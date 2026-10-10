"""Streaming inference from raw L0 echo lines: front end (range compression, binning, row averaging,
normalization; ai4onboardsar/frontend.py) followed by the trained network, one batch of echo lines
at a time with the recurrent state carried over.

The echo lines of a processed scene are replayed in acquisition order (same lines and row grid as
the training input), so the result can be compared with the labels and with the prediction from the
cached input (scripts/evaluate.py).

Output in the run directory: prob_raw_<scene>.npy (rows, bins; aligned to labels, NaN for the last
`delay` rows) and infer_raw_<scene>.json (metrics, agreement with the cached input, throughput).

Usage: python scripts/infer_raw.py --run <runs_dir>/<run> --scene s4_20250329 [--rows 1000 1500]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ai4onboardsar import l0  # noqa: E402
from ai4onboardsar.data import DB_SCALE, Scene, load_calib  # noqa: E402
from ai4onboardsar.evaluation import metrics, stream_predict  # noqa: E402
from ai4onboardsar.frontend import Frontend  # noqa: E402
from ai4onboardsar.models import build_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--rows", nargs=2, type=int, help="row interval [a, b) to process (default: all)")
    ap.add_argument("--batch-rows", type=int, default=64, help="rows (x lines_per_row echo lines) per batch")
    args = ap.parse_args()
    run = Path(args.run)
    cfg = yaml.safe_load(open(run / "config.yaml"))
    torch.set_num_threads(cfg.get("threads", 8))
    root = Path(yaml.safe_load(open(cfg["scenes_config"]))["data_root"])
    d = root / "processed" / args.scene
    rc_meta = json.load(open(d / "rc_meta.json"))
    K = json.load(open(d / "annot_meta.json"))["lines_per_row"]
    D = cfg["delay"]

    # --- front end on the scene's slant-range grid
    beam = rc_meta["raw"][0].split("_")[1]
    calib = load_calib(root, beam) if cfg["norm"] == "calib" else None
    fe = Frontend(rc_meta["tau_start"], rc_meta["dtau"], rc_meta["n_bins"], rc_meta["range_dec"], K,
                  cfg["norm"], cfg.get("db_offset"), calib, n_pol=len(rc_meta["pols"]))

    # --- echo lines of the scene: PRI count -> (decoder, meta, packet row) per polarisation
    pri = np.load(d / "lines.npz")["pri_count"]
    n_rows = len(pri) // K
    a, b = args.rows or (0, n_rows)
    src = {p: [] for p in rc_meta["pols"]}
    for raw in rc_meta["raw"]:
        files = l0.l0_files(root / "raw" / raw)
        for p in rc_meta["pols"]:
            src[p].append(l0.read_echo_meta(files[p]))
    where = {p: {} for p in src}
    for p, slices in src.items():
        for si, (_, m) in enumerate(slices):
            for ri, c in enumerate(m.pri_count):
                where[p].setdefault(int(c), (si, ri))

    def raw_lines(counts):
        """Decode the echo lines with these PRI counts -> front-end line power (N, n_pol, n_bins)."""
        out = torch.full((len(counts), len(src), fe.n_bins), float("nan"), dtype=torch.float64)
        for pi, p in enumerate(src):
            loc = np.array([where[p][int(c)] for c in counts])
            for si in np.unique(loc[:, 0]):
                dec, m = src[p][si]
                idx = np.flatnonzero(loc[:, 0] == si)
                for rws, echo, gain, rep in l0.decode_echoes(dec, m, loc[idx, 1]):
                    dst = idx[np.searchsorted(loc[idx, 1], rws)]
                    pw = fe.compress(torch.from_numpy(echo), torch.from_numpy(rep), torch.from_numpy(gain.copy()))
                    out[dst, pi] = fe.to_grid(pw, m.tau0[rws]).double()
        return out

    # --- model
    model = build_model(cfg["model"]["name"], cin=len(src) + 1, **cfg["model"].get("kwargs", {}))
    model.load_state_dict(torch.load(run / args.ckpt))
    model.eval()

    x_raw, logits, state = [], [], None
    t_fe = t_net = 0.0
    with torch.no_grad():
        for r in range(a, b, args.batch_rows):
            r1 = min(b, r + args.batch_rows)
            t = time.time()
            rows = fe.push_lines(raw_lines(pri[r * K:r1 * K]))
            t_fe += time.time() - t
            t = time.time()
            out, state = model(rows[None], state)
            t_net += time.time() - t
            x_raw.append(rows.numpy())
            logits.append(out[0].numpy())
            print(f"rows {r1 - a}/{b - a}  front end {t_fe:.0f}s  network {t_net:.1f}s", end="\r", flush=True)
    print()
    x_raw = np.concatenate(x_raw)
    prob = np.full((b - a, fe.n_bins), np.nan, np.float32)
    prob[:b - a - D] = 1 / (1 + np.exp(-np.concatenate(logits)[D:]))  # output at t -> row t - D
    np.save(run / f"prob_raw_{args.scene}.npy", prob.astype(np.float16))

    # --- comparison with the cached input and the labels
    sc = Scene(d, cfg["norm"], cfg.get("db_offset"))
    x_c = sc.x[a:b]
    prob_c = stream_predict(model, x_c, D)
    ok = np.isfinite(prob) & np.isfinite(prob_c)
    skip = 0 if a else cfg["window"]["warmup"]
    n_lines = (b - a) * K
    res = {"scene": args.scene, "rows": [a, b], "echo_lines": n_lines,
           "input_max_abs_diff_db": float(np.abs(x_raw - x_c)[:, :-1].max() * DB_SCALE),
           "valid_identical": bool((x_raw[:, -1] == x_c[:, -1]).all()),
           "prob_max_abs_diff": float(np.nanmax(np.abs(prob - prob_c))),
           "decisions_differing_frac": float(np.mean((prob >= 0.5)[ok] != (prob_c >= 0.5)[ok])),
           "model_raw": metrics(prob[skip:], sc.y[a:b][skip:]),
           "model_cached": metrics(prob_c[skip:], sc.y[a:b][skip:]),
           "frontend_echo_lines_per_s": n_lines / t_fe, "network_echo_lines_per_s": n_lines / t_net}
    json.dump(res, open(run / f"infer_raw_{args.scene}.json", "w"), indent=2)
    print(json.dumps({k: v for k, v in res.items() if not k.startswith("model")}, indent=1))
    print(f"IoU from raw {res['model_raw']['iou']:.3f}, from cached input {res['model_cached']['iou']:.3f}")


if __name__ == "__main__":
    main()
