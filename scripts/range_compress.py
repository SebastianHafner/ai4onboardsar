"""Decode the RAW (L0) slices of a scene, range-compress every echo line and write the result on a
common slant-range grid.

Output in <data_root>/processed/<scene>/:
  rc_db.npy     float16 (n_lines, n_pol, n_bins): 10*log10 of range-compressed power, block-averaged
                over `--range-dec` samples (NaN outside the receive window)
  lines.npz     t_utc (s since l0.TIME_REF), pri_count per line
  rc_meta.json  range grid (tau_start, dtau), polarisations, sampling rate, ...

Usage: python scripts/range_compress.py --scene s4_20250422
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ai4onboardsar import l0  # noqa: E402

AZ_MARGIN_S = 1.0  # keep this much data before/after the GRD coverage (> half synthetic aperture)


def grd_time_span(grd_names):
    fmt = "%Y%m%dT%H%M%S"
    t0 = min(datetime.strptime(n.split("_")[4], fmt) for n in grd_names)
    t1 = max(datetime.strptime(n.split("_")[5], fmt) for n in grd_names)
    return l0.utc_seconds(t0), l0.utc_seconds(t1) + 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/scenes.yaml")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--bin-m", type=float, default=21.3,
                    help="target slant-range bin size; range decimation = round(bin / sample spacing)")
    ap.add_argument("--pols", nargs="+", default=["vv", "vh"])
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    root = Path(cfg["data_root"])
    manifest = json.load(open(root / "scenes" / f"{args.scene}.json"))
    out_dir = root / "processed" / args.scene
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start, t_stop = grd_time_span(manifest["grd"])
    t_start -= AZ_MARGIN_S
    t_stop += AZ_MARGIN_S

    # --- pass 1: metadata of all slices / polarisations
    sources = {p: [] for p in args.pols}  # pol -> [(decoder, meta)]
    for raw_name in manifest["raw"]:
        files = l0.l0_files(root / "raw" / raw_name)
        for p in args.pols:
            print(f"reading headers {raw_name} {p}")
            sources[p].append(l0.read_echo_meta(files[p]))

    # common set of lines: PRI counts present in every polarisation, within the time span
    def lines_of(p):
        d = {}
        for si, (_, m) in enumerate(sources[p]):
            for ri in np.flatnonzero((m.t_utc >= t_start) & (m.t_utc <= t_stop)):
                d.setdefault(int(m.pri_count[ri]), (si, ri))
        return d

    per_pol = {p: lines_of(p) for p in args.pols}
    pri_common = sorted(set.intersection(*[set(d) for d in per_pol.values()]))
    ref = per_pol[args.pols[0]]
    t_utc = np.array([sources[args.pols[0]][ref[c][0]][1].t_utc[ref[c][1]] for c in pri_common])
    order = np.argsort(t_utc, kind="stable")
    pri_common = [pri_common[i] for i in order]
    t_utc = t_utc[order]

    # common slant-range grid
    all_m = [m for p in args.pols for _, m in sources[p]]
    fs_all = np.unique(np.concatenate([m.fs for m in all_m]))
    assert len(fs_all) == 1, f"varying range sampling rate {fs_all}"
    fs = float(fs_all[0])
    args.range_dec = max(1, int(round(args.bin_m * 2 * fs / 299792458.0)))
    dtau = args.range_dec / fs
    n_samp = max(int(m.packets["Number of Quads"].max()) * 2 for m in all_m)
    tau_lo = min(m.tau0.min() for m in all_m)
    tau_hi = max(m.tau0.max() for m in all_m) + n_samp / fs
    n_bins = int(np.ceil((tau_hi - tau_lo) / dtau))
    n_lines = len(pri_common)
    print(f"{n_lines} lines, fs={fs / 1e6:.3f} MHz, {n_samp} samples -> {n_bins} bins "
          f"({dtau * 3e8 / 2:.1f} m slant)")

    out = np.lib.format.open_memmap(out_dir / "rc_db.npy", mode="w+", dtype=np.float16,
                                    shape=(n_lines, len(args.pols), n_bins))
    out[:] = np.nan
    line_of_pri = {c: i for i, c in enumerate(pri_common)}

    # --- pass 2: decode + range compress + resample onto grid
    for pi, p in enumerate(args.pols):
        for si, (dec, m) in enumerate(sources[p]):
            rows = np.array(sorted(ri for c, (s, ri) in per_pol[p].items() if s == si and c in line_of_pri))
            if len(rows) == 0:
                continue
            for rws, pw, n_rep in tqdm(l0.decode_range_compressed_power(dec, m, rows),
                                       desc=f"{p} slice {si}", total=int(np.ceil(len(rows) / 1024))):
                n_valid = pw.shape[1] - n_rep + 1
                pw = l0.block_mean(pw[:, :n_valid], args.range_dec)
                db = (10 * np.log10(np.maximum(pw, 1e-6))).astype(np.float16)
                dst = np.array([line_of_pri[int(c)] for c in m.pri_count[rws]])
                off = np.round((m.tau0[rws] - tau_lo) / dtau).astype(int)
                for o in np.unique(off):
                    k = off == o
                    out[dst[k], pi, o:o + db.shape[1]] = db[k]
    out.flush()

    np.savez(out_dir / "lines.npz", t_utc=t_utc, pri_count=np.array(pri_common))
    json.dump({"scene": args.scene, "pols": args.pols, "fs": fs, "range_dec": args.range_dec,
               "tau_start": tau_lo, "dtau": dtau, "n_lines": n_lines, "n_bins": n_bins,
               "prf_mean": float(1 / np.median(np.diff(t_utc))), "raw": manifest["raw"]},
              open(out_dir / "rc_meta.json", "w"), indent=2)
    print("done", out_dir)


if __name__ == "__main__":
    main()
