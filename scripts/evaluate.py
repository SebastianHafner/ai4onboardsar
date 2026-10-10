"""Evaluate a trained run on the test scenes (full-scene streaming inference) and compare with the
threshold baseline (darkness of the azimuth-averaged, range-normalized RC power, threshold fitted on
the training scene).

Usage: python scripts/evaluate.py --run <runs_dir>/<run> [--scenes s4_20250427 s2_20250405]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ai4onboardsar.data import Scene  # noqa: E402
from ai4onboardsar.evaluation import (fit_threshold, metrics, stream_predict,  # noqa: E402
                                      threshold_baseline_score)
from ai4onboardsar.models import build_model  # noqa: E402
from train import val_rows  # noqa: E402


def quicklook(path, x, y, prob, score, thr, title):
    sub = max(1, x.shape[0] // 1200)
    fig, ax = plt.subplots(1, 4, figsize=(20, 10), sharex=True, sharey=True)
    ax[0].imshow(x[::sub, 0], cmap="gray", vmin=-2, vmax=2, aspect="auto", interpolation="none")
    ax[0].set_title("input VV (normalized)")
    ax[1].imshow(y[::sub], cmap="Blues", vmin=0, vmax=1, aspect="auto", interpolation="none")
    ax[1].set_title("label (GRD water)")
    ax[2].imshow(prob[::sub], cmap="Blues", vmin=0, vmax=1, aspect="auto", interpolation="none")
    ax[2].set_title("model probability")
    ax[3].imshow((score >= thr)[::sub], cmap="Blues", vmin=0, vmax=1, aspect="auto", interpolation="none")
    ax[3].set_title("threshold baseline")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=70)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--scenes", nargs="*")
    args = ap.parse_args()
    run = Path(args.run)
    cfg = yaml.safe_load(open(run / "config.yaml"))
    torch.set_num_threads(cfg.get("threads", 8))
    data_root = Path(yaml.safe_load(open(cfg["scenes_config"]))["data_root"])
    D = cfg["delay"]

    train = Scene(data_root / "processed" / cfg["train_scene"], cfg["norm"], cfg.get("db_offset"))
    v_a, v_b = val_rows(cfg, train.n_rows)
    tr_rows = np.r_[0:v_a - 2 * D, v_b + 2 * D:train.n_rows]
    model = build_model(cfg["model"]["name"], cin=train.x.shape[1], **cfg["model"].get("kwargs", {}))
    model.load_state_dict(torch.load(run / args.ckpt))
    model.eval()

    # baseline threshold fitted on the training part of the training scene
    tr_score = threshold_baseline_score(train.x, D)[tr_rows]
    thr = fit_threshold(tr_score, train.y[tr_rows])

    results = {"n_params": sum(p.numel() for p in model.parameters()), "baseline_threshold": thr}
    lines_per_row = train.meta["lines_per_row"]
    scenes = {"val": (train, v_a, v_b)}
    for s in args.scenes or cfg["test_scenes"]:
        sc = Scene(data_root / "processed" / s, cfg["norm"], cfg.get("db_offset"))
        scenes[s] = (sc, 0, sc.n_rows)

    for name, (sc, r0, r1) in scenes.items():
        a = max(0, r0 - cfg["window"]["warmup"])
        x = sc.x[a:min(sc.n_rows, r1 + D)]
        t = time.time()
        prob = stream_predict(model, x, D)[r0 - a:r1 - a]
        dt = time.time() - t
        x, y = x[r0 - a:r1 - a], sc.y[r0:r1]
        # the first `warmup` rows of a scene stream have no history; exclude them from scoring
        skip = 0 if r0 else cfg["window"]["warmup"]
        score = threshold_baseline_score(x, D)
        m_model = metrics(prob[skip:], y[skip:])
        m_base = metrics(score[skip:], y[skip:], thr)
        rows_per_s = x.shape[0] / dt
        results[name] = {"model": m_model, "threshold_baseline": m_base,
                         "rows_per_s": rows_per_s, "echo_lines_per_s": rows_per_s * lines_per_row,
                         "n_rows": int(x.shape[0]), "width_bins": int(x.shape[2])}
        print(f"{name:14s} model IoU {m_model['iou']:.3f} F1 {m_model['f1']:.3f} | "
              f"baseline IoU {m_base['iou']:.3f} F1 {m_base['f1']:.3f} | water label/model/baseline "
              f"{m_model['water_frac_label']:.2%}/{m_model['water_frac_pred']:.2%}/{m_base['water_frac_pred']:.2%} | "
              f"{rows_per_s * lines_per_row:.0f} echo lines/s on CPU")
        np.save(run / f"prob_{name}.npy", prob.astype(np.float16))
        quicklook(run / f"eval_{name}.png", x, y, prob, score, thr, f"{name} ({run.name})")
    json.dump(results, open(run / "eval.json", "w"), indent=2)


if __name__ == "__main__":
    main()
