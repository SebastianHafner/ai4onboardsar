"""Train a streaming water segmentation model on one range-compressed scene.

An azimuth block of the scene's rows ([val_start, val_start + val_frac) as fractions) is held out for
validation, separated from the training rows by gaps of 2*delay rows (the synthetic aperture smears
targets across the boundary).

Usage: python scripts/train.py --config configs/train.yaml
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ai4onboardsar.data import Scene, WindowDataset  # noqa: E402
from ai4onboardsar.evaluation import metrics, stream_predict  # noqa: E402
from ai4onboardsar.models import build_model  # noqa: E402


def loss_fn(logits, y, pos_weight):
    m = torch.isfinite(y)
    yl, lg = y[m], logits[m]
    bce = F.binary_cross_entropy_with_logits(lg, yl, pos_weight=torch.tensor(pos_weight))
    p = torch.sigmoid(lg)
    dice = 1 - (2 * (p * yl).sum() + 1) / (p.sum() + yl.sum() + 1)
    return bce + dice


def val_rows(cfg, n_rows):
    a = int(n_rows * cfg["val_start"])
    return a, min(n_rows, a + int(n_rows * cfg["val_frac"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train.yaml")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    torch.manual_seed(cfg["seed"])
    torch.set_num_threads(cfg.get("threads", 8))
    data_root = Path(yaml.safe_load(open(cfg["scenes_config"]))["data_root"])

    out = Path(args.out or Path(cfg["runs_dir"]) / f"{cfg['model']['name']}_{time.strftime('%Y%m%d_%H%M%S')}")
    out.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump(cfg, open(out / "config.yaml", "w"))

    scene = Scene(data_root / "processed" / cfg["train_scene"], cfg["norm"], cfg.get("db_offset"))
    R, D = scene.n_rows, cfg["delay"]
    v_a, v_b = val_rows(cfg, R)
    w = cfg["window"]
    ds = WindowDataset(scene, [(0, v_a - 2 * D), (v_b + 2 * D, R)], warmup=w["warmup"], delay=D, target_len=w["target_len"],
                       width=w["width"], n_samples=cfg["steps_per_epoch"] * cfg["batch_size"],
                       water_oversample=cfg["water_oversample"], seed=cfg["seed"])
    dl = DataLoader(ds, batch_size=cfg["batch_size"], num_workers=0)
    y_tr = np.concatenate([scene.y[:v_a - 2 * D], scene.y[v_b + 2 * D:]])
    wf = np.nanmean(y_tr[np.isfinite(y_tr)] > 0.5)
    pos_weight = float(min(cfg["max_pos_weight"], (1 - wf) / max(wf, 1e-6)))
    print(f"train scene {cfg['train_scene']}: {R} rows x {scene.x.shape[2]} bins, water {wf:.2%}, "
          f"pos_weight {pos_weight:.1f}")

    model = build_model(cfg["model"]["name"], cin=scene.x.shape[1], **cfg["model"].get("kwargs", {}))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model {cfg['model']['name']}: {n_params / 1e3:.1f} k params")
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=cfg["epochs"] * len(dl))

    # validation stream starts `warmup` rows before the validation block
    v0 = v_a - w["warmup"]
    best, log = -1.0, []
    for ep in range(cfg["epochs"]):
        model.train()
        t0, tot = time.time(), 0.0
        for x, y in dl:
            logits, _ = model(x)
            logits = logits[:, w["warmup"] + D:]
            loss = loss_fn(logits, y, pos_weight)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
        prob = stream_predict(model, scene.x[v0:v_b + D], D)[w["warmup"]:v_b - v0]
        m = metrics(prob, scene.y[v_a:v_b])
        rec = {"epoch": ep, "loss": tot / len(dl), "time_s": time.time() - t0, **{f"val_{k}": v for k, v in m.items()}}
        log.append(rec)
        print(f"ep {ep:3d} loss {rec['loss']:.4f} val IoU {m['iou']:.3f} F1 {m['f1']:.3f} "
              f"P {m['precision']:.3f} R {m['recall']:.3f} ({rec['time_s']:.0f}s)", flush=True)
        if m["iou"] > best:
            best = m["iou"]
            torch.save(model.state_dict(), out / "best.pt")
        torch.save(model.state_dict(), out / "last.pt")
        json.dump(log, open(out / "log.json", "w"), indent=1)
    print("best val IoU", best, "->", out)


if __name__ == "__main__":
    main()
