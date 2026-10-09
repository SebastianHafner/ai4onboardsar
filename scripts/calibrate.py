"""Estimate the range gain profile of a beam (elevation antenna pattern x range spreading x typical
land backscatter) from the land cells of one or more annotated scenes.

Onboard, this profile is a fixed per-beam lookup over slant-range time (it could also be derived from
the antenna model); normalizing by it keeps absolute radiometry, unlike the running-mean
normalization.

Output: <data_root>/calib/<beam>.npz with tau [s], vv, vh [dB] (land median, smoothed).

Usage: python scripts/calibrate.py --scenes s4_20250422
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def scene_beam(proc_dir: Path) -> str:
    return json.load(open(proc_dir / "rc_meta.json"))["raw"][0].split("_")[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/scenes.yaml")
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--smooth-bins", type=float, default=10)
    args = ap.parse_args()
    root = Path(yaml.safe_load(open(args.config))["data_root"])

    beams = {scene_beam(root / "processed" / s) for s in args.scenes}
    assert len(beams) == 1, f"scenes from several beams: {beams}"
    beam = beams.pop()

    taus, vals = [], []
    for s in args.scenes:
        d = root / "processed" / s
        m = json.load(open(d / "rc_meta.json"))
        x = np.load(d / "rc_rows_db.npy")
        y = np.load(d / "labels.npy")
        land = np.isfinite(y) & (y < 0.5)
        tau = m["tau_start"] + (np.arange(x.shape[2]) + 0.5) * m["dtau"]
        prof = np.stack([np.nanmedian(np.where(land, x[:, p], np.nan), 0) for p in range(x.shape[1])])
        taus.append(tau)
        vals.append(prof)
    # common grid over all scenes, average where several scenes overlap
    dtau = min(np.diff(t[:2])[0] for t in taus)
    grid = np.arange(min(t[0] for t in taus), max(t[-1] for t in taus) + dtau, dtau)
    acc = np.zeros((2, len(grid)))
    cnt = np.zeros((2, len(grid)))
    for t, v in zip(taus, vals):
        vi = np.stack([np.interp(grid, t, np.where(np.isfinite(vp), vp, np.nan), left=np.nan, right=np.nan)
                       for vp in v])
        ok = np.isfinite(vi)
        acc[ok] += vi[ok]
        cnt[ok] += 1
    prof = acc / np.maximum(cnt, 1)
    ok = cnt.min(0) > 0
    grid, prof = grid[ok], prof[:, ok]
    prof = gaussian_filter1d(prof, args.smooth_bins, axis=1, mode="nearest")
    (root / "calib").mkdir(exist_ok=True)
    np.savez(root / "calib" / f"{beam}.npz", tau=grid, vv=prof[0], vh=prof[1], scenes=np.array(args.scenes))
    print(f"beam {beam}: tau {grid[0] * 1e3:.4f}-{grid[-1] * 1e3:.4f} ms, VV {prof[0].min():.1f}..{prof[0].max():.1f} dB")


if __name__ == "__main__":
    main()
