"""Annotate range-compressed data with water labels derived from the GRD products of the same datatake.

GRD sigma0 (VV) -> multilook + boxcar -> scene-wide tiled-Otsu water mask -> each GRD pixel is mapped to
(zero-Doppler azimuth time, slant-range time) via the GRD geolocation grid -> (echo line, range bin)
of the range-compressed data -> aggregated into label cells of `--lines-per-row` echo lines x 1 range bin.

The residual offset between both geometries (timing conventions, squint) is estimated by
cross-correlating the projected GRD backscatter with the azimuth-averaged range-compressed power,
and removed before the final projection.

Output in <data_root>/processed/<scene>/:
  labels.npy        float32 (n_rows, n_bins) water fraction, NaN = no label
  grd_db.npy        float32 (n_rows, n_bins) projected GRD VV sigma0 [dB] (reference)
  rc_rows_db.npy    float32 (n_rows, n_pol, n_bins) RC power averaged over each row's lines [dB]
  annot_meta.json, quicklook.png

Usage: python scripts/annotate.py --scene s4_20250422
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import yaml
from scipy import ndimage

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ai4onboardsar import grd  # noqa: E402


def project(points_t, points_tau, values, t_lines, k, tau_start, dtau, shape):
    """Scatter-average values at (azimuth time, slant-range time) onto the (row, bin) label grid."""
    line = np.interp(points_t, t_lines, np.arange(len(t_lines)), left=-1, right=-1)
    row = np.floor(line / k).astype(int)
    col = np.floor((points_tau - tau_start) / dtau).astype(int)
    ok = (line >= 0) & (row < shape[0]) & (col >= 0) & (col < shape[1])
    idx = row[ok] * shape[1] + col[ok]
    n = np.bincount(idx, minlength=shape[0] * shape[1])
    s = np.bincount(idx, weights=values[ok], minlength=shape[0] * shape[1])
    return s.reshape(shape), n.reshape(shape)


def rows_mean_db(rc_db: np.ndarray, k: int, chunk: int = 4096) -> np.ndarray:
    """Average RC power (linear) over blocks of k lines -> (n_rows, n_pol, n_bins) in dB."""
    n_rows = rc_db.shape[0] // k
    out = np.empty((n_rows, *rc_db.shape[1:]), np.float32)
    step = (chunk // k) * k
    for s in range(0, n_rows * k, step):
        x = rc_db[s:min(s + step, n_rows * k)].astype(np.float32)
        x = 10 ** (x / 10)
        x = np.nanmean(x.reshape(-1, k, *x.shape[1:]), axis=1)
        out[s // k:s // k + len(x)] = 10 * np.log10(x)
    return out


def highpass(x: np.ndarray, sigma: float = 8.0) -> np.ndarray:
    m = np.isfinite(x)
    x = np.where(m, x, np.nanmean(x))
    x = x - ndimage.gaussian_filter(x, sigma)
    return np.where(m, x, 0.0)


def estimate_shift(a: np.ndarray, b: np.ndarray, max_shift: int = 64):
    """Shift (rows, cols) to apply to b's coordinates to best match a (normalized cross-correlation
    of high-passed images, integer peak + parabolic refinement)."""
    a, b = highpass(a), highpass(b)
    fa, fb = np.fft.rfft2(a), np.fft.rfft2(b)
    xc = np.fft.irfft2(fa * np.conj(fb), s=a.shape)
    xc = np.fft.fftshift(xc)
    c0 = np.array(a.shape) // 2
    win = xc[c0[0] - max_shift:c0[0] + max_shift + 1, c0[1] - max_shift:c0[1] + max_shift + 1]
    i, j = np.unravel_index(np.argmax(win), win.shape)

    def parab(m, c, p):
        d = m - 2 * c + p
        return 0.0 if d == 0 else 0.5 * (m - p) / d
    di = parab(win[i - 1, j], win[i, j], win[i + 1, j]) if 0 < i < win.shape[0] - 1 else 0.0
    dj = parab(win[i, j - 1], win[i, j], win[i, j + 1]) if 0 < j < win.shape[1] - 1 else 0.0
    peak_ratio = win[i, j] / np.sort(win.ravel())[-int(0.01 * win.size)]
    return i - max_shift + di, j - max_shift + dj, float(peak_ratio)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/scenes.yaml")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--lines-per-row", type=int, default=16)
    ap.add_argument("--grd-looks", type=int, default=2, help="GRD multilook factor (10 m pixels)")
    ap.add_argument("--no-coreg", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    root = Path(cfg["data_root"])
    d = root / "processed" / args.scene
    meta = json.load(open(d / "rc_meta.json"))
    t_lines = np.load(d / "lines.npz")["t_utc"]
    k = args.lines_per_row
    shape = (meta["n_lines"] // k, meta["n_bins"])

    rc_db = np.load(d / "rc_db.npy", mmap_mode="r")
    print("averaging RC rows ...")
    rc_rows = rows_mean_db(rc_db, k)
    np.save(d / "rc_rows_db.npy", rc_rows)

    # GRD -> per-pixel (t, tau, sigma0_db, water); one water threshold for the whole scene
    grd_names = json.load(open(root / "scenes" / f"{args.scene}.json"))["grd"]
    slices, tile_ts, tile_px = [], [], []
    for name in grd_names:
        print("GRD", name)
        db, valid = grd.prepare_vv(grd.block_mean2d(grd.sigma0(root / "grd" / name, "vv"), args.grd_looks))
        slices.append((name, db, valid))
        ts, px = grd.tile_thresholds(db, valid)
        tile_ts += ts
        tile_px.append(px)
    thr = grd.scene_threshold(np.concatenate(tile_px))
    print(f"  water threshold {thr:.2f} dB (VV) from {len(tile_ts)} bimodal tiles")
    pts = {"t": [], "tau": [], "db": [], "water": []}
    thresholds = {"vv_threshold_db": thr, "bimodal_tiles": len(tile_ts),
                  "tile_otsu_median_db": float(np.median(tile_ts)) if tile_ts else None, "slices": []}
    for name, db, valid in slices:
        water = grd.water_mask(db, valid, thr)
        thresholds["slices"].append({"grd": name, "water_fraction": float(water[valid].mean())})
        print(f"  {name[:40]}: water {water[valid].mean():.2%}")
        az, rg = grd.radar_coords(root / "grd" / name, "vv")
        g = args.grd_looks
        L, P = np.meshgrid((np.arange(valid.shape[0]) + 0.5) * g - 0.5,
                           (np.arange(valid.shape[1]) + 0.5) * g - 0.5, indexing="ij")
        v = valid.ravel()
        pts["t"].append(az(L.ravel()[v], P.ravel()[v]))
        pts["tau"].append(rg(L.ravel()[v], P.ravel()[v]))
        pts["db"].append(db.ravel()[v])
        pts["water"].append(water.ravel()[v].astype(np.float32))
        del water, L, P
    del slices
    pts = {key: np.concatenate(v) for key, v in pts.items()}

    def project_all(dt=0.0, dtau_off=0.0):
        s, n = project(pts["t"] + dt, pts["tau"] + dtau_off, pts["water"], t_lines, k,
                       meta["tau_start"], meta["dtau"], shape)
        sd, _ = project(pts["t"] + dt, pts["tau"] + dtau_off, pts["db"], t_lines, k,
                        meta["tau_start"], meta["dtau"], shape)
        with np.errstate(invalid="ignore", divide="ignore"):
            return s / n, sd / n, n

    labels, grd_db, n = project_all()

    # coregistration: compare projected GRD dB with the unfocused RC image (VV)
    shift = {"rows": 0.0, "bins": 0.0, "dt_s": 0.0, "dtau_s": 0.0, "peak_ratio": None}
    if not args.no_coreg:
        rc_vv = rc_rows[:, 0]
        ok = np.isfinite(grd_db) & np.isfinite(rc_vv)
        r = np.flatnonzero(ok.any(1))
        c = np.flatnonzero(ok.any(0))
        a = np.where(ok, rc_vv, np.nan)[r[0]:r[-1] + 1, c[0]:c[-1] + 1]
        b = np.where(ok, grd_db, np.nan)[r[0]:r[-1] + 1, c[0]:c[-1] + 1]
        dr, dc, pr = estimate_shift(a, b)
        prf = meta["prf_mean"]
        shift = {"rows": float(dr), "bins": float(dc), "dt_s": float(dr * k / prf),
                 "dtau_s": float(dc * meta["dtau"]), "peak_ratio": pr}
        print(f"coreg shift: {dr:+.2f} rows ({dr * k:+.1f} lines), {dc:+.2f} bins, "
              f"peak ratio {pr:.2f}")
        labels, grd_db, n = project_all(shift["dt_s"], shift["dtau_s"])

    # cells with too few GRD samples are unreliable (swath edges)
    labels[n < 2] = np.nan
    grd_db[n < 2] = np.nan
    np.save(d / "labels.npy", labels.astype(np.float32))
    np.save(d / "grd_db.npy", grd_db.astype(np.float32))
    json.dump({"lines_per_row": k, "grd_looks": args.grd_looks, "n_rows": shape[0],
               "n_bins": shape[1], "thresholds": thresholds, "coreg": shift,
               "labelled_fraction": float(np.isfinite(labels).mean()),
               "water_fraction": float(np.nanmean(labels > 0.5))},
              open(d / "annot_meta.json", "w"), indent=2)

    # quicklook (rows = azimuth, cols = slant range), subsampled
    sub = max(1, shape[0] // 1500)
    fig, ax = plt.subplots(1, 3, figsize=(18, 10), sharex=True, sharey=True)
    vv = rc_rows[::sub, 0]
    vv = vv - np.nanmedian(vv, 0, keepdims=True)  # remove range-dependent gain for display
    ax[0].imshow(vv, cmap="gray", vmin=-8, vmax=6, aspect="auto", interpolation="none")
    ax[0].set_title("RC power (VV), azimuth-averaged, range-normalized")
    ax[1].imshow(grd_db[::sub], cmap="gray", vmin=-25, vmax=0, aspect="auto", interpolation="none")
    ax[1].set_title("GRD sigma0 VV projected [dB]")
    ax[2].imshow(vv, cmap="gray", vmin=-8, vmax=6, aspect="auto", interpolation="none")
    ax[2].imshow(np.ma.masked_less(labels[::sub], 0.5), cmap="cool", alpha=0.6, aspect="auto",
                 interpolation="none")
    ax[2].set_title("water labels on RC")
    for a_ in ax:
        a_.set_xlabel("range bin")
    ax[0].set_ylabel(f"row ({k} echo lines, every {sub})")
    fig.tight_layout()
    fig.savefig(d / "quicklook.png", dpi=80)
    print("done", d)


if __name__ == "__main__":
    main()
