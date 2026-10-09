"""Sentinel-1 GRD reading: calibrated sigma0, per-pixel radar coordinates, and water thresholding."""
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator
from skimage.filters import threshold_otsu

from .l0 import utc_seconds


def _t(s: str) -> float:
    return utc_seconds(datetime.fromisoformat(s))


def _files(safe: Path, pol: str):
    ann = next((safe / "annotation").glob(f"*-{pol}-*.xml"))
    cal = next((safe / "annotation" / "calibration").glob(f"calibration-*-{pol}-*.xml"))
    tif = next((safe / "measurement").glob(f"*-{pol}-*.tiff"))
    return ann, cal, tif


def _grid(points, value_fn, method):
    """Interpolator over a regular (line, pixel) grid of annotation points."""
    lines = np.array(sorted({int(p.find("line").text) for p in points}))
    pixels = np.array(sorted({int(p.find("pixel").text) for p in points}))
    v = np.full((len(lines), len(pixels)), np.nan)
    li = {l: i for i, l in enumerate(lines)}
    pj = {p: j for j, p in enumerate(pixels)}
    for p in points:
        v[li[int(p.find("line").text)], pj[int(p.find("pixel").text)]] = value_fn(p)
    assert not np.isnan(v).any(), "irregular annotation grid"
    return RegularGridInterpolator((lines, pixels), v, method=method, bounds_error=False, fill_value=None)


def radar_coords(safe: Path, pol: str = "vv"):
    """Interpolators (line, pixel) -> zero-Doppler azimuth time [s since TIME_REF] and two-way
    slant-range time [s], from the geolocation grid."""
    ann, _, _ = _files(safe, pol)
    pts = ET.parse(ann).getroot().findall(".//geolocationGridPoint")
    t0 = _t(pts[0].find("azimuthTime").text)
    az = _grid(pts, lambda p: _t(p.find("azimuthTime").text) - t0, "linear")
    rg = _grid(pts, lambda p: float(p.find("slantRangeTime").text), "cubic")
    return (lambda l, p: az(np.stack([l, p], -1)) + t0), (lambda l, p: rg(np.stack([l, p], -1)))


def sigma0(safe: Path, pol: str) -> np.ndarray:
    """Calibrated sigma0 (linear) of a GRD polarisation channel; 0 where no data."""
    _, cal, tif = _files(safe, pol)
    dn = tifffile.imread(tif).astype(np.float32)
    vecs = ET.parse(cal).getroot().findall(".//calibrationVector")
    lines = np.array([int(v.find("line").text) for v in vecs])
    pixels = np.array([int(x) for x in vecs[0].find("pixel").text.split()])
    lut = np.array([[float(x) for x in v.find("sigmaNought").text.split()] for v in vecs])
    interp = RegularGridInterpolator((lines, pixels), lut, bounds_error=False, fill_value=None)
    out = np.empty_like(dn)
    pp = np.arange(dn.shape[1])
    for s in range(0, dn.shape[0], 1024):
        ll = np.arange(s, min(s + 1024, dn.shape[0]))
        L, P = np.meshgrid(ll, pp, indexing="ij")
        a = interp(np.stack([L, P], -1)).astype(np.float32)
        out[ll] = dn[ll] ** 2 / a**2
    return out


def block_mean2d(x: np.ndarray, k: int) -> np.ndarray:
    h, w = (x.shape[0] // k) * k, (x.shape[1] // k) * k
    return x[:h, :w].reshape(h // k, k, w // k, k).mean((1, 3))


def tile_thresholds(db: np.ndarray, valid: np.ndarray, lo: float = -24.0, hi: float = -14.0,
                    dark_max: float = -18.0, tile: int = 128, min_valid: float = 0.9) -> list:
    """Otsu thresholds of tiles with a bimodal water/land histogram (split-based approach).

    A tile is used when its Otsu threshold lies in [lo, hi], both classes are well separated
    (Ashman's D > 2), the minority class covers >= 10 % and the dark class mean is water-like
    (< dark_max). Returns the tile thresholds and the (subsampled) pixels of the selected tiles."""
    ts, px = [], []
    for i in range(0, db.shape[0] - tile + 1, tile):
        for j in range(0, db.shape[1] - tile + 1, tile):
            v = valid[i:i + tile, j:j + tile]
            if v.mean() < min_valid:
                continue
            x = db[i:i + tile, j:j + tile][v]
            t = threshold_otsu(x)
            a, b = x[x < t], x[x >= t]
            if not (lo <= t <= hi) or min(len(a), len(b)) < 0.1 * len(x) or a.mean() > dark_max:
                continue
            d = np.sqrt(2) * abs(a.mean() - b.mean()) / np.sqrt(a.var() + b.var())
            if d > 2:
                ts.append(t)
                px.append(x[::4])
    return ts, (np.concatenate(px) if px else np.empty(0, np.float32))


def scene_threshold(pixels: np.ndarray, default: float = -18.0, min_pixels: int = 10000,
                    lo: float = -23.0, hi: float = -14.0) -> float:
    """Threshold at the valley of the smoothed histogram of pixels pooled from bimodal tiles (more
    robust than the median tile-Otsu when wind-roughened water broadens the water mode)."""
    if len(pixels) < min_pixels:
        return default
    edges = np.arange(-35, 5.01, 0.1)
    h, _ = np.histogram(pixels, edges)
    h = ndimage.gaussian_filter1d(h.astype(float), 5)
    c = 0.5 * (edges[1:] + edges[:-1])
    water_mode = c[(c < -17)][np.argmax(h[c < -17])]
    m = (c > water_mode) & (c >= lo) & (c <= hi)
    return float(c[m][np.argmin(h[m])])


def prepare_vv(s0_vv: np.ndarray, smooth: int = 5):
    """Boxcar-smoothed VV sigma0 in dB (on multilooked pixels) and the valid-data mask (swath
    borders and low-valued no-data columns removed)."""
    valid = ndimage.binary_erosion(s0_vv > 0, iterations=3)
    db = 10 * np.log10(np.maximum(ndimage.uniform_filter(s0_vv, smooth), 1e-6))
    col_med = np.nanmedian(np.where(valid, db, np.nan), axis=0)
    valid &= ~ndimage.binary_dilation(~(col_med > -25), iterations=5)[None]
    return db, valid


def water_mask(db: np.ndarray, valid: np.ndarray, threshold: float, min_size: int = 10):
    """VV dB below threshold; small water objects and small land holes removed."""
    water = (db < threshold) & valid
    lab, n = ndimage.label(water)
    sizes = ndimage.sum(water, lab, np.arange(1, n + 1))
    water = np.isin(lab, np.flatnonzero(sizes >= min_size) + 1)
    lab, n = ndimage.label(~water & valid)
    sizes = ndimage.sum(~water, lab, np.arange(1, n + 1))
    water |= np.isin(lab, np.flatnonzero(sizes < min_size) + 1)
    return water & valid
