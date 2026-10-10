"""Datasets over range-compressed scenes.

The model consumes "rows": incoherent averages of `lines_per_row` consecutive echo lines (an onboard
running accumulator), so one recurrent step = one row. Inputs are normalized with fixed, streaming-
compatible offsets (`norm`):
  "none":  subtract one global offset per polarisation (`db_offset`, the median RC power of the
           training scene); the range gain profile (antenna pattern) stays in the input
  "calib": subtract a fixed per-beam range gain profile over slant-range time (scripts/calibrate.py)
Both keep absolute radiometry. A validity channel is always appended.
"""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

DB_SCALE = 3.0  # dB -> network units
NORMS = ("none", "calib")


def offset_normalize(x: np.ndarray, db_offset) -> np.ndarray:
    """Subtract one fixed offset per polarisation (axis 1)."""
    off = np.asarray(db_offset, np.float32)[None, :, None]
    return np.where(np.isfinite(x), (x - off) / DB_SCALE, 0.0)


def calib_normalize(x: np.ndarray, tau: np.ndarray, calib: dict) -> np.ndarray:
    """Subtract the per-beam land profile interpolated at each bin's slant-range time."""
    prof = np.stack([np.interp(tau, calib["tau"], calib[p]) for p in ("vv", "vh")])  # (2, B)
    return np.where(np.isfinite(x), (x - prof[None]) / DB_SCALE, 0.0)


def normalize_rows(rc_rows_db: np.ndarray, norm: str = "none", db_offset=None, tau=None,
                   calib=None) -> np.ndarray:
    """(n_rows, n_pol, n_bins) dB -> normalized float32 channels + validity channel."""
    x = rc_rows_db.astype(np.float32)
    valid = np.isfinite(x).all(1, keepdims=True)
    if norm == "none":
        xn = offset_normalize(x, db_offset)
    elif norm == "calib":
        xn = calib_normalize(x, tau, calib)
    else:
        raise ValueError(f"unknown norm {norm!r}, expected one of {NORMS}")
    return np.concatenate([xn, valid.astype(np.float32)], 1).astype(np.float32)


def load_calib(data_root: Path, beam: str) -> dict:
    c = np.load(Path(data_root) / "calib" / f"{beam}.npz")
    return {k: c[k] for k in ("tau", "vv", "vh")}


class Scene:
    def __init__(self, proc_dir: Path, norm: str = "none", db_offset=None):
        self.dir = Path(proc_dir)
        self.meta = json.load(open(self.dir / "annot_meta.json"))
        rc_meta = json.load(open(self.dir / "rc_meta.json"))
        self.beam = rc_meta["raw"][0].split("_")[1]
        rows = np.load(self.dir / "rc_rows_db.npy")
        tau = rc_meta["tau_start"] + (np.arange(rows.shape[2]) + 0.5) * rc_meta["dtau"]
        calib = load_calib(self.dir.parents[1], self.beam) if norm == "calib" else None
        self.x = normalize_rows(rows, norm, db_offset, tau, calib)  # (R, C, B)
        self.y = np.load(self.dir / "labels.npy")  # (R, B), NaN = unlabelled
        assert self.x.shape[0] == self.y.shape[0]

    @property
    def n_rows(self):
        return self.x.shape[0]


class WindowDataset(Dataset):
    """Random (azimuth window x range crop) samples from row intervals of a scene.

    Each sample has T = warmup + delay + target_len rows. The loss is applied to model outputs at
    steps t >= warmup + delay, which predict rows t - delay (so every supervised row has `delay` rows
    of look-ahead and at least `warmup` rows of history)."""

    def __init__(self, scene: Scene, row_ranges, warmup=64, delay=32, target_len=128, width=512,
                 n_samples=2000, water_oversample=0.5, seed=0):
        self.s = scene
        self.T = warmup + delay + target_len
        self.ranges = [(a, b) for a, b in row_ranges if b - a > self.T]
        lens = np.array([b - a for a, b in self.ranges], float)
        self.range_p = lens / lens.sum()
        self.warmup, self.delay, self.width = warmup, delay, width
        self.n = n_samples
        self.rng = np.random.default_rng(seed)
        self.water_oversample = water_oversample
        # rows containing water (for oversampling), per range
        self.water_rows = [np.flatnonzero((np.nan_to_num(scene.y[a:b]) > 0.5).any(1)) + a
                           for a, b in self.ranges]

    def __len__(self):
        return self.n

    def _start(self):
        B = self.s.x.shape[2]
        k = int(self.rng.choice(len(self.ranges), p=self.range_p))
        r0, r1 = self.ranges[k]
        if self.rng.random() < self.water_oversample and len(self.water_rows[k]):
            r = int(self.rng.choice(self.water_rows[k]))
            t0 = r - self.warmup - self.delay - int(self.rng.integers(0, self.T - self.warmup - self.delay))
            cols = np.flatnonzero(np.nan_to_num(self.s.y[r]) > 0.5)
            c = int(self.rng.choice(cols)) - int(self.rng.integers(0, self.width))
        else:
            t0 = int(self.rng.integers(r0, r1 - self.T))
            c = int(self.rng.integers(0, B - self.width))
        t0 = int(np.clip(t0, r0, r1 - self.T))
        c = int(np.clip(c, 0, B - self.width))
        return t0, c

    def __getitem__(self, i):
        t0, c = self._start()
        x = self.s.x[t0:t0 + self.T, :, c:c + self.width].copy()          # (T, C, W)
        y = self.s.y[t0 + self.warmup:t0 + self.T - self.delay, c:c + self.width].copy()  # (L, W)
        if self.rng.random() < 0.5:  # range flip
            x, y = x[..., ::-1].copy(), y[..., ::-1].copy()
        x[:, :-1] += self.rng.normal(0, 0.1)  # global gain jitter (normalized units)
        if self.rng.random() < 0.1:  # polarisation dropout (VH)
            x[:, 1] = 0
        return torch.from_numpy(x), torch.from_numpy(y)
