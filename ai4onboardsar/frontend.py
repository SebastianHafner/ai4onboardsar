"""Fixed (non-trainable) front end: raw echo lines -> normalized model input rows.

This is the deployable form of the preprocessing that produces the training input
(scripts/range_compress.py + the row averaging in scripts/annotate.py + data.normalize_rows):

  echo line (complex samples, per polarisation)
    -> range compression: matched filter with the nominal chirp replica (FFT), |.|^2, Rx gain
    -> range bins: incoherent mean over `range_dec` samples, placed on the fixed slant-range grid
    -> rows: incoherent mean over `lines_per_row` consecutive lines (running accumulator), dB
    -> normalization (fixed offset or per-beam profile) + validity channel

Training uses the cached output of the same chain, so the deployed model is `Frontend` followed by
the trained network. Onboard, the echo samples are available before FDBAQ compression, so the L0
decoding step (sentinel1decoder) is not part of the deployed chain.
"""
import numpy as np
import torch
import torch.nn as nn

from .data import DB_SCALE


class RangeCompression(nn.Module):
    """Matched filter along fast time, power and Rx gain correction. Output sample k corresponds to
    slant-range time tau0 + k / fs; only fully supported samples (n_samples - len(replica) + 1)
    are returned. Same as l0.range_compress, in torch."""

    @staticmethod
    def forward(echo: torch.Tensor, replica: torch.Tensor, rx_gain_db: torch.Tensor) -> torch.Tensor:
        n_s, n_r = echo.shape[-1], replica.shape[-1]
        n_fft = 1 << int(np.ceil(np.log2(n_s + n_r)))
        H = torch.conj(torch.fft.fft(replica, n_fft)) / np.sqrt(n_r)  # unit noise gain
        rc = torch.fft.ifft(torch.fft.fft(echo, n_fft) * H)[..., :n_s - n_r + 1]
        pw = rc.real ** 2 + rc.imag ** 2
        return pw / 10 ** (rx_gain_db[:, None].to(pw.dtype) / 10)


class Frontend(nn.Module):
    """Streaming front end. Feed echo lines in acquisition order with `push`; completed rows are
    returned as (n_rows, n_pol + 1, n_bins) model input."""

    def __init__(self, tau_start: float, dtau: float, n_bins: int, range_dec: int, lines_per_row: int = 16,
                 norm: str = "none", db_offset=None, calib: dict | None = None, n_pol: int = 2):
        super().__init__()
        self.tau_start, self.dtau, self.n_bins = tau_start, dtau, n_bins
        self.range_dec, self.k, self.n_pol = range_dec, lines_per_row, n_pol
        self.compress = RangeCompression()
        if norm == "none":
            ref = torch.tensor(db_offset, dtype=torch.float32)[:, None].expand(n_pol, n_bins)
        elif norm == "calib":
            tau = tau_start + (np.arange(n_bins) + 0.5) * dtau
            ref = torch.from_numpy(np.stack([np.interp(tau, calib["tau"], calib[p]) for p in ("vv", "vh")]))
        else:
            raise ValueError(norm)
        self.register_buffer("ref_db", ref.float().contiguous())  # (n_pol, n_bins)
        self.reset()

    def reset(self):
        self.acc = torch.zeros(self.n_pol, self.n_bins, dtype=torch.float64)
        self.cnt = torch.zeros(self.n_pol, self.n_bins, dtype=torch.float64)
        self.n_acc = 0

    def to_grid(self, pw: torch.Tensor, tau0: np.ndarray) -> torch.Tensor:
        """(N, n_samples) power -> (N, n_bins) on the slant-range grid (NaN outside the window)."""
        n = pw.shape[1] // self.range_dec
        binned = pw[:, :n * self.range_dec].reshape(len(pw), n, self.range_dec).mean(-1)
        out = torch.full((len(pw), self.n_bins), float("nan"), dtype=pw.dtype)
        off = np.round((np.asarray(tau0) - self.tau_start) / self.dtau).astype(int)
        for o in np.unique(off):  # SWST changes shift the window
            k = torch.from_numpy(off == o)
            w = min(n, self.n_bins - o)
            out[k, o:o + w] = binned[k, :w]
        return out

    def lines(self, echoes, replicas, rx_gains_db, tau0s) -> torch.Tensor:
        """Per polarisation: echoes (N, n_samples) complex, replica, gains (N,), tau0 (N,) ->
        power of N lines on the grid, (N, n_pol, n_bins)."""
        grids = []
        for echo, rep, gain, tau0 in zip(echoes, replicas, rx_gains_db, tau0s):
            pw = self.compress(torch.as_tensor(echo), torch.as_tensor(rep), torch.as_tensor(gain))
            grids.append(self.to_grid(pw, tau0))
        return torch.stack(grids, 1)

    def push_lines(self, pw_lines: torch.Tensor) -> torch.Tensor:
        """Accumulate (N, n_pol, n_bins) line power; return the completed, normalized rows."""
        rows = []
        for line in pw_lines.double():
            ok = torch.isfinite(line)
            self.acc += torch.where(ok, line, 0.0)
            self.cnt += ok
            self.n_acc += 1
            if self.n_acc == self.k:
                rows.append(self.normalize(10 * torch.log10(self.acc / self.cnt)))  # NaN where no data
                self.reset()
        if not rows:
            return torch.zeros(0, self.n_pol + 1, self.n_bins)
        return torch.stack(rows)

    def normalize(self, row_db: torch.Tensor) -> torch.Tensor:
        """(n_pol, n_bins) dB -> (n_pol + 1, n_bins) normalized input with validity channel."""
        row_db = row_db.float()
        ok = torch.isfinite(row_db)
        x = torch.where(ok, (row_db - self.ref_db) / DB_SCALE, 0.0)
        return torch.cat([x, ok.all(0, keepdim=True).float()])

    def push(self, echoes, replicas, rx_gains_db, tau0s) -> torch.Tensor:
        """Raw echo lines (per polarisation, same N lines in acquisition order) -> completed rows."""
        return self.push_lines(self.lines(echoes, replicas, rx_gains_db, tau0s))
