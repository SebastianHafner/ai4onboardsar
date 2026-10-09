"""Sentinel-1 Level-0 decoding and range compression (no azimuth compression).

Output convention (per scene): an array of shape (n_lines, n_pol, n_bins) holding range-compressed
power, block-averaged over `range_dec` samples in range (incoherent), on a fixed slant-range-time
grid that is common to all lines (compensating SWST changes). One row = one echo line (one PRI).
"""
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sentinel1decoder import Level0Decoder, SignalType
from sentinel1decoder import _field_names as fn
from sentinel1decoder.constants import F_REF

GPS_EPOCH = datetime(1980, 1, 6, tzinfo=timezone.utc)
GPS_UTC_LEAP = 18.0  # GPS - UTC [s], valid since 2017
# Delay between the SWST reference and the first sample (S1-IF-ASD-PL-0007, 320 / (8 F_REF))
DELTA_T_SUPPRESSED = 320 / (8 * F_REF)
# Reference epoch for storing times as float seconds (UTC)
TIME_REF = datetime(2025, 1, 1, tzinfo=timezone.utc)


def utc_seconds(dt: datetime) -> float:
    """Seconds since TIME_REF for a (naive = UTC) datetime."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return (dt - TIME_REF).total_seconds()


def l0_files(safe_dir: Path) -> dict[str, Path]:
    """Map polarisation ('vv', 'vh', ...) to the measurement .dat file of a RAW SAFE product."""
    out = {}
    for f in sorted(Path(safe_dir).glob("*.dat")):
        if f.name.endswith(("-annot.dat", "-index.dat")):
            continue
        out[f.name.split("-")[4]] = f
    return out


@dataclass
class EchoMeta:
    """Per-echo-line metadata for one L0 measurement file (echo packets only)."""
    packets: pd.DataFrame  # decoded headers of echo packets, original multiindex
    t_utc: np.ndarray      # sensing time [s since TIME_REF]
    tau0: np.ndarray       # slant-range (two-way) time of first sample [s]
    fs: np.ndarray         # range sampling rate [Hz]
    pri_count: np.ndarray


def read_echo_meta(dat_file: Path) -> tuple[Level0Decoder, EchoMeta]:
    dec = Level0Decoder(str(dat_file))
    df = dec.decode_metadata()
    df = df[df[fn.SIGNAL_TYPE_DECODED] == SignalType.ECHO]
    gps = df[fn.COARSE_TIME_DECODED].to_numpy(dtype=np.float64) + df[fn.FINE_TIME_DECODED].to_numpy(np.float64)
    t_utc = gps - GPS_UTC_LEAP + (GPS_EPOCH - TIME_REF).total_seconds()
    pri = df[fn.PRI_DECODED].to_numpy(np.float64)
    # Packet time stamps are quantized (2^-16 s); within a datatake the PRI is constant, so use
    # t = t0 + PRI * (PRI count - c0) with t0 fitted robustly.
    count = df[fn.PRI_COUNT_DECODED].to_numpy(np.int64)
    pri_c = float(np.median(pri))
    assert np.allclose(pri, pri_c), "PRI changes within file"
    rel = (count - count[0]) * pri_c
    t_utc = np.median(t_utc - rel) + rel
    rank = df[fn.RANK_DECODED].to_numpy(np.float64)
    swst = df[fn.SWST_DECODED].to_numpy(np.float64)
    tau0 = rank * pri + swst + DELTA_T_SUPPRESSED
    fs = np.array([r.sample_rate_hz for r in df[fn.RANGE_DEC_DECODED]])
    return dec, EchoMeta(df, t_utc, tau0, fs, df[fn.PRI_COUNT_DECODED].to_numpy(np.int64))


def chirp_replica(txpl: float, txprr: float, txpsf: float, fs: float) -> np.ndarray:
    """Nominal transmitted chirp sampled at fs (from packet header parameters)."""
    n = int(np.floor(txpl * fs))
    t = np.arange(n) / fs
    return np.exp(2j * np.pi * (txpsf * t + 0.5 * txprr * t**2)).astype(np.complex64)


def range_compress(echo: np.ndarray, replica: np.ndarray) -> np.ndarray:
    """Matched filter along fast time. Output sample k corresponds to a target whose echo starts at
    sample k of the receive window, i.e. at slant-range time tau0 + k / fs. Samples whose replica
    support extends past the window end are set to 0."""
    n_s = echo.shape[-1]
    n_fft = int(2 ** np.ceil(np.log2(n_s + len(replica))))
    H = np.conj(np.fft.fft(replica, n_fft))
    H /= np.sqrt(len(replica))  # unit noise gain
    rc = np.fft.ifft(np.fft.fft(echo, n_fft, axis=-1) * H, axis=-1)[..., :n_s]
    rc[..., n_s - len(replica) + 1:] = 0
    return rc.astype(np.complex64)


def decode_range_compressed_power(dec: Level0Decoder, meta: EchoMeta, rows: np.ndarray,
                                  batch: int = 1024):
    """Yield (row_indices, power) for the given echo rows (indices into meta.packets), where power is
    |range-compressed|^2 corrected for Rx gain, at full range sampling. Rows are grouped into
    batches with constant packet geometry (same chunk)."""
    pk = meta.packets.iloc[rows]
    chunk_ids = pk.index.get_level_values(fn.ACQUISITION_CHUNK_NUM_DECODED).to_numpy()
    # split into runs of constant chunk id, then into batches
    breaks = np.flatnonzero(np.diff(chunk_ids)) + 1
    for run in np.split(np.arange(len(rows)), breaks):
        for s in range(0, len(run), batch):
            sel = run[s:s + batch]
            sub = pk.iloc[sel]
            echo = dec.decode_packets(sub)
            r0 = sub.iloc[0]
            fs = r0[fn.RANGE_DEC_DECODED].sample_rate_hz
            rep = chirp_replica(r0[fn.TX_PULSE_LEN_DECODED], r0[fn.TX_RAMP_RATE_DECODED],
                                r0[fn.TX_PULSE_START_FREQ_DECODED], fs)
            rc = range_compress(echo, rep)
            pw = (rc.real**2 + rc.imag**2).astype(np.float32)
            gain = sub[fn.RX_GAIN_DECODED].to_numpy(np.float64)  # [dB]
            pw /= (10 ** (gain / 10)).astype(np.float32)[:, None]
            yield rows[sel], pw, len(rep)


def block_mean(x: np.ndarray, k: int) -> np.ndarray:
    n = x.shape[-1] // k
    return x[..., :n * k].reshape(*x.shape[:-1], n, k).mean(-1)
