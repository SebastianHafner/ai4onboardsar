# ai4onboardsar

Water detection on **range-compressed** (not azimuth-compressed) Sentinel-1 stripmap data, with a
streaming model that consumes echo lines as they arrive — a feasibility study for onboard SAR flood
detection.

## Pipeline

```
download.py        GRD (labels) + matching RAW L0 (input) from CDSE, per scene in configs/scenes.yaml
range_compress.py  L0 decode (sentinel1decoder) -> chirp replica from packet headers -> matched filter
                   -> |.|^2, Rx-gain corrected, block-averaged in range to ~21 m slant bins, on a fixed
                   slant-range grid (SWST changes compensated)        -> rc_db.npy (lines, pol, bins)
annotate.py        GRD sigma0 VV -> scene-wide threshold water mask -> GRD pixels mapped to (zero-Doppler
                   time, slant-range time) via the geolocation grid -> (echo line, range bin)
                   -> label cells of 16 lines x 1 bin; residual offset estimated by cross-correlating
                   the projected GRD with the unfocused RC image  -> labels.npy, rc_rows_db.npy, quicklook.png
calibrate.py       per-beam range gain profile (land median vs slant-range time) -> calib/<beam>.npz
train.py           streaming model on one scene (a held-out azimuth block = validation)
evaluate.py        full-scene streaming inference on the test scenes + threshold baseline
infer_raw.py       deployable chain: raw L0 echo lines -> front end (frontend.py) -> network, streaming
```

Data lives in `data_root` (configs/scenes.yaml); CDSE credentials in `.env`
(`CDSE_USERNAME`, `CDSE_PASSWORD`).

```bash
conda activate ai4onboardsar
python scripts/download.py
for s in s4_20250422 s4_20250329 s4_20250427 s2_20250405; do
  python scripts/range_compress.py --scene $s && python scripts/annotate.py --scene $s
done
python scripts/calibrate.py --scenes s4_20250422   # one per beam
python scripts/calibrate.py --scenes s2_20250405
python scripts/train.py --config configs/train.yaml
python scripts/evaluate.py --run <data_root>/runs/<run>
python scripts/infer_raw.py --run <data_root>/runs/<run> --scene s4_20250329
```

## Model

The input stream is one echo line per PRI. Lines are accumulated incoherently into rows of 16 lines
(about 66 m in azimuth); each row is one recurrent step:

1. **Normalization**, set by `norm`, both fixed offsets that keep absolute radiometry:
   - `none` (default): subtract one global offset per polarisation (`db_offset`, the median RC
     power of the training scene). The range gain profile (antenna pattern) stays in the input.
   - `calib`: subtract a fixed per-beam range profile over slant-range time (antenna pattern and
     spreading loss removed). Likely needed to transfer across beams; not yet testable.
   A relative normalization (running mean per range bin) was dropped: smooth, dark snowfields then
   look like water.
2. **Conv along range** (kernel 5, 16 channels, ReLU), applied to every row.
3. **GRU per range bin** over slow time (16 hidden units). Range compression has already focused
   energy in range, so the remaining smear (the synthetic aperture, about 3.8 km or about 56 rows)
   is purely along azimuth. Weights are shared across range bins; each bin has its own state.
4. **Linear output** per bin: water logit.

This is `minimal` (default, 1.9k parameters, about 4.8M multiply-adds per row of 2667 bins).
`bingru` adds a 3-layer dilated CNN encoder and a conv head (7.2k parameters, 4x the compute) and
was not better in the screening experiments. Output at step *t* predicts row *t − D* (`delay`,
default 32 rows, about half the aperture), so the latency is fixed at about 2 km of flight. Other
alternatives: `convgru` (ConvGRU along range, slower) and `tcn` (causal dilated conv).

**Deployment.** Range compression, range binning, row averaging and normalization are a fixed
front end without trainable parameters (`ai4onboardsar/frontend.py`, torch). Training uses its
cached output; `infer_raw.py` runs front end + network on raw echo lines and reproduces the cached
input to within 0.01 dB. The front end (about 17 GFLOP/s for two polarisations at the S1 PRF)
dominates the compute; the network needs about 1 GFLOP/s.

Baseline: threshold on the normalized VV power averaged over ±D rows, with the threshold fitted
on the training scene.

## Scenes

| id | split | beam | notes |
|---|---|---|---|
| s4_20250422 | train (+val) | S4 | Belgium to the Rhine–Meuse delta; two GRD slices, one datatake |
| s4_20250329 | test | S4 | Thames / London / Channel coast |
| s4_20250427 | false-positive test | S4 | northern Sweden (67°N), lakes ice-covered, no open water |
| s2_20250405 | false-positive test | S2 | northern Sweden, other beam; S2 calib profile from this scene |

The water labels come from thresholding the GRD VV: 2×2 multilook, 5×5 boxcar, and the threshold
at the valley of the histogram of bimodal tiles. They are good for open water (sea, estuaries,
large rivers), with some false positives on smooth fields and none under ice.
