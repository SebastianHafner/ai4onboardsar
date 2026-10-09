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
```

## Model

The input stream is one echo line per PRI. Lines are accumulated incoherently into rows of 16 lines
(about 66 m in azimuth); each row is one recurrent step:

1. **Normalization (causal)**, set by `norm`:
   - `calib`: subtract a fixed per-beam range profile over slant-range time. This removes the
     antenna pattern and spreading loss but keeps absolute radiometry.
   - `ema`: subtract a running mean over past rows per range bin (relative contrast only).
     Smooth, dark snowfields then look like water.
2. **Row encoder**: a 1D dilated CNN along range.
3. **Per-range-bin GRU** over slow time (`bingru`, default, ~7k parameters). Range compression has
   already focused energy in range, so the remaining smear (the synthetic aperture, about 3.8 km or
   about 56 rows) is purely along azimuth.
4. **Head**: a conv along range producing water logits.

Output at step *t* predicts row *t − D* (`delay`, default 32 rows, about half the aperture), so the
latency is fixed at about 2 km of flight. Alternatives: `convgru` (ConvGRU along range, slower) and
`tcn` (causal dilated conv).

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
