# Inference

Inference takes a measured TG-FROG trace, its physical axes and carrier
wavelength, and optionally an existing COPRA field. It never reads a target
field or uses pulse MAE to select a reconstruction.

```bash
uv run copra-flow infer --config configs/inference_report.json \
  --checkpoint checkpoints/copra-flow.pt --input measurement.npz --out-dir runs/pulse
```

A `.txt` input is prepared using the paper background subtraction;
a prepared `.npz` input is validated and peak-normalized. Trace-only inputs
run 400 COPRA iterations with seed 11. Supply `--source-field field.npy` to
reuse a COPRA field and skip retrieval. An input NPZ can alternatively contain
`source_field`. A separate supplied file takes precedence.

The source field is centered by intensity centroid, rotated so the peak field
has zero phase, and RMS-normalized. Both direct and conjugate-reversed branches
are refined. The branch with lower measured-trace NRMSE is retained, using only
measurement information. Outputs remain centered and retain FROG ambiguities;
absolute arrival time, phase and calibrated field amplitude are not recovered.

Default sampling is 64-step Euler. `--method rk4` is available for numerical
experiments, but is not the paper setting. Input grids must match the
checkpoint's field length at 2.5 fs spacing; native traces are downsampled or
zero-padded using the fixed physical grid and never upsampled to invent
resolution. Unsupported grids fail explicitly.

`prediction.npz` contains `source_field`, `predicted_field`, `measured_trace`,
`source_reconstructed_trace`, `predicted_reconstructed_trace`, `time_axis`,
`delay_axis`, `freq_axis`, `wl0`, and `metadata_json`. `metrics.json` records
trace metrics, selected branch, centering shift, source RMS, settings and
checkpoint checksum. It reports no pulse accuracy without ground truth.

## Experimental benchmark

`copra-flow benchmark-experimental` accepts a JSON manifest with a `records`
list containing `uid` and `path`. Paths are relative to the manifest. Inputs
can be native text or prepared NPZ files. The paper's ten UIDs are listed in
`reproduction/selections/experimental_10.json`; experimental measurements must be supplied separately.

```bash
uv run copra-flow benchmark-experimental --config configs/experimental_report.json \
  --manifest experimental/manifest.json --checkpoint checkpoints/copra-flow.pt \
  --out-dir runs/experimental
```

All methods receive identical input and paired initializations for seeds
11, 29, and 47. COPRA Flow starts from each paired COPRA field; PnP-COPRA uses
Hann denoising, inferred soft support and trace-SSIM backtracking. All per-seed
fields and metrics are persisted. `metrics.json` also records the seed selected
for each paper panel by minimum summed NRMSE across methods. Aggregate metrics
use all seeds, not just the panel seed. Roughness is RMS second difference of
unit-peak amplitude; FWHM uses intensity around the main lobe.
