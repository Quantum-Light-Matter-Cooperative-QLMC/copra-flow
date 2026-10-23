# Array formats and physical conventions

All NPZ files use numeric arrays or JSON stored as Unicode scalars. Load them
with `numpy.load(path, allow_pickle=False)`. Fields are complex electric-field
envelopes; amplitude is `abs(field)` and intensity is `abs(field)**2`.

## Prepared inference inputs

| Key | Shape | Meaning |
| --- | --- | --- |
| `measured_trace` | N × N | Nonnegative intensity; rows = delay, columns = frequency |
| `time_axis` | N | Centered time in seconds, spacing 2.5e-15 |
| `delay_axis` | N | Centered delay in seconds, spacing 2.5e-15 |
| `freq_axis` | N | Relative angular frequency in rad/s, increasing |
| `wl0` | scalar | Physical carrier wavelength in metres |
| `source_field` | N, optional | Existing COPRA field on `time_axis` |

`trace_linear` is accepted as an alias for `measured_trace`. Absolute angular
frequencies are also accepted if subtracting the carrier gives the exact model
grid. The released checkpoint expects N=256. Do not pass Hz, PHz, wavelength
coordinates or transposed arrays under `freq_axis`. The code checks the physical
grid rather than silently stretching a trace.

Generate a valid example with `copra-flow example --out data/example.npz`.
For native measurements, a tab-delimited text file has one numeric header row
containing delays in fs, one header column containing wavelengths in nm, and
a numeric top-left placeholder. Interior rows are wavelengths and columns are
delays. Negative detector values are allowed before preprocessing. Native
axes must be strictly monotone with no duplicates or nonfinite entries.

## Synthetic training shards

One pulse UID identifies one shard. Important arrays are:

| Key | Shape / dtype | Meaning |
| --- | --- | --- |
| `target_field` | N, complex64 | Clean field, RMS-normalized |
| `source_fields` | K × N, complex64 | Sources aligned to the target and normalized identically |
| `source_fields_raw` | K × N, complex64 | Before target ambiguity alignment, still normalized |
| `source_spectra_raw` | K × N, complex64 | Original COPRA/augmentation spectra |
| `source_spectra_aligned` | K × N, complex64 | Aligned source spectra |
| `measured_trace` | N × N, float32 | Peak-normalized simulated TG-FROG trace plus noise/clipping |
| `time_axis`, `delay_axis` | N, float64 | Seconds |
| `freq_axis` | N, float64 | Absolute angular-frequency axis in rad/s at 800 nm |
| `time_window` | N, float32 | Trace-derived soft temporal support |
| `source_kind`, `source_detail` | K, Unicode | Source provenance |
| `source_weight` | K, float32 | Training loss weight |
| `hard_negative` | K, bool | Candidate marked as difficult or augmented |
| `mae_aligned`, `trace_error`, `trace_ssim` | K | Stored diagnostics |
| `spec_json`, `metadata_json` | Unicode scalar | Pulse specification and generation provenance |

The dataset uses K=30: 17 COPRA snapshots, one final COPRA field, direct Hann
and guarded Hann final fields, two smoothed snapshots, and eight hard negatives.
The `hard_negative` mask also flags COPRA candidates that meet diagnostic
difficulty criteria; its count can exceed eight. The released dataset has
80,000 generated hard-negative candidates and 115,347 flagged candidates in
total. Use `source_kind` to distinguish generated augmentations from flagged
COPRA reconstructions.
Raw source arrays in a shard are not physical calibrated fields: they have the
stored RMS normalization. Recover original scales from metadata if needed.
Training and synthetic evaluation use target-aligned `source_fields`, whereas
experimental inference applies target-free centering and branch selection.

Additional arrays preserve normalization scales and diagnostics from the
original generator. Their presence does not make them model inputs: the model
uses two-channel real/imaginary fields and a peak-normalized, square-root trace.
`target_spectrum` is stored in the generator's physical scaling, unlike the
RMS-normalized `target_field`; use the documented field arrays for training.

## Split manifests

A version-1 JSON split has `unit: "pulse_uid"` and disjoint `train` and `val`
lists. `excluded` optionally records unused UIDs in limited training runs.
The complete manifest must match the dataset membership; relocation preserves
UIDs. Shuffling individual source rows would leak related pulses across splits.

## Prediction artifacts

Synthetic benchmark prediction shards store target, source, prediction, Hann
baselines, axes and measurement. They support metrics and visualization without
rerunning the neural sampler. Synthetic predictions and target-free inference
outputs have different roles and are documented by their respective commands.
