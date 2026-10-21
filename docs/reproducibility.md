# Reproducibility and limits

The release preserves the report's pulse catalogue, normalized measurement
simulation, source construction, architecture, loss, optimizer settings,
Euler integration and metrics. `reproduction/source_manifest.json` records
where the initial curated sources came from and their original checksums.
Portability and validation changes are described below; generated binaries
and historical research files are excluded.

## Scientific reference

- 10,000 independent pulses from 17 families; 30 sources per pulse.
- Grid length 256, spacing 2.5 fs, TG-FROG carrier 800 nm for synthetic data.
- Unit-peak field/clean trace before measurement noise; 1% Gaussian noise
  followed by clipping at zero. Stored target/source fields are RMS-normalized.
- 9,000 training and 1,000 validation pulse UIDs, split with seed 42. The
  explicit report split is `reproduction/selections/synthetic_split.json`.
- 50 training epochs; AdamW; mixed precision on A100; best checkpoint at epoch
  46, update 194074, validation loss 0.017534218672072208.
- Main benchmark uses final COPRA sources only and 64 explicit Euler steps.
- The separate 30,000-source-row validation evaluation is not the headline
  1,000-pulse comparison. The validation partition also selected the checkpoint,
  so it is not an independent test set.

Run `reproduction/synthetic_benchmark.py` with the released data and weights.
The script compares six mean metrics with the stored reference at absolute
numerical tolerance 1e-4. It saves fields, scores, and the comparison result.
A different checkpoint, source selection, pulse partition, ODE solver, or
preprocessing is a different experiment. Numerical differences across devices
can occur; this does not promise identical retraining weights.

## Metric conventions

Pulse MAE compares complex fields after circular shift, complex scale and the
conjugate-reversal ambiguity. Alignment uses the ground truth only in synthetic
data preparation/evaluation. Trace NRMSE uses best-fit intensity scaling and
measured-trace peak normalization. Trace SSIM is **global** mean/variance/
covariance SSIM on independently unit-peak images; it is not window-averaged
`skimage` SSIM. Direct Hann and trace-guarded Hann are distinct baselines.
The best-Hann-by-target-MAE diagnostic is an oracle and not a deployable rule.

The saved report means are COPRA MAE 0.4685310486, direct Hann MAE 0.3344596439,
and Flow MAE 0.0674247687. Their mean Trace SSIM values are 0.9709196563,
0.8206889966 and 0.9102954187. Lower pulse error comes with a decrease in trace
SSIM relative to COPRA. Trace conditioning does not enforce trace consistency.

## Experimental workflow

Native data use wavelength-delay coordinates. The final report preprocessing
uses median edge estimates over the outer 18% of delay columns, a 9-row median
filter, a 21-row quadratic Savitzky–Golay filter, linear interpolation between
left/right edge estimates, subtraction and clipping at zero. Carrier estimation
retains the report's separate 5% border-median convention. Frequency and delay
resampling preserve the fixed grid; traces are then peak-normalized.

The report selected the five highest-SNR eligible IR and five UV traces using
the historical eligibility manifest. Their UIDs are frozen in
`reproduction/selections/experimental_10.json`; do not rerank a new collection.
Run seeds 11, 29, 47 with paired initializations and 400 retrieval iterations.
The learned result is selected between two ambiguity branches by trace NRMSE.
Aggregate metrics use all seeds; report panels choose the seed minimizing
summed NRMSE across the methods. There is no experimental target field, and
roughness/FWHM are not recovery-accuracy metrics.

Experimental data are not included in the code distribution. Do not substitute
the old 20-trace inference collection for the final ten-trace benchmark: the
background subtraction and selection differ.

## Portability changes

There is one package and public CLI. Bundled pypret is installed in the wheel;
no parent checkout or sys.path mutation is needed. Paths are CLI/configuration
inputs. Explicit pulse split manifests can be supplied and are saved per run.
Checkpoint loading uses `weights_only=True`. Configuration validation and
native-input checks fail on unsupported data. Training saves environment
metadata and always writes a final `last.pt` even when checkpoint cadence does
not divide the epoch count. New target-free inference and experimental JSON
outputs consolidate the numerical report helpers without changing their
background, carrier or branch-selection conventions.

## Historical dataset provenance

The original dataset metadata records revision `5056619`, but normalization
changes were committed later in `a7f9791`. The data already contain the later
normalization fields. The committed revision alone therefore does not fully
identify the generator working tree. The public generator incorporates those
normalization changes. Keep the released original shards and checksums as the
canonical training inputs; record regeneration comparisons in verification
results rather than silently rewriting the historical metadata.

Dataset generation records a generator source fingerprint in `run_config.json`.
Prediction directories are bound to a checkpoint checksum, pulse membership and
sampling settings; use a new directory when changing the experiment.

## Historical experimental metrics do not match exactly

The October 20, 2026 verification ran all ten traces, three seeds, and three
methods (90 outputs). It produced these mean values:

| Method | Trace NRMSE | Trace SSIM | Amplitude roughness |
| --- | ---: | ---: | ---: |
| COPRA | 0.013426 | 0.913082 | 0.179189 |
| PnP-COPRA | 0.014067 | 0.924997 | 0.158948 |
| COPRA Flow | 0.014747 | 0.920358 | 0.085489 |

These differ from the historical report table stored in
`reproduction/reference_results/experimental_metrics.json`. Do not substitute
these rerun values for the original report or call the historical comparison
an exact reproduction. The public implementation's prepared arrays match the
current original research scripts byte-for-byte on all ten inputs. A controlled
run on one trace with seed 11 also gives byte-identical reconstructed traces
for COPRA, PnP-COPRA and Flow in the original and public implementations.
The original scripts rerun today therefore inherit the historical discrepancy;
it is not evidence that the port changed those controlled calculations.

The original ten-trace experiment did not preserve complete per-seed numeric
outputs and environment metadata. Its exact historical numeric match remains
unresolved. The release now saves both, pins NumPy/SciPy for each supported
Python version, records source/checkpoint fingerprints, and retains the old
reference separately from current verification results.

`benchmark-experimental --reference-results FILE` always writes the numeric
comparison. A successful benchmark exit means the methods ran, not that the
historical reference matched. Add `--strict-reference` to turn a reference
mismatch into a failing exit code. Synthetic headline verification remains
strict and passes against the report within its documented tolerance.
