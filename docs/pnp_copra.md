# Synthetic PnP-COPRA benchmark

This benchmark runs the existing `AbhiPnPCOPRAClassic` retriever on the
same noisy synthetic TG-FROG measurements and initialization seeds used
for the COPRA baseline. It retains the Hann denoiser, soft temporal support,
400 iterations, COPRA step scale 0.25, and pyPret's original rule for
returning the spectrum with the best trace-error score.

## Backtracking tolerance

After a global COPRA update, the retriever blends its spectrum with a
Hann-smoothed copy. Let `S(E)` be the global SSIM of the measured and
simulated traces. A mixture is accepted when

```text
E_mix = (1 - lambda) E_COPRA_update + lambda E_smoothed
S(E_mix) >= (1 - delta) S(E_COPRA_update)
```

The mixing weight starts at 0.7 and is halved after a rejection while
it remains at least 0.05. If no mixture passes, the COPRA update is kept.
The reported result uses `delta = accept_tol = 0.01`: a maximum 1% relative
SSIM decrease per smoothing decision. This is a comparison with the
unsmoothed update, not a 1% bound on the final trace's difference from the
measurement. The separate final SSIM floor of 0.30 is a reporting check;
it does not replace or reject the solver's output.

Only the acceptance tolerance was changed for the reported comparison.
The library's default tolerance remains `1e-6`. Direct Hann is a separate
post-retrieval smoothing baseline.

## Results and selection

A proportional 50-pulse sample across all 17 pulse families was selected
with seed 42. Its UIDs are fixed in
[`pnp_copra_pilot.json`](../reproduction/selections/pnp_copra_pilot.json).
The coarse sweep tested 0.0, 0.1, ..., 0.7. A finer tolerance-only follow-up
selected 0.01, which reduced pilot mean pulse MAE from 0.4585 to 0.3984.
This setting was fixed before evaluating the remaining pulses.

| Method | Mean pulse MAE | Mean trace SSIM |
| --- | ---: | ---: |
| COPRA | 0.468531 | 0.970920 |
| PnP-COPRA, delta=0.01 | 0.410690 | 0.946159 |
| COPRA + direct Hann | 0.334460 | 0.820689 |
| COPRA Flow | 0.067425 | 0.910295 |

PnP-COPRA improves field MAE for 884 of 1,000 pulses and reduces its mean
by 12.3% relative to COPRA. Mean MAE is 0.411339 on the 950 pulses outside
the pilot. It preserves more trace agreement than direct Hann, while
its field MAE remains higher. The saved results are in
[`reference_results/pnp_copra`](../reproduction/reference_results/pnp_copra/).
`provenance.json` records settings, versions, and result-file checksums.

## Reproduce

Install the package and download the dataset and checkpoint as described
in the main README. First run `reproduction/synthetic_benchmark.py` to
create `runs/benchmark/predictions` for the fixed validation split.
PnP retrieval then uses those predictions only for paired baseline scores
and the saved target used for evaluation; the target never enters retrieval.

Run the full comparison at the fixed tolerance:

```bash
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 \
uv run python reproduction/pnp_copra_benchmark.py \
  --data-dir data/flow_matching_10k \
  --predictions-dir runs/benchmark/predictions \
  --out-dir runs/pnp-copra --tolerances 0.01 --workers 4
```

Add `--pilot` to run only the fixed 50 pulses. Use a separate output
directory and `--tolerances 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7` for the
coarse sweep, or `--tolerances 0.01 0.02` for the finer follow-up.
The script never automatically launches a full run after a pilot.

Rerun the same command to resume. It saves each returned field and metrics
under `runs/tol_*/`, followed by a per-pulse CSV and summary JSON.
Input-manifest, code, version, and settings checks prevent resuming into
a directory from a different run. `--summarize-only` recomputes the CSV
and summary from completed results; `--limit 2` makes a small smoke run.

The default `--backend cached` reuses unchanged traces with stock pyPret
FFTs. `--backend stock` disables this acceleration. The archived full run
used `--backend scalar`, which also batches array operations across delays
while retaining scalar FFTs. That backend needs SciPy's DUCC internals
(the recorded run used SciPy 1.18.0). Use `--verify-only` to require exact
agreement of returned and last spectra and backtracking decisions with
stock pyPret on two pulses before a benchmark.

The runner checks regenerated traces against their saved float32 values.
When they match, it retains the original float64 trace to reproduce the
dataset-generation calculation. Otherwise, it uses the saved trace and
records that precision restoration was unavailable. All 1,000 archived
results restored the original precision. Exact numerical reproduction
also depends on the recorded NumPy/SciPy versions.
