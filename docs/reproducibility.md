# Paper results and evaluation

The results below are those presented in the paper.

## Synthetic pulse reconstruction

The synthetic evaluation uses 1,000 validation pulses, the final COPRA
reconstruction for each pulse, and 64 Euler refinement steps.

| Method | Mean pulse MAE ↓ | Mean trace SSIM ↑ |
| --- | ---: | ---: |
| COPRA | 0.468531 | 0.970920 |
| COPRA + direct Hann | 0.334460 | 0.820689 |
| COPRA Flow | 0.067425 | 0.910295 |

The model was trained on 9,000 pulses, with 30 source–target pairs per pulse.
The validation partition was used for checkpoint selection. Pulse MAE compares
complex fields after resolving circular shift, complex scaling, and
conjugate-reversal ambiguities. Trace SSIM uses global image statistics on
independently peak-normalized traces.

## Experimental TG-FROG measurements

The paper evaluates five IR and five UV measurements. Results are averaged
across paired initializations with seeds 11, 29, and 47. Retrieval uses 400
iterations; COPRA Flow uses 64 Euler refinement steps.

| Method | Mean trace NRMSE ↓ | Mean trace SSIM ↑ | Mean amplitude roughness ↓ |
| --- | ---: | ---: | ---: |
| COPRA | 0.013481 | 0.910208 | 0.179120 |
| PnP-COPRA | 0.014060 | 0.925819 | 0.159540 |
| COPRA Flow | 0.014705 | 0.919202 | 0.087340 |

Trace NRMSE uses best-fit intensity scaling. Amplitude roughness is the RMS
second difference of the peak-normalized temporal amplitude. Experimental
measurements do not provide a ground-truth pulse; roughness measures smoothness.

## Evaluation commands

To evaluate the released model on the synthetic validation set:

```bash
uv run python reproduction/synthetic_benchmark.py \
  --checkpoint checkpoints/copra-flow.pt --data-dir data/flow_matching_10k \
  --out-dir runs/benchmark
```

The command saves predictions and calculated metrics in the output directory.
The paper's numerical results are provided in `reproduction/reference_results/`.
The pulse split is provided in `reproduction/selections/synthetic_split.json`.

For experimental inputs and preprocessing, see the [inference guide](inference.md).
Experimental measurements must be supplied separately.
