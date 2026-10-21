# Report reproduction

`synthetic_benchmark.py` regenerates 64-step Euler predictions for the fixed
1,000 validation pulses and compares the quantitative result with
`reference_results/synthetic_metrics.json`. Install the package before running.
No copies of the weights or training data are embedded in this repository.

Use `copra-flow visualize --predictions-dir RUN/predictions --out-dir RUN/figures
--selection-json reproduction/selections/validation_20.json` for the fixed
20-pulse panel. The output includes PNG, PDF and a JSON manifest.

`copra-flow benchmark-experimental` runs the final experimental comparison;
see `docs/inference.md`. `selections/experimental_10.json` lists the exact
report UIDs. The measurements must be supplied separately with a path manifest.

The source manifest records pre-curation file checksums, not checksums of the
modified public files. Verification results must record the final release
source checksums as well as dataset and checkpoint checksums.
