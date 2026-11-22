# Benchmarks and paper results

`synthetic_benchmark.py` evaluates the fixed 1,000-pulse validation set using
64 Euler steps. It saves predictions and calculated metrics. Install the
package and download the model and dataset before running it.

`reference_results/` contains the numerical results presented in the paper.
`selections/synthetic_split.json` defines the training and validation pulses.

`pnp_copra_benchmark.py` reproduces the in-loop PnP-COPRA comparison using
the same dataset and saved synthetic predictions. It supports the fixed
50-pulse tolerance pilot, the full validation set, and resuming a run.
`pnp_copra_backend.py` contains the trace caching and scalar FFT acceleration
used during the experiment. The retrieval algorithm remains in
`src/pnp_copra/retrieval.py`.

See the [PnP-COPRA guide](../docs/pnp_copra.md) for commands.
`reference_results/pnp_copra/` contains the full per-pulse CSV, summaries,
pilot results, and stock-solver verification. The reference CSV is provided
to audit the reported statistics without rerunning retrieval.

To plot the paper's 20 selected synthetic examples, use:

```bash
uv run copra-flow visualize --predictions-dir runs/benchmark/predictions \
  --out-dir runs/benchmark/figures \
  --selection-json reproduction/selections/validation_20.json
```

For experimental measurements, use `copra-flow benchmark-experimental` as
described in [the inference guide](../docs/inference.md).
`selections/experimental_10.json` lists the paper's ten measurement UIDs.
Experimental measurements must be supplied separately.
