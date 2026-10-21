# Code verification — October 20, 2026

The curated release was tested independently from the research checkout.
Machine-readable evidence is in
`reproduction/reference_results/code_verification.json`.

| Check | Result |
| --- | --- |
| Automated tests | 24 passed |
| Dataset generation | Serial and parallel generation; resumption and configuration guards |
| Generator numerical regression | One original shard from each of 17 families: field/trace/weight arrays identical |
| Training | Real optimizer updates, validation, best/last checkpoints and reload |
| Full report architecture | Short MPS training run on original dataset shards |
| Inference | Trace-only, supplied-COPRA-field, Euler and RK4 |
| Evaluation | Direct evaluation and saved prediction evaluation agree |
| Full synthetic report benchmark | 1,000 validation pulses; six reference means within 1e-4 |
| Report panel | All 20 selected PNG/PDF comparisons generated; visual inspection performed |
| Experimental execution | Ten traces × three seeds × three methods; 90 outputs |
| Experimental preparation port | All ten prepared model inputs identical to original code |
| Original retrieval control | One trace, seed 11: all three reconstructed traces identical |
| Distribution | Wheel and source distribution built; installed wheel exercised outside checkout |

The experimental historical reference comparison is **different**, not passed:
the current original research code also fails to match the old table exactly.
See the actual values and controls in `reproducibility.md`. Successful execution
is distinguished from historical numerical agreement.

The reference interpreter is Python 3.12.2 with PyTorch 2.5.1, NumPy 2.5.1 and
SciPy 1.18.0. The initial complete experimental rerun used NumPy 2.5.4 and SciPy 1.18.1;
the original-code control and the final test environment used the pinned
versions above. Per-check environments are recorded with the JSON results.
The published lockfile fixes compatible versions for the other
supported Python versions. CPU and Apple MPS paths were exercised. CUDA/AMP
execution needs an NVIDIA device and was not tested on this Mac. A short
training run verifies execution; the full 50-epoch A100 training was not rerun.

Run `uv run pytest -q` and `uv run python scripts/verify_release.py` for local
checks. The latter creates its own small dataset and model. Reproducing the
scientific benchmark additionally requires the separately released checkpoint
and complete dataset. GitHub Actions runs the data-free checks on Linux.
