# Example input

Create an input without downloading any data:

```bash
copra-flow example --out data/example.npz
```

The command simulates a Gaussian pulse with a small smooth phase modulation,
adds reproducible Gaussian trace noise, and writes measurement/axis arrays
only. It does not include the target pulse. Use it with `copra-flow infer` and
a checkpoint. `scripts/verify_release.py` trains its own small checkpoint when
verifying the complete workflow without external artifacts.
