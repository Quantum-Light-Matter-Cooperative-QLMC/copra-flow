from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from pnp_copra.checkpoints import load_checkpoint, dataset_settings
from pnp_copra.cli import main
from pnp_copra.flow_matching.config import DatasetConfig, ModelConfig, TrainingConfig
from pnp_copra.flow_matching.trainer import train_flow_matching, load_model_from_checkpoint
from pnp_copra.flow_matching.sampling import sample
from pnp_copra.flow_matching.config import SamplingConfig
from pnp_copra.inference import infer, load_measurement
from pnp_copra.experimental import GridSpec, prepare_native, parse_linshan_trace
from pnp_copra.splits import split_payload, read_splits


@pytest.fixture(scope="session", autouse=True)
def limited_threads():
    torch.set_num_threads(2)


def tiny_model(n=32):
    return ModelConfig(field_length=n, hidden_channels=8, residual_blocks=1,
                       condition_dim=16, time_embedding_dim=8, trace_base_channels=4, trace_layers=2)


def make_dataset(root: Path, n=32):
    from pypret import FourierTransform, PNPS, Pulse
    root.mkdir()
    (root / "shards").mkdir()
    ft = FourierTransform(n, dt=2.5e-15)
    for index in range(4):
        pulse = Pulse(ft, 800e-9)
        pulse.field = np.exp(-0.5*(ft.t/(8e-15+index*1e-15))**2).astype(complex)
        field = pulse.field / np.sqrt(np.mean(abs(pulse.field)**2))
        source = field * np.exp(0.1j*np.sin(np.arange(n)))
        pnps = PNPS(pulse, "frog", "tg")
        trace = pnps.calculate(pulse.spectrum, ft.t).copy()
        trace /= trace.max()
        np.savez_compressed(root / "shards" / f"pulse_{index:05d}_gaussian_test.npz",
            target_field=field.astype(np.complex64), source_fields=source[None].astype(np.complex64),
            source_kind=np.asarray(["copra_final"]), source_weight=np.ones(1, dtype=np.float32),
            hard_negative=np.zeros(1, dtype=bool), measured_trace=trace.astype(np.float32),
            time_axis=ft.t, delay_axis=ft.t, freq_axis=pnps.process_w, time_window=np.ones(n))


def test_train_checkpoint_evaluate_and_prediction_roundtrip(tmp_path):
    data = tmp_path / "data"
    make_dataset(data)
    run = tmp_path / "run"
    train_flow_matching(DatasetConfig(data, validation_fraction=0.25), tiny_model(),
                        TrainingConfig(run, epochs=2, batch_size=2, num_workers=0, device="cpu"))
    ck = load_checkpoint(run / "checkpoints/best.pt")
    assert ck["global_step"] > 0
    assert np.isfinite(ck["best_val_loss"])
    assert len(json.loads((run / "history.json").read_text())["history"]) == 2
    assert (run / "checkpoints/last.pt").is_file()
    assert len(read_splits(run / "splits.json", sorted((data / "shards").glob("*.npz")))["val"]) == 1
    args = ["--checkpoint", str(run / "checkpoints/best.pt"), "--data-dir", str(data),
            "--steps", "2", "--workers", "0", "--device", "cpu"]
    out = tmp_path / "metrics.json"
    assert main(["evaluate", *args, "--source-kind", "copra_final", "--trace-metrics",
                 "--out-json", str(out)]) == 0
    metrics = json.loads(out.read_text())
    assert metrics["evaluated_examples"] == 1
    pred = tmp_path / "predictions"
    assert main(["save-predictions", *args, "--out-dir", str(pred)]) == 0
    assert main(["evaluate-predictions", "--predictions-dir", str(pred), "--trace-metrics",
                 "--out-json", str(tmp_path / "saved_metrics.json")]) == 0
    restored = json.loads((tmp_path / "saved_metrics.json").read_text())
    assert restored["predicted_pulse_mae"]["mean"] == pytest.approx(metrics["predicted_pulse_mae"]["mean"])
    assert restored["target_trace_ssim"]["count"] == 1
    assert main(["save-predictions", *args, "--out-dir", str(pred)]) == 0
    assert json.loads((pred / "manifest.json").read_text())["skipped_existing"] == 1
    changed_checkpoint = tmp_path / "different.pt"
    changed = dict(ck)
    changed["epoch"] = 99
    torch.save(changed, changed_checkpoint)
    with pytest.raises(ValueError, match="different or unverified run"):
        main(["save-predictions", *args, "--checkpoint", str(changed_checkpoint), "--out-dir", str(pred)])
    # A relocated run must use its local split file, not its original absolute path.
    relocated = tmp_path / "relocated"
    import shutil
    shutil.copytree(run, relocated)
    shutil.rmtree(run)
    settings = dataset_settings(ck, relocated / "checkpoints/best.pt", data)
    assert Path(settings["split_file"]) == relocated / "splits.json"


def test_inference_needs_no_target_and_ignores_target_payload(tmp_path):
    from pnp_copra.flow_matching.model import FlowMatchingModel
    n = 256
    config = tiny_model(n)
    model = FlowMatchingModel(config)
    from dataclasses import asdict
    checkpoint = tmp_path / "model.pt"
    torch.save({"model": model.state_dict(), "configs": {"model": asdict(config),
               "dataset": {"trace_transform": "sqrt", "normalize_trace": True}}}, checkpoint)
    input_path = tmp_path / "measurement.npz"
    main(["example", "--out", str(input_path)])
    first = infer(input_path, checkpoint, tmp_path / "trace", steps=2, maxiter=2, device="cpu")
    assert first["target_used"] is False
    with np.load(tmp_path / "trace/prediction.npz", allow_pickle=False) as arrays:
        source = arrays["source_field"].copy()
        assert np.isfinite(arrays["predicted_field"]).all()
        assert "target_field" not in arrays
    with np.load(input_path, allow_pickle=False) as handle:
        arrays = {key: handle[key] for key in handle.files}
    arrays["source_field"] = source
    # An object-valued target would fail to load with allow_pickle=False if accessed.
    arrays["target_field"] = np.asarray([{"must_not_be_read": True}], dtype=object)
    np.savez(input_path, **arrays)
    second = infer(input_path, checkpoint, tmp_path / "supplied", steps=2, device="cpu")
    assert second["source_origin"] == "input_source_field"
    with np.load(tmp_path / "trace/prediction.npz") as a, np.load(tmp_path / "supplied/prediction.npz") as b:
        np.testing.assert_array_equal(a["predicted_field"], b["predicted_field"])


@pytest.mark.parametrize("method", ["euler", "rk4"])
def test_sampler_integrates_constant_velocity(method):
    class Velocity(torch.nn.Module):
        def forward(self, x, t, trace, source=None):
            return torch.ones_like(x)*0.5
    source = torch.zeros(2, 2, 16)
    result, trajectory = sample(Velocity(), source, torch.zeros(2, 1, 16, 16),
                                SamplingConfig(steps=4, method=method, return_trajectory=True))
    torch.testing.assert_close(result, torch.full_like(source, 0.5))
    assert trajectory.shape == (2, 5, 2, 16)


def test_splits_reject_leakage_and_missing_files(tmp_path):
    paths = [tmp_path / f"pulse_{i}.npz" for i in range(10)]
    payload = split_payload(paths, 0.1, 42)
    file = tmp_path / "split.json"
    file.write_text(json.dumps(payload))
    assert len(read_splits(file, paths)["val"]) == 1
    payload["train"].append(payload["val"][0])
    file.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="overlap"):
        read_splits(file, paths)


def test_report_preprocessing_removes_linear_background(tmp_path):
    # Construct a physical native wavelength-delay grid with enough resolution.
    wavelength = np.linspace(450, 1200, 700)
    delay = np.linspace(-400, 400, 801)
    signal = np.exp(-0.5*((wavelength[:,None]-800)/20)**2) * np.exp(-0.5*(delay[None,:]/35)**2)
    background = 2 + 0.0001*(wavelength[:,None]-800) + 0.0002*delay[None,:]
    raw = signal + background
    table = np.empty((wavelength.size+1, delay.size+1))
    table[0,0] = 0
    table[0,1:] = delay
    table[1:,0] = wavelength
    table[1:,1:] = raw
    native = tmp_path / "native.txt"
    np.savetxt(native, table, delimiter="\t")
    prepared = prepare_native(native)
    assert prepared["trace_linear"].shape == (256,256)
    assert prepared["trace_linear"].max() == pytest.approx(1)
    np.testing.assert_allclose(prepared["background"], background, atol=2e-8)
    assert json.loads(prepared["metadata_json"].item())["preprocessing"] == "report_far_delay_v1"
    assert main(["prepare", "--input", str(native), "--out", str(tmp_path / "prepared.npz")]) == 0
    assert load_measurement(tmp_path / "prepared.npz")["trace_linear"].shape == (256, 256)
    # Nonmonotone coordinates must not silently interpolate.
    table[10,0] = table[12,0] + 0.1
    np.savetxt(native, table, delimiter="\t")
    with pytest.raises(ValueError, match="monotone"):
        parse_linshan_trace(native)


def test_json_config_overrides_and_unknown_false_flag(tmp_path):
    from pnp_copra.configuration import with_config
    from pnp_copra.flow_matching.cli import parse_args
    config = tmp_path / 'train.json'
    config.write_text(json.dumps({'dataset': {'seed': 42}, 'training': {'epochs': 50, 'amp': True},
                                  'model': {'condition_on_source': False}, 'loss': {'time_eps': 1e-4}}))
    arguments = with_config(['--config', str(config), '--epochs', '1', '--no-amp'],
                            sections=('dataset', 'training', 'model', 'loss'))
    parsed = parse_args(['train', *arguments])
    assert parsed.epochs == 1 and parsed.amp is False and not parsed.condition_on_source
    config.write_text(json.dumps({'typo': False}))
    with pytest.raises(SystemExit):
        main(['inspect', '--config', str(config)])


def test_generation_refuses_mixed_settings(tmp_path):
    out = tmp_path / 'mixed'
    out.mkdir()
    (out / 'run_config.json').write_text(json.dumps({'maxiter': 400}))
    with pytest.raises(ValueError, match='different settings'):
        main(['generate', '--count', '1', '--maxiter', '4', '--out-dir', str(out)])
    assert json.loads((out / 'run_config.json').read_text())['maxiter'] == 400


def test_inference_rejects_checkpoint_with_unsupported_physical_grid(tmp_path):
    from dataclasses import asdict
    path = tmp_path / "different_grid.pt"
    torch.save({"model": {}, "configs": {"model": asdict(tiny_model()), "grid": {"dt_s": 5e-15}}}, path)
    with pytest.raises(ValueError, match="different spacing"):
        GridSpec.from_checkpoint_or_default(path)
