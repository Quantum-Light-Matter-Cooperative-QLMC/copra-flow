from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pnp_copra.flow_matching.complex import channels_to_complex, complex_to_channels
from pnp_copra.flow_matching.config import DatasetConfig, FlowLossConfig, ModelConfig
from pnp_copra.flow_matching.inspection import dataset_inventory
from pnp_copra.flow_matching.metrics import (
    align_field_to_target,
    aligned_pulse_mae,
    hann_denoise_time,
    infer_pulse_type,
)


def test_complex_channel_round_trip() -> None:
    values = np.asarray([1.0 + 2.0j, -3.0 + 0.5j], dtype=np.complex64)
    channels = complex_to_channels(values)
    assert channels.shape == (2, 2)
    restored = channels_to_complex(channels)
    np.testing.assert_allclose(restored, values)


def test_pulse_mae_alignment_and_uid_parsing() -> None:
    target = np.asarray([0.0, 1.0 + 1.0j, 2.0, 0.0], dtype=np.complex64)
    shifted_scaled = 3.0j * np.roll(target, 1)
    noisy = shifted_scaled + np.asarray([0.0, 0.0, 0.4, 0.0], dtype=np.complex64)
    assert aligned_pulse_mae(target, shifted_scaled) < 1e-6
    assert aligned_pulse_mae(target, noisy) > aligned_pulse_mae(target, shifted_scaled)
    assert infer_pulse_type("pulse_00042_random_spectral_abcdef12") == "random_spectral"


def test_field_alignment_returns_the_selected_aligned_field() -> None:
    target = np.asarray([0.0, 1.0 + 1.0j, 2.0, 0.0], dtype=np.complex64)
    estimate = 3.0j * np.roll(target, 1)
    mae, aligned = align_field_to_target(target, estimate)
    assert mae < 1e-6
    np.testing.assert_allclose(aligned, target, atol=1e-6)


def test_hann_denoise_time_preserves_complex_shape() -> None:
    values = np.asarray([0.0, 1.0 + 1.0j, 0.2 - 0.4j, 0.0], dtype=np.complex64)
    denoised = hann_denoise_time(values)
    assert denoised.shape == values.shape
    assert np.iscomplexobj(denoised)
    assert np.isfinite(denoised).all()


def write_test_shard(
    root: Path,
    *,
    n: int = 16,
    source_count: int = 3,
    include_hann_baselines: bool = False,
) -> Path:
    rng = np.random.default_rng(123)
    shard_dir = root / "shards"
    shard_dir.mkdir(parents=True)
    target = (rng.normal(size=n) + 1j * rng.normal(size=n)).astype(np.complex64)
    sources = np.stack(
        [
            target + 0.1 * (rng.normal(size=n) + 1j * rng.normal(size=n))
            for _ in range(source_count)
        ],
        axis=0,
    ).astype(np.complex64)
    trace = rng.random((n, n), dtype=np.float32)
    path = shard_dir / "pulse_test.npz"
    arrays = {
        "target_field": target,
        "source_fields": sources,
        "measured_trace": trace,
        "source_weight": np.asarray([1.0, 1.5, 2.0], dtype=np.float32),
        "hard_negative": np.asarray([False, True, False], dtype=np.bool_),
    }
    if include_hann_baselines:
        arrays["source_kind"] = np.asarray(
            ["copra_snapshot", "copra_denoised_final", "copra_denoised_guarded_final"],
            dtype="U64",
        )
        arrays["trace_ssim"] = np.asarray([0.9, 0.8, 0.85], dtype=np.float32)
        arrays["trace_error"] = np.asarray([0.1, 0.2, 0.15], dtype=np.float32)
        arrays["delay_axis"] = np.linspace(-1.0, 1.0, n, dtype=np.float64)
        arrays["freq_axis"] = np.linspace(-1.0, 1.0, n, dtype=np.float64)
        arrays["time_axis"] = np.linspace(-1.0, 1.0, n, dtype=np.float64)
        arrays["time_window"] = np.ones(n, dtype=np.float32)
    np.savez_compressed(path, **arrays)
    return path


def write_prediction_artifact(
    root: Path,
    *,
    uid: str = "pulse_00042_gaussian_abcdef12",
    n: int = 32,
) -> tuple[Path, str]:
    shard_dir = root / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    time = np.linspace(-40e-15, 40e-15, n, dtype=np.float64)
    target = np.exp(-0.5 * (time / 12e-15) ** 2).astype(np.complex64)
    phase_error = np.exp(0.15j * np.sin(np.linspace(-np.pi, np.pi, n)))
    source = (target * phase_error).astype(np.complex64)
    hann = hann_denoise_time(source)
    predicted = (0.9 * target + 0.1 * source).astype(np.complex64)
    trace = np.outer(np.abs(target), np.abs(target)).astype(np.float32)
    path = shard_dir / f"{uid}_src17_copra_final.npz"
    np.savez_compressed(
        path,
        uid=np.asarray(uid),
        target_field=target,
        source_field=source,
        copra_hann_direct_final_field=hann,
        predicted_field=predicted,
        source_trace_ssim=np.asarray(0.97, dtype=np.float32),
        copra_hann_direct_final_trace_ssim=np.asarray(0.91, dtype=np.float32),
        time_axis=time,
        delay_axis=time,
        freq_axis=np.linspace(-1e15, 1e15, n, dtype=np.float64),
        measured_trace=trace,
    )
    return path, uid


def test_dataset_inventory_reads_npz_without_torch(tmp_path: Path) -> None:
    write_test_shard(tmp_path)
    inventory = dataset_inventory(tmp_path)
    assert inventory["shards"] == 1
    assert inventory["source_examples"] == 3
    assert inventory["hard_negatives"] == 1


def test_flow_matching_dataset_shapes(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from pnp_copra.flow_matching.data import FlowMatchingDataset

    write_test_shard(tmp_path)
    dataset = FlowMatchingDataset(DatasetConfig(data_dir=tmp_path, split="all"))
    sample = dataset[0]
    assert len(dataset) == 3
    assert tuple(sample["source"].shape) == (2, 16)
    assert tuple(sample["target"].shape) == (2, 16)
    assert tuple(sample["trace"].shape) == (1, 16, 16)
    assert "time_window" not in sample
    assert "copra_hann_direct_final" not in sample


def test_flow_matching_dataset_exposes_hann_baselines(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from pnp_copra.flow_matching.data import FlowMatchingDataset

    write_test_shard(tmp_path, include_hann_baselines=True)
    dataset = FlowMatchingDataset(
        DatasetConfig(data_dir=tmp_path, split="all", include_evaluation_fields=True)
    )
    sample = dataset[0]
    assert sample["copra_hann_direct_final_available"]
    assert sample["copra_hann_guarded_final_available"]
    assert sample["source_kind"] == "copra_snapshot"
    assert tuple(sample["time_window"].shape) == (16,)
    assert tuple(sample["measured_trace_raw"].shape) == (16, 16)
    assert float(sample["source_trace_ssim"]) == pytest.approx(0.9)


def test_flow_matching_dataset_filters_source_kind(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from pnp_copra.flow_matching.data import FlowMatchingDataset

    write_test_shard(tmp_path, include_hann_baselines=True)
    dataset = FlowMatchingDataset(
        DatasetConfig(
            data_dir=tmp_path,
            split="all",
            source_kind_filter="copra_denoised_final",
            include_evaluation_fields=True,
        )
    )
    assert len(dataset) == 1
    assert dataset[0]["source_kind"] == "copra_denoised_final"


def test_prediction_visualization_writes_four_field_comparison(tmp_path: Path) -> None:
    from pnp_copra.flow_matching.visualization import (
        PulseVisualizationConfig,
        visualize_prediction_artifacts,
    )

    predictions_dir = tmp_path / "predictions"
    _, uid = write_prediction_artifact(predictions_dir)
    out_dir = tmp_path / "figures"
    result = visualize_prediction_artifacts(
        PulseVisualizationConfig(
            predictions_dir=predictions_dir,
            out_dir=out_dir,
            pulse_uids=(uid,),
            include_trace_diagnostics=False,
            dpi=72,
        )
    )
    assert result["artifact_count"] == 1
    assert (out_dir / f"{uid}.png").is_file()
    assert (out_dir / f"{uid}.pdf").is_file()
    assert (out_dir / "manifest.json").is_file()
    assert result["records"][0]["pulse_mae"]["flow"] < result["records"][0]["pulse_mae"]["copra"]


def test_prediction_visualization_uses_multiple_workers(tmp_path: Path) -> None:
    from pnp_copra.flow_matching.visualization import (
        PulseVisualizationConfig,
        visualize_prediction_artifacts,
    )

    predictions_dir = tmp_path / "predictions"
    _, first_uid = write_prediction_artifact(predictions_dir)
    _, second_uid = write_prediction_artifact(
        predictions_dir,
        uid="pulse_00043_sech_12345678",
    )
    out_dir = tmp_path / "figures"
    result = visualize_prediction_artifacts(
        PulseVisualizationConfig(
            predictions_dir=predictions_dir,
            out_dir=out_dir,
            pulse_uids=(first_uid, second_uid),
            include_trace_diagnostics=True,
            dpi=48,
            num_workers=2,
        )
    )
    assert result["artifact_count"] == 2
    assert result["num_workers"] == 2
    assert [record["pulse_uid"] for record in result["records"]] == [first_uid, second_uid]


def test_model_loss_backward_smoke(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from pnp_copra.flow_matching.data import FlowMatchingDataset
    from pnp_copra.flow_matching.losses import flow_matching_loss
    from pnp_copra.flow_matching.model import FlowMatchingModel

    write_test_shard(tmp_path)
    dataset = FlowMatchingDataset(DatasetConfig(data_dir=tmp_path, split="all"))
    samples = [dataset[index] for index in range(2)]
    batch = {
        "source": torch.stack([sample["source"] for sample in samples]),
        "target": torch.stack([sample["target"] for sample in samples]),
        "trace": torch.stack([sample["trace"] for sample in samples]),
        "weight": torch.stack([sample["weight"] for sample in samples]),
    }
    model = FlowMatchingModel(
        ModelConfig(
            field_length=16,
            hidden_channels=16,
            residual_blocks=2,
            condition_dim=32,
            time_embedding_dim=16,
            trace_base_channels=4,
            trace_layers=2,
        )
    )
    metrics = flow_matching_loss(model, batch, FlowLossConfig())
    assert torch.isfinite(metrics["loss"])
    metrics["loss"].backward()
    assert any(param.grad is not None for param in model.parameters())
