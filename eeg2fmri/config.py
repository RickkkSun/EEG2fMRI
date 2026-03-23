from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    root: str = "./data"
    eeg_dirname: str = "EEG_set"
    fmri_dirname: str = "fMRI_difumo64"
    eeg_glob: str = "*.set"
    fmri_suffix: str = "_difumo64_roi.pkl"
    tr_seconds: float = 2.1
    eeg_fs: float = 200.0
    context_seconds: float = 16.0
    chunk_length: int = 8
    chunk_stride: int = 1
    n_rois: int = 64
    group_split_seed: int = 7
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    eeg_normalize: str = "robust"
    scan_cache_size: int = 4
    max_scans: int | None = None


@dataclass
class ModelConfig:
    patch_size: int = 128
    patch_stride: int = 64
    d_model: int = 256
    eeg_conv_kernel: int = 9
    eeg_conv_layers: int = 3
    eeg_dropout: float = 0.1
    spectral_bands: tuple[tuple[float, float], ...] = (
        (0.5, 4.0),
        (4.0, 8.0),
        (8.0, 13.0),
        (13.0, 30.0),
        (30.0, 45.0),
    )
    condition_heads: int = 4
    condition_ff_mult: int = 4
    hrf_peak_seconds: float = 5.0
    hrf_sigma_seconds: float = 3.0
    hrf_max_seconds: float = 16.0
    flow_layers: int = 4
    flow_heads: int = 4
    flow_ff_mult: int = 4
    flow_dropout: float = 0.1
    noise_sigma: float = 1.0


@dataclass
class LossConfig:
    mean_weight: float = 1.0
    temporal_weight: float = 0.1
    cfm_weight: float = 1.0
    huber_delta: float = 1.0


@dataclass
class OptimConfig:
    lr: float = 3e-4
    weight_decay: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.95)
    batch_size: int = 8
    num_workers: int = 0
    max_epochs: int = 80
    mean_only_epochs: int = 10
    grad_clip_norm: float = 1.0
    amp: bool = False
    log_every: int = 20
    val_every: int = 1
    seed: int = 42


@dataclass
class EvalConfig:
    num_samples: int = 8
    ode_steps: int = 16
    ode_solver: str = "heun"
    save_predictions: bool = False


@dataclass
class RuntimeConfig:
    output_dir: str = "./outputs/neuroflowmatch"
    device: str = "auto"


@dataclass
class TrainConfig:
    experiment_name: str = "neuroflowmatch_base"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


def _update_dataclass(instance: Any, values: dict[str, Any]) -> Any:
    for key, value in values.items():
        current = getattr(instance, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _update_dataclass(current, value)
        else:
            setattr(instance, key, value)
    return instance


def load_config(path: str | Path) -> TrainConfig:
    config = TrainConfig()
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    _update_dataclass(config, data)
    return config
