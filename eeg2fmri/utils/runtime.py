from __future__ import annotations

import argparse

import torch

from eeg2fmri.config import TrainConfig


def str_to_bool(value: str | bool | None) -> bool | None:
    if value is None or isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Could not parse boolean value from {value!r}")


def str_to_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    parts = [item.strip() for item in value.split(",")]
    return [item for item in parts if item]


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def add_common_override_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data.root", dest="data_root", type=str, default=None, help="Override dataset root.")
    parser.add_argument("--data.eeg_dirname", dest="eeg_dirname", type=str, default=None, help="Override EEG directory under data root.")
    parser.add_argument("--data.fmri_dirname", dest="fmri_dirname", type=str, default=None, help="Override fMRI directory under data root.")
    parser.add_argument("--data.eeg_glob", dest="eeg_glob", type=str, default=None, help="Override EEG filename glob.")
    parser.add_argument("--data.eeg_fs", dest="eeg_fs", type=float, default=None, help="Override EEG sampling rate.")
    parser.add_argument("--data.resample_eeg_to_target_fs", dest="resample_eeg_to_target_fs", type=str_to_bool, default=None, help="Resample EEG to the configured target sampling rate.")
    parser.add_argument("--data.split_strategy", dest="split_strategy", type=str, default=None, help="Override split strategy.")
    parser.add_argument("--data.split_manifest", dest="split_manifest", type=str, default=None, help="Path to an explicit train/val/test scan assignment file (csv/tsv/xlsx).")
    parser.add_argument("--data.cv_folds", dest="cv_folds", type=int, default=None, help="Override number of subject-wise CV folds.")
    parser.add_argument("--data.fold_index", dest="fold_index", type=int, default=None, help="Override fold index.")
    parser.add_argument("--data.loso_subject", dest="loso_subject", type=str, default=None, help="Override LOSO held-out subject.")
    parser.add_argument("--data.temporal_gap_trs", dest="temporal_gap_trs", type=int, default=None, help="Override temporal gap for intra-subject splits.")
    parser.add_argument("--data.max_scans", dest="max_scans", type=int, default=None, help="Limit the number of discovered scans.")
    parser.add_argument("--data.target_columns", dest="target_columns", type=str_to_list, default=None, help="Comma-separated ROI/global-signal target names.")
    parser.add_argument("--data.include_global_signal_clean", dest="include_global_signal_clean", type=str_to_bool, default=None, help="Include global signal clean target.")
    parser.add_argument("--data.include_global_signal_raw", dest="include_global_signal_raw", type=str_to_bool, default=None, help="Include global signal raw target.")
    parser.add_argument("--optim.batch_size", dest="batch_size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--optim.max_epochs", dest="max_epochs", type=int, default=None, help="Override max epochs.")
    parser.add_argument("--optim.mean_only_epochs", dest="mean_only_epochs", type=int, default=None, help="Override deterministic warmup epochs.")
    parser.add_argument("--optim.num_workers", dest="num_workers", type=int, default=None, help="Override dataloader workers.")
    parser.add_argument("--eval.num_samples", dest="num_samples", type=int, default=None, help="Override number of flow samples at evaluation.")
    parser.add_argument("--eval.ode_steps", dest="ode_steps", type=int, default=None, help="Override ODE solver steps at evaluation.")
    parser.add_argument("--eval.save_predictions", dest="save_predictions", type=str_to_bool, default=None, help="Override whether evaluation stores scan predictions.")
    parser.add_argument("--runtime.output_dir", dest="output_dir", type=str, default=None, help="Override output dir.")
    parser.add_argument("--runtime.device", dest="device", type=str, default=None, help="Override device.")


def apply_common_overrides(config: TrainConfig, args: argparse.Namespace) -> TrainConfig:
    if args.data_root is not None:
        config.data.root = args.data_root
    if args.eeg_dirname is not None:
        config.data.eeg_dirname = args.eeg_dirname
    if args.fmri_dirname is not None:
        config.data.fmri_dirname = args.fmri_dirname
    if args.eeg_glob is not None:
        config.data.eeg_glob = args.eeg_glob
    if args.eeg_fs is not None:
        config.data.eeg_fs = args.eeg_fs
    if args.resample_eeg_to_target_fs is not None:
        config.data.resample_eeg_to_target_fs = bool(args.resample_eeg_to_target_fs)
    if args.split_strategy is not None:
        config.data.split_strategy = args.split_strategy
    if args.split_manifest is not None:
        config.data.split_manifest = args.split_manifest
    if args.cv_folds is not None:
        config.data.cv_folds = args.cv_folds
    if args.fold_index is not None:
        config.data.fold_index = args.fold_index
    if args.loso_subject is not None:
        config.data.loso_subject = args.loso_subject
    if args.temporal_gap_trs is not None:
        config.data.temporal_gap_trs = args.temporal_gap_trs
    if args.max_scans is not None:
        config.data.max_scans = args.max_scans
    if args.target_columns is not None:
        config.data.target_columns = list(args.target_columns)
    if args.include_global_signal_clean is not None:
        config.data.include_global_signal_clean = bool(args.include_global_signal_clean)
    if args.include_global_signal_raw is not None:
        config.data.include_global_signal_raw = bool(args.include_global_signal_raw)
    if args.batch_size is not None:
        config.optim.batch_size = args.batch_size
    if args.max_epochs is not None:
        config.optim.max_epochs = args.max_epochs
    if args.mean_only_epochs is not None:
        config.optim.mean_only_epochs = args.mean_only_epochs
    if args.num_workers is not None:
        config.optim.num_workers = args.num_workers
    if args.num_samples is not None:
        config.eval.num_samples = args.num_samples
    if args.ode_steps is not None:
        config.eval.ode_steps = args.ode_steps
    if args.save_predictions is not None:
        config.eval.save_predictions = bool(args.save_predictions)
    if args.output_dir is not None:
        config.runtime.output_dir = args.output_dir
    if args.device is not None:
        config.runtime.device = args.device
    return config
