from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from scipy.signal import resample_poly
from torch.utils.data import DataLoader, Dataset

from eeg2fmri.config import DataConfig, OptimConfig
from eeg2fmri.data.io import load_eeg, load_roi_timeseries


GLOBAL_SIGNAL_CLEAN = "global signal clean"
GLOBAL_SIGNAL_RAW = "global signal raw"
GLOBAL_SIGNAL_NAMES = {GLOBAL_SIGNAL_CLEAN, GLOBAL_SIGNAL_RAW}

DEFAULT_SCALP_CHANNELS = [
    "Fp1",
    "Fp2",
    "F3",
    "F4",
    "C3",
    "C4",
    "P3",
    "P4",
    "O1",
    "O2",
    "F7",
    "F8",
    "T7",
    "T8",
    "P7",
    "P8",
    "FPz",
    "Fz",
    "Cz",
    "Pz",
    "POz",
    "Oz",
    "FT9",
    "FT10",
    "TP9'",
    "TP10'",
]

EXCLUDED_AUX_CHANNELS = {"ECG", "EOG1", "EOG2", "EMG1", "EMG2", "EMG3", "CWL1", "CWL2", "CWL3", "CWL4"}


@dataclass(frozen=True)
class ScanRecord:
    subject_id: str
    scan_id: str
    eeg_path: str
    fmri_path: str


@dataclass(frozen=True)
class ExperimentSplit:
    name: str
    train_records: list[ScanRecord]
    val_records: list[ScanRecord]
    test_records: list[ScanRecord]
    train_ranges: dict[str, list[tuple[int, int]]] | None = None
    val_ranges: dict[str, list[tuple[int, int]]] | None = None
    test_ranges: dict[str, list[tuple[int, int]]] | None = None


def _parse_scan_id(stem: str) -> tuple[str, str]:
    base = stem
    for suffix in ("_difumo64_roi", "_eeg", "_EEG", "_roi"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    parts = base.split("-")
    if len(parts) < 2:
        raise ValueError(f"Unexpected scan stem: {stem}")
    subject_id = parts[0]
    return subject_id, base


def discover_scans(config: DataConfig) -> list[ScanRecord]:
    root = Path(config.root)
    eeg_dir = root / config.eeg_dirname
    fmri_dir = root / config.fmri_dirname
    eeg_files = sorted(eeg_dir.glob(config.eeg_glob))
    fmri_files = sorted(fmri_dir.glob(f"*{config.fmri_suffix}"))

    eeg_by_scan: dict[str, ScanRecord] = {}
    for path in eeg_files:
        subject_id, scan_id = _parse_scan_id(path.stem)
        eeg_by_scan[scan_id] = ScanRecord(
            subject_id=subject_id,
            scan_id=scan_id,
            eeg_path=str(path),
            fmri_path="",
        )

    records: list[ScanRecord] = []
    for path in fmri_files:
        subject_id, scan_id = _parse_scan_id(path.stem)
        if scan_id not in eeg_by_scan:
            continue
        base = eeg_by_scan[scan_id]
        records.append(
            ScanRecord(
                subject_id=subject_id,
                scan_id=scan_id,
                eeg_path=base.eeg_path,
                fmri_path=str(path),
            )
        )

    if config.max_scans is not None:
        records = records[: config.max_scans]
    if not records:
        raise FileNotFoundError(
            f"No paired scans found under {eeg_dir} and {fmri_dir}. "
            "Expected NeuroBOLT-style filenames such as sub01-scan01.set and "
            "sub01-scan01_difumo64_roi.pkl."
        )
    return records


def _select_targets(
    roi: np.ndarray,
    roi_names: list[str] | None,
    config: DataConfig,
) -> tuple[np.ndarray, list[str] | None]:
    if roi_names:
        normalized = [str(name).strip() for name in roi_names]
        if config.target_columns:
            index_by_name = {name.lower(): idx for idx, name in enumerate(normalized)}
            selected = []
            for requested in config.target_columns:
                key = requested.strip().lower()
                if key not in index_by_name:
                    raise ValueError(
                        f"Requested target column {requested!r} was not found in ROI columns. "
                        f"Available examples: {normalized[:10]}"
                    )
                selected.append(index_by_name[key])
            selected_names = [normalized[idx] for idx in selected]
            return roi[:, selected].astype(np.float32), selected_names
        roi_indices = [idx for idx, name in enumerate(normalized) if name.lower() not in GLOBAL_SIGNAL_NAMES]
        roi_indices = roi_indices[: config.n_rois]
        selected = list(roi_indices)
        if config.include_global_signal_clean:
            found_clean = False
            for idx, name in enumerate(normalized):
                if name.lower() == GLOBAL_SIGNAL_CLEAN:
                    selected.append(idx)
                    found_clean = True
                    break
            if not found_clean:
                raise ValueError("Requested `global signal clean` but it was not found in ROI columns.")
        if config.include_global_signal_raw:
            found_raw = False
            for idx, name in enumerate(normalized):
                if name.lower() == GLOBAL_SIGNAL_RAW:
                    selected.append(idx)
                    found_raw = True
                    break
            if not found_raw:
                raise ValueError("Requested `global signal raw` but it was not found in ROI columns.")
        selected_names = [normalized[idx] for idx in selected]
        return roi[:, selected].astype(np.float32), selected_names

    extra = int(config.include_global_signal_clean) + int(config.include_global_signal_raw)
    if config.target_columns:
        raise ValueError("Named target selection requires ROI column names, but none were provided by the file.")
    target_dim = config.n_rois + extra
    if roi.shape[1] < target_dim:
        raise ValueError(
            f"Expected at least {target_dim} target columns, but found {roi.shape[1]}. "
            "Disable the extra global-signal targets or provide ROI names so they can be selected explicitly."
        )
    return roi[:, :target_dim].astype(np.float32), None


def _load_split_manifest(records: list[ScanRecord], manifest_path: str | Path) -> ExperimentSplit:
    path = Path(manifest_path)
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        frame = pd.read_csv(path)
    elif suffix == ".tsv":
        frame = pd.read_csv(path, sep="\t")
    elif suffix in {".xlsx", ".xls"}:
        frame = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported split manifest format: {path}")

    columns = {str(col).strip().lower(): str(col) for col in frame.columns}
    scan_col = next((columns[key] for key in columns if "scan" in key), None)
    split_col = next((columns[key] for key in columns if key in {"split", "set", "group"} or "split" in key), None)
    if scan_col is None or split_col is None:
        raise ValueError(
            f"Split manifest must contain scan and split columns. Found columns: {list(frame.columns)}"
        )

    by_scan = {record.scan_id: record for record in records}
    partitions = {"train": [], "val": [], "test": []}
    for _, row in frame.iterrows():
        scan_id = str(row[scan_col]).strip()
        split_name = str(row[split_col]).strip().lower()
        if scan_id not in by_scan:
            continue
        if split_name in {"train", "training"}:
            partitions["train"].append(by_scan[scan_id])
        elif split_name in {"val", "valid", "validation", "dev"}:
            partitions["val"].append(by_scan[scan_id])
        elif split_name in {"test", "testing"}:
            partitions["test"].append(by_scan[scan_id])

    if not partitions["train"] or not partitions["test"]:
        counts = {key: len(value) for key, value in partitions.items()}
        raise ValueError(
            f"Split manifest {path} did not produce non-empty train/test sets. "
            f"Resolved counts: {counts}"
        )
    return ExperimentSplit(
        name=path.stem,
        train_records=partitions["train"],
        val_records=partitions["val"],
        test_records=partitions["test"],
    )


def _get_target_length(record: ScanRecord, config: DataConfig) -> int:
    roi, roi_names = load_roi_timeseries(record.fmri_path, n_rois=config.n_rois + 2, return_names=True)
    selected, _ = _select_targets(roi, roi_names, config)
    return int(selected.shape[0])


def _split_train_val_from_records(
    records: list[ScanRecord],
    config: DataConfig,
    seed_offset: int = 0,
) -> tuple[list[ScanRecord], list[ScanRecord]]:
    if not records:
        return [], []
    if config.val_ratio <= 0:
        return records, []
    unique_groups = sorted({record.subject_id for record in records})
    if len(unique_groups) < 2:
        return records, []
    groups = np.array([record.subject_id for record in records])
    indices = np.arange(len(records))
    val_fraction = config.val_ratio / max(config.train_ratio + config.val_ratio, 1e-8)
    splitter = GroupShuffleSplit(
        n_splits=1,
        train_size=max(1e-8, 1.0 - val_fraction),
        random_state=config.group_split_seed + seed_offset,
    )
    train_idx, val_idx = next(splitter.split(indices, groups=groups))
    return [records[i] for i in train_idx.tolist()], [records[i] for i in val_idx.tolist()]


def _build_holdout_split(records: list[ScanRecord], config: DataConfig) -> list[ExperimentSplit]:
    groups = np.array([record.subject_id for record in records])
    indices = np.arange(len(records))
    if len(np.unique(groups)) < 2:
        raise ValueError("inter_subject_holdout requires at least two unique subjects.")
    remain_fraction = config.train_ratio + config.val_ratio
    splitter = GroupShuffleSplit(
        n_splits=1,
        train_size=max(1e-8, remain_fraction),
        random_state=config.group_split_seed,
    )
    train_val_idx, test_idx = next(splitter.split(indices, groups=groups))
    train_val_records = [records[i] for i in train_val_idx.tolist()]
    test_records = [records[i] for i in test_idx.tolist()]
    train_records, val_records = _split_train_val_from_records(train_val_records, config, seed_offset=1)
    return [
        ExperimentSplit(
            name="holdout",
            train_records=train_records,
            val_records=val_records,
            test_records=test_records,
        )
    ]


def _build_kfold_splits(records: list[ScanRecord], config: DataConfig) -> list[ExperimentSplit]:
    groups = np.array([record.subject_id for record in records])
    indices = np.arange(len(records))
    n_subjects = len(np.unique(groups))
    if n_subjects < 2:
        raise ValueError("inter_subject_kfold requires at least two unique subjects.")
    splitter = GroupKFold(n_splits=min(config.cv_folds, n_subjects))
    splits: list[ExperimentSplit] = []
    for fold_id, (train_val_idx, test_idx) in enumerate(splitter.split(indices, groups=groups)):
        train_val_records = [records[i] for i in train_val_idx.tolist()]
        test_records = [records[i] for i in test_idx.tolist()]
        train_records, val_records = _split_train_val_from_records(train_val_records, config, seed_offset=100 + fold_id)
        splits.append(
            ExperimentSplit(
                name=f"fold_{fold_id:02d}",
                train_records=train_records,
                val_records=val_records,
                test_records=test_records,
            )
        )
    if config.fold_index is not None:
        return [splits[int(config.fold_index)]]
    return splits


def _build_loso_splits(records: list[ScanRecord], config: DataConfig) -> list[ExperimentSplit]:
    subject_ids = sorted({record.subject_id for record in records})
    if config.loso_subject is not None:
        subject_ids = [subject_id for subject_id in subject_ids if subject_id == config.loso_subject]
        if not subject_ids:
            raise ValueError(f"Requested loso_subject={config.loso_subject!r}, but it was not found in the dataset.")
    splits: list[ExperimentSplit] = []
    for fold_id, subject_id in enumerate(subject_ids):
        test_records = [record for record in records if record.subject_id == subject_id]
        train_val_records = [record for record in records if record.subject_id != subject_id]
        train_records, val_records = _split_train_val_from_records(train_val_records, config, seed_offset=200 + fold_id)
        splits.append(
            ExperimentSplit(
                name=f"loso_{subject_id}",
                train_records=train_records,
                val_records=val_records,
                test_records=test_records,
            )
        )
    if config.fold_index is not None and splits:
        return [splits[int(config.fold_index)]]
    return splits


def _make_temporal_ranges(total_starts: int, config: DataConfig) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]:
    gap = max(0, config.temporal_gap_trs)
    train_end = int(total_starts * config.train_ratio)
    val_end = int(total_starts * (config.train_ratio + config.val_ratio))

    train_range = (0, max(0, train_end - gap))
    val_range = (min(total_starts, train_end + gap), max(min(total_starts, train_end + gap), val_end - gap))
    test_range = (min(total_starts, val_end + gap), total_starts)

    train_ranges = [train_range] if train_range[1] > train_range[0] else []
    val_ranges = [val_range] if val_range[1] > val_range[0] else []
    test_ranges = [test_range] if test_range[1] > test_range[0] else []
    return train_ranges, val_ranges, test_ranges


def _build_intra_subject_splits(records: list[ScanRecord], config: DataConfig) -> list[ExperimentSplit]:
    splits: list[ExperimentSplit] = []
    for record in records:
        total_length = _get_target_length(record, config)
        total_starts = max(0, total_length - config.chunk_length + 1)
        train_ranges, val_ranges, test_ranges = _make_temporal_ranges(total_starts, config)
        if not train_ranges or not test_ranges:
            continue
        splits.append(
            ExperimentSplit(
                name=f"intra_{record.scan_id}",
                train_records=[record],
                val_records=[record] if val_ranges else [],
                test_records=[record],
                train_ranges={record.scan_id: train_ranges},
                val_ranges={record.scan_id: val_ranges} if val_ranges else {},
                test_ranges={record.scan_id: test_ranges},
            )
        )
    if config.fold_index is not None and splits:
        return [splits[int(config.fold_index)]]
    return splits


def build_experiment_splits(records: list[ScanRecord], config: DataConfig) -> list[ExperimentSplit]:
    if config.split_manifest:
        return [_load_split_manifest(records, config.split_manifest)]
    strategy = config.split_strategy
    if strategy == "inter_subject_holdout":
        return _build_holdout_split(records, config)
    if strategy == "inter_subject_kfold":
        return _build_kfold_splits(records, config)
    if strategy == "loso":
        return _build_loso_splits(records, config)
    if strategy == "intra_subject_temporal":
        return _build_intra_subject_splits(records, config)
    raise ValueError(f"Unknown split_strategy: {strategy}")


class ScanCache:
    def __init__(self, max_size: int) -> None:
        self.max_size = max_size
        self._cache: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def get(self, key: str) -> dict[str, Any] | None:
        if key not in self._cache:
            return None
        value = self._cache.pop(key)
        self._cache[key] = value
        return value

    def put(self, key: str, value: dict[str, Any]) -> None:
        if key in self._cache:
            self._cache.pop(key)
        self._cache[key] = value
        while len(self._cache) > self.max_size:
            self._cache.popitem(last=False)


class NeuroBoltWindowDataset(Dataset):
    def __init__(
        self,
        records: list[ScanRecord],
        data_config: DataConfig,
        split_name: str,
        allowed_ranges: dict[str, list[tuple[int, int]]] | None = None,
    ) -> None:
        self.records = records
        self.config = data_config
        self.split_name = split_name
        self.allowed_ranges = allowed_ranges or {}
        self.cache = ScanCache(max_size=data_config.scan_cache_size)
        self.chunk_length = data_config.chunk_length
        self.chunk_stride = data_config.chunk_stride
        self.selected_channels = DEFAULT_SCALP_CHANNELS
        self.samples_per_tr = int(round(data_config.tr_seconds * data_config.eeg_fs))
        self.context_samples = int(round(data_config.context_seconds * data_config.eeg_fs))
        self.input_samples = self.context_samples + (self.chunk_length - 1) * self.samples_per_tr
        self.index: list[tuple[int, int]] = []
        self.scan_lengths: dict[str, int] = {}
        self._build_index()

    def _build_index(self) -> None:
        for record_idx, record in enumerate(self.records):
            roi, roi_names = load_roi_timeseries(
                record.fmri_path,
                n_rois=self.config.n_rois + 2,
                return_names=True,
            )
            selected, _ = _select_targets(roi, roi_names, self.config)
            length = selected.shape[0]
            self.scan_lengths[record.scan_id] = length
            max_start = length - self.chunk_length + 1
            allowed = self.allowed_ranges.get(record.scan_id)
            if allowed:
                for start_range, end_range in allowed:
                    end_range = min(end_range, max_start)
                    for start in range(start_range, end_range, self.chunk_stride):
                        self.index.append((record_idx, start))
            else:
                for start in range(0, max_start, self.chunk_stride):
                    self.index.append((record_idx, start))

    def __len__(self) -> int:
        return len(self.index)

    def _normalize_eeg(self, eeg: np.ndarray) -> np.ndarray:
        if self.config.eeg_normalize == "none":
            return eeg
        if self.config.eeg_normalize == "zscore":
            mean = eeg.mean(axis=1, keepdims=True)
            std = eeg.std(axis=1, keepdims=True) + 1e-6
            return (eeg - mean) / std
        median = np.median(eeg, axis=1, keepdims=True)
        mad = np.median(np.abs(eeg - median), axis=1, keepdims=True) + 1e-6
        return (eeg - median) / mad

    def _load_scan(self, record: ScanRecord) -> dict[str, Any]:
        cached = self.cache.get(record.scan_id)
        if cached is not None:
            return cached

        eeg, fs, channel_names = load_eeg(record.eeg_path)
        roi, roi_names = load_roi_timeseries(
            record.fmri_path,
            n_rois=self.config.n_rois + 2,
            return_names=True,
        )
        roi, roi_names = _select_targets(roi, roi_names, self.config)

        if eeg.ndim != 2:
            raise ValueError(f"Expected 2D EEG array [channels, samples], got {eeg.shape}")
        if channel_names:
            channel_index = {name: idx for idx, name in enumerate(channel_names)}
            if all(name in channel_index for name in self.selected_channels):
                eeg = np.stack([eeg[channel_index[name]] for name in self.selected_channels], axis=0)
                channel_names = list(self.selected_channels)
            else:
                keep = [idx for idx, name in enumerate(channel_names) if name not in EXCLUDED_AUX_CHANNELS]
                eeg = eeg[keep]
                channel_names = [channel_names[idx] for idx in keep]
        if abs(fs - self.config.eeg_fs) > 1.0:
            if not self.config.resample_eeg_to_target_fs:
                raise ValueError(
                    f"EEG sampling rate mismatch for {record.scan_id}: "
                    f"found {fs}, expected about {self.config.eeg_fs}"
                )
            up = int(round(self.config.eeg_fs))
            down = int(round(fs))
            eeg = resample_poly(eeg, up=up, down=down, axis=1).astype(np.float32)
            fs = self.config.eeg_fs
        eeg = self._normalize_eeg(eeg.astype(np.float32))
        scan = {
            "eeg": eeg,
            "roi": roi.astype(np.float32),
            "roi_names": roi_names,
            "channel_names": channel_names,
        }
        self.cache.put(record.scan_id, scan)
        return scan

    def __getitem__(self, index: int) -> dict[str, Any]:
        record_idx, start_tr = self.index[index]
        record = self.records[record_idx]
        scan = self._load_scan(record)
        eeg = scan["eeg"]
        roi = scan["roi"]

        end_tr = start_tr + self.chunk_length - 1
        end_sample = (end_tr + 1) * self.samples_per_tr
        start_sample = end_sample - self.input_samples
        if start_sample < 0:
            pad = np.zeros((eeg.shape[0], -start_sample), dtype=np.float32)
            eeg_segment = np.concatenate([pad, eeg[:, :end_sample]], axis=1)
        else:
            eeg_segment = eeg[:, start_sample:end_sample]

        if eeg_segment.shape[1] < self.input_samples:
            pad = np.zeros((eeg.shape[0], self.input_samples - eeg_segment.shape[1]), dtype=np.float32)
            eeg_segment = np.concatenate([pad, eeg_segment], axis=1)

        target = roi[start_tr : start_tr + self.chunk_length]

        return {
            "eeg": torch.from_numpy(eeg_segment.astype(np.float32)),
            "target": torch.from_numpy(target.astype(np.float32)),
            "scan_id": record.scan_id,
            "subject_id": record.subject_id,
            "roi_names": scan.get("roi_names"),
            "start_tr": start_tr,
            "length": self.scan_lengths[record.scan_id],
        }


def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "eeg": torch.stack([item["eeg"] for item in batch], dim=0),
        "target": torch.stack([item["target"] for item in batch], dim=0),
        "scan_id": [item["scan_id"] for item in batch],
        "subject_id": [item["subject_id"] for item in batch],
        "roi_names": [item["roi_names"] for item in batch],
        "start_tr": torch.tensor([item["start_tr"] for item in batch], dtype=torch.long),
        "length": torch.tensor([item["length"] for item in batch], dtype=torch.long),
    }


class NeuroBoltDataModule:
    def __init__(
        self,
        data_config: DataConfig,
        optim_config: OptimConfig,
        split: ExperimentSplit | None = None,
        records: list[ScanRecord] | None = None,
    ) -> None:
        self.data_config = data_config
        self.optim_config = optim_config
        self.records = records or discover_scans(data_config)
        self.available_splits = build_experiment_splits(self.records, data_config)
        self.experiment_split = split or self.available_splits[0]
        self.datasets: dict[str, NeuroBoltWindowDataset] = {}

    def setup(self) -> None:
        self.datasets["train"] = NeuroBoltWindowDataset(
            records=self.experiment_split.train_records,
            data_config=self.data_config,
            split_name=f"{self.experiment_split.name}_train",
            allowed_ranges=self.experiment_split.train_ranges,
        )
        self.datasets["val"] = NeuroBoltWindowDataset(
            records=self.experiment_split.val_records,
            data_config=self.data_config,
            split_name=f"{self.experiment_split.name}_val",
            allowed_ranges=self.experiment_split.val_ranges,
        )
        self.datasets["test"] = NeuroBoltWindowDataset(
            records=self.experiment_split.test_records,
            data_config=self.data_config,
            split_name=f"{self.experiment_split.name}_test",
            allowed_ranges=self.experiment_split.test_ranges,
        )

    def dataloader(self, split_name: str, shuffle: bool) -> DataLoader:
        dataset = self.datasets[split_name]
        return DataLoader(
            dataset,
            batch_size=self.optim_config.batch_size,
            shuffle=shuffle,
            num_workers=self.optim_config.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=_collate,
        )

    def train_dataloader(self) -> DataLoader:
        return self.dataloader("train", shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self.dataloader("val", shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self.dataloader("test", shuffle=False)

    def target_names(self) -> list[str] | None:
        for split_name in ("train", "val", "test"):
            dataset = self.datasets.get(split_name)
            if dataset and len(dataset) > 0:
                first = dataset[0]
                return first["roi_names"]
        return None

    def target_dim(self) -> int:
        for split_name in ("train", "val", "test"):
            dataset = self.datasets.get(split_name)
            if dataset and len(dataset) > 0:
                first = dataset[0]
                return int(first["target"].shape[-1])
        raise RuntimeError("Could not infer target dimension from empty datasets.")

    def n_channels(self) -> int:
        for split_name in ("train", "val", "test"):
            dataset = self.datasets.get(split_name)
            if dataset and len(dataset) > 0:
                first = dataset[0]
                return int(first["eeg"].shape[0])
        raise RuntimeError("Could not infer EEG channel count from empty datasets.")

    def split_manifest(self) -> dict[str, Any]:
        manifest = {"name": self.experiment_split.name, "strategy": self.data_config.split_strategy}
        for split_name, records in (
            ("train", self.experiment_split.train_records),
            ("val", self.experiment_split.val_records),
            ("test", self.experiment_split.test_records),
        ):
            manifest[split_name] = [
                {
                    "subject_id": record.subject_id,
                    "scan_id": record.scan_id,
                    "eeg_path": record.eeg_path,
                    "fmri_path": record.fmri_path,
                }
                for record in records
            ]
        return manifest

    def describe(self) -> dict[str, Any]:
        info = {
            "experiment_split": self.experiment_split.name,
            "strategy": self.data_config.split_strategy,
            "train": len(self.experiment_split.train_records),
            "val": len(self.experiment_split.val_records),
            "test": len(self.experiment_split.test_records),
            "target_dim": None,
            "n_channels": None,
        }
        if self.datasets:
            try:
                info["target_dim"] = self.target_dim()
                info["n_channels"] = self.n_channels()
            except RuntimeError:
                pass
        info["subjects"] = {
            "train": len({record.subject_id for record in self.experiment_split.train_records}),
            "val": len({record.subject_id for record in self.experiment_split.val_records}),
            "test": len({record.subject_id for record in self.experiment_split.test_records}),
        }
        return info
