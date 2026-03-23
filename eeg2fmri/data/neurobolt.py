from __future__ import annotations

import math
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset

from eeg2fmri.config import DataConfig, OptimConfig
from eeg2fmri.data.io import load_eeg, load_roi_timeseries


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


@dataclass(frozen=True)
class ScanRecord:
    subject_id: str
    scan_id: str
    eeg_path: str
    fmri_path: str


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

    eeg_by_scan = {}
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


def split_records(records: list[ScanRecord], config: DataConfig) -> dict[str, list[ScanRecord]]:
    groups = np.array([record.subject_id for record in records])
    indices = np.arange(len(records))
    splitter = GroupShuffleSplit(
        n_splits=1,
        train_size=config.train_ratio,
        random_state=config.group_split_seed,
    )
    train_index, remain_index = next(splitter.split(indices, groups=groups))
    remain_groups = groups[remain_index]
    remain_size = config.val_ratio + config.test_ratio
    if remain_size <= 0:
        raise ValueError("val_ratio + test_ratio must be positive.")
    val_fraction = config.val_ratio / remain_size
    val_splitter = GroupShuffleSplit(
        n_splits=1,
        train_size=val_fraction,
        random_state=config.group_split_seed + 1,
    )
    val_subindex, test_subindex = next(
        val_splitter.split(remain_index, groups=remain_groups)
    )
    val_index = remain_index[val_subindex]
    test_index = remain_index[test_subindex]
    return {
        "train": [records[i] for i in train_index.tolist()],
        "val": [records[i] for i in val_index.tolist()],
        "test": [records[i] for i in test_index.tolist()],
    }


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
    ) -> None:
        self.records = records
        self.config = data_config
        self.split_name = split_name
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
            roi = load_roi_timeseries(record.fmri_path, n_rois=self.config.n_rois)
            length = roi.shape[0]
            self.scan_lengths[record.scan_id] = length
            max_start = length - self.chunk_length + 1
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
        roi = load_roi_timeseries(record.fmri_path, n_rois=self.config.n_rois)
        if eeg.ndim != 2:
            raise ValueError(f"Expected 2D EEG array [channels, samples], got {eeg.shape}")
        if abs(fs - self.config.eeg_fs) > 1.0:
            raise ValueError(
                f"EEG sampling rate mismatch for {record.scan_id}: "
                f"found {fs}, expected about {self.config.eeg_fs}"
            )
        if channel_names:
            channel_index = {name: idx for idx, name in enumerate(channel_names)}
            if all(name in channel_index for name in self.selected_channels):
                eeg = np.stack([eeg[channel_index[name]] for name in self.selected_channels], axis=0)
                channel_names = list(self.selected_channels)
            else:
                extras = {"ECG", "EOG1", "EOG2", "EMG1", "EMG2", "EMG3", "CWL1", "CWL2", "CWL3", "CWL4"}
                keep = [i for i, name in enumerate(channel_names) if name not in extras]
                eeg = eeg[keep]
                channel_names = [channel_names[i] for i in keep]
        eeg = self._normalize_eeg(eeg.astype(np.float32))
        scan = {
            "eeg": eeg,
            "roi": roi.astype(np.float32),
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
            "start_tr": start_tr,
            "length": self.scan_lengths[record.scan_id],
        }


def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "eeg": torch.stack([item["eeg"] for item in batch], dim=0),
        "target": torch.stack([item["target"] for item in batch], dim=0),
        "scan_id": [item["scan_id"] for item in batch],
        "subject_id": [item["subject_id"] for item in batch],
        "start_tr": torch.tensor([item["start_tr"] for item in batch], dtype=torch.long),
        "length": torch.tensor([item["length"] for item in batch], dtype=torch.long),
    }


class NeuroBoltDataModule:
    def __init__(self, data_config: DataConfig, optim_config: OptimConfig) -> None:
        self.data_config = data_config
        self.optim_config = optim_config
        self.records = discover_scans(data_config)
        self.splits = split_records(self.records, data_config)
        self.datasets: dict[str, NeuroBoltWindowDataset] = {}

    def setup(self) -> None:
        for split_name, records in self.splits.items():
            self.datasets[split_name] = NeuroBoltWindowDataset(
                records=records,
                data_config=self.data_config,
                split_name=split_name,
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

    def describe(self) -> dict[str, Any]:
        info = {split: len(records) for split, records in self.splits.items()}
        info["subjects"] = {
            split: len({record.subject_id for record in records})
            for split, records in self.splits.items()
        }
        return info
