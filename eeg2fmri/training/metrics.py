from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ScanPrediction:
    target: np.ndarray
    prediction: np.ndarray
    samples: np.ndarray | None


class OverlapAccumulator:
    def __init__(self, num_samples: int | None = None) -> None:
        self.num_samples = num_samples
        self.storage: dict[str, dict[str, np.ndarray]] = {}

    def _ensure(self, scan_id: str, length: int, n_rois: int) -> None:
        if scan_id in self.storage:
            return
        state = {
            "pred_sum": np.zeros((length, n_rois), dtype=np.float32),
            "target_sum": np.zeros((length, n_rois), dtype=np.float32),
            "count": np.zeros((length, 1), dtype=np.float32),
        }
        if self.num_samples is not None and self.num_samples > 0:
            state["sample_sum"] = np.zeros((self.num_samples, length, n_rois), dtype=np.float32)
        self.storage[scan_id] = state

    def add(
        self,
        scan_id: str,
        start: int,
        length: int,
        prediction: np.ndarray,
        target: np.ndarray,
        samples: np.ndarray | None = None,
    ) -> None:
        n_steps, n_rois = prediction.shape
        self._ensure(scan_id, length=length, n_rois=n_rois)
        state = self.storage[scan_id]
        end = start + n_steps
        state["pred_sum"][start:end] += prediction
        state["target_sum"][start:end] += target
        state["count"][start:end] += 1.0
        if samples is not None and "sample_sum" in state:
            state["sample_sum"][:, start:end] += samples

    def finalize(self) -> dict[str, ScanPrediction]:
        result = {}
        for scan_id, state in self.storage.items():
            count = np.clip(state["count"], a_min=1.0, a_max=None)
            prediction = state["pred_sum"] / count
            target = state["target_sum"] / count
            samples = None
            if "sample_sum" in state:
                samples = state["sample_sum"] / count[None]
            result[scan_id] = ScanPrediction(
                target=target,
                prediction=prediction,
                samples=samples,
            )
        return result


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if denom < 1e-8:
        return 0.0
    return float(np.dot(x, y) / denom)


def pearson_r(prediction: np.ndarray, target: np.ndarray) -> float:
    values = [_safe_corr(prediction[:, i], target[:, i]) for i in range(target.shape[1])]
    return float(np.mean(values))


def rmse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((prediction - target) ** 2)))


def r2_score(prediction: np.ndarray, target: np.ndarray) -> float:
    ss_res = np.sum((target - prediction) ** 2)
    ss_tot = np.sum((target - target.mean(axis=0, keepdims=True)) ** 2)
    if ss_tot < 1e-8:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def psd_correlation(prediction: np.ndarray, target: np.ndarray) -> float:
    pred_fft = np.abs(np.fft.rfft(prediction - prediction.mean(axis=0, keepdims=True), axis=0)) ** 2
    true_fft = np.abs(np.fft.rfft(target - target.mean(axis=0, keepdims=True), axis=0)) ** 2
    values = [_safe_corr(np.log1p(pred_fft[:, i]), np.log1p(true_fft[:, i])) for i in range(target.shape[1])]
    return float(np.mean(values))


def fc_correlation(prediction: np.ndarray, target: np.ndarray) -> float:
    if prediction.shape[0] < 3:
        return 0.0
    pred_fc = np.nan_to_num(np.corrcoef(prediction, rowvar=False), nan=0.0)
    true_fc = np.nan_to_num(np.corrcoef(target, rowvar=False), nan=0.0)
    iu = np.triu_indices_from(pred_fc, k=1)
    return _safe_corr(pred_fc[iu], true_fc[iu])


def crps_ensemble(samples: np.ndarray, target: np.ndarray) -> float:
    first = np.mean(np.abs(samples - target[None]), axis=0)
    pair = np.mean(np.abs(samples[:, None] - samples[None, :]), axis=(0, 1))
    return float(np.mean(first - 0.5 * pair))


def interval_metrics(samples: np.ndarray, target: np.ndarray, alpha: float = 0.1) -> tuple[float, float]:
    lower = np.quantile(samples, alpha / 2.0, axis=0)
    upper = np.quantile(samples, 1.0 - alpha / 2.0, axis=0)
    covered = ((target >= lower) & (target <= upper)).astype(np.float32)
    width = upper - lower
    return float(covered.mean()), float(width.mean())


def energy_score(samples: np.ndarray, target: np.ndarray) -> float:
    sample_flat = samples.reshape(samples.shape[0], -1)
    target_flat = target.reshape(-1)
    first = np.linalg.norm(sample_flat - target_flat[None], axis=1).mean()
    pair = np.linalg.norm(sample_flat[:, None, :] - sample_flat[None, :, :], axis=-1).mean()
    return float(first - 0.5 * pair)


def summarize_scan_metrics(scan_predictions: dict[str, ScanPrediction]) -> dict[str, float]:
    metrics: dict[str, list[float]] = {
        "pearson_r": [],
        "rmse": [],
        "r2": [],
        "psd_corr": [],
        "fc_corr": [],
    }
    probabilistic: dict[str, list[float]] = {
        "crps": [],
        "coverage_90": [],
        "width_90": [],
        "energy_score": [],
    }
    for prediction in scan_predictions.values():
        metrics["pearson_r"].append(pearson_r(prediction.prediction, prediction.target))
        metrics["rmse"].append(rmse(prediction.prediction, prediction.target))
        metrics["r2"].append(r2_score(prediction.prediction, prediction.target))
        metrics["psd_corr"].append(psd_correlation(prediction.prediction, prediction.target))
        metrics["fc_corr"].append(fc_correlation(prediction.prediction, prediction.target))
        if prediction.samples is not None:
            probabilistic["crps"].append(crps_ensemble(prediction.samples, prediction.target))
            coverage, width = interval_metrics(prediction.samples, prediction.target, alpha=0.1)
            probabilistic["coverage_90"].append(coverage)
            probabilistic["width_90"].append(width)
            probabilistic["energy_score"].append(energy_score(prediction.samples, prediction.target))

    summary = {name: float(np.mean(values)) for name, values in metrics.items()}
    for name, values in probabilistic.items():
        if values:
            summary[name] = float(np.mean(values))
    return summary
