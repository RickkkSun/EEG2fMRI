from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ScanPrediction:
    target: np.ndarray
    prediction: np.ndarray
    samples: np.ndarray | None
    roi_names: list[str] | None
    subject_id: str | None = None


BENCHMARK_ROI_ALIASES = {
    "cuneus": ["Cuneus"],
    "heschls_gyrus": ["Heschl’s gyrus", "Heschl's gyrus"],
    "middle_frontal_gyrus_anterior": ["Middle frontal gyrus anterior"],
    "precuneus_anterior": ["Precuneus anterior"],
    "putamen": ["Putamen"],
    "thalamus": ["Thalamus"],
    "global_signal": ["global signal clean", "Global signal", "global signal raw"],
}

BENCHMARK_GROUPS = {
    "primary_sensory": ["cuneus", "heschls_gyrus"],
    "high_level_cognitive": ["middle_frontal_gyrus_anterior", "precuneus_anterior"],
    "subcortical": ["putamen", "thalamus"],
}


class OverlapAccumulator:
    def __init__(self, num_samples: int | None = None) -> None:
        self.num_samples = num_samples
        self.storage: dict[str, dict[str, Any]] = {}

    def _ensure(self, scan_id: str, length: int, n_rois: int) -> None:
        if scan_id in self.storage:
            return
        state: dict[str, Any] = {
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
        roi_names: list[str] | None = None,
        samples: np.ndarray | None = None,
        subject_id: str | None = None,
    ) -> None:
        n_steps, n_rois = prediction.shape
        self._ensure(scan_id, length=length, n_rois=n_rois)
        state = self.storage[scan_id]
        if roi_names is not None and "roi_names" not in state:
            state["roi_names"] = np.array(roi_names, dtype=object)
        if subject_id is not None and "subject_id" not in state:
            state["subject_id"] = subject_id
        end = start + n_steps
        state["pred_sum"][start:end] += prediction
        state["target_sum"][start:end] += target
        state["count"][start:end] += 1.0
        if samples is not None and "sample_sum" in state:
            state["sample_sum"][:, start:end] += samples

    def finalize(self) -> dict[str, ScanPrediction]:
        result: dict[str, ScanPrediction] = {}
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
                roi_names=state.get("roi_names", None).tolist() if "roi_names" in state else None,
                subject_id=state.get("subject_id"),
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


def mse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((prediction - target) ** 2))


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


def _resolve_roi_index(roi_names: list[str] | None, aliases: list[str]) -> int | None:
    if not roi_names:
        return None
    normalized = {name.strip().lower(): idx for idx, name in enumerate(roi_names)}
    for alias in aliases:
        key = alias.strip().lower()
        if key in normalized:
            return normalized[key]
    return None


def benchmark_roi_indices(roi_names: list[str] | None) -> dict[str, int | None]:
    return {
        key: _resolve_roi_index(roi_names, aliases)
        for key, aliases in BENCHMARK_ROI_ALIASES.items()
    }


def benchmark_scan_metrics(scan_id: str, prediction: ScanPrediction) -> dict[str, float]:
    roi_indices = benchmark_roi_indices(prediction.roi_names)
    row: dict[str, float] = {}
    selected: list[float] = []

    for roi_key, roi_idx in roi_indices.items():
        if roi_idx is None or roi_idx >= prediction.target.shape[1]:
            continue
        score = _safe_corr(prediction.prediction[:, roi_idx], prediction.target[:, roi_idx])
        row[f"roi_{roi_key}_pearson_r"] = float(score)
        selected.append(float(score))

    for group_name, members in BENCHMARK_GROUPS.items():
        values = [row[f"roi_{member}_pearson_r"] for member in members if f"roi_{member}_pearson_r" in row]
        if values:
            row[f"group_{group_name}_pearson_r"] = float(np.mean(values))

    if selected:
        row["benchmark_avg_r"] = float(np.mean(selected))
    return row


def scan_metric_rows(scan_predictions: dict[str, ScanPrediction]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scan_id, prediction in scan_predictions.items():
        row: dict[str, Any] = {
            "scan_id": scan_id,
            "subject_id": prediction.subject_id,
            "num_trs": int(prediction.target.shape[0]),
            "num_targets": int(prediction.target.shape[1]),
            "pearson_r": pearson_r(prediction.prediction, prediction.target),
            "mse": mse(prediction.prediction, prediction.target),
            "rmse": rmse(prediction.prediction, prediction.target),
            "r2": r2_score(prediction.prediction, prediction.target),
            "psd_corr": psd_correlation(prediction.prediction, prediction.target),
            "fc_corr": fc_correlation(prediction.prediction, prediction.target),
        }
        if prediction.samples is not None:
            coverage, width = interval_metrics(prediction.samples, prediction.target, alpha=0.1)
            row["crps"] = crps_ensemble(prediction.samples, prediction.target)
            row["coverage_90"] = coverage
            row["width_90"] = width
            row["energy_score"] = energy_score(prediction.samples, prediction.target)
        row.update(benchmark_scan_metrics(scan_id, prediction))
        rows.append(row)
    return rows


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted({key for row in rows for key, value in row.items() if isinstance(value, (int, float, np.floating))})
    summary: dict[str, float] = {}
    for key in keys:
        if key in {"num_trs", "num_targets"}:
            continue
        values = [float(row[key]) for row in rows if key in row]
        if values:
            summary[key] = float(np.mean(values))
    return summary


def summarize_scan_metrics(scan_predictions: dict[str, ScanPrediction]) -> dict[str, float]:
    return summarize_rows(scan_metric_rows(scan_predictions))


def roi_metric_rows(scan_predictions: dict[str, ScanPrediction]) -> list[dict[str, Any]]:
    if not scan_predictions:
        return []
    first = next(iter(scan_predictions.values()))
    roi_names = first.roi_names or [f"roi_{idx:02d}" for idx in range(first.target.shape[1])]
    rows: list[dict[str, Any]] = []

    for roi_idx, roi_name in enumerate(roi_names):
        correlations = []
        rmses = []
        for prediction in scan_predictions.values():
            correlations.append(_safe_corr(prediction.prediction[:, roi_idx], prediction.target[:, roi_idx]))
            rmses.append(float(np.sqrt(np.mean((prediction.prediction[:, roi_idx] - prediction.target[:, roi_idx]) ** 2))))
        rows.append(
            {
                "roi_index": roi_idx,
                "roi_name": roi_name,
                "mean_pearson_r": float(np.mean(correlations)),
                "std_pearson_r": float(np.std(correlations)),
                "mean_rmse": float(np.mean(rmses)),
                "std_rmse": float(np.std(rmses)),
            }
        )
    return rows


def benchmark_table_rows(scan_predictions: dict[str, ScanPrediction]) -> list[dict[str, Any]]:
    scan_rows = scan_metric_rows(scan_predictions)
    if not scan_rows:
        return []

    metric_names = []
    for roi_key in BENCHMARK_ROI_ALIASES:
        metric_names.append(f"roi_{roi_key}_pearson_r")
    for group_name in BENCHMARK_GROUPS:
        metric_names.append(f"group_{group_name}_pearson_r")
    metric_names.append("benchmark_avg_r")
    metric_names.append("pearson_r")

    rows: list[dict[str, Any]] = []
    for metric_name in metric_names:
        values = [float(row[metric_name]) for row in scan_rows if metric_name in row]
        if not values:
            continue
        rows.append(
            {
                "metric": metric_name,
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "n_scans": len(values),
                "formatted": f"{np.mean(values):.3f} ± {np.std(values):.3f}",
            }
        )
    return rows


def representative_scan_id(
    scan_predictions: dict[str, ScanPrediction],
    metric_key: str = "pearson_r",
    benchmark_roi: str | None = None,
) -> str | None:
    rows = scan_metric_rows(scan_predictions)
    if not rows:
        return None
    if benchmark_roi is not None:
        metric_key = f"roi_{benchmark_roi}_pearson_r"
    values = [row[metric_key] for row in rows if metric_key in row]
    if not values:
        return rows[0]["scan_id"]
    mean_value = float(np.mean(values))
    best_row = min(
        [row for row in rows if metric_key in row],
        key=lambda row: abs(float(row[metric_key]) - mean_value),
    )
    return str(best_row["scan_id"])


def protocol_summary_rows(split_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not split_summaries:
        return []
    metric_names = sorted(
        {
            key
            for row in split_summaries
            for key, value in row.items()
            if isinstance(value, (int, float, np.floating)) and key not in {"fold_index"}
        }
    )
    rows: list[dict[str, Any]] = []
    for metric_name in metric_names:
        values = [float(row[metric_name]) for row in split_summaries if metric_name in row]
        if not values:
            continue
        rows.append(
            {
                "metric": metric_name,
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "n_splits": len(values),
                "formatted": f"{np.mean(values):.3f} ± {np.std(values):.3f}",
            }
        )
    return rows
