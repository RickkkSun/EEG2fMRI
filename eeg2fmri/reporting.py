from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = str(Path(tempfile.gettempdir()) / "eeg2fmri-mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from eeg2fmri.training.metrics import (
    BENCHMARK_ROI_ALIASES,
    ScanPrediction,
    benchmark_roi_indices,
    benchmark_table_rows,
    protocol_summary_rows,
    representative_scan_id,
    roi_metric_rows,
    scan_metric_rows,
    summarize_scan_metrics,
)


def _write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(target, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _save_predictions(path: str | Path, scan_predictions: dict[str, ScanPrediction]) -> None:
    arrays: dict[str, Any] = {"scan_ids": np.array(list(scan_predictions.keys()), dtype=object)}
    for scan_id, prediction in scan_predictions.items():
        arrays[f"{scan_id}__prediction"] = prediction.prediction.astype(np.float32)
        arrays[f"{scan_id}__target"] = prediction.target.astype(np.float32)
        if prediction.samples is not None:
            arrays[f"{scan_id}__samples"] = prediction.samples.astype(np.float32)
        if prediction.roi_names is not None:
            arrays[f"{scan_id}__roi_names"] = np.array(prediction.roi_names, dtype=object)
        if prediction.subject_id is not None:
            arrays[f"{scan_id}__subject_id"] = np.array(prediction.subject_id, dtype=object)
    np.savez_compressed(Path(path), **arrays)


def _benchmark_roi_keys(scan_predictions: dict[str, ScanPrediction]) -> list[str]:
    if not scan_predictions:
        return []
    first = next(iter(scan_predictions.values()))
    roi_indices = benchmark_roi_indices(first.roi_names)
    keys = [key for key in BENCHMARK_ROI_ALIASES if roi_indices.get(key) is not None]
    return keys


def _plot_benchmark_overview(path: str | Path, scan_predictions: dict[str, ScanPrediction]) -> None:
    roi_keys = _benchmark_roi_keys(scan_predictions)
    if not roi_keys:
        return

    scan_rows = scan_metric_rows(scan_predictions)
    n_rows = len(roi_keys)
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 3.0 * n_rows), constrained_layout=True)
    axes = np.atleast_2d(axes)

    for row_idx, roi_key in enumerate(roi_keys):
        hist_ax = axes[row_idx, 0]
        ts_ax = axes[row_idx, 1]
        metric_key = f"roi_{roi_key}_pearson_r"
        values = [float(row[metric_key]) for row in scan_rows if metric_key in row]
        if not values:
            continue

        hist_ax.hist(values, bins=min(8, max(4, len(values))), color="#92c5de", alpha=0.85, edgecolor="white")
        hist_ax.axvline(float(np.mean(values)), color="#404040", linestyle="--", linewidth=1.5)
        hist_ax.set_title(metric_key)
        hist_ax.set_xlabel("Pearson r")
        hist_ax.set_ylabel("Count")

        scan_id = representative_scan_id(scan_predictions, benchmark_roi=roi_key)
        if scan_id is None:
            continue
        prediction = scan_predictions[scan_id]
        roi_idx = benchmark_roi_indices(prediction.roi_names).get(roi_key)
        if roi_idx is None:
            continue

        true_signal = prediction.target[:, roi_idx]
        pred_signal = prediction.prediction[:, roi_idx]
        trs = np.arange(true_signal.shape[0], dtype=np.int32)
        ts_ax.plot(trs, true_signal, color="#4d4d4d", linestyle="--", linewidth=1.6, label="True")
        ts_ax.plot(trs, pred_signal, color="#1b9e77", linewidth=1.8, label="Pred")
        if prediction.samples is not None:
            lower = np.quantile(prediction.samples[:, :, roi_idx], 0.05, axis=0)
            upper = np.quantile(prediction.samples[:, :, roi_idx], 0.95, axis=0)
            ts_ax.fill_between(trs, lower, upper, color="#1b9e77", alpha=0.18, label="90% interval")
        ts_ax.set_title(f"{scan_id} ({metric_key})")
        ts_ax.set_xlabel("Time (TR)")
        ts_ax.set_ylabel("Normalized signal")
        if row_idx == 0:
            ts_ax.legend(loc="upper right")

    fig.savefig(Path(path), dpi=180)
    plt.close(fig)


def _plot_fc_psd_diagnostics(path: str | Path, scan_predictions: dict[str, ScanPrediction]) -> None:
    scan_id = representative_scan_id(scan_predictions, metric_key="pearson_r")
    if scan_id is None:
        return
    prediction = scan_predictions[scan_id]
    pred_fc = np.nan_to_num(np.corrcoef(prediction.prediction, rowvar=False), nan=0.0)
    true_fc = np.nan_to_num(np.corrcoef(prediction.target, rowvar=False), nan=0.0)

    pred_fft = np.abs(np.fft.rfft(prediction.prediction - prediction.prediction.mean(axis=0, keepdims=True), axis=0)) ** 2
    true_fft = np.abs(np.fft.rfft(prediction.target - prediction.target.mean(axis=0, keepdims=True), axis=0)) ** 2
    freq = np.arange(pred_fft.shape[0], dtype=np.int32)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    im0 = axes[0, 0].imshow(true_fc, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    axes[0, 0].set_title("True FC")
    im1 = axes[0, 1].imshow(pred_fc, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    axes[0, 1].set_title("Predicted FC")
    axes[1, 0].imshow(pred_fc - true_fc, cmap="coolwarm")
    axes[1, 0].set_title("FC Difference")
    axes[1, 1].plot(freq, np.log1p(true_fft.mean(axis=1)), color="#4d4d4d", linestyle="--", linewidth=1.6, label="True PSD")
    axes[1, 1].plot(freq, np.log1p(pred_fft.mean(axis=1)), color="#d95f02", linewidth=1.8, label="Pred PSD")
    axes[1, 1].set_title(f"Representative Scan: {scan_id}")
    axes[1, 1].set_xlabel("Frequency Bin")
    axes[1, 1].set_ylabel("log(PSD)")
    axes[1, 1].legend(loc="upper right")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
    fig.savefig(Path(path), dpi=180)
    plt.close(fig)


def export_evaluation_report(
    output_dir: str | Path,
    split_name: str,
    scan_predictions: dict[str, ScanPrediction],
    save_predictions: bool = False,
) -> dict[str, float]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    summary = summarize_scan_metrics(scan_predictions)
    _write_json(output / f"summary_{split_name}.json", summary)
    _write_csv(output / f"per_scan_metrics_{split_name}.csv", scan_metric_rows(scan_predictions))
    _write_csv(output / f"per_roi_metrics_{split_name}.csv", roi_metric_rows(scan_predictions))
    _write_csv(output / f"benchmark_table_{split_name}.csv", benchmark_table_rows(scan_predictions))
    _plot_benchmark_overview(output / f"benchmark_overview_{split_name}.png", scan_predictions)
    _plot_fc_psd_diagnostics(output / f"diagnostics_{split_name}.png", scan_predictions)
    if save_predictions:
        _save_predictions(output / f"predictions_{split_name}.npz", scan_predictions)
    return summary


def export_protocol_report(
    output_dir: str | Path,
    split_summaries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "split_summaries.csv", split_summaries)
    aggregated = protocol_summary_rows(split_summaries)
    _write_csv(output / "aggregate_summary.csv", aggregated)
    _write_json(output / "aggregate_summary.json", aggregated)
    return aggregated
