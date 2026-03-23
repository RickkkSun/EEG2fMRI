# EEG2fMRI

Residual conditional flow matching for probabilistic EEG-to-fMRI ROI time-course prediction on NeuroBOLT-style data.

## What is implemented

- NeuroBOLT-format dataset discovery:
  - `data/EEG/subXX-scanYY_eeg.set`
  - `data/fMRI/subXX-scanYY_difumo64_roi.pkl`
- Subject-wise train/val/test split
- Inter-subject holdout, inter-subject k-fold, LOSO, and intra-subject temporal protocols
- Chunk-level training with causal EEG context
- EEG encoder with temporal and spectral branches
- HRF-aligned condition tokens
- Deterministic mean head plus residual conditional flow matching decoder
- Scan-level overlap-add reconstruction
- Optional `global signal clean` / `global signal raw` targets in addition to the 64 ROI channels
- Evaluation metrics:
  - Pearson `r`
  - MSE
  - `R^2`
  - RMSE
  - PSD correlation
  - FC correlation
  - CRPS
  - 90% interval coverage / width
  - Energy score
- Paper-style reporting:
  - per-scan metrics CSV
  - per-ROI metrics CSV
  - benchmark ROI / group table
  - benchmark histogram + representative time-course figure
  - representative FC / PSD diagnostic figure

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data layout

Point the config `data.root` to a folder containing:

```text
data/
├── EEG/
│   ├── sub01-scan01_eeg.set
│   ├── sub02-scan01_eeg.set
│   └── ...
└── fMRI/
    ├── sub01-scan01_difumo64_roi.pkl
    ├── sub02-scan01_difumo64_roi.pkl
    └── ...
```

The loader also supports `.npy/.npz` EEG and `.npy/.npz/.csv/.tsv` ROI files if you adjust `data.eeg_glob` and `data.fmri_suffix`.

The default config now matches the public NeuroBOLT release more closely. If your local folder names differ, override them explicitly:

```bash
--data.root /Users/yourname/Desktop \
--data.eeg_dirname EEG \
--data.fmri_dirname fMRI \
--data.eeg_glob "*_eeg.set"
```

If your local `.set` files are stored at `250 Hz`, the loader will resample them to the paper-aligned `200 Hz` model input by default.

## Train

```bash
python train.py \
  --config configs/neuroflowmatch_base.yaml \
  --data.root /absolute/path/to/data \
  --runtime.output_dir /absolute/path/to/outputs/neuroflowmatch_base
```

## Evaluate

```bash
python evaluate.py \
  --config configs/neuroflowmatch_base.yaml \
  --checkpoint /absolute/path/to/outputs/neuroflowmatch_base/best.pt \
  --data.root /absolute/path/to/data \
  --split test
```

Evaluation writes:

- `summary_test.json`
- `per_scan_metrics_test.csv`
- `per_roi_metrics_test.csv`
- `benchmark_table_test.csv`
- `benchmark_overview_test.png`
- `diagnostics_test.png`
- optionally `predictions_test.npz`

## Run Full Protocols

Subject-wise and intra-subject experiments are intended to be run with `run_protocol.py`.

Inter-subject k-fold:

```bash
python run_protocol.py \
  --config configs/neuroflowmatch_base.yaml \
  --data.root /absolute/path/to/data \
  --data.split_strategy inter_subject_kfold \
  --runtime.output_dir /absolute/path/to/outputs/inter_subject_kfold
```

Leave-one-subject-out:

```bash
python run_protocol.py \
  --config configs/neuroflowmatch_base.yaml \
  --data.root /absolute/path/to/data \
  --data.split_strategy loso \
  --runtime.output_dir /absolute/path/to/outputs/loso
```

Intra-subject temporal:

```bash
python run_protocol.py \
  --config configs/neuroflowmatch_base.yaml \
  --data.root /absolute/path/to/data \
  --data.split_strategy intra_subject_temporal \
  --runtime.output_dir /absolute/path/to/outputs/intra_subject
```

Protocol outputs include per-split subdirectories plus:

- `split_summaries.csv`
- `aggregate_summary.csv`
- `aggregate_summary.json`
- `protocol_manifest.json`

To mirror the official NeuroBOLT cross-subject setup more closely, you can provide an explicit train/val/test assignment file:

```bash
python run_protocol.py \
  --config configs/neuroflowmatch_base.yaml \
  --data.root /absolute/path/to/data \
  --data.split_manifest /absolute/path/to/scan_split_example.xlsx \
  --runtime.output_dir /absolute/path/to/outputs/official_split
```

The manifest parser accepts `csv`, `tsv`, and `xlsx` files with a scan identifier column and a split column.

To reproduce the original single-ROI setting for a benchmark region, pass a named target:

```bash
python train.py \
  --config configs/neuroflowmatch_base.yaml \
  --data.root /absolute/path/to/data \
  --data.target_columns Thalamus \
  --runtime.output_dir /absolute/path/to/outputs/thalamus_single_roi
```

## Notes

- The `.set` loader supports EEGLAB MAT files and external `.fdt` payloads.
- ROI `.pkl` loading is intentionally permissive because public NeuroBOLT files can be wrapped in dictionaries with different keys.
- The loader aligns mixed-channel public EEG files to the common 26 scalp channels used across scans and can resample public `250 Hz` files to the paper-aligned `200 Hz` model input.
- The first `mean_only_epochs` train only the deterministic head. After that, the residual flow branch is enabled.
- The deterministic reconstruction term is configurable, but defaults to `MSE` to stay aligned with the original NeuroBOLT training objective.
