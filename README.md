# EEG2fMRI

Residual conditional flow matching for probabilistic EEG-to-fMRI ROI time-course prediction on NeuroBOLT-style data.

## What is implemented

- NeuroBOLT-format dataset discovery:
  - `data/EEG_set/subXX-scanYY.set`
  - `data/fMRI_difumo64/subXX-scanYY_difumo64_roi.pkl`
- Subject-wise train/val/test split
- Chunk-level training with causal EEG context
- EEG encoder with temporal and spectral branches
- HRF-aligned condition tokens
- Deterministic mean head plus residual conditional flow matching decoder
- Scan-level overlap-add reconstruction
- Evaluation metrics:
  - Pearson `r`
  - `R^2`
  - RMSE
  - PSD correlation
  - FC correlation
  - CRPS
  - 90% interval coverage / width
  - Energy score

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
├── EEG_set/
│   ├── sub01-scan01.set
│   ├── sub02-scan01.set
│   └── ...
└── fMRI_difumo64/
    ├── sub01-scan01_difumo64_roi.pkl
    ├── sub02-scan01_difumo64_roi.pkl
    └── ...
```

The loader also supports `.npy/.npz` EEG and `.npy/.npz/.csv/.tsv` ROI files if you adjust `data.eeg_glob` and `data.fmri_suffix`.

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

## Notes

- The `.set` loader supports EEGLAB MAT files and external `.fdt` payloads.
- ROI `.pkl` loading is intentionally permissive because public NeuroBOLT files can be wrapped in dictionaries with different keys.
- The current implementation assumes the EEG has already been resampled to about `200 Hz` and aligned to fMRI acquisition, matching the public NeuroBOLT release.
- The first `mean_only_epochs` train only the deterministic head. After that, the residual flow branch is enabled.
