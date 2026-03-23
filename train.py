from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from eeg2fmri.config import load_config
from eeg2fmri.data import NeuroBoltDataModule
from eeg2fmri.models import NeuroFlowMatch
from eeg2fmri.training import Trainer
from eeg2fmri.utils import seed_everything


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train NeuroFlowMatch on NeuroBOLT-format data.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--data.root", dest="data_root", type=str, default=None, help="Override dataset root.")
    parser.add_argument("--data.eeg_dirname", dest="eeg_dirname", type=str, default=None, help="Override EEG directory under data root.")
    parser.add_argument("--data.fmri_dirname", dest="fmri_dirname", type=str, default=None, help="Override fMRI directory under data root.")
    parser.add_argument("--data.eeg_glob", dest="eeg_glob", type=str, default=None, help="Override EEG filename glob.")
    parser.add_argument("--data.eeg_fs", dest="eeg_fs", type=float, default=None, help="Override EEG sampling rate.")
    parser.add_argument("--runtime.output_dir", dest="output_dir", type=str, default=None, help="Override output dir.")
    parser.add_argument("--runtime.device", dest="device", type=str, default=None, help="Override device.")
    args = parser.parse_args()

    config = load_config(args.config)
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
    if args.output_dir is not None:
        config.runtime.output_dir = args.output_dir
    if args.device is not None:
        config.runtime.device = args.device

    seed_everything(config.optim.seed)
    device = resolve_device(config.runtime.device)

    datamodule = NeuroBoltDataModule(config.data, config.optim)
    datamodule.setup()
    print(json.dumps(datamodule.describe(), indent=2))
    n_channels = datamodule.datasets["train"][0]["eeg"].shape[0]

    model = NeuroFlowMatch(
        data_config=config.data,
        model_config=config.model,
        n_channels=int(n_channels),
    )
    trainer = Trainer(
        model=model,
        config=config,
        output_dir=Path(config.runtime.output_dir),
        device=device,
    )
    best = trainer.fit(
        train_loader=datamodule.train_dataloader(),
        val_loader=datamodule.val_dataloader(),
    )
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
