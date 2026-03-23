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
    parser = argparse.ArgumentParser(description="Evaluate a trained NeuroFlowMatch checkpoint.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--data.root", dest="data_root", type=str, default=None)
    parser.add_argument("--data.eeg_dirname", dest="eeg_dirname", type=str, default=None)
    parser.add_argument("--data.fmri_dirname", dest="fmri_dirname", type=str, default=None)
    parser.add_argument("--data.eeg_glob", dest="eeg_glob", type=str, default=None)
    parser.add_argument("--data.eeg_fs", dest="eeg_fs", type=float, default=None)
    parser.add_argument("--runtime.device", dest="device", type=str, default=None)
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
    if args.device is not None:
        config.runtime.device = args.device

    seed_everything(config.optim.seed)
    device = resolve_device(config.runtime.device)
    datamodule = NeuroBoltDataModule(config.data, config.optim)
    datamodule.setup()
    n_channels = datamodule.datasets["train"][0]["eeg"].shape[0]

    model = NeuroFlowMatch(
        data_config=config.data,
        model_config=config.model,
        n_channels=int(n_channels),
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model"])

    trainer = Trainer(
        model=model,
        config=config,
        output_dir=Path(config.runtime.output_dir),
        device=device,
    )

    if args.split == "train":
        loader = datamodule.train_dataloader()
    elif args.split == "val":
        loader = datamodule.val_dataloader()
    else:
        loader = datamodule.test_dataloader()
    summary = trainer.evaluate(loader, split_name=args.split)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
