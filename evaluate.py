from __future__ import annotations

import argparse
import json
from pathlib import Path

from eeg2fmri.config import load_config
from eeg2fmri.data import NeuroBoltDataModule
from eeg2fmri.models import NeuroFlowMatch
from eeg2fmri.reporting import export_evaluation_report
from eeg2fmri.training import Trainer
from eeg2fmri.utils import add_common_override_args, apply_common_overrides, resolve_device, seed_everything
import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained NeuroFlowMatch checkpoint.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    add_common_override_args(parser)
    args = parser.parse_args()

    config = load_config(args.config)
    config = apply_common_overrides(config, args)

    seed_everything(config.optim.seed)
    device = resolve_device(config.runtime.device)
    datamodule = NeuroBoltDataModule(config.data, config.optim)
    datamodule.setup()
    n_channels = datamodule.n_channels()
    target_dim = datamodule.target_dim()

    model = NeuroFlowMatch(
        data_config=config.data,
        model_config=config.model,
        n_channels=int(n_channels),
        target_dim=int(target_dim),
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
    prefixed_summary, scan_predictions = trainer.evaluate(loader, split_name=args.split, return_predictions=True)
    summary = export_evaluation_report(
        output_dir=Path(config.runtime.output_dir),
        split_name=args.split,
        scan_predictions=scan_predictions,
        save_predictions=config.eval.save_predictions,
    )
    print(json.dumps({"prefixed": prefixed_summary, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
