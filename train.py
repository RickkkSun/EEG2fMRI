from __future__ import annotations

import argparse
import json
from pathlib import Path

from eeg2fmri.config import load_config
from eeg2fmri.data import NeuroBoltDataModule
from eeg2fmri.models import NeuroFlowMatch
from eeg2fmri.training import Trainer
from eeg2fmri.utils import add_common_override_args, apply_common_overrides, resolve_device, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Train NeuroFlowMatch on NeuroBOLT-format data.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    add_common_override_args(parser)
    args = parser.parse_args()

    config = load_config(args.config)
    config = apply_common_overrides(config, args)

    seed_everything(config.optim.seed)
    device = resolve_device(config.runtime.device)

    datamodule = NeuroBoltDataModule(config.data, config.optim)
    datamodule.setup()
    output_dir = Path(config.runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "split_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(datamodule.split_manifest(), handle, indent=2)
    print(json.dumps(datamodule.describe(), indent=2))
    n_channels = datamodule.n_channels()
    target_dim = datamodule.target_dim()

    model = NeuroFlowMatch(
        data_config=config.data,
        model_config=config.model,
        n_channels=int(n_channels),
        target_dim=int(target_dim),
    )
    trainer = Trainer(
        model=model,
        config=config,
        output_dir=output_dir,
        device=device,
    )
    best = trainer.fit(
        train_loader=datamodule.train_dataloader(),
        val_loader=datamodule.val_dataloader(),
    )
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
