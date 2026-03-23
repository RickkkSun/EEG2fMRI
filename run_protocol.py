from __future__ import annotations

import argparse
import json
from pathlib import Path

from eeg2fmri.config import load_config
from eeg2fmri.data import NeuroBoltDataModule
from eeg2fmri.models import NeuroFlowMatch
from eeg2fmri.reporting import export_evaluation_report, export_protocol_report
from eeg2fmri.training import Trainer
from eeg2fmri.utils import add_common_override_args, apply_common_overrides, resolve_device, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a full NeuroFlowMatch protocol over all splits of a strategy.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    add_common_override_args(parser)
    args = parser.parse_args()

    config = load_config(args.config)
    config = apply_common_overrides(config, args)
    if config.data.split_strategy in {"inter_subject_kfold", "loso", "intra_subject_temporal"} and args.fold_index is None:
        config.data.fold_index = None

    seed_everything(config.optim.seed)
    device = resolve_device(config.runtime.device)

    probe_datamodule = NeuroBoltDataModule(config.data, config.optim)
    split_manifests = []
    split_summaries = []
    base_output = Path(config.runtime.output_dir)
    base_output.mkdir(parents=True, exist_ok=True)

    for split_index, split in enumerate(probe_datamodule.available_splits):
        seed_everything(config.optim.seed + split_index)
        split_output = base_output / split.name
        split_output.mkdir(parents=True, exist_ok=True)

        datamodule = NeuroBoltDataModule(
            config.data,
            config.optim,
            split=split,
            records=probe_datamodule.records,
        )
        datamodule.setup()
        with open(split_output / "split_manifest.json", "w", encoding="utf-8") as handle:
            json.dump(datamodule.split_manifest(), handle, indent=2)

        model = NeuroFlowMatch(
            data_config=config.data,
            model_config=config.model,
            n_channels=datamodule.n_channels(),
            target_dim=datamodule.target_dim(),
        )
        trainer = Trainer(
            model=model,
            config=config,
            output_dir=split_output,
            device=device,
        )
        best_summary = trainer.fit(
            train_loader=datamodule.train_dataloader(),
            val_loader=datamodule.val_dataloader(),
        )
        prefixed_test, scan_predictions = trainer.evaluate(
            datamodule.test_dataloader(),
            split_name="test",
            return_predictions=True,
        )
        raw_test_summary = export_evaluation_report(
            output_dir=split_output,
            split_name="test",
            scan_predictions=scan_predictions,
            save_predictions=config.eval.save_predictions,
        )
        split_row = {
            "split_name": split.name,
            "fold_index": split_index,
            "train_scans": len(split.train_records),
            "val_scans": len(split.val_records),
            "test_scans": len(split.test_records),
            "train_subjects": len({record.subject_id for record in split.train_records}),
            "val_subjects": len({record.subject_id for record in split.val_records}),
            "test_subjects": len({record.subject_id for record in split.test_records}),
            **prefixed_test,
        }
        if best_summary:
            for key, value in best_summary.items():
                if isinstance(value, (int, float)):
                    split_row[f"best_{key}"] = value
        split_summaries.append(split_row)
        split_manifests.append(
            {
                "split_name": split.name,
                "summary": raw_test_summary,
                "manifest": datamodule.split_manifest(),
            }
        )

    with open(base_output / "protocol_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "strategy": config.data.split_strategy,
                "num_splits": len(split_manifests),
                "splits": split_manifests,
            },
            handle,
            indent=2,
        )
    aggregate = export_protocol_report(base_output, split_summaries)
    print(json.dumps({"num_splits": len(split_summaries), "aggregate": aggregate}, indent=2))


if __name__ == "__main__":
    main()
