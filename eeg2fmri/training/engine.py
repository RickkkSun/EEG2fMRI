from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch import amp
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm

from eeg2fmri.config import TrainConfig
from eeg2fmri.models import NeuroFlowMatch
from eeg2fmri.training.losses import (
    conditional_flow_matching_loss,
    mean_prediction_loss,
    temporal_difference_loss,
)
from eeg2fmri.training.metrics import OverlapAccumulator, summarize_scan_metrics
from eeg2fmri.utils.checkpoint import save_checkpoint


class Trainer:
    def __init__(
        self,
        model: NeuroFlowMatch,
        config: TrainConfig,
        output_dir: str | Path,
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.optim.lr,
            weight_decay=config.optim.weight_decay,
            betas=config.optim.betas,
        )
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config.optim.max_epochs)
        self.scaler = amp.GradScaler(device="cuda", enabled=config.optim.amp and device.type == "cuda")
        self.history_path = self.output_dir / "history.csv"
        self.best_path = self.output_dir / "best.pt"
        self.last_path = self.output_dir / "last.pt"
        self.config_path = self.output_dir / "config.json"
        with open(self.config_path, "w", encoding="utf-8") as handle:
            json.dump(asdict(config), handle, indent=2)

    def _move_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        return {
            "eeg": batch["eeg"].to(self.device, non_blocking=True),
            "target": batch["target"].to(self.device, non_blocking=True),
            "scan_id": batch["scan_id"],
            "subject_id": batch["subject_id"],
            "start_tr": batch["start_tr"],
            "length": batch["length"],
        }

    def _compute_losses(self, batch: dict[str, Any], epoch: int) -> tuple[torch.Tensor, dict[str, float]]:
        eeg = batch["eeg"]
        target = batch["target"]
        condition_tokens, mean_prediction = self.model.encode_condition(eeg)
        mean_loss = mean_prediction_loss(
            mean_prediction,
            target,
            delta=self.config.loss.huber_delta,
        )
        temporal_loss = temporal_difference_loss(mean_prediction, target)
        total = (
            self.config.loss.mean_weight * mean_loss
            + self.config.loss.temporal_weight * temporal_loss
        )
        metrics = {
            "loss": float(total.detach().item()),
            "mean_loss": float(mean_loss.detach().item()),
            "temporal_loss": float(temporal_loss.detach().item()),
            "cfm_loss": 0.0,
        }

        if epoch >= self.config.optim.mean_only_epochs:
            sigma = self.config.model.noise_sigma
            residual_target = target - mean_prediction.detach()
            eps = torch.randn_like(residual_target)
            t = torch.rand(target.shape[0], device=target.device, dtype=target.dtype)
            t_view = t[:, None, None]
            x_t = (1.0 - t_view) * sigma * eps + t_view * residual_target
            velocity_target = residual_target - sigma * eps
            velocity_prediction = self.model.velocity(x_t, t, condition_tokens)
            cfm_loss = conditional_flow_matching_loss(velocity_prediction, velocity_target)
            total = total + self.config.loss.cfm_weight * cfm_loss
            metrics["loss"] = float(total.detach().item())
            metrics["cfm_loss"] = float(cfm_loss.detach().item())

        return total, metrics

    def _write_history(self, rows: list[dict[str, float]]) -> None:
        if not rows:
            return
        write_header = not self.history_path.exists()
        with open(self.history_path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            if write_header:
                writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def train_epoch(self, loader: torch.utils.data.DataLoader, epoch: int) -> dict[str, float]:
        self.model.train()
        running: dict[str, list[float]] = {"loss": [], "mean_loss": [], "temporal_loss": [], "cfm_loss": []}
        iterator = tqdm(loader, desc=f"train {epoch:03d}", leave=False)
        for step, batch in enumerate(iterator, start=1):
            batch = self._move_batch(batch)
            self.optimizer.zero_grad(set_to_none=True)
            with amp.autocast("cuda", enabled=self.config.optim.amp and self.device.type == "cuda"):
                loss, metrics = self._compute_losses(batch, epoch=epoch)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.optim.grad_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            for key, value in metrics.items():
                running[key].append(value)
            if step % self.config.optim.log_every == 0:
                iterator.set_postfix({key: f"{np.mean(values):.4f}" for key, values in running.items()})

        return {key: float(np.mean(values)) for key, values in running.items()}

    @torch.no_grad()
    def evaluate(
        self,
        loader: torch.utils.data.DataLoader,
        split_name: str,
    ) -> dict[str, float]:
        self.model.eval()
        accumulator = OverlapAccumulator(num_samples=self.config.eval.num_samples)
        iterator = tqdm(loader, desc=f"{split_name}", leave=False)
        for batch in iterator:
            batch = self._move_batch(batch)
            eeg = batch["eeg"]
            target = batch["target"]
            condition_tokens, mean_prediction = self.model.encode_condition(eeg)
            residual_samples = self.model.sample_residual(
                condition_tokens=condition_tokens,
                ode_steps=self.config.eval.ode_steps,
                solver=self.config.eval.ode_solver,
                sigma=self.config.model.noise_sigma,
                num_samples=self.config.eval.num_samples,
            )
            prediction_samples = mean_prediction[:, None] + residual_samples
            prediction = prediction_samples.mean(dim=1)

            prediction_np = prediction.detach().cpu().numpy()
            target_np = target.detach().cpu().numpy()
            sample_np = prediction_samples.detach().cpu().numpy()
            start_tr = batch["start_tr"].cpu().numpy()
            length = batch["length"].cpu().numpy()

            for idx, scan_id in enumerate(batch["scan_id"]):
                accumulator.add(
                    scan_id=scan_id,
                    start=int(start_tr[idx]),
                    length=int(length[idx]),
                    prediction=prediction_np[idx],
                    target=target_np[idx],
                    samples=sample_np[idx],
                )

        summary = summarize_scan_metrics(accumulator.finalize())
        return {f"{split_name}_{key}": value for key, value in summary.items()}

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
    ) -> dict[str, float]:
        best_metric = float("-inf")
        best_summary: dict[str, float] = {}
        history_rows: list[dict[str, float]] = []

        for epoch in range(1, self.config.optim.max_epochs + 1):
            train_summary = self.train_epoch(train_loader, epoch)
            row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_summary.items()}}

            if epoch % self.config.optim.val_every == 0:
                val_summary = self.evaluate(val_loader, split_name="val")
                row.update(val_summary)
                current_metric = val_summary.get("val_pearson_r", float("-inf"))
                if current_metric > best_metric:
                    best_metric = current_metric
                    best_summary = dict(row)
                    save_checkpoint(
                        self.best_path,
                        model=self.model,
                        optimizer=self.optimizer,
                        epoch=epoch,
                        best_metric=best_metric,
                        config=asdict(self.config),
                    )

            save_checkpoint(
                self.last_path,
                model=self.model,
                optimizer=self.optimizer,
                epoch=epoch,
                best_metric=best_metric,
                config=asdict(self.config),
            )
            self.scheduler.step()
            history_rows.append(row)
            self._write_history([row])

        return best_summary
