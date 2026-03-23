from __future__ import annotations

import math

import torch
from torch import nn

from eeg2fmri.config import DataConfig, ModelConfig
from eeg2fmri.models.components import (
    AttentionBlock,
    ConditionProjector,
    CrossAttentionBlock,
    LearnedPositionalEncoding,
    SinusoidalTimeEmbedding,
)


class HRFAligner(nn.Module):
    def __init__(
        self,
        chunk_length: int,
        dim: int,
        patch_count: int,
        patch_stride: int,
        patch_size: int,
        fs: float,
        tr_seconds: float,
        context_seconds: float,
        peak_seconds: float,
        sigma_seconds: float,
        max_seconds: float,
        heads: int,
        ff_mult: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.chunk_length = chunk_length
        self.query = nn.Parameter(torch.randn(1, chunk_length, dim) * 0.02)
        self.cross = CrossAttentionBlock(dim=dim, heads=heads, ff_mult=ff_mult, dropout=dropout)
        self.self_block = AttentionBlock(dim=dim, heads=heads, ff_mult=ff_mult, dropout=dropout)

        patch_centers = (
            torch.arange(patch_count, dtype=torch.float32) * patch_stride + patch_size / 2.0
        ) / fs
        target_times = context_seconds + torch.arange(chunk_length, dtype=torch.float32) * tr_seconds
        deltas = target_times[:, None] - patch_centers[None, :]
        invalid = (deltas < 0.0) | (deltas > max_seconds)
        bias = -((deltas - peak_seconds) ** 2) / (2.0 * sigma_seconds**2 + 1e-6)
        bias = bias.masked_fill(invalid, float("-inf"))
        self.register_buffer("attn_bias", bias, persistent=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch_size = tokens.shape[0]
        query = self.query.expand(batch_size, -1, -1)
        query = self.cross(query, tokens, attn_mask=self.attn_bias)
        return self.self_block(query)


class FlowBlock(nn.Module):
    def __init__(self, dim: int, heads: int, ff_mult: int, dropout: float) -> None:
        super().__init__()
        self.self_block = AttentionBlock(dim=dim, heads=heads, ff_mult=ff_mult, dropout=dropout)
        self.cross_block = CrossAttentionBlock(dim=dim, heads=heads, ff_mult=ff_mult, dropout=dropout)
        self.cond_gate = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim * 2))
        self.time_gate = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim * 2))
        self.out_norm = nn.LayerNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
        time_embedding: torch.Tensor,
        condition_summary: torch.Tensor,
    ) -> torch.Tensor:
        cond_shift, cond_scale = self.cond_gate(condition_summary).chunk(2, dim=-1)
        time_shift, time_scale = self.time_gate(time_embedding).chunk(2, dim=-1)
        x = x * (1.0 + cond_scale[:, None] + time_scale[:, None]) + cond_shift[:, None] + time_shift[:, None]
        x = self.self_block(x)
        x = self.cross_block(x, condition)
        return self.out_norm(x)


class ResidualFlowDecoder(nn.Module):
    def __init__(
        self,
        chunk_length: int,
        target_dim: int,
        dim: int,
        layers: int,
        heads: int,
        ff_mult: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.chunk_length = chunk_length
        self.target_dim = target_dim
        self.input_proj = nn.Linear(target_dim, dim)
        self.position = LearnedPositionalEncoding(length=chunk_length, dim=dim)
        self.time_embedding = SinusoidalTimeEmbedding(dim)
        self.blocks = nn.ModuleList(
            [FlowBlock(dim=dim, heads=heads, ff_mult=ff_mult, dropout=dropout) for _ in range(layers)]
        )
        self.output_proj = nn.Linear(dim, target_dim)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, condition_tokens: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x_t)
        x = self.position(x)
        time_embed = self.time_embedding(t)
        condition_summary = condition_tokens.mean(dim=1)
        for block in self.blocks:
            x = block(x, condition_tokens, time_embed, condition_summary)
        return self.output_proj(x)


class NeuroFlowMatch(nn.Module):
    def __init__(
        self,
        data_config: DataConfig,
        model_config: ModelConfig,
        n_channels: int = 26,
        target_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.data_config = data_config
        self.model_config = model_config
        extra_targets = int(data_config.include_global_signal_clean) + int(data_config.include_global_signal_raw)
        self.target_dim = target_dim if target_dim is not None else data_config.n_rois + extra_targets
        self.chunk_length = data_config.chunk_length
        input_samples = int(round(data_config.context_seconds * data_config.eeg_fs)) + (
            data_config.chunk_length - 1
        ) * int(round(data_config.tr_seconds * data_config.eeg_fs))
        patch_count = 1 + max(0, (input_samples - model_config.patch_size) // model_config.patch_stride)

        self.condition_encoder = ConditionProjector(
            n_channels=n_channels,
            dim=model_config.d_model,
            patch_size=model_config.patch_size,
            patch_stride=model_config.patch_stride,
            kernel_size=model_config.eeg_conv_kernel,
            layers=model_config.eeg_conv_layers,
            dropout=model_config.eeg_dropout,
            bands=model_config.spectral_bands,
            fs=data_config.eeg_fs,
            token_count=patch_count,
        )
        self.hrf_aligner = HRFAligner(
            chunk_length=data_config.chunk_length,
            dim=model_config.d_model,
            patch_count=patch_count,
            patch_stride=model_config.patch_stride,
            patch_size=model_config.patch_size,
            fs=data_config.eeg_fs,
            tr_seconds=data_config.tr_seconds,
            context_seconds=data_config.context_seconds,
            peak_seconds=model_config.hrf_peak_seconds,
            sigma_seconds=model_config.hrf_sigma_seconds,
            max_seconds=model_config.hrf_max_seconds,
            heads=model_config.condition_heads,
            ff_mult=model_config.condition_ff_mult,
            dropout=model_config.eeg_dropout,
        )
        self.mean_head = nn.Sequential(
            nn.LayerNorm(model_config.d_model),
            nn.Linear(model_config.d_model, model_config.d_model),
            nn.GELU(),
            nn.Linear(model_config.d_model, self.target_dim),
        )
        self.flow = ResidualFlowDecoder(
            chunk_length=data_config.chunk_length,
            target_dim=self.target_dim,
            dim=model_config.d_model,
            layers=model_config.flow_layers,
            heads=model_config.flow_heads,
            ff_mult=model_config.flow_ff_mult,
            dropout=model_config.flow_dropout,
        )

    def encode_condition(self, eeg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.condition_encoder(eeg)
        condition = self.hrf_aligner(tokens)
        mean = self.mean_head(condition)
        return condition, mean

    def velocity(self, x_t: torch.Tensor, t: torch.Tensor, condition_tokens: torch.Tensor) -> torch.Tensor:
        return self.flow(x_t, t, condition_tokens)

    @torch.no_grad()
    def sample_residual(
        self,
        condition_tokens: torch.Tensor,
        ode_steps: int,
        solver: str,
        sigma: float,
        num_samples: int,
    ) -> torch.Tensor:
        batch_size = condition_tokens.shape[0]
        samples = []
        for _ in range(num_samples):
            state = sigma * torch.randn(
                batch_size,
                self.chunk_length,
                self.target_dim,
                device=condition_tokens.device,
                dtype=condition_tokens.dtype,
            )
            dt = 1.0 / ode_steps
            for step in range(ode_steps):
                t0 = torch.full((batch_size,), step * dt, device=state.device, dtype=state.dtype)
                if solver == "euler":
                    state = state + dt * self.velocity(state, t0, condition_tokens)
                else:
                    k1 = self.velocity(state, t0, condition_tokens)
                    midpoint = state + dt * k1
                    t1 = torch.full((batch_size,), (step + 1) * dt, device=state.device, dtype=state.dtype)
                    k2 = self.velocity(midpoint, t1, condition_tokens)
                    state = state + 0.5 * dt * (k1 + k2)
            samples.append(state)
        return torch.stack(samples, dim=1)
