from __future__ import annotations

import math

import torch
from einops import rearrange
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 0:
            t = t[None]
        t = t.view(-1)
        half_dim = self.dim // 2
        scale = math.log(10000.0) / max(half_dim - 1, 1)
        freqs = torch.exp(
            torch.arange(half_dim, device=t.device, dtype=t.dtype) * (-scale)
        )
        angles = t[:, None] * freqs[None, :]
        embedding = torch.cat([angles.sin(), angles.cos()], dim=-1)
        if embedding.shape[-1] < self.dim:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return self.proj(embedding)


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, length: int, dim: int) -> None:
        super().__init__()
        self.embedding = nn.Parameter(torch.zeros(1, length, dim))
        nn.init.trunc_normal_(self.embedding, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.embedding[:, : x.shape[1]]


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int, dropout: float) -> None:
        super().__init__()
        hidden = dim * mult
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int, ff_mult: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, mult=ff_mult, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x, _ = self.self_attn(x, x, x, need_weights=False)
        x = residual + x
        return x + self.ff(self.norm2(x))


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int, ff_mult: int, dropout: float) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_ff = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, mult=ff_mult, dropout=dropout)

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = query
        query_norm = self.norm_q(query)
        context_norm = self.norm_kv(context)
        out, _ = self.cross_attn(
            query_norm,
            context_norm,
            context_norm,
            attn_mask=attn_mask,
            need_weights=False,
        )
        query = residual + out
        return query + self.ff(self.norm_ff(query))


class SpectralPatchEncoder(nn.Module):
    def __init__(
        self,
        n_channels: int,
        patch_size: int,
        patch_stride: int,
        bands: tuple[tuple[float, float], ...],
        fs: float,
        dim: int,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.bands = bands
        self.fs = fs
        self.proj = nn.Sequential(
            nn.Linear(n_channels * len(bands), dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = x.unfold(-1, self.patch_size, self.patch_stride)
        patches = rearrange(patches, "b c n p -> b n c p")
        spectrum = torch.fft.rfft(patches, dim=-1)
        power = spectrum.abs().pow(2)
        freqs = torch.fft.rfftfreq(self.patch_size, d=1.0 / self.fs).to(x.device)

        band_features = []
        for low, high in self.bands:
            mask = (freqs >= low) & (freqs < high)
            if mask.any():
                band_power = power[..., mask].mean(dim=-1)
            else:
                band_power = power[..., :1].mean(dim=-1) * 0.0
            band_features.append(torch.log1p(band_power))
        features = torch.cat(band_features, dim=-1)
        return self.proj(features)


class TemporalConvPatchEncoder(nn.Module):
    def __init__(
        self,
        n_channels: int,
        dim: int,
        patch_size: int,
        patch_stride: int,
        kernel_size: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        blocks = []
        in_channels = n_channels
        for _ in range(layers - 1):
            blocks.extend(
                [
                    nn.Conv1d(in_channels, dim, kernel_size, padding=kernel_size // 2),
                    nn.GroupNorm(8, dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            in_channels = dim
        blocks.append(
            nn.Conv1d(in_channels, dim, kernel_size=patch_size, stride=patch_stride)
        )
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.net(x)
        return tokens.transpose(1, 2)


class ConditionProjector(nn.Module):
    def __init__(
        self,
        n_channels: int,
        dim: int,
        patch_size: int,
        patch_stride: int,
        kernel_size: int,
        layers: int,
        dropout: float,
        bands: tuple[tuple[float, float], ...],
        fs: float,
        token_count: int,
    ) -> None:
        super().__init__()
        self.temporal = TemporalConvPatchEncoder(
            n_channels=n_channels,
            dim=dim,
            patch_size=patch_size,
            patch_stride=patch_stride,
            kernel_size=kernel_size,
            layers=layers,
            dropout=dropout,
        )
        self.spectral = SpectralPatchEncoder(
            n_channels=n_channels,
            patch_size=patch_size,
            patch_stride=patch_stride,
            bands=bands,
            fs=fs,
            dim=dim,
        )
        self.positional = LearnedPositionalEncoding(length=token_count, dim=dim)
        self.fuse = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        temporal_tokens = self.temporal(x)
        spectral_tokens = self.spectral(x)
        tokens = temporal_tokens + spectral_tokens
        tokens = self.positional(tokens)
        return self.fuse(tokens)
