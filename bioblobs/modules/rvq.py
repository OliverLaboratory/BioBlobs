"""Residual vector quantization of blob embeddings.

Each blob embedding is L2-normalized and quantized by a stack of codebooks, where level q
quantizes the residual left by levels 0..q-1; the blob is replaced by the sum of its codes.
Codebooks are updated by exponential moving averages, initialized from data and refreshed
by dead-code revival.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Quantized:
    """RVQ output for a flat set of blob embeddings.

    Attributes:
        embeddings: Straight-through quantized embeddings, zero for invalid blobs, ``[M, d]``.
        codes: Level-0 code index of each blob, ``-1`` for invalid blobs or during warmup,
            ``[M]``.
        commitment: ``beta * ||z - sg(z_q)||^2`` averaged over valid blobs (scalar).
        metrics: Detached codebook-usage diagnostics.
    """

    embeddings: torch.Tensor
    codes: torch.Tensor
    commitment: torch.Tensor
    metrics: dict[str, torch.Tensor] = field(default_factory=dict)


class ResidualVQ(nn.Module):
    """Residual vector quantizer with EMA codebooks.

    Args:
        input_dim: Blob embedding width D.
        num_codes: Codes per level V.
        num_quantizers: Residual depth Q.
        code_dim: Quantization width (None keeps ``input_dim``; otherwise a linear map).
        beta: Commitment weight.
        ema_decay: EMA decay of the codebook statistics.
        ema_eps: Laplace smoothing of the EMA cluster sizes.
        normalize_input: L2-normalize embeddings before quantizing.
        dead_code_revival: Re-seed rarely used codes from the current batch.
        revive_every_steps: Revival period in training steps.
        revive_threshold: Minimum EMA usage below which a code is revived.
        revive_frac: Alternative threshold as a fraction of the mean usage.
        warmup_epochs: Leading epochs in which quantization is bypassed.
        data_init: Initialize codebooks from the first batch with at least V valid blobs.
    """

    def __init__(
        self,
        input_dim: int,
        num_codes: int = 1024,
        num_quantizers: int = 8,
        code_dim: int | None = None,
        beta: float = 0.25,
        ema_decay: float = 0.99,
        ema_eps: float = 1e-5,
        normalize_input: bool = True,
        dead_code_revival: bool = True,
        revive_every_steps: int = 250,
        revive_threshold: float = 1.0,
        revive_frac: float = 0.0,
        warmup_epochs: int = 0,
        data_init: bool = True,
    ) -> None:
        super().__init__()
        code_dim = int(code_dim) if code_dim is not None else int(input_dim)
        self.code_dim = code_dim
        self.num_codes = int(num_codes)
        self.num_quantizers = int(num_quantizers)
        self.beta = float(beta)
        self.ema_decay = float(ema_decay)
        self.ema_eps = float(ema_eps)
        self.normalize_input = bool(normalize_input)
        self.dead_code_revival = bool(dead_code_revival)
        self.revive_every_steps = int(revive_every_steps)
        self.revive_threshold = float(revive_threshold)
        self.revive_frac = float(revive_frac)
        self.warmup_epochs = int(warmup_epochs)
        self.projection = (
            nn.Identity() if code_dim == input_dim else nn.Linear(input_dim, code_dim, bias=False)
        )
        init = torch.randn(self.num_quantizers, self.num_codes, code_dim) * code_dim**-0.5
        self.register_buffer("embeddings", init)
        self.register_buffer("ema_cluster_size", torch.zeros(self.num_quantizers, self.num_codes))
        self.register_buffer("ema_embed_avg", init.clone())
        self.register_buffer("epoch", torch.tensor(0, dtype=torch.long))
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))
        self.register_buffer("initialized", torch.tensor(not data_init))

    @property
    def output_dim(self) -> int:
        """Width of the quantized embeddings."""
        return self.code_dim

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch that controls the quantization warmup."""
        self.epoch.fill_(max(int(epoch), 0))

    @property
    def in_warmup(self) -> bool:
        """Whether quantization is currently bypassed."""
        return int(self.epoch.item()) < self.warmup_epochs

    def forward(self, blob_embeddings: torch.Tensor, valid: torch.Tensor) -> Quantized:
        """Quantize blob embeddings.

        Args:
            blob_embeddings: Flat blob embeddings, ``[M, D]``.
            valid: True for blobs with a real seed, ``[M]``.

        Returns:
            Quantized embeddings ``[M, d]``, level-0 codes and the commitment loss.
        """
        z = self.projection(blob_embeddings)
        if self.normalize_input:
            z = F.normalize(z, dim=-1)
        valid = valid.bool()
        valid_f = valid.float().unsqueeze(-1)
        zero = z.new_zeros(())
        if self.in_warmup:
            codes = torch.full((z.shape[0],), -1, dtype=torch.long, device=z.device)
            return Quantized(z * valid_f, codes, zero, {"codebook_bypass": z.new_ones(())})

        if self.training and not bool(self.initialized.item()):
            z_valid = z[valid].detach()
            if z_valid.shape[0] >= self.num_codes:
                self._data_init(z_valid)

        residual, z_q = z, torch.zeros_like(z)
        level_codes, level_inputs = [], []
        for level in range(self.num_quantizers):
            codebook = self.embeddings[level]
            idx = torch.cdist(residual, codebook).argmin(dim=-1)
            code = codebook[idx]
            level_inputs.append(residual.detach())
            level_codes.append(idx)
            z_q = z_q + code
            residual = residual - code

        n_valid = valid.float().sum().clamp_min(1.0)
        commit = ((z - z_q.detach()).pow(2).sum(-1) * valid.float()).sum() / n_valid
        quantized = (z + (z_q - z).detach()) * valid_f
        if self.training:
            self._update_codebooks(level_inputs, level_codes, valid)
        codes = level_codes[0].masked_fill(~valid, -1)
        return Quantized(
            quantized, codes, self.beta * commit, self._usage(level_codes, valid, commit)
        )

    @torch.no_grad()
    def _data_init(self, z_valid: torch.Tensor) -> None:
        residual = z_valid
        for level in range(self.num_quantizers):
            perm = torch.randperm(residual.shape[0], device=residual.device)[: self.num_codes]
            centroids = residual[perm].clone()
            if centroids.shape[0] < self.num_codes:
                centroids = torch.cat([centroids, residual[: self.num_codes - centroids.shape[0]]])
            self.embeddings[level].copy_(centroids)
            self.ema_embed_avg[level].copy_(centroids)
            residual = residual - centroids[torch.cdist(residual, centroids).argmin(dim=-1)]
        self.ema_cluster_size.zero_()
        self.initialized.fill_(True)

    @torch.no_grad()
    def _update_codebooks(
        self, level_inputs: list[torch.Tensor], level_codes: list[torch.Tensor], valid: torch.Tensor
    ) -> None:
        self.step += 1
        revive = (
            self.dead_code_revival
            and self.revive_every_steps > 0
            and int(self.step.item()) % self.revive_every_steps == 0
        )
        for level in range(self.num_quantizers):
            x, idx = level_inputs[level][valid], level_codes[level][valid]
            if idx.numel() == 0:
                continue
            self._ema_update(level, x, idx)
            if revive:
                self._revive(level, x)

    @torch.no_grad()
    def _ema_update(self, level: int, x: torch.Tensor, idx: torch.Tensor) -> None:
        onehot = F.one_hot(idx, self.num_codes).to(x.dtype)
        decay = self.ema_decay
        self.ema_cluster_size[level].mul_(decay).add_(onehot.sum(0), alpha=1 - decay)
        self.ema_embed_avg[level].mul_(decay).add_(onehot.t() @ x, alpha=1 - decay)
        n = self.ema_cluster_size[level].sum()
        smoothed = (
            (self.ema_cluster_size[level] + self.ema_eps) / (n + self.num_codes * self.ema_eps) * n
        )
        self.embeddings[level].copy_(
            self.ema_embed_avg[level] / smoothed.unsqueeze(1).clamp_min(self.ema_eps)
        )

    @torch.no_grad()
    def _revive(self, level: int, x: torch.Tensor) -> None:
        usage = self.ema_cluster_size[level]
        dead = usage < max(self.revive_threshold, self.revive_frac * float(usage.mean()))
        n_dead = int(dead.sum().item())
        if n_dead == 0:
            return
        pick = torch.randint(0, x.shape[0], (n_dead,), device=x.device)
        self.embeddings[level][dead] = x[pick]
        self.ema_embed_avg[level][dead] = x[pick]
        self.ema_cluster_size[level][dead] = 1.0

    @torch.no_grad()
    def _usage(
        self, level_codes: list[torch.Tensor], valid: torch.Tensor, commit: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        perplexities, active = [], []
        for idx in level_codes:
            counts = torch.bincount(idx[valid], minlength=self.num_codes).float()
            p = counts / counts.sum().clamp_min(1.0)
            nz = p > 0
            perplexities.append(torch.exp(-(p[nz] * p[nz].log()).sum()))
            active.append((counts > 0).sum().float())
        return {
            "codebook_perplexity": perplexities[0],
            "codebook_perplexity_mean": torch.stack(perplexities).mean(),
            "codebook_active": active[0],
            "codebook_active_mean": torch.stack(active).mean(),
            "codebook_commit": commit.detach(),
            "codebook_bypass": commit.new_zeros(()),
        }
