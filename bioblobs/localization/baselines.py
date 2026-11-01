"""Residue-score baselines: uniform random, burial, integrated gradients and attention weights."""

from __future__ import annotations

from collections.abc import Callable

import torch


def random_scores(num_residues: int, generator: torch.Generator) -> torch.Tensor:
    """Uniform random residue scores in [0, 1), ``[N]``."""
    return torch.rand(num_residues, generator=generator)


def burial_scores(coords: torch.Tensor, radius: float = 10.0) -> torch.Tensor:
    """Number of C-alpha atoms within ``radius`` angstroms of each residue (itself included).

    Args:
        coords: C-alpha coordinates of one protein, ``[N, 3]``; non-finite rows are allowed.
        radius: Neighborhood radius.

    Returns:
        Burial counts, ``[N]``; zero for residues without a finite C-alpha.
    """
    finite = torch.isfinite(coords.sum(-1))
    counts = ((torch.cdist(coords, coords) < radius) & finite[None, :]).sum(-1).float()
    counts[~finite] = 0.0
    return counts


def integrated_gradients(
    forward: Callable[[torch.Tensor], torch.Tensor],
    features: torch.Tensor,
    target: torch.Tensor | int,
    steps: int = 16,
    method: str = "riemann_right",
) -> torch.Tensor:
    """Integrated gradients of target logits with respect to residue embeddings, zero reference.

    The residue score is the attribution summed over embedding dimensions (signed).

    Args:
        forward: Maps residue embeddings ``[N_total, D]`` to logits ``[B, C]``.
        features: Residue embeddings, ``[N_total, D]``.
        target: Class index per protein, ``[B]`` (or one int for a single protein).
        steps: Number of interpolation steps.
        method: ``"riemann_right"`` (alpha = k / steps, k = 1..steps) or a captum
            integration method such as ``"gausslegendre"``.

    Returns:
        Signed residue attributions, ``[N_total]``.
    """
    x = features.detach()
    target = torch.as_tensor(target, device=x.device).view(-1)
    if method != "riemann_right":
        from captum.attr import IntegratedGradients

        attr = IntegratedGradients(forward).attribute(
            x, baselines=torch.zeros_like(x), target=int(target[0]), n_steps=steps, method=method
        )
        return attr.sum(dim=-1).detach()
    total = torch.zeros_like(x)
    for k in range(1, steps + 1):
        xa = (k / steps * x).requires_grad_(True)
        selected = forward(xa).gather(1, target[:, None]).sum()
        total += torch.autograd.grad(selected, xa)[0]
    return (x * total / steps).sum(dim=-1).detach()
