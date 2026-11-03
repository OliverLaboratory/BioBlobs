#!/usr/bin/env python
"""Attribution baselines through the mean-pooling EC-L3 classifier (``+experiment=mcsa/meanpool``).

Each method attributes the logit of the annotated EC-L3 class (the predicted class when that class
is not in the classifier's vocabulary) to the frozen residue embeddings ``Z`` of the M-CSA feature
file, and writes one score file per method:

- ``<out>_ig.json``: integrated gradients, zero reference, 32 steps, ``|sum_d IG_id|``.
- ``<out>_grad_x_input.json``: ``|sum_d Z_id * d f / d Z_id|``.
- ``<out>_gradcam.json``: ``ReLU(sum_d w_d Z_id)``, ``w`` the residue-averaged gradient.

Usage::

    python baselines/mcsa/meanpool_saliency.py --checkpoint outputs/mcsa/meanpool/seed0/best.ckpt \\
        --token-map data/swissprot_ec/token_map.json --features data/mcsa/features.pt \\
        --out scores/meanpool
"""

from __future__ import annotations

import argparse
import json

import torch
from captum.attr import IntegratedGradients
from inhouse import DEFAULT_IDS, DEFAULT_LABELS, load_features, read_ids, write_scores

from bioblobs.inference import load_pretrained


def logits_fn(model):
    """``Z [B, N, D] -> logits [B, C]``: B copies of one protein through the mean-pool model."""

    def forward(z: torch.Tensor) -> torch.Tensor:
        b, n, d = z.shape
        batch = torch.arange(b, device=z.device).repeat_interleave(n)
        mask = torch.ones(b * n, dtype=torch.bool, device=z.device)
        return model.predictor(model.encoder(z.reshape(b * n, d)), batch, mask)[0]

    return forward


def target_class(entry: dict, token_map: dict[str, int], logits: torch.Tensor) -> int:
    """Annotated EC-L3 class if the classifier knows it, else the predicted class."""
    l3 = ".".join(str(entry.get("ec_number") or "").split(".")[:3])
    return token_map[l3] if l3 in token_map else int(logits.argmax())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", required=True, help="mean-pool .ckpt or model directory")
    p.add_argument("--token-map", required=True, help="EC-L3 -> class index of the classifier")
    p.add_argument("--features", required=True, help="M-CSA feature file")
    p.add_argument("--ids", default=str(DEFAULT_IDS))
    p.add_argument("--labels", default=str(DEFAULT_LABELS))
    p.add_argument("--steps", type=int, default=32, help="integrated-gradients steps")
    p.add_argument("--out", required=True, help="output prefix")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    model, cfg, _ = load_pretrained(args.checkpoint, args.device)
    if cfg.partitioners.name != "none" or cfg.decoders.name != "mean_pool":
        raise ValueError("expected a mean-pool classifier without a partitioner")
    forward = logits_fn(model)
    token_map = json.load(open(args.token_map))
    entries = {f"{e['pdb_id']}_{e['chain_used']}": e for e in json.load(open(args.labels))}
    features = load_features(args.features)
    out = {"ig": {}, "grad_x_input": {}, "gradcam": {}}
    for key in read_ids(args.ids):
        z = features[key]["nf"].float().to(args.device)[None]  # [1, N, D]
        z.requires_grad_(True)
        logits = forward(z)
        target = target_class(entries[key], token_map, logits.detach()[0])
        (grad,) = torch.autograd.grad(logits[0, target], z)
        z, grad = z.detach()[0], grad[0]
        ig = IntegratedGradients(forward).attribute(
            z[None], baselines=torch.zeros_like(z[None]), target=target, n_steps=args.steps
        )[0]
        out["ig"][key] = ig.sum(-1).abs().cpu().tolist()
        out["grad_x_input"][key] = (grad * z).sum(-1).abs().cpu().tolist()
        out["gradcam"][key] = torch.relu((z * grad.mean(0)).sum(-1)).cpu().tolist()
    for name, scores in out.items():
        write_scores(f"{args.out}_{name}.json", scores)
    print(f"wrote {len(out['ig'])} chains to {args.out}_{{ig,grad_x_input,gradcam}}.json")


if __name__ == "__main__":
    main()
