#!/usr/bin/env python
"""ESM2 attention baseline: the attention each residue receives in ESM2-T30, averaged over all
30 layers and 20 heads. Uses no labels.

Usage::

    python baselines/mcsa/esm2_attention.py --features data/mcsa/features.pt \\
        --pdb-dir data/mcsa_raw/pdbs --out scores/esm2_attention.json
"""

from __future__ import annotations

import argparse
import json

import torch
from inhouse import (
    DEFAULT_IDS,
    DEFAULT_LABELS,
    chain_sequence,
    load_esm2,
    load_features,
    read_ids,
    write_scores,
)


@torch.no_grad()
def received_attention(tokenizer, model, sequence: str) -> torch.Tensor:
    """Attention received per residue (summed over queries), mean over heads and layers."""
    device = next(model.parameters()).device
    tokens = tokenizer(sequence, return_tensors="pt").to(device)
    attentions = model(**tokens, output_attentions=True).attentions  # layers x [1, H, S, S]
    received = torch.stack([a[0].sum(dim=1) for a in attentions])  # [layers, H, S]
    return received[:, :, 1 : len(sequence) + 1].float().mean(dim=(0, 1))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--features", required=True, help="M-CSA feature file (residue order)")
    p.add_argument("--pdb-dir", required=True)
    p.add_argument("--ids", default=str(DEFAULT_IDS))
    p.add_argument("--labels", default=str(DEFAULT_LABELS))
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    features = load_features(args.features)
    entries = {f"{e['pdb_id']}_{e['chain_used']}": e for e in json.load(open(args.labels))}
    tokenizer, model = load_esm2(args.device, eager_attention=True)
    scores = {}
    for key in read_ids(args.ids):
        parsed = chain_sequence(args.pdb_dir, entries[key])
        if parsed is None or parsed[1] != [int(r) for r in features[key]["resnums"]]:
            print(f"skip {key}: chain does not match the feature file")
            continue
        scores[key] = received_attention(tokenizer, model, parsed[0]).cpu().tolist()
    write_scores(args.out, scores)
    print(f"wrote {len(scores)} chains to {args.out}")


if __name__ == "__main__":
    main()
