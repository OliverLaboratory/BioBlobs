#!/usr/bin/env python
"""BatchTopK sparse autoencoder (SAE) baseline on final-layer ESM2-T30 embeddings.

``train`` fits the SAE without labels on 4,000 SwissProt EC training proteins that share no
30%-identity cluster with an M-CSA chain (``clusters_30.tsv`` from
``preprocessing/mcsa/build_homology_excluded_train.py``). ``score`` encodes each M-CSA chain as one
batch, keeps the top ``k * L`` activations of the whole ``[L, latents]`` matrix, and writes two
residue scores: the summed retained activation (``<out>_magnitude.json``) and the number of
retained latents (``<out>_count.json``).

Usage::

    python baselines/mcsa/sae.py train --corpus data/swissprot_ec/train_split_train.csv \\
        --clusters preprocessing_work/mcsa_homology_filter/clusters_30.tsv --out sae.pt
    python baselines/mcsa/sae.py score --sae sae.pt --features data/mcsa/features.pt \\
        --pdb-dir data/mcsa_raw/pdbs --out scores/sae
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from inhouse import (
    DEFAULT_IDS,
    DEFAULT_LABELS,
    chain_sequence,
    esm2_layer,
    load_esm2,
    load_features,
    read_ids,
    write_scores,
)

DIM = 640


class BatchTopKSAE(nn.Module):
    """ReLU encoder, BatchTopK sparsity over the whole batch, unit-norm linear decoder."""

    def __init__(self, d: int = DIM, m: int = 5120, k: int = 32) -> None:
        super().__init__()
        self.k = k
        self.b_dec = nn.Parameter(torch.zeros(d))
        self.W_enc = nn.Parameter(torch.randn(d, m) * (1.0 / d**0.5))
        self.b_enc = nn.Parameter(torch.zeros(m))
        self.W_dec = nn.Parameter(self.W_enc.detach().t().clone())
        self.normalize_dec()

    @torch.no_grad()
    def normalize_dec(self) -> None:
        self.W_dec.data = F.normalize(self.W_dec.data, dim=1)

    def preacts(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu((x - self.b_dec) @ self.W_enc + self.b_enc)

    def batchtopk(self, pre: torch.Tensor) -> torch.Tensor:
        """Keep the top ``k * B`` activations of the whole ``[B, M]`` matrix."""
        budget = max(1, self.k * pre.shape[0])
        if budget >= pre.numel():
            return pre
        threshold = torch.topk(pre.reshape(-1), budget, sorted=False).values.min()
        return torch.where(pre >= threshold, pre, torch.zeros_like(pre))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pre = self.preacts(x)
        acts = self.batchtopk(pre)
        return acts @ self.W_dec + self.b_dec, acts, pre


def corpus_sequences(corpus: str, clusters: str, n: int) -> list[str]:
    """Every ``len // n``-th corpus sequence outside the clusters that hold an M-CSA chain."""
    members, has_mcsa = defaultdict(list), set()
    with open(clusters) as handle:
        next(handle)
        for line in handle:
            rep, member = line.rstrip("\n").split("\t")
            members[rep].append(member)
            if member.startswith("MCSA__"):
                has_mcsa.add(rep)
    excluded = {m for rep in has_mcsa for m in members[rep] if not m.startswith("MCSA__")}
    with open(corpus) as handle:
        seqs = [row["seq"] for row in csv.DictReader(handle) if row["id"] not in excluded]
    return seqs[:: max(1, len(seqs) // n)][:n]


def train(args: argparse.Namespace) -> None:
    tokenizer, esm = load_esm2(args.device)
    seqs = corpus_sequences(args.corpus, args.clusters, args.n_proteins)
    tokens, total = [], 0
    for i, seq in enumerate(seqs):
        if 10 <= len(seq) <= 800:
            tokens.append(esm2_layer(tokenizer, esm, seq, args.layer).cpu())
            total += tokens[-1].shape[0]
        if (i + 1) % 200 == 0 and total > args.max_tokens:
            break
    x_all = torch.cat(tokens).to(args.device)
    print(f"{len(seqs)} corpus proteins, {x_all.shape[0]} training residues")

    m = args.expansion * DIM
    sae = BatchTopKSAE(DIM, m, args.k).to(args.device)
    sae.b_dec.data = x_all.mean(0)
    opt = torch.optim.Adam(sae.parameters(), lr=args.lr)
    last_fired = torch.zeros(m, device=args.device)
    gen = torch.Generator(device=args.device).manual_seed(0)
    for step in range(args.steps):
        x = x_all[
            torch.randint(0, x_all.shape[0], (args.batch,), device=args.device, generator=gen)
        ]
        recon, acts, pre = sae(x)
        loss = F.mse_loss(recon, x)
        # Auxiliary loss: latents silent for 200 steps reconstruct the residual.
        last_fired = torch.where((acts > 0).any(0), torch.zeros_like(last_fired), last_fired + 1)
        dead = last_fired > 200
        if dead.any() and step > 500:
            pre_dead = pre.clone()
            pre_dead[:, ~dead] = 0
            top = torch.topk(pre_dead, min(int(dead.sum()), 512), dim=1)
            aux_acts = torch.zeros_like(pre_dead).scatter(1, top.indices, top.values)
            loss = loss + 0.03 * F.mse_loss(aux_acts @ sae.W_dec, x - recon.detach())
        opt.zero_grad()
        loss.backward()
        opt.step()
        sae.normalize_dec()
        if step % 2000 == 0 or step == args.steps - 1:
            unexplained = ((recon - x) ** 2).sum() / ((x - x.mean(0)) ** 2).sum()
            print(f"step {step:>6}  1-EV {unexplained.item():.3f}  dead {int(dead.sum())}/{m}")
    state = {"state": sae.state_dict(), "d": DIM, "m": m, "k": args.k, "layer": args.layer}
    torch.save(state, args.out)
    print(f"wrote {args.out}")


@torch.no_grad()
def score(args: argparse.Namespace) -> None:
    ckpt = torch.load(args.sae, map_location=args.device, weights_only=True)
    sae = BatchTopKSAE(ckpt["d"], ckpt["m"], ckpt["k"]).to(args.device).eval()
    sae.load_state_dict(ckpt["state"])
    tokenizer, esm = load_esm2(args.device)
    features = load_features(args.features)
    entries = {f"{e['pdb_id']}_{e['chain_used']}": e for e in json.load(open(args.labels))}
    magnitude, count = {}, {}
    for key in read_ids(args.ids):
        parsed = chain_sequence(args.pdb_dir, entries[key])
        if parsed is None or parsed[1] != [int(r) for r in features[key]["resnums"]]:
            print(f"skip {key}: chain does not match the feature file")
            continue
        acts = sae.batchtopk(sae.preacts(esm2_layer(tokenizer, esm, parsed[0], ckpt["layer"])))
        magnitude[key] = acts.sum(1).cpu().tolist()
        count[key] = (acts > 0).sum(1).float().cpu().tolist()
    write_scores(f"{args.out}_magnitude.json", magnitude)
    write_scores(f"{args.out}_count.json", count)
    print(f"wrote {len(magnitude)} chains to {args.out}_{{magnitude,count}}.json")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="mode", required=True)
    t = sub.add_parser("train")
    t.add_argument("--corpus", required=True, help="SwissProt EC train CSV (id, seq)")
    t.add_argument("--clusters", required=True, help="clusters_30.tsv (rep, member)")
    t.add_argument("--out", required=True)
    t.add_argument("--layer", type=int, default=30, help="ESM2 layer (30 = final)")
    t.add_argument("--expansion", type=int, default=8, help="latents = expansion * 640")
    t.add_argument("--k", type=int, default=32)
    t.add_argument("--steps", type=int, default=20000)
    t.add_argument("--batch", type=int, default=4096)
    t.add_argument("--lr", type=float, default=4e-4)
    t.add_argument("--n-proteins", type=int, default=4000)
    t.add_argument("--max-tokens", type=int, default=1_500_000)
    s = sub.add_parser("score")
    s.add_argument("--sae", required=True)
    s.add_argument("--features", required=True, help="M-CSA feature file (residue order)")
    s.add_argument("--pdb-dir", required=True)
    s.add_argument("--ids", default=str(DEFAULT_IDS))
    s.add_argument("--labels", default=str(DEFAULT_LABELS))
    s.add_argument("--out", required=True, help="output prefix")
    for sp in (t, s):
        sp.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    (train if args.mode == "train" else score)(args)


if __name__ == "__main__":
    main()
