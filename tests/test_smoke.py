"""CPU smoke tests on synthetic proteins: every model family runs forward and backward."""

from __future__ import annotations

import pytest
import torch
from hydra import compose, initialize
from torch_geometric.data import Batch, Data

from bioblobs.localization.metrics import residue_ranking_metrics
from bioblobs.localization.scores import max_membership, weighted_membership
from bioblobs.objectives import hoyer_sparsity
from bioblobs.training.lightning_module import BioBlobsModule
from bioblobs.training.metrics import fmax, multiclass_metrics

NUM_CLASSES = 5


def synthetic_batch(
    dim: int = 640,
    multi_label: bool = False,
    seed: int = 0,
    sizes: tuple[int, int, int] = (30, 45, 6),
) -> Batch:
    """Three small proteins with random embeddings, C-alpha traces and site labels."""
    g = torch.Generator().manual_seed(seed)
    items = []
    for i, n in enumerate(sizes):
        x = torch.cumsum(torch.randn(n, 3, generator=g) * 2.0, dim=0)
        mask = torch.ones(n, dtype=torch.bool)
        mask[0] = i != 1  # one protein lacks the C-alpha of residue 0
        x[~mask] = float("inf")
        y = (
            (torch.rand(NUM_CLASSES, generator=g) > 0.5).float()
            if multi_label
            else torch.tensor(i % NUM_CLASSES)
        )
        site = torch.zeros(n)
        site[: max(1, n // 5)] = 1.0
        edges = torch.stack([torch.arange(n - 1), torch.arange(1, n)])
        edges = torch.cat([edges, edges.flip(0)], dim=1)
        items.append(
            Data(
                node_features=torch.randn(n, dim, generator=g),
                x=x,
                mask=mask,
                y=y,
                residue_target=site,
                edge_index=edges,
                num_nodes=n,
            )
        )
    return Batch.from_data_list(items)


def config(*overrides: str):
    with initialize(version_base="1.3", config_path="../conf"):
        return compose("train_bioblobs", overrides=list(overrides))


CASES = [
    ("+experiment=proteinshake/ec_sequence_esm2",),
    ("+experiment=proteinshake/scop_fam_sequence_esm2",),
    ("+experiment=proteinshake/go_mf_sequence_esm2",),
    ("+experiment=proteinshake/ec_sequence_saprot",),
    ("+experiment=proteinshake/go_mf_structure_saprot",),
    ("+experiment=venusx/act_esm2_seedalign",),
    ("+experiment=venusx/act_saprot_labelfree",),
    ("+experiment=mcsa/rvq_k8", "codebooks.warmup_epochs=0", "codebooks.num_codes=8"),
    ("partitioners=none", "decoders=mean_pool"),
    ("partitioners=none", "decoders=attention_pool"),
    ("partitioners=none", "decoders=mean_pool", "encoders=gat_esm2"),
    ("partitioners=diffpool",),
    ("partitioners=topk",),
    ("partitioners=sagpool",),
]


@pytest.mark.parametrize("overrides", CASES, ids=lambda o: o[0].split("=")[-1])
def test_forward_backward(overrides: tuple[str, ...]) -> None:
    cfg = config(*overrides)
    torch.manual_seed(0)
    module = BioBlobsModule(cfg, NUM_CLASSES)
    multi_label = cfg.tasks.problem_type == "multi_label"
    batch = synthetic_batch(cfg.encoders.output_dim, multi_label)
    module.train()
    out = module(batch)
    loss, terms = module.loss(batch, out)
    assert out.logits.shape == (3, NUM_CLASSES)
    assert torch.isfinite(loss) and all(torch.isfinite(v) for v in terms.values())
    loss.backward()
    grads = [p.grad for p in module.parameters() if p.requires_grad and p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    if out.partition is not None and out.importance is not None:
        assert torch.allclose(out.importance.sum(-1), torch.ones(3), atol=1e-5)


def test_partition_properties() -> None:
    module = BioBlobsModule(config("+experiment=proteinshake/ec_sequence_esm2"), NUM_CLASSES).eval()
    batch = synthetic_batch()
    with torch.no_grad():
        out = module(batch)
    part = out.partition
    assert part.blob_mask[2].sum() == batch[2].num_nodes and part.blob_mask[0].all()
    assert ((part.memberships >= 0) & (part.memberships <= 1)).all()
    assert (part.memberships[~part.candidate_mask] == 0).all()
    seeds = part.seed_indices[part.blob_mask]
    assert seeds.min() >= 0
    hoyer = hoyer_sparsity(part.memberships, part.candidate_mask, part.blob_mask)
    assert 0 < float(hoyer) <= 1
    scores = max_membership(part.memberships, part.blob_mask)
    assert torch.equal(scores[part.residue_mask].clamp(0, 1), scores[part.residue_mask])
    weighted = weighted_membership(part.memberships, out.importance)
    assert weighted.shape == scores.shape


def test_empty_seed_slots_stay_finite() -> None:
    """Fewer valid residues than K and no C-alpha at residue 0 leave empty seed slots."""
    cfg = config("+experiment=proteinshake/go_mf_structure_saprot")
    torch.manual_seed(0)
    module = BioBlobsModule(cfg, NUM_CLASSES).train()
    batch = synthetic_batch(cfg.encoders.output_dim, multi_label=True, sizes=(40, 28, 6))
    out = module(batch)
    assert int(out.partition.blob_mask[1].sum()) == 27 < cfg.partitioners.num_blobs
    loss, terms = module.loss(batch, out)
    assert torch.isfinite(loss) and all(torch.isfinite(v) for v in terms.values())
    loss.backward()
    grads = [p.grad for p in module.parameters() if p.requires_grad and p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_metrics() -> None:
    scores = torch.tensor([0.9, 0.8, 0.1, 0.2, 0.7])
    sites = torch.tensor([1, 0, 0, 0, 1])
    m = residue_ranking_metrics(scores, sites)
    assert m["f1_at_l"] == 0.5 and 0 < m["auprc"] <= 1 and m["auroc"] == pytest.approx(5 / 6)
    assert residue_ranking_metrics(scores, torch.zeros(5)) is None
    logits = torch.eye(3).numpy()
    assert multiclass_metrics([0, 1, 2], logits, 4)["macro_f1"] == 1.0
    assert multiclass_metrics([0, 1, 2], logits, 4)["macro_f1_fixed_vocab"] == 0.75
    assert fmax(torch.eye(3).numpy(), (torch.eye(3) * 10 - 5).numpy())["fmax"] == 1.0
