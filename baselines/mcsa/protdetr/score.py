#!/usr/bin/env python
"""ProtDETR attention-based residue scores (encoder + decoder read-outs).

ProtDETR (Yang et al., 2025) frames EC prediction as DETR-style set
prediction over residues: an ESM-1b-embedded sequence is passed through a
transformer encoder/decoder with learned "functional" queries. This
wrapper reproduces the two attention read-outs used as M-CSA localization
baselines in the BioBlobs paper:

    protdetr:     encoder self-attention received per residue, averaged
                  over encoder layers and query positions, then
                  L2-normalized.
    protdetr_dec: decoder cross-attention for the functional query with
                  the largest non-null EC logit, averaged over decoder
                  layers, then L2-normalized.

Both read-outs use PyTorch's default head-averaged attention weights.
Sequences are truncated to 1,022 residues (the ESM-1b limit). Unlike the
other two baselines, ProtDETR's class/query selection does not use the
annotated EC label at all -- the decoder query is chosen purely from the
model's own highest non-null-class logit, so this script never reads
``ec`` from the input records.

Typical usage example:

    python score.py \\
        --inputs mcsa_inputs.jsonl \\
        --upstream-dir /path/to/ProtDETR \\
        --out scores_protdetr.json

``--upstream-dir`` is the cloned ProtDETR repository
(github.com/yangzhao1230/ProtDETR @ aa49809) with
``saved_models/ProtDETR_split100.pt`` and ``data/multi_func/split100.csv``
in place (see ../README.md for where to get them). Requires torch and
fair-esm (see ../README.md for the pinned versions); the first run
downloads the ESM-1b weights via torch hub unless they are already
cached.

Output: two JSON files -- ``--out`` (encoder score) and a sibling file
with ``_dec`` appended to its stem (decoder score) -- each a JSON object
``{chain_id: [score, ...]}`` with one float per residue of the (possibly
truncated) ``seq``.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import load_records, write_scores  # noqa: E402

LOGGER = logging.getLogger("protdetr_score")

MAXLEN = 1022
ESM_LAYER = 32


def l2norm(v: np.ndarray) -> np.ndarray:
    """L2-normalizes a 1-D array; returns it unchanged if its norm is zero."""
    n = float(np.sqrt((v ** 2).sum()))
    return v / n if n else v


def _load_upstream(upstream_dir: str) -> Tuple:
    """Imports the ProtDETR modules needed for inference from upstream_dir.

    Args:
        upstream_dir: Path to the cloned ProtDETR repository.

    Returns:
        A tuple ``(esm, build_model, get_esm_emb, get_ec_id_dict,
        get_id_seq_dict, ProtSeqDETRDataset, get_args_parser)``.
    """
    sys.path.insert(0, upstream_dir)
    cwd = os.getcwd()
    os.chdir(upstream_dir)  # the repo's dataset/label-map code uses relative ./data paths
    try:
        import esm  # fair-esm
        from datasets.protdataset import ProtSeqDETRDataset
        from engine import get_esm_emb
        from infer_important_site import get_args_parser
        from models import build_model
        from util.clean import get_ec_id_dict, get_id_seq_dict
    finally:
        os.chdir(cwd)
    return (esm, build_model, get_esm_emb, get_ec_id_dict, get_id_seq_dict,
            ProtSeqDETRDataset, get_args_parser)


@torch.no_grad()
def score_records(
    records: List[dict],
    upstream_dir: str,
    device: str = "cpu",
) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    """Computes the ProtDETR encoder and decoder attention scores.

    Args:
        records: Input records with "id" and "seq" fields (see
            ../common.py / ../README.md).
        upstream_dir: Path to the cloned ProtDETR repository, with
            ``saved_models/ProtDETR_split100.pt`` and
            ``data/multi_func/split100.csv`` in place.
        device: torch device string, e.g. "cpu" or "cuda:0".

    Returns:
        A tuple ``(encoder_scores, decoder_scores)``, each mapping chain
        id to a list of per-residue scores.
    """
    (esm, build_model, get_esm_emb, get_ec_id_dict, get_id_seq_dict,
     ProtSeqDETRDataset, get_args_parser) = _load_upstream(upstream_dir)

    dev = torch.device(device)
    args = get_args_parser().parse_args([])
    args.device = device
    args.esm_layer = ESM_LAYER

    split100 = os.path.join(upstream_dir, "data", "multi_func", "split100.csv")
    id_ec, ec_id = get_ec_id_dict(split100)
    id_seq = get_id_seq_dict(split100)
    ds = ProtSeqDETRDataset(id_ec, ec_id, id_seq, max_labes=args.num_queries, esm_layer=args.esm_layer)
    args.num_classes = len(ds.ec_to_label)

    esm_model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    esm_model.eval().to(dev)
    model, _ = build_model(args, ds.ec_weight)
    ckpt = os.path.join(upstream_dir, "saved_models", "ProtDETR_split100.pt")
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval().to(dev)
    LOGGER.info("loaded ProtDETR (enc_layers=%d dec_layers=%d num_queries=%d num_classes=%d)",
                args.enc_layers, args.dec_layers, args.num_queries, args.num_classes)

    enc_out: Dict[str, List[float]] = {}
    dec_out: Dict[str, List[float]] = {}
    for rec in records:
        cid = rec["id"]
        seq = rec["seq"][:MAXLEN]
        length = len(seq)
        if length < 10:
            LOGGER.warning("%s: sequence too short, skipping", cid)
            continue

        enc_w: List[List[torch.Tensor]] = [[] for _ in range(args.enc_layers)]
        dec_w: List[List[torch.Tensor]] = [[] for _ in range(args.dec_layers)]
        hooks = []
        for i in range(args.enc_layers):
            hooks.append(model.transformer.encoder.layers[i].self_attn.register_forward_hook(
                lambda mod, inp, out, idx=i: enc_w[idx].append(out[1].squeeze(0))))
        for i in range(args.dec_layers):
            hooks.append(model.transformer.decoder.layers[i].multihead_attn.register_forward_hook(
                lambda mod, inp, out, idx=i: dec_w[idx].append(out[1].squeeze(0))))
        try:
            embs, masks = get_esm_emb(esm_model, alphabet, args.esm_layer, [seq], dev)
            out = model(embs, masks.to(dev))
        finally:
            for h in hooks:
                h.remove()

        logits = out["pred_logits"].squeeze(0).cpu().numpy()                  # (Q, C)
        enc = torch.stack([torch.cat(w, 0) for w in enc_w], 0).cpu().numpy()  # (enc_layers, L, L)
        dec = torch.stack([torch.cat(w, 0) for w in dec_w], 0).cpu().numpy()  # (dec_layers, Q, L)

        # Encoder: self-attention received per residue, averaged over layers then queries.
        enc_score = l2norm(enc.mean(0).mean(0))[:length]
        # Decoder: cross-attention of the query with the largest non-null-class logit,
        # averaged over layers (drop the last "no EC" logit column).
        q = int(np.argmax(logits[:, :-1].max(1)))
        dec_score = l2norm(dec[:, q, :].mean(0))[:length]

        if enc_score.std() > 0:
            enc_out[cid] = [float(v) for v in enc_score]
        else:
            LOGGER.warning("%s: constant encoder score, skipping", cid)
        if dec_score.std() > 0:
            dec_out[cid] = [float(v) for v in dec_score]
        else:
            LOGGER.warning("%s: constant decoder score, skipping", cid)

    return enc_out, dec_out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inputs", required=True, help="Common-format input file (see ../README.md).")
    parser.add_argument("--upstream-dir", required=True,
                         help="Cloned ProtDETR repo with saved_models/ and data/multi_func/ in place.")
    parser.add_argument("--out", required=True,
                         help="Output JSON path for the encoder score; the decoder score is "
                              "written alongside it with '_dec' appended to the stem.")
    parser.add_argument("--device", default="cpu", help="torch device, e.g. 'cpu' or 'cuda:0'.")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    records = load_records(args.inputs)
    LOGGER.info("loaded %d records from %s", len(records), args.inputs)
    enc_out, dec_out = score_records(records, args.upstream_dir, device=args.device)
    LOGGER.info("scored %d/%d (encoder), %d/%d (decoder)",
                len(enc_out), len(records), len(dec_out), len(records))

    out_path = Path(args.out)
    write_scores(str(out_path), enc_out)
    dec_path = out_path.with_name(out_path.stem + "_dec" + out_path.suffix)
    write_scores(str(dec_path), dec_out)
    LOGGER.info("wrote %s and %s", out_path, dec_path)


if __name__ == "__main__":
    main()
