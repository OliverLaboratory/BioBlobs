"""Graph datasets over prepared structures and cached residue embeddings, and their loaders."""

from __future__ import annotations

import random
from collections.abc import Iterator, Sequence

import torch
from torch.utils.data import Dataset, Sampler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from bioblobs.data.plm_cache import EmbeddingCache
from bioblobs.data.structure import Structure, to_graph


class ProteinGraphDataset(Dataset):
    """Graph samples built on the fly from structures and an embedding cache.

    Args:
        structures: Structure dicts (see ``bioblobs.data.structure.to_graph``).
        embeddings: Residue embedding cache.
        key_field: Structure field used as the embedding-cache key.
    """

    def __init__(
        self,
        structures: Sequence[Structure],
        embeddings: EmbeddingCache,
        key_field: str = "name",
    ) -> None:
        self.structures = structures
        self.embeddings = embeddings
        self.key_field = key_field
        self.node_counts = [len(s["seq"]) for s in structures]

    def __len__(self) -> int:
        return len(self.structures)

    def __getitem__(self, index: int) -> Data:
        structure = self.structures[index]
        return to_graph(structure, self.embeddings.get(structure[self.key_field]))


class LengthBucketBatchSampler(Sampler[list[int]]):
    """Batches of similar-length proteins, in shuffled batch order, to limit padding.

    Args:
        lengths: Residues per protein.
        batch_size: Proteins per batch.
        shuffle: Shuffle the order of batches every epoch.
        drop_last: Drop a final incomplete batch.
        seed: Base seed; epoch e uses ``seed + e``.
    """

    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 0,
    ) -> None:
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = int(seed)
        self._epoch = 0
        self._order = sorted(range(len(lengths)), key=lambda i: int(lengths[i]))

    def __iter__(self) -> Iterator[list[int]]:
        batches = [
            self._order[i : i + self.batch_size]
            for i in range(0, len(self._order), self.batch_size)
        ]
        if self.drop_last and batches and len(batches[-1]) < self.batch_size:
            batches = batches[:-1]
        if self.shuffle:
            random.Random(self.seed + self._epoch).shuffle(batches)
        self._epoch += 1
        yield from batches

    def __len__(self) -> int:
        full, rest = divmod(len(self._order), self.batch_size)
        return full if self.drop_last or rest == 0 else full + 1


def make_loader(
    dataset: ProteinGraphDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    bucket_by_size: bool = False,
    seed: int = 0,
) -> DataLoader:
    """DataLoader over graph samples, optionally with length bucketing.

    Args:
        dataset: Graph dataset.
        batch_size: Proteins per batch.
        shuffle: Shuffle the training order.
        num_workers: Loader worker processes.
        bucket_by_size: Group proteins of similar length into batches.
        seed: Seed of the bucket order.

    Returns:
        A ``torch_geometric`` DataLoader.
    """
    kwargs = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = 4
    if bucket_by_size:
        sampler = LengthBucketBatchSampler(dataset.node_counts, batch_size, shuffle, seed=seed)
        return DataLoader(dataset, batch_sampler=sampler, **kwargs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
