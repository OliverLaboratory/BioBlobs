"""Length-aware batch samplers for protein datasets."""

from __future__ import annotations

import random
from typing import Iterable, Sequence

from torch.utils.data import Sampler


class LengthBucketBatchSampler(Sampler[list[int]]):
    """Group similarly sized examples into batches to reduce dense padding waste."""

    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int,
        *,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 0,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        self.lengths = [int(length) for length in lengths]
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self._iteration = 0
        self._sorted_indices = sorted(
            range(len(self.lengths)),
            key=self.lengths.__getitem__,
        )

    def __iter__(self) -> Iterable[list[int]]:
        batches = [
            self._sorted_indices[start : start + self.batch_size]
            for start in range(0, len(self._sorted_indices), self.batch_size)
        ]
        if self.drop_last and batches and len(batches[-1]) < self.batch_size:
            batches = batches[:-1]

        if self.shuffle:
            rng = random.Random(self.seed + self._iteration)
            rng.shuffle(batches)
        self._iteration += 1

        yield from batches

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.lengths) // self.batch_size
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size
