"""Generic dataset wrapper for protein-level tasks."""

from __future__ import annotations

import numpy as np
import torch.utils.data as data


class TaskDataset(data.Dataset):
    """Map-style dataset that delegates graph creation to the configured featurizer."""

    def __init__(
        self,
        data_list,
        *,
        num_classes: int | None = None,
        featurizer=None,
    ) -> None:
        super().__init__()

        if featurizer is None:
            raise ValueError("featurizer cannot be None")

        self.data_list = data_list
        self.node_counts = [len(entry["seq"]) for entry in data_list]
        self.num_classes = num_classes or self._infer_num_classes(data_list)
        self.featurizer = featurizer

    def _infer_num_classes(self, data_list) -> int:
        labels = [item.get("label") for item in data_list if "label" in item]
        if not labels:
            return 1

        first_label = labels[0]
        if isinstance(first_label, np.ndarray):
            if first_label.ndim == 0:
                return int(first_label.item()) + 1
            return int(first_label.shape[0])
        if isinstance(first_label, (list, tuple)):
            return len(first_label)
        return max(int(label) for label in labels) + 1

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, index):
        return self.featurizer.featurize(self.data_list[index])
