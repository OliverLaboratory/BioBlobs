"""ProteinShake dataset adapters."""

from .adapter import (
    ensure_prepared_dataset,
    get_prepared_dataset_root,
    load_prepared_structures,
)
from .task_registry import (
    ProteinShakeTaskSpec,
    get_dataset_spec,
    get_prepared_root_name,
)

__all__ = [
    "ProteinShakeTaskSpec",
    "ensure_prepared_dataset",
    "get_dataset_spec",
    "get_prepared_dataset_root",
    "get_prepared_root_name",
    "load_prepared_structures",
]
