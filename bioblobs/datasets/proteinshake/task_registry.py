"""Registry for ProteinShake task metadata used by BioBlobs."""

from __future__ import annotations

from dataclasses import dataclass


GO_BRANCHES = (
    "molecular_function",
    "biological_process",
    "cellular_component",
)

GO_ALIAS_TO_BRANCH: dict[str, str] = {
    "go_mf": "molecular_function",
    "go_bp": "biological_process",
    "go_cc": "cellular_component",
}

SCOP_LEVELS: dict[str, str] = {
    "scop_fam": "SCOP-FA",
    "scop_sf": "SCOP-SF",
}


@dataclass(frozen=True)
class ProteinShakeTaskSpec:
    dataset_name: str
    problem_type: str
    prepared_root: str
    task_class_name: str
    target_format: str


def _normalize_go_branch(go_branch: str | None) -> str:
    branch = go_branch or "molecular_function"
    if branch not in GO_BRANCHES:
        raise ValueError(
            f"Unsupported GO branch {branch!r}. Expected one of: {', '.join(GO_BRANCHES)}"
        )
    return branch


def get_prepared_root_name(dataset_name: str, go_branch: str | None = None) -> str:
    if dataset_name in GO_ALIAS_TO_BRANCH:
        branch = GO_ALIAS_TO_BRANCH[dataset_name]
        return f"go_{branch}_proteinshake"
    if dataset_name == "go":
        branch = _normalize_go_branch(go_branch)
        return f"go_{branch}_proteinshake"
    if dataset_name == "pfam":
        return "pfam_proteinshake"
    if dataset_name == "ec":
        return "ec_proteinshake"
    if dataset_name in SCOP_LEVELS:
        return f"{dataset_name}_proteinshake"
    raise ValueError(f"Unsupported ProteinShake dataset {dataset_name!r}")


def get_dataset_spec(dataset_name: str, go_branch: str | None = None) -> ProteinShakeTaskSpec:
    if dataset_name in GO_ALIAS_TO_BRANCH:
        return ProteinShakeTaskSpec(
            dataset_name=dataset_name,
            problem_type="multi_label",
            prepared_root=get_prepared_root_name(dataset_name),
            task_class_name="GeneOntologyTask",
            target_format="sparse_multi_label",
        )
    if dataset_name == "go":
        return ProteinShakeTaskSpec(
            dataset_name="go",
            problem_type="multi_label",
            prepared_root=get_prepared_root_name(dataset_name, go_branch),
            task_class_name="GeneOntologyTask",
            target_format="sparse_multi_label",
        )
    if dataset_name in SCOP_LEVELS:
        return ProteinShakeTaskSpec(
            dataset_name=dataset_name,
            problem_type="multi_class",
            prepared_root=get_prepared_root_name(dataset_name),
            task_class_name="StructuralClassTask",
            target_format="label_id",
        )
    if dataset_name == "pfam":
        return ProteinShakeTaskSpec(
            dataset_name="pfam",
            problem_type="multi_class",
            prepared_root=get_prepared_root_name(dataset_name, go_branch),
            task_class_name="ProteinFamilyTask",
            target_format="label_id",
        )
    raise ValueError(f"Unsupported ProteinShake dataset {dataset_name!r}")


def build_task_kwargs(ds_cfg) -> dict[str, object]:
    dataset_name = ds_cfg.dataset_name
    kwargs: dict[str, object] = {
        "split": ds_cfg.split,
        "split_similarity_threshold": ds_cfg.get("split_similarity_threshold", 0.7),
        "root": ds_cfg.data_dir,
    }
    if dataset_name in GO_ALIAS_TO_BRANCH:
        kwargs["branch"] = GO_ALIAS_TO_BRANCH[dataset_name]
    elif dataset_name == "go":
        kwargs["branch"] = _normalize_go_branch(ds_cfg.get("go_branch"))
    elif dataset_name in SCOP_LEVELS:
        kwargs["scop_level"] = SCOP_LEVELS[dataset_name]
    return kwargs
