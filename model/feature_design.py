from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler

from utils.features import (
    edge_indices,
    impute_topology,
    nbs_weights,
    ranked_features,
)
from utils.mask_loader import (
    discover_pre_surgery_mask_sheets,
    load_fc_mask,
)


@dataclass(frozen=True)
class FeatureSpec:
    k: int
    linear_svc_c: float
    mask_kind: str = "none"


def _mask_workbook(config: dict[str, Any]) -> Path:
    workbook = config.get("topology_prior_selection", {}).get(
        "workbook", "mask.xlsx"
    )
    return Path(config["data_root"]) / workbook


def _infer_mask_kind(
    network: str,
    feature_mode: str,
    config: dict[str, Any],
) -> str:
    applicable = []
    for mask in discover_pre_surgery_mask_sheets(
        _mask_workbook(config)
    ):
        if mask.network != network:
            continue
        if (
            mask.feature_family == "topology"
            and "topology" in feature_mode
        ):
            applicable.append("topology_mask")
        elif mask.feature_family == "FC" and "FC" in feature_mode:
            applicable.append("smn_nbs_score")

    if len(applicable) != 1:
        raise ValueError(
            f"Cannot infer one mask kind for {network}/{feature_mode} "
            f"from mask.xlsx; found {applicable}"
        )
    return applicable[0]


def specification(
    variant: str,
    network: str,
    feature_mode: str,
    config: dict[str, Any],
) -> FeatureSpec:
    configured = config["reference_specs"]
    original_values = configured["Original"][network][feature_mode]
    original = FeatureSpec(
        k=int(original_values["K"]),
        linear_svc_c=float(original_values["LinearSVC_C"]),
    )
    if variant == "Original":
        return original
    if variant != "Mask":
        raise ValueError(f"Unknown feature variant: {variant}")

    mask_values = (
        configured.get("Mask", {}).get(network, {}).get(feature_mode)
    )
    if mask_values is None:
        return replace(original, mask_kind="no_mask_fallback")
    return FeatureSpec(
        k=int(mask_values["K"]),
        linear_svc_c=float(mask_values["LinearSVC_C"]),
        mask_kind=_infer_mask_kind(network, feature_mode, config),
    )


def _base_design(
    network: str,
    feature_mode: str,
    dataset: Any,
    train: np.ndarray,
    test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if feature_mode == "FC":
        return dataset.fc[network][train], dataset.fc[network][test]

    topology_train, topology_test, _ = impute_topology(
        dataset.topology[network][train],
        dataset.topology[network][test],
    )
    if feature_mode == "topology":
        return topology_train, topology_test
    if feature_mode == "FC+topology":
        return (
            np.column_stack((dataset.fc[network][train], topology_train)),
            np.column_stack((dataset.fc[network][test], topology_test)),
        )
    raise ValueError(f"Unknown feature mode: {feature_mode}")


def build_design(
    *,
    network: str,
    feature_mode: str,
    feature_spec: FeatureSpec,
    dataset: Any,
    train: np.ndarray,
    test: np.ndarray,
    labels: np.ndarray,
    config: dict[str, Any],
    topology_priors: dict[str, dict[str, Any] | None],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    x_train, x_test = _base_design(
        network, feature_mode, dataset, train, test
    )
    selected = ranked_features(x_train, labels[train])[
        : min(feature_spec.k, x_train.shape[1])
    ].copy()

    if feature_spec.mask_kind == "topology_mask":
        prior = topology_priors.get(network)
        if prior is not None:
            mask_index = int(prior["feature_index"])
            if feature_mode == "FC+topology":
                mask_index += dataset.fc[network].shape[1]
            if mask_index not in selected:
                selected[-1] = mask_index
        train_design = x_train[:, selected]
        test_design = x_test[:, selected]
    elif feature_spec.mask_kind == "smn_nbs_score":
        node_count = dataset.node_counts[network]
        mask = load_fc_mask(
            _mask_workbook(config),
            network,
            node_count,
        )
        mask_edges = edge_indices(node_count, mask.edges)
        edge_scaler = StandardScaler()
        train_edges = edge_scaler.fit_transform(
            dataset.fc[network][train][:, mask_edges]
        )
        test_edges = edge_scaler.transform(
            dataset.fc[network][test][:, mask_edges]
        )
        weights = nbs_weights(
            np.asarray(mask.statistics, dtype=float),
            4.0,
        )
        train_design = np.column_stack(
            (x_train[:, selected], train_edges @ weights)
        )
        test_design = np.column_stack(
            (x_test[:, selected], test_edges @ weights)
        )
    else:
        train_design = x_train[:, selected]
        test_design = x_test[:, selected]

    return train_design, test_design, {
        "feature_count": int(train_design.shape[1]),
        "mask_kind": feature_spec.mask_kind,
    }
