from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import ttest_ind

from utils.data_loader import Dataset
from utils.mask_loader import load_topology_mask_values


def _matching_feature_index(
    dataset: Dataset,
    network: str,
    dNCR_values: np.ndarray,
    non_dNCR_values: np.ndarray,
) -> int:
    matches = []
    values = dataset.topology[network]
    for feature_index in range(values.shape[1]):
        positive = values[dataset.labels == 1, feature_index]
        negative = values[dataset.labels == 0, feature_index]
        positive = positive[np.isfinite(positive)]
        negative = negative[np.isfinite(negative)]
        if (
            len(positive) == len(dNCR_values)
            and len(negative) == len(non_dNCR_values)
            and np.allclose(
                np.sort(positive),
                np.sort(dNCR_values),
                rtol=0.0,
                atol=1e-12,
            )
            and np.allclose(
                np.sort(negative),
                np.sort(non_dNCR_values),
                rtol=0.0,
                atol=1e-12,
            )
        ):
            matches.append(feature_index)
    if len(matches) != 1:
        raise ValueError(
            f"{network}: mask workbook values matched "
            f"{len(matches)} topology features; expected exactly one"
        )
    return matches[0]


def select_topology_priors(
    dataset: Dataset,
    config: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Discover topology priors in mask.xlsx and recompute t-tests."""
    settings = config["topology_prior_selection"]
    alpha = float(settings["alpha"])
    if not 0.0 < alpha < 1.0:
        raise ValueError("topology prior alpha must be between 0 and 1")

    workbook_path = (
        Path(config["data_root"]) / settings.get("workbook", "mask.xlsx")
    )
    selected: dict[str, dict[str, Any]] = {}
    for mask in load_topology_mask_values(workbook_path):
        positive = np.asarray(mask.dNCR_values, dtype=float)
        negative = np.asarray(mask.non_dNCR_values, dtype=float)
        feature_index = _matching_feature_index(
            dataset,
            mask.network,
            positive,
            negative,
        )
        statistic, p_value = ttest_ind(
            positive,
            negative,
            equal_var=True,
        )
        if not np.isfinite(p_value) or p_value >= alpha:
            raise ValueError(
                f"{mask.network}/{mask.sheet_name}: topology mask is not "
                f"significant at alpha={alpha} (P={p_value:.6g})"
            )
        selected[mask.network] = {
            "source_sheet": mask.sheet_name,
            "feature_name": dataset.topology_feature_names[
                mask.network
            ][feature_index],
            "feature_index": int(feature_index),
            "t_statistic": float(statistic),
            "p_value": float(p_value),
            "alpha": alpha,
            "dNCR_n": int(len(positive)),
            "non_dNCR_n": int(len(negative)),
        }
    return selected
