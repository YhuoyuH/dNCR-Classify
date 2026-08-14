from __future__ import annotations

import numpy as np
from sklearn.feature_selection import f_classif


def impute_topology(
    train_values: np.ndarray, other_values: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Median-impute topology using training rows only."""
    with np.errstate(all="ignore"):
        medians = np.nanmedian(train_values, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    train = np.where(np.isfinite(train_values), train_values, medians)
    other = np.where(np.isfinite(other_values), other_values, medians)
    return train, other, medians


def ranked_features(values: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Return stable descending ANOVA-F feature ranks."""
    with np.errstate(all="ignore"):
        scores, _ = f_classif(values, labels)
    scores = np.nan_to_num(
        scores, nan=-np.inf, posinf=np.inf, neginf=-np.inf
    )
    return np.argsort(scores, kind="stable")[::-1]


def edge_indices(
    node_count: int, edges: list[list[int]] | list[tuple[int, int]]
) -> np.ndarray:
    row, column = np.triu_indices(node_count, k=1)
    lookup = {
        (int(node_i), int(node_j)): index
        for index, (node_i, node_j) in enumerate(
            zip(row, column)
        )
    }
    indices = []
    for raw_i, raw_j in edges:
        node_i, node_j = sorted((int(raw_i), int(raw_j)))
        try:
            indices.append(lookup[(node_i, node_j)])
        except KeyError as error:
            raise ValueError(
                f"Invalid edge ({raw_i}, {raw_j}) for {node_count} nodes"
            ) from error
    if len(indices) != len(set(indices)):
        raise ValueError("Mask contains duplicate FC edges")
    return np.asarray(indices, dtype=int)


def nbs_weights(statistics: np.ndarray, power: float) -> np.ndarray:
    weights = np.abs(np.asarray(statistics, dtype=float)) ** float(power)
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0:
        raise ValueError("NBS weights cannot be normalized")
    return weights / total
