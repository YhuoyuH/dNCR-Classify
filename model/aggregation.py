from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np

from model.constants import (
    FEATURE_MODES,
    METRIC_NAMES,
    MODEL_ORDER,
    NETWORKS,
    VARIANTS,
)


TaskKey = tuple[str, str, str, str]


def group_by_task(
    rows: list[dict[str, Any]],
) -> dict[TaskKey, list[dict[str, Any]]]:
    grouped: dict[TaskKey, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["model"]),
            str(row["variant"]),
            str(row["network"]),
            str(row["feature_mode"]),
        )
        grouped[key].append(row)
    return grouped


def task_summaries(
    metric_rows: list[dict[str, Any]],
    n_splits: int,
) -> list[dict[str, Any]]:
    grouped = group_by_task(metric_rows)
    summaries = []
    for model in MODEL_ORDER:
        for variant in VARIANTS:
            for network in NETWORKS:
                for feature_mode in FEATURE_MODES:
                    rows = grouped[
                        (model, variant, network, feature_mode)
                    ]
                    if len(rows) != n_splits:
                        raise RuntimeError(
                            f"{model}/{variant}/{network}/{feature_mode}: "
                            f"expected {n_splits} rows, found {len(rows)}"
                        )
                    metrics = {}
                    for metric in METRIC_NAMES:
                        values = np.asarray(
                            [float(row[metric]) for row in rows],
                            dtype=float,
                        )
                        metrics[metric] = {
                            "mean": float(np.mean(values)),
                            "std": float(np.std(values, ddof=1)),
                            "median": float(np.median(values)),
                            "min": float(np.min(values)),
                            "max": float(np.max(values)),
                        }
                    result_sources = {
                        str(row["result_source"]) for row in rows
                    }
                    if len(result_sources) != 1:
                        raise RuntimeError(
                            f"{model}/{variant}/{network}/{feature_mode}: "
                            "inconsistent result sources"
                        )
                    summaries.append(
                        {
                            "model": model,
                            "variant": variant,
                            "network": network,
                            "feature_mode": feature_mode,
                            "result_source": result_sources.pop(),
                            "metrics": metrics,
                        }
                    )
    return summaries


def feature_summaries(
    metric_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[
        tuple[str, str, str, str], list[dict[str, Any]]
    ] = defaultdict(list)
    for row in metric_rows:
        key = (
            str(row["model"]),
            str(row["variant"]),
            str(row["network"]),
            str(row["feature_mode"]),
        )
        grouped[key].append(row)

    summaries = []
    for model in MODEL_ORDER:
        for variant in VARIANTS:
            for network in NETWORKS:
                for feature_mode in FEATURE_MODES:
                    rows = grouped[
                        (model, variant, network, feature_mode)
                    ]
                    summaries.append(
                        {
                            "model": model,
                            "variant": variant,
                            "network": network,
                            "feature_mode": feature_mode,
                            **feature_selection_summary(rows),
                        }
                    )
    return summaries


def feature_selection_summary(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if not rows:
        raise ValueError("No feature-selection rows")
    first = rows[0]
    constant_fields = (
        "reference_k",
        "selection_method",
        "mask_kind",
        "result_source",
    )
    if any(
        row[field] != first[field]
        for row in rows
        for field in constant_fields
    ):
        raise RuntimeError("Inconsistent feature-selection metadata")

    k_counts = Counter(int(row["k"]) for row in rows)
    feature_count_counts = Counter(
        int(row["feature_count"]) for row in rows
    )
    summary = {
        "selection_method": str(first["selection_method"]),
        "reference_k": int(first["reference_k"]),
        "selected_by_inner_cv_count": int(
            sum(bool(row["selected_by_inner_cv"]) for row in rows)
        ),
        "split_count": len(rows),
        "selected_k_counts": [
            {"k": k, "count": count}
            for k, count in sorted(
                k_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ],
        "feature_count_counts": [
            {"feature_count": count_value, "count": count}
            for count_value, count in sorted(
                feature_count_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ],
        "mask_kind": str(first["mask_kind"]),
        "result_source": str(first["result_source"]),
    }
    if first["model"] == "LinearSVC":
        summary["linear_svc_c"] = float(first["linear_svc_c"])
    return summary


def model_summaries(
    tasks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    summaries = []
    for model in MODEL_ORDER:
        auc = np.asarray(
            [
                row["metrics"]["roc_auc"]["mean"]
                for row in tasks
                if row["model"] == model
                and row["result_source"] == "trained"
            ],
            dtype=float,
        )
        summaries.append(
            {
                "model": model,
                "task_count": int(len(auc)),
                "roc_auc_mean": float(np.mean(auc)),
                "roc_auc_min": float(np.min(auc)),
                "roc_auc_max": float(np.max(auc)),
            }
        )
    return summaries
