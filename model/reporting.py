from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import confusion_matrix

from model.aggregation import (
    feature_selection_summary,
    feature_summaries,
    group_by_task,
    model_summaries,
    task_summaries,
)
from model.constants import (
    FEATURE_MODES,
    METRIC_NAMES,
    MODE_DISPLAY,
    MODE_FOLDERS,
    MODEL_ORDER,
    NETWORKS,
    VARIANTS,
)
from utils.io import write_csv, write_json
from utils.plotting import (
    plot_combined_roc,
    plot_confusion_matrix,
    plot_pr_curve,
    plot_roc_curve,
)


def render_comparison_markdown(
    tasks: list[dict[str, Any]],
    models: list[dict[str, Any]],
) -> str:
    lookup = {
        (
            row["model"],
            row["variant"],
            row["network"],
            row["feature_mode"],
        ): row
        for row in tasks
    }
    lines = [
        "# dNCR模型ROC-AUC对比",
        "",
        "## 模型总体结果",
        "",
        "| 模型 | 任务平均AUC | 最低AUC | 最高AUC |",
        "|---|---:|---:|---:|",
    ]
    for row in models:
        lines.append(
            f"| {row['model']} | {row['roc_auc_mean']:.3f} | "
            f"{row['roc_auc_min']:.3f} | {row['roc_auc_max']:.3f} |"
        )

    for variant in VARIANTS:
        lines.extend(
            [
                "",
                f"## {variant}",
                "",
                "| 网络 | 特征 | LinearSVC | LDA | RF | DT | XGBoost |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for network in NETWORKS:
            for feature_mode in FEATURE_MODES:
                formatted = []
                for model in MODEL_ORDER:
                    metric = lookup[
                        (model, variant, network, feature_mode)
                    ]["metrics"]["roc_auc"]
                    formatted.append(
                        f"{metric['mean']:.3f} ± {metric['std']:.3f}"
                    )
                lines.append(
                    f"| {network} | {MODE_DISPLAY[feature_mode]} | "
                    + " | ".join(formatted)
                    + " |"
                )
    return "\n".join(lines) + "\n"


def _save_task_artifacts(
    *,
    results_root: Path,
    tasks: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    n_splits: int,
) -> None:
    task_lookup = {
        (
            row["model"],
            row["variant"],
            row["network"],
            row["feature_mode"],
        ): row
        for row in tasks
    }
    metric_lookup = group_by_task(metric_rows)
    prediction_lookup = group_by_task(prediction_rows)

    for model in MODEL_ORDER:
        for variant in VARIANTS:
            for network in NETWORKS:
                combined_labels: np.ndarray | None = None
                combined_scores: dict[str, np.ndarray] = {}
                for feature_mode in FEATURE_MODES:
                    key = (model, variant, network, feature_mode)
                    predictions = sorted(
                        prediction_lookup[key],
                        key=lambda row: (
                            int(row["split"]),
                            int(row["subject_id"]),
                        ),
                    )
                    labels = np.asarray(
                        [int(row["label"]) for row in predictions],
                        dtype=int,
                    )
                    scores = np.asarray(
                        [float(row["score"]) for row in predictions],
                        dtype=float,
                    )
                    per_split = metric_lookup[key]
                    threshold = float(per_split[0]["threshold"])
                    if combined_labels is None:
                        combined_labels = labels
                    elif not np.array_equal(combined_labels, labels):
                        raise RuntimeError(
                            f"Prediction order mismatch for {key}"
                        )
                    combined_scores[MODE_DISPLAY[feature_mode]] = scores

                    mode_root = (
                        results_root
                        / model
                        / variant
                        / network
                        / MODE_FOLDERS[feature_mode]
                    )
                    title = (
                        f"{model} | {variant} | {network} | "
                        f"{MODE_DISPLAY[feature_mode]}"
                    )
                    plot_roc_curve(
                        labels,
                        scores,
                        mode_root / "ROC曲线.png",
                        title,
                    )
                    plot_pr_curve(
                        labels,
                        scores,
                        mode_root / "PR曲线.png",
                        title,
                    )
                    plot_confusion_matrix(
                        labels,
                        scores,
                        threshold,
                        mode_root / "混淆矩阵.png",
                        title,
                    )

                    first = per_split[0]
                    task = task_lookup[key]
                    parameter_counts = Counter(
                        tuple(sorted(row["model_parameters"].items()))
                        for row in per_split
                    )
                    write_json(
                        mode_root / "summary.json",
                        {
                            "schema_version": 3,
                            "task": {
                                "model": model,
                                "variant": variant,
                                "network": network,
                                "feature_mode": feature_mode,
                            },
                            "n_splits": n_splits,
                            "feature_selection": (
                                feature_selection_summary(per_split)
                            ),
                            "threshold": threshold,
                            "model_parameter_counts": [
                                {
                                    "parameters": dict(parameters),
                                    "count": count,
                                }
                                for parameters, count in sorted(
                                    parameter_counts.items(),
                                    key=lambda item: -item[1],
                                )
                            ],
                            "metrics": task["metrics"],
                            "pooled_confusion_matrix": {
                                "labels": ["non_dNCR", "dNCR"],
                                "values": confusion_matrix(
                                    labels,
                                    scores >= threshold,
                                    labels=[0, 1],
                                ).tolist(),
                            },
                        },
                    )

                if combined_labels is None:
                    raise RuntimeError("No combined ROC labels")
                plot_combined_roc(
                    combined_labels,
                    combined_scores,
                    (
                        results_root
                        / model
                        / variant
                        / network
                        / "三模式_ROC曲线.png"
                    ),
                    f"{model} | {variant} | {network}",
                )


def save_outputs(
    *,
    results_root: Path,
    config: dict[str, Any],
    split_info: dict[str, Any],
    assignment_rows: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    topology_priors: dict[str, dict[str, Any]],
    parallel_jobs: int,
) -> dict[str, Any]:
    n_splits = int(split_info["n_splits"])
    tasks = task_summaries(metric_rows, n_splits)
    features = feature_summaries(metric_rows)
    models = model_summaries(tasks)

    write_csv(
        results_root / "split_assignments.csv",
        assignment_rows,
    )
    metric_columns = (
        "split",
        "model",
        "variant",
        "network",
        "feature_mode",
        "result_source",
        *METRIC_NAMES,
    )
    write_csv(
        results_root / "per_split_metrics.csv",
        metric_rows,
        metric_columns,
    )
    hyperparameter_rows = metric_rows
    hyperparameter_columns = (
        "split",
        "model",
        "variant",
        "network",
        "feature_mode",
        "result_source",
        "selection_method",
        "selection_model",
        "reference_k",
        "k",
        "linear_svc_c",
        "selected_by_inner_cv",
        "reference_inner_auc",
        "selected_inner_auc",
        "selected_mean_auc_gain",
        "selected_repeat_win_rate",
        "candidate_count",
        "inner_cv_model_fits",
    )
    write_csv(
        results_root / "per_split_hyperparameters.csv",
        hyperparameter_rows,
        hyperparameter_columns,
    )
    prediction_columns = (
        "split",
        "model",
        "variant",
        "network",
        "feature_mode",
        "result_source",
        "subject_id",
        "label",
        "score",
        "prediction",
    )
    write_csv(
        results_root / "predictions.csv",
        prediction_rows,
        prediction_columns,
    )
    _save_task_artifacts(
        results_root=results_root,
        tasks=tasks,
        metric_rows=metric_rows,
        prediction_rows=prediction_rows,
        n_splits=n_splits,
    )

    payload = {
        "schema_version": 3,
        "experiment": {
            "random_seed": int(config["random_seed"]),
            "split_count": n_splits,
            "split_source": split_info["source"],
            "models": list(MODEL_ORDER),
            "variants": list(VARIANTS),
            "networks": list(NETWORKS),
            "feature_modes": list(FEATURE_MODES),
            "execution": {
                "parallel_jobs": parallel_jobs,
                "trained_model_fits": sum(
                    row["result_source"] == "trained"
                    for row in metric_rows
                ),
                "reused_original_results": sum(
                    row["result_source"] == "reused_original"
                    for row in metric_rows
                ),
                "inner_cv_model_fits": sum(
                    int(row["inner_cv_model_fits"])
                    for row in metric_rows
                    if row["result_source"] == "trained"
                ),
                "selected_by_inner_cv": sum(
                    bool(row["selected_by_inner_cv"])
                    for row in metric_rows
                    if row["result_source"] == "trained"
                ),
                "total_result_rows": len(metric_rows),
            },
        },
        "artifacts": {
            "split_assignments": "split_assignments.csv",
            "per_split_metrics": "per_split_metrics.csv",
            "per_split_hyperparameters": (
                "per_split_hyperparameters.csv"
            ),
            "predictions": "predictions.csv",
            "comparison_markdown": "model_auc_comparison.md",
            "comparison_pdf": "model_auc_comparison.pdf",
        },
        "feature_summaries": features,
        "topology_prior_selection": topology_priors,
        "model_summaries": models,
        "task_summaries": tasks,
    }
    write_json(results_root / "summary.json", payload)
    (results_root / "model_auc_comparison.md").write_text(
        render_comparison_markdown(tasks, models),
        encoding="utf-8",
    )
    return payload
