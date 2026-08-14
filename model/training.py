from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
from joblib import Parallel, cpu_count, delayed
from sklearn.preprocessing import StandardScaler

from model.constants import (
    FEATURE_MODES,
    MODEL_ORDER,
    NETWORKS,
    SUBJECT_ORDER_SHA256,
    VARIANTS,
)
from model.feature_design import build_design, specification
from model.hyperparameter_selection import (
    make_inner_folds,
    select_specs_by_model,
)
from model.modeling import make_model, model_scores
from model.prior_selection import select_topology_priors
from model.reporting import save_outputs
from model.split_selection import get_patient_splits
from utils.data_loader import Dataset, load_dataset
from utils.metrics import classification_metrics


def _resolve_parallel_jobs(config: dict[str, Any]) -> int:
    configured = config.get("runtime", {}).get("parallel_jobs", "auto")
    if configured == "auto":
        return max(1, cpu_count(only_physical_cores=True) - 1)
    jobs = int(configured)
    if jobs < 1:
        raise ValueError("runtime.parallel_jobs must be 'auto' or >= 1")
    return jobs


def _validate_config(config: dict[str, Any]) -> None:
    if tuple(config["models"]) != MODEL_ORDER:
        raise ValueError(
            f"models must be ordered as: {', '.join(MODEL_ORDER)}"
        )
    if tuple(config["networks"]) != NETWORKS:
        raise ValueError(
            f"networks must be ordered as: {', '.join(NETWORKS)}"
        )


def _evaluate_split(
    split_index: int,
    train: np.ndarray,
    test: np.ndarray,
    dataset: Dataset,
    config: dict[str, Any],
    topology_priors: dict[str, dict[str, Any] | None],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    assignments = [
        {
            "split": split_index,
            "subject_id": dataset.subject_ids[patient_index],
            "label": int(dataset.labels[patient_index]),
            "set": set_name,
        }
        for set_name, indices in (("train", train), ("test", test))
        for patient_index in indices
    ]
    metrics = []
    predictions = []
    inner_folds = make_inner_folds(
        outer_train=train,
        labels=dataset.labels,
        split_index=split_index,
        config=config,
    )
    original_results: dict[
        tuple[str, str, str],
        tuple[dict[str, Any], list[dict[str, Any]]],
    ] = {}

    for variant in VARIANTS:
        for network in NETWORKS:
            for feature_mode in FEATURE_MODES:
                reference_spec = specification(
                    variant, network, feature_mode, config
                )
                if reference_spec.mask_kind == "no_mask_fallback":
                    for model_name in MODEL_ORDER:
                        cache_key = (network, feature_mode, model_name)
                        try:
                            original_row, original_predictions = (
                                original_results[cache_key]
                            )
                        except KeyError as error:
                            raise RuntimeError(
                                "Original result was not cached before "
                                f"Mask fallback: {cache_key}"
                            ) from error
                        metrics.append(
                            {
                                **original_row,
                                "variant": variant,
                                "mask_kind": "no_mask_fallback",
                                "result_source": "reused_original",
                            }
                        )
                        predictions.extend(
                            {
                                **row,
                                "variant": variant,
                                "result_source": "reused_original",
                            }
                            for row in original_predictions
                        )
                    continue

                feature_specs, selection_details = (
                    select_specs_by_model(
                        reference=reference_spec,
                        inner_folds=inner_folds,
                        network=network,
                        feature_mode=feature_mode,
                        dataset=dataset,
                        config=config,
                        topology_priors=topology_priors,
                    )
                )
                for model_name in MODEL_ORDER:
                    feature_spec = feature_specs[model_name]
                    x_train, x_test, feature_details = build_design(
                        network=network,
                        feature_mode=feature_mode,
                        feature_spec=feature_spec,
                        dataset=dataset,
                        train=train,
                        test=test,
                        labels=dataset.labels,
                        config=config,
                        topology_priors=topology_priors,
                    )
                    scaler = StandardScaler()
                    train_scaled = scaler.fit_transform(x_train)
                    test_scaled = scaler.transform(x_test)
                    model, threshold, model_parameters = make_model(
                        model_name,
                        dataset.labels[train],
                        (
                            feature_spec.linear_svc_c
                            if model_name == "LinearSVC"
                            else None
                        ),
                        config,
                    )
                    model.fit(train_scaled, dataset.labels[train])
                    test_scores = model_scores(model, test_scaled)
                    row = {
                        "split": split_index,
                        "model": model_name,
                        "variant": variant,
                        "network": network,
                        "feature_mode": feature_mode,
                        "k": feature_spec.k,
                        "linear_svc_c": (
                            feature_spec.linear_svc_c
                            if model_name == "LinearSVC"
                            else None
                        ),
                        "feature_count": feature_details[
                            "feature_count"
                        ],
                        "mask_kind": feature_details["mask_kind"],
                        "result_source": "trained",
                        "threshold": threshold,
                        "model_parameters": model_parameters,
                        **selection_details[model_name],
                        **classification_metrics(
                            dataset.labels[test],
                            test_scores,
                            threshold,
                        ),
                    }
                    task_predictions = [
                        {
                            "split": split_index,
                            "model": model_name,
                            "variant": variant,
                            "network": network,
                            "feature_mode": feature_mode,
                            "result_source": "trained",
                            "subject_id": dataset.subject_ids[
                                patient_index
                            ],
                            "label": int(dataset.labels[patient_index]),
                            "score": float(test_scores[local_index]),
                            "prediction": int(
                                test_scores[local_index] >= threshold
                            ),
                        }
                        for local_index, patient_index in enumerate(test)
                    ]
                    metrics.append(row)
                    predictions.extend(task_predictions)
                    if variant == "Original":
                        original_results[
                            (network, feature_mode, model_name)
                        ] = (row, task_predictions)
    return metrics, assignments, predictions


def train_all(
    *,
    config: dict[str, Any],
    results_root: Path,
) -> dict[str, Any]:
    _validate_config(config)
    dataset = load_dataset(config)
    topology_priors = select_topology_priors(dataset, config)
    subject_hash = hashlib.sha256(
        "|".join(dataset.subject_ids).encode()
    ).hexdigest()
    if subject_hash != SUBJECT_ORDER_SHA256:
        raise RuntimeError(
            "Dataset subject order changed; verify the input cohort"
        )

    splits, split_info = get_patient_splits(dataset, config)
    parallel_jobs = _resolve_parallel_jobs(config)
    evaluated = Parallel(
        n_jobs=parallel_jobs,
        verbose=10,
    )(
        delayed(_evaluate_split)(
            split_index,
            train,
            test,
            dataset,
            config,
            topology_priors,
        )
        for split_index, train, test in splits
    )
    metric_rows = [
        row for split_metrics, _, _ in evaluated for row in split_metrics
    ]
    assignment_rows = [
        row for _, split_assignments, _ in evaluated
        for row in split_assignments
    ]
    prediction_rows = [
        row for _, _, split_predictions in evaluated
        for row in split_predictions
    ]
    payload = save_outputs(
        results_root=results_root,
        config=config,
        split_info=split_info,
        assignment_rows=assignment_rows,
        metric_rows=metric_rows,
        prediction_rows=prediction_rows,
        topology_priors=topology_priors,
        parallel_jobs=parallel_jobs,
    )
    return {
        "results_root": str(results_root.resolve()),
        "split_info": split_info,
        "model_summaries": payload["model_summaries"],
        "topology_priors": topology_priors,
        "execution": payload["experiment"]["execution"],
    }
