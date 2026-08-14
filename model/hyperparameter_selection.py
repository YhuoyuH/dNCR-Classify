from __future__ import annotations

import math
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

from model.constants import MODEL_ORDER
from model.feature_design import FeatureSpec, build_design
from model.modeling import make_model, model_scores
from utils.data_loader import Dataset


SELECTION_METHOD = (
    "per_model_repeated_stratified_kfold_best_mean_auc"
)


def make_inner_folds(
    *,
    outer_train: np.ndarray,
    labels: np.ndarray,
    split_index: int,
    config: dict[str, Any],
) -> list[tuple[np.ndarray, np.ndarray]]:
    settings = config["k_selection"]
    n_splits = int(settings["inner_splits"])
    n_repeats = int(settings["inner_repeats"])
    if n_splits < 2 or n_repeats < 1:
        raise ValueError(
            "k_selection inner_splits must be >= 2 and "
            "inner_repeats must be >= 1"
        )
    splitter = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=int(config["random_seed"]) + split_index,
    )
    return [
        (outer_train[local_train], outer_train[local_valid])
        for local_train, local_valid in splitter.split(
            np.zeros(len(outer_train)), labels[outer_train]
        )
    ]


def _feature_width(
    dataset: Dataset,
    network: str,
    feature_mode: str,
) -> int:
    if feature_mode == "FC":
        return int(dataset.fc[network].shape[1])
    if feature_mode == "topology":
        return int(dataset.topology[network].shape[1])
    if feature_mode == "FC+topology":
        return int(
            dataset.fc[network].shape[1]
            + dataset.topology[network].shape[1]
        )
    raise ValueError(f"Unknown feature mode: {feature_mode}")


def _candidate_specs(
    reference: FeatureSpec,
    feature_width: int,
    config: dict[str, Any],
) -> list[FeatureSpec]:
    settings = config["k_selection"]
    lower_ratio = float(settings["k_lower_ratio"])
    upper_ratio = float(settings["k_upper_ratio"])
    maximum_candidates = int(settings["maximum_k_candidates"])
    if not 0.0 < lower_ratio <= 1.0 <= upper_ratio:
        raise ValueError(
            "K search ratios must satisfy 0 < lower <= 1 <= upper"
        )
    if maximum_candidates < 2:
        raise ValueError("maximum_k_candidates must be >= 2")

    lower = max(1, int(math.floor(reference.k * lower_ratio)))
    upper = min(
        feature_width,
        int(math.ceil(reference.k * upper_ratio)),
    )
    if upper - lower + 1 <= maximum_candidates:
        k_values = set(range(lower, upper + 1))
    else:
        k_values = {
            int(round(value))
            for value in np.linspace(lower, upper, maximum_candidates)
        }
    k_values.add(min(feature_width, reference.k))
    return [
        FeatureSpec(
            k=k,
            linear_svc_c=reference.linear_svc_c,
            mask_kind=reference.mask_kind,
        )
        for k in sorted(k_values)
    ]


def _inner_aucs(
    *,
    train: np.ndarray,
    valid: np.ndarray,
    model_names: tuple[str, ...],
    network: str,
    feature_mode: str,
    feature_spec: FeatureSpec,
    dataset: Dataset,
    config: dict[str, Any],
    topology_priors: dict[str, dict[str, Any] | None],
) -> dict[str, float]:
    x_train, x_valid, _ = build_design(
        network=network,
        feature_mode=feature_mode,
        feature_spec=feature_spec,
        dataset=dataset,
        train=train,
        test=valid,
        labels=dataset.labels,
        config=config,
        topology_priors=topology_priors,
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_valid = scaler.transform(x_valid)
    aucs = {}
    for model_name in model_names:
        model, _, _ = make_model(
            model_name,
            dataset.labels[train],
            (
                feature_spec.linear_svc_c
                if model_name == "LinearSVC"
                else None
            ),
            config,
        )
        model.fit(x_train, dataset.labels[train])
        scores = model_scores(model, x_valid)
        aucs[model_name] = float(
            roc_auc_score(dataset.labels[valid], scores)
        )
    return aucs


def select_specs_by_model(
    *,
    reference: FeatureSpec,
    inner_folds: list[tuple[np.ndarray, np.ndarray]],
    network: str,
    feature_mode: str,
    dataset: Dataset,
    config: dict[str, Any],
    topology_priors: dict[str, dict[str, Any] | None],
) -> tuple[
    dict[str, FeatureSpec],
    dict[str, dict[str, Any]],
]:
    """Select K separately for every model using inner-training data."""
    settings = config["k_selection"]
    n_splits = int(settings["inner_splits"])
    n_repeats = int(settings["inner_repeats"])
    if len(inner_folds) != n_splits * n_repeats:
        raise ValueError("Unexpected number of inner CV folds")

    candidates = _candidate_specs(
        reference,
        _feature_width(dataset, network, feature_mode),
        config,
    )
    scores: dict[str, dict[int, list[float]]] = {
        model_name: {} for model_name in MODEL_ORDER
    }
    for candidate in candidates:
        key = candidate.k
        for model_name in MODEL_ORDER:
            scores[model_name][key] = []
        for train, valid in inner_folds:
            fold_aucs = _inner_aucs(
                train=train,
                valid=valid,
                model_names=MODEL_ORDER,
                network=network,
                feature_mode=feature_mode,
                feature_spec=candidate,
                dataset=dataset,
                config=config,
                topology_priors=topology_priors,
            )
            for model_name, auc in fold_aucs.items():
                scores[model_name][key].append(auc)

    reference_key = reference.k
    selected_specs = {}
    selection_details = {}
    for model_name in MODEL_ORDER:
        model_scores_by_candidate = {
            key: np.asarray(values, dtype=float)
            for key, values in scores[model_name].items()
        }
        reference_scores = model_scores_by_candidate[reference_key]
        reference_mean = float(np.mean(reference_scores))
        selected = max(
            candidates,
            key=lambda candidate: (
                float(
                    np.mean(
                        model_scores_by_candidate[
                            candidate.k
                        ]
                    )
                ),
                -candidate.k,
            ),
        )
        selected_scores = model_scores_by_candidate[
            selected.k
        ]
        repeat_delta = np.mean(
            (selected_scores - reference_scores).reshape(
                n_repeats, n_splits
            ),
            axis=1,
        )
        selected_specs[model_name] = selected
        selection_details[model_name] = {
            "selection_method": SELECTION_METHOD,
            "selection_model": model_name,
            "reference_k": reference.k,
            "selected_by_inner_cv": selected.k != reference.k,
            "reference_inner_auc": reference_mean,
            "selected_inner_auc": float(np.mean(selected_scores)),
            "selected_mean_auc_gain": float(
                np.mean(selected_scores) - reference_mean
            ),
            "selected_repeat_win_rate": float(
                np.mean(repeat_delta > 0.0)
            ),
            "candidate_count": len(candidates),
            "inner_cv_model_fits": len(candidates) * len(inner_folds),
        }
    return selected_specs, selection_details
