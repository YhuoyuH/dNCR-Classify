from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from configs.settings import PROJECT_ROOT


def _load_test_subject_ids(
    config: dict[str, Any],
) -> tuple[list[list[str]], Path]:
    split_path = Path(config["split"]["fixed_splits_path"]).expanduser()
    if not split_path.is_absolute():
        split_path = PROJECT_ROOT / split_path
    split_path = split_path.resolve()

    payload = json.loads(split_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid split file: {split_path}")
    if payload.get("format") != "fixed_test_subject_ids":
        raise ValueError(f"Unsupported split format: {split_path}")

    raw_splits = payload.get("test_subject_ids")
    if not isinstance(raw_splits, list) or not raw_splits:
        raise ValueError("test_subject_ids must be a non-empty list")
    if payload.get("n_splits") != len(raw_splits):
        raise ValueError("n_splits does not match test_subject_ids")

    parsed = []
    for split_index, raw_ids in enumerate(raw_splits, start=1):
        if (
            not isinstance(raw_ids, list)
            or not all(isinstance(value, str) for value in raw_ids)
        ):
            raise TypeError(
                f"Split {split_index} must contain subject ID strings"
            )
        subject_ids = [value.strip() for value in raw_ids]
        if any(not value for value in subject_ids):
            raise ValueError(f"Split {split_index} contains an empty ID")
        if len(subject_ids) != len(set(subject_ids)):
            raise ValueError(f"Split {split_index} contains duplicate IDs")
        parsed.append(subject_ids)
    return parsed, split_path


def get_patient_splits(
    dataset: Any,
    config: dict[str, Any],
) -> tuple[
    list[tuple[int, np.ndarray, np.ndarray]],
    dict[str, Any],
]:
    configured_splits, split_path = _load_test_subject_ids(config)
    subject_to_index = {
        subject_id: index
        for index, subject_id in enumerate(dataset.subject_ids)
    }
    all_subjects = set(subject_to_index)
    all_indices = np.arange(len(dataset.subject_ids), dtype=int)

    splits = []
    seen_tests: set[tuple[str, ...]] = set()
    expected_sizes: tuple[int, int] | None = None
    expected_class_counts: tuple[int, int] | None = None

    for split_index, test_subject_ids in enumerate(
        configured_splits, start=1
    ):
        unknown = sorted(set(test_subject_ids) - all_subjects)
        if unknown:
            raise ValueError(
                f"Split {split_index} contains unknown subjects: {unknown}"
            )

        signature = tuple(sorted(test_subject_ids))
        if signature in seen_tests:
            raise ValueError(f"Split {split_index} is duplicated")
        seen_tests.add(signature)

        test = np.asarray(
            [subject_to_index[subject] for subject in test_subject_ids],
            dtype=int,
        )
        train = np.setdiff1d(all_indices, test, assume_unique=True)

        sizes = (len(train), len(test))
        class_counts = (
            int(dataset.labels[train].sum()),
            int(dataset.labels[test].sum()),
        )
        if expected_sizes is None:
            expected_sizes = sizes
            expected_class_counts = class_counts
        elif sizes != expected_sizes:
            raise ValueError(f"Split {split_index} has inconsistent sizes")
        elif class_counts != expected_class_counts:
            raise ValueError(
                f"Split {split_index} has inconsistent class counts"
            )
        splits.append((split_index, train, test))

    if expected_sizes is None:
        raise RuntimeError("No patient splits were loaded")
    try:
        source = split_path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        source = str(split_path)

    return splits, {
        "source": source,
        "n_splits": len(splits),
        "train_n": expected_sizes[0],
        "test_n": expected_sizes[1],
    }
