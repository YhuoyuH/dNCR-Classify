from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat


GROUPS = (("dNCR", 1), ("non_dNCR", 0))
GLOBAL_METRIC_PATHS = (
    "Assortativity/ar.txt",
    "Hierarchy/ab.txt",
    "NetworkEfficiency/aEg.txt",
    "NetworkEfficiency/aEloc.txt",
    "SmallWorld/aCp.txt",
    "SmallWorld/aGamma.txt",
    "SmallWorld/aLambda.txt",
    "SmallWorld/aLp.txt",
    "SmallWorld/aSigma.txt",
    "Synchronization/as.txt",
)
NODAL_METRIC_PATHS = (
    "BetweennessCentrality/aBc.txt",
    "DegreeCentrality/aDc.txt",
    "NodalClustCoeff/aNCp.txt",
    "NodalEfficiency/aNe.txt",
    "NodalLocalEfficiency/aNLe.txt",
    "NodalShortestPath/aNLp.txt",
)
TOPOLOGY_METRIC_PATHS = GLOBAL_METRIC_PATHS + NODAL_METRIC_PATHS


@dataclass(frozen=True)
class Dataset:
    subject_ids: list[str]
    labels: np.ndarray
    fc: dict[str, np.ndarray]
    node_counts: dict[str, int]
    topology: dict[str, np.ndarray]
    topology_feature_names: dict[str, tuple[str, ...]]


def patient_id(value: str) -> str:
    match = re.search(r"(?:z|r)sub_(\d+)", value)
    if not match:
        raise ValueError(f"Cannot parse subject ID from {value!r}")
    return match.group(1)


def _load_fc_matrices(
    data_root: Path, network_folder: str
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    matrices: dict[str, np.ndarray] = {}
    labels: dict[str, int] = {}
    for group, label in GROUPS:
        folder = (
            data_root
            / "winner_takes_all_FC"
            / network_folder
            / "pre_surgery"
            / group
        )
        paths = sorted(folder.glob("zsub_*.txt"))
        if not paths:
            raise FileNotFoundError(f"No FC files found in {folder}")
        for path in paths:
            subject = patient_id(path.name)
            matrix = np.asarray(np.loadtxt(path), dtype=float)
            if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
                raise ValueError(f"FC matrix is not square: {path}")
            if subject in matrices:
                raise ValueError(f"Duplicate FC subject {subject}: {path}")
            matrices[subject] = matrix
            labels[subject] = label
    return matrices, labels


def _topology_row_ids(folder: Path) -> list[str]:
    pipe_path = folder / "GretnaLogs" / "NetAnalysis" / "PIPE.mat"
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The default value for `spmatrix` is changing",
            category=DeprecationWarning,
        )
        pipe = loadmat(pipe_path, simplify_cells=True)
    files_in = pipe["files_in"]
    keys = sorted(
        (key for key in files_in if key.startswith("ThresMat")),
        key=lambda key: int(key.removeprefix("ThresMat")),
    )
    return [patient_id(str(files_in[key])) for key in keys]


def _load_topology(
    data_root: Path, network_folder: str
) -> tuple[
    dict[str, np.ndarray],
    dict[str, int],
    tuple[str, ...],
]:
    rows: dict[str, np.ndarray] = {}
    labels: dict[str, int] = {}
    expected_widths: tuple[int, ...] | None = None
    for group, label in GROUPS:
        folder = (
            data_root
            / "winner_takes_all_topology"
            / network_folder
            / "pre_surgery"
            / f"{group}_r"
        )
        subject_ids = _topology_row_ids(folder)
        blocks: list[np.ndarray] = []
        widths: list[int] = []
        for relative in TOPOLOGY_METRIC_PATHS:
            path = folder / relative
            values = np.asarray(np.loadtxt(path), dtype=float)
            if values.ndim == 1:
                values = values.reshape(-1, 1)
            if values.shape[0] != len(subject_ids):
                raise ValueError(
                    f"{path}: {values.shape[0]} rows; "
                    f"expected {len(subject_ids)}"
                )
            blocks.append(values)
            widths.append(values.shape[1])
        data = np.concatenate(blocks, axis=1)
        data[~np.isfinite(data)] = np.nan
        if expected_widths is None:
            expected_widths = tuple(widths)
        elif expected_widths != tuple(widths):
            raise ValueError(
                f"Topology columns differ across groups: {network_folder}"
            )
        if len(subject_ids) != len(data):
            raise ValueError(
                f"Topology subject/row counts differ: {network_folder}"
            )
        for subject, vector in zip(subject_ids, data):
            if subject in rows:
                raise ValueError(
                    f"Duplicate topology subject {subject}: {folder}"
                )
            rows[subject] = vector
            labels[subject] = label
    if expected_widths is None:
        raise ValueError(f"No topology data for {network_folder}")
    feature_names = []
    for relative, width in zip(TOPOLOGY_METRIC_PATHS, expected_widths):
        if width == 1:
            feature_names.append(relative)
        else:
            feature_names.extend(
                f"{relative}#column_{column + 1}"
                for column in range(width)
            )
    return rows, labels, tuple(feature_names)


def _upper_triangle(
    matrices: dict[str, np.ndarray], subject_ids: list[str]
) -> np.ndarray:
    first = matrices[subject_ids[0]]
    row, column = np.triu_indices(first.shape[0], k=1)
    return np.vstack(
        [matrices[subject][row, column] for subject in subject_ids]
    )


def load_dataset(config: dict[str, Any]) -> Dataset:
    """Load and align all four networks from the configured data root."""
    data_root = Path(config["data_root"])
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    all_fc_matrices: dict[str, dict[str, np.ndarray]] = {}
    fc_labels: dict[str, int] | None = None
    for network, definition in config["networks"].items():
        matrices, labels = _load_fc_matrices(
            data_root, definition["folder"]
        )
        all_fc_matrices[network] = matrices
        if fc_labels is None:
            fc_labels = labels
        elif fc_labels != labels:
            raise ValueError(f"FC subject labels differ for {network}")
    if fc_labels is None:
        raise ValueError("No FC data loaded")

    subject_ids = sorted(
        fc_labels, key=lambda subject: (-fc_labels[subject], int(subject))
    )
    labels = np.asarray(
        [fc_labels[subject] for subject in subject_ids], dtype=int
    )

    fc: dict[str, np.ndarray] = {}
    node_counts: dict[str, int] = {}
    topology: dict[str, np.ndarray] = {}
    topology_feature_names: dict[str, tuple[str, ...]] = {}
    for network, definition in config["networks"].items():
        matrix_sizes = {
            matrix.shape[0]
            for matrix in all_fc_matrices[network].values()
        }
        if len(matrix_sizes) != 1:
            raise ValueError(
                f"{network}: inconsistent FC matrix sizes: "
                f"{sorted(matrix_sizes)}"
            )
        node_counts[network] = matrix_sizes.pop()
        fc[network] = _upper_triangle(
            all_fc_matrices[network], subject_ids
        )
        topology_rows, topology_labels, feature_names = _load_topology(
            data_root, definition["folder"]
        )
        if topology_labels != fc_labels:
            raise ValueError(
                f"FC and topology labels differ for {network}"
            )
        topology[network] = np.vstack(
            [topology_rows[subject] for subject in subject_ids]
        )
        if len(feature_names) != topology[network].shape[1]:
            raise RuntimeError(
                f"{network}: topology feature-name count mismatch"
            )
        topology_feature_names[network] = feature_names

    return Dataset(
        subject_ids=subject_ids,
        labels=labels,
        fc=fc,
        node_counts=node_counts,
        topology=topology,
        topology_feature_names=topology_feature_names,
    )
