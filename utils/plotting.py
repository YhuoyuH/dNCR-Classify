from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


NAVY = "#123047"
TEAL = "#0E7490"
GREEN = "#15803D"
GOLD = "#D97706"
RED = "#B91C1C"
GRAY = "#64748B"
LIGHT_GRID = "#CBD5E1"
BACKGROUND = "#F8FAFC"
MODE_COLORS = {
    "FC": "#0E7490",
    "Topology": "#D97706",
    "FC+Topology": "#15803D",
}


def _prepare_axes(title: str, xlabel: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(6.4, 5.2), dpi=180)
    fig.patch.set_facecolor("white")
    ax.set_facecolor(BACKGROUND)
    ax.set_title(title, fontsize=14, fontweight="bold", color=NAVY, pad=13)
    ax.set_xlabel(xlabel, fontsize=11, color=NAVY)
    ax.set_ylabel(ylabel, fontsize=11, color=NAVY)
    ax.grid(True, color=LIGHT_GRID, linewidth=0.7, alpha=0.65)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(LIGHT_GRID)
    ax.tick_params(colors=GRAY)
    return fig, ax


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(
        path,
        dpi=220,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def plot_roc_curve(
    labels: np.ndarray,
    scores: np.ndarray,
    path: Path,
    title: str,
) -> None:
    fpr, tpr, _ = roc_curve(labels, scores)
    auc_value = roc_auc_score(labels, scores)
    fig, ax = _prepare_axes(
        title, "False positive rate", "True positive rate"
    )
    ax.plot(
        fpr,
        tpr,
        color=TEAL,
        linewidth=3,
        label=f"ROC-AUC = {auc_value:.3f}",
    )
    ax.fill_between(fpr, tpr, color=TEAL, alpha=0.12)
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color=GRAY,
        linewidth=1.4,
        label="Chance",
    )
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(
        loc="lower right",
        frameon=True,
        facecolor="white",
        edgecolor=LIGHT_GRID,
        fontsize=10,
    )
    _save(fig, path)


def plot_pr_curve(
    labels: np.ndarray,
    scores: np.ndarray,
    path: Path,
    title: str,
) -> None:
    precision, recall, _ = precision_recall_curve(labels, scores)
    average_precision = average_precision_score(labels, scores)
    prevalence = float(np.mean(labels))
    fig, ax = _prepare_axes(title, "Recall", "Precision")
    ax.plot(
        recall,
        precision,
        color=GOLD,
        linewidth=3,
        label=f"PR-AUC = {average_precision:.3f}",
    )
    ax.fill_between(recall, precision, color=GOLD, alpha=0.12)
    ax.axhline(
        prevalence,
        linestyle="--",
        color=GRAY,
        linewidth=1.4,
        label=f"Prevalence = {prevalence:.3f}",
    )
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor=LIGHT_GRID,
        fontsize=10,
    )
    _save(fig, path)


def plot_confusion_matrix(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    path: Path,
    title: str,
) -> None:
    predictions = (scores >= threshold).astype(int)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    color_map = LinearSegmentedColormap.from_list(
        "dncr_teal", ["#F8FAFC", "#67C4D4", "#0E7490", "#123047"]
    )
    fig, ax = plt.subplots(figsize=(5.6, 5.0), dpi=180)
    fig.patch.set_facecolor("white")
    image = ax.imshow(matrix, cmap=color_map, vmin=0)
    ax.set_title(title, fontsize=14, fontweight="bold", color=NAVY, pad=13)
    ax.set_xlabel("Predicted label", fontsize=11, color=NAVY)
    ax.set_ylabel("True label", fontsize=11, color=NAVY)
    ax.set_xticks([0, 1], labels=["non-dNCR", "dNCR"])
    ax.set_yticks([0, 1], labels=["non-dNCR", "dNCR"])
    ax.tick_params(colors=GRAY)
    for row in range(2):
        row_total = int(matrix[row].sum())
        for column in range(2):
            count = int(matrix[row, column])
            percentage = count / row_total if row_total else 0.0
            ax.text(
                column,
                row,
                f"{count}\n{percentage:.1%}",
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
                color=(
                    "white"
                    if count > float(matrix.max()) * 0.55
                    else NAVY
                ),
            )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.ax.tick_params(colors=GRAY)
    for spine in ax.spines.values():
        spine.set_color(LIGHT_GRID)
    _save(fig, path)


def plot_combined_roc(
    labels: np.ndarray,
    scores_by_mode: Mapping[str, np.ndarray],
    path: Path,
    title: str,
) -> None:
    fig, ax = _prepare_axes(
        title, "False positive rate", "True positive rate"
    )
    for display_mode, scores in scores_by_mode.items():
        fpr, tpr, _ = roc_curve(labels, scores)
        auc_value = roc_auc_score(labels, scores)
        ax.plot(
            fpr,
            tpr,
            linewidth=2.8,
            color=MODE_COLORS[display_mode],
            label=f"{display_mode} (AUC {auc_value:.3f})",
        )
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color=GRAY,
        linewidth=1.3,
        label="Chance",
    )
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(
        loc="lower right",
        frameon=True,
        facecolor="white",
        edgecolor=LIGHT_GRID,
        fontsize=9.5,
    )
    _save(fig, path)
