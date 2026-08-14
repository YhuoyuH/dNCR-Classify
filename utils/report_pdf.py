from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

from configs.settings import load_config
from model.constants import FEATURE_MODES, MODEL_ORDER, NETWORKS


FONT = "Helvetica"
FONT_BOLD = "Helvetica-Bold"

NAVY = colors.HexColor("#15364D")
TEAL = colors.HexColor("#008F91")
TEXT = colors.HexColor("#17324A")
MUTED = colors.HexColor("#60758A")
GRID = colors.HexColor("#B9CAD6")
PALE_BLUE = colors.HexColor("#EAF1F6")
PALE_GREEN = colors.HexColor("#E4F4F0")
ALT_ROW = colors.HexColor("#F5F8FA")
WHITE = colors.white

MODE_LABELS = {
    "FC": "FC",
    "topology": "Topology",
    "FC+topology": "FC+Topology",
}

MODEL_INSIGHTS = {
    "LinearSVC": (
        "It was the highest-performing and most stable model overall."
    ),
    "LDA": (
        "It ranked second overall, with relatively strong FC-related "
        "performance but weaker topology-only performance in some "
        "subnetworks."
    ),
    "RF": (
        "It ranked third overall; several WB tasks were competitive, "
        "whereas performance varied across subnetworks and SMN Mask "
        "tasks."
    ),
    "DT": (
        "It had the lowest overall AUC and showed substantial variation "
        "across task combinations."
    ),
    "XGBoost": (
        "It ranked fourth overall and approached an AUC of 0.80 in its "
        "best task, but performance was lower in several subnetwork tasks."
    ),
}


def styles() -> dict[str, ParagraphStyle]:
    sample = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title",
            parent=sample["Title"],
            fontName=FONT,
            fontSize=20,
            leading=23,
            textColor=TEXT,
            alignment=TA_LEFT,
            spaceAfter=1.8 * mm,
        ),
        "section": ParagraphStyle(
            "section",
            parent=sample["Heading2"],
            fontName=FONT,
            fontSize=11.5,
            leading=14,
            textColor=TEXT,
            alignment=TA_LEFT,
            spaceBefore=1.2 * mm,
            spaceAfter=1.1 * mm,
        ),
        "body": ParagraphStyle(
            "body",
            parent=sample["BodyText"],
            fontName=FONT,
            fontSize=7.8,
            leading=10,
            textColor=TEXT,
            alignment=TA_LEFT,
            spaceAfter=1.4 * mm,
        ),
        "note": ParagraphStyle(
            "note",
            parent=sample["BodyText"],
            fontName=FONT,
            fontSize=6.7,
            leading=8.3,
            textColor=MUTED,
            alignment=TA_LEFT,
        ),
        "header": ParagraphStyle(
            "header",
            parent=sample["Normal"],
            fontName=FONT_BOLD,
            fontSize=6.5,
            leading=7.4,
            textColor=WHITE,
            alignment=1,
        ),
    }


def metric_text(row: dict[str, Any], metric_name: str) -> str:
    metric = row["metrics"][metric_name]
    return f"{float(metric['mean']):.3f} +/- {float(metric['std']):.3f}"


def _base_table_style(
    *,
    header_rows: int = 1,
    font_size: float = 6.4,
) -> TableStyle:
    return TableStyle(
        [
            ("FONTNAME", (0, 0), (-1, -1), FONT),
            ("FONTNAME", (0, 0), (-1, header_rows - 1), FONT_BOLD),
            ("FONTSIZE", (0, header_rows), (-1, -1), font_size),
            ("TEXTCOLOR", (0, header_rows), (-1, -1), TEXT),
            ("BACKGROUND", (0, 0), (-1, header_rows - 1), NAVY),
            ("GRID", (0, 0), (-1, -1), 0.35, GRID),
            ("BOX", (0, 0), (-1, -1), 0.7, NAVY),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 2.5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 2.5),
            ("TOPPADDING", (0, 0), (-1, -1), 1.3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 1.3),
        ]
    )


def _add_alternating_rows(
    table_style: TableStyle,
    row_count: int,
    *,
    start_row: int = 1,
) -> None:
    for row in range(start_row, row_count):
        if (row - start_row) % 2 == 1:
            table_style.add("BACKGROUND", (0, row), (-1, row), ALT_ROW)


def _overview_table(
    model_name: str,
    lookup: dict[tuple[str, str, str, str], dict[str, Any]],
    style: dict[str, ParagraphStyle],
) -> Table:
    headers: list[Any] = [Paragraph("Network", style["header"])]
    for mode in FEATURE_MODES:
        label = MODE_LABELS[mode]
        headers.extend(
            [
                Paragraph(f"{label}<br/>no-mask", style["header"]),
                Paragraph(f"{label}<br/>mask", style["header"]),
            ]
        )
    data: list[list[Any]] = [headers]
    for network in NETWORKS:
        row: list[Any] = [network]
        for mode in FEATURE_MODES:
            row.append(
                metric_text(
                    lookup[(model_name, "Original", network, mode)],
                    "roc_auc",
                )
            )
            row.append(
                metric_text(
                    lookup[(model_name, "Mask", network, mode)],
                    "roc_auc",
                )
            )
        data.append(row)

    table = Table(
        data,
        colWidths=[42, 118, 118, 118, 118, 128, 128],
        rowHeights=[17, 14, 14, 14, 14],
        repeatRows=1,
    )
    table_style = _base_table_style(font_size=6.8)
    table_style.add("ALIGN", (0, 0), (-1, -1), "CENTER")
    table_style.add("BACKGROUND", (0, 1), (0, -1), PALE_BLUE)
    _add_alternating_rows(table_style, len(data))
    for row_index, network in enumerate(NETWORKS, start=1):
        for pair_index, mode in enumerate(FEATURE_MODES):
            original = float(
                lookup[
                    (model_name, "Original", network, mode)
                ]["metrics"]["roc_auc"]["mean"]
            )
            mask = float(
                lookup[
                    (model_name, "Mask", network, mode)
                ]["metrics"]["roc_auc"]["mean"]
            )
            if original > mask + 1e-12:
                column = 1 + 2 * pair_index
            elif mask > original + 1e-12:
                column = 2 + 2 * pair_index
            else:
                continue
            table_style.add(
                "BACKGROUND",
                (column, row_index),
                (column, row_index),
                PALE_GREEN,
            )
            table_style.add(
                "TEXTCOLOR",
                (column, row_index),
                (column, row_index),
                TEAL,
            )
    table.setStyle(table_style)
    return table


def _detail_table(
    model_name: str,
    variant: str,
    lookup: dict[tuple[str, str, str, str], dict[str, Any]],
    style: dict[str, ParagraphStyle],
) -> Table:
    metric_columns = (
        ("ROC-AUC", "roc_auc"),
        ("PR-AUC", "pr_auc"),
        ("Balanced<br/>Accuracy", "balanced_accuracy"),
        ("Sensitivity", "sensitivity"),
        ("Specificity", "specificity"),
        ("Accuracy", "accuracy"),
        ("F1", "f1"),
    )
    data: list[list[Any]] = [
        [
            Paragraph("Network", style["header"]),
            Paragraph("Features", style["header"]),
            *(
                Paragraph(label, style["header"])
                for label, _ in metric_columns
            ),
        ]
    ]
    for network in NETWORKS:
        for mode in FEATURE_MODES:
            task = lookup[(model_name, variant, network, mode)]
            data.append(
                [
                    network,
                    MODE_LABELS[mode],
                    *(
                        metric_text(task, metric_name)
                        for _, metric_name in metric_columns
                    ),
                ]
            )

    table = Table(
        data,
        colWidths=[39, 65, 94, 94, 103, 94, 94, 94, 91],
        rowHeights=[14, *(10.5 for _ in range(12))],
        repeatRows=1,
    )
    table_style = _base_table_style(font_size=5.7)
    table_style.add("ALIGN", (2, 1), (-1, -1), "CENTER")
    _add_alternating_rows(table_style, len(data))
    table.setStyle(table_style)
    return table


def _method_table(
    n_splits: int,
    train_n: int,
    test_n: int,
    config: dict[str, Any],
    style: dict[str, ParagraphStyle],
) -> Table:
    k_selection = config["k_selection"]
    rows = [
        [
            Paragraph("Step", style["header"]),
            Paragraph("Implementation", style["header"]),
            Paragraph("Test data used?", style["header"]),
        ],
        [
            "Patient splits",
            (
                f"{n_splits} fixed stratified holdout splits; "
                f"{train_n} training and {test_n} test subjects per split"
            ),
            "Only to define evaluation sets",
        ],
        ["Topology imputation", "Training-set median imputation", "No"],
        [
            "Feature selection",
            (
                "Training-fold ANOVA-F ranking; each model selects K "
                f"using {k_selection['inner_repeats']} repeats of "
                f"stratified {k_selection['inner_splits']}-fold CV"
            ),
            "No",
        ],
        [
            "Mask construction",
            (
                "WB/DMN preserve topology priors; SMN appends the "
                "NBS-weighted score"
            ),
            "No",
        ],
        [
            "Scaling",
            "StandardScaler fitted only on the corresponding training data",
            "No",
        ],
        [
            "Model fitting",
            "Five classifiers trained separately using their own selected K",
            "No",
        ],
        [
            "Evaluation",
            (
                f"Metrics computed in {test_n} test subjects and summarized "
                f"across {n_splits} splits"
            ),
            "Yes",
        ],
    ]
    table = Table(rows, colWidths=[112, 515, 125], rowHeights=[17, 19, 16, 23, 21, 17, 17, 20])
    table_style = _base_table_style(font_size=7.2)
    _add_alternating_rows(table_style, len(rows))
    table.setStyle(table_style)
    return table


def _model_parameter_table(
    config: dict[str, Any],
    style: dict[str, ParagraphStyle],
) -> Table:
    models = config["models"]
    rows: list[list[Any]] = [
        [
            Paragraph("Model", style["header"]),
            Paragraph("Fixed parameters", style["header"]),
        ],
        [
            "LinearSVC",
            (
                "class_weight=balanced; task-specific fixed C; "
                f"max_iter={models['LinearSVC']['max_iter']}"
            ),
        ],
        [
            "LDA",
            (
                f"solver={models['LDA']['solver']}; "
                f"shrinkage={models['LDA']['shrinkage']}"
            ),
        ],
        [
            "RF",
            (
                f"n_estimators={models['RF']['n_estimators']}; "
                f"max_features={models['RF']['max_features']}; "
                f"min_samples_leaf={models['RF']['min_samples_leaf']}; "
                "class_weight=balanced"
            ),
        ],
        [
            "DT",
            (
                f"max_depth={models['DT']['max_depth']}; "
                f"min_samples_leaf={models['DT']['min_samples_leaf']}; "
                "class_weight=balanced"
            ),
        ],
        [
            "XGBoost",
            (
                f"n_estimators={models['XGBoost']['n_estimators']}; "
                f"max_depth={models['XGBoost']['max_depth']}; "
                f"learning_rate={models['XGBoost']['learning_rate']}; "
                f"subsample={models['XGBoost']['subsample']}; "
                f"colsample_bytree={models['XGBoost']['colsample_bytree']}"
            ),
        ],
    ]
    table = Table(rows, colWidths=[112, 640], rowHeights=[17, 17, 17, 19, 17, 21])
    table_style = _base_table_style(font_size=7.2)
    _add_alternating_rows(table_style, len(rows))
    table.setStyle(table_style)
    return table


def _split_sizes(results_root: Path) -> tuple[int, int]:
    with (results_root / "split_assignments.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        first_split = [
            row for row in csv.DictReader(handle) if int(row["split"]) == 1
        ]
    train_n = sum(row["set"] == "train" for row in first_split)
    test_n = sum(row["set"] == "test" for row in first_split)
    if not train_n or not test_n:
        raise ValueError("Could not determine training/test split sizes")
    return train_n, test_n


def _overall_sentence(model_summaries: list[dict[str, Any]]) -> str:
    summaries = {
        row["model"]: row for row in model_summaries
    }

    def result(model: str, *, figure: bool = False) -> str:
        row = summaries[model]
        text = (
            f"mean AUC: {float(row['roc_auc_mean']):.3f}, "
            f"range: {float(row['roc_auc_min']):.3f}-"
            f"{float(row['roc_auc_max']):.3f}"
        )
        return f"{text}; Fig. 3A" if figure else text

    return (
        "Among all models, LinearSVC was identified as the most suitable "
        f"model ({result('LinearSVC', figure=True)}), followed by LDA "
        f"({result('LDA')}), RF ({result('RF')}), XGBoost "
        f"({result('XGBoost')}), and DT ({result('DT')})."
    )


def _draw_page_chrome(
    canvas,
    document,
    *,
    n_splits: int,
    train_n: int,
    test_n: int,
) -> None:
    canvas.saveState()
    width, height = landscape(A4)
    canvas.setFillColor(NAVY)
    canvas.rect(0, height - 24, width, 24, stroke=0, fill=1)
    canvas.setFillColor(WHITE)
    canvas.setFont(FONT, 7.5)
    canvas.drawString(
        12 * mm,
        height - 15.5,
        "dNCR Classification Model Comparison - Repeated Holdout Evaluation",
    )

    left = 12 * mm
    right = width - 12 * mm
    line_y = 12 * mm
    canvas.setStrokeColor(colors.HexColor("#C9D5DE"))
    canvas.setLineWidth(0.45)
    canvas.line(left, line_y, right, line_y)
    canvas.setFont(FONT, 7)
    canvas.setFillColor(MUTED)
    canvas.drawString(
        left,
        7.5 * mm,
        (
            f"Same {n_splits} patient splits | "
            f"{train_n} training / {test_n} test subjects"
        ),
    )
    canvas.drawRightString(right, 7.5 * mm, f"Page {document.page}")
    canvas.restoreState()


def _model_page(
    model_name: str,
    lookup: dict[tuple[str, str, str, str], dict[str, Any]],
    model_summary: dict[str, Any],
    style: dict[str, ParagraphStyle],
) -> list[Any]:
    trained = [
        row
        for row in lookup.values()
        if row["model"] == model_name
        and row["result_source"] == "trained"
    ]
    reached = sum(
        float(row["metrics"]["roc_auc"]["mean"]) >= 0.80
        for row in trained
    )
    summary = (
        f"Across the {len(trained)} actually trained tasks, {model_name} "
        f"achieved a mean AUC of "
        f"{float(model_summary['roc_auc_mean']):.3f} "
        f"(range {float(model_summary['roc_auc_min']):.3f}-"
        f"{float(model_summary['roc_auc_max']):.3f}); "
        f"{reached}/{len(trained)} tasks reached AUC >= 0.80. "
        f"{MODEL_INSIGHTS[model_name]}"
    )
    return [
        Paragraph(f"{model_name}: Detailed Results", style["title"]),
        Paragraph(summary, style["body"]),
        Paragraph(
            "ROC-AUC Overview (mean +/- SD across 50 splits)",
            style["section"],
        ),
        _overview_table(model_name, lookup, style),
        Spacer(1, 1.2 * mm),
        Paragraph(
            "Original: Detailed Metrics (mean +/- SD across 50 splits)",
            style["section"],
        ),
        _detail_table(model_name, "Original", lookup, style),
        Spacer(1, 0.8 * mm),
        Paragraph(
            "Mask: Detailed Metrics (mean +/- SD across 50 splits)",
            style["section"],
        ),
        _detail_table(model_name, "Mask", lookup, style),
        Spacer(1, 1 * mm),
        Paragraph(
            "Where no Mask rule was defined, the corresponding Original "
            "result is displayed for completeness; it was not retrained "
            "and was excluded from the 18-task overall model summary.",
            style["note"],
        ),
    ]


def build_pdf(results_root: Path) -> Path:
    results_root = Path(results_root)
    summary_path = results_root / "summary.json"
    output_pdf = results_root / "model_auc_comparison.pdf"

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    task_rows = payload["task_summaries"]
    model_summaries = payload["model_summaries"]
    n_splits = int(payload["experiment"]["split_count"])
    train_n, test_n = _split_sizes(results_root)
    config = load_config()
    lookup = {
        (
            row["model"],
            row["variant"],
            row["network"],
            row["feature_mode"],
        ): row
        for row in task_rows
    }
    models = {row["model"]: row for row in model_summaries}
    style = styles()
    width, height = landscape(A4)
    document = BaseDocTemplate(
        str(output_pdf),
        pagesize=landscape(A4),
        leftMargin=12 * mm,
        rightMargin=12 * mm,
        topMargin=16 * mm,
        bottomMargin=15 * mm,
        title="dNCR Classification Model Comparison",
        subject="Comparison of LinearSVC, LDA, RF, DT, and XGBoost",
        author="dNCR classification project",
    )
    frame = Frame(
        document.leftMargin,
        document.bottomMargin,
        width - document.leftMargin - document.rightMargin,
        height - document.topMargin - document.bottomMargin,
        id="content",
    )
    document.addPageTemplates(
        [
            PageTemplate(
                id="main",
                frames=[frame],
                onPage=lambda canvas, doc: _draw_page_chrome(
                    canvas,
                    doc,
                    n_splits=n_splits,
                    train_n=train_n,
                    test_n=test_n,
                ),
            )
        ]
    )

    story: list[Any] = [
        Paragraph("Evaluation Design and Model Settings", style["title"]),
        Paragraph("Evaluation design", style["section"]),
        _method_table(n_splits, train_n, test_n, config, style),
        Spacer(1, 2.5 * mm),
        Paragraph("Fixed model parameters", style["section"]),
        _model_parameter_table(config, style),
        Spacer(1, 2.5 * mm),
        Paragraph("Overall model comparison", style["section"]),
        Paragraph(_overall_sentence(model_summaries), style["body"]),
        Paragraph(
            "All models used the same outer patient splits and the same "
            "K-selection rule, while selecting K independently within the "
            "training data. Outer test subjects were not used for "
            "preprocessing or K selection.",
            style["note"],
        ),
    ]
    for model_name in MODEL_ORDER:
        story.append(PageBreak())
        story.extend(
            _model_page(
                model_name,
                lookup,
                models[model_name],
                style,
            )
        )

    document.build(story)
    return output_pdf
