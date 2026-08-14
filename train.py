from __future__ import annotations

import argparse
from pathlib import Path

from configs.settings import PROJECT_ROOT, load_config
from model.training import train_all
from utils.report_pdf import build_pdf


def _project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run five dNCR classifiers on the configured fixed patient "
            "splits and generate the complete results tree."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "experiment.yml",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=PROJECT_ROOT / "results",
    )
    args = parser.parse_args()

    config = load_config(_project_path(args.config))
    result = train_all(
        config=config,
        results_root=_project_path(args.results),
    )
    split_info = result["split_info"]
    print(
        "完成：5个模型 × Original/Mask × 4个网络 × 3种特征，"
        f"每项{split_info['n_splits']}次重复；"
        f"每次{split_info['train_n']}例训练、"
        f"{split_info['test_n']}例测试。"
    )
    for row in result["model_summaries"]:
        print(
            f"{row['model']}: {row['task_count']}项平均AUC="
            f"{row['roc_auc_mean']:.3f}, "
            f"范围={row['roc_auc_min']:.3f}-"
            f"{row['roc_auc_max']:.3f}"
        )
    for network, prior in result["topology_priors"].items():
        print(
            f"在线拓扑先验 {network}: {prior['feature_name']}, "
            f"t={prior['t_statistic']:.3f}, "
            f"P={prior['p_value']:.4f}, "
            f"自动索引={prior['feature_index']}"
        )
    execution = result["execution"]
    print(
        f"自动使用{execution['parallel_jobs']}个并行进程；"
        f"内层交叉验证拟合"
        f"{execution['inner_cv_model_fits']}个候选模型，"
        f"最终训练{execution['trained_model_fits']}个模型，"
        f"复用Original结果{execution['reused_original_results']}项。"
    )
    print(
        "各模型独立K选择规则在"
        f"{execution['selected_by_inner_cv']}个实际任务划分中"
        "选择了非参考参数。"
    )
    print(f"结果目录：{result['results_root']}")
    print(f"完整JSON：{Path(result['results_root']) / 'summary.json'}")
    pdf_path = build_pdf(Path(result["results_root"]))
    print(f"PDF报告：{pdf_path}")


if __name__ == "__main__":
    main()
