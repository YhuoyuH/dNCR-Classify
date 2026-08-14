from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load experiment YAML and resolve project-relative paths."""
    config_path = (
        _resolve_project_path(path)
        if path is not None
        else PROJECT_ROOT / "configs" / "experiment.yml"
    )
    with config_path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Invalid configuration: {config_path}")

    config = loaded
    config["data_root"] = str(
        _resolve_project_path(config.get("data_root", "data"))
    )
    return config
