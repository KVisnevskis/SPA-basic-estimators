from __future__ import annotations

import json
import pickle
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from spa_basic_estimators.utils.config import load_yaml, resolve_path
from spa_basic_estimators.utils.data_loader import DataConfig


@dataclass(frozen=True)
class LinearLeastSquaresConfig:
    config_path: Path
    name: str
    fit_intercept: bool
    output_dir: Path


@dataclass(frozen=True)
class LinearLeastSquaresTrainingResult:
    config: LinearLeastSquaresConfig
    feature_columns: list[str]
    model: LinearRegression
    validation_metrics: dict[str, float]
    held_out_metrics: dict[str, float]
    validation_predictions: pd.DataFrame
    held_out_predictions: pd.DataFrame
    coefficient_table: pd.DataFrame
    artifact_dir: Path
    all_dataset_predictions_path: Path


def load_linear_least_squares_config(
    path: str | Path,
    *,
    default_name: str,
    default_output_dir: str,
) -> LinearLeastSquaresConfig:
    config_path = Path(path).resolve()
    raw = load_yaml(config_path)
    config_dir = config_path.parent
    project_root = config_dir.parent.parent

    return LinearLeastSquaresConfig(
        config_path=config_path,
        name=str(raw.get("name", default_name)),
        fit_intercept=bool(raw.get("fit_intercept", True)),
        output_dir=_resolve_config_reference(
            config_dir,
            project_root,
            raw.get("output_dir", default_output_dir),
        ),
    )


def save_linear_least_squares_artifacts(
    *,
    artifact_dir: Path,
    estimator_config: LinearLeastSquaresConfig,
    data_config: DataConfig,
    model: LinearRegression,
    validation_metrics: dict[str, float],
    held_out_metrics: dict[str, float],
    validation_predictions: pd.DataFrame,
    held_out_predictions: pd.DataFrame,
    coefficient_table: pd.DataFrame,
    extra_summary_fields: Mapping[str, Any] | None = None,
    additional_artifact_names: Iterable[str] | None = None,
    obsolete_artifacts: Iterable[str] | None = None,
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)

    with (artifact_dir / "linear_model.pkl").open("wb") as handle:
        pickle.dump(model, handle)

    for artifact_name in obsolete_artifacts or []:
        artifact_path = artifact_dir / artifact_name
        if artifact_path.exists():
            artifact_path.unlink()

    validation_predictions.to_csv(artifact_dir / "validation_predictions.csv", index=False)
    held_out_predictions.to_csv(artifact_dir / "held_out_predictions.csv", index=False)
    coefficient_table.to_csv(artifact_dir / "coefficient_table.csv", index=False)

    (artifact_dir / "validation_metrics.json").write_text(
        json.dumps(_normalise_json_floats(validation_metrics), indent=2),
        encoding="utf-8",
    )
    (artifact_dir / "held_out_metrics.json").write_text(
        json.dumps(_normalise_json_floats(held_out_metrics), indent=2),
        encoding="utf-8",
    )

    project_root = data_config.config_path.parent.parent
    summary_payload: dict[str, Any] = {
        "estimator_name": estimator_config.name,
        "model_family": "ordinary_least_squares",
        "solver_class": "sklearn.linear_model.LinearRegression",
        "fit_intercept": estimator_config.fit_intercept,
        "feature_columns": list(coefficient_table["feature"]),
        "data_config_path": _summary_path(data_config.config_path, project_root),
        "model_config_path": _summary_path(estimator_config.config_path, project_root),
        "artifacts_saved": [
            "linear_model.pkl",
            "validation_predictions.csv",
            "held_out_predictions.csv",
            "coefficient_table.csv",
            "validation_metrics.json",
            "held_out_metrics.json",
        ],
    }
    if additional_artifact_names:
        summary_payload["artifacts_saved"].extend(list(additional_artifact_names))
    if extra_summary_fields:
        summary_payload.update(dict(extra_summary_fields))

    (artifact_dir / "run_summary.json").write_text(
        json.dumps(summary_payload, indent=2),
        encoding="utf-8",
    )

    shutil.copyfile(data_config.config_path, artifact_dir / data_config.config_path.name)
    shutil.copyfile(estimator_config.config_path, artifact_dir / estimator_config.config_path.name)


def _normalise_json_floats(payload: Mapping[str, float]) -> dict[str, float | None]:
    normalised: dict[str, float | None] = {}
    for key, value in payload.items():
        normalised[key] = None if np.isnan(value) else float(value)
    return normalised


def _summary_path(path: str | Path, project_root: Path) -> str:
    resolved_path = Path(path).resolve()
    resolved_root = project_root.resolve()
    try:
        return resolved_path.relative_to(resolved_root).as_posix()
    except ValueError:
        return str(resolved_path)


def _resolve_config_reference(
    config_dir: Path,
    project_root: Path,
    raw_path: str | Path,
) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path

    config_relative = resolve_path(config_dir, path)
    project_relative = resolve_path(project_root, path)

    if config_relative.exists():
        return config_relative
    if project_relative.exists():
        return project_relative
    return project_relative
