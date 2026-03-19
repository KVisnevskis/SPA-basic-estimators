from __future__ import annotations

import json
import pickle
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score

from spa_basic_estimators.utils.config import load_yaml, resolve_path
from spa_basic_estimators.utils.data_loader import DataConfig
from spa_basic_estimators.utils.splits import HDF5_KEY_COLUMN, SPLIT_COLUMN

DEFAULT_ALPHA_GRID = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
SAMPLE_INDEX_COLUMN = "sample_index"


@dataclass(frozen=True)
class PressureRidgeConfig:
    config_path: Path
    name: str
    alpha_grid: list[float]
    fit_intercept: bool
    output_dir: Path


@dataclass(frozen=True)
class SplitDesignMatrix:
    X: np.ndarray
    y: np.ndarray
    metadata: pd.DataFrame


@dataclass(frozen=True)
class PressureOnlyDataset:
    train: SplitDesignMatrix
    val: SplitDesignMatrix
    held_out: SplitDesignMatrix
    feature_columns: list[str]
    target_column: str


@dataclass(frozen=True)
class PressureRidgeResult:
    config: PressureRidgeConfig
    feature_columns: list[str]
    selected_alpha: float
    model: Ridge
    validation_search: pd.DataFrame
    validation_metrics: dict[str, float]
    held_out_metrics: dict[str, float]
    validation_predictions: pd.DataFrame
    held_out_predictions: pd.DataFrame
    coefficient_table: pd.DataFrame
    artifact_dir: Path


def load_pressure_ridge_config(
    path: str | Path,
    *,
    default_name: str,
    default_output_dir: str,
) -> PressureRidgeConfig:
    config_path = Path(path).resolve()
    raw = load_yaml(config_path)
    config_dir = config_path.parent
    project_root = config_dir.parent.parent

    return PressureRidgeConfig(
        config_path=config_path,
        name=str(raw.get("name", default_name)),
        alpha_grid=[float(alpha) for alpha in raw.get("alpha_grid", DEFAULT_ALPHA_GRID)],
        fit_intercept=bool(raw.get("fit_intercept", True)),
        output_dir=_resolve_config_reference(
            config_dir,
            project_root,
            raw.get("output_dir", default_output_dir),
        ),
    )


def build_pressure_only_dataset(
    runs: Mapping[str, pd.DataFrame],
    data_config: DataConfig,
) -> PressureOnlyDataset:
    feature_columns = list(data_config.schema.pressure_columns)
    if not feature_columns:
        raise ValueError("Pressure-only ridge requires at least one configured pressure column")

    return PressureOnlyDataset(
        train=_build_split_design_matrix("train", runs, data_config, feature_columns),
        val=_build_split_design_matrix("val", runs, data_config, feature_columns),
        held_out=_build_split_design_matrix("held_out", runs, data_config, feature_columns),
        feature_columns=feature_columns,
        target_column=data_config.schema.target_column,
    )


def build_prediction_table(
    split_dataset: SplitDesignMatrix,
    predictions: np.ndarray,
    target_column: str,
) -> pd.DataFrame:
    table = split_dataset.metadata.copy()
    table[target_column] = split_dataset.y
    table["prediction"] = np.asarray(predictions, dtype=float)
    table["error"] = table["prediction"] - table[target_column]
    return table


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true_array = np.asarray(y_true, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)
    error = y_pred_array - y_true_array

    rmse = float(np.sqrt(np.mean(np.square(error))))
    mae = float(mean_absolute_error(y_true_array, y_pred_array))
    bias = float(np.mean(error))
    r2 = float(r2_score(y_true_array, y_pred_array))

    if len(y_true_array) < 2 or np.std(y_true_array) == 0.0 or np.std(y_pred_array) == 0.0:
        pearson_r = float("nan")
    else:
        pearson_r = float(np.corrcoef(y_true_array, y_pred_array)[0, 1])

    return {
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
        "r2": r2,
        "pearson_r": pearson_r,
    }


def save_pressure_ridge_artifacts(
    *,
    artifact_dir: Path,
    estimator_config: PressureRidgeConfig,
    data_config: DataConfig,
    model: Ridge,
    validation_search: pd.DataFrame,
    validation_metrics: dict[str, float],
    held_out_metrics: dict[str, float],
    validation_predictions: pd.DataFrame,
    held_out_predictions: pd.DataFrame,
    coefficient_table: pd.DataFrame,
    selected_alpha: float,
    extra_pickled_artifacts: Mapping[str, Any] | None = None,
    obsolete_artifacts: Iterable[str] | None = None,
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)

    with (artifact_dir / "ridge_model.pkl").open("wb") as handle:
        pickle.dump(model, handle)

    for artifact_name in obsolete_artifacts or []:
        artifact_path = artifact_dir / artifact_name
        if artifact_path.exists():
            artifact_path.unlink()

    saved_artifact_names = ["ridge_model.pkl"]
    for artifact_name, artifact in (extra_pickled_artifacts or {}).items():
        with (artifact_dir / artifact_name).open("wb") as handle:
            pickle.dump(artifact, handle)
        saved_artifact_names.append(artifact_name)

    validation_search.to_csv(artifact_dir / "validation_search.csv", index=False)
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
    (artifact_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "estimator_name": estimator_config.name,
                "selected_alpha": float(selected_alpha),
                "fit_intercept": estimator_config.fit_intercept,
                "feature_columns": list(coefficient_table["feature"]),
                "data_config_path": str(data_config.config_path),
                "model_config_path": str(estimator_config.config_path),
                "artifacts_saved": saved_artifact_names
                + [
                    "validation_search.csv",
                    "validation_predictions.csv",
                    "held_out_predictions.csv",
                    "coefficient_table.csv",
                    "validation_metrics.json",
                    "held_out_metrics.json",
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    shutil.copyfile(data_config.config_path, artifact_dir / data_config.config_path.name)
    shutil.copyfile(estimator_config.config_path, artifact_dir / estimator_config.config_path.name)


def _build_split_design_matrix(
    split_name: str,
    runs: Mapping[str, pd.DataFrame],
    data_config: DataConfig,
    feature_columns: list[str],
) -> SplitDesignMatrix:
    split_frames: list[pd.DataFrame] = []
    for run_id in data_config.splits[split_name]:
        if run_id not in runs:
            raise KeyError(f"Run '{run_id}' is missing from the loaded run dictionary")

        frame = runs[run_id].copy()
        frame[SAMPLE_INDEX_COLUMN] = frame.index.to_numpy()
        split_frames.append(frame)

    if not split_frames:
        raise ValueError(f"Split '{split_name}' does not contain any runs")

    combined = pd.concat(split_frames, ignore_index=True)
    metadata_columns = [
        data_config.schema.run_id_column,
        data_config.schema.time_column,
        SPLIT_COLUMN,
        HDF5_KEY_COLUMN,
        SAMPLE_INDEX_COLUMN,
    ]

    return SplitDesignMatrix(
        X=combined[feature_columns].to_numpy(dtype=float),
        y=combined[data_config.schema.target_column].to_numpy(dtype=float),
        metadata=combined[metadata_columns].copy(),
    )


def _normalise_json_floats(payload: dict[str, float]) -> dict[str, float | None]:
    normalised: dict[str, float | None] = {}
    for key, value in payload.items():
        normalised[key] = None if np.isnan(value) else float(value)
    return normalised


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
