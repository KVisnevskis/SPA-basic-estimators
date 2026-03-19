from __future__ import annotations

import json
import pickle
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score

from spa_basic_estimators.utils.config import load_yaml, resolve_path
from spa_basic_estimators.utils.data_loader import DataConfig, load_data_config, load_runs
from spa_basic_estimators.utils.splits import HDF5_KEY_COLUMN, SPLIT_COLUMN

DEFAULT_ALPHA_GRID = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
SAMPLE_INDEX_COLUMN = "sample_index"


@dataclass(frozen=True)
class PressureRidgeLinearConfig:
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
class PressureRidgeLinearResult:
    config: PressureRidgeLinearConfig
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


def load_pressure_ridge_linear_config(path: str | Path) -> PressureRidgeLinearConfig:
    config_path = Path(path).resolve()
    raw = load_yaml(config_path)
    config_dir = config_path.parent
    project_root = config_dir.parent.parent

    return PressureRidgeLinearConfig(
        config_path=config_path,
        name=str(raw.get("name", "pressure_ridge_linear")),
        alpha_grid=[float(alpha) for alpha in raw.get("alpha_grid", DEFAULT_ALPHA_GRID)],
        fit_intercept=bool(raw.get("fit_intercept", True)),
        output_dir=_resolve_config_reference(
            config_dir,
            project_root,
            raw.get("output_dir", "outputs/pressure_ridge_linear"),
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


def train_pressure_ridge_linear(
    runs: Mapping[str, pd.DataFrame],
    data_config: DataConfig,
    estimator_config: PressureRidgeLinearConfig,
) -> PressureRidgeLinearResult:
    dataset = build_pressure_only_dataset(runs, data_config)

    best_alpha: float | None = None
    best_rmse: float | None = None
    search_rows: list[dict[str, float]] = []

    for alpha in estimator_config.alpha_grid:
        model = Ridge(alpha=alpha, fit_intercept=estimator_config.fit_intercept)
        model.fit(dataset.train.X, dataset.train.y)

        val_predictions = model.predict(dataset.val.X)
        val_metrics = _compute_regression_metrics(dataset.val.y, val_predictions)
        search_rows.append({"alpha": float(alpha), **val_metrics})

        if best_rmse is None or val_metrics["rmse"] < best_rmse:
            best_alpha = float(alpha)
            best_rmse = float(val_metrics["rmse"])

    if best_alpha is None:
        raise ValueError("Alpha grid is empty; cannot train pressure-only ridge model")

    final_model = Ridge(alpha=best_alpha, fit_intercept=estimator_config.fit_intercept)
    final_model.fit(dataset.train.X, dataset.train.y)

    validation_predictions = final_model.predict(dataset.val.X)
    held_out_predictions = final_model.predict(dataset.held_out.X)

    validation_metrics = _compute_regression_metrics(dataset.val.y, validation_predictions)
    held_out_metrics = _compute_regression_metrics(dataset.held_out.y, held_out_predictions)
    validation_search = pd.DataFrame(search_rows)

    validation_table = _build_prediction_table(
        dataset.val,
        validation_predictions,
        dataset.target_column,
    )
    held_out_table = _build_prediction_table(
        dataset.held_out,
        held_out_predictions,
        dataset.target_column,
    )
    coefficient_table = pd.DataFrame(
        {
            "feature": dataset.feature_columns,
            "coefficient": np.ravel(final_model.coef_),
        }
    )

    artifact_dir = estimator_config.output_dir
    _save_pressure_ridge_linear_artifacts(
        artifact_dir=artifact_dir,
        estimator_config=estimator_config,
        data_config=data_config,
        model=final_model,
        validation_search=validation_search,
        validation_metrics=validation_metrics,
        held_out_metrics=held_out_metrics,
        validation_predictions=validation_table,
        held_out_predictions=held_out_table,
        coefficient_table=coefficient_table,
        selected_alpha=best_alpha,
    )

    return PressureRidgeLinearResult(
        config=estimator_config,
        feature_columns=dataset.feature_columns,
        selected_alpha=best_alpha,
        model=final_model,
        validation_search=validation_search,
        validation_metrics=validation_metrics,
        held_out_metrics=held_out_metrics,
        validation_predictions=validation_table,
        held_out_predictions=held_out_table,
        coefficient_table=coefficient_table,
        artifact_dir=artifact_dir,
    )


def run_pressure_ridge_linear(
    data_config_path: str | Path = "configs/data.yaml",
    model_config_path: str | Path = "configs/models/pressure_ridge_linear.yaml",
) -> PressureRidgeLinearResult:
    data_config = load_data_config(data_config_path)
    estimator_config = load_pressure_ridge_linear_config(model_config_path)
    runs = load_runs(data_config)
    return train_pressure_ridge_linear(runs, data_config, estimator_config)


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


def _build_prediction_table(
    split_dataset: SplitDesignMatrix,
    predictions: np.ndarray,
    target_column: str,
) -> pd.DataFrame:
    table = split_dataset.metadata.copy()
    table[target_column] = split_dataset.y
    table["prediction"] = np.asarray(predictions, dtype=float)
    table["error"] = table["prediction"] - table[target_column]
    return table


def _compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
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


def _save_pressure_ridge_linear_artifacts(
    artifact_dir: Path,
    estimator_config: PressureRidgeLinearConfig,
    data_config: DataConfig,
    model: Ridge,
    validation_search: pd.DataFrame,
    validation_metrics: dict[str, float],
    held_out_metrics: dict[str, float],
    validation_predictions: pd.DataFrame,
    held_out_predictions: pd.DataFrame,
    coefficient_table: pd.DataFrame,
    selected_alpha: float,
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)

    with (artifact_dir / "ridge_model.pkl").open("wb") as handle:
        pickle.dump(model, handle)
    obsolete_scaler_path = artifact_dir / "input_scaler.pkl"
    if obsolete_scaler_path.exists():
        obsolete_scaler_path.unlink()

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
                "artifacts_saved": [
                    "ridge_model.pkl",
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
