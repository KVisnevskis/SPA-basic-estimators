from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from spa_basic_estimators.estimators.pressure_ridge_common import (
    SAMPLE_INDEX_COLUMN,
    PressureOnlyDataset,
    PressureRidgeConfig,
    PressureRidgeResult,
    SplitDesignMatrix,
    build_prediction_table,
    compute_regression_metrics,
    inverse_scale_array,
    load_pressure_ridge_config,
    save_pressure_ridge_artifacts,
)
from spa_basic_estimators.utils.config import load_yaml
from spa_basic_estimators.utils.data_loader import (
    DataConfig,
    load_all_runs,
    load_data_config,
    load_runs,
    load_scaler_bounds,
)
from spa_basic_estimators.utils.splits import HDF5_KEY_COLUMN, SPLIT_COLUMN

DEFAULT_LAG_GRID = [1, 2, 3, 5, 10, 20]


@dataclass(frozen=True)
class LaggedPressureAccelRidgeConfig(PressureRidgeConfig):
    lag_grid: list[int]


@dataclass(frozen=True)
class LaggedPressureAccelRidgeResult(PressureRidgeResult):
    selected_lag: int
    raw_feature_columns: list[str]


def load_lagged_pressure_accel_ridge_config(
    path: str | Path,
) -> LaggedPressureAccelRidgeConfig:
    common_config = load_pressure_ridge_config(
        path,
        default_name="lagged_pressure_accel_ridge",
        default_output_dir="outputs/lagged_pressure_accel_ridge",
    )
    raw = load_yaml(Path(path).resolve())
    lag_grid = [int(value) for value in raw.get("lag_grid", DEFAULT_LAG_GRID)]
    lag_grid = list(dict.fromkeys(lag_grid))
    if not lag_grid:
        raise ValueError("Lagged pressure+accel ridge requires at least one lag length")
    if any(lag_length < 1 for lag_length in lag_grid):
        raise ValueError("All lag lengths must be positive integers")

    return LaggedPressureAccelRidgeConfig(
        config_path=common_config.config_path,
        name=common_config.name,
        alpha_grid=common_config.alpha_grid,
        fit_intercept=common_config.fit_intercept,
        output_dir=common_config.output_dir,
        lag_grid=lag_grid,
    )


def build_lagged_pressure_accel_dataset(
    runs: Mapping[str, pd.DataFrame],
    data_config: DataConfig,
    lag_length: int,
) -> PressureOnlyDataset:
    raw_feature_columns = _raw_feature_columns(data_config)
    feature_columns = _lagged_feature_names(raw_feature_columns, lag_length)

    return PressureOnlyDataset(
        train=_build_lagged_split_design_matrix(
            "train",
            runs,
            data_config,
            raw_feature_columns,
            feature_columns,
            lag_length,
        ),
        val=_build_lagged_split_design_matrix(
            "val",
            runs,
            data_config,
            raw_feature_columns,
            feature_columns,
            lag_length,
        ),
        held_out=_build_lagged_split_design_matrix(
            "held_out",
            runs,
            data_config,
            raw_feature_columns,
            feature_columns,
            lag_length,
        ),
        feature_columns=feature_columns,
        target_column=data_config.schema.target_column,
    )


def train_lagged_pressure_accel_ridge(
    runs: Mapping[str, pd.DataFrame],
    data_config: DataConfig,
    estimator_config: LaggedPressureAccelRidgeConfig,
) -> LaggedPressureAccelRidgeResult:
    raw_feature_columns = _raw_feature_columns(data_config)
    datasets_by_lag: dict[int, PressureOnlyDataset] = {}

    best_alpha: float | None = None
    best_lag: int | None = None
    best_rmse: float | None = None
    search_rows: list[dict[str, float | int]] = []

    for lag_length in estimator_config.lag_grid:
        dataset = build_lagged_pressure_accel_dataset(runs, data_config, lag_length)
        datasets_by_lag[lag_length] = dataset

        for alpha in estimator_config.alpha_grid:
            model = Ridge(alpha=alpha, fit_intercept=estimator_config.fit_intercept)
            model.fit(dataset.train.X, dataset.train.y)

            val_predictions = model.predict(dataset.val.X)
            val_metrics = compute_regression_metrics(dataset.val.y, val_predictions)
            search_rows.append(
                {
                    "lag_length": int(lag_length),
                    "alpha": float(alpha),
                    **val_metrics,
                }
            )

            if best_rmse is None or val_metrics["rmse"] < best_rmse:
                best_alpha = float(alpha)
                best_lag = int(lag_length)
                best_rmse = float(val_metrics["rmse"])

    if best_alpha is None or best_lag is None:
        raise ValueError("Lagged pressure+accel ridge could not select a best lag/alpha pair")

    final_dataset = datasets_by_lag[best_lag]
    final_model = Ridge(alpha=best_alpha, fit_intercept=estimator_config.fit_intercept)
    final_model.fit(final_dataset.train.X, final_dataset.train.y)

    validation_predictions = final_model.predict(final_dataset.val.X)
    held_out_predictions = final_model.predict(final_dataset.held_out.X)

    validation_metrics = compute_regression_metrics(final_dataset.val.y, validation_predictions)
    held_out_metrics = compute_regression_metrics(final_dataset.held_out.y, held_out_predictions)
    validation_search = pd.DataFrame(search_rows)

    validation_table = build_prediction_table(
        final_dataset.val,
        validation_predictions,
        final_dataset.target_column,
    )
    held_out_table = build_prediction_table(
        final_dataset.held_out,
        held_out_predictions,
        final_dataset.target_column,
    )
    coefficient_table = pd.DataFrame(
        {
            "feature": final_dataset.feature_columns,
            "source_feature": [
                _source_feature_from_lagged_feature(feature_name)
                for feature_name in final_dataset.feature_columns
            ],
            "lag": [_lag_from_feature_name(feature_name) for feature_name in final_dataset.feature_columns],
            "feature_group": [
                _feature_group_from_base_feature(
                    _source_feature_from_lagged_feature(feature_name),
                    data_config,
                )
                for feature_name in final_dataset.feature_columns
            ],
            "coefficient": np.ravel(final_model.coef_),
        }
    )
    coefficient_table["abs_coefficient"] = coefficient_table["coefficient"].abs()

    artifact_dir = estimator_config.output_dir
    all_dataset_predictions_path = _predict_all_datasets_lagged(
        data_config=data_config,
        artifact_dir=artifact_dir,
        raw_feature_columns=raw_feature_columns,
        lag_length=best_lag,
        model=final_model,
    )
    save_pressure_ridge_artifacts(
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
        extra_summary_fields={
            "selected_lag": int(best_lag),
            "raw_feature_columns": raw_feature_columns,
        },
        additional_artifact_names=[all_dataset_predictions_path.name],
        obsolete_artifacts=["input_scaler.pkl", "polynomial_transformer.pkl"],
    )

    return LaggedPressureAccelRidgeResult(
        config=estimator_config,
        feature_columns=final_dataset.feature_columns,
        selected_alpha=best_alpha,
        model=final_model,
        validation_search=validation_search,
        validation_metrics=validation_metrics,
        held_out_metrics=held_out_metrics,
        validation_predictions=validation_table,
        held_out_predictions=held_out_table,
        coefficient_table=coefficient_table,
        artifact_dir=artifact_dir,
        all_dataset_predictions_path=all_dataset_predictions_path,
        selected_lag=best_lag,
        raw_feature_columns=raw_feature_columns,
    )


def run_lagged_pressure_accel_ridge(
    data_config_path: str | Path = "configs/data.yaml",
    model_config_path: str | Path = "configs/models/lagged_pressure_accel_ridge.yaml",
) -> LaggedPressureAccelRidgeResult:
    data_config = load_data_config(data_config_path)
    estimator_config = load_lagged_pressure_accel_ridge_config(model_config_path)
    runs = load_runs(data_config)
    return train_lagged_pressure_accel_ridge(runs, data_config, estimator_config)


def _build_lagged_split_design_matrix(
    split_name: str,
    runs: Mapping[str, pd.DataFrame],
    data_config: DataConfig,
    raw_feature_columns: list[str],
    feature_columns: list[str],
    lag_length: int,
) -> SplitDesignMatrix:
    matrices: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    metadata_frames: list[pd.DataFrame] = []

    for run_id in data_config.splits[split_name]:
        if run_id not in runs:
            raise KeyError(f"Run '{run_id}' is missing from the loaded run dictionary")

        frame = runs[run_id]
        run_matrix, run_targets, run_metadata = _build_lagged_run_design_matrix(
            frame,
            data_config,
            raw_feature_columns,
            lag_length,
        )
        matrices.append(run_matrix)
        targets.append(run_targets)
        metadata_frames.append(run_metadata)

    if not matrices:
        raise ValueError(f"Split '{split_name}' does not contain any runs")

    return SplitDesignMatrix(
        X=np.vstack(matrices),
        y=np.concatenate(targets),
        metadata=pd.concat(metadata_frames, ignore_index=True),
    )


def _build_lagged_run_design_matrix(
    frame: pd.DataFrame,
    data_config: DataConfig,
    raw_feature_columns: list[str],
    lag_length: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    if lag_length < 1:
        raise ValueError("Lag length must be a positive integer")
    if len(frame) < lag_length:
        run_name = str(frame[data_config.schema.run_id_column].iloc[0])
        raise ValueError(
            f"Run '{run_name}' has {len(frame)} rows, which is shorter than lag length {lag_length}"
        )

    working_frame = frame.copy()
    working_frame[SAMPLE_INDEX_COLUMN] = working_frame.index.to_numpy()

    values = working_frame[raw_feature_columns].to_numpy(dtype=float)
    num_rows = len(working_frame)
    lagged_blocks: list[np.ndarray] = []
    for lag in range(lag_length):
        lagged_blocks.append(values[lag_length - 1 - lag : num_rows - lag, :])

    matrix = np.hstack(lagged_blocks)
    targets = working_frame[data_config.schema.target_column].to_numpy(dtype=float)[lag_length - 1 :]
    metadata_columns = [
        data_config.schema.run_id_column,
        data_config.schema.time_column,
        SPLIT_COLUMN,
        HDF5_KEY_COLUMN,
        SAMPLE_INDEX_COLUMN,
    ]
    metadata = (
        working_frame.iloc[lag_length - 1 :][metadata_columns].copy().reset_index(drop=True)
    )
    return matrix, targets, metadata


def _predict_all_datasets_lagged(
    *,
    data_config: DataConfig,
    artifact_dir: Path,
    raw_feature_columns: list[str],
    lag_length: int,
    model: Ridge,
) -> Path:
    all_runs = load_all_runs(data_config)
    scaler_bounds = load_scaler_bounds(data_config)
    target_column = data_config.schema.target_column
    missing_bounds = [
        column for column in [*raw_feature_columns, target_column] if column not in scaler_bounds
    ]
    if missing_bounds:
        raise KeyError(
            "Missing scaler bounds for required output columns: " + ", ".join(missing_bounds)
        )

    artifact_dir.mkdir(parents=True, exist_ok=True)
    output_path = artifact_dir / "all_dataset_predictions.h5"
    meta_rows: list[dict[str, int | str]] = []

    with pd.HDFStore(output_path, mode="w") as store:
        for run_id, frame in all_runs.items():
            run_matrix, _, run_metadata = _build_lagged_run_design_matrix(
                frame,
                data_config,
                raw_feature_columns,
                lag_length,
            )
            predictions_scaled = model.predict(run_matrix)
            current_frame = frame.iloc[lag_length - 1 :].reset_index(drop=True)

            prediction_frame = pd.DataFrame(
                {
                    data_config.schema.run_id_column: run_metadata[
                        data_config.schema.run_id_column
                    ].astype(str),
                    data_config.schema.time_column: run_metadata[
                        data_config.schema.time_column
                    ].to_numpy(dtype=float),
                    "split": run_metadata[SPLIT_COLUMN].astype(str),
                    SAMPLE_INDEX_COLUMN: run_metadata[SAMPLE_INDEX_COLUMN].to_numpy(dtype=int),
                }
            )

            for column in raw_feature_columns:
                prediction_frame[column] = inverse_scale_array(
                    current_frame[column].to_numpy(dtype=float),
                    scaler_bounds[column],
                )

            target_true = inverse_scale_array(
                current_frame[target_column].to_numpy(dtype=float),
                scaler_bounds[target_column],
            )
            target_prediction = inverse_scale_array(
                predictions_scaled,
                scaler_bounds[target_column],
            )
            prediction_frame[f"{target_column}_true"] = target_true
            prediction_frame[f"{target_column}_prediction"] = target_prediction
            prediction_frame[f"{target_column}_error"] = target_prediction - target_true

            key = f"/predictions/{run_id}"
            store.put(key, prediction_frame, format="fixed")
            meta_rows.append(
                {
                    "run_id": run_id,
                    "split": str(run_metadata[SPLIT_COLUMN].iloc[0]),
                    "source_hdf5_key": str(run_metadata[HDF5_KEY_COLUMN].iloc[0]),
                    "prediction_hdf5_key": key,
                    "rows_saved": len(prediction_frame),
                    "lag_length": int(lag_length),
                    "trimmed_initial_rows": int(lag_length - 1),
                }
            )

        store.put("/meta/runs", pd.DataFrame(meta_rows), format="fixed")

    return output_path


def _raw_feature_columns(data_config: DataConfig) -> list[str]:
    pressure_columns = list(data_config.schema.pressure_columns)
    accel_columns = list(data_config.schema.accel_columns)
    if not pressure_columns:
        raise ValueError("Lagged pressure+accel ridge requires at least one pressure column")
    if not accel_columns:
        raise ValueError(
            "Lagged pressure+accel ridge requires at least one accelerometer column"
        )
    return pressure_columns + accel_columns


def _lagged_feature_names(base_feature_columns: list[str], lag_length: int) -> list[str]:
    if lag_length < 1:
        raise ValueError("Lag length must be a positive integer")

    feature_names: list[str] = []
    for lag in range(lag_length):
        for column in base_feature_columns:
            feature_names.append(f"{column}_lag_{lag}")
    return feature_names


def _source_feature_from_lagged_feature(feature_name: str) -> str:
    return feature_name.rsplit("_lag_", 1)[0]


def _lag_from_feature_name(feature_name: str) -> int:
    return int(feature_name.rsplit("_lag_", 1)[1])


def _feature_group_from_base_feature(base_feature: str, data_config: DataConfig) -> str:
    if base_feature in data_config.schema.pressure_columns:
        return "pressure"
    if base_feature in data_config.schema.accel_columns:
        return "accel"
    return "other"
