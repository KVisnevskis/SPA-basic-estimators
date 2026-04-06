from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from spa_basic_estimators.estimators.lagged_ridge_common import (
    DEFAULT_LAG_GRID,
    _feature_group_from_base_feature,
    _lag_from_feature_name,
    _predict_all_datasets_lagged,
    _source_feature_from_lagged_feature,
    accel_feature_columns,
    build_lagged_dataset_from_columns,
)
from spa_basic_estimators.estimators.pressure_ridge_common import (
    DatasetMatrices,
    RidgeModelConfig,
    RidgeTrainingResult,
    build_prediction_table,
    compute_regression_metrics,
    load_ridge_model_config,
    save_ridge_artifacts,
)
from spa_basic_estimators.utils.config import load_yaml
from spa_basic_estimators.utils.data_loader import (
    DataConfig,
    load_data_config,
    load_runs,
)


@dataclass(frozen=True)
class LaggedAccelRidgeConfig(RidgeModelConfig):
    lag_grid: list[int]


@dataclass(frozen=True)
class LaggedAccelRidgeResult(RidgeTrainingResult):
    selected_lag: int
    raw_feature_columns: list[str]


def load_lagged_accel_ridge_config(path: str | Path) -> LaggedAccelRidgeConfig:
    common_config = load_ridge_model_config(
        path,
        default_name="lagged_accel_ridge",
        default_output_dir="outputs/lagged_accel_ridge",
    )
    raw = load_yaml(Path(path).resolve())
    lag_grid = [int(value) for value in raw.get("lag_grid", DEFAULT_LAG_GRID)]
    lag_grid = list(dict.fromkeys(lag_grid))
    if not lag_grid:
        raise ValueError("Lagged accel ridge requires at least one lag length")
    if any(lag_length < 1 for lag_length in lag_grid):
        raise ValueError("All lag lengths must be positive integers")

    return LaggedAccelRidgeConfig(
        config_path=common_config.config_path,
        name=common_config.name,
        alpha_grid=common_config.alpha_grid,
        fit_intercept=common_config.fit_intercept,
        output_dir=common_config.output_dir,
        lag_grid=lag_grid,
    )


def build_lagged_accel_dataset(
    runs: Mapping[str, pd.DataFrame],
    data_config: DataConfig,
    lag_length: int,
) -> DatasetMatrices:
    raw_feature_columns = accel_feature_columns(data_config)
    return build_lagged_dataset_from_columns(runs, data_config, raw_feature_columns, lag_length)


def train_lagged_accel_ridge(
    runs: Mapping[str, pd.DataFrame],
    data_config: DataConfig,
    estimator_config: LaggedAccelRidgeConfig,
) -> LaggedAccelRidgeResult:
    raw_feature_columns = accel_feature_columns(data_config)
    datasets_by_lag: dict[int, DatasetMatrices] = {}

    best_alpha: float | None = None
    best_lag: int | None = None
    best_rmse: float | None = None
    search_rows: list[dict[str, float | int]] = []

    for lag_length in estimator_config.lag_grid:
        dataset = build_lagged_dataset_from_columns(runs, data_config, raw_feature_columns, lag_length)
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
        raise ValueError("Lagged accel ridge could not select a best lag/alpha pair")

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
    save_ridge_artifacts(
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

    return LaggedAccelRidgeResult(
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


def run_lagged_accel_ridge(
    data_config_path: str | Path = "configs/data.yaml",
    model_config_path: str | Path = "configs/models/lagged_accel_ridge.yaml",
) -> LaggedAccelRidgeResult:
    data_config = load_data_config(data_config_path)
    estimator_config = load_lagged_accel_ridge_config(model_config_path)
    runs = load_runs(data_config)
    return train_lagged_accel_ridge(runs, data_config, estimator_config)
