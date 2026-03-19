from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from spa_basic_estimators.estimators.pressure_ridge_common import (
    PressureOnlyDataset,
    PressureRidgeConfig,
    PressureRidgeResult,
    build_prediction_table,
    build_pressure_only_dataset as build_pressure_only_dataset_common,
    compute_regression_metrics,
    load_pressure_ridge_config,
    predict_all_datasets,
    save_pressure_ridge_artifacts,
)
from spa_basic_estimators.utils.data_loader import DataConfig, load_data_config, load_runs

PressureRidgeLinearConfig = PressureRidgeConfig
PressureRidgeLinearResult = PressureRidgeResult


def load_pressure_ridge_linear_config(path: str | Path) -> PressureRidgeLinearConfig:
    common_config = load_pressure_ridge_config(
        path,
        default_name="pressure_ridge_linear",
        default_output_dir="outputs/pressure_ridge_linear",
    )
    return PressureRidgeLinearConfig(**common_config.__dict__)


def build_pressure_only_dataset(
    runs: Mapping[str, pd.DataFrame],
    data_config: DataConfig,
) -> PressureOnlyDataset:
    return build_pressure_only_dataset_common(runs, data_config)


def train_pressure_ridge_linear(
    runs: Mapping[str, pd.DataFrame],
    data_config: DataConfig,
    estimator_config: PressureRidgeLinearConfig,
) -> PressureRidgeLinearResult:
    dataset = build_pressure_only_dataset_common(runs, data_config)

    best_alpha: float | None = None
    best_rmse: float | None = None
    search_rows: list[dict[str, float]] = []

    for alpha in estimator_config.alpha_grid:
        model = Ridge(alpha=alpha, fit_intercept=estimator_config.fit_intercept)
        model.fit(dataset.train.X, dataset.train.y)

        val_predictions = model.predict(dataset.val.X)
        val_metrics = compute_regression_metrics(dataset.val.y, val_predictions)
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

    validation_metrics = compute_regression_metrics(dataset.val.y, validation_predictions)
    held_out_metrics = compute_regression_metrics(dataset.held_out.y, held_out_predictions)
    validation_search = pd.DataFrame(search_rows)

    validation_table = build_prediction_table(
        dataset.val,
        validation_predictions,
        dataset.target_column,
    )
    held_out_table = build_prediction_table(
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
    all_dataset_predictions_path = predict_all_datasets(
        runs=runs,
        data_config=data_config,
        artifact_dir=artifact_dir,
        input_columns=dataset.feature_columns,
        predict_fn=lambda frame: final_model.predict(
            frame[dataset.feature_columns].to_numpy(dtype=float)
        ),
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
        additional_artifact_names=[all_dataset_predictions_path.name],
        obsolete_artifacts=["input_scaler.pkl", "polynomial_transformer.pkl"],
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
        all_dataset_predictions_path=all_dataset_predictions_path,
    )


def run_pressure_ridge_linear(
    data_config_path: str | Path = "configs/data.yaml",
    model_config_path: str | Path = "configs/models/pressure_ridge_linear.yaml",
) -> PressureRidgeLinearResult:
    data_config = load_data_config(data_config_path)
    estimator_config = load_pressure_ridge_linear_config(model_config_path)
    runs = load_runs(data_config)
    return train_pressure_ridge_linear(runs, data_config, estimator_config)
