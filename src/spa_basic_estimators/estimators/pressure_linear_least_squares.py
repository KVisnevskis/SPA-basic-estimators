from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from spa_basic_estimators.estimators.linear_least_squares_common import (
    LinearLeastSquaresConfig,
    LinearLeastSquaresTrainingResult,
    load_linear_least_squares_config,
    save_linear_least_squares_artifacts,
)
from spa_basic_estimators.estimators.pressure_ridge_common import (
    DatasetMatrices,
    build_prediction_table,
    build_pressure_only_dataset as build_pressure_only_dataset_common,
    compute_regression_metrics,
    predict_all_datasets,
)
from spa_basic_estimators.utils.data_loader import DataConfig, load_data_config, load_runs

PressureLinearLeastSquaresConfig = LinearLeastSquaresConfig
PressureLinearLeastSquaresResult = LinearLeastSquaresTrainingResult


def load_pressure_linear_least_squares_config(
    path: str | Path,
) -> PressureLinearLeastSquaresConfig:
    common_config = load_linear_least_squares_config(
        path,
        default_name="pressure_linear_least_squares",
        default_output_dir="outputs/pressure_linear_least_squares",
    )
    return PressureLinearLeastSquaresConfig(**common_config.__dict__)


def build_pressure_only_dataset(
    runs: Mapping[str, pd.DataFrame],
    data_config: DataConfig,
) -> DatasetMatrices:
    return build_pressure_only_dataset_common(runs, data_config)


def train_pressure_linear_least_squares(
    runs: Mapping[str, pd.DataFrame],
    data_config: DataConfig,
    estimator_config: PressureLinearLeastSquaresConfig,
) -> PressureLinearLeastSquaresResult:
    dataset = build_pressure_only_dataset_common(runs, data_config)

    final_model = LinearRegression(fit_intercept=estimator_config.fit_intercept)
    final_model.fit(dataset.train.X, dataset.train.y)

    validation_predictions = final_model.predict(dataset.val.X)
    held_out_predictions = final_model.predict(dataset.held_out.X)

    validation_metrics = compute_regression_metrics(dataset.val.y, validation_predictions)
    held_out_metrics = compute_regression_metrics(dataset.held_out.y, held_out_predictions)

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
    coefficient_table["abs_coefficient"] = coefficient_table["coefficient"].abs()

    artifact_dir = estimator_config.output_dir
    all_dataset_predictions_path = predict_all_datasets(
        data_config=data_config,
        artifact_dir=artifact_dir,
        input_columns=dataset.feature_columns,
        predict_fn=lambda frame: final_model.predict(
            frame[dataset.feature_columns].to_numpy(dtype=float)
        ),
    )
    save_linear_least_squares_artifacts(
        artifact_dir=artifact_dir,
        estimator_config=estimator_config,
        data_config=data_config,
        model=final_model,
        validation_metrics=validation_metrics,
        held_out_metrics=held_out_metrics,
        validation_predictions=validation_table,
        held_out_predictions=held_out_table,
        coefficient_table=coefficient_table,
        additional_artifact_names=[all_dataset_predictions_path.name],
        obsolete_artifacts=[
            "input_scaler.pkl",
            "polynomial_transformer.pkl",
            "ridge_model.pkl",
            "validation_search.csv",
        ],
    )

    return PressureLinearLeastSquaresResult(
        config=estimator_config,
        feature_columns=dataset.feature_columns,
        model=final_model,
        validation_metrics=validation_metrics,
        held_out_metrics=held_out_metrics,
        validation_predictions=validation_table,
        held_out_predictions=held_out_table,
        coefficient_table=coefficient_table,
        artifact_dir=artifact_dir,
        all_dataset_predictions_path=all_dataset_predictions_path,
    )


def run_pressure_linear_least_squares(
    data_config_path: str | Path = "configs/data.yaml",
    model_config_path: str | Path = "configs/models/pressure_linear_least_squares.yaml",
) -> PressureLinearLeastSquaresResult:
    data_config = load_data_config(data_config_path)
    estimator_config = load_pressure_linear_least_squares_config(model_config_path)
    runs = load_runs(data_config)
    return train_pressure_linear_least_squares(runs, data_config, estimator_config)
