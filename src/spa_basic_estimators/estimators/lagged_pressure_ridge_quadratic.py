from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

from spa_basic_estimators.estimators.lagged_pressure_accel_ridge_quadratic import (
    _feature_group,
    _predict_all_datasets_lagged_quadratic,
    _source_features,
    _term_type,
)
from spa_basic_estimators.estimators.lagged_ridge_common import (
    DEFAULT_LAG_GRID,
    _lag_from_feature_name,
    build_lagged_dataset_from_columns,
    pressure_feature_columns,
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
class LaggedPressureRidgeQuadraticConfig(RidgeModelConfig):
    lag_grid: list[int]
    degree: int


@dataclass(frozen=True)
class LaggedPressureRidgeQuadraticResult(RidgeTrainingResult):
    selected_lag: int
    raw_feature_columns: list[str]


def load_lagged_pressure_ridge_quadratic_config(
    path: str | Path,
) -> LaggedPressureRidgeQuadraticConfig:
    common_config = load_ridge_model_config(
        path,
        default_name="lagged_pressure_ridge_quadratic",
        default_output_dir="outputs/lagged_pressure_ridge_quadratic",
    )
    raw = load_yaml(Path(path).resolve())
    lag_grid = [int(value) for value in raw.get("lag_grid", DEFAULT_LAG_GRID)]
    lag_grid = list(dict.fromkeys(lag_grid))
    if not lag_grid:
        raise ValueError("Lagged pressure quadratic ridge requires at least one lag length")
    if any(lag_length < 1 for lag_length in lag_grid):
        raise ValueError("All lag lengths must be positive integers")

    degree = int(raw.get("degree", 2))
    if degree != 2:
        raise ValueError("Lagged pressure quadratic ridge currently supports degree=2 only")

    return LaggedPressureRidgeQuadraticConfig(
        config_path=common_config.config_path,
        name=common_config.name,
        alpha_grid=common_config.alpha_grid,
        fit_intercept=common_config.fit_intercept,
        output_dir=common_config.output_dir,
        lag_grid=lag_grid,
        degree=degree,
    )


def build_lagged_pressure_dataset(
    runs: Mapping[str, pd.DataFrame],
    data_config: DataConfig,
    lag_length: int,
) -> DatasetMatrices:
    raw_feature_columns = pressure_feature_columns(data_config)
    return build_lagged_dataset_from_columns(runs, data_config, raw_feature_columns, lag_length)


def train_lagged_pressure_ridge_quadratic(
    runs: Mapping[str, pd.DataFrame],
    data_config: DataConfig,
    estimator_config: LaggedPressureRidgeQuadraticConfig,
) -> LaggedPressureRidgeQuadraticResult:
    raw_feature_columns = pressure_feature_columns(data_config)
    datasets_by_lag: dict[int, DatasetMatrices] = {}
    polynomials_by_lag: dict[int, PolynomialFeatures] = {}
    feature_names_by_lag: dict[int, list[str]] = {}

    best_alpha: float | None = None
    best_lag: int | None = None
    best_rmse: float | None = None
    search_rows: list[dict[str, float | int]] = []

    for lag_length in estimator_config.lag_grid:
        dataset = build_lagged_dataset_from_columns(runs, data_config, raw_feature_columns, lag_length)
        polynomial = PolynomialFeatures(degree=estimator_config.degree, include_bias=False)
        X_train_poly = polynomial.fit_transform(dataset.train.X)
        feature_names = polynomial.get_feature_names_out(dataset.feature_columns).tolist()

        datasets_by_lag[lag_length] = dataset
        polynomials_by_lag[lag_length] = polynomial
        feature_names_by_lag[lag_length] = feature_names

        for alpha in estimator_config.alpha_grid:
            model = Ridge(alpha=alpha, fit_intercept=estimator_config.fit_intercept)
            model.fit(X_train_poly, dataset.train.y)

            val_predictions = model.predict(polynomial.transform(dataset.val.X))
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
        raise ValueError("Lagged pressure quadratic ridge could not select a best lag/alpha pair")

    final_dataset = datasets_by_lag[best_lag]
    final_polynomial = polynomials_by_lag[best_lag]
    final_feature_names = feature_names_by_lag[best_lag]
    final_model = Ridge(alpha=best_alpha, fit_intercept=estimator_config.fit_intercept)
    final_model.fit(final_polynomial.transform(final_dataset.train.X), final_dataset.train.y)

    validation_predictions = final_model.predict(final_polynomial.transform(final_dataset.val.X))
    held_out_predictions = final_model.predict(final_polynomial.transform(final_dataset.held_out.X))

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
            "feature": final_feature_names,
            "feature_group": [
                _feature_group(feature_name, data_config) for feature_name in final_feature_names
            ],
            "term_type": [_term_type(feature_name) for feature_name in final_feature_names],
            "source_features": [
                ", ".join(_source_features(feature_name)) for feature_name in final_feature_names
            ],
            "source_lags": [
                ", ".join(str(_lag_from_feature_name(name)) for name in _source_features(feature_name))
                for feature_name in final_feature_names
            ],
            "coefficient": np.ravel(final_model.coef_),
        }
    )
    coefficient_table["abs_coefficient"] = coefficient_table["coefficient"].abs()

    artifact_dir = estimator_config.output_dir
    all_dataset_predictions_path = _predict_all_datasets_lagged_quadratic(
        data_config=data_config,
        artifact_dir=artifact_dir,
        raw_feature_columns=raw_feature_columns,
        lag_length=best_lag,
        model=final_model,
        polynomial=final_polynomial,
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
        extra_pickled_artifacts={"polynomial_transformer.pkl": final_polynomial},
        extra_summary_fields={
            "selected_lag": int(best_lag),
            "raw_feature_columns": raw_feature_columns,
        },
        additional_artifact_names=[all_dataset_predictions_path.name],
        obsolete_artifacts=["input_scaler.pkl"],
    )

    return LaggedPressureRidgeQuadraticResult(
        config=estimator_config,
        feature_columns=final_feature_names,
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


def run_lagged_pressure_ridge_quadratic(
    data_config_path: str | Path = "configs/data.yaml",
    model_config_path: str | Path = "configs/models/lagged_pressure_ridge_quadratic.yaml",
) -> LaggedPressureRidgeQuadraticResult:
    data_config = load_data_config(data_config_path)
    estimator_config = load_lagged_pressure_ridge_quadratic_config(model_config_path)
    runs = load_runs(data_config)
    return train_lagged_pressure_ridge_quadratic(runs, data_config, estimator_config)
