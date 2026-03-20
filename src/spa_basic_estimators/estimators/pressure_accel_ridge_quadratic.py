from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

from spa_basic_estimators.estimators.pressure_ridge_common import (
    PressureOnlyDataset,
    PressureRidgeConfig,
    PressureRidgeResult,
    build_prediction_table,
    build_pressure_accel_dataset as build_pressure_accel_dataset_common,
    compute_regression_metrics,
    load_pressure_ridge_config,
    predict_all_datasets,
    save_pressure_ridge_artifacts,
)
from spa_basic_estimators.utils.config import load_yaml
from spa_basic_estimators.utils.data_loader import DataConfig, load_data_config, load_runs


@dataclass(frozen=True)
class PressureAccelRidgeQuadraticConfig(PressureRidgeConfig):
    degree: int


PressureAccelRidgeQuadraticResult = PressureRidgeResult


def load_pressure_accel_ridge_quadratic_config(
    path: str | Path,
) -> PressureAccelRidgeQuadraticConfig:
    common_config = load_pressure_ridge_config(
        path,
        default_name="pressure_accel_ridge_quadratic",
        default_output_dir="outputs/pressure_accel_ridge_quadratic",
    )
    raw = load_yaml(Path(path).resolve())
    degree = int(raw.get("degree", 2))
    if degree != 2:
        raise ValueError("Phase 6 pressure+accel quadratic currently supports degree=2 only")

    return PressureAccelRidgeQuadraticConfig(
        config_path=common_config.config_path,
        name=common_config.name,
        alpha_grid=common_config.alpha_grid,
        fit_intercept=common_config.fit_intercept,
        output_dir=common_config.output_dir,
        degree=degree,
    )


def build_pressure_accel_dataset(
    runs: Mapping[str, pd.DataFrame],
    data_config: DataConfig,
) -> PressureOnlyDataset:
    return build_pressure_accel_dataset_common(runs, data_config)


def train_pressure_accel_ridge_quadratic(
    runs: Mapping[str, pd.DataFrame],
    data_config: DataConfig,
    estimator_config: PressureAccelRidgeQuadraticConfig,
) -> PressureAccelRidgeQuadraticResult:
    dataset = build_pressure_accel_dataset_common(runs, data_config)
    polynomial = PolynomialFeatures(degree=estimator_config.degree, include_bias=False)
    X_train_poly = polynomial.fit_transform(dataset.train.X)
    feature_names = polynomial.get_feature_names_out(dataset.feature_columns).tolist()

    best_alpha: float | None = None
    best_rmse: float | None = None
    search_rows: list[dict[str, float]] = []

    for alpha in estimator_config.alpha_grid:
        model = Ridge(alpha=alpha, fit_intercept=estimator_config.fit_intercept)
        model.fit(X_train_poly, dataset.train.y)

        val_predictions = model.predict(polynomial.transform(dataset.val.X))
        val_metrics = compute_regression_metrics(dataset.val.y, val_predictions)
        search_rows.append({"alpha": float(alpha), **val_metrics})

        if best_rmse is None or val_metrics["rmse"] < best_rmse:
            best_alpha = float(alpha)
            best_rmse = float(val_metrics["rmse"])

    if best_alpha is None:
        raise ValueError("Alpha grid is empty; cannot train pressure+accel quadratic ridge model")

    final_model = Ridge(alpha=best_alpha, fit_intercept=estimator_config.fit_intercept)
    final_model.fit(X_train_poly, dataset.train.y)

    validation_predictions = final_model.predict(polynomial.transform(dataset.val.X))
    held_out_predictions = final_model.predict(polynomial.transform(dataset.held_out.X))

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
            "feature": feature_names,
            "feature_group": [
                _feature_group(feature_name, data_config) for feature_name in feature_names
            ],
            "term_type": [_term_type(feature_name) for feature_name in feature_names],
            "source_features": [
                ", ".join(_source_features(feature_name)) for feature_name in feature_names
            ],
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
            polynomial.transform(frame[dataset.feature_columns].to_numpy(dtype=float))
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
        extra_pickled_artifacts={"polynomial_transformer.pkl": polynomial},
        additional_artifact_names=[all_dataset_predictions_path.name],
        obsolete_artifacts=["input_scaler.pkl"],
    )

    return PressureAccelRidgeQuadraticResult(
        config=estimator_config,
        feature_columns=feature_names,
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


def run_pressure_accel_ridge_quadratic(
    data_config_path: str | Path = "configs/data.yaml",
    model_config_path: str | Path = "configs/models/pressure_accel_ridge_quadratic.yaml",
) -> PressureAccelRidgeQuadraticResult:
    data_config = load_data_config(data_config_path)
    estimator_config = load_pressure_accel_ridge_quadratic_config(model_config_path)
    runs = load_runs(data_config)
    return train_pressure_accel_ridge_quadratic(runs, data_config, estimator_config)


def _term_type(feature_name: str) -> str:
    if " " in feature_name:
        return "interaction"
    if "^2" in feature_name:
        return "squared"
    return "linear"


def _source_features(feature_name: str) -> list[str]:
    if " " in feature_name:
        return [part.strip() for part in feature_name.split(" ") if part.strip()]
    if "^2" in feature_name:
        return [feature_name.split("^", 1)[0]]
    return [feature_name]


def _feature_group(feature_name: str, data_config: DataConfig) -> str:
    source_features = _source_features(feature_name)
    if all(feature in data_config.schema.pressure_columns for feature in source_features):
        return "pressure"
    if all(feature in data_config.schema.accel_columns for feature in source_features):
        return "accel"
    return "mixed"
