from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

from spa_basic_estimators.estimators.pressure_ridge_common import (
    DatasetMatrices,
    RidgeModelConfig,
    RidgeTrainingResult,
    build_prediction_table,
    compute_regression_metrics,
    load_ridge_model_config,
    predict_all_datasets,
    save_ridge_artifacts,
)
from spa_basic_estimators.utils.config import load_yaml
from spa_basic_estimators.utils.data_loader import DataConfig

DEFAULT_DEGREE_GRID = [1, 2, 3]


@dataclass(frozen=True)
class PolynomialSearchRidgeConfig(RidgeModelConfig):
    degree_grid: list[int]


@dataclass(frozen=True)
class PolynomialSearchRidgeResult(RidgeTrainingResult):
    selected_degree: int


def load_polynomial_search_ridge_config(
    path: str | Path,
    *,
    default_name: str,
    default_output_dir: str,
    default_degree_grid: list[int] | None = None,
) -> PolynomialSearchRidgeConfig:
    common_config = load_ridge_model_config(
        path,
        default_name=default_name,
        default_output_dir=default_output_dir,
    )
    raw = load_yaml(Path(path).resolve())
    raw_degree_grid = raw.get("degree_grid", default_degree_grid or DEFAULT_DEGREE_GRID)
    degree_grid = _normalise_degree_grid(raw_degree_grid)

    return PolynomialSearchRidgeConfig(
        config_path=common_config.config_path,
        name=common_config.name,
        alpha_grid=common_config.alpha_grid,
        fit_intercept=common_config.fit_intercept,
        output_dir=common_config.output_dir,
        degree_grid=degree_grid,
    )


def train_static_polynomial_ridge(
    *,
    runs: Mapping[str, pd.DataFrame],
    data_config: DataConfig,
    estimator_config: PolynomialSearchRidgeConfig,
    dataset_builder: Callable[[Mapping[str, pd.DataFrame], DataConfig], DatasetMatrices],
) -> PolynomialSearchRidgeResult:
    dataset = dataset_builder(runs, data_config)

    best_alpha: float | None = None
    best_degree: int | None = None
    best_rmse: float | None = None
    search_rows: list[dict[str, float | int]] = []

    for degree in estimator_config.degree_grid:
        polynomial = PolynomialFeatures(degree=degree, include_bias=False)
        X_train_poly = polynomial.fit_transform(dataset.train.X)
        X_val_poly = polynomial.transform(dataset.val.X)

        for alpha in estimator_config.alpha_grid:
            model = Ridge(alpha=alpha, fit_intercept=estimator_config.fit_intercept)
            model.fit(X_train_poly, dataset.train.y)

            val_predictions = model.predict(X_val_poly)
            val_metrics = compute_regression_metrics(dataset.val.y, val_predictions)
            search_rows.append({"degree": int(degree), "alpha": float(alpha), **val_metrics})

            if best_rmse is None or val_metrics["rmse"] < best_rmse:
                best_alpha = float(alpha)
                best_degree = int(degree)
                best_rmse = float(val_metrics["rmse"])

    if best_alpha is None or best_degree is None:
        raise ValueError("Search grids are empty; cannot train static polynomial ridge model")

    final_polynomial = PolynomialFeatures(degree=best_degree, include_bias=False)
    X_train_poly = final_polynomial.fit_transform(dataset.train.X)
    feature_names = final_polynomial.get_feature_names_out(dataset.feature_columns).tolist()

    final_model = Ridge(alpha=best_alpha, fit_intercept=estimator_config.fit_intercept)
    final_model.fit(X_train_poly, dataset.train.y)

    validation_predictions = final_model.predict(final_polynomial.transform(dataset.val.X))
    held_out_predictions = final_model.predict(final_polynomial.transform(dataset.held_out.X))

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
    coefficient_table = build_polynomial_coefficient_table(
        feature_names=feature_names,
        coefficients=np.ravel(final_model.coef_),
        data_config=data_config,
    )

    artifact_dir = estimator_config.output_dir
    all_dataset_predictions_path = predict_all_datasets(
        data_config=data_config,
        artifact_dir=artifact_dir,
        input_columns=dataset.feature_columns,
        predict_fn=lambda frame: final_model.predict(
            final_polynomial.transform(frame[dataset.feature_columns].to_numpy(dtype=float))
        ),
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
            "selected_degree": best_degree,
            "degree_grid": list(estimator_config.degree_grid),
        },
        additional_artifact_names=[all_dataset_predictions_path.name],
        obsolete_artifacts=["input_scaler.pkl"],
    )

    return PolynomialSearchRidgeResult(
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
        selected_degree=best_degree,
    )


def build_polynomial_coefficient_table(
    *,
    feature_names: list[str],
    coefficients: np.ndarray,
    data_config: DataConfig,
) -> pd.DataFrame:
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
            "total_degree": [_total_degree(feature_name) for feature_name in feature_names],
            "coefficient": np.asarray(coefficients, dtype=float),
        }
    )
    coefficient_table["abs_coefficient"] = coefficient_table["coefficient"].abs()
    return coefficient_table


def _normalise_degree_grid(raw_degree_grid: object) -> list[int]:
    if not isinstance(raw_degree_grid, list):
        raise ValueError("degree_grid must be a list of positive integers")

    normalised: list[int] = []
    seen: set[int] = set()
    for raw_degree in raw_degree_grid:
        degree = int(raw_degree)
        if degree < 1:
            raise ValueError("Polynomial degree grid entries must be >= 1")
        if degree in seen:
            continue
        seen.add(degree)
        normalised.append(degree)

    if not normalised:
        raise ValueError("degree_grid cannot be empty")
    return normalised


def _parse_feature_tokens(feature_name: str) -> list[tuple[str, int]]:
    parsed: list[tuple[str, int]] = []
    for token in feature_name.split(" "):
        stripped = token.strip()
        if not stripped:
            continue
        if "^" in stripped:
            base_name, power_text = stripped.split("^", 1)
            parsed.append((base_name, int(power_text)))
        else:
            parsed.append((stripped, 1))
    return parsed


def _source_features(feature_name: str) -> list[str]:
    return [base_name for base_name, _ in _parse_feature_tokens(feature_name)]


def _total_degree(feature_name: str) -> int:
    return sum(power for _, power in _parse_feature_tokens(feature_name))


def _term_type(feature_name: str) -> str:
    source_features = _source_features(feature_name)
    total_degree = _total_degree(feature_name)
    if len(source_features) == 1 and total_degree == 1:
        return "linear"
    if len(source_features) == 1:
        return "polynomial_power"
    return "interaction"


def _feature_group(feature_name: str, data_config: DataConfig) -> str:
    source_features = _source_features(feature_name)
    if all(feature in data_config.schema.pressure_columns for feature in source_features):
        return "pressure"
    if all(feature in data_config.schema.accel_columns for feature in source_features):
        return "accel"
    return "mixed"
