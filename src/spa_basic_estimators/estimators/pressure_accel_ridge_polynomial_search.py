from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd

from spa_basic_estimators.estimators.pressure_ridge_common import build_pressure_accel_dataset
from spa_basic_estimators.estimators.static_polynomial_ridge_common import (
    PolynomialSearchRidgeConfig,
    PolynomialSearchRidgeResult,
    load_polynomial_search_ridge_config,
    train_static_polynomial_ridge,
)
from spa_basic_estimators.utils.data_loader import DataConfig, load_data_config, load_runs

PressureAccelRidgePolynomialSearchConfig = PolynomialSearchRidgeConfig
PressureAccelRidgePolynomialSearchResult = PolynomialSearchRidgeResult


def load_pressure_accel_ridge_polynomial_search_config(
    path: str | Path,
) -> PressureAccelRidgePolynomialSearchConfig:
    return load_polynomial_search_ridge_config(
        path,
        default_name="pressure_accel_ridge_polynomial_search",
        default_output_dir="outputs/pressure_accel_ridge_polynomial_search",
        default_degree_grid=[1, 2, 3],
    )


def train_pressure_accel_ridge_polynomial_search(
    runs: Mapping[str, pd.DataFrame],
    data_config: DataConfig,
    estimator_config: PressureAccelRidgePolynomialSearchConfig,
) -> PressureAccelRidgePolynomialSearchResult:
    return train_static_polynomial_ridge(
        runs=runs,
        data_config=data_config,
        estimator_config=estimator_config,
        dataset_builder=build_pressure_accel_dataset,
    )


def run_pressure_accel_ridge_polynomial_search(
    data_config_path: str | Path = "configs/data.yaml",
    model_config_path: str | Path = "configs/models/pressure_accel_ridge_polynomial_search.yaml",
) -> PressureAccelRidgePolynomialSearchResult:
    data_config = load_data_config(data_config_path)
    estimator_config = load_pressure_accel_ridge_polynomial_search_config(model_config_path)
    runs = load_runs(data_config)
    return train_pressure_accel_ridge_polynomial_search(runs, data_config, estimator_config)
