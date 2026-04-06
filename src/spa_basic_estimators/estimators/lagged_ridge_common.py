from __future__ import annotations

from typing import Mapping

import pandas as pd

from spa_basic_estimators.estimators.lagged_pressure_accel_ridge import (
    DEFAULT_LAG_GRID,
    _build_lagged_run_design_matrix,
    _build_lagged_split_design_matrix,
    _feature_group_from_base_feature,
    _lag_from_feature_name,
    _lagged_feature_names,
    _predict_all_datasets_lagged,
    _source_feature_from_lagged_feature,
)
from spa_basic_estimators.estimators.pressure_ridge_common import DatasetMatrices
from spa_basic_estimators.utils.data_loader import DataConfig


def pressure_feature_columns(data_config: DataConfig) -> list[str]:
    pressure_columns = list(data_config.schema.pressure_columns)
    if not pressure_columns:
        raise ValueError("Lagged pressure ridge requires at least one pressure column")
    return pressure_columns


def accel_feature_columns(data_config: DataConfig) -> list[str]:
    accel_columns = list(data_config.schema.accel_columns)
    if not accel_columns:
        raise ValueError("Lagged accel ridge requires at least one accelerometer column")
    return accel_columns


def build_lagged_dataset_from_columns(
    runs: Mapping[str, pd.DataFrame],
    data_config: DataConfig,
    raw_feature_columns: list[str],
    lag_length: int,
) -> DatasetMatrices:
    feature_columns = _lagged_feature_names(raw_feature_columns, lag_length)

    return DatasetMatrices(
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


__all__ = [
    "DEFAULT_LAG_GRID",
    "_build_lagged_run_design_matrix",
    "_feature_group_from_base_feature",
    "_lag_from_feature_name",
    "_predict_all_datasets_lagged",
    "_source_feature_from_lagged_feature",
    "accel_feature_columns",
    "build_lagged_dataset_from_columns",
    "pressure_feature_columns",
]
