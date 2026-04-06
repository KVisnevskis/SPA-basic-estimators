from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.preprocessing import PolynomialFeatures

from spa_basic_estimators.estimators.lagged_accel_ridge import (
    load_lagged_accel_ridge_config,
    train_lagged_accel_ridge,
)
from spa_basic_estimators.estimators.lagged_accel_ridge_quadratic import (
    load_lagged_accel_ridge_quadratic_config,
    train_lagged_accel_ridge_quadratic,
)
from spa_basic_estimators.estimators.lagged_pressure_ridge import (
    build_lagged_pressure_dataset,
    load_lagged_pressure_ridge_config,
    train_lagged_pressure_ridge,
)
from spa_basic_estimators.estimators.lagged_pressure_ridge_quadratic import (
    load_lagged_pressure_ridge_quadratic_config,
    train_lagged_pressure_ridge_quadratic,
)
from spa_basic_estimators.utils.data_loader import load_data_config, load_runs
from spa_basic_estimators.utils.splits import UNASSIGNED_SPLIT


def _write_linear_model_config(
    path: Path,
    name: str,
    output_dir: str,
    lag_grid: list[int],
    alpha_grid: list[float],
) -> None:
    payload = {
        "name": name,
        "fit_intercept": True,
        "lag_grid": lag_grid,
        "alpha_grid": alpha_grid,
        "output_dir": output_dir,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _write_quadratic_model_config(
    path: Path,
    name: str,
    output_dir: str,
    lag_grid: list[int],
    alpha_grid: list[float],
) -> None:
    payload = {
        "name": name,
        "degree": 2,
        "fit_intercept": True,
        "lag_grid": lag_grid,
        "alpha_grid": alpha_grid,
        "output_dir": output_dir,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


@pytest.mark.parametrize(
    ("name", "output_dir", "loader_fn", "train_fn", "writer_fn", "expected_raw_columns", "expect_polynomial"),
    [
        (
            "lagged_pressure_ridge",
            "outputs/lagged_pressure_ridge",
            load_lagged_pressure_ridge_config,
            train_lagged_pressure_ridge,
            _write_linear_model_config,
            ["pressure"],
            False,
        ),
        (
            "lagged_accel_ridge",
            "outputs/lagged_accel_ridge",
            load_lagged_accel_ridge_config,
            train_lagged_accel_ridge,
            _write_linear_model_config,
            ["acc_x", "acc_y", "acc_z"],
            False,
        ),
        (
            "lagged_pressure_ridge_quadratic",
            "outputs/lagged_pressure_ridge_quadratic",
            load_lagged_pressure_ridge_quadratic_config,
            train_lagged_pressure_ridge_quadratic,
            _write_quadratic_model_config,
            ["pressure"],
            True,
        ),
        (
            "lagged_accel_ridge_quadratic",
            "outputs/lagged_accel_ridge_quadratic",
            load_lagged_accel_ridge_quadratic_config,
            train_lagged_accel_ridge_quadratic,
            _write_quadratic_model_config,
            ["acc_x", "acc_y", "acc_z"],
            True,
        ),
    ],
)
def test_lagged_ablation_smoke_runs(
    synthetic_loader_case: dict[str, Path],
    name: str,
    output_dir: str,
    loader_fn,
    train_fn,
    writer_fn,
    expected_raw_columns: list[str],
    expect_polynomial: bool,
) -> None:
    lag_length = 2
    project_root = synthetic_loader_case["project_root"]
    model_config_path = project_root / "configs" / "models" / f"{name}.yaml"
    writer_fn(model_config_path, name, output_dir, [lag_length], [1e-6, 1e-3, 1.0])

    data_config = load_data_config(synthetic_loader_case["config_path"])
    model_config = loader_fn(model_config_path)
    runs = load_runs(data_config)

    result = train_fn(runs, data_config, model_config)

    assert result.selected_lag == lag_length
    assert result.selected_alpha in model_config.alpha_grid
    assert result.raw_feature_columns == expected_raw_columns
    assert result.validation_predictions["__split__"].unique().tolist() == ["val"]
    assert result.held_out_predictions["__split__"].unique().tolist() == ["held_out"]
    assert result.validation_predictions["sample_index"].min() == lag_length - 1
    assert result.held_out_predictions["sample_index"].min() == lag_length - 1
    assert set(result.coefficient_table["feature_group"]) == {
        "pressure" if expected_raw_columns == ["pressure"] else "accel"
    }

    assert (result.artifact_dir / "ridge_model.pkl").exists()
    assert (result.artifact_dir / "validation_search.csv").exists()
    assert (result.artifact_dir / "held_out_predictions.csv").exists()
    assert (result.artifact_dir / "coefficient_table.csv").exists()
    assert not (result.artifact_dir / "input_scaler.pkl").exists()
    if expect_polynomial:
        assert (result.artifact_dir / "polynomial_transformer.pkl").exists()
    else:
        assert not (result.artifact_dir / "polynomial_transformer.pkl").exists()
    assert result.all_dataset_predictions_path.exists()

    with pd.HDFStore(result.all_dataset_predictions_path, mode="r") as store:
        assert set(store.keys()) == {
            "/meta/runs",
            "/predictions/run_extra_1",
            "/predictions/run_test_1",
            "/predictions/run_train_1",
            "/predictions/run_val_1",
        }
        per_run = store["/predictions/run_train_1"]
        assert set(expected_raw_columns + ["phi_true", "phi_prediction", "phi_error", "sample_index"]).issubset(
            per_run.columns
        )
        assert int(per_run["sample_index"].iloc[0]) == lag_length - 1
        extra_run = store["/predictions/run_extra_1"]
        assert extra_run["split"].unique().tolist() == [UNASSIGNED_SPLIT]


def test_lagged_pressure_feature_builder_has_expected_shape(
    synthetic_loader_case: dict[str, Path],
) -> None:
    data_config = load_data_config(synthetic_loader_case["config_path"])
    runs = load_runs(data_config)
    dataset = build_lagged_pressure_dataset(runs, data_config, lag_length=3)

    assert dataset.feature_columns == [
        "pressure_lag_0",
        "pressure_lag_1",
        "pressure_lag_2",
    ]
    assert dataset.train.X.shape == (1, 3)
    assert dataset.train.metadata["sample_index"].tolist() == [2]


def test_lagged_pressure_ridge_fits_known_linear_history_relationship(
    synthetic_loader_case: dict[str, Path],
) -> None:
    lag_length = 3
    rng = np.random.default_rng(17)

    def make_frame(num_rows: int, time_offset: float) -> pd.DataFrame:
        pressure = rng.uniform(-1.0, 1.0, size=num_rows)
        frame = pd.DataFrame(
            {
                "pressure": pressure,
                "acc_x": rng.uniform(-1.0, 1.0, size=num_rows),
                "acc_y": rng.uniform(-1.0, 1.0, size=num_rows),
                "acc_z": rng.uniform(-1.0, 1.0, size=num_rows),
                "Time": time_offset + np.arange(num_rows, dtype=float) * 0.1,
            }
        )
        phi = np.zeros(num_rows, dtype=float)
        for t in range(lag_length - 1, num_rows):
            phi[t] = 0.25 + 1.5 * pressure[t] - 2.0 * pressure[t - 1] + 3.5 * pressure[t - 2]
        frame["phi"] = phi
        return frame

    train_frame = make_frame(30, 0.0)
    val_frame = make_frame(10, 10.0)
    test_frame = make_frame(10, 20.0)
    _overwrite_split_runs(synthetic_loader_case["h5_path"], train_frame, val_frame, test_frame)

    project_root = synthetic_loader_case["project_root"]
    model_config_path = project_root / "configs" / "models" / "lagged_pressure_ridge.yaml"
    _write_linear_model_config(
        model_config_path,
        "lagged_pressure_ridge",
        "outputs/lagged_pressure_ridge",
        [lag_length],
        [1e-12],
    )

    data_config = load_data_config(synthetic_loader_case["config_path"])
    model_config = load_lagged_pressure_ridge_config(model_config_path)
    runs = load_runs(data_config)

    result = train_lagged_pressure_ridge(runs, data_config, model_config)

    assert result.selected_lag == lag_length
    assert result.validation_metrics["rmse"] < 1e-8
    assert result.held_out_metrics["rmse"] < 1e-8


def test_lagged_accel_ridge_fits_known_linear_history_relationship(
    synthetic_loader_case: dict[str, Path],
) -> None:
    lag_length = 2
    rng = np.random.default_rng(21)

    def make_frame(num_rows: int, time_offset: float) -> pd.DataFrame:
        accel = rng.uniform(-1.0, 1.0, size=(num_rows, 3))
        frame = pd.DataFrame(
            {
                "pressure": rng.uniform(-1.0, 1.0, size=num_rows),
                "acc_x": accel[:, 0],
                "acc_y": accel[:, 1],
                "acc_z": accel[:, 2],
                "Time": time_offset + np.arange(num_rows, dtype=float) * 0.1,
            }
        )
        phi = np.zeros(num_rows, dtype=float)
        for t in range(lag_length - 1, num_rows):
            phi[t] = (
                -0.5
                + 2.0 * frame.loc[t, "acc_x"]
                - 3.0 * frame.loc[t, "acc_y"]
                + 4.0 * frame.loc[t - 1, "acc_z"]
            )
        frame["phi"] = phi
        return frame

    train_frame = make_frame(30, 0.0)
    val_frame = make_frame(10, 10.0)
    test_frame = make_frame(10, 20.0)
    _overwrite_split_runs(synthetic_loader_case["h5_path"], train_frame, val_frame, test_frame)

    project_root = synthetic_loader_case["project_root"]
    model_config_path = project_root / "configs" / "models" / "lagged_accel_ridge.yaml"
    _write_linear_model_config(
        model_config_path,
        "lagged_accel_ridge",
        "outputs/lagged_accel_ridge",
        [lag_length],
        [1e-12],
    )

    data_config = load_data_config(synthetic_loader_case["config_path"])
    model_config = load_lagged_accel_ridge_config(model_config_path)
    runs = load_runs(data_config)

    result = train_lagged_accel_ridge(runs, data_config, model_config)

    assert result.selected_lag == lag_length
    assert result.validation_metrics["rmse"] < 1e-8
    assert result.held_out_metrics["rmse"] < 1e-8


def test_lagged_pressure_ridge_quadratic_fits_known_degree_two_history_relationship(
    synthetic_loader_case: dict[str, Path],
) -> None:
    lag_length = 2
    rng = np.random.default_rng(23)

    def make_frame(num_rows: int, time_offset: float) -> pd.DataFrame:
        pressure = rng.uniform(-1.0, 1.0, size=num_rows)
        frame = pd.DataFrame(
            {
                "pressure": pressure,
                "acc_x": rng.uniform(-1.0, 1.0, size=num_rows),
                "acc_y": rng.uniform(-1.0, 1.0, size=num_rows),
                "acc_z": rng.uniform(-1.0, 1.0, size=num_rows),
                "Time": time_offset + np.arange(num_rows, dtype=float) * 0.1,
            }
        )
        phi = np.zeros(num_rows, dtype=float)
        for t in range(lag_length - 1, num_rows):
            phi[t] = (
                0.5
                + 1.5 * pressure[t]
                - 2.0 * pressure[t - 1]
                + 3.0 * pressure[t] ** 2
                - 4.0 * pressure[t] * pressure[t - 1]
                + 5.0 * pressure[t - 1] ** 2
            )
        frame["phi"] = phi
        return frame

    train_frame = make_frame(48, 0.0)
    val_frame = make_frame(16, 10.0)
    test_frame = make_frame(16, 20.0)
    _overwrite_split_runs(synthetic_loader_case["h5_path"], train_frame, val_frame, test_frame)

    project_root = synthetic_loader_case["project_root"]
    model_config_path = (
        project_root / "configs" / "models" / "lagged_pressure_ridge_quadratic.yaml"
    )
    _write_quadratic_model_config(
        model_config_path,
        "lagged_pressure_ridge_quadratic",
        "outputs/lagged_pressure_ridge_quadratic",
        [lag_length],
        [1e-12],
    )

    data_config = load_data_config(synthetic_loader_case["config_path"])
    model_config = load_lagged_pressure_ridge_quadratic_config(model_config_path)
    runs = load_runs(data_config)

    result = train_lagged_pressure_ridge_quadratic(runs, data_config, model_config)

    lagged_features = ["pressure_lag_0", "pressure_lag_1"]
    polynomial = PolynomialFeatures(degree=2, include_bias=False)
    polynomial.fit(np.zeros((1, len(lagged_features))))
    expected_features = polynomial.get_feature_names_out(lagged_features).tolist()

    assert result.feature_columns == expected_features
    assert result.validation_metrics["rmse"] < 1e-8
    assert result.held_out_metrics["rmse"] < 1e-8


def _overwrite_split_runs(
    h5_path: Path,
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
) -> None:
    with pd.HDFStore(h5_path, mode="a") as store:
        store.put("/runs/run_train_1", train_frame, format="fixed")
        store.put("/runs/run_val_1", val_frame, format="fixed")
        store.put("/runs/run_test_1", test_frame, format="fixed")
