from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from spa_basic_estimators.estimators.lagged_pressure_accel_ridge import (
    build_lagged_pressure_accel_dataset,
    load_lagged_pressure_accel_ridge_config,
    train_lagged_pressure_accel_ridge,
)
from spa_basic_estimators.utils.data_loader import load_data_config, load_runs
from spa_basic_estimators.utils.splits import UNASSIGNED_SPLIT


def _write_model_config(
    path: Path,
    output_dir: str,
    lag_grid: list[int],
    alpha_grid: list[float],
) -> None:
    payload = {
        "name": "lagged_pressure_accel_ridge",
        "fit_intercept": True,
        "lag_grid": lag_grid,
        "alpha_grid": alpha_grid,
        "output_dir": output_dir,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_lagged_pressure_accel_ridge_smoke_run(
    synthetic_loader_case: dict[str, Path],
) -> None:
    project_root = synthetic_loader_case["project_root"]
    model_config_path = (
        project_root / "configs" / "models" / "lagged_pressure_accel_ridge.yaml"
    )
    _write_model_config(
        model_config_path,
        "outputs/lagged_pressure_accel_ridge",
        [1, 2],
        [1e-6, 1e-3, 1.0],
    )

    data_config = load_data_config(synthetic_loader_case["config_path"])
    model_config = load_lagged_pressure_accel_ridge_config(model_config_path)
    runs = load_runs(data_config)

    result = train_lagged_pressure_accel_ridge(runs, data_config, model_config)

    assert result.selected_lag in model_config.lag_grid
    assert result.selected_alpha in model_config.alpha_grid
    assert result.raw_feature_columns == ["pressure", "acc_x", "acc_y", "acc_z"]
    assert result.validation_predictions["__split__"].unique().tolist() == ["val"]
    assert result.held_out_predictions["__split__"].unique().tolist() == ["held_out"]
    assert result.validation_predictions["sample_index"].min() == result.selected_lag - 1
    assert result.held_out_predictions["sample_index"].min() == result.selected_lag - 1
    assert set(result.coefficient_table["feature_group"]) == {"pressure", "accel"}
    assert set(result.coefficient_table["lag"]) == set(range(result.selected_lag))

    assert (result.artifact_dir / "ridge_model.pkl").exists()
    assert (result.artifact_dir / "validation_search.csv").exists()
    assert (result.artifact_dir / "held_out_predictions.csv").exists()
    assert (result.artifact_dir / "coefficient_table.csv").exists()
    assert not (result.artifact_dir / "input_scaler.pkl").exists()
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
        assert {
            "pressure",
            "acc_x",
            "acc_y",
            "acc_z",
            "phi_true",
            "phi_prediction",
            "phi_error",
            "sample_index",
        }.issubset(per_run.columns)
        assert len(per_run) == 4 - result.selected_lag
        assert int(per_run["sample_index"].iloc[0]) == result.selected_lag - 1
        extra_run = store["/predictions/run_extra_1"]
        assert extra_run["split"].unique().tolist() == [UNASSIGNED_SPLIT]


def test_lagged_feature_builder_preserves_alignment_and_run_boundaries(
    synthetic_loader_case: dict[str, Path],
) -> None:
    train_frame = pd.DataFrame(
        {
            "pressure": [10.0, 11.0, 12.0, 13.0],
            "acc_x": [20.0, 21.0, 22.0, 23.0],
            "acc_y": [30.0, 31.0, 32.0, 33.0],
            "acc_z": [40.0, 41.0, 42.0, 43.0],
            "phi": [100.0, 101.0, 102.0, 103.0],
            "Time": [0.0, 0.1, 0.2, 0.3],
        }
    )
    extra_train_frame = pd.DataFrame(
        {
            "pressure": [110.0, 111.0, 112.0, 113.0],
            "acc_x": [120.0, 121.0, 122.0, 123.0],
            "acc_y": [130.0, 131.0, 132.0, 133.0],
            "acc_z": [140.0, 141.0, 142.0, 143.0],
            "phi": [200.0, 201.0, 202.0, 203.0],
            "Time": [1.0, 1.1, 1.2, 1.3],
        }
    )
    val_frame = pd.DataFrame(
        {
            "pressure": [210.0, 211.0, 212.0, 213.0],
            "acc_x": [220.0, 221.0, 222.0, 223.0],
            "acc_y": [230.0, 231.0, 232.0, 233.0],
            "acc_z": [240.0, 241.0, 242.0, 243.0],
            "phi": [300.0, 301.0, 302.0, 303.0],
            "Time": [2.0, 2.1, 2.2, 2.3],
        }
    )
    test_frame = pd.DataFrame(
        {
            "pressure": [310.0, 311.0, 312.0, 313.0],
            "acc_x": [320.0, 321.0, 322.0, 323.0],
            "acc_y": [330.0, 331.0, 332.0, 333.0],
            "acc_z": [340.0, 341.0, 342.0, 343.0],
            "phi": [400.0, 401.0, 402.0, 403.0],
            "Time": [3.0, 3.1, 3.2, 3.3],
        }
    )

    h5_path = synthetic_loader_case["h5_path"]
    with pd.HDFStore(h5_path, mode="a") as store:
        store.put("/runs/run_train_1", train_frame, format="fixed")
        store.put("/runs/run_extra_1", extra_train_frame, format="fixed")
        store.put("/runs/run_val_1", val_frame, format="fixed")
        store.put("/runs/run_test_1", test_frame, format="fixed")

    config_path = synthetic_loader_case["config_path"]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["splits"]["train"] = ["run_train_1", "run_extra_1"]
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    data_config = load_data_config(config_path)
    runs = load_runs(data_config)
    dataset = build_lagged_pressure_accel_dataset(runs, data_config, lag_length=3)

    assert dataset.feature_columns == [
        "pressure_lag_0",
        "acc_x_lag_0",
        "acc_y_lag_0",
        "acc_z_lag_0",
        "pressure_lag_1",
        "acc_x_lag_1",
        "acc_y_lag_1",
        "acc_z_lag_1",
        "pressure_lag_2",
        "acc_x_lag_2",
        "acc_y_lag_2",
        "acc_z_lag_2",
    ]

    assert dataset.train.X.shape == (4, 12)
    assert dataset.train.y.tolist() == [102.0, 103.0, 202.0, 203.0]
    assert dataset.train.metadata["sample_index"].tolist() == [2, 3, 2, 3]
    assert dataset.train.metadata["run_id"].tolist() == [
        "run_train_1",
        "run_train_1",
        "run_extra_1",
        "run_extra_1",
    ]

    assert dataset.train.X[0].tolist() == [
        12.0,
        22.0,
        32.0,
        42.0,
        11.0,
        21.0,
        31.0,
        41.0,
        10.0,
        20.0,
        30.0,
        40.0,
    ]
    assert dataset.train.X[2].tolist() == [
        112.0,
        122.0,
        132.0,
        142.0,
        111.0,
        121.0,
        131.0,
        141.0,
        110.0,
        120.0,
        130.0,
        140.0,
    ]
    assert dataset.val.X[0].tolist() == [
        212.0,
        222.0,
        232.0,
        242.0,
        211.0,
        221.0,
        231.0,
        241.0,
        210.0,
        220.0,
        230.0,
        240.0,
    ]


def test_lagged_pressure_accel_ridge_fits_known_linear_history_relationship(
    synthetic_loader_case: dict[str, Path],
) -> None:
    lag_length = 3
    rng = np.random.default_rng(17)

    def make_frame(num_rows: int, time_offset: float) -> pd.DataFrame:
        values = rng.uniform(-1.0, 1.0, size=(num_rows, 4))
        frame = pd.DataFrame(
            values,
            columns=["pressure", "acc_x", "acc_y", "acc_z"],
        )
        frame["Time"] = time_offset + np.arange(num_rows, dtype=float) * 0.1
        phi = np.zeros(num_rows, dtype=float)
        for t in range(lag_length - 1, num_rows):
            phi[t] = (
                0.25
                + 1.5 * frame.loc[t, "pressure"]
                - 2.5 * frame.loc[t, "acc_x"]
                + 3.5 * frame.loc[t - 1, "acc_y"]
                - 4.5 * frame.loc[t - 1, "acc_z"]
                + 5.5 * frame.loc[t - 2, "pressure"]
                - 6.5 * frame.loc[t - 2, "acc_x"]
            )
        frame["phi"] = phi
        return frame

    train_frame = make_frame(30, 0.0)
    val_frame = make_frame(10, 10.0)
    test_frame = make_frame(10, 20.0)

    h5_path = synthetic_loader_case["h5_path"]
    with pd.HDFStore(h5_path, mode="a") as store:
        store.put("/runs/run_train_1", train_frame, format="fixed")
        store.put("/runs/run_val_1", val_frame, format="fixed")
        store.put("/runs/run_test_1", test_frame, format="fixed")

    project_root = synthetic_loader_case["project_root"]
    model_config_path = (
        project_root / "configs" / "models" / "lagged_pressure_accel_ridge.yaml"
    )
    _write_model_config(
        model_config_path,
        "outputs/lagged_pressure_accel_ridge",
        [lag_length],
        [1e-12],
    )

    data_config = load_data_config(synthetic_loader_case["config_path"])
    model_config = load_lagged_pressure_accel_ridge_config(model_config_path)
    runs = load_runs(data_config)

    result = train_lagged_pressure_accel_ridge(runs, data_config, model_config)

    assert result.selected_lag == lag_length
    assert result.validation_metrics["rmse"] < 1e-8
    assert result.held_out_metrics["rmse"] < 1e-8
    assert result.all_dataset_predictions_path.exists()
