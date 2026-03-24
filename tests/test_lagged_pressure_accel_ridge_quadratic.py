from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import PolynomialFeatures

from spa_basic_estimators.estimators.lagged_pressure_accel_ridge_quadratic import (
    load_lagged_pressure_accel_ridge_quadratic_config,
    train_lagged_pressure_accel_ridge_quadratic,
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
        "name": "lagged_pressure_accel_ridge_quadratic",
        "degree": 2,
        "fit_intercept": True,
        "lag_grid": lag_grid,
        "alpha_grid": alpha_grid,
        "output_dir": output_dir,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_lagged_pressure_accel_ridge_quadratic_smoke_run(
    synthetic_loader_case: dict[str, Path],
) -> None:
    lag_length = 2
    project_root = synthetic_loader_case["project_root"]
    model_config_path = (
        project_root / "configs" / "models" / "lagged_pressure_accel_ridge_quadratic.yaml"
    )
    _write_model_config(
        model_config_path,
        "outputs/lagged_pressure_accel_ridge_quadratic",
        [lag_length],
        [1e-6, 1e-3, 1.0],
    )

    data_config = load_data_config(synthetic_loader_case["config_path"])
    model_config = load_lagged_pressure_accel_ridge_quadratic_config(model_config_path)
    runs = load_runs(data_config)

    result = train_lagged_pressure_accel_ridge_quadratic(runs, data_config, model_config)

    lagged_features = [
        "pressure_lag_0",
        "acc_x_lag_0",
        "acc_y_lag_0",
        "acc_z_lag_0",
        "pressure_lag_1",
        "acc_x_lag_1",
        "acc_y_lag_1",
        "acc_z_lag_1",
    ]
    polynomial = PolynomialFeatures(degree=2, include_bias=False)
    polynomial.fit(np.zeros((1, len(lagged_features))))
    expected_features = polynomial.get_feature_names_out(lagged_features).tolist()

    assert result.selected_lag == lag_length
    assert result.selected_alpha in model_config.alpha_grid
    assert result.raw_feature_columns == ["pressure", "acc_x", "acc_y", "acc_z"]
    assert result.feature_columns == expected_features
    assert result.validation_predictions["__split__"].unique().tolist() == ["val"]
    assert result.held_out_predictions["__split__"].unique().tolist() == ["held_out"]
    assert result.validation_predictions["sample_index"].min() == lag_length - 1
    assert result.held_out_predictions["sample_index"].min() == lag_length - 1

    coefficient_table = result.coefficient_table.set_index("feature")
    assert coefficient_table.loc["pressure_lag_0", "term_type"] == "linear"
    assert coefficient_table.loc["pressure_lag_0^2", "term_type"] == "squared"
    assert coefficient_table.loc["acc_x_lag_0 acc_y_lag_1", "term_type"] == "interaction"
    assert coefficient_table.loc["pressure_lag_0^2", "feature_group"] == "pressure"
    assert coefficient_table.loc["acc_x_lag_0 acc_y_lag_1", "feature_group"] == "accel"
    assert coefficient_table.loc["pressure_lag_0 acc_x_lag_1", "feature_group"] == "mixed"
    assert coefficient_table.loc["pressure_lag_0 acc_x_lag_1", "source_lags"] == "0, 1"

    assert (result.artifact_dir / "ridge_model.pkl").exists()
    assert (result.artifact_dir / "polynomial_transformer.pkl").exists()
    assert (result.artifact_dir / "validation_search.csv").exists()
    assert (result.artifact_dir / "held_out_predictions.csv").exists()
    assert (result.artifact_dir / "coefficient_table.csv").exists()
    assert not (result.artifact_dir / "input_scaler.pkl").exists()
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
        assert len(per_run) == 2
        assert int(per_run["sample_index"].iloc[0]) == lag_length - 1
        extra_run = store["/predictions/run_extra_1"]
        assert extra_run["split"].unique().tolist() == [UNASSIGNED_SPLIT]


def test_lagged_pressure_accel_ridge_quadratic_fits_known_degree_two_history_relationship(
    synthetic_loader_case: dict[str, Path],
) -> None:
    lag_length = 2
    rng = np.random.default_rng(23)

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
                0.5
                + 1.5 * frame.loc[t, "pressure"]
                - 2.5 * frame.loc[t, "acc_x"]
                + 3.5 * frame.loc[t - 1, "pressure"]
                - 4.5 * frame.loc[t - 1, "acc_y"]
                + 5.5 * frame.loc[t, "pressure"] ** 2
                - 6.5 * frame.loc[t, "pressure"] * frame.loc[t, "acc_x"]
                + 7.5 * frame.loc[t, "pressure"] * frame.loc[t - 1, "pressure"]
                - 8.5 * frame.loc[t - 1, "acc_z"] ** 2
            )
        frame["phi"] = phi
        return frame

    train_frame = make_frame(96, 0.0)
    val_frame = make_frame(24, 10.0)
    test_frame = make_frame(24, 20.0)

    h5_path = synthetic_loader_case["h5_path"]
    with pd.HDFStore(h5_path, mode="a") as store:
        store.put("/runs/run_train_1", train_frame, format="fixed")
        store.put("/runs/run_val_1", val_frame, format="fixed")
        store.put("/runs/run_test_1", test_frame, format="fixed")

    project_root = synthetic_loader_case["project_root"]
    model_config_path = (
        project_root / "configs" / "models" / "lagged_pressure_accel_ridge_quadratic.yaml"
    )
    _write_model_config(
        model_config_path,
        "outputs/lagged_pressure_accel_ridge_quadratic",
        [lag_length],
        [1e-12],
    )

    data_config = load_data_config(synthetic_loader_case["config_path"])
    model_config = load_lagged_pressure_accel_ridge_quadratic_config(model_config_path)
    runs = load_runs(data_config)

    result = train_lagged_pressure_accel_ridge_quadratic(runs, data_config, model_config)

    assert result.selected_lag == lag_length
    assert result.validation_metrics["rmse"] < 1e-8
    assert result.held_out_metrics["rmse"] < 1e-8
    assert result.all_dataset_predictions_path.exists()
