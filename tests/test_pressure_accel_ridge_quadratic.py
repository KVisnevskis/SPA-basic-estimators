from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from spa_basic_estimators.estimators.pressure_accel_ridge_quadratic import (
    load_pressure_accel_ridge_quadratic_config,
    train_pressure_accel_ridge_quadratic,
)
from spa_basic_estimators.utils.data_loader import load_data_config, load_runs
from spa_basic_estimators.utils.splits import UNASSIGNED_SPLIT


def _write_model_config(path: Path, output_dir: str, alpha_grid: list[float]) -> None:
    payload = {
        "name": "pressure_accel_ridge_quadratic",
        "degree": 2,
        "fit_intercept": True,
        "alpha_grid": alpha_grid,
        "output_dir": output_dir,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_pressure_accel_ridge_quadratic_smoke_run(
    synthetic_loader_case: dict[str, Path],
) -> None:
    project_root = synthetic_loader_case["project_root"]
    model_config_path = (
        project_root / "configs" / "models" / "pressure_accel_ridge_quadratic.yaml"
    )
    _write_model_config(
        model_config_path,
        "outputs/pressure_accel_ridge_quadratic",
        [1e-6, 1e-3, 1.0],
    )

    data_config = load_data_config(synthetic_loader_case["config_path"])
    model_config = load_pressure_accel_ridge_quadratic_config(model_config_path)
    runs = load_runs(data_config)

    result = train_pressure_accel_ridge_quadratic(runs, data_config, model_config)

    expected_features = [
        "pressure",
        "acc_x",
        "acc_y",
        "acc_z",
        "pressure^2",
        "pressure acc_x",
        "pressure acc_y",
        "pressure acc_z",
        "acc_x^2",
        "acc_x acc_y",
        "acc_x acc_z",
        "acc_y^2",
        "acc_y acc_z",
        "acc_z^2",
    ]

    assert result.selected_alpha in model_config.alpha_grid
    assert result.feature_columns == expected_features
    assert result.validation_predictions["__split__"].unique().tolist() == ["val"]
    assert result.held_out_predictions["__split__"].unique().tolist() == ["held_out"]
    assert list(result.coefficient_table["feature"]) == expected_features

    coefficient_table = result.coefficient_table.set_index("feature")
    assert coefficient_table.loc["pressure", "feature_group"] == "pressure"
    assert coefficient_table.loc["pressure", "term_type"] == "linear"
    assert coefficient_table.loc["pressure^2", "feature_group"] == "pressure"
    assert coefficient_table.loc["pressure^2", "term_type"] == "squared"
    assert coefficient_table.loc["acc_x acc_y", "feature_group"] == "accel"
    assert coefficient_table.loc["acc_x acc_y", "term_type"] == "interaction"
    assert coefficient_table.loc["pressure acc_x", "feature_group"] == "mixed"
    assert coefficient_table.loc["pressure acc_x", "term_type"] == "interaction"

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
        }.issubset(per_run.columns)
        extra_run = store["/predictions/run_extra_1"]
        assert extra_run["split"].unique().tolist() == [UNASSIGNED_SPLIT]


def test_pressure_accel_ridge_quadratic_fits_known_degree_two_relationship(
    synthetic_loader_case: dict[str, Path],
) -> None:
    rng = np.random.default_rng(7)

    def make_frame(num_rows: int, time_offset: float) -> pd.DataFrame:
        values = rng.uniform(-1.0, 1.0, size=(num_rows, 4))
        frame = pd.DataFrame(
            values,
            columns=["pressure", "acc_x", "acc_y", "acc_z"],
        )
        frame["Time"] = time_offset + np.arange(num_rows, dtype=float) * 0.1
        frame["phi"] = (
            1.0
            + 2.0 * frame["pressure"]
            - 3.0 * frame["acc_x"]
            + 4.0 * frame["acc_y"]
            - 5.0 * frame["acc_z"]
            + 6.0 * frame["pressure"] ** 2
            + 7.0 * frame["pressure"] * frame["acc_x"]
            - 8.0 * frame["pressure"] * frame["acc_y"]
            + 9.0 * frame["pressure"] * frame["acc_z"]
            - 10.0 * frame["acc_x"] ** 2
            + 11.0 * frame["acc_x"] * frame["acc_y"]
            - 12.0 * frame["acc_x"] * frame["acc_z"]
            + 13.0 * frame["acc_y"] ** 2
            - 14.0 * frame["acc_y"] * frame["acc_z"]
            + 15.0 * frame["acc_z"] ** 2
        )
        return frame

    train_frame = make_frame(24, 0.0)
    val_frame = make_frame(8, 10.0)
    test_frame = make_frame(8, 20.0)

    h5_path = synthetic_loader_case["h5_path"]
    with pd.HDFStore(h5_path, mode="a") as store:
        store.put("/runs/run_train_1", train_frame, format="fixed")
        store.put("/runs/run_val_1", val_frame, format="fixed")
        store.put("/runs/run_test_1", test_frame, format="fixed")

    project_root = synthetic_loader_case["project_root"]
    model_config_path = (
        project_root / "configs" / "models" / "pressure_accel_ridge_quadratic.yaml"
    )
    _write_model_config(model_config_path, "outputs/pressure_accel_ridge_quadratic", [1e-12])

    data_config = load_data_config(synthetic_loader_case["config_path"])
    model_config = load_pressure_accel_ridge_quadratic_config(model_config_path)
    runs = load_runs(data_config)

    result = train_pressure_accel_ridge_quadratic(runs, data_config, model_config)

    assert result.validation_metrics["rmse"] < 1e-8
    assert result.held_out_metrics["rmse"] < 1e-8
    assert result.all_dataset_predictions_path.exists()
