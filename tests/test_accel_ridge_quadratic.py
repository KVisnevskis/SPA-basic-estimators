from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from spa_basic_estimators.estimators.accel_ridge_quadratic import (
    load_accel_ridge_quadratic_config,
    train_accel_ridge_quadratic,
)
from spa_basic_estimators.utils.data_loader import load_data_config, load_runs
from spa_basic_estimators.utils.splits import UNASSIGNED_SPLIT


def _write_model_config(path: Path, output_dir: str, alpha_grid: list[float]) -> None:
    payload = {
        "name": "accel_ridge_quadratic",
        "degree": 2,
        "fit_intercept": True,
        "alpha_grid": alpha_grid,
        "output_dir": output_dir,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_accel_ridge_quadratic_smoke_run(
    synthetic_loader_case: dict[str, Path],
) -> None:
    project_root = synthetic_loader_case["project_root"]
    model_config_path = project_root / "configs" / "models" / "accel_ridge_quadratic.yaml"
    _write_model_config(model_config_path, "outputs/accel_ridge_quadratic", [1e-6, 1e-3, 1.0])

    data_config = load_data_config(synthetic_loader_case["config_path"])
    model_config = load_accel_ridge_quadratic_config(model_config_path)
    runs = load_runs(data_config)

    result = train_accel_ridge_quadratic(runs, data_config, model_config)

    expected_features = [
        "acc_x",
        "acc_y",
        "acc_z",
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
    assert set(result.coefficient_table["feature_group"]) == {"accel"}

    coefficient_table = result.coefficient_table.set_index("feature")
    assert coefficient_table.loc["acc_x", "term_type"] == "linear"
    assert coefficient_table.loc["acc_x^2", "term_type"] == "squared"
    assert coefficient_table.loc["acc_x acc_y", "term_type"] == "interaction"
    assert coefficient_table.loc["acc_x acc_y", "source_features"] == "acc_x, acc_y"

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
            "acc_x",
            "acc_y",
            "acc_z",
            "phi_true",
            "phi_prediction",
            "phi_error",
        }.issubset(per_run.columns)
        assert "pressure" not in per_run.columns
        extra_run = store["/predictions/run_extra_1"]
        assert extra_run["split"].unique().tolist() == [UNASSIGNED_SPLIT]


def test_accel_ridge_quadratic_fits_known_degree_two_relationship(
    synthetic_loader_case: dict[str, Path],
) -> None:
    rng = np.random.default_rng(11)

    def make_frame(num_rows: int, time_offset: float) -> pd.DataFrame:
        values = rng.uniform(-1.0, 1.0, size=(num_rows, 3))
        frame = pd.DataFrame(values, columns=["acc_x", "acc_y", "acc_z"])
        frame["pressure"] = rng.uniform(-1.0, 1.0, size=num_rows)
        frame["Time"] = time_offset + np.arange(num_rows, dtype=float) * 0.1
        frame["phi"] = (
            0.25
            + 1.5 * frame["acc_x"]
            - 2.5 * frame["acc_y"]
            + 3.5 * frame["acc_z"]
            + 4.5 * frame["acc_x"] ** 2
            - 5.5 * frame["acc_x"] * frame["acc_y"]
            + 6.5 * frame["acc_x"] * frame["acc_z"]
            + 7.5 * frame["acc_y"] ** 2
            - 8.5 * frame["acc_y"] * frame["acc_z"]
            + 9.5 * frame["acc_z"] ** 2
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
    model_config_path = project_root / "configs" / "models" / "accel_ridge_quadratic.yaml"
    _write_model_config(model_config_path, "outputs/accel_ridge_quadratic", [1e-12])

    data_config = load_data_config(synthetic_loader_case["config_path"])
    model_config = load_accel_ridge_quadratic_config(model_config_path)
    runs = load_runs(data_config)

    result = train_accel_ridge_quadratic(runs, data_config, model_config)

    assert result.validation_metrics["rmse"] < 1e-8
    assert result.held_out_metrics["rmse"] < 1e-8
    assert result.all_dataset_predictions_path.exists()
