from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from spa_basic_estimators.estimators.accel_linear_least_squares import (
    load_accel_linear_least_squares_config,
    train_accel_linear_least_squares,
)
from spa_basic_estimators.estimators.pressure_accel_linear_least_squares import (
    load_pressure_accel_linear_least_squares_config,
    train_pressure_accel_linear_least_squares,
)
from spa_basic_estimators.estimators.pressure_linear_least_squares import (
    load_pressure_linear_least_squares_config,
    train_pressure_linear_least_squares,
)
from spa_basic_estimators.utils.data_loader import load_data_config, load_runs
from spa_basic_estimators.utils.splits import UNASSIGNED_SPLIT


def _write_model_config(path: Path, name: str, output_dir: str) -> None:
    payload = {
        "name": name,
        "fit_intercept": True,
        "output_dir": output_dir,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


@pytest.mark.parametrize(
    ("name", "output_dir", "loader_fn", "train_fn", "expected_features", "expected_feature_groups"),
    [
        (
            "pressure_linear_least_squares",
            "outputs/pressure_linear_least_squares",
            load_pressure_linear_least_squares_config,
            train_pressure_linear_least_squares,
            ["pressure"],
            None,
        ),
        (
            "accel_linear_least_squares",
            "outputs/accel_linear_least_squares",
            load_accel_linear_least_squares_config,
            train_accel_linear_least_squares,
            ["acc_x", "acc_y", "acc_z"],
            ["accel", "accel", "accel"],
        ),
        (
            "pressure_accel_linear_least_squares",
            "outputs/pressure_accel_linear_least_squares",
            load_pressure_accel_linear_least_squares_config,
            train_pressure_accel_linear_least_squares,
            ["pressure", "acc_x", "acc_y", "acc_z"],
            ["pressure", "accel", "accel", "accel"],
        ),
    ],
)
def test_linear_least_squares_smoke_run(
    synthetic_loader_case: dict[str, Path],
    name: str,
    output_dir: str,
    loader_fn,
    train_fn,
    expected_features: list[str],
    expected_feature_groups: list[str] | None,
) -> None:
    project_root = synthetic_loader_case["project_root"]
    model_config_path = project_root / "configs" / "models" / f"{name}.yaml"
    _write_model_config(model_config_path, name, output_dir)

    data_config = load_data_config(synthetic_loader_case["config_path"])
    model_config = loader_fn(model_config_path)
    runs = load_runs(data_config)

    result = train_fn(runs, data_config, model_config)

    assert result.feature_columns == expected_features
    assert result.validation_predictions["__split__"].unique().tolist() == ["val"]
    assert result.held_out_predictions["__split__"].unique().tolist() == ["held_out"]
    assert list(result.coefficient_table["feature"]) == expected_features
    if expected_feature_groups is not None:
        assert result.coefficient_table["feature_group"].tolist() == expected_feature_groups

    assert (result.artifact_dir / "linear_model.pkl").exists()
    assert (result.artifact_dir / "validation_predictions.csv").exists()
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
        expected_columns = set(expected_features) | {"phi_true", "phi_prediction", "phi_error"}
        assert expected_columns.issubset(per_run.columns)
        extra_run = store["/predictions/run_extra_1"]
        assert extra_run["split"].unique().tolist() == [UNASSIGNED_SPLIT]


def test_pressure_linear_least_squares_recovers_known_linear_mapping(
    synthetic_loader_case: dict[str, Path],
) -> None:
    intercept = 1.0
    pressure_weight = 2.0

    def phi_values(frame: pd.DataFrame) -> list[float]:
        return (intercept + pressure_weight * frame["pressure"]).tolist()

    train_frame, val_frame, test_frame = _build_shared_frames(phi_values)
    _overwrite_split_runs(synthetic_loader_case["h5_path"], train_frame, val_frame, test_frame)

    project_root = synthetic_loader_case["project_root"]
    model_config_path = project_root / "configs" / "models" / "pressure_linear_least_squares.yaml"
    _write_model_config(
        model_config_path,
        "pressure_linear_least_squares",
        "outputs/pressure_linear_least_squares",
    )

    data_config = load_data_config(synthetic_loader_case["config_path"])
    model_config = load_pressure_linear_least_squares_config(model_config_path)
    runs = load_runs(data_config)

    result = train_pressure_linear_least_squares(runs, data_config, model_config)

    assert abs(float(result.model.intercept_) - intercept) < 1e-6
    assert abs(float(result.model.coef_[0]) - pressure_weight) < 1e-6
    assert result.validation_metrics["rmse"] < 1e-10
    assert result.held_out_metrics["rmse"] < 1e-10


def test_accel_linear_least_squares_recovers_known_linear_mapping(
    synthetic_loader_case: dict[str, Path],
) -> None:
    coefficients = {
        "intercept": 1.0,
        "acc_x": -3.0,
        "acc_y": 4.0,
        "acc_z": -5.0,
    }

    def phi_values(frame: pd.DataFrame) -> list[float]:
        return (
            coefficients["intercept"]
            + coefficients["acc_x"] * frame["acc_x"]
            + coefficients["acc_y"] * frame["acc_y"]
            + coefficients["acc_z"] * frame["acc_z"]
        ).tolist()

    train_frame, val_frame, test_frame = _build_shared_frames(phi_values)
    _overwrite_split_runs(synthetic_loader_case["h5_path"], train_frame, val_frame, test_frame)

    project_root = synthetic_loader_case["project_root"]
    model_config_path = project_root / "configs" / "models" / "accel_linear_least_squares.yaml"
    _write_model_config(
        model_config_path,
        "accel_linear_least_squares",
        "outputs/accel_linear_least_squares",
    )

    data_config = load_data_config(synthetic_loader_case["config_path"])
    model_config = load_accel_linear_least_squares_config(model_config_path)
    runs = load_runs(data_config)

    result = train_accel_linear_least_squares(runs, data_config, model_config)

    assert abs(float(result.model.intercept_) - coefficients["intercept"]) < 1e-6
    assert abs(float(result.model.coef_[0]) - coefficients["acc_x"]) < 1e-6
    assert abs(float(result.model.coef_[1]) - coefficients["acc_y"]) < 1e-6
    assert abs(float(result.model.coef_[2]) - coefficients["acc_z"]) < 1e-6
    assert result.validation_metrics["rmse"] < 1e-10
    assert result.held_out_metrics["rmse"] < 1e-10


def test_pressure_accel_linear_least_squares_recovers_known_linear_mapping(
    synthetic_loader_case: dict[str, Path],
) -> None:
    coefficients = {
        "intercept": 1.0,
        "pressure": 2.0,
        "acc_x": -3.0,
        "acc_y": 4.0,
        "acc_z": -5.0,
    }

    def phi_values(frame: pd.DataFrame) -> list[float]:
        return (
            coefficients["intercept"]
            + coefficients["pressure"] * frame["pressure"]
            + coefficients["acc_x"] * frame["acc_x"]
            + coefficients["acc_y"] * frame["acc_y"]
            + coefficients["acc_z"] * frame["acc_z"]
        ).tolist()

    train_frame, val_frame, test_frame = _build_shared_frames(phi_values)
    _overwrite_split_runs(synthetic_loader_case["h5_path"], train_frame, val_frame, test_frame)

    project_root = synthetic_loader_case["project_root"]
    model_config_path = (
        project_root / "configs" / "models" / "pressure_accel_linear_least_squares.yaml"
    )
    _write_model_config(
        model_config_path,
        "pressure_accel_linear_least_squares",
        "outputs/pressure_accel_linear_least_squares",
    )

    data_config = load_data_config(synthetic_loader_case["config_path"])
    model_config = load_pressure_accel_linear_least_squares_config(model_config_path)
    runs = load_runs(data_config)

    result = train_pressure_accel_linear_least_squares(runs, data_config, model_config)

    assert abs(float(result.model.intercept_) - coefficients["intercept"]) < 1e-6
    assert abs(float(result.model.coef_[0]) - coefficients["pressure"]) < 1e-6
    assert abs(float(result.model.coef_[1]) - coefficients["acc_x"]) < 1e-6
    assert abs(float(result.model.coef_[2]) - coefficients["acc_y"]) < 1e-6
    assert abs(float(result.model.coef_[3]) - coefficients["acc_z"]) < 1e-6
    assert result.validation_metrics["rmse"] < 1e-10
    assert result.held_out_metrics["rmse"] < 1e-10


def _build_shared_frames(phi_values_fn):
    train_frame = pd.DataFrame(
        {
            "pressure": [-1.0, -0.5, 0.0, 0.5, 1.0, 0.25],
            "acc_x": [0.2, -0.4, 0.6, -0.8, 0.1, -0.2],
            "acc_y": [-0.3, 0.5, -0.7, 0.9, -0.1, 0.4],
            "acc_z": [0.7, -0.6, 0.5, -0.4, 0.3, -0.2],
            "Time": [0.1 * index for index in range(6)],
        }
    )
    train_frame["phi"] = phi_values_fn(train_frame)

    val_frame = pd.DataFrame(
        {
            "pressure": [-0.75, 0.75, 0.1],
            "acc_x": [0.15, -0.35, 0.45],
            "acc_y": [-0.25, 0.55, -0.15],
            "acc_z": [0.65, -0.45, 0.25],
            "Time": [0.0, 0.1, 0.2],
        }
    )
    val_frame["phi"] = phi_values_fn(val_frame)

    test_frame = pd.DataFrame(
        {
            "pressure": [-0.25, 0.25, -0.9],
            "acc_x": [0.05, -0.15, 0.3],
            "acc_y": [-0.05, 0.35, -0.45],
            "acc_z": [0.15, -0.05, 0.55],
            "Time": [0.0, 0.1, 0.2],
        }
    )
    test_frame["phi"] = phi_values_fn(test_frame)
    return train_frame, val_frame, test_frame


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
