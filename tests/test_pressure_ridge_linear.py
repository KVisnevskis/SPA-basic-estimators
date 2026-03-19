from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from spa_basic_estimators.estimators.pressure_ridge_linear import (
    load_pressure_ridge_linear_config,
    train_pressure_ridge_linear,
)
from spa_basic_estimators.utils.data_loader import load_data_config, load_runs


def _write_model_config(path: Path, output_dir: str) -> None:
    payload = {
        "name": "pressure_ridge_linear",
        "fit_intercept": True,
        "alpha_grid": [1e-6, 1e-3, 1.0],
        "output_dir": output_dir,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_pressure_ridge_linear_smoke_run(
    synthetic_loader_case: dict[str, Path],
) -> None:
    project_root = synthetic_loader_case["project_root"]
    model_config_path = project_root / "configs" / "models" / "pressure_ridge_linear.yaml"
    _write_model_config(model_config_path, "outputs/pressure_ridge_linear")

    data_config = load_data_config(synthetic_loader_case["config_path"])
    model_config = load_pressure_ridge_linear_config(model_config_path)
    runs = load_runs(data_config)

    result = train_pressure_ridge_linear(runs, data_config, model_config)

    assert result.selected_alpha in model_config.alpha_grid
    assert result.feature_columns == ["pressure"]
    assert result.validation_predictions["__split__"].unique().tolist() == ["val"]
    assert result.held_out_predictions["__split__"].unique().tolist() == ["held_out"]
    assert list(result.coefficient_table["feature"]) == ["pressure"]

    assert (result.artifact_dir / "ridge_model.pkl").exists()
    assert (result.artifact_dir / "validation_search.csv").exists()
    assert (result.artifact_dir / "held_out_predictions.csv").exists()
    assert (result.artifact_dir / "coefficient_table.csv").exists()
    assert not (result.artifact_dir / "input_scaler.pkl").exists()
    assert result.all_dataset_predictions_path.exists()

    with pd.HDFStore(result.all_dataset_predictions_path, mode="r") as store:
        assert set(store.keys()) == {
            "/meta/runs",
            "/predictions/run_test_1",
            "/predictions/run_train_1",
            "/predictions/run_val_1",
        }
        per_run = store["/predictions/run_train_1"]
        assert {"pressure", "phi_true", "phi_prediction", "phi_error"}.issubset(per_run.columns)
        assert abs(float(per_run["pressure"].iloc[0]) - 55.0) < 1e-9


def test_pressure_ridge_linear_uses_prescaled_input_directly(
    synthetic_loader_case: dict[str, Path],
) -> None:
    h5_path = synthetic_loader_case["h5_path"]
    with pd.HDFStore(h5_path, mode="a") as store:
        store.put(
            "/runs/run_train_1",
            pd.DataFrame(
                {
                    "pressure": [0.0, 1.0, 2.0],
                    "acc_x": [0.0, 0.0, 0.0],
                    "acc_y": [0.0, 0.0, 0.0],
                    "acc_z": [0.0, 0.0, 0.0],
                    "phi": [1.0, 3.0, 5.0],
                    "Time": [0.0, 0.1, 0.2],
                }
            ),
            format="fixed",
        )
        store.put(
            "/runs/run_val_1",
            pd.DataFrame(
                {
                    "pressure": [100.0, 101.0, 102.0],
                    "acc_x": [0.0, 0.0, 0.0],
                    "acc_y": [0.0, 0.0, 0.0],
                    "acc_z": [0.0, 0.0, 0.0],
                    "phi": [201.0, 203.0, 205.0],
                    "Time": [0.0, 0.1, 0.2],
                }
            ),
            format="fixed",
        )
        store.put(
            "/runs/run_test_1",
            pd.DataFrame(
                {
                    "pressure": [200.0, 201.0, 202.0],
                    "acc_x": [0.0, 0.0, 0.0],
                    "acc_y": [0.0, 0.0, 0.0],
                    "acc_z": [0.0, 0.0, 0.0],
                    "phi": [401.0, 403.0, 405.0],
                    "Time": [0.0, 0.1, 0.2],
                }
            ),
            format="fixed",
        )

    project_root = synthetic_loader_case["project_root"]
    model_config_path = project_root / "configs" / "models" / "pressure_ridge_linear.yaml"
    _write_model_config(model_config_path, "outputs/pressure_ridge_linear")

    data_config = load_data_config(synthetic_loader_case["config_path"])
    model_config = load_pressure_ridge_linear_config(model_config_path)
    runs = load_runs(data_config)

    result = train_pressure_ridge_linear(runs, data_config, model_config)

    assert abs(float(result.model.coef_[0]) - 2.0) < 1e-3
    assert abs(float(result.model.intercept_) - 1.0) < 1e-3
    assert result.all_dataset_predictions_path.exists()
