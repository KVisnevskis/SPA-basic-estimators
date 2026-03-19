from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from spa_basic_estimators.estimators.pressure_ridge_quadratic import (
    load_pressure_ridge_quadratic_config,
    train_pressure_ridge_quadratic,
)
from spa_basic_estimators.utils.data_loader import load_data_config, load_runs


def _write_model_config(path: Path, output_dir: str, alpha_grid: list[float]) -> None:
    payload = {
        "name": "pressure_ridge_quadratic",
        "degree": 2,
        "fit_intercept": True,
        "alpha_grid": alpha_grid,
        "output_dir": output_dir,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_pressure_ridge_quadratic_smoke_run(
    synthetic_loader_case: dict[str, Path],
) -> None:
    project_root = synthetic_loader_case["project_root"]
    model_config_path = project_root / "configs" / "models" / "pressure_ridge_quadratic.yaml"
    _write_model_config(model_config_path, "outputs/pressure_ridge_quadratic", [1e-6, 1e-3, 1.0])

    data_config = load_data_config(synthetic_loader_case["config_path"])
    model_config = load_pressure_ridge_quadratic_config(model_config_path)
    runs = load_runs(data_config)

    result = train_pressure_ridge_quadratic(runs, data_config, model_config)

    assert result.selected_alpha in model_config.alpha_grid
    assert result.feature_columns == ["pressure", "pressure^2"]
    assert result.validation_predictions["__split__"].unique().tolist() == ["val"]
    assert result.held_out_predictions["__split__"].unique().tolist() == ["held_out"]
    assert list(result.coefficient_table["feature"]) == ["pressure", "pressure^2"]

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
            "/predictions/run_test_1",
            "/predictions/run_train_1",
            "/predictions/run_val_1",
        }
        per_run = store["/predictions/run_train_1"]
        assert {"pressure", "phi_true", "phi_prediction", "phi_error"}.issubset(per_run.columns)
        assert abs(float(per_run["pressure"].iloc[0]) - 55.0) < 1e-9


def test_pressure_ridge_quadratic_fits_degree_two_relationship(
    synthetic_loader_case: dict[str, Path],
) -> None:
    train_pressure = [-1.0, -0.5, 0.0, 0.5, 1.0]
    val_pressure = [-0.75, 0.75]
    test_pressure = [-0.25, 0.25]

    def phi_values(pressure_values: list[float]) -> list[float]:
        return [1.0 + 2.0 * value + 3.0 * (value**2) for value in pressure_values]

    h5_path = synthetic_loader_case["h5_path"]
    with pd.HDFStore(h5_path, mode="a") as store:
        for key, pressure_values in {
            "/runs/run_train_1": train_pressure,
            "/runs/run_val_1": val_pressure,
            "/runs/run_test_1": test_pressure,
        }.items():
            store.put(
                key,
                pd.DataFrame(
                    {
                        "pressure": pressure_values,
                        "acc_x": [0.0] * len(pressure_values),
                        "acc_y": [0.0] * len(pressure_values),
                        "acc_z": [0.0] * len(pressure_values),
                        "phi": phi_values(pressure_values),
                        "Time": [0.1 * index for index in range(len(pressure_values))],
                    }
                ),
                format="fixed",
            )

    project_root = synthetic_loader_case["project_root"]
    model_config_path = project_root / "configs" / "models" / "pressure_ridge_quadratic.yaml"
    _write_model_config(model_config_path, "outputs/pressure_ridge_quadratic", [1e-12])

    data_config = load_data_config(synthetic_loader_case["config_path"])
    model_config = load_pressure_ridge_quadratic_config(model_config_path)
    runs = load_runs(data_config)

    result = train_pressure_ridge_quadratic(runs, data_config, model_config)

    assert abs(float(result.model.intercept_) - 1.0) < 1e-6
    assert abs(float(result.model.coef_[0]) - 2.0) < 1e-6
    assert abs(float(result.model.coef_[1]) - 3.0) < 1e-6
    assert result.all_dataset_predictions_path.exists()
