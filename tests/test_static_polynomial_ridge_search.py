from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from spa_basic_estimators.estimators.accel_ridge_polynomial_search import (
    load_accel_ridge_polynomial_search_config,
    train_accel_ridge_polynomial_search,
)
from spa_basic_estimators.estimators.pressure_accel_ridge_polynomial_search import (
    load_pressure_accel_ridge_polynomial_search_config,
    train_pressure_accel_ridge_polynomial_search,
)
from spa_basic_estimators.estimators.pressure_ridge_polynomial_search import (
    load_pressure_ridge_polynomial_search_config,
    train_pressure_ridge_polynomial_search,
)
from spa_basic_estimators.utils.data_loader import load_data_config, load_runs


def _write_model_config(
    path: Path,
    *,
    name: str,
    output_dir: str,
    alpha_grid: list[float],
    degree_grid: list[int],
) -> None:
    payload = {
        "name": name,
        "degree_grid": degree_grid,
        "fit_intercept": True,
        "alpha_grid": alpha_grid,
        "output_dir": output_dir,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_pressure_polynomial_search_smoke_and_degree_selection(
    synthetic_loader_case: dict[str, Path],
) -> None:
    train_pressure = [-1.0, -0.6, -0.2, 0.2, 0.6, 1.0]
    val_pressure = [-0.8, -0.4, 0.4, 0.8]
    test_pressure = [-0.7, -0.3, 0.3, 0.7]

    def phi_values(values: list[float]) -> list[float]:
        return [0.5 + 1.2 * value + 2.3 * (value**2) for value in values]

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
    model_config_path = project_root / "configs" / "models" / "pressure_ridge_polynomial_search.yaml"
    _write_model_config(
        model_config_path,
        name="pressure_ridge_polynomial_search",
        output_dir="outputs/pressure_ridge_polynomial_search",
        alpha_grid=[1e-12],
        degree_grid=[1, 2, 3],
    )

    data_config = load_data_config(synthetic_loader_case["config_path"])
    model_config = load_pressure_ridge_polynomial_search_config(model_config_path)
    runs = load_runs(data_config)

    result = train_pressure_ridge_polynomial_search(runs, data_config, model_config)

    assert result.selected_degree == 2
    assert result.selected_alpha == 1e-12
    assert list(result.validation_search.columns[:2]) == ["degree", "alpha"]
    assert set(result.validation_search["degree"]) == {1, 2, 3}
    assert "pressure^2" in result.feature_columns
    assert result.validation_metrics["rmse"] < 1e-8

    run_summary = json.loads((result.artifact_dir / "run_summary.json").read_text(encoding="utf-8"))
    assert run_summary["selected_degree"] == 2
    assert run_summary["degree_grid"] == [1, 2, 3]


def test_accel_polynomial_search_can_select_degree_three(
    synthetic_loader_case: dict[str, Path],
) -> None:
    rng = np.random.default_rng(19)

    def make_frame(num_rows: int, time_offset: float) -> pd.DataFrame:
        values = rng.uniform(-1.0, 1.0, size=(num_rows, 3))
        frame = pd.DataFrame(values, columns=["acc_x", "acc_y", "acc_z"])
        frame["pressure"] = rng.uniform(-1.0, 1.0, size=num_rows)
        frame["Time"] = time_offset + np.arange(num_rows, dtype=float) * 0.1
        frame["phi"] = (
            0.2
            + 1.1 * frame["acc_x"]
            - 0.7 * frame["acc_y"]
            + 0.4 * frame["acc_z"]
            + 2.5 * frame["acc_x"] ** 3
            - 1.4 * frame["acc_y"] ** 2
            + 0.8 * frame["acc_x"] * frame["acc_z"]
        )
        return frame

    h5_path = synthetic_loader_case["h5_path"]
    with pd.HDFStore(h5_path, mode="a") as store:
        store.put("/runs/run_train_1", make_frame(32, 0.0), format="fixed")
        store.put("/runs/run_val_1", make_frame(12, 10.0), format="fixed")
        store.put("/runs/run_test_1", make_frame(12, 20.0), format="fixed")

    project_root = synthetic_loader_case["project_root"]
    model_config_path = project_root / "configs" / "models" / "accel_ridge_polynomial_search.yaml"
    _write_model_config(
        model_config_path,
        name="accel_ridge_polynomial_search",
        output_dir="outputs/accel_ridge_polynomial_search",
        alpha_grid=[1e-12],
        degree_grid=[1, 2, 3],
    )

    data_config = load_data_config(synthetic_loader_case["config_path"])
    model_config = load_accel_ridge_polynomial_search_config(model_config_path)
    runs = load_runs(data_config)

    result = train_accel_ridge_polynomial_search(runs, data_config, model_config)

    assert result.selected_degree == 3
    assert result.validation_metrics["rmse"] < 1e-8
    assert result.held_out_metrics["rmse"] < 1e-8
    coefficient_table = result.coefficient_table.set_index("feature")
    assert coefficient_table.loc["acc_x^3", "term_type"] == "polynomial_power"
    assert coefficient_table.loc["acc_x acc_z", "term_type"] == "interaction"
    assert coefficient_table.loc["acc_x acc_z", "feature_group"] == "accel"


def test_pressure_accel_polynomial_search_tracks_selected_degree_and_groups(
    synthetic_loader_case: dict[str, Path],
) -> None:
    rng = np.random.default_rng(23)

    def make_frame(num_rows: int, time_offset: float) -> pd.DataFrame:
        values = rng.uniform(-1.0, 1.0, size=(num_rows, 4))
        frame = pd.DataFrame(values, columns=["pressure", "acc_x", "acc_y", "acc_z"])
        frame["Time"] = time_offset + np.arange(num_rows, dtype=float) * 0.1
        frame["phi"] = (
            -0.1
            + 0.8 * frame["pressure"]
            - 0.6 * frame["acc_x"]
            + 0.4 * frame["acc_y"]
            + 1.7 * frame["pressure"] ** 2
            + 1.2 * frame["pressure"] * frame["acc_x"]
            - 0.9 * frame["acc_y"] * frame["acc_z"]
        )
        return frame

    h5_path = synthetic_loader_case["h5_path"]
    with pd.HDFStore(h5_path, mode="a") as store:
        store.put("/runs/run_train_1", make_frame(32, 0.0), format="fixed")
        store.put("/runs/run_val_1", make_frame(12, 10.0), format="fixed")
        store.put("/runs/run_test_1", make_frame(12, 20.0), format="fixed")

    project_root = synthetic_loader_case["project_root"]
    model_config_path = (
        project_root / "configs" / "models" / "pressure_accel_ridge_polynomial_search.yaml"
    )
    _write_model_config(
        model_config_path,
        name="pressure_accel_ridge_polynomial_search",
        output_dir="outputs/pressure_accel_ridge_polynomial_search",
        alpha_grid=[1e-12],
        degree_grid=[1, 2, 3],
    )

    data_config = load_data_config(synthetic_loader_case["config_path"])
    model_config = load_pressure_accel_ridge_polynomial_search_config(model_config_path)
    runs = load_runs(data_config)

    result = train_pressure_accel_ridge_polynomial_search(runs, data_config, model_config)

    assert result.selected_degree == 2
    assert result.validation_metrics["rmse"] < 1e-8
    coefficient_table = result.coefficient_table.set_index("feature")
    assert coefficient_table.loc["pressure^2", "feature_group"] == "pressure"
    assert coefficient_table.loc["pressure^2", "term_type"] == "polynomial_power"
    assert coefficient_table.loc["pressure acc_x", "feature_group"] == "mixed"
    assert coefficient_table.loc["acc_y acc_z", "feature_group"] == "accel"
    assert result.all_dataset_predictions_path.exists()
