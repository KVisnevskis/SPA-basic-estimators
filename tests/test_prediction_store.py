from __future__ import annotations

from pathlib import Path

import yaml

from spa_basic_estimators.estimators.pressure_ridge_linear import (
    load_pressure_ridge_linear_config,
    train_pressure_ridge_linear,
)
from spa_basic_estimators.evaluation.prediction_store import (
    choose_x_axis_column,
    compute_run_rmse,
    default_selected_columns,
    discover_prediction_stores,
    list_plottable_columns,
    load_run_catalog,
    load_run_frame,
)
from spa_basic_estimators.utils.data_loader import load_data_config, load_runs
from spa_basic_estimators.utils.splits import UNASSIGNED_SPLIT


def _write_model_config(path: Path, output_dir: str) -> None:
    payload = {
        "name": "pressure_ridge_linear",
        "fit_intercept": True,
        "alpha_grid": [1e-6, 1e-3, 1.0],
        "output_dir": output_dir,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_prediction_store_helpers_discover_and_load_runs(
    synthetic_loader_case: dict[str, Path],
) -> None:
    project_root = synthetic_loader_case["project_root"]
    model_config_path = project_root / "configs" / "models" / "pressure_ridge_linear.yaml"
    _write_model_config(model_config_path, "outputs/pressure_ridge_linear")

    data_config = load_data_config(synthetic_loader_case["config_path"])
    model_config = load_pressure_ridge_linear_config(model_config_path)
    runs = load_runs(data_config)
    result = train_pressure_ridge_linear(runs, data_config, model_config)

    stores = discover_prediction_stores(project_root / "outputs")

    assert len(stores) == 1
    assert stores[0].model_name == "pressure_ridge_linear"
    assert stores[0].store_path == result.all_dataset_predictions_path

    catalog = load_run_catalog(stores[0].store_path)
    assert catalog["run_id"].tolist() == ["run_test_1", "run_extra_1", "run_train_1", "run_val_1"]
    assert set(catalog["split"]) == {"held_out", UNASSIGNED_SPLIT, "train", "val"}

    run_frame = load_run_frame(stores[0].store_path, "run_val_1")
    assert {"pressure", "phi_true", "phi_prediction", "phi_error"}.issubset(run_frame.columns)
    assert choose_x_axis_column(run_frame) == "Time"

    plottable_columns = list_plottable_columns(run_frame)
    assert "Time" in plottable_columns
    assert "pressure" in plottable_columns
    assert "phi_true" in plottable_columns
    assert "phi_prediction" in plottable_columns

    assert default_selected_columns(run_frame) == ["phi_true", "phi_prediction"]

    expected_rmse = float((run_frame["phi_error"].pow(2).mean()) ** 0.5)
    assert compute_run_rmse(run_frame) == expected_rmse

    extra_frame = load_run_frame(stores[0].store_path, "run_extra_1")
    assert extra_frame["split"].unique().tolist() == [UNASSIGNED_SPLIT]


def test_load_run_frame_rejects_unknown_run(synthetic_loader_case: dict[str, Path]) -> None:
    project_root = synthetic_loader_case["project_root"]
    model_config_path = project_root / "configs" / "models" / "pressure_ridge_linear.yaml"
    _write_model_config(model_config_path, "outputs/pressure_ridge_linear")

    data_config = load_data_config(synthetic_loader_case["config_path"])
    model_config = load_pressure_ridge_linear_config(model_config_path)
    runs = load_runs(data_config)
    result = train_pressure_ridge_linear(runs, data_config, model_config)

    try:
        load_run_frame(result.all_dataset_predictions_path, "missing_run")
        raise AssertionError("Expected load_run_frame to fail for an unknown run id")
    except KeyError as exc:
        assert "missing_run" in str(exc)
