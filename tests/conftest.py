from __future__ import annotations

import shutil
from uuid import uuid4
from pathlib import Path

import pandas as pd
import pytest
import yaml


def _run_frame(include_phi: bool = True, include_missing: bool = False) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "pressure": [0.1, 0.2, 0.3],
            "acc_x": [0.0, 0.1, 0.2],
            "acc_y": [0.3, 0.4, 0.5],
            "acc_z": [0.6, 0.7, 0.8],
            "Time": [0.0, 0.1, 0.2],
        }
    )
    if include_phi:
        frame["phi"] = [0.9, 1.0, 1.1]
    if include_missing:
        frame.loc[1, "acc_y"] = None
    return frame


def _write_config(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


@pytest.fixture()
def synthetic_loader_case() -> dict[str, Path]:
    artifacts_root = Path(__file__).resolve().parent / ".artifacts"
    artifacts_root.mkdir(parents=True, exist_ok=True)
    project_root = artifacts_root / f"loader_case_{uuid4().hex}"
    project_root.mkdir()
    data_dir = project_root / "data"
    configs_dir = project_root / "configs"
    data_dir.mkdir()
    configs_dir.mkdir()

    try:
        h5_path = data_dir / "synthetic_runs.h5"
        meta_runs = pd.DataFrame(
            {
                "run_id": ["source_train", "source_val", "source_test", "source_extra"],
                "hdf5_key": [
                    "/runs/run_train_1",
                    "/runs/run_val_1",
                    "/runs/run_test_1",
                    "/runs/run_extra_1",
                ],
            }
        )
        scaler_parameters = pd.DataFrame(
            {
                "column": ["pressure", "acc_x", "acc_y", "acc_z", "phi"],
                "min": [0.0, -10.0, -10.0, -10.0, -10.0],
                "max": [100.0, 10.0, 10.0, 10.0, 10.0],
                "range": [100.0, 20.0, 20.0, 20.0, 20.0],
                "is_constant": [False, False, False, False, False],
            }
        )

        with pd.HDFStore(h5_path, mode="w") as store:
            store.put("/runs/run_train_1", _run_frame(), format="fixed")
            store.put("/runs/run_val_1", _run_frame(), format="fixed")
            store.put("/runs/run_test_1", _run_frame(), format="fixed")
            store.put("/runs/run_extra_1", _run_frame(), format="fixed")
            store.put("/meta/runs", meta_runs, format="fixed")
            store.put("/meta/scaler_parameters", scaler_parameters, format="fixed")

        schema_doc = data_dir / "schema.md"
        schema_doc.write_text("# synthetic schema\n", encoding="utf-8")
        source_experiment = data_dir / "baseline.yaml"
        source_experiment.write_text("name: synthetic\n", encoding="utf-8")

        config = {
            "storage": {
                "format": "hdf5",
                "path": "data/synthetic_runs.h5",
                "source_experiment_config": "data/baseline.yaml",
                "runs": {
                    "key_pattern": "/runs/<run_id>",
                    "meta_runs_key": "/meta/runs",
                },
                "meta": {
                    "runs": "/meta/runs",
                    "scaler_parameters": "/meta/scaler_parameters",
                },
            },
            "schema": {
                "schema_doc": "data/schema.md",
                "split_id_convention": "normalized_hdf5_run_key",
                "run_id_column": "run_id",
                "run_key_column": "hdf5_key",
                "time_column": "Time",
                "target_column": "phi",
                "pressure_columns": ["pressure"],
                "accel_columns": ["acc_x", "acc_y", "acc_z"],
                "required_run_columns": ["pressure", "acc_x", "acc_y", "acc_z", "phi", "Time"],
            },
            "conventions": {
                "time_units": "s",
                "target_units": "rad",
            },
            "splits": {
                "train": ["run_train_1"],
                "val": ["run_val_1"],
                "held_out": ["run_test_1"],
            },
            "validation": {
                "fail_on_missing_columns": True,
                "fail_on_missing_values": True,
                "require_unique_run_ids": True,
            },
        }
        config_path = configs_dir / "data.yaml"
        _write_config(config_path, config)

        yield {
            "project_root": project_root,
            "config_path": config_path,
            "h5_path": h5_path,
        }
    finally:
        shutil.rmtree(project_root, ignore_errors=True)
