from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from spa_basic_estimators.utils.data_loader import load_data_config, load_runs
from spa_basic_estimators.utils.splits import HDF5_KEY_COLUMN, SPLIT_COLUMN


def test_load_runs_from_hdf5_assigns_split_labels(
    synthetic_loader_case: dict[str, Path],
) -> None:
    config = load_data_config(synthetic_loader_case["config_path"])
    runs = load_runs(config)

    assert set(runs) == {"run_train_1", "run_val_1", "run_test_1"}
    assert runs["run_train_1"][SPLIT_COLUMN].unique().tolist() == ["train"]
    assert runs["run_val_1"][SPLIT_COLUMN].unique().tolist() == ["val"]
    assert runs["run_test_1"][SPLIT_COLUMN].unique().tolist() == ["held_out"]
    assert runs["run_train_1"][HDF5_KEY_COLUMN].unique().tolist() == ["/runs/run_train_1"]
    assert runs["run_train_1"]["run_id"].unique().tolist() == ["run_train_1"]


def test_load_runs_fails_on_missing_required_column(
    synthetic_loader_case: dict[str, Path],
) -> None:
    h5_path = synthetic_loader_case["h5_path"]
    with pd.HDFStore(h5_path, mode="a") as store:
        bad_frame = pd.DataFrame(
            {
                "pressure": [0.1, 0.2],
                "acc_x": [0.0, 0.1],
                "acc_y": [0.3, 0.4],
                "acc_z": [0.5, 0.6],
                "Time": [0.0, 0.1],
            }
        )
        store.put("/runs/run_train_1", bad_frame, format="fixed")

    config = load_data_config(synthetic_loader_case["config_path"])
    with pytest.raises(ValueError, match="missing required columns"):
        load_runs(config)


def test_load_data_config_fails_on_duplicate_run_ids_across_splits(
    synthetic_loader_case: dict[str, Path],
) -> None:
    config_path = synthetic_loader_case["config_path"]
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["splits"]["val"] = ["run_train_1"]
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="appears in both"):
        load_data_config(config_path)


def test_load_runs_fails_on_duplicate_canonical_run_ids_in_metadata(
    synthetic_loader_case: dict[str, Path],
) -> None:
    h5_path = synthetic_loader_case["h5_path"]
    duplicate_meta = pd.DataFrame(
        {
            "run_id": ["source_train", "source_train_dup", "source_test"],
            "hdf5_key": ["/runs/run_train_1", "/runs/run_train_1", "/runs/run_test_1"],
        }
    )
    with pd.HDFStore(h5_path, mode="a") as store:
        store.put("/meta/runs", duplicate_meta, format="fixed")

    config_path = synthetic_loader_case["config_path"]
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["splits"] = {
        "train": ["run_train_1"],
        "val": ["run_test_1"],
        "held_out": [],
    }
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    config = load_data_config(config_path)
    with pytest.raises(ValueError, match="Duplicate canonical run ID"):
        load_runs(config)


def test_load_runs_fails_fast_on_missing_values_by_default(
    synthetic_loader_case: dict[str, Path],
) -> None:
    h5_path = synthetic_loader_case["h5_path"]
    with pd.HDFStore(h5_path, mode="a") as store:
        frame = pd.DataFrame(
            {
                "pressure": [0.1, 0.2, 0.3],
                "acc_x": [0.0, 0.1, 0.2],
                "acc_y": [0.3, None, 0.5],
                "acc_z": [0.6, 0.7, 0.8],
                "phi": [0.9, 1.0, 1.1],
                "Time": [0.0, 0.1, 0.2],
            }
        )
        store.put("/runs/run_train_1", frame, format="fixed")

    config = load_data_config(synthetic_loader_case["config_path"])
    with pytest.raises(ValueError, match="contains missing values"):
        load_runs(config)


def test_load_runs_fails_when_configured_run_is_missing_from_metadata(
    synthetic_loader_case: dict[str, Path],
) -> None:
    config_path = synthetic_loader_case["config_path"]
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["splits"]["held_out"] = ["run_missing_1"]
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    config = load_data_config(config_path)
    with pytest.raises(FileNotFoundError, match="Configured split run IDs were not found"):
        load_runs(config)
