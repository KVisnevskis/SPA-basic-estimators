from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from spa_basic_estimators.utils.config import load_yaml, resolve_path
from spa_basic_estimators.utils.splits import (
    HDF5_KEY_COLUMN,
    SPLIT_COLUMN,
    build_run_to_split_map,
    validate_splits,
)


@dataclass(frozen=True)
class StorageConfig:
    format: str
    path: Path
    source_experiment_config: Path | None
    runs_key_pattern: str
    meta_runs_key: str
    meta_keys: dict[str, str]


@dataclass(frozen=True)
class SchemaConfig:
    schema_doc: Path | None
    split_id_convention: str
    split_id_note: str | None
    run_id_column: str
    run_key_column: str
    time_column: str
    target_column: str
    pressure_columns: list[str]
    accel_columns: list[str]
    required_run_columns: list[str]


@dataclass(frozen=True)
class ValidationConfig:
    fail_on_missing_columns: bool
    fail_on_missing_values: bool
    require_unique_run_ids: bool


@dataclass(frozen=True)
class ScalerBounds:
    column: str
    min_value: float
    max_value: float
    range_value: float
    is_constant: bool


@dataclass(frozen=True)
class DataConfig:
    config_path: Path
    storage: StorageConfig
    schema: SchemaConfig
    conventions: dict[str, Any]
    splits: dict[str, list[str]]
    validation: ValidationConfig

    @property
    def expected_run_ids(self) -> list[str]:
        return [run_id for split_ids in self.splits.values() for run_id in split_ids]


def load_data_config(path: str | Path) -> DataConfig:
    config_path = Path(path).resolve()
    raw = load_yaml(config_path)
    config_dir = config_path.parent
    project_root = config_dir.parent

    storage_raw = raw["storage"]
    schema_raw = raw["schema"]
    validation_raw = raw.get("validation", {})

    storage = StorageConfig(
        format=str(storage_raw["format"]).lower(),
        path=_resolve_config_reference(config_dir, project_root, storage_raw["path"]),
        source_experiment_config=(
            _resolve_config_reference(
                config_dir,
                project_root,
                storage_raw["source_experiment_config"],
            )
            if storage_raw.get("source_experiment_config")
            else None
        ),
        runs_key_pattern=str(storage_raw["runs"]["key_pattern"]),
        meta_runs_key=str(storage_raw["runs"]["meta_runs_key"]),
        meta_keys={str(name): str(value) for name, value in storage_raw.get("meta", {}).items()},
    )

    schema = SchemaConfig(
        schema_doc=(
            _resolve_config_reference(
                config_dir,
                project_root,
                schema_raw["schema_doc"],
            )
            if schema_raw.get("schema_doc")
            else None
        ),
        split_id_convention=str(schema_raw.get("split_id_convention", "hdf5_key_suffix")),
        split_id_note=(
            str(schema_raw["split_id_note"])
            if schema_raw.get("split_id_note") is not None
            else None
        ),
        run_id_column=str(schema_raw["run_id_column"]),
        run_key_column=str(schema_raw["run_key_column"]),
        time_column=str(schema_raw["time_column"]),
        target_column=str(schema_raw["target_column"]),
        pressure_columns=[str(item) for item in schema_raw.get("pressure_columns", [])],
        accel_columns=[str(item) for item in schema_raw.get("accel_columns", [])],
        required_run_columns=[str(item) for item in schema_raw["required_run_columns"]],
    )

    validation = ValidationConfig(
        fail_on_missing_columns=bool(validation_raw.get("fail_on_missing_columns", True)),
        fail_on_missing_values=bool(validation_raw.get("fail_on_missing_values", True)),
        require_unique_run_ids=bool(validation_raw.get("require_unique_run_ids", True)),
    )

    splits = validate_splits(
        raw["splits"],
        require_unique_run_ids=validation.require_unique_run_ids,
    )

    return DataConfig(
        config_path=config_path,
        storage=storage,
        schema=schema,
        conventions=dict(raw.get("conventions", {})),
        splits=splits,
        validation=validation,
    )


def load_runs(config: DataConfig) -> dict[str, pd.DataFrame]:
    if config.storage.format not in {"hdf", "hdf5"}:
        raise NotImplementedError(
            f"Phase 1C currently supports HDF5 only, not {config.storage.format!r}"
        )

    if not config.storage.path.exists():
        raise FileNotFoundError(f"HDF5 dataset not found: {config.storage.path}")

    run_to_split = build_run_to_split_map(
        config.splits,
        require_unique_run_ids=config.validation.require_unique_run_ids,
    )
    runs_metadata = _load_runs_metadata(config)

    missing_configured_runs = sorted(set(run_to_split) - set(runs_metadata))
    if missing_configured_runs:
        raise FileNotFoundError(
            "Configured split run IDs were not found in HDF5 metadata: "
            + ", ".join(missing_configured_runs)
        )

    loaded_runs: dict[str, pd.DataFrame] = {}
    with pd.HDFStore(config.storage.path, mode="r") as store:
        for split_name in ("train", "val", "held_out"):
            for run_id in config.splits[split_name]:
                metadata = runs_metadata[run_id]
                frame = store[metadata["hdf5_key"]].copy()
                prepared = _prepare_run_frame(
                    frame=frame,
                    config=config,
                    run_id=run_id,
                    split_name=split_name,
                    hdf5_key=metadata["hdf5_key"],
                )

                if run_id in loaded_runs:
                    raise ValueError(f"Duplicate run ID loaded more than once: {run_id}")
                loaded_runs[run_id] = prepared

    return loaded_runs


def load_scaler_bounds(config: DataConfig) -> dict[str, ScalerBounds]:
    scaler_key = config.storage.meta_keys.get("scaler_parameters", "/meta/scaler_parameters")
    normalised_key = _normalise_hdf_key(scaler_key)

    with pd.HDFStore(config.storage.path, mode="r") as store:
        available_keys = set(store.keys())
        if normalised_key not in available_keys:
            raise KeyError(f"Scaler parameter table not found in HDF5 store: {normalised_key}")

        scaler_parameters = store[normalised_key]

    required_columns = ["column", "min", "max", "range", "is_constant"]
    missing_columns = [
        column for column in required_columns if column not in scaler_parameters.columns
    ]
    if missing_columns:
        raise ValueError(
            "Missing required columns in scaler parameter table: "
            + ", ".join(missing_columns)
        )

    bounds_by_column: dict[str, ScalerBounds] = {}
    for _, row in scaler_parameters.iterrows():
        column = str(row["column"])
        bounds_by_column[column] = ScalerBounds(
            column=column,
            min_value=float(row["min"]),
            max_value=float(row["max"]),
            range_value=float(row["range"]),
            is_constant=bool(row["is_constant"]),
        )

    return bounds_by_column


def _load_runs_metadata(config: DataConfig) -> dict[str, dict[str, str]]:
    meta_runs_key = _normalise_hdf_key(config.storage.meta_runs_key)
    with pd.HDFStore(config.storage.path, mode="r") as store:
        available_keys = set(store.keys())
        if meta_runs_key not in available_keys:
            raise KeyError(f"Metadata table not found in HDF5 store: {meta_runs_key}")

        meta_runs = store[meta_runs_key]

    required_meta_columns = [config.schema.run_id_column, config.schema.run_key_column]
    missing_meta_columns = [
        column for column in required_meta_columns if column not in meta_runs.columns
    ]
    if missing_meta_columns:
        raise ValueError(
            "Missing required columns in meta runs table: "
            + ", ".join(missing_meta_columns)
        )

    key_prefix = _run_key_prefix(config.storage.runs_key_pattern)
    runs_metadata: dict[str, dict[str, str]] = {}
    for _, row in meta_runs.iterrows():
        hdf5_key = _normalise_hdf_key(str(row[config.schema.run_key_column]))
        canonical_run_id = _canonical_run_id_from_key(hdf5_key, key_prefix)
        source_run_id = str(row[config.schema.run_id_column])

        if canonical_run_id in runs_metadata:
            raise ValueError(
                f"Duplicate canonical run ID found in metadata: {canonical_run_id}"
            )

        runs_metadata[canonical_run_id] = {
            "hdf5_key": hdf5_key,
            "source_run_id": source_run_id,
        }

    return runs_metadata


def _prepare_run_frame(
    frame: pd.DataFrame,
    config: DataConfig,
    run_id: str,
    split_name: str,
    hdf5_key: str,
) -> pd.DataFrame:
    _validate_required_columns(frame, config, run_id, hdf5_key)
    _validate_missing_values(frame, config, run_id, hdf5_key)

    prepared = frame.copy()
    prepared[config.schema.run_id_column] = run_id
    prepared[SPLIT_COLUMN] = split_name
    prepared[HDF5_KEY_COLUMN] = hdf5_key
    return prepared


def _validate_required_columns(
    frame: pd.DataFrame,
    config: DataConfig,
    run_id: str,
    hdf5_key: str,
) -> None:
    if not config.validation.fail_on_missing_columns:
        return

    missing_columns = [
        column
        for column in config.schema.required_run_columns
        if column not in frame.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Run '{run_id}' at '{hdf5_key}' is missing required columns: {missing_columns}"
        )


def _validate_missing_values(
    frame: pd.DataFrame,
    config: DataConfig,
    run_id: str,
    hdf5_key: str,
) -> None:
    if not config.validation.fail_on_missing_values:
        return

    relevant_columns = [
        column
        for column in config.schema.required_run_columns
        if column in frame.columns
    ]
    missing_value_columns = [
        column
        for column, count in frame[relevant_columns].isna().sum().items()
        if int(count) > 0
    ]
    if missing_value_columns:
        raise ValueError(
            f"Run '{run_id}' at '{hdf5_key}' contains missing values in columns: "
            f"{missing_value_columns}"
        )


def _normalise_hdf_key(key: str) -> str:
    return key if key.startswith("/") else f"/{key}"


def _run_key_prefix(key_pattern: str) -> str:
    if "<run_id>" not in key_pattern:
        raise ValueError(
            "storage.runs.key_pattern must include the '<run_id>' placeholder"
        )
    return _normalise_hdf_key(key_pattern.split("<run_id>", 1)[0])


def _canonical_run_id_from_key(hdf5_key: str, key_prefix: str) -> str:
    if not hdf5_key.startswith(key_prefix):
        raise ValueError(
            f"HDF5 key '{hdf5_key}' does not match expected run prefix '{key_prefix}'"
        )

    canonical_run_id = hdf5_key[len(key_prefix) :]
    if not canonical_run_id:
        raise ValueError(f"Could not derive run ID from HDF5 key: {hdf5_key}")
    return canonical_run_id


def _resolve_config_reference(
    config_dir: Path,
    project_root: Path,
    raw_path: str | Path,
) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path

    config_relative = resolve_path(config_dir, path)
    project_relative = resolve_path(project_root, path)

    if config_relative.exists():
        return config_relative
    if project_relative.exists():
        return project_relative

    # Prefer repo-root-relative paths for project data assets when neither exists yet.
    return project_relative
