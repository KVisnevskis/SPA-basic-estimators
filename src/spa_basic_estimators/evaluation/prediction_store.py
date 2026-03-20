from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_PREDICTION_STORE_NAME = "all_dataset_predictions.h5"
META_RUNS_KEY = "/meta/runs"
DEFAULT_TRUTH_COLUMN = "phi_true"
DEFAULT_PREDICTION_COLUMN = "phi_prediction"


@dataclass(frozen=True)
class PredictionStoreInfo:
    model_name: str
    artifact_dir: Path
    store_path: Path


def discover_prediction_stores(outputs_dir: str | Path = "outputs") -> list[PredictionStoreInfo]:
    root = Path(outputs_dir)
    if not root.exists():
        return []

    stores: list[PredictionStoreInfo] = []
    for artifact_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        store_path = artifact_dir / DEFAULT_PREDICTION_STORE_NAME
        if not store_path.exists():
            continue

        stores.append(
            PredictionStoreInfo(
                model_name=_load_model_name(artifact_dir),
                artifact_dir=artifact_dir,
                store_path=store_path,
            )
        )

    return stores


def load_run_catalog(store_path: str | Path) -> pd.DataFrame:
    path = Path(store_path)
    with pd.HDFStore(path, mode="r") as store:
        if META_RUNS_KEY not in store:
            raise KeyError(f"Prediction store '{path}' is missing '{META_RUNS_KEY}'")
        catalog = store[META_RUNS_KEY].copy()

    if "run_id" not in catalog.columns:
        raise KeyError(f"Prediction store '{path}' metadata is missing the 'run_id' column")

    catalog["run_id"] = catalog["run_id"].astype(str)

    sort_columns = [column for column in ["split", "run_id"] if column in catalog.columns]
    if sort_columns:
        catalog = catalog.sort_values(sort_columns, kind="stable").reset_index(drop=True)

    return catalog


def load_run_frame(store_path: str | Path, run_id: str) -> pd.DataFrame:
    path = Path(store_path)
    catalog = load_run_catalog(path)
    matches = catalog.loc[catalog["run_id"] == str(run_id)]
    if matches.empty:
        raise KeyError(f"Run '{run_id}' was not found in prediction store '{path}'")
    if len(matches) > 1:
        raise ValueError(f"Run '{run_id}' appears multiple times in prediction store '{path}'")

    prediction_key = str(matches.iloc[0].get("prediction_hdf5_key", f"/predictions/{run_id}"))
    with pd.HDFStore(path, mode="r") as store:
        if prediction_key not in store:
            raise KeyError(f"Prediction store '{path}' is missing '{prediction_key}'")
        return store[prediction_key].copy()


def list_plottable_columns(run_frame: pd.DataFrame) -> list[str]:
    excluded_columns = {"run_id", "split"}
    columns: list[str] = []
    for column in run_frame.columns:
        if column in excluded_columns:
            continue
        if pd.api.types.is_numeric_dtype(run_frame[column]):
            columns.append(column)
    return columns


def default_selected_columns(run_frame: pd.DataFrame) -> list[str]:
    plottable = list_plottable_columns(run_frame)
    preferred = [
        column
        for column in [DEFAULT_TRUTH_COLUMN, DEFAULT_PREDICTION_COLUMN]
        if column in plottable
    ]
    if preferred:
        return preferred

    x_axis = choose_x_axis_column(run_frame)
    fallback = [column for column in plottable if column != x_axis]
    return fallback[: min(2, len(fallback))]


def choose_x_axis_column(run_frame: pd.DataFrame) -> str:
    for column in ["Time", "time", "sample_index"]:
        if column in run_frame.columns and pd.api.types.is_numeric_dtype(run_frame[column]):
            return column

    plottable = list_plottable_columns(run_frame)
    if not plottable:
        raise ValueError("Run frame does not contain any numeric columns to use as an x-axis")
    return plottable[0]


def compute_run_rmse(
    run_frame: pd.DataFrame,
    *,
    truth_column: str = DEFAULT_TRUTH_COLUMN,
    prediction_column: str = DEFAULT_PREDICTION_COLUMN,
) -> float:
    if truth_column not in run_frame.columns:
        raise KeyError(f"Run frame is missing truth column '{truth_column}'")
    if prediction_column not in run_frame.columns:
        raise KeyError(f"Run frame is missing prediction column '{prediction_column}'")

    y_true = run_frame[truth_column].to_numpy(dtype=float)
    y_pred = run_frame[prediction_column].to_numpy(dtype=float)
    return float(np.sqrt(np.mean(np.square(y_pred - y_true))))


def _load_model_name(artifact_dir: Path) -> str:
    summary_path = artifact_dir / "run_summary.json"
    if not summary_path.exists():
        return artifact_dir.name

    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return artifact_dir.name

    model_name = summary.get("estimator_name")
    return str(model_name) if model_name else artifact_dir.name
