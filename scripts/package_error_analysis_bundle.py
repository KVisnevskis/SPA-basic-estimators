#!/usr/bin/env python3
"""Package basic-estimator predictions and metadata for external error analysis.

The bundle is designed for a dedicated downstream repository that wants:

- full samplewise predictions per model and per run
- bending-angle columns in degrees
- traceable links back to original model artifact folders
- model coefficients and selected hyperparameters alongside predictions
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import shutil
from typing import Any

import pandas as pd


RAD_TO_DEG = 180.0 / math.pi
PREDICTION_STORE_NAME = "basic_estimators_error_analysis.h5"
EXCLUDED_OUTPUT_DIRS = {"mlp_external"}


@dataclass(frozen=True)
class ModelArtifact:
    model_name: str
    artifact_dir: Path
    prediction_store_path: Path
    run_summary_path: Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _relative_to_repo(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return loaded


def _load_model_name(run_summary_path: Path, fallback_name: str) -> str:
    summary = _load_json(run_summary_path)
    value = summary.get("estimator_name")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return fallback_name


def _discover_model_artifacts(outputs_root: Path) -> list[ModelArtifact]:
    discovered: list[ModelArtifact] = []
    for artifact_dir in sorted(path for path in outputs_root.iterdir() if path.is_dir()):
        if artifact_dir.name in EXCLUDED_OUTPUT_DIRS:
            continue

        prediction_store_path = artifact_dir / "all_dataset_predictions.h5"
        run_summary_path = artifact_dir / "run_summary.json"
        if not prediction_store_path.exists() or not run_summary_path.exists():
            continue

        model_name = _load_model_name(run_summary_path, artifact_dir.name)
        discovered.append(
            ModelArtifact(
                model_name=model_name,
                artifact_dir=artifact_dir,
                prediction_store_path=prediction_store_path,
                run_summary_path=run_summary_path,
            )
        )

    if not discovered:
        raise ValueError(f"No packagable model artifacts found in {outputs_root}")
    return discovered


def _classify_model(model_name: str) -> dict[str, str]:
    tokens = model_name.split("_")
    temporal_mode = "lagged" if tokens[0] == "lagged" else "static"
    base_name = model_name[len("lagged_") :] if temporal_mode == "lagged" else model_name
    fixed_degree_match = re.fullmatch(r"(.+)_ridge_polynomial_degree_(\d+)", base_name)

    if base_name.endswith("_linear_least_squares"):
        estimator_family = "linear_least_squares"
        feature_expansion = "linear"
        input_group = base_name[: -len("_linear_least_squares")]
    elif fixed_degree_match:
        estimator_family = "ridge"
        feature_expansion = "polynomial_fixed_degree"
        input_group = fixed_degree_match.group(1)
    elif base_name.endswith("_ridge_polynomial_search"):
        estimator_family = "ridge"
        feature_expansion = "polynomial_search"
        input_group = base_name[: -len("_ridge_polynomial_search")]
    elif base_name.endswith("_ridge_linear"):
        estimator_family = "ridge"
        feature_expansion = "linear"
        input_group = base_name[: -len("_ridge_linear")]
    elif base_name.endswith("_ridge_quadratic"):
        estimator_family = "ridge"
        feature_expansion = "quadratic"
        input_group = base_name[: -len("_ridge_quadratic")]
    elif base_name.endswith("_ridge"):
        estimator_family = "ridge"
        feature_expansion = "linear"
        input_group = base_name[: -len("_ridge")]
    else:
        estimator_family = "unknown"
        feature_expansion = "unknown"
        input_group = base_name

    return {
        "estimator_family": estimator_family,
        "feature_expansion": feature_expansion,
        "temporal_mode": temporal_mode,
        "input_group": input_group,
    }


def _metrics_or_blank(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return _load_json(path)


def _model_catalog_row(
    artifact: ModelArtifact,
    repo_root: Path,
    run_summary: dict[str, Any],
    validation_metrics: dict[str, Any],
    held_out_metrics: dict[str, Any],
    coefficient_count: int,
    run_count: int,
    prediction_row_count: int,
) -> dict[str, Any]:
    classification = _classify_model(artifact.model_name)
    row: dict[str, Any] = {
        "model_name": artifact.model_name,
        "artifact_dir": _relative_to_repo(artifact.artifact_dir, repo_root),
        "prediction_store_path": _relative_to_repo(artifact.prediction_store_path, repo_root),
        "run_summary_path": _relative_to_repo(artifact.run_summary_path, repo_root),
        "data_config_path": str(run_summary.get("data_config_path", "")),
        "model_config_path": str(run_summary.get("model_config_path", "")),
        "fit_intercept": run_summary.get("fit_intercept"),
        "selected_alpha": run_summary.get("selected_alpha"),
        "selected_lag": run_summary.get("selected_lag"),
        "selected_degree": run_summary.get("selected_degree"),
        "degree": run_summary.get("degree"),
        "searched_degree_grid": json.dumps(run_summary.get("degree_grid"))
        if run_summary.get("degree_grid") is not None
        else None,
        "feature_count": len(run_summary.get("feature_columns", [])),
        "raw_feature_count": len(run_summary.get("raw_feature_columns", [])),
        "coefficient_count": coefficient_count,
        "run_count": run_count,
        "prediction_row_count": prediction_row_count,
        "packaged_prediction_root": f"/predictions/{artifact.model_name}",
        "packaged_coefficient_key": f"/coefficients/{artifact.model_name}",
        "has_validation_search": (artifact.artifact_dir / "validation_search.csv").exists(),
        "validation_rmse_stored_units": validation_metrics.get("rmse"),
        "held_out_rmse_stored_units": held_out_metrics.get("rmse"),
        "validation_mae_stored_units": validation_metrics.get("mae"),
        "held_out_mae_stored_units": held_out_metrics.get("mae"),
        "validation_r2": validation_metrics.get("r2"),
        "held_out_r2": held_out_metrics.get("r2"),
        "validation_pearson_r": validation_metrics.get("pearson_r"),
        "held_out_pearson_r": held_out_metrics.get("pearson_r"),
    }
    row.update(classification)
    return row


def _prediction_export_frame(frame: pd.DataFrame, model_name: str, run_id: str) -> pd.DataFrame:
    required_columns = {"Time", "split", "sample_index", "phi_true", "phi_prediction", "phi_error"}
    missing = required_columns.difference(frame.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Prediction frame for {model_name}/{run_id} is missing columns: {missing_text}")

    export_frame = frame.loc[:, ["Time", "split", "sample_index", "phi_true", "phi_prediction", "phi_error"]].copy()
    export_frame.insert(0, "run_id", run_id)
    export_frame.insert(0, "model_name", model_name)
    export_frame["phi_true_deg"] = export_frame["phi_true"] * RAD_TO_DEG
    export_frame["phi_prediction_deg"] = export_frame["phi_prediction"] * RAD_TO_DEG
    export_frame["phi_error_deg"] = export_frame["phi_error"] * RAD_TO_DEG
    return export_frame


def _copy_snapshot_files(artifact: ModelArtifact, snapshot_dir: Path, run_summary: dict[str, Any], repo_root: Path) -> list[str]:
    copied: list[str] = []
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    candidate_files = [
        artifact.run_summary_path,
        artifact.artifact_dir / "coefficient_table.csv",
        artifact.artifact_dir / "held_out_metrics.json",
        artifact.artifact_dir / "validation_metrics.json",
        artifact.artifact_dir / "validation_search.csv",
    ]

    data_config_path = run_summary.get("data_config_path")
    if isinstance(data_config_path, str) and data_config_path.strip():
        candidate_files.append(Path(data_config_path))

    model_config_path = run_summary.get("model_config_path")
    if isinstance(model_config_path, str) and model_config_path.strip():
        candidate_files.append(Path(model_config_path))

    seen_sources: set[Path] = set()
    for source_path in candidate_files:
        source_path = source_path.resolve()
        if source_path in seen_sources or not source_path.exists():
            continue
        seen_sources.add(source_path)
        destination_path = snapshot_dir / source_path.name
        shutil.copy2(source_path, destination_path)
        copied.append(_relative_to_repo(destination_path, repo_root))

    return copied


def _write_readme(
    readme_path: Path,
    bundle_store_path: Path,
    included_model_names: list[str],
) -> None:
    pressure_accel_degree_sweep_models = [
        name
        for name in included_model_names
        if name.startswith("pressure_accel_ridge_polynomial_degree_")
    ]
    readme = f"""# Basic Estimator Error Analysis Bundle

This bundle packages the samplewise prediction outputs for the current basic estimator models in a form intended for a dedicated downstream error-analysis repository.

## Bundle Contents

- `{bundle_store_path.name}`: HDF5 store containing:
  - `/meta/models`: one row per packaged model
  - `/meta/runs`: one row per `(model, run)` pair
  - `/coefficients/<model_name>`: coefficient tables copied into the bundle store
  - `/validation_search/<model_name>`: validation search tables when present
  - `/predictions/<model_name>/<run_id>`: samplewise prediction tables
- `models.csv`: flat model catalog for quick filtering and joining
- `runs.csv`: flat `(model, run)` catalog with split labels and sample counts
- `coefficients_all_models.csv`: one consolidated coefficient table across all packaged models
- `source_snapshots/`: copied config and summary files from the original artifact directories
- `manifest.json`: generation metadata and traceability information

## Prediction Table Schema

Each table under `/predictions/<model_name>/<run_id>` contains:

- `model_name`
- `run_id`
- `Time`
- `split`
- `sample_index`
- `phi_true`
- `phi_prediction`
- `phi_error`
- `phi_true_deg`
- `phi_prediction_deg`
- `phi_error_deg`

The `phi_*` columns are the original unscaled physical-angle values from the source prediction stores, in radians.
The `phi_*_deg` columns are the same quantities converted to degrees for downstream plotting and error analysis.

## Traceability

Traceability is preserved through three mechanisms:

1. `models.csv` and `/meta/models` record the original artifact directory, source config paths, and selected hyperparameters.
2. `runs.csv` and `/meta/runs` record the source prediction-table key for every `(model, run)` pair.
3. `source_snapshots/` contains copies of the run summary, model config, data config, coefficient table, and metrics files used to produce each exported model entry.

## Included Models

{chr(10).join(f"- `{name}`" for name in included_model_names)}

## Highlighted Degree Sweep

The bundle includes a fixed-degree pressure-plus-accelerometer polynomial sweep to support explicit analysis of how increasing non-linearity affects prediction quality:

{chr(10).join(f"- `{name}`" for name in pressure_accel_degree_sweep_models) if pressure_accel_degree_sweep_models else "- none"}

## Exclusions

- `mlp_external` is not included because the current repository only contains grouped per-run RMSE for that model, not the full samplewise prediction traces required for this bundle.
"""
    readme_path.write_text(readme, encoding="utf-8")


def main() -> None:
    repo_root = _repo_root()
    outputs_root = repo_root / "outputs"
    bundle_root = outputs_root / "error_analysis_bundle"
    snapshot_root = bundle_root / "source_snapshots"
    bundle_store_path = bundle_root / PREDICTION_STORE_NAME
    manifest_path = bundle_root / "manifest.json"
    models_csv_path = bundle_root / "models.csv"
    runs_csv_path = bundle_root / "runs.csv"
    coefficients_csv_path = bundle_root / "coefficients_all_models.csv"
    readme_path = bundle_root / "README.md"

    if bundle_root.exists():
        shutil.rmtree(bundle_root)
    bundle_root.mkdir(parents=True, exist_ok=True)
    snapshot_root.mkdir(parents=True, exist_ok=True)

    artifacts = _discover_model_artifacts(outputs_root)

    model_catalog_rows: list[dict[str, Any]] = []
    run_catalog_rows: list[dict[str, Any]] = []
    coefficient_frames: list[pd.DataFrame] = []
    manifest_models: list[dict[str, Any]] = []

    total_prediction_tables = 0
    total_prediction_rows = 0

    with pd.HDFStore(bundle_store_path, mode="w", complevel=9, complib="zlib") as bundle_store:
        for artifact in artifacts:
            run_summary = _load_json(artifact.run_summary_path)
            validation_metrics = _metrics_or_blank(artifact.artifact_dir / "validation_metrics.json")
            held_out_metrics = _metrics_or_blank(artifact.artifact_dir / "held_out_metrics.json")

            coefficient_path = artifact.artifact_dir / "coefficient_table.csv"
            coefficient_frame = pd.read_csv(coefficient_path) if coefficient_path.exists() else pd.DataFrame()
            if not coefficient_frame.empty:
                coefficient_frame.insert(0, "model_name", artifact.model_name)
                coefficient_frames.append(coefficient_frame)
                bundle_store.put(f"/coefficients/{artifact.model_name}", coefficient_frame, format="table")

            validation_search_path = artifact.artifact_dir / "validation_search.csv"
            if validation_search_path.exists():
                validation_search_frame = pd.read_csv(validation_search_path)
                validation_search_frame.insert(0, "model_name", artifact.model_name)
                bundle_store.put(
                    f"/validation_search/{artifact.model_name}",
                    validation_search_frame,
                    format="table",
                )

            snapshot_dir = snapshot_root / artifact.model_name
            snapshot_files = _copy_snapshot_files(artifact, snapshot_dir, run_summary, repo_root)

            with pd.HDFStore(artifact.prediction_store_path, mode="r") as source_store:
                meta_runs = source_store["/meta/runs"].copy()
                prediction_rows_for_model = 0

                for _, meta_row in meta_runs.iterrows():
                    run_id = str(meta_row["run_id"])
                    prediction_key = str(meta_row["prediction_hdf5_key"])
                    source_frame = source_store[prediction_key]
                    export_frame = _prediction_export_frame(source_frame, artifact.model_name, run_id)

                    bundle_prediction_key = f"/predictions/{artifact.model_name}/{run_id}"
                    bundle_store.put(bundle_prediction_key, export_frame, format="table")

                    row_count = int(len(export_frame))
                    prediction_rows_for_model += row_count
                    total_prediction_rows += row_count
                    total_prediction_tables += 1

                    run_catalog_row = {
                        "model_name": artifact.model_name,
                        "run_id": run_id,
                        "split": str(meta_row["split"]),
                        "rows_saved": int(meta_row["rows_saved"]),
                        "source_hdf5_key": str(meta_row["source_hdf5_key"]),
                        "source_prediction_hdf5_key": prediction_key,
                        "packaged_prediction_hdf5_key": bundle_prediction_key,
                        "artifact_dir": _relative_to_repo(artifact.artifact_dir, repo_root),
                        "prediction_store_path": _relative_to_repo(artifact.prediction_store_path, repo_root),
                    }
                    for optional_column in ("lag_length", "trimmed_initial_rows"):
                        if optional_column in meta_runs.columns:
                            run_catalog_row[optional_column] = meta_row.get(optional_column)
                    run_catalog_rows.append(run_catalog_row)

                model_catalog_rows.append(
                    _model_catalog_row(
                        artifact=artifact,
                        repo_root=repo_root,
                        run_summary=run_summary,
                        validation_metrics=validation_metrics,
                        held_out_metrics=held_out_metrics,
                        coefficient_count=int(len(coefficient_frame)),
                        run_count=int(len(meta_runs)),
                        prediction_row_count=prediction_rows_for_model,
                    )
                )

                manifest_models.append(
                    {
                        "model_name": artifact.model_name,
                        "artifact_dir": _relative_to_repo(artifact.artifact_dir, repo_root),
                        "prediction_store_path": _relative_to_repo(artifact.prediction_store_path, repo_root),
                        "snapshot_files": snapshot_files,
                    }
                )

        models_frame = pd.DataFrame(model_catalog_rows).sort_values("model_name").reset_index(drop=True)
        runs_frame = pd.DataFrame(run_catalog_rows).sort_values(["model_name", "run_id"]).reset_index(drop=True)
        coefficients_all_frame = (
            pd.concat(coefficient_frames, ignore_index=True)
            if coefficient_frames
            else pd.DataFrame()
        )

        bundle_store.put("/meta/models", models_frame, format="table")
        bundle_store.put("/meta/runs", runs_frame, format="table")

    models_frame.to_csv(models_csv_path, index=False)
    runs_frame.to_csv(runs_csv_path, index=False)
    coefficients_all_frame.to_csv(coefficients_csv_path, index=False)

    _write_readme(
        readme_path=readme_path,
        bundle_store_path=bundle_store_path,
        included_model_names=[artifact.model_name for artifact in artifacts],
    )

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root.resolve()),
        "bundle_root": _relative_to_repo(bundle_root, repo_root),
        "bundle_store_path": _relative_to_repo(bundle_store_path, repo_root),
        "included_model_count": len(artifacts),
        "included_run_table_count": total_prediction_tables,
        "included_prediction_row_count": total_prediction_rows,
        "prediction_angle_units": {
            "phi_true": "radians",
            "phi_prediction": "radians",
            "phi_error": "radians",
            "phi_true_deg": "degrees",
            "phi_prediction_deg": "degrees",
            "phi_error_deg": "degrees",
        },
        "excluded_outputs": [
            {
                "name": "mlp_external",
                "reason": "Only grouped per-run RMSE is available in this repository; samplewise prediction traces are not present.",
            }
        ],
        "highlighted_model_groups": {
            "pressure_accel_polynomial_degree_sweep": [
                artifact.model_name
                for artifact in artifacts
                if artifact.model_name.startswith("pressure_accel_ridge_polynomial_degree_")
            ]
        },
        "models": manifest_models,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Packaged models: {len(artifacts)}")
    print(f"Prediction tables written: {total_prediction_tables}")
    print(f"Prediction rows written: {total_prediction_rows}")
    print(f"Bundle directory: {bundle_root}")


if __name__ == "__main__":
    main()
