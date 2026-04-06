#!/usr/bin/env python3
"""Generate a grouped LaTeX table with per-run RMSE for the basic estimator models.

This script reads the current basic-model prediction artifacts written under
`outputs/<model_name>/all_dataset_predictions.h5`, computes per-run RMSE from
the unscaled `phi_error` column, converts the values to degrees, and writes a
LaTeX longtable with one column per model.

Rows are grouped by split role:

train -> val -> eval -> unseen

The existing color rule is preserved:
- values above 100 deg are highlighted in yellow as anomalies
- non-outlier cells use the softer blue->orange gradient
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


SPLIT_ROLE_ORDER: dict[str, int] = {
    "train": 0,
    "val": 1,
    "eval": 2,
    "unseen": 3,
}

SECTION_ROW_LABEL: dict[str, str] = {
    "train": "training datasets",
    "val": "validation datasets",
    "eval": "evaluation datasets",
    "unseen": "unseen datasets",
}

EXCLUDED_MODEL_NAMES = {
    "accel_ridge_quadratic",
    "pressure_ridge_quadratic",
}

PREFERRED_MODEL_ORDER = [
    "accel_ridge_linear",
    "accel_linear_least_squares",
    "pressure_ridge_linear",
    "pressure_linear_least_squares",
    "pressure_accel_ridge_linear",
    "pressure_accel_linear_least_squares",
    "pressure_accel_ridge_quadratic",
    "lagged_pressure_ridge",
    "lagged_accel_ridge",
    "lagged_pressure_accel_ridge",
    "lagged_pressure_ridge_quadratic",
    "lagged_accel_ridge_quadratic",
    "lagged_pressure_accel_ridge_quadratic",
    "mlp_external_best",
]

MODEL_LABELS: dict[str, str] = {
    "accel_ridge_linear": "A-LR",
    "accel_linear_least_squares": "A-LS",
    "pressure_ridge_linear": "P-LR",
    "pressure_linear_least_squares": "P-LS",
    "pressure_accel_ridge_linear": "PA-LR",
    "pressure_accel_linear_least_squares": "PA-LS",
    "pressure_accel_ridge_quadratic": "PA-QR",
    "lagged_pressure_ridge": "LP-LR",
    "lagged_accel_ridge": "LA-LR",
    "lagged_pressure_accel_ridge": "LPA-LR",
    "lagged_pressure_ridge_quadratic": "LP-QR",
    "lagged_accel_ridge_quadratic": "LA-QR",
    "lagged_pressure_accel_ridge_quadratic": "LPA-QR",
    "mlp_external_best": "MLP",
}

RAD_TO_DEG = 180.0 / math.pi
DEFAULT_EXTERNAL_MLP_CSV = Path("outputs/mlp_external/mlp_hpo_rmse_per_run_grouped.csv")


@dataclass(frozen=True)
class ModelSource:
    model_name: str
    display_label: str
    source_kind: str
    source_path: Path
    selected_variant: str | None = None


def _latex_escape(text: str) -> str:
    escaped = text
    replacements = (
        ("\\", "\\textbackslash{}"),
        ("&", "\\&"),
        ("%", "\\%"),
        ("$", "\\$"),
        ("#", "\\#"),
        ("_", "\\_"),
        ("{", "\\{"),
        ("}", "\\}"),
        ("~", "\\textasciitilde{}"),
        ("^", "\\textasciicircum{}"),
    )
    for old, new in replacements:
        escaped = escaped.replace(old, new)
    return escaped


def _load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _discover_model_sources(repo_root: Path, outputs_root: Path) -> list[ModelSource]:
    discovered: list[ModelSource] = []
    for artifact_dir in sorted(path for path in outputs_root.iterdir() if path.is_dir()):
        store_path = artifact_dir / "all_dataset_predictions.h5"
        if not store_path.exists():
            continue

        model_name = _load_model_name(artifact_dir)
        if model_name in EXCLUDED_MODEL_NAMES:
            continue

        discovered.append(
            ModelSource(
                model_name=model_name,
                display_label=_model_display_label(model_name),
                source_kind="prediction_store",
                source_path=store_path,
            )
        )

    external_mlp_csv = repo_root / DEFAULT_EXTERNAL_MLP_CSV
    if external_mlp_csv.exists():
        selected_variant = _select_best_external_variant(external_mlp_csv)
        discovered.append(
            ModelSource(
                model_name="mlp_external_best",
                display_label=_model_display_label("mlp_external_best"),
                source_kind="external_grouped_csv",
                source_path=external_mlp_csv,
                selected_variant=selected_variant,
            )
        )

    if not discovered:
        raise ValueError(
            f"No model outputs found in {outputs_root}"
        )

    order_index = {model_name: index for index, model_name in enumerate(PREFERRED_MODEL_ORDER)}
    discovered.sort(
        key=lambda item: (
            order_index.get(item.model_name, len(order_index)),
            item.display_label.lower(),
        )
    )
    return discovered


def _load_model_name(artifact_dir: Path) -> str:
    summary_path = artifact_dir / "run_summary.json"
    if not summary_path.exists():
        return artifact_dir.name

    try:
        summary = _load_json(summary_path)
    except (OSError, json.JSONDecodeError):
        return artifact_dir.name

    if isinstance(summary, dict):
        raw_name = summary.get("estimator_name")
        if isinstance(raw_name, str) and raw_name.strip():
            return raw_name.strip()
    return artifact_dir.name


def _model_display_label(model_name: str) -> str:
    if model_name in MODEL_LABELS:
        return MODEL_LABELS[model_name]
    return model_name.replace("_", " ")


def _split_to_role(split_name: str) -> str:
    normalised = split_name.strip().lower()
    if normalised in {"train", "val"}:
        return normalised
    if normalised == "held_out":
        return "eval"
    if normalised == "not_in_split":
        return "unseen"
    return normalised


def _extract_per_run(store_path: Path) -> dict[str, tuple[str, float]]:
    out: dict[str, tuple[str, float]] = {}
    with pd.HDFStore(store_path, mode="r") as store:
        meta = store["/meta/runs"]

        for _, row in meta.iterrows():
            run_name = str(row["run_id"]).strip()
            if not run_name:
                continue

            split_role = _split_to_role(str(row["split"]))
            if split_role not in SPLIT_ROLE_ORDER:
                continue

            prediction_key = str(row["prediction_hdf5_key"])
            frame = store[prediction_key]

            if "phi_error" in frame.columns:
                rmse_rad = float((frame["phi_error"].pow(2).mean()) ** 0.5)
            elif {"phi_true", "phi_prediction"}.issubset(frame.columns):
                diff = frame["phi_prediction"] - frame["phi_true"]
                rmse_rad = float((diff.pow(2).mean()) ** 0.5)
            else:
                raise ValueError(
                    f"Prediction frame '{prediction_key}' in '{store_path}' is missing phi error columns"
                )

            out[run_name] = (split_role, rmse_rad * RAD_TO_DEG)

    return out


def _select_best_external_variant(csv_path: Path) -> str:
    frame = pd.read_csv(csv_path)
    candidate_columns = [
        column
        for column in frame.columns
        if column not in {"run_name", "split_role"}
    ]
    if not candidate_columns:
        raise ValueError(f"No candidate model columns found in {csv_path}")

    val_frame = frame[frame["split_role"].astype(str).str.strip().str.lower() == "val"]
    if val_frame.empty:
        raise ValueError(f"No validation rows found in {csv_path}")

    ranked_candidates: list[tuple[float, str]] = []
    for column in candidate_columns:
        numeric = pd.to_numeric(val_frame[column], errors="coerce")
        mean_rmse = float(numeric.mean())
        if math.isfinite(mean_rmse):
            ranked_candidates.append((mean_rmse, column))

    if not ranked_candidates:
        raise ValueError(f"Could not compute validation means for any MLP candidates in {csv_path}")

    ranked_candidates.sort(key=lambda item: (item[0], item[1]))
    return ranked_candidates[0][1]


def _extract_per_run_from_external_grouped_csv(
    csv_path: Path,
    variant_name: str,
) -> dict[str, tuple[str, float]]:
    frame = pd.read_csv(csv_path)
    required_columns = {"run_name", "split_role", variant_name}
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns in {csv_path}: {missing}")

    out: dict[str, tuple[str, float]] = {}
    for _, row in frame.iterrows():
        run_name = str(row["run_name"]).strip()
        if not run_name:
            continue

        split_role = _split_to_role(str(row["split_role"]))
        if split_role not in SPLIT_ROLE_ORDER:
            continue

        rmse_deg = float(row[variant_name])
        out[run_name] = (split_role, rmse_deg)

    return out


def _build_rows(model_runs: list[ModelSource]) -> tuple[list[str], list[dict[str, Any]]]:
    per_model: dict[str, dict[str, tuple[str, float]]] = {}
    model_labels = [model_run.display_label for model_run in model_runs]

    for model_run in model_runs:
        if model_run.source_kind == "prediction_store":
            per_model[model_run.display_label] = _extract_per_run(model_run.source_path)
            continue

        if model_run.source_kind == "external_grouped_csv":
            if model_run.selected_variant is None:
                raise ValueError(
                    f"External grouped CSV source '{model_run.source_path}' is missing a selected variant"
                )
            per_model[model_run.display_label] = _extract_per_run_from_external_grouped_csv(
                model_run.source_path,
                model_run.selected_variant,
            )
            continue

        raise ValueError(f"Unsupported model source kind: {model_run.source_kind}")

    run_names = sorted({run_name for model_map in per_model.values() for run_name in model_map})
    merged_rows: list[dict[str, Any]] = []

    for run_name in run_names:
        roles = {
            model_map[run_name][0]
            for model_map in per_model.values()
            if run_name in model_map
        }
        if not roles:
            continue
        if len(roles) > 1:
            raise ValueError(f"Inconsistent split role across models for run '{run_name}': {roles}")

        split_role = next(iter(roles))
        row: dict[str, Any] = {
            "run_name": run_name,
            "split_role": split_role,
        }
        for model_label in model_labels:
            row[model_label] = per_model[model_label].get(run_name, (split_role, float("nan")))[1]
        merged_rows.append(row)

    merged_rows.sort(
        key=lambda row: (
            SPLIT_ROLE_ORDER[str(row["split_role"])],
            str(row["run_name"]).lower(),
        )
    )
    return model_labels, merged_rows


def _render_longtable(
    model_labels: list[str],
    rows: list[dict[str, Any]],
    caption: str,
    label: str,
    decimals: int,
    colorize: bool,
) -> str:
    col_spec = "l" + ("r" * len(model_labels))
    top_header = rf"Run name & \multicolumn{{{len(model_labels)}}}{{c}}{{RMSE [deg]}} \\"
    bottom_header = "& " + " & ".join(_latex_escape(model_label) for model_label in model_labels) + r" \\"
    cmidrule = rf"\cmidrule(lr){{2-{len(model_labels) + 1}}}"

    non_outlier_values = [
        float(row[model_label])
        for row in rows
        for model_label in model_labels
        if isinstance(row.get(model_label), (int, float))
        and math.isfinite(float(row[model_label]))
        and float(row[model_label]) <= 100.0
    ]
    if not non_outlier_values:
        raise ValueError("No non-outlier RMSE values available for color scaling.")

    min_non_outlier = min(non_outlier_values)
    max_non_outlier = max(non_outlier_values)

    def _format_cell(value: float) -> str:
        value_text = f"{value:.{decimals}f}"
        if not colorize or not math.isfinite(value):
            return value_text
        if value > 100.0:
            return rf"\cellcolor{{RMSEAnomaly}}{value_text}"
        if max_non_outlier <= min_non_outlier:
            norm = 0.0
        else:
            norm = (value - min_non_outlier) / (max_non_outlier - min_non_outlier)
        norm = max(0.0, min(1.0, norm))
        color_pct = int(round(norm * 100))
        return rf"\cellcolor{{RMSELow!{color_pct}!RMSEHigh}}{value_text}"

    lines: list[str] = []
    if colorize:
        lines.append(r"% Requires: \usepackage[table]{xcolor}")
        lines.append(
            r"% Colorblind-friendly, softer blue->orange scale. "
            r"Outliers (>100 deg) highlighted in contrasting yellow."
        )
        lines.append(r"% Non-outlier gradient bounds are min..max over values <= 100 deg.")
        lines.append(r"\definecolor{RMSELow}{HTML}{E8F1F2}")
        lines.append(r"\definecolor{RMSEHigh}{HTML}{E38B5B}")
        lines.append(r"\definecolor{RMSEAnomaly}{HTML}{FFF176}")

    lines.append(r"\begin{longtable}{" + col_spec + "}")
    lines.append(r"\caption{" + _latex_escape(caption) + r"}\label{" + _latex_escape(label) + r"} \\")
    lines.append(r"\toprule")
    lines.append(top_header)
    lines.append(cmidrule)
    lines.append(bottom_header)
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")
    lines.append(r"\toprule")
    lines.append(top_header)
    lines.append(cmidrule)
    lines.append(bottom_header)
    lines.append(r"\midrule")
    lines.append(r"\endhead")
    lines.append(r"\bottomrule")
    lines.append(r"\endfoot")

    last_split_role: str | None = None
    for row in rows:
        split_role = str(row["split_role"])
        if split_role != last_split_role:
            last_split_role = split_role
            section_text = _latex_escape(SECTION_ROW_LABEL.get(split_role, f"{split_role} datasets"))
            lines.append(r"\midrule")
            lines.append(rf"\multicolumn{{{len(model_labels) + 1}}}{{l}}{{\textbf{{{section_text}}}}} \\")
            lines.append(r"\midrule")

        rendered_model_cells = " & ".join(
            _format_cell(float(row[model_label])) for model_label in model_labels
        )
        lines.append(f"{_latex_escape(str(row['run_name']))} & {rendered_model_cells} \\\\")

    lines.append(r"\end{longtable}")
    lines.append("")
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    default_repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Generate a grouped LaTeX table comparing RMSE across the current basic estimator models."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=default_repo_root,
        help=f"Repository root path (default: {default_repo_root})",
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=Path("outputs"),
        help="Directory containing model artifact subdirectories.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/basic_models_rmse_grouped.tex"),
        help="Output .tex path.",
    )
    parser.add_argument(
        "--caption",
        default=(
            "Per-run RMSE [deg] across the current basic estimator models grouped by split role. "
            "Acronyms: A-LR = accelerometer-only linear ridge, "
            "A-LS = accelerometer-only linear least squares, "
            "P-LR = pressure-only linear ridge, "
            "P-LS = pressure-only linear least squares, "
            "PA-LR = pressure plus accelerometer linear ridge, "
            "PA-LS = pressure plus accelerometer linear least squares, "
            "PA-QR = pressure plus accelerometer quadratic ridge, "
            "LP-LR = lagged pressure linear ridge, "
            "LA-LR = lagged accelerometer linear ridge, "
            "LPA-LR = lagged pressure plus accelerometer ridge, "
            "LP-QR = lagged pressure quadratic ridge, "
            "LA-QR = lagged accelerometer quadratic ridge, "
            "LPA-QR = lagged pressure plus accelerometer quadratic ridge, "
            "MLP = externally trained multilayer perceptron selected by mean validation RMSE."
        ),
        help="LaTeX table caption.",
    )
    parser.add_argument(
        "--label",
        default="tab:basic-model-rmse-grouped",
        help="LaTeX table label.",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=2,
        help="Decimal places for displayed RMSE values (default: 2).",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable cell background gradient coloring.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    repo_root = args.repo_root.resolve()
    outputs_root = (
        args.outputs_root if args.outputs_root.is_absolute() else (repo_root / args.outputs_root)
    )
    output_path = args.output if args.output.is_absolute() else (repo_root / args.output)

    model_runs = _discover_model_sources(repo_root, outputs_root)
    model_labels, rows = _build_rows(model_runs)
    table_tex = _render_longtable(
        model_labels=model_labels,
        rows=rows,
        caption=args.caption,
        label=args.label,
        decimals=args.decimals,
        colorize=(not args.no_color),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(table_tex, encoding="utf-8")
    print(f"Wrote table: {output_path}")
    print(f"Model columns: {', '.join(model_labels)}")
    for model_run in model_runs:
        if model_run.model_name == "mlp_external_best" and model_run.selected_variant is not None:
            print(f"External MLP column uses validation-selected variant: {model_run.selected_variant}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
