#!/usr/bin/env python3
"""Common-overlap lag investigation for lagged pressure+accel ridge."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from spa_basic_estimators.estimators.lagged_pressure_accel_ridge import (
    build_lagged_pressure_accel_dataset,
    load_lagged_pressure_accel_ridge_config,
)
from spa_basic_estimators.estimators.pressure_ridge_common import (
    SAMPLE_INDEX_COLUMN,
    compute_regression_metrics,
)
from spa_basic_estimators.utils.data_loader import load_data_config, load_runs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a lag sweep for the lagged pressure+accel ridge model "
            "using common-overlap evaluation samples."
        )
    )
    parser.add_argument(
        "--data-config",
        default="configs/data.yaml",
        help="Path to the shared data config YAML.",
    )
    parser.add_argument(
        "--model-config",
        default="configs/models/lagged_pressure_accel_ridge.yaml",
        help="Path to the base lagged pressure+accel ridge config YAML.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/investigation/lagged_pressure_accel_ridge_common_overlap_max500",
        help="Directory for investigation outputs.",
    )
    parser.add_argument(
        "--max-lag",
        type=int,
        default=500,
        help="Maximum lag to assess. Default: 500.",
    )
    return parser


def build_investigation_lag_grid(max_lag: int) -> list[int]:
    if max_lag < 1:
        raise ValueError("max_lag must be positive")

    base = [1, 2, 3, 5, 7, 10, 15, 20, 30, 40, 60, 80, 90, 100]
    extensions = [
        110,
        120,
        130,
        140,
        150,
        160,
        180,
        200,
        225,
        250,
        275,
        300,
        325,
        350,
        375,
        400,
        425,
        450,
        475,
        500,
    ]
    grid = [lag for lag in [*base, *extensions] if lag <= max_lag]
    if max_lag not in grid:
        grid.append(max_lag)
    return sorted(dict.fromkeys(grid))


def common_overlap_mask(metadata: pd.DataFrame, common_start_index: int) -> np.ndarray:
    return metadata[SAMPLE_INDEX_COLUMN].to_numpy(dtype=int) >= common_start_index


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    data_config = load_data_config(args.data_config)
    base_config = load_lagged_pressure_accel_ridge_config(args.model_config)
    runs = load_runs(data_config)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    lag_grid = build_investigation_lag_grid(args.max_lag)
    common_start_index = max(lag_grid) - 1

    min_split_lengths: dict[str, int] = {}
    for split_name in ("train", "val", "held_out"):
        min_split_lengths[split_name] = min(len(runs[run_id]) for run_id in data_config.splits[split_name])
        if min_split_lengths[split_name] < max(lag_grid):
            raise ValueError(
                f"Split '{split_name}' contains a run shorter than the requested max lag "
                f"({min_split_lengths[split_name]} < {max(lag_grid)})"
            )

    search_rows: list[dict[str, float | int]] = []
    best_rows: list[dict[str, float | int]] = []

    for lag_length in lag_grid:
        dataset = build_lagged_pressure_accel_dataset(runs, data_config, lag_length)
        val_mask = common_overlap_mask(dataset.val.metadata, common_start_index)
        held_mask = common_overlap_mask(dataset.held_out.metadata, common_start_index)

        if not np.any(val_mask):
            raise ValueError(f"No validation samples remain for lag {lag_length} under common-overlap masking")
        if not np.any(held_mask):
            raise ValueError(f"No held-out samples remain for lag {lag_length} under common-overlap masking")

        best_alpha: float | None = None
        best_val_rmse: float | None = None

        for alpha in base_config.alpha_grid:
            model = Ridge(alpha=alpha, fit_intercept=base_config.fit_intercept)
            model.fit(dataset.train.X, dataset.train.y)

            val_predictions = model.predict(dataset.val.X[val_mask])
            val_metrics = compute_regression_metrics(dataset.val.y[val_mask], val_predictions)
            search_rows.append(
                {
                    "lag_length": int(lag_length),
                    "alpha": float(alpha),
                    "common_overlap_start_index": int(common_start_index),
                    "validation_row_count": int(np.sum(val_mask)),
                    "held_out_row_count": int(np.sum(held_mask)),
                    **{f"validation_{key}": value for key, value in val_metrics.items()},
                }
            )

            if best_val_rmse is None or val_metrics["rmse"] < best_val_rmse:
                best_alpha = float(alpha)
                best_val_rmse = float(val_metrics["rmse"])

        if best_alpha is None:
            raise ValueError(f"Could not select best alpha for lag {lag_length}")

        final_model = Ridge(alpha=best_alpha, fit_intercept=base_config.fit_intercept)
        final_model.fit(dataset.train.X, dataset.train.y)

        val_predictions = final_model.predict(dataset.val.X[val_mask])
        held_predictions = final_model.predict(dataset.held_out.X[held_mask])
        validation_metrics = compute_regression_metrics(dataset.val.y[val_mask], val_predictions)
        held_out_metrics = compute_regression_metrics(dataset.held_out.y[held_mask], held_predictions)

        best_rows.append(
            {
                "lag_length": int(lag_length),
                "selected_alpha": float(best_alpha),
                "common_overlap_start_index": int(common_start_index),
                "validation_row_count": int(np.sum(val_mask)),
                "held_out_row_count": int(np.sum(held_mask)),
                **{f"validation_{key}": value for key, value in validation_metrics.items()},
                **{f"held_out_{key}": value for key, value in held_out_metrics.items()},
            }
        )

    search_frame = pd.DataFrame(search_rows).sort_values(["lag_length", "alpha"]).reset_index(drop=True)
    best_frame = pd.DataFrame(best_rows).sort_values("lag_length").reset_index(drop=True)

    search_frame.to_csv(output_dir / "common_overlap_search.csv", index=False)
    best_frame.to_csv(output_dir / "common_overlap_best_per_lag.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(
        best_frame["lag_length"],
        best_frame["validation_rmse"],
        marker="o",
        linewidth=2.0,
        markersize=4.5,
        color="#2F5D62",
        label="Validation RMSE",
    )
    ax.plot(
        best_frame["lag_length"],
        best_frame["held_out_rmse"],
        marker="s",
        linewidth=1.8,
        markersize=4.0,
        color="#D97D54",
        label="Held-out RMSE",
    )

    best_val_row = best_frame.loc[best_frame["validation_rmse"].idxmin()]
    ax.scatter(
        [best_val_row["lag_length"]],
        [best_val_row["validation_rmse"]],
        color="#153B50",
        s=70,
        zorder=4,
    )
    ax.annotate(
        f"Best val: L={int(best_val_row['lag_length'])}\nRMSE={best_val_row['validation_rmse']:.4f}",
        xy=(best_val_row["lag_length"], best_val_row["validation_rmse"]),
        xytext=(10, -32),
        textcoords="offset points",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#B0B0B0"},
        arrowprops={"arrowstyle": "->", "color": "#666666", "linewidth": 0.8},
    )

    ax.set_title("Lagged Pressure+Accel Ridge: Common-Overlap RMSE vs Lag")
    ax.set_xlabel("Lag Length")
    ax.set_ylabel("RMSE [stored units]")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "common_overlap_lag_vs_rmse.png", dpi=180)
    plt.close(fig)

    summary = {
        "investigation_type": "common_overlap_lag_sweep",
        "base_model_config_path": str(Path(args.model_config).resolve()),
        "output_dir": str(output_dir),
        "lag_grid": lag_grid,
        "alpha_grid": list(base_config.alpha_grid),
        "common_overlap_start_index": int(common_start_index),
        "min_split_lengths": min_split_lengths,
        "best_validation_lag": int(best_val_row["lag_length"]),
        "best_validation_alpha": float(best_val_row["selected_alpha"]),
        "best_validation_rmse": float(best_val_row["validation_rmse"]),
        "best_validation_held_out_rmse": float(best_val_row["held_out_rmse"]),
    }
    (output_dir / "common_overlap_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(f"Wrote search grid results to: {output_dir / 'common_overlap_search.csv'}")
    print(f"Wrote best-per-lag summary to: {output_dir / 'common_overlap_best_per_lag.csv'}")
    print(f"Wrote plot to: {output_dir / 'common_overlap_lag_vs_rmse.png'}")
    print(
        "Best validation lag on common-overlap samples: "
        f"{int(best_val_row['lag_length'])} (alpha={float(best_val_row['selected_alpha'])})"
    )


if __name__ == "__main__":
    main()
