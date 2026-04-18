#!/usr/bin/env python3
"""Temporary plotting helper for lag-vs-RMSE investigation."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot best validation RMSE against lag for a lagged-model search table."
    )
    parser.add_argument(
        "--validation-search",
        default="outputs/investigation/lagged_pressure_accel_ridge_wider_lag_search/validation_search.csv",
        help="Path to validation_search.csv for the lagged model.",
    )
    parser.add_argument(
        "--output",
        default="outputs/investigation/lagged_pressure_accel_ridge_wider_lag_search/lag_vs_validation_rmse.png",
        help="Output image path.",
    )
    parser.add_argument(
        "--title",
        default="Lagged Pressure+Accel Ridge: Best Validation RMSE vs Lag",
        help="Plot title.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    validation_search_path = Path(args.validation_search).resolve()
    output_path = Path(args.output).resolve()

    frame = pd.read_csv(validation_search_path)
    if not {"lag_length", "alpha", "rmse"}.issubset(frame.columns):
        raise ValueError(
            f"{validation_search_path} must contain lag_length, alpha, and rmse columns"
        )

    best_per_lag = (
        frame.sort_values(["lag_length", "rmse", "alpha"])
        .groupby("lag_length", as_index=False)
        .first()
        .sort_values("lag_length")
        .reset_index(drop=True)
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(
        best_per_lag["lag_length"],
        best_per_lag["rmse"],
        marker="o",
        linewidth=2.0,
        markersize=5.0,
        color="#2F5D62",
    )

    best_row = best_per_lag.loc[best_per_lag["rmse"].idxmin()]
    ax.scatter(
        [best_row["lag_length"]],
        [best_row["rmse"]],
        color="#D97D54",
        s=70,
        zorder=3,
        label=f"Best lag = {int(best_row['lag_length'])}",
    )
    ax.annotate(
        f"L={int(best_row['lag_length'])}\nRMSE={best_row['rmse']:.4f}",
        xy=(best_row["lag_length"], best_row["rmse"]),
        xytext=(10, -35),
        textcoords="offset points",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#B0B0B0"},
        arrowprops={"arrowstyle": "->", "color": "#666666", "linewidth": 0.8},
    )

    for _, row in best_per_lag.iterrows():
        ax.annotate(
            f"{int(row['lag_length'])}",
            xy=(row["lag_length"], row["rmse"]),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=7,
            color="#555555",
        )

    ax.set_title(args.title)
    ax.set_xlabel("Lag Length")
    ax.set_ylabel("Best Validation RMSE [stored units]")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    best_per_lag.to_csv(output_path.with_suffix(".csv"), index=False)
    print(f"Wrote plot: {output_path}")
    print(f"Wrote summary: {output_path.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
