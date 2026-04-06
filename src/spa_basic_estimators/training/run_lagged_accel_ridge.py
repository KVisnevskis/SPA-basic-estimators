from __future__ import annotations

import argparse

from spa_basic_estimators.estimators.lagged_accel_ridge import (
    run_lagged_accel_ridge,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the lagged accelerometer-only ridge baseline."
    )
    parser.add_argument(
        "--data-config",
        default="configs/data.yaml",
        help="Path to the shared data config YAML.",
    )
    parser.add_argument(
        "--model-config",
        default="configs/models/lagged_accel_ridge.yaml",
        help="Path to the lagged accelerometer-only ridge model config YAML.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    result = run_lagged_accel_ridge(
        data_config_path=args.data_config,
        model_config_path=args.model_config,
    )

    print(f"Artifacts saved to: {result.artifact_dir}")
    print(f"Selected lag: {result.selected_lag}")
    print(f"Selected alpha: {result.selected_alpha}")
    print(f"Validation RMSE: {result.validation_metrics['rmse']:.6f}")
    print(f"Held-out RMSE: {result.held_out_metrics['rmse']:.6f}")


if __name__ == "__main__":
    main()
