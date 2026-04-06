from __future__ import annotations

import argparse

from spa_basic_estimators.estimators.pressure_accel_linear_least_squares import (
    run_pressure_accel_linear_least_squares,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the pressure+accelerometer ordinary least-squares baseline."
    )
    parser.add_argument(
        "--data-config",
        default="configs/data.yaml",
        help="Path to the shared data config YAML.",
    )
    parser.add_argument(
        "--model-config",
        default="configs/models/pressure_accel_linear_least_squares.yaml",
        help="Path to the pressure+accelerometer least-squares model config YAML.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    result = run_pressure_accel_linear_least_squares(
        data_config_path=args.data_config,
        model_config_path=args.model_config,
    )

    print(f"Artifacts saved to: {result.artifact_dir}")
    print(f"Validation RMSE: {result.validation_metrics['rmse']:.6f}")
    print(f"Held-out RMSE: {result.held_out_metrics['rmse']:.6f}")


if __name__ == "__main__":
    main()
