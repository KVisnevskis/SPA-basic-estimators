#!/usr/bin/env python3
"""Run the configured SPA basic-estimator benchmark suite."""

from __future__ import annotations

import argparse
import importlib
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Callable

from spa_basic_estimators.utils.config import load_yaml


@dataclass(frozen=True)
class ModelSpec:
    name: str
    runner: str
    model_config: Path


@dataclass(frozen=True)
class SuiteSpec:
    name: str
    data_config: Path
    models: list[ModelSpec]


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _resolve_repo_path(repo_root: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def _load_suite(path: Path, repo_root: Path) -> SuiteSpec:
    raw = load_yaml(path)
    raw_models = raw.get("models")
    if not isinstance(raw_models, list) or not raw_models:
        raise ValueError(f"Benchmark suite must define a non-empty models list: {path}")

    models: list[ModelSpec] = []
    for index, raw_model in enumerate(raw_models, start=1):
        if not isinstance(raw_model, dict):
            raise ValueError(f"Model entry {index} in {path} must be a mapping")

        try:
            name = str(raw_model["name"])
            runner = str(raw_model["runner"])
            model_config = _resolve_repo_path(repo_root, raw_model["model_config"])
        except KeyError as exc:
            raise ValueError(f"Model entry {index} in {path} is missing {exc.args[0]!r}") from exc

        models.append(ModelSpec(name=name, runner=runner, model_config=model_config))

    return SuiteSpec(
        name=str(raw.get("name", path.stem)),
        data_config=_resolve_repo_path(repo_root, raw.get("data_config", "configs/data.yaml")),
        models=models,
    )


def _load_runner(dotted_path: str) -> Callable[..., Any]:
    try:
        module_name, function_name = dotted_path.rsplit(".", 1)
    except ValueError as exc:
        raise ValueError(f"Runner must be a dotted function path: {dotted_path!r}") from exc

    module = importlib.import_module(module_name)
    runner = getattr(module, function_name)
    if not callable(runner):
        raise TypeError(f"Runner is not callable: {dotted_path}")
    return runner


def _selected_models(models: list[ModelSpec], requested_names: list[str] | None) -> list[ModelSpec]:
    if not requested_names:
        return models

    by_name = {model.name: model for model in models}
    missing = [name for name in requested_names if name not in by_name]
    if missing:
        available = ", ".join(sorted(by_name))
        raise ValueError(
            "Unknown model name(s): "
            + ", ".join(missing)
            + f". Available models: {available}"
        )
    return [by_name[name] for name in requested_names]


def _result_summary(result: Any) -> str:
    parts: list[str] = []
    artifact_dir = getattr(result, "artifact_dir", None)
    if artifact_dir is not None:
        parts.append(f"artifacts={artifact_dir}")

    for field_name, label in (
        ("selected_degree", "degree"),
        ("selected_lag", "lag"),
        ("selected_alpha", "alpha"),
    ):
        value = getattr(result, field_name, None)
        if value is not None:
            parts.append(f"{label}={value}")

    validation_metrics = getattr(result, "validation_metrics", None)
    if isinstance(validation_metrics, dict) and "rmse" in validation_metrics:
        parts.append(f"val_rmse={float(validation_metrics['rmse']):.6f}")

    held_out_metrics = getattr(result, "held_out_metrics", None)
    if isinstance(held_out_metrics, dict) and "rmse" in held_out_metrics:
        parts.append(f"held_out_rmse={float(held_out_metrics['rmse']):.6f}")

    return ", ".join(parts) if parts else "completed"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the configured SPA basic-estimator benchmark suite."
    )
    parser.add_argument(
        "--suite-config",
        default="configs/benchmark_suite.yaml",
        help="Benchmark suite YAML path.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Optional subset of model names from the suite to run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List the selected model runs without executing them.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep running later models if one model fails.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    repo_root = _repo_root()
    suite_path = _resolve_repo_path(repo_root, args.suite_config)
    suite = _load_suite(suite_path, repo_root)
    models = _selected_models(suite.models, args.models)

    print(f"Suite: {suite.name}")
    print(f"Data config: {suite.data_config}")
    print(f"Models selected: {len(models)}")

    if args.dry_run:
        for model in models:
            print(f"- {model.name}: {model.model_config}")
        return 0

    failures: list[tuple[str, str]] = []
    suite_start = time.perf_counter()
    for index, model in enumerate(models, start=1):
        print(f"[{index}/{len(models)}] Running {model.name}")
        model_start = time.perf_counter()
        try:
            runner = _load_runner(model.runner)
            result = runner(
                data_config_path=suite.data_config,
                model_config_path=model.model_config,
            )
        except Exception as exc:
            failures.append((model.name, str(exc)))
            print(f"[{index}/{len(models)}] FAILED {model.name}: {exc}")
            if not args.continue_on_error:
                break
            continue

        elapsed_s = time.perf_counter() - model_start
        print(f"[{index}/{len(models)}] Done {model.name} in {elapsed_s:.1f}s: {_result_summary(result)}")

    suite_elapsed_s = time.perf_counter() - suite_start
    if failures:
        print("")
        print("Failures:")
        for model_name, message in failures:
            print(f"- {model_name}: {message}")
        print(f"Suite stopped/finished in {suite_elapsed_s:.1f}s with {len(failures)} failure(s).")
        return 1

    print(f"Suite completed in {suite_elapsed_s:.1f}s.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
