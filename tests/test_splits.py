from __future__ import annotations

from spa_basic_estimators.utils.data_loader import load_data_config
from spa_basic_estimators.utils.splits import build_run_to_split_map


def test_build_run_to_split_map_returns_expected_mapping() -> None:
    splits = {
        "train": ["run_a", "run_b"],
        "val": ["run_c"],
        "held_out": ["run_d"],
    }

    assert build_run_to_split_map(splits) == {
        "run_a": "train",
        "run_b": "train",
        "run_c": "val",
        "run_d": "held_out",
    }


def test_build_run_to_split_map_matches_repo_data_config() -> None:
    config = load_data_config("configs/data.yaml")

    assert build_run_to_split_map(config.splits) == {
        "Freehand_tt_1": "train",
        "Freehand_static_03V_1": "train",
        "Freehand_static_09V_1": "train",
        "Freehand_sin_1": "train",
        "run_0roll_0pitch_tt_1": "train",
        "Freehand_tt_2": "val",
        "Freehand_static_03V_2": "val",
        "Freehand_static_09V_2": "val",
        "Freehand_sin_2": "val",
        "run_0roll_0pitch_tt_2": "val",
        "run_0roll_90pitch_tt_1": "held_out",
        "Freehand_tt_3": "held_out",
        "Freehand_static_03V_3": "held_out",
        "Freehand_static_06V_3": "held_out",
        "Freehand_static_09V_3": "held_out",
        "Freehand_sin_3": "held_out",
    }
