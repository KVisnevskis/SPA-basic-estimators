from __future__ import annotations

from collections.abc import Mapping

REQUIRED_SPLITS = ("train", "val", "held_out")
SPLIT_COLUMN = "__split__"
HDF5_KEY_COLUMN = "__hdf5_key__"


def validate_splits(
    splits: Mapping[str, list[str]],
    require_unique_run_ids: bool = True,
) -> dict[str, list[str]]:
    missing_splits = [name for name in REQUIRED_SPLITS if name not in splits]
    if missing_splits:
        raise ValueError(f"Missing required splits: {missing_splits}")

    normalised: dict[str, list[str]] = {}
    seen_run_ids: dict[str, str] = {}

    for split_name in REQUIRED_SPLITS:
        run_ids = [str(run_id) for run_id in splits.get(split_name, [])]
        normalised[split_name] = run_ids

        if not require_unique_run_ids:
            continue

        split_duplicates = sorted({run_id for run_id in run_ids if run_ids.count(run_id) > 1})
        if split_duplicates:
            raise ValueError(
                f"Duplicate run IDs found within split '{split_name}': {split_duplicates}"
            )

        for run_id in run_ids:
            previous_split = seen_run_ids.get(run_id)
            if previous_split is not None:
                raise ValueError(
                    f"Run ID '{run_id}' appears in both '{previous_split}' and '{split_name}'"
                )
            seen_run_ids[run_id] = split_name

    return normalised


def build_run_to_split_map(
    splits: Mapping[str, list[str]],
    require_unique_run_ids: bool = True,
) -> dict[str, str]:
    validated = validate_splits(
        splits,
        require_unique_run_ids=require_unique_run_ids,
    )
    return {
        run_id: split_name
        for split_name, run_ids in validated.items()
        for run_id in run_ids
    }

