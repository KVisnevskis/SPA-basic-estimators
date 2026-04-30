# SPA Basic Estimators

This repository contains the linear least-squares, ridge, polynomial ridge, and lagged ridge baseline estimators used in the PhD thesis *Data-Driven Bending Angle Estimation for Soft Pneumatic Actuators: Dataset, Methods, and Comparative Evaluation*. The code loads the preprocessed soft pneumatic actuator bending dataset, fits the configured baseline estimators, writes validation and held-out metrics, and generates grouped RMSE tables for comparison with the companion LSTM and MLP workflows.

## Data Dependency

The estimator configs expect the preprocessed HDF5 dataset at:

```text
data/preprocessed_all_trials.h5
```

The dataset is archived on Zenodo at https://doi.org/10.5281/zenodo.18697336. The preprocessing workflow that produces the expected HDF5 schema is in the companion repository https://github.com/KVisnevskis/SPA-data-pre-processing.

The default `configs/data.yaml` uses the expanded development split from the final baseline study. The original 5-train/5-validation development split is preserved in `configs/data_original_development_split.yaml`.

The repository intentionally ignores `data/*` and `outputs/*`; keep downloaded datasets and generated artifacts local.

## Reproducing The Main Results

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
# Place the Zenodo/preprocessed HDF5 file at data/preprocessed_all_trials.h5
.\.venv\Scripts\python.exe scripts\run_all_basic_estimators.py
.\.venv\Scripts\python.exe scripts\basic_models_rmse_grouped_table.py
```

The per-model runs write artifacts under `outputs/<model_name>/`, including:

- `validation_metrics.json` and `held_out_metrics.json`
- `validation_search.csv`
- `validation_predictions.csv` and `held_out_predictions.csv`
- `coefficient_table.csv`
- `run_summary.json`
- `all_dataset_predictions.h5`

The grouped thesis table is written to `outputs/basic_models_rmse_grouped.tex`.

For the selected-hyperparameter expanded-development-set study, run:

```powershell
.\.venv\Scripts\python.exe scripts\run_all_basic_estimators.py --suite-config configs/benchmark_suite_expanded_development_selected.yaml
```

Those configs live under `configs/models_additional_sets/` and write artifacts under `outputs/additional_sets/`.

## Companion Repositories

- Data preprocessing: https://github.com/KVisnevskis/SPA-data-pre-processing
- Basic estimators: https://github.com/KVisnevskis/SPA-basic-estimators
- LSTM training and evaluation: https://github.com/KVisnevskis/SPA-LSTM-training

MLP values used in grouped comparison tables are treated as external inputs when `outputs/mlp_external/mlp_hpo_rmse_per_run_grouped.csv` is present; MLP training itself is not implemented in this repository.

## Environment

The project targets Python 3.11 or newer. The HDF5 files are read and written through pandas/PyTables, so installing from `requirements.txt` is the recommended setup path.

## Citation

If you use this repository, please cite the companion thesis:

```bibtex
@phdthesis{visnevskis2026spa,
  author = {Visnevskis, Krisjanis},
  title = {Data-Driven Bending Angle Estimation for Soft Pneumatic Actuators: Dataset, Methods, and Comparative Evaluation},
  school = {University of Aberdeen},
  year = {2026}
}
```
