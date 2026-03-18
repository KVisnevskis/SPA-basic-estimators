# Technical Specification: Basic Benchmark Estimators for SPA Bending Angle Estimation

## 1. Purpose

This repository will implement a compact suite of simple benchmark estimators for SPA bending angle estimation. The purpose of these estimators is to provide fair, reproducible, and interpretable baselines against which the existing LSTM models can be compared.

The benchmark suite must answer the following question:

> How much performance improvement does the LSTM provide over simpler static and low-complexity dynamic models when all models are trained and evaluated on the same pre-processed dataset splits?

The benchmark suite is intentionally limited to simple methods that are:

- easy to implement and reproduce,
- computationally lightweight,
- straightforward to interpret,
- plausible alternatives that a reviewer may reasonably expect as baselines.

---

## 2. Scope

The repository will implement the following six benchmark models:

1. **Pressure-only linear regression with ridge regularisation**
2. **Pressure-only quadratic regression with ridge regularisation**
3. **Pressure + accelerometer linear regression with ridge regularisation**
4. **Pressure + accelerometer quadratic regression with ridge regularisation**
5. **Pressure + accelerometer minimal neural network (MLP)**
6. **Lagged pressure + accelerometer ridge regression**

These models will use the same pre-processed datasets as the existing LSTM pipeline.

Dataset split:

- **5 dedicated training runs**
- **5 dedicated validation runs**
- **6 dedicated held-out runs**

All benchmark models must use the exact same outer split policy as the LSTM experiments.

---

## 3. High-Level Requirements

### 3.1 Fair comparison requirements

To preserve comparability with the LSTM models:

- All benchmark models must be trained using only the **training runs**.
- Hyperparameter selection must use only the **validation runs**.
- Final reported benchmark performance must be computed only on the **held-out runs**.
- No information from validation or held-out runs may be used during preprocessing fitting, feature fitting, model fitting, or scaler fitting.

### 3.2 General repository goals

The repository must:

- load the pre-processed runs from disk,
- construct feature matrices for each model family,
- train and validate all benchmark models,
- store trained models and preprocessing objects,
- generate evaluation outputs on held-out runs,
- export metrics, predictions, and plots,
- support reproducible reruns using configuration files,
- support extension to future benchmarks.

---

## 4. Data Assumptions and Interface

### 4.1 Input dataset assumptions

The starting point is the same pre-processed dataset used by the LSTM training pipeline.

Each run is assumed to contain time-aligned samples of:

- pressure features,
- accelerometer features,
- target bending angle.

The repository should **not** implement raw-data decoding or time synchronisation. Those steps are assumed to have already been completed upstream.

### 4.2 Expected per-sample quantities

Let each time step be indexed by \( t \in \{1, 2, \dots, T\} \).

For a given run:

- pressure feature vector:
  \[
  \mathbf{p}(t) \in \mathbb{R}^{d_p}
  \]
- accelerometer feature vector:
  \[
  \mathbf{a}(t) \in \mathbb{R}^{3}
  \]
- target bending angle:
  \[
  \phi(t) \in \mathbb{R}
  \]

Where:

- \( d_p \) is the number of pressure channels/features retained by the preprocessing pipeline,
- \( \phi(t) \) is the ground-truth bending angle in degrees.

### 4.3 Expected data storage format

The repository should support loading pre-processed runs from a structured storage format such as:

- HDF5,
- Parquet,
- CSV per run,
- pickled pandas DataFrames.

The exact format will be configurable.

Each run must be identifiable by a unique run ID and assigned to one of:

- `train`
- `val`
- `held_out`

### 4.4 Required columns

At minimum, each run must expose:

- run identifier,
- time index or timestamp,
- one or more pressure columns,
- accelerometer columns,
- target angle column.

Example required logical fields:

- `run_id`
- `time`
- `pressure_*`
- `acc_x`
- `acc_y`
- `acc_z`
- `target_angle_deg`

Actual column names must be configurable via YAML.

---

## 5. Split Policy

### 5.1 Fixed split usage

The repository must use three fixed disjoint sets:

- training runs,
- validation runs,
- held-out runs.

These splits must be defined explicitly in configuration and never inferred automatically from sample-level shuffling.

### 5.2 Run-level isolation

All train/validation/test separation must occur at the **run level**, not at the individual sample level.

This is critical because adjacent samples within a run are highly correlated. Sample-level random shuffling would create leakage and unrealistically optimistic results.

### 5.3 Validation usage

Validation runs are used only for:

- selecting ridge penalty,
- selecting polynomial degree where relevant,
- selecting MLP hidden dimension / weight decay / learning rate,
- selecting lag window size for lagged regression,
- selecting any optional preprocessing settings that are not fixed in advance.

Held-out runs are used only once for final reporting after model selection is complete.

---

## 6. Preprocessing Policy

## 6.1 General principle

All preprocessing objects that learn parameters from data must be fit on **training runs only**.

This includes:

- input feature scalers,
- target scaler if used,
- polynomial feature expansion fitting if implemented using a fitted transformer,
- any learned feature normalisation.

### 6.2 Feature scaling

The default approach should be:

- fit input scaler on concatenated training samples,
- transform training, validation, and held-out features using the fitted training scaler.

Recommended default:

- `StandardScaler` for regression and MLP feature inputs.

Alternative allowed if desired:

- min-max scaling to match the LSTM preprocessing convention, but only if this is already part of the upstream preprocessed data.

### 6.3 Target scaling

The target angle may be left in physical units (degrees), which simplifies interpretability.

If target scaling is introduced for the MLP:

- the scaling object must be fit on training targets only,
- predictions must be inverse-transformed before metric computation.

Default recommendation:

- do **not** scale the target for ridge models,
- optional target scaling for MLP only if training stability benefits from it.

### 6.4 Missing data handling

The repository should assume the pre-processed runs are already cleaned.

If missing values are present:

- fail fast with a clear error by default,
- optional imputation support can be added later, but is not required for the first implementation.

### 6.5 Sequence boundaries

For lagged models:

- lagged windows must be constructed **within each run only**,
- no lagged sample may span across a run boundary.

---

## 7. Mathematical Formulation of the Regression Models

## 7.1 General supervised regression setting

For each time sample \( i \), define:

- feature vector \( \mathbf{x}\_i \in \mathbb{R}^{m} \),
- target \( y_i = \phi_i \in \mathbb{R} \).

The dataset formed from all training runs is:
\[
\mathcal{D}_{\text{train}} = \{(\mathbf{x}\_i, y_i)\}_{i=1}^{N}
\]

The goal is to learn a mapping:
\[
f: \mathbb{R}^{m} \rightarrow \mathbb{R}
\]
such that:
\[
\hat{y}\_i = f(\mathbf{x}\_i)
\]
approximates the target bending angle.

---

## 7.2 Linear ridge regression

For standard linear regression:
\[
\hat{y}\_i = \mathbf{w}^\top \mathbf{x}\_i + b
\]

Ridge regression estimates parameters by minimising:
\[
\mathcal{L}(\mathbf{w}, b) =
\sum\_{i=1}^{N}
\left(
y_i - (\mathbf{w}^\top \mathbf{x}\_i + b)
\right)^2

- \lambda \|\mathbf{w}\|\_2^2
  \]

Where:

- \( \mathbf{w} \) is the coefficient vector,
- \( b \) is the intercept,
- \( \lambda \ge 0 \) is the ridge regularisation coefficient.

This penalty discourages large coefficient magnitudes and improves stability in the presence of correlated features.

---

## 7.3 Polynomial regression with ridge regularisation

Polynomial regression is implemented by first mapping the original feature vector \( \mathbf{x} \) into a nonlinear feature space:
\[
\boldsymbol{\psi}(\mathbf{x})
\]

For degree-2 polynomial regression:
\[
\boldsymbol{\psi}(\mathbf{x}) =
[
1,\,
x_1,\dots,x_m,\,
x_1^2,\dots,x_m^2,\,
x_1x_2,\dots
]
\]

The model is then linear in the expanded feature space:
\[
\hat{y}\_i = \mathbf{w}^\top \boldsymbol{\psi}(\mathbf{x}\_i) + b
\]

The optimisation becomes:
\[
\mathcal{L}(\mathbf{w}, b) =
\sum\_{i=1}^{N}
\left(
y_i - (\mathbf{w}^\top \boldsymbol{\psi}(\mathbf{x}\_i) + b)
\right)^2

- \lambda \|\mathbf{w}\|\_2^2
  \]

This allows the model to represent nonlinear but memoryless input-output relationships.

---

## 7.4 Lagged regression

For lagged regression, the feature vector at time \( t \) includes current and recent past input samples.

For lag window length \( L \), define:
\[
\mathbf{x}^{\text{lag}}(t) =
[
\mathbf{u}(t),
\mathbf{u}(t-1),
\dots,
\mathbf{u}(t-L+1)
]
\]

Where:
\[
\mathbf{u}(t) =
[\mathbf{p}(t), \mathbf{a}(t)]
\]

The prediction becomes:
\[
\hat{\phi}(t) = \mathbf{w}^\top \mathbf{x}^{\text{lag}}(t) + b
\]

The optimisation is again ridge regression:
\[
\mathcal{L}(\mathbf{w}, b) =
\sum\_{t=L}^{T}
\left(
\phi(t) - (\mathbf{w}^\top \mathbf{x}^{\text{lag}}(t) + b)
\right)^2

- \lambda \|\mathbf{w}\|\_2^2
  \]

This provides a simple dynamic baseline with finite memory.

---

## 7.5 Minimal MLP

The minimal MLP is a feedforward neural network:
\[
\hat{y} = f\_{\theta}(\mathbf{x})
\]

A minimal 1-hidden-layer architecture is sufficient:
\[
\mathbf{h} = \sigma(\mathbf{W}\_1 \mathbf{x} + \mathbf{b}\_1)
\]
\[
\hat{y} = \mathbf{W}\_2 \mathbf{h} + b_2
\]

Where:

- \( \sigma \) is a nonlinear activation such as ReLU or tanh,
- \( \theta = \{\mathbf{W}\_1,\mathbf{b}\_1,\mathbf{W}\_2,b_2\} \) are trainable parameters.

Training objective:
\[
\mathcal{L}(\theta) =
\frac{1}{N}
\sum\_{i=1}^{N}
(y_i - \hat{y}\_i)^2

- \alpha \cdot \Omega(\theta)
  \]

Where:

- the first term is mean squared error,
- \( \Omega(\theta) \) is optional L2 regularisation,
- \( \alpha \) is the weight decay coefficient.

This serves as a minimal nonlinear static benchmark.

---

## 8. Benchmark Model Definitions

## 8.1 Model 1: Pressure-only linear ridge

### Purpose

This is the minimum plausible benchmark. It tests how much of the bending angle can be explained using only pressure through a linear static map.

### Inputs

\[
\mathbf{x}(t) = \mathbf{p}(t)
\]

### Model form

\[
\hat{\phi}(t) = \mathbf{w}^\top \mathbf{p}(t) + b
\]

### Implementation notes

- Use `Ridge` from scikit-learn.
- Default `fit_intercept=True`.
- Use input scaling.
- No polynomial features.
- Hyperparameter to tune: `alpha`.

### Suggested alpha grid

\[
\{10^{-6}, 10^{-5}, 10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}, 1, 10, 100\}
\]

### Output artifacts

- fitted ridge model,
- input scaler,
- validation metrics,
- held-out metrics,
- coefficient table.

---

## 8.2 Model 2: Pressure-only quadratic ridge

### Purpose

This tests whether a simple nonlinear static pressure-angle mapping is sufficient.

### Inputs

\[
\mathbf{x}(t) = \mathbf{p}(t)
\]

### Feature expansion

Degree-2 polynomial expansion:
\[
\boldsymbol{\psi}(\mathbf{p}(t))
\]

### Model form

\[
\hat{\phi}(t) = \mathbf{w}^\top \boldsymbol{\psi}(\mathbf{p}(t)) + b
\]

### Implementation notes

- Use `PolynomialFeatures(degree=2, include_bias=False)`.
- Then use `Ridge`.
- Scaling should be applied to the original input features before polynomial expansion, or to the expanded features depending on implementation choice. The approach must be consistent and documented.
- Recommended implementation:
  - scale raw inputs,
  - generate polynomial features,
  - optionally scale expanded features if needed.

### Hyperparameters

- `alpha`
- polynomial degree fixed at 2 for this benchmark

### Output artifacts

- polynomial transformer,
- scaler(s),
- ridge model,
- coefficient table mapped to feature names,
- metrics and predictions.

---

## 8.3 Model 3: Pressure + accelerometer linear ridge

### Purpose

This benchmark tests whether simple gravity-related orientation cues from the accelerometer, combined with pressure, are enough without nonlinearities or temporal memory.

### Inputs

\[
\mathbf{x}(t) =
[
\mathbf{p}(t),\,
\mathbf{a}(t)
]
\]

If pressure has \( d_p \) channels:
\[
\mathbf{x}(t) \in \mathbb{R}^{d_p + 3}
\]

### Model form

\[
\hat{\phi}(t) = \mathbf{w}^\top \mathbf{x}(t) + b
\]

### Implementation notes

- Use `Ridge`.
- Scale all input features.
- Hyperparameter to tune: `alpha`.

### Output artifacts

- fitted model,
- input scaler,
- coefficient analysis showing relative contribution of pressure and accel terms,
- metrics and predictions.

---

## 8.4 Model 4: Pressure + accelerometer quadratic ridge

### Purpose

This is the strongest static regression baseline. It captures nonlinear interactions between pressure and accelerometer channels without temporal memory.

### Inputs

\[
\mathbf{x}(t) =
[
\mathbf{p}(t),\,
\mathbf{a}(t)
]
\]

### Feature expansion

Degree-2 polynomial expansion:
\[
\boldsymbol{\psi}(\mathbf{x}(t))
\]

This includes:

- linear terms,
- squared terms,
- pairwise interaction terms.

### Model form

\[
\hat{\phi}(t) = \mathbf{w}^\top \boldsymbol{\psi}(\mathbf{x}(t)) + b
\]

### Implementation notes

- Use scikit-learn pipeline if convenient:
  - scaler,
  - polynomial expansion,
  - ridge regression.
- Tune `alpha`.
- Keep degree fixed at 2 for first implementation.
- Ensure feature name tracking for interpretability.

### Output artifacts

- transformer + model,
- top coefficient magnitudes,
- validation and held-out metrics,
- prediction traces for each run.

---

## 8.5 Model 5: Pressure + accelerometer minimal MLP

### Purpose

This tests whether a very small nonlinear feedforward network can match the LSTM without using temporal recurrence.

### Inputs

\[
\mathbf{x}(t) =
[
\mathbf{p}(t),\,
\mathbf{a}(t)
]
\]

### Recommended architecture

A minimal default architecture:

- input layer of size \( d_p + 3 \),
- one hidden layer,
- one scalar output.

Example:

- Dense(16, activation=ReLU)
- Dense(1, linear)

Alternative small sizes allowed for tuning:

- 8, 16, 32 hidden units

### Training objective

Mean squared error loss.

### Regularisation

At least one of:

- L2 weight decay,
- early stopping,
- optional dropout set to very small values or omitted entirely.

Recommended default:

- L2 weight decay + early stopping
- no dropout initially

### Optimiser

- Adam

### Suggested hyperparameters

- hidden units: `{8, 16, 32}`
- learning rate: `{1e-4, 3e-4, 1e-3}`
- weight decay / L2: `{0, 1e-6, 1e-5, 1e-4}`
- batch size: `{128, 256, 512}`

### Training protocol

- concatenate all training samples from all training runs,
- shuffle training samples between epochs,
- do **not** mix validation samples into training,
- use early stopping based on validation RMSE or validation loss,
- restore best weights.

### Framework

Keras is preferred for consistency with the existing LSTM ecosystem, but PyTorch is also acceptable if the repository remains simple.

### Output artifacts

- saved trained model,
- training history,
- best epoch,
- validation curves,
- held-out predictions,
- metrics.

---

## 8.6 Model 6: Lagged pressure + accelerometer ridge

### Purpose

This is a simple dynamic baseline with finite memory. It is the key comparator against the LSTM because it tests whether short recent history alone already explains most of the temporal structure.

### Inputs

Define the instantaneous input vector:
\[
\mathbf{u}(t) =
[
\mathbf{p}(t),\,
\mathbf{a}(t)
]
\]

For lag length \( L \), construct:
\[
\mathbf{x}^{\text{lag}}(t) =
[
\mathbf{u}(t),
\mathbf{u}(t-1),
\dots,
\mathbf{u}(t-L+1)
]
\]

### Model form

\[
\hat{\phi}(t) = \mathbf{w}^\top \mathbf{x}^{\text{lag}}(t) + b
\]

### Lag construction rules

- windows must be built independently for each run,
- the first \( L-1 \) samples of each run are dropped,
- no padding across run boundaries,
- the target remains \( \phi(t) \) at the current time.

### Candidate lag lengths

Initial grid:

- \( L \in \{3, 5, 10, 20\} \)

The exact lag values may later be interpreted in seconds using the dataset sample rate.

### Hyperparameters

- lag length \( L \)
- ridge `alpha`

### Implementation notes

- fit input scaler on training lagged feature matrix only,
- apply same scaler to validation and held-out lagged matrices,
- use ridge regression on flattened lagged features.

### Output artifacts

- best lag length,
- best alpha,
- coefficient vector,
- metrics,
- predictions aligned to trimmed time series.

---

## 9. Feature Construction Requirements

## 9.1 Pressure feature handling

The number and identity of pressure inputs must be configurable.

Examples:

- single pressure channel,
- two chamber pressures,
- differential pressure,
- multiple engineered pressure features already provided upstream.

The repository must not hard-code a single pressure representation. The config should specify which columns are used.

### 9.2 Accelerometer feature handling

The default accelerometer inputs are:

- `acc_x`
- `acc_y`
- `acc_z`

No derived accelerometer features are required for first implementation.

Optional future extensions may include:

- vector norm,
- tilt angles,
- low-pass filtered accel,
- gravity/body-frame projections.

### 9.3 Time feature usage

Time itself should **not** be used as a predictor in the baseline models unless explicitly added in a future extension.

### 9.4 Run identifiers

Run identifiers are metadata only and must never be passed as model inputs.

---

## 10. Training and Model Selection Procedure

## 10.1 Data aggregation

For static models (Models 1 to 5):

- concatenate all samples from all training runs into a single training table,
- concatenate all samples from all validation runs into a single validation table,
- preserve run IDs in metadata for later per-run evaluation.

For lagged models:

- first construct lagged samples within each run,
- then concatenate lagged samples across runs.

## 10.2 Hyperparameter search strategy

A simple grid search is sufficient.

For each model family:

1. build training features,
2. fit preprocessing on training only,
3. fit candidate models on training data,
4. evaluate on validation data,
5. select the configuration with the best validation metric,
6. retrain on training data only using the chosen configuration,
7. evaluate once on held-out runs.

### 10.3 Primary model-selection metric

Recommended primary validation metric:

- **overall RMSE**

Secondary metrics:

- MAE
- \( R^2 \)
- Pearson correlation
- bias / mean signed error

### 10.4 Optional final refit

This repository should support, but not require, an optional mode in which the final selected hyperparameters are retrained on `train + val` before final held-out evaluation.

However, for direct comparability with the current LSTM work, the default mode should remain:

- fit on train,
- select on val,
- report on held-out.

---

## 11. Evaluation Requirements

## 11.1 Metrics

At minimum, compute:

- RMSE
- MAE
- mean signed error (bias)
- \( R^2 \)
- Pearson correlation coefficient

Metrics must be computed:

- per run,
- aggregated over all held-out runs.

### Definitions

For predictions \( \hat{\phi}\_i \) and targets \( \phi_i \), \( i=1,\dots,N \):

#### RMSE

\[
\mathrm{RMSE} =
\sqrt{
\frac{1}{N}
\sum\_{i=1}^{N}
(\hat{\phi}\_i - \phi_i)^2
}
\]

#### MAE

\[
\mathrm{MAE} =
\frac{1}{N}
\sum\_{i=1}^{N}
|\hat{\phi}\_i - \phi_i|
\]

#### Bias

\[
\mathrm{Bias} =
\frac{1}{N}
\sum\_{i=1}^{N}
(\hat{\phi}\_i - \phi_i)
\]

#### Coefficient of determination

\[
R^2 =
1 -
\frac{
\sum*{i=1}^{N} (\phi_i - \hat{\phi}\_i)^2
}{
\sum*{i=1}^{N} (\phi_i - \bar{\phi})^2
}
\]

#### Pearson correlation

\[
r =
\frac{
\sum*{i=1}^{N}
(\phi_i - \bar{\phi})(\hat{\phi}\_i - \bar{\hat{\phi}})
}{
\sqrt{
\sum*{i=1}^{N}(\phi*i - \bar{\phi})^2
}
\sqrt{
\sum*{i=1}^{N}(\hat{\phi}\_i - \bar{\hat{\phi}})^2
}
}
\]

---

## 11.2 Evaluation outputs

For each trained model, the repository must export:

### Numeric outputs

- validation summary table,
- held-out summary table,
- per-run held-out metrics,
- overall held-out metrics,
- selected hyperparameters.

### Prediction outputs

For every held-out run:

- timestamp or sample index,
- ground truth angle,
- predicted angle,
- error,
- run ID,
- model name.

Suggested format:

- CSV or Parquet

### Plot outputs

For each held-out run:

- angle prediction trace vs ground truth,
- optional error trace,
- optional scatter plot of predicted vs actual.

For each model:

- held-out parity plot,
- histogram of prediction errors,
- optional coefficient magnitude plot for regression models,
- training history plot for MLP.

---

## 12. Reproducibility Requirements

## 12.1 Random seeds

The repository must support explicit global seeding for:

- NumPy,
- Python random,
- TensorFlow/Keras if MLP uses Keras.

Ridge regression is deterministic given fixed data, but the MLP is not unless seeded.

### 12.2 Configuration-driven runs

All experiment settings must be provided through configuration files, not hard-coded into the training scripts.

This includes:

- data paths,
- run split IDs,
- feature column definitions,
- hyperparameter grids,
- output paths,
- random seed,
- selected metrics.

### 12.3 Versioned outputs

Each experiment run should create a timestamped output directory containing:

- config snapshot,
- selected model parameters,
- metrics,
- plots,
- fitted objects,
- logs.

---

## 13. Repository Structure

Suggested structure:

````text
basic_benchmark_repo/
├── README.md
├── pyproject.toml
├── requirements.txt
├── configs/
│   ├── data.yaml
│   ├── evaluation.yaml
│   ├── benchmark_suite.yaml
│   └── models/
│       ├── pressure_ridge_linear.yaml
│       ├── pressure_ridge_quadratic.yaml
│       ├── pressure_accel_ridge_linear.yaml
│       ├── pressure_accel_ridge_quadratic.yaml
│       ├── pressure_accel_mlp.yaml
│       └── lagged_pressure_accel_ridge.yaml
├── src/
│   ├── data/
│   │   ├── loader.py
│   │   ├── splits.py
│   │   └── schemas.py
│   ├── features/
│   │   ├── static_features.py
│   │   ├── lagged_features.py
│   │   └── preprocess.py
│   ├── models/
│   │   ├── ridge_models.py
│   │   ├── mlp_model.py
│   │   └── registry.py
│   ├── training/
│   │   ├── train_one_model.py
│   │   ├── tune_model.py
│   │   └── run_suite.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── predict.py
│   │   └── summarise.py
│   ├── plotting/
│   │   ├── traces.py
│   │   ├── parity.py
│   │   └── errors.py
│   └── utils/
│       ├── io.py
│       ├── config.py
│       ├── logging.py
│       └── seed.py
├── outputs/
└── tests/
    ├── test_loader.py
    ├── test_feature_building.py
    ├── test_no_leakage.py
    ├── test_metrics.py
    └── test_training_smoke.py

## 14. Script-Level Functional Requirements

### 14.1 Data loading script

Responsibilities:
- load all run files,
- validate expected columns,
- attach split labels,
- return run-wise data structures.

Recommended interface:

```python
runs = load_runs(config)
````

Where `runs` is a mapping:

```python
{
    "run_id_001": dataframe,
    "run_id_002": dataframe,
    ...
}
```

---

### 14.2 Feature construction modules

#### Static feature builder

Builds:

- `X_train`, `y_train`
- `X_val`, `y_val`
- `X_test`, `y_test`

plus metadata arrays for run IDs and timestamps.

#### Lagged feature builder

Builds lagged samples independently for each run.

Outputs must preserve:

- sample-to-run mapping,
- trimmed timestamps after lagging,
- target alignment.

---

### 14.3 Training script

A model training script must:

1. read config,
2. load data,
3. build features,
4. fit preprocessing on training data,
5. train candidate configurations,
6. select best validation configuration,
7. refit chosen configuration on training data,
8. export artifacts,
9. evaluate on held-out data,
10. save outputs.

---

### 14.4 Suite runner

A suite runner must execute the full benchmark suite end-to-end.

It should:

- iterate over the six benchmark model configs,
- launch training/evaluation,
- collect summary tables across models,
- generate a cross-model comparison CSV.

Expected output:

- one summary table comparing all six models on validation,
- one summary table comparing all six models on held-out data.

---

## 15. Detailed Model Implementation Notes

### 15.1 Common implementation abstraction

A clean design is to define a common interface for all models:

```python
class BenchmarkModel:
    def fit(self, X_train, y_train, X_val=None, y_val=None): ...
    def predict(self, X): ...
    def save(self, output_dir): ...
    def get_params(self): ...
```

This allows the suite runner to treat all models uniformly.

### 15.2 Scikit-learn pipelines

For regression baselines, prefer using scikit-learn `Pipeline` objects where practical:

- scaler,
- polynomial expansion if relevant,
- ridge estimator.

This reduces implementation bugs and keeps fitting logic consistent.

### 15.3 MLP wrapper

The MLP should expose a scikit-learn-like interface even if implemented in Keras.

The wrapper must:

- fit on training data,
- evaluate on validation data,
- store training history,
- provide `predict()` returning NumPy arrays.

---

## 16. Logging and Reporting Requirements

The repository must log:

- selected features,
- number of samples per split,
- lag settings,
- scaler statistics summary,
- validation results per configuration,
- final selected hyperparameters,
- held-out results.

Each experiment should produce:

- a machine-readable summary file, e.g. JSON,
- a human-readable summary file, e.g. Markdown or TXT.

---

## 17. Unit and Integration Testing Requirements

At minimum, implement the following tests:

### 17.1 Data loading test

Verify:

- required columns exist,
- split mapping is correct,
- run IDs are unique.

### 17.2 No-leakage test

Verify:

- scalers are fit only on training data,
- lag windows never cross run boundaries,
- held-out data is never used in training.

### 17.3 Feature construction test

Verify:

- expected feature matrix shapes for each model,
- polynomial expansion output shape,
- lagged feature output shape,
- target alignment is correct.

### 17.4 Metric test

Verify:

- RMSE / MAE / bias / `R^2` implementations match known values on synthetic data.

### 17.5 Training smoke test

Run a tiny synthetic example to ensure each model can:

- fit,
- predict,
- save outputs.

---

## 18. Initial Hyperparameter Defaults

These are recommended starting defaults for first implementation.

### 18.1 Ridge models

- `alpha_grid = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]`

### 18.2 Polynomial models

- `degree = 2`

### 18.3 MLP

- hidden units: `[8, 16, 32]`
- learning rates: `[1e-4, 3e-4, 1e-3]`
- L2: `[0.0, 1e-6, 1e-5, 1e-4]`
- batch size: `[128, 256, 512]`
- max epochs: `200`
- early stopping patience: `15`

### 18.4 Lagged ridge

- lag lengths: `[3, 5, 10, 20]`
- alpha grid same as ridge models

These ranges are intentionally modest to keep the baseline suite lightweight.

---

## 19. Recommended Development Order

Implementation should proceed in the following order:

1. data loader + split handling
2. metrics module
3. static feature builder
4. pressure-only linear ridge
5. pressure-only quadratic ridge
6. pressure + accel linear ridge
7. pressure + accel quadratic ridge
8. lagged feature builder
9. lagged ridge
10. minimal MLP
11. plotting and summary tables
12. test suite

This order ensures that the most useful and simplest baselines become available first.

---

## 20. Deliverables

The completed repository must be able to produce the following deliverables:

### 20.1 Per-model deliverables

- trained model file,
- preprocessing objects,
- validation metrics,
- held-out metrics,
- per-run predictions,
- plots.

### 20.2 Benchmark-suite deliverables

- comparison table across all six baselines,
- selected hyperparameter table,
- held-out metrics summary table,
- directory of held-out prediction plots,
- directory of per-run prediction CSV files.

### 20.3 Thesis/paper-ready outputs

The repository should make it easy to generate:

- a main benchmark comparison table,
- representative prediction trace figures,
- per-run error summary tables.

---

## 21. Non-Goals

The first version of this repository does **not** need to include:

- raw data decoding,
- OptiTrack/IMU synchronisation,
- advanced feature engineering,
- hysteresis-specific physics models,
- full system identification toolbox methods,
- Kalman filtering,
- sequence models beyond the lagged ridge baseline,
- nested cross-validation.

These may be added later if needed, but they are outside the current scope.

---

## 22. Summary

This repository will implement a simple, rigorous, and fair benchmark suite for SPA bending angle estimation using the same pre-processed data splits as the LSTM experiments.

The six benchmark models cover:

- the simplest plausible pressure-only map,
- nonlinear static regression,
- multimodal static regression with pressure and accelerometer data,
- minimal feedforward neural estimation,
- a finite-memory dynamic regression baseline.

Together, these baselines will establish a stronger experimental position for the LSTM by showing whether its gains arise from:

- nonlinear modelling capacity,
- multimodal sensing,
- temporal memory,
- or some combination of these.

The implementation must prioritise:

- run-level split integrity,
- leakage-free preprocessing,
- reproducibility,
- simple configuration-driven experimentation,
- easy export of benchmark tables and figures.
