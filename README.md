# SPA Basic Estimators

This repository is intentionally reset to a minimal Phase 0 state.

The goal is to rebuild the benchmark suite from `tech_spec.md` in a clear, maintainable, phase-by-phase way rather than carrying forward a large scaffold that is hard to reason about.

## Current Status

What exists right now:

- the technical specification in `tech_spec.md`,
- the implementation roadmap in `PHASE_PLAN.md`,
- minimal packaging,
- the target directory skeleton for the rebuild.

What does not exist right now:

- estimator implementations,
- data loader logic,
- training pipeline logic,
- tests beyond the future placeholders.

## Intended Structure

```text
README.md
tech_spec.md
PHASE_PLAN.md
pyproject.toml
requirements.txt
configs/
  benchmark_suite.yaml
  data.yaml
  evaluation.yaml
  models/
src/
  spa_basic_estimators/
    data/
    evaluation/
    estimators/
    training/
    utils/
tests/
outputs/
```

## Build Approach

The rebuild will follow `PHASE_PLAN.md`.

Key rules:

1. Each of the six estimators gets its own explicit implementation phase.
2. Shared code stays small and limited to clearly common concerns.
3. Run-level split integrity and leakage prevention are built in from the beginning.
4. We finish one understandable slice at a time.

## Next Step

The next implementation step is Phase 1:

- define the HDF5 data contract,
- implement the run loader,
- implement split validation,
- add basic loader tests.

