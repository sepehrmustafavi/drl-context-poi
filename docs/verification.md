# Verification record

Date: 2026-10-19

## Executed

- `python -m pytest -q`: **13 passed**, 52.77 seconds in the final recorded run.
- `python -m ruff check src tests scripts --select E9,F`: passed.
- `python -m ruff format --check src tests scripts`: all 24 Python files already formatted.
- Python bytecode compilation of source and scripts passed.
- Package installed in editable mode; public CLI help and workflow exercised.

Test environment: Python 3.12.14; PyTorch 2.14.1+cpu; NumPy 2.3.5; pandas 2.2.3; PyYAML 6.0.3; pytest 9.1.1; Ruff 0.17.1. No CUDA GPU was available in this verification environment.

## Covered behaviors

The end-to-end fixture trains independent ensemble members, generates policy/validation/test subset caches, trains all eight learned controller variants, evaluates them and checks initial raw-vs-cache decisions/ranks. It simulates interruption after a committed controller epoch and confirms identical CPU continuation parameters compared with uninterrupted training. It exercises cost profiling, raw latency, external score coverage/provenance validation and out-of-catalog target handling.

The public CLI test covers conversion, fitting, profiling, caching, controller fitting, cached evaluation, raw inference, fixed-subset timing and a bounded HPO configuration. Further tests cover future-label mutation invariance, strict historical graph cutoffs, lazy retrieval, empty-context fusion, M=1 disagreement, GRU/SASRec-style forward/backward, masked Double DQN/DQN targets, explicit terminal rewards, zero budget, exact ranks/ties, clustered statistics, experiment DAG/seed coverage, a CPU worker and single-run reporting.

## Not established by these tests

No NYC/Tokyo/Gowalla benchmark accuracy, real GPU duration, CUDA peak memory, or external baseline reproduction has been measured here. The fixtures are synthetic software tests, not research evidence. CUDA precision/runtime agreement, dataset manifests and all publication experiments must be checked on the actual final software/hardware. CPU tests do not guarantee identical floating-point outputs across GPU types or PyTorch versions.
