# Verification record

Date: 2026-10-21. Version 0.2.1 re-audited from repository commit `9dd7bb9` (initial implementation derived from `7c7df21`).

## Executed software checks

- Full test suite, including actual pinned upstream-class tests: **34 passed, 3 skipped**. Synthetic fixtures only. The three explicitly skipped tests require CUDA: two FP16 attention cases and checkpoint resume.
- `ruff check src tests scripts --select E9,F`: passed.
- Python source formatting and bytecode compilation checked.
- Public CLI conversion/fitting/profiling/caching/training/evaluation/inference/HPO paths exercised.
- Full-data calibration path exercised on a synthetic fixture, including actual ensemble validation selection, controller selection, entropy threshold and frozen-cost provenance. It does not generate a test cache.
- Four upstream repositories freshly fetched with the new sparse checkout helper; commit/model-byte verification succeeded. Their real model classes completed CPU training, backward propagation, deterministic evaluation, unknown-user handling, label-independent inference and complete OOC-inclusive test coverage.
- Exact pinned Gowalla-CA archive converted and audited in full: metadata in `gowalla-ca-audit.json`.
- Notebook Python cells compile; notebook JSON and CI YAML parse.
- Fresh full-plan generation: **517 jobs**, including 94 controller fits, 36 external baseline fits and 110 isolated timing jobs.
- FP16 masked attention forward/backward reproduced and fixed on CPU autocast; this exercises FP16 arithmetic.
- Failed predictor resume preserves existing metadata/checkpoints; mismatched source order, relabeled predictor weights and changed frozen protocol files are rejected.
- iPCM executes with noncontiguous region labels; the scheduler tolerates another worker releasing a lock during claim inspection.

Environment: macOS arm64 CPU, Python 3.12.14, PyTorch 2.14.1, NumPy 2.5.4, pandas 3.0.6, PyYAML 6.0.3, pytest 9.1.1, Ruff 0.17.1, SciPy 1.18.1, scikit-learn 1.9.1, torch-geometric 2.8.1. CUDA was unavailable. Upstream/dependency warnings include sparse-invariant-check defaults, Transformer nested-tensor eligibility, deprecated torch.jit.script, and the sandboxed macOS physical-core query; the checks did not fail.

## Important invariants exercised

Raw/cached decisions and ranks agree on tested examples. Cached request prediction matches full recomputation while encoding the base once and each newly acquired source once. Tests cover strict historical graph cutoffs, target mutation invariance, missing category IDs, exact UTC units under pandas 3, indexed periodic counts, all feasible subsets, budgets including zero, explicit terminal NLL, masked Double DQN targets and deterministic CPU resume. DAG checks cover all datasets/seeds, external baseline jobs, explicit sensitivity values, architecture/source ablations and a separate timing phase. Statistical tests cover matched request coverage, user-cluster resampling, sample SD and Holm correction.

## Execution protocol

CPU execution does not guarantee bitwise agreement across GPU/software versions. The adapters use the common chronological split/request and cold-start conventions documented in `external.md`. Full-scale DCHL still has quadratic contrastive compute despite reduced activation storage.

Run calibration, final training, CUDA agreement checks and idle-reference-GPU timing in the intended environment before reporting article results. Source/config/data identities in the generated plan and result artifacts support reproducibility.
