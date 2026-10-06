# Wyckoff recognition exam implementation plan

> Execute inline with superpowers:executing-plans. The user delegated decisions
> to the quant and requested uninterrupted completion. No paid market assessments.

**Goal:** source-adjudicated 12-case semantic exam of the actual native path.
**Spec:** `docs/superpowers/specs/2026-10-05-wyckoff-recognition-exam-design.md`.
**Architecture:** deterministic input packet, reviewed seal, actual-source observer,
conditional sizing probe, and an immutable offline receipt. No strategy clone.
**Stack:** existing Python, pandas/numpy, pytest, hashlib, AST, JSON.

## Global constraints

Existing quant branch/dirty tree preserved by user preference. No detector/config
edits, deployment, threshold search, M2 activation, economics, commit/push/PR.
Synthetic cases are conceptual examples, not historical X charts. Inaccessible
posts cannot establish grading facts. Expectations frozen before detector runs.
Engineering pass is distinct from semantic failure. One sequential 600s/100MiB
exam; no result-dependent replacement. Preserve progress ledger (no commits).

### Task 1: Fixture packet and independent adjudication

**Files:** create `scripts/research/wyckoff_recognition_cases.py`,
`tests/research/test_wyckoff_recognition_cases.py`; generate new local
`results/wyckoff_recognition_2026_10_05/input_v1/`.
**Produces:** JSON packet schema `wyckoff-recognition-input-v1` with source ledger,
12 IDs/categories/directions, exact start-stamped OHLCV, semantic landmarks,
parent boundaries, checkpoint expectations and uncertainty; canonical SHA256.
No detector imports or output-derived decisions in the fixture module.

1. Write failing tests for exact 4/4/4 balance, deterministic UTC candles, positive
   volumes/valid OHLC, chronological checkpoints, mirrors, malformed inputs and
   no overwrite. Run `python3 -m pytest -q tests/research/test_wyckoff_recognition_cases.py`.
   Expected: new API unavailable assertion fails before implementation.
2. Implement fixed warmup and price/volume anchor interpolation, with explicit
   event candles and named landmarks. No parameter search. Validate each case
   before serialization, fail closed on extra/invalid input fields.
3. Run same tests. Expected: all pass; no detector invoked. Export packet.
4. Fresh source-only quant receives this packet and cited readable sources only;
   no PROJECT/MEMORY, engine source/results. Review actual candles, chronology,
   qualitative evidence and permissible interpretations. Correct only genuine
   pre-freeze input/annotation defects. Save exact reviewed packet hash, 12
   adjudications and signed identity/approval in `review_v1/approval.json`.
   Expected: 12 adequate cards approved, or explicitly unresolved cards not scored.

### Task 2: Actual-source trace and fail-closed runner

**Files:** create `scripts/research/wyckoff_recognition_exam.py`,
`tests/research/test_wyckoff_recognition_exam.py`.
**Consumes:** Task 1 packet/hash + reviewer approval.
**Produces:** per-prefix raw/validated event records and states, feature status,
long/short Wyckoff scores, conditional phase-sizing response, sampled causal
witnesses, code/config bindings and three-layer verdict ledger.

1. Tests first: malformed/missing approval and drift rejection before computation;
   actual-source all-flat non-exam smoke; raw/validated trace presence; directional
   scorer; AST consumer long C_accum 1.25 vs short/nonqualifying 1.0; ambiguous AST
   and hash failure; future perturbation equality; no overwrite/resource bounds.
   Run `python3 -m pytest -q tests/research/test_wyckoff_recognition_exam.py`.
   Expected: unavailable API assertions fail before implementation.
2. Use LiveFeatureProcessor import boundary and deny_network, feed OHLCV prefixes
   into actual `_wyckoff_features`; wrap state validator/process_bar observably
   without modifying return data. Capture active TF config and parent lifecycle.
   Call actual directional scorer. Source/hash-bind unique phase-boost AST branch
   in V11ShadowRunner.process_bar; inert explicit inputs, no runner construction.
3. Validate input/review/source seals before executing. Deadline includes all
   preflight; byte budget applies to every artifact. Mark failure with bounded
   error receipt, never publish a partial success. Check identical prefix and
   a changed unobserved tail at fixed checkpoints, using independent instances.
4. Separate expected relationships, recognized native events/phase and consumer
   response. Missing positive milestones are misses, not passes due to M2-off.
   Preserve provisional/ambiguous status and unsupported nested lineage. No P&L.
5. Run both new test files plus existing evidence-integrity regressions.
   Expected: harness tests green without running the 12 frozen exam cases.

### Task 3: Independent software review, one exam and handoff

**Files:** local `exam_v1/`; create
`docs/knowledge/wyckoff_recognition_exam_2026_10_05.md`; update PROJECT/MEMORY.
**Consumes:** Tasks 1/2 and approved source packet; **Produces:** verified report.

1. Fresh most-capable software reviewer checks actual new files and scoped dirty
   diff (not an empty HEAD diff), spec/plan/ledger. Reproduce Important findings
   test-first and fix harness only; never tune semantic inputs on outcomes.
2. Freeze packet, review and runtime/source/config hashes together. Run one bounded
   exam with immutable output; verify counts, every artifact hash, source drift,
   and semantic discrepancies. Quant interprets results; no automatic strategy fix.
3. Run selected regressions, then bare `python3 -m pytest -q` to report actual
   broad-suite status, including known collection blockers. Expected: focused
   suite green, broad failures explicitly reported (not misrepresented as green).
4. Update continuity with exact results, unresolved concepts, X access limitations,
   local dependencies, no running jobs and the single most useful next repair.
   No commit/push. User receives digestible results and links, not another pause.

## Review focus

Examine causal prefix truncation, wrapper observational equivalence, native daily
absence, incomplete higher-timeframe candles, timeout coverage during imports and
preflight, NaN/JSON handling, approval/hash drift and path traversal, stale results,
source-only independence, configured-M2 versus semantic recognition distinction,
scoring/phase sizing that survive rejected evidence, AST uniqueness and isolation.
Tests cannot prove exhaustive semantic fidelity; report that boundary explicitly.
