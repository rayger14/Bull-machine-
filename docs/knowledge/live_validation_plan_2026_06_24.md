# Live Validation Plan + Latent Gate Finding (2026-06-24)

## Decision (user, 2026-06-24)
Live engine bleeding: $72,034 equity, **−28%, PF 0.67, 66% WR, 175 trades**. The
66% WR with PF 0.67 means direction is fine — losers are bigger than winners
(long-only dip-buyer in a bear; stops dominate). Two jobs: stop the bleed (config
change) and start winning (deploy the validated edge, currently locked out).

**Chosen path: keep collecting + validate offline first** (NOT change live yet).
Standing orders stay intact. **Normal 2% sizing** when we eventually deploy.
Rationale: prove wick_trap_v14rq fires + performs on the real live feature stream
before any production change.

## Validation harness (built, working)
`scripts/champion/validate_live_offline.py` — replays wick_trap_v14rq against
`results/coinbase_paper/live_features/*.jsonl` (V14 tail for warmup), via the
real backtester + champion config. Mirrors the validated battery conditions.

**Status 2026-06-24**: 0 wick_trap trades on 5 days / 141 live bars — INSUFFICIENT.
Expected (wick_trap fires a few times/month). Verdict needs ~30 setups ≈ several
weeks of logs. Harness is ready; rerun as data accumulates.

### Promotion gate (pre-registered)
Deploy wick_trap_v14rq to live paper only when offline replay shows, on
accumulated live features: ≥30 trades, PF ≥ 1.3, signal timing matching backtest
expectation, no execution surprises. Then live paper at 2% sizing, then real.

## Latent finding: wick_trap `instability` gate is INERT
wick_trap.yaml hard_gate reads feature `instability` (max 0.45), but the engine
produces `instability_score` — **neither V14 nor the live logs contain
`instability`**. The gate has `nan_policy: skip`, so it has been silently SKIPPED
in every backtest, including the holdout PF 1.43. Implications:
- The validated wick_trap_v14rq results are WITH this gate inert — consistent and
  fine, but the "low-instability" noise filter never actually applied.
- DANGER: if anyone later "fixes" the column name, wick_trap behavior changes and
  PF 1.43 no longer holds. Treat instability-gate activation as a NEW hypothesis
  requiring its own WFO, not a bugfix.
- Parity-check enhancement worth adding: flag gate features ENTIRELY ABSENT from
  the store (name mismatch), not just unreachable thresholds — this class would
  have been caught earlier.

## Cross-references
[[v14_champion_hunt_2026_06_11]] · [[sizing_studies_verdict_2026_06_16]] ·
[[industry_study_backtest_live_parity_2026_06_11]]
