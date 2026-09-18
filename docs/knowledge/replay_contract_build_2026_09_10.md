# Research replay contract build — September 10, 2026

## Delivered

The approved initial research interfaces are implemented, reviewed and exercised. **63 research tests pass**: the previous 35 plus 13 clock, eight selected-feature and seven report tests. This is not a full-repository test result or a profitable-strategy certificate.

- `scripts/research/replay_clock.py`: typed observation identity, explicit event/availability/receipt/expiry times, UTC candle grid and OHLCV checks, completed/developing context, chronological processor replay, prefix-bound checkpoint validation and deterministic full-prehistory restart.
- `scripts/research/replay_features.py`: exact selected source helpers, unmodified feature/gate reference, named finite-FVG/OI-liquidity candidate, dependency invalidation and actual-source threshold probes.
- `scripts/research/replay_contract_report.py`: version-aware paired historical report, all 17 archetypes, sparse JSON key preservation, ambiguous/incompatible pair exclusion, hashes and explicit certification blockers.

No engine, runner, archetype configuration or original input data was edited. No orders, deployment, optimizer, full-store overwrite or public push. Branch stays local: `quant/archetype-evidence-audit`.

## What the clock guarantees—and what it does not

Observation visibility uses the later of recorded `available_at` and `received_at`, when receipt time is known. An unknown availability time rejects the contract; it is never inferred from a candle label. Source, formula version, units and instrument changes are rejected within a feature stream. Validity/expiry problems reach the raw reference but taint certification, including retained warmup state.

Candles are labeled by open time and processed at close. Higher-timeframe context uses only completed constituent candles, with completed, developing and incomplete buckets distinguished. Prefix checks run on fixtures and on recovered June 19 minute bars at 00:59, 01:00 and 04:00 UTC. At 00:59, an hourly bucket has 59 minute constituents and is still developing; at 01:00 it is completed. The raw minute schema uses `vol`; the diagnostic explicitly renamed it to `volume` in memory and did not alter the source file.

Restart reconstructs processor state by replaying the identical supplied prehistory through a fresh factory, discarding already-emitted rows. A changed prefix or processor contract identity rejects the checkpoint. It does not attempt partial serialization of opaque detector memory. Tests cover multiple restart cuts for a stateful accumulator and the actual selected-feature processor.

**Scope limit:** `replay(...).certified` describes the supplied observation-clock/processor contract only. It does not establish that every required strategy input was supplied, that a source assertion is externally verified, or that the entire LiveFeatureComputer/state/book was reproduced. `SelectedFeatureProcessor` explicitly lacks full funding and detector reconstruction. The historical report cannot issue an end-to-end certificate.

## Actual-data exercise

V23 store: 74,436 rows. Archived live JSONL: 533 rows. Seven rows at three duplicated timestamps are excluded, leaving **526 paired timestamps** from June 18 23:00 through July 10 23:00 UTC. Known instrument/venue mismatches would be excluded; missing identity metadata remains a blocker instead of an assumption of compatibility.

| Selected feature diagnostic | Historical store | Archived live |
|---|---:|---:|
| `any_fvg` predicate changes under finite-value correction | 352 | 0 |
| Liquidity values changed using saved OI/current helper | 526 / 526 | 526 / 526 |

The 352 store changes are reference-versus-candidate within the store. They are distinct from the previous 354 store-versus-live FVG disagreements. None is a trade count or an estimate of profit improvement.

The candidate recomputes liquidity and its listed fusion descendants. It explicitly invalidates unimplemented OI/divergence and regime-context descendants, including `macro_regime`. Because missing-input policies can skip gates, a predicate becoming permissive after invalidation is **not a useful strategy improvement**. Candidate gates remain diagnostic and uncertified. Funding history/Z reconstruction is also explicitly uncertified.

The actual-source threshold matrix still shows **48/192 acceptance mismatches**. The gate-to-boundary probe records gate status/penalty, hard/soft mode, supplied pre-gate and post-gate score, threshold, bypass/enforcement controls and acceptance. It holds the supplied pre-gate score fixed and regime adjustment inactive; it is not a full archetype fusion, structural, cooldown or allocation replay.

## Why historical certification remains false

The final CLI exits **2** with `--require-certification` and lists:

- Ambiguous duplicate timestamps.
- Missing `available_at` history.
- Missing instrument/venue provenance and source versions.
- Historical internal state not reconstructed.
- Full pipeline not replayed.
- Known threshold-boundary mismatch.

The report intentionally cannot promote this paired-input inventory into an end-to-end certificate. The absence of sufficient provenance is not repaired with assumed timestamps, guessed source identity, or retroactive formula changes.

## Review corrections

Independent read-only review found seven edge cases. Each was reproduced with a failing regression before correction:

1. Invalid warmup history could be omitted from certification issues.
2. Mutable processor outputs could rewrite prior emitted rows.
3. A display NaN marker could collide with a literal dictionary in checkpoint hashing; hashing now uses separate type-tagged canonical encoding.
4. Candidate invalidation missed OI divergence fields and the regime alias.
5. Finite NumPy scalars were incorrectly treated as absent.
6. Rectangularizing sparse JSON inserted artificial NaNs into the reference.
7. Known instrument/venue mismatches were not excluded from timestamp pairing.

Reviewer confirmed fixes. Original sparse dictionaries now reach gate evaluation unchanged. No production fixes were bundled into this work.

## Reproduction

```sh
python3 -m unittest discover -s tests/research -v
python3 scripts/research/replay_contract_report.py --store ../data/features_mtf/BTC_1H_FEATURES_V23_PARITY_2018_2026.parquet --live-jsonl ../results/coinbase_paper/live_features/2026-06.jsonl ../results/coinbase_paper/live_features/2026-07.jsonl --config configs/champion_paper.json --out results/research_validation_2026_09_10/contract --require-certification
# Expected: report written, exit 2, certified=false.
git diff e1c422e HEAD --exit-code -- engine bin configs
```

The ignored local output is `results/research_validation_2026_09_10/contract/replay_contract_report.json`. Its config, source and raw input SHA-256 values were verified against disk after the final run.

Implementation commits: `3856473`, `9b46453`, `faf59fa`, `101fc25`, `4ebba88`, `654c9a2`; plan `9e06ee1`.

## Remaining work

Connect the full feature/state path to this clock under explicit source/availability/formula contracts; preserve the production-faithful reference alongside defect-corrected candidates. Compare full signals and book sequencing before interpreting P&L changes. Where historical release/state evidence is unavailable, label any historical reconstruction provisional and reserve forward shadow evidence for validation.

Hourly and minute BTC remain equally eligible. No novel combined archetype, tuned gate, strategy graduation or profitability claim is delivered by this infrastructure layer.
