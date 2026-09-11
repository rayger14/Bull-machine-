# Research validation results — September 10, 2026

## Decision

The approved research-only validation utilities are implemented. They do **not** certify a profitable archetype. The recovered minute strategy's positive result does not survive chronological event selection. The local live and standalone backtest threshold boundaries also disagree. Feature availability differs materially between the historical store and archived live inputs.

Keep all 17 production archetypes and settings unchanged. Do not optimize on the invalid minute population or treat the standalone backtester as a faithful live baseline yet. This is a measurement milestone, not strategy graduation or deployment approval.

This extends [the trader-to-archetype audit](archetype_translation_audit_2026_09_09.md), which contains the 17-archetype translation table, recovered data inventory, position-level live accounting and original causality reproducer. No new live-history claim is made here.

## 1. Minute BTC: chronological replay rejects the recovered headline

Input: recovered Binance BTCUSDT futures minute OHLC, January 1, 2021 through August 31, 2026 UTC, 2,979,360 complete minute bars. This is not the separate CME futures dataset.

Frozen setup: confirmed 31-bar equal-low pivots; prior touch at least 30 minutes old within 24 hours and 0.1%; sweep 0.02% below level; reclaim within 30 minutes. Candidates are ordered by the time their reclaim is observable, then oldest pivot. The 60-minute sweep-spacing condition cannot reserve cooldown from a future event.

Frozen execution: $50,000 notional per position, stop 0.15% below the sweep low, four-hour lockout from entry even after an early stop, 12 bps round-trip cost ($60 per completed position), no compounding. Default next-open simulation checks the entry minute for stops, fills adverse stop gaps at the open, and exits at the four-hour deadline **open**, before any subsequent entry at that open. Close mode instead enters/exits at closes and is an execution diagnostic only.

| Event population / execution | Events | Completed positions | Net P&L | PF | Win rate |
|---|---:|---:|---:|---:|---:|
| Legacy cached events / same close simulator | 7,952 | 5,032 | $204,495.52 | 1.2955 | 39.23% |
| Chronological events / same close simulator | 15,910 | 7,708 | −$496,914.87 | 0.5755 | 22.50% |
| Chronological events / next-open baseline | 15,910 | 7,708 | −$496,873.92 | 0.5755 | 22.50% |

The first two rows use **identical accounting and execution code**: changing close-fill timing does not explain the sign reversal. The old cached population is known to depend on future history: appending bars deletes previously observable events. The chronological population passes all 68 monthly prefix comparisons, plus the January 15 counterexample check; April 1 also passes explicitly. Prefix checks require exact event dictionaries, not just equal counts.

There are two explicit boundary corrections besides candidate ordering: permit a level with exactly two pivots (legacy loop began at the third), and include the final available bar (legacy excluded it). These follow the stated two-touch setup and prefix invariance; no parameters were searched. This is a causal interpretation of the recovered setup, not proof that every possible causal implementation loses.

The legacy comparison leaves its final unresolved position open, explaining 5,032 completed versus the old headline's 5,033 forced-completed trades. Its final marked P&L is $204,527.72, including that open position and its entry fee. Neither corrected chronological run has an open position at the end.

Next-open yearly results, grouped by exit year:

| Year | Completed | Net P&L |
|---|---:|---:|
| 2021 | 1,286 | −$107,173.69 |
| 2022 | 1,391 | −$87,637.92 |
| 2023 | 1,400 | −$81,128.48 |
| 2024 | 1,352 | −$83,873.48 |
| 2025 | 1,364 | −$74,604.49 |
| 2026 through August | 915 | −$62,455.86 |

Average initial stop risk is $151.33. Minute-close marked dollar drawdown is $499,910.06. Total assumed fees are $462,480; gross P&L is still negative at −$34,393.92. Funding and market impact are not modeled.

**These totals are not funded-account returns or actual losses.** Starting equity is unspecified; $50,000 is trade notional, not account capital. The simulator does not constrain margin, stop at insolvency, simulate liquidations, or establish obtainable fills. Minute-close drawdown omits intraminute excursions. This historical sample was already researched and is not untouched out-of-sample evidence.

## 2. All 17 archetypes: inputs and gate behavior

The tool resolves champion YAMLs and main-config gate mode/value overrides, then invokes the actual `ArchetypeInstance._evaluate_gates` for each gate/input row. It emits 600 source/archetype/gate/year profiles, including the roster's no-gate entries. Every populated gate profile's status counts reconcile to its input row count.

Sources: V23 historical store, 74,436 hourly rows from March 1, 2018 to August 30, 2026; archived live JSONL, 533 rows from June 18 to July 10, 2026, with four duplicate timestamps preserved and explicitly reported. These archives do not establish today's server state or full live-period coverage.

| Input and rule | Historical store | Archived live snapshots | Consequence at the individual gate |
|---|---|---|---|
| Funding divergence: `funding_Z <= -0.5` | Zero on all 74,436 rows; all fail | 144 pass, 389 fail | Historical zero is not missing and does not trigger `nan_policy: skip` |
| Long squeeze: `funding_Z >= 0.5` | Zero on all 74,436 rows; all fail | 229 pass, 304 fail | Same named feature behaves materially differently across sources |
| Failed continuation / volume fade: `effort_result_ratio` | Missing on all 14,535 rows in 2025–2026 | Absent on all 533 rows | Configured skip policy makes each gate skip on these rows |
| Long squeeze: `vol_shock` | Column absent; skips all 74,436 rows | Absent; skips all 533 rows | This rule cannot discriminate inputs in these sources |

Synthetic regression also confirms a missing/NaN RSI can be defaulted to zero by the production derived function and **pass** `derived:rsi_extreme_65`. The audit flags dependency absence even when the evaluator returns a value. This fixture is not a claim that every live RSI is missing.

These are individual predicate outcomes across all bars, **not** trade rejection counts. Soft mode can penalize rather than reject; structural checks, bypass enforcement, thresholds and deduplication operate elsewhere. Derived dependency discovery conservatively lists every literal `dict.get` input, including fallback alternatives: a missing alternative is not by itself proof of a defective value. Availability also does not establish point-in-time correctness.

## 3. Live/backtest decision boundary: parity fails

The probe executes the actual unmodified AST threshold loops from local source, without constructing runners or invoking network/order adapters:

- `bin/live/v11_shadow_runner.py`, lines 1121–1227.
- `bin/backtest_v11_standalone.py`, lines 899–956.

Of 192 controlled cases, **48 disagree** on acceptance. `--require-parity` correctly exits with status 2. This is an expected diagnostic failure; passing regression tests means the discrepancy is reproduced, not repaired.

| Controlled boundary input | Live | Standalone backtest |
|---|---|---|
| Below threshold, gates pass, global bypass on, per-archetype bypass off | Accept | Reject |
| Below threshold, gates pass, global bypass off, per-archetype bypass on | Reject | Accept |
| Threshold equality, gates pass, no bypass | Accept | Accept |

The live loop has a global bypass branch; the standalone loop instead uses a per-archetype bypass set at this boundary. No attempt was made to decide which policy to change or to repair production.

This is **not full end-to-end parity**. The 192 fixtures are a finite branch matrix, not a mismatch rate in real trades; some may be unreachable after upstream checks. Feature generation, structural eligibility, cooldown, regime adjustments, signal selection, sizing, execution and exits remain outside this probe. No claim is made that the running server has these exact local source hashes.

## Verification and reproducibility

Research suite: **35 tests pass** (9 accounting, 11 minute, 8 gate, 7 boundary). Independent read-only review caught a next-open deadline collision in the first research simulator; the corrected implementation exits at deadline open and has a failing-then-passing regression. Review and main-agent output reconciliation also exposed wholly missing-column rows disappearing from gate aggregation; the corrected tool preserves every timestamp and empty-source policy metadata. These were research-tool corrections, not production edits.

Commits: plan `589d357`; minute `3e0c9a6` and phase correction `4ce121b`; gates `09b1f3e` and row-preservation correction `abca69e`; boundary `289c9be`. Local results remain ignored under `results/research_validation_2026_09_10/`. No public push or production/configuration change.

Run from the nested checkout:

```sh
python3 -m unittest discover -s tests/research -v
python3 scripts/research/minute_sweep_validation.py --bars /private/tmp/claude-501/-Users-rayghandchi-Bull-Machine-Bull-machine-/833eefef-c5b5-45bb-af8d-fd9afb9e129c/scratchpad/btc_1m_2021_2026.parquet --out results/research_validation_2026_09_10/minute_next_open --check-prefixes
python3 scripts/research/minute_sweep_validation.py --bars /private/tmp/claude-501/-Users-rayghandchi-Bull-Machine-Bull-machine-/833eefef-c5b5-45bb-af8d-fd9afb9e129c/scratchpad/btc_1m_2021_2026.parquet --out results/research_validation_2026_09_10/minute_close_diagnostic --entry-mode close
python3 scripts/research/gate_observability.py --config configs/champion_paper.json --store ../data/features_mtf/BTC_1H_FEATURES_V23_PARITY_2018_2026.parquet --live-jsonl ../results/coinbase_paper/live_features/2026-06.jsonl ../results/coinbase_paper/live_features/2026-07.jsonl --out results/research_validation_2026_09_10/gates
python3 scripts/research/decision_boundary_parity.py --out results/research_validation_2026_09_10/parity --require-parity
# Last command must exit 2 while the documented boundary mismatch exists.
git diff 3b526a4 HEAD --exit-code -- engine bin configs
```

Same-simulator cached comparison: load the scratchpad `scalper_events_idx.parquet`, map each row to `reclaim_idx=int(ridx), sweep_low=float(sweep_low)`, and invoke `simulate_events(..., entry_mode='close')`. For the simulator's event-schema validation only, use `pivot_idx=0, sweep_idx=int(ridx)` because the cache lacks original pivot/sweep indices; these placeholders are **not causality evidence**. The saved comparison is `legacy_comparison.json`. The prior report reproduces original selection from bars independently.

Input SHA-256:

- Minute bars: `5b8a4533f70b8ccd0dc533984469e6886ec73ce315d48f150c667ccb97e84035`.
- Legacy event cache: `8020ed91431648f4e8ed5822e45088b567702439c5f8ca9e830554dae718623b`.
- Hourly store: `0e8604d0dacf7435e42e9759fe64aca15bfa1e3d66f14c1d21c142f41d1ab61f`.

CLI JSONs include source/config/input hashes and assumptions. The recovered minute files are in a temporary directory; their existence should be checked before later reproduction.

## What this means for the all-seeing-eye objective

The objective remains nested timeframe context, separately defined entry mechanisms, and coherent invalidation/management—not merging scores indiscriminately. Hourly and minute BTC remain equally eligible for research; neither gets a reliability exemption.

Next research work should establish point-in-time feature parity and a shared decision contract, then test one explicit higher-timeframe-context → lower-timeframe-trigger hypothesis with chronological holdouts, realistic costs, trade-level risk, and an experiment ledger. The previous live accounting makes liquidity compression a candidate to investigate, not a certified winner. A novel combined archetype and gate tuning remain unimplemented; no safe or consistently profitable strategy is claimed.
