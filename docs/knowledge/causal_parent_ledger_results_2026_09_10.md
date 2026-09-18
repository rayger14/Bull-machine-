# Causal parent structure: implementation and first data witness

## Accomplished

The research-only `scripts/research/causal_parent_ledger.py` now provides `build_parent_ledger`, `parent_asof` and `bind_parent`. It hash-verifies and guards the recovered sibling pivot/range functions, preserves their rules, and adds the identities and clocks needed to test parent/child setups without hindsight. Local commits: `494bcf2` and numerical validation fix `901156e`; [approved design](../superpowers/specs/2026-09-10-causal-parent-ledger-design.md).

Every pivot binds its instrument, occurrence, confirmation and exact causal supporting anchor window. Each range formation starts a lineage; floor tightening produces a new immutable version. Break/sweep annotations refer to the pre-update boundary, and later changes cannot rewrite an earlier child's frozen binding. A parent becomes available at the source hourly close, not its candle-open label. Binding requires parent availability strictly before the child's first sweep open.

The adapter rejects gaps, duplicates, incomplete interior evidence, unaligned/naive timestamps, invalid ATR and source-hash mismatch. Only complete 4-hour/daily buckets enter pivot detection. It excludes partial leading buckets and reports developing trailing ones. UTC-normalizable inputs preserve same-epoch identity. NaN/zero ATR is preserved and flagged rather than replaced. Parent carry expires at the next missing hourly update, rather than remaining active indefinitely.

All four explicit source hypotheses remain separate: 4H/N=3, 4H/N=5, 1D/N=3, 1D/N=5. These anchor choices and the recovered numeric constants are project hypotheses, not trader-certified settings. This is an initial higher-timeframe reference; it does not claim weekly/monthly Wyckoff fidelity.

## Verification

- 47 focused parent tests pass, including seven local real-source integration cases with no skips.
- 219 full research tests pass in the final root verification (11.61 seconds), with the one pre-existing urllib3/LibreSSL warning; no new parent warning.
- Source parity covers all four anchor/N combinations. Same-data minute aggregation and direct hourly input share parent identities. No different-venue parity is claimed.
- Prefix and fresh full-history restart tests pass; serialized partial checkpoint restore is not implemented.
- Controlled annotation seams pin old-floor/new-floor sweep semantics, old-anchor reformation and equal-priced pivot identity. These supplement rather than replace real-source integrations.
- Source bytes, adapter/helpers, runtime, parameters and caller ATR contract are manifest-bound. Input data hash is a run-level field, not a future-dependent event ID.
- `git diff b26a5b8 -- engine bin configs` is empty. No original data or sibling source was edited.

The first real-data attempt correctly exposed a test gap: recovered pandas resample sums and independent grouped volume sums differed by 1–2 floating-point ULPs (37 of 180 four-hour buckets, maximum 9.094947017729282e-13). The initial exact comparator rejected the run. The reviewed fix permits at most eight ULPs for finite nonnegative volume **comparison only**, records that convention in the contract, and leaves prices/times exact. Tests accept the eighth ULP, reject the ninth, material differences, OHLC drift and nonfinite values. Inputs and source logic are not rounded or rewritten.

Independent task review approved the parent component, and broad cross-component review found no critical or important findings. Its one minor test-quality finding was resolved in `c36dcac`: the minute/hour fixture now asserts actual aggregated volume exactly instead of overwriting it. The final full research suite passed after that test-only change. Approval is for local research use, not live capital. The existing `quant/archetype-evidence-audit` branch is retained; no merge or push.

## June 2026 historical annotation, not strategy performance

Input: 720 hourly rows, June 1 00:00 through July 1 00:00 UTC, from the saved V23 OHLCV/ATR store. This is a reused research window. ATR is passed unchanged with explicit `saved_atr_14_formula_unverified` identity; its historical formula/acquisition lineage is not certified. Close-time availability is assumed. No entries, costs, returns, equity or per-trade risk are evaluated by this parent-only diagnostic.

| Hypothesis | Confirmed pivots | Immutable versions | Range lineages | Active hours | Forming hours | Break hours |
|---|---:|---:|---:|---:|---:|---:|
| 4H / N=3 | 34 | 20 | 14 | 398 | 308 | 14 |
| 4H / N=5 | 18 | 11 | 8 | 370 | 342 | 8 |
| 1D / N=3 | 5 | 4 | 3 | 180 | 538 | 2 |
| 1D / N=5 | 2 | 0 | 0 | 0 | 720 | 0 |

All four pass 360-hour prefix comparisons for pivots, versions and transitions filtered by availability, plus exact fresh full-history rebuild equality. Higher coverage is not higher quality, child signal count or a profitable edge. The daily/N=5 result means no active parent under this supplied history and construction; it does not prove that daily structure is useless. Changing supplied prehistory can change the available structural state.

Private ignored artifact: `results/research_validation_2026_09_10/parent_ledger/june_720h_four_hypotheses.json`, SHA-256 `d38ff094e74b962a14cc7309dcc0c8f6b57ce0f92b8aa86404f47e6e9477336f`.

## What is not done

This ledger is not integrated into native candidates, live orders, sizing or exits. Its Python side-effect guard is not an OS sandbox for arbitrary hostile native code. Input/release/receipt provenance remains incomplete. All outputs remain `certified=False`; no configuration has been selected as a profitable strategy.

The [H1/H2 sidecar](context_permission_audit_2026_09_10.md) and this ledger solve different prerequisites. H1 cannot grant actual observed-input permission until consumed-field provenance is captured at the native computation boundary. H2's loose numeric prerequisite does not by itself establish compression or edge.

## Next bounded work

1. Bind the already-defined hourly LC reclaim and minute child-sweep hypotheses to these frozen parents, retaining all four anchor configurations as declared alternatives. Report absent/unconfirmed/invalidated context and pass/fail reasons, not only winners.
2. Complete actual consumed-field provenance before enforcing evidence permission. Keep raw feed freshness, formula identity and fallback state explicit.
3. Replay each intervention independently through full selection/lockout/cooldown/book history, preserving all 17 archetype identities and disclosing displaced opportunities.
4. Keep management and executable-fill changes in separate experiments; evaluate chronological holdouts and genuinely new forward evidence after freezing designs. The September 8 winner is a diagnostic case, not a parameter target.

The practical all-seeing eye is a shared, time-correct explanation of location and sequence. Each archetype remains a separate decision hypothesis; unknown context remains unknown rather than a confidence score.
