# Execution timing and recovered parent structure — September 10, 2026

## Scope and research decision

The quant reviewer approved a bounded same-minute-data execution-timing sensitivity test. The hourly native-book reference is not silently corrected. Unresolved hourly/minute pairing prohibits interpreting cross-dataset P&L as a latency effect. No production files, settings, original data, live orders, or risk allocations are changed.

Verification: **152 research tests pass**, including 23 minute-selector/simulator tests. The extension's 11 initial new tests failed before implementation; the existing 11 tests passed at that RED checkpoint. A further early-stop lockout witness was added during verification. The existing urllib3/LibreSSL warning remains. `git diff e46789d -- engine bin configs` is empty; whitespace checks pass.

The experiment uses only two fixed scenarios: zero additional delay and 90 seconds after the reclaim candle closes. The 90-second value is a sensitivity assumption, **not measured latency of a minute trading host**. Both reuse the same frozen causal minute events and absolute sweep-derived stops.

Eligible reference entry is the first exact minute open at or after readiness. Thus a 90-second request on aligned minute candles samples at +120 seconds, with 30 seconds of grid rounding. Signal availability, delayed order readiness, and sampled entry are separate timestamps. No pre-entry low stops a position that does not yet exist. This is not a new pending-order cancellation rule: an intervening price excursion does not cancel a frozen signal; a stop at/above the eventual entry price rejects it.

The holding period and lockout start at actual sampled entry. Entry-bar stop checks, adverse gap fills, fee model and fixed notional are preserved. That means delay can alter which subsequent events are entered or skipped. A per-event status ledger is required; scenario aggregate differences are not a matched-position latency effect.

## Completed minute experiment

Data: January 1, 2021 00:00 through August 31, 2026 23:59 UTC, inclusive; 2,979,360 minute bars. Fresh detection exactly reproduces all 15,910 previously recorded causal events. Both runs use the same events and $50,000 fixed position notional, a stop 0.15% below the sweep low, four-hour holding/lockout period and 12 bps round-trip cost. **Starting equity is unspecified; this is not a funded account simulation.** No compounding, funding, liquidation, market impact or independently calibrated additional slippage is modeled.

| Requested delay / sampled delay | Completed positions | Average initial stop risk | Diagnostic net P&L | Profit factor |
|---|---:|---:|---:|---:|
| 0s / 0s after reclaim close | 7,708 | $151.33 | −$496,873.92 | 0.57547 |
| 90s / 120s after reclaim close | 7,618 | $153.23 | −$481,844.63 | 0.57882 |

Neither run has an open/censored or unfilled event at the dataset end. Baseline skips 8,202 events for lockout. Delayed simulation skips 8,153 for lockout and rejects 139 invalid stops. Of entered event identities, 7,502 are shared, 206 appear only in the baseline, and 116 only in the delayed arm. The detailed status transitions are: completed→completed 7,502; busy→busy 8,078; completed→invalid 131; busy→completed 116; completed→busy 75; busy→invalid 8. All sum to the same 15,910 frozen events.

Baseline fees total $462,480, leaving negative gross P&L of −$34,393.92. Delayed fees total $457,080, leaving negative gross P&L of −$24,764.63. Both are negative in every exit-year stratum, including partial 2026. Minute-close marked dollar drawdowns are $499,910.06 and $483,906.20; these are diagnostic series, not capital-constrained drawdowns or actual losses, and omit intraminute excursions.

Zero delay exactly matches the prior artifact's summary and every pre-existing trade field. Independent code review also matched zero-delay source behavior on 200 randomized fixtures across next-open and close modes. Source, input and baseline-artifact hashes were checked; source was unchanged during the historical run. No delay was chosen as a strategy improvement. The aggregate difference includes selection/displacement, not just delayed prices for matched trades.

Ignored artifact: `results/research_validation_2026_09_10/minute_delay_sensitivity/fixed_0_90.json`, SHA-256 `f52d35aa97b9bc315ed7e97205e95a988e54a8acc7cc8e9d50fc22c6cddee3e8`. It retains both complete event-status ledgers and trade records. Frozen event-list SHA-256: `0512931562035ebfd071aca3aaa11c075dd1f6d38c49683821ef1e8f01beddd5` (canonical sorted-key compact JSON).

## Input pairing: what was established

Window: June 10 00:00 through June 20 00:00 UTC, end exclusive. The minute input contains 14,400 unique, aligned, contiguous observations. It is aggregated into 240 left-labelled hourly OHLCV rows without interpolation. `vol` is renamed `volume` only in memory.

- V23, `ohlcv_1h_full_2018_2026.parquet`, and the Coinbase-labelled hourly cache have exact equality on every OHLCV column for all 240 timestamps.
- No V23 close exactly matches the minute aggregate close in this window. Mean absolute basis is 3.7917227607 bps, calculated as `mean(abs(hourly_close / minute_close - 1) * 10000)`.
- The Binance-labelled hourly cache is different from V23 on all five columns at all 240 timestamps, and also has no exactly matching minute close.
- V12 ends in December 2024 and supplies no rows in this window. The V23 builder's default V12 path alone cannot establish the provenance of its 2026 extension.

The minute study header names Binance BTCUSDT futures; the hourly downloader inspected names Binance spot. Neither is a download receipt bound to every recovered file. Numerical identity with a Coinbase-labelled cache is evidence about this window, not proof of acquisition provenance, full-file equality, or the historical live deployment.

The reviewer independently reproduced the counts and verified all nine input/source hashes. Local ignored artifact: `results/research_validation_2026_09_10/native_pipeline/input_pairing_240h.json`; SHA-256 `977c1b0d23538f49f5a75fc9f63e6975a35cacea7c715f33c764249a0dc931d9`.

## Recovered multi-timeframe code

A wider local search found the previously referenced study modules in the sibling project:

- `/Users/rayghandchi/Bull Machine/one-strategy/idea_lab/structural_range.py`, SHA-256 `22e3e1ab21d8db585d89569c3d1d4a3fe3c74ce17ca72c5080a0ca7cf0caa8ca`.
- `/Users/rayghandchi/Bull Machine/one-strategy/idea_lab/htf_pivots.py`, SHA-256 `0c37d254cba93f40638ffeb2422d53649ea326adecee0a0690ed417b2f492f0b`.

The pivot module explicitly delays visibility until right-side confirming HTF bars close. The range module is a persistent state machine with body-close breaks and wick sweeps, but also permits floor tightening. Consequently, an event-bound parent needs an immutable version, not just a mutable range name or a backfilled price level. The modules remain untouched; discovery does not establish validated causality or an economic edge.

### Reproduced contract gaps

Quant review and independent root probes reproduced these behaviors without changing the sibling files:

1. `htf_pivots.py:62–73` accepts an internal-gap four-hour bucket containing only three hourly rows. Its final-bin test is not a constituent-completeness check. The reviewer also reproduced acceptance of half-hour-shifted hourly starts; require aligned, unique, ordered, contiguous closed inputs and exact bucket coverage before reuse.
2. `htf_pivots.py:105–132` strips timezone from confirmation storage, then UTC-aware hourly input fails at `merge_asof` with incompatible aware/naive keys. Preserve an explicit UTC clock; dropping timezone labels is not the contract.
3. `structural_range.py:136–233` uses the current hourly close but emits the original hour-open index. State is available no earlier than hour-open + one hour; minute consumers must not forward-fill it from hour-open.
4. Sweep checks precede floor tightening. With old bounds 90/120, new confirmed low 100, candle low 95 and close 110, the output floor becomes 100 while sweep-low remains zero. That zero refers to the old floor 90. Preserve pre/post bounds, event-evaluation version and availability, rather than reinterpret it against the new floor.
5. Unchanged pivots 90/120 and closes 110→121→110 yield active→broken-up→active, with a new formation age of zero. Reformation may reuse old anchors; requiring fresh anchors would change the strategy. Equal-priced distinct pivots also need IDs because level-change detection cannot distinguish them.

The reviewer found nominal complete/aligned/timezone-naive synthetic inputs prefix-consistent at six cutoffs. That limited positive result does not negate these contract gaps. Reuse the math and source state machine only behind a source-hashed research adapter with witnesses, explicit pivot IDs, range lineage/version IDs, pre/post state and `available_at`. Do not repair the sibling project or silently change its reformation policy.

The separately reviewed [candidate contracts](candidate_rule_contracts_2026_09_10.md) retain the distinction between evidence validity, prior compression, and fixed parent/child structure. Exact numerical anchoring rules remain project hypotheses rather than attributed trader facts.

## Next work

1. Implement H1 observed-evidence decision witnesses and H2's single prior-hour compression rule as separate research variants; report decision changes before outcomes.
2. Build the guarded parent ledger from the recovered source, with the five gaps above pinned by tests. Freeze the range contract before a historical parent-gate comparison.
3. Compare parent-only hourly and minute candidates independently, then test any conjunction on a declared protocol. No threshold optimizer, candidate graduation, live changes or profit claim is authorized by this checkpoint.
