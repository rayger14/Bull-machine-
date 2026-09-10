# Archetype translation and evidence audit — 2026-09-09

## 1. Executive summary

- All **17 configured archetypes** are included below. HOB is an additional research candidate, not a replacement for one of the 17. Graduation is not the scope boundary.
- The captured paper ledger has 532 exits, not 532 independent trades. The March-9-entry-cutoff view has 220 apparently completed identified positions, PF 1.1301, 44.55% position win rate, and $13,537.33 completed-position realized P&L. This is not yet evidence of consistent profitability.
- Trader concepts often survived as single-bar proxies. Nested structure, event ordering, direction, and level identity are not interchangeable with an indicator score. Restoring a teaching is a hypothesis, not an automatic improvement.
- **New minute-study blocker:** the original event selector uses future sweeps to suppress earlier entries. Full reproduction matches all 7,952 cached events, but prefix replay changes historical signals. Its headline $204,498 simulation is reproducible, not causally validated.
- **New hourly-store findings:** inspected V23 funding_Z is constant zero; effort_result_ratio disappears in 2025–2026. Gate tuning requires effective-rule and feature-availability parity first. Production code/configuration remains unchanged.

## 2. Methodology and data inventory

This is an accounting, source-translation, and deterministic reproduction audit. **No new WFO/CPCV result or optimized strategy is claimed.** Previously examined historical windows remain research data, not newly pristine holdouts.

Working checkout: `Bull-machine-/Bull-machine-`, branch `quant/archetype-evidence-audit`, research base `9a9d12e`. The richer original checkout one directory above is at `d2fe814`; its newer minute-study source and final specification were read separately. The inspected source is not a deployment fingerprint for every historical trade.

### Recovered local inputs

Paths below are relative to the original checkout unless explicitly qualified. Raw data has not been uploaded or added to Git.

| Input | Inspected scope | Role / limitation |
|---|---|---|
| `data/features_mtf/BTC_1H_FEATURES_V23_PARITY_2018_2026.parquet` | 74,436 rows, 364 columns; 2018-03-01 through 2026-08-30 00:00 UTC | Hourly research substrate; name alone does not establish parity |
| `data/cache/derivatives_hourly_full.parquet` | 50,617 rows, 9 columns | Historical derivatives source; publication timing/provenance still needs verification |
| `data/cache/ohlcv_1h_full_2018_2026.parquet` | Located | Raw hourly price history |
| `data/cache/macro_daily_history.parquet` | Located | Daily macro inputs; require as-of availability, not calendar-date joins |
| `data/cache/coinbase_btcusd_1h_2024_11_to_2026_08.parquet` | Located | Cross-venue price comparison; spot is not perpetual execution |
| `data/databento/btc_fut_1m_2021_2026.parquet` | 2,633,326 rows, 7 columns | Distinct futures instrument/source; do not merge with Binance as identical bars |
| `data/databento/btc_fut_oi_2021_2026.parquet`, `btc_fut_trades_2021_2026.parquet` | Located | OI / trade data; contract and roll treatment not audited |
| `results/coinbase_paper/live_features/2026-06.jsonl`, `2026-07.jsonl` | 533 rows total; June 18–July 10 | Partial archived live snapshots, not full-period coverage or necessarily unique bars |
| Sibling `one-strategy/idea_lab/` | Prior unified strategies, structural-range work, HTF pivots, captured results | Avoid repeating prior failed studies; older V22 substrate is a separate limitation |
| Siblings `risk-neutral-deploy/`, `wyckoff-campaign-v2/`, `Bull-machine--1/` | Located / filename inventory | Additional versions, not independent datasets by default |
| Project-specific temporary research scratchpad | 2,979,360 Binance minute bars, 7,952 cached events, previous study scripts and execution variants | Recovered executable minute evidence; see reproducibility appendix |
| `/tmp/bull-live-audit.oGDiLt/{trades,status}.json` | Full captured API responses | Snapshot used by accounting tool; not a fresh atomic server snapshot |

The project tree contains older V12–V22 stores, rebuild shards, Binance funding/metrics archives, aggregate-trade flow history, and prior study outputs. They are useful provenance, not interchangeable validation samples. This is a targeted project-folder search, not a claim that every relevant file on the computer has been inspected.

### Accounting

`scripts/research/archetype_evidence_audit.py` groups scale-outs by position ID, separates open partials and unknown legacy IDs, and preserves zero-trade archetypes. Nine unit tests cover the grouping/error contracts. JSON output includes input hashes, configuration hashes, timestamp coverage, and limitations.

The whole ledger contains 68 exits without position IDs: they remain unidentified exits. The March-9 view has 450 exits across 221 identified positions with exits: 220 apparently closed and one still open with $125.94 realized. Closure is inferred from terminal exit reason and captured open-state IDs, not verified against an independent entry/fill ledger.

The API's $13,663.25 realized total differs from summed rounded exit values ($13,663.27) by two cents. Separate funding is $124.21; the completed-position table does not allocate that funding. The snapshot's $2m reference cash reflects recapitalization and changing size; dollar P&L is not a constant-capital or constant-risk return. Reconstructed average initial risk for completed post-cutoff positions is approximately $931.78, subject to historical stop/quantity metadata quality. The dashboard's historical 32.7% drawdown is not a clean return statistic on the current $2m basis.

## 3. Findings

### A. All 17: paper-position evidence

Entries from 2026-03-09, observed exits through 2026-09-09. PF is gross completed-position profits divided by gross completed-position losses, before separately recorded funding. Counts below are **not exits** and do not establish independence of correlated BTC exposures.

| Archetype | Completed positions | Realized P&L | PF | Interpretation |
|---|---:|---:|---:|---|
| liquidity_compression | 36 | $12,539.71 | 2.168 | Strongest hourly lead in this snapshot; uncertain, changing config/size |
| funding_divergence | 9 | $4,713.06 | 2.503 | Small sample; funding feature parity essential |
| liquidity_sweep | 21 | $4,288.11 | 1.478 | Positive proxy; not proof of textbook swept-level edge |
| order_block_retest | 4 | $3,656.07 | 5.364 | Far too small to infer dependability |
| exhaustion_reversal | 7 | $3,463.32 | 2.159 | Small sample; direction rule changed historically |
| trap_within_trend | 21 | $1,657.93 | 1.130 | Thin, direction semantics need audit |
| failed_continuation | 4 | $497.63 | 1.165 | Very small; effort/result gate parity unresolved |
| long_squeeze | 5 | -$177.19 | 0.857 | Small, derivatives-gate parity unresolved |
| fvg_continuation | 11 | -$256.46 | 0.899 | Gap/break direction association incomplete |
| wick_trap | 12 | -$950.89 | 0.908 | Excludes $125.94 partial realized on an open position |
| retest_cluster | 9 | -$1,596.48 | 0.659 | Small; time/cluster semantics need explicit specification |
| spring | 3 | -$1,604.89 | 0.028 | Tiny post-cutoff sample |
| confluence_breakout | 62 | -$6,203.43 | 0.810 | Largest sample; gate/selection interactions matter |
| oi_divergence | 16 | -$6,489.16 | 0.235 | Negative evidence; not permission to disable |
| liquidity_vacuum | 0 | $0.00 | — | No post-cutoff closed evidence; older exits do exist |
| whipsaw | 0 | $0.00 | — | Neutral direction cannot emit through generic detect path |
| volume_fade_chop | 0 | $0.00 | — | Same neutral-direction limitation |

The descriptive weekly-cluster bootstrap interval for overall mean position dollars is approximately [-$181, +$351]; liquidity compression's interval also crosses zero. This is not a probability-of-future-profit estimate and does not adjust for every past research trial.

### B. Trader intent versus operative checks

Primary references in this checkout: `engine/archetypes/logic.py`, `engine/archetypes/archetype_instance.py`, `engine/archetypes/structural_check.py`, and **`configs/champion/archetypes_v14rq/`**, resolved through `configs/champion_paper.json`. The similarly named default YAML directory is not the inspected live config directory.

H = YAML hard mode, S = YAML soft mode; these are not the complete runtime decision. Threshold bypass, downstream gate enforcement, missing-value policy, cooldown, and dedup remain separate.

| Archetype | Intended concept / lineage | Operative identity and important YAML gates | Translation / test question |
|---|---|---|---|
| spring | Wyckoff/PTI spring or bear trap | Bullish PTI trap type; PTI >=0.1 (S, NaN skip) | Track range and reclaim identity, not merely a trap label; older UTAD acceptance was changed |
| order_block_retest | SMC/HOB retest family | Near a recent BOS close, either direction; any 1H BOS + fib-time >=0.1 (S) | No required order-block candle, zone, freshness, or directional linkage in this identity |
| fvg_continuation | SMC gap continuation | Any 1H/4H FVG + recent bullish OR bearish BOS; any-BOS/any-FVG (H) | Are the gap, break, and long direction from the same structure? |
| failed_continuation | Failing continuation / effort-result | FVG + falling ADX + vol_z<1.5; RSI<=55, effort/result<=1.4, chop<=0.25 (H) | Weak ADX is not a full failed-break-and-reclaim sequence; missing effort/result changes rule |
| liquidity_compression | Volume climax/absorption reversal | Climax, absorption, or volume+RSI extreme; vol_z>=3, RSI extreme, BB<=0.06, chop<=0.5 (H) | An exhaustion proxy, not simply quiet compression; RSI extreme is two-sided |
| exhaustion_reversal | Exhaustion reversal family | RSI extreme; current config enables oversold-only identity; ATR percentile>=0.5, vol_z>=0.3 (H) | Preserve repaired direction; do not attribute older trades to the newer rule |
| liquidity_sweep | ZeroIka / liquidity-reclaim family | Dominant lower wick; liquidity>=0.35 and lower-wick/range>=0.5 (S) | No prior level or multi-bar reclaim required; test native minute setup separately |
| trap_within_trend | Bojan/wick-trap with trend | Wick anomaly + ADX>=10 + EMA/HTF-fusion context; volume>=0, lower wick>=0.25, pivot age<=110 (H) | Bearish HTF-fusion fallback can pass despite below-EMA long context |
| wick_trap | Bojan-style wick/trap proxy | Wick anomaly; lower wick>=0.35, vol_z>=0 (H); structural exodus option | Lacks persistent magnet/level and trap-reset sequence; source wick magnitudes need provenance |
| retest_cluster | Volume exhaustion + temporal/fib cluster | Volume+RSI extreme; fib cluster positive when valid; temporal score>=0.45 (S) | Missing/null cluster can bypass check; missing key defaults to zero/reject. What level is being retested? |
| confluence_breakout | Compression/value-area breakout | ATR percentile<0.3, near POC, BOMS confirmation by default; volume>=0.5 and context gates (S) | Gate-enforcement opt-out remains; tighten-only experiments can be inert or change dedup ranking |
| liquidity_vacuum | Capitulation / vacuum reversal | Bearish BOS + climax/absorption; low liquidity, volume, wick exhaustion (H); direction long | Requires evidence of reversal if trading against the break; none in this identity |
| whipsaw | Failed upside break / weakness | Upper wick>2x body + SOW or vol_z>=2; low-volume YAML + SOW (S) | Neutral direction returns None; high-volume structural alternative conflicts with low-volume gate |
| funding_divergence | Crowded-short squeeze | Negative raw funding OR funding Z extreme; crowding/OI/resilience gates (S, mostly skip) | Persistent crowded state is not necessarily entry timing; raw and normalized funding differ |
| long_squeeze | Crowded-long unwind (short) | Positive raw funding OR funding Z extreme; RSI>=60, crowding/OI/vol-shock gates (S) | Short-side setup distinct from long dip buying; vol_shock gate availability missing |
| volume_fade_chop | Low-volume range fade | vol_z<0.5, ADX<25; RSI extreme and effort/result<=1.5 (S) | Neutral direction prevents execution; needs explicit long/short range geometry |
| oi_divergence | OI capitulation / price divergence | Optional price-extreme identity; otherwise returns True; falling OI, RSI<=35, volume/taker/context (S) | In inspected default path there is no mandatory structural price-divergence identity |

**Source honesty:** the July founding-knowledge inventory is useful but the August-7 provenance addendum is more precise: structural objects are trader-sourced; many numerical magnitudes are project hypotheses. Moneytaur's locally documented corpus is thin. Do not attribute the entire dominance/breadth doctrine, a universal wick percentage, or an exact target formula to him. The source-backed concepts include fixed structural ranges, body-close versus wick-break distinction, and price/time context; their profitability remains a separate empirical question. See `wyckoff_audit.md` addenda 35–36.

### C. What hard/soft actually means

`ArchetypeInstance._evaluate_gates()` skips missing values when configured to skip. If all gates are skipped, it returns passed. Hard mode rejects a failed gate; soft mode multiplies fusion by the fraction of evaluated gates that passed.

`bin/live/v11_shadow_runner.py` has additional enforcement on the **below-threshold bypass branch**, default on, but `confluence_breakout.yaml` explicitly opts out. Signals already above threshold enter a different branch. Therefore the old May blanket statement "soft gates never matter under bypass" is not an accurate complete description of the present source. Nor does a column called `hard_gates` imply unconditional enforcement.

Dedup selects the highest-fusion candidate per direction. A soft gate can change which archetype owns a bar even when it does not prevent a trade. Report both the changed archetype and the full book; a higher system P&L can be a routing change, not better detection. Compare independent research sleeves as well as integrated replay.

Collection mode also changes position/allocation controls; do not assume configured max positions implies the same live risk cap. Per-archetype signal cooldown is a separate mechanism. YAML exit ladders are not automatically consumed by the shared exit engine: trace the runtime exit JSON and runner separately before changing an apparently relevant YAML field.

### D. New feature-availability findings

Directly inspected on the 74,436-row V23 extended parquet:

| Feature | Finding | Consequence to investigate |
|---|---|---|
| funding_Z | Exactly one unique value: 0.0 on every row | Both funding-direction YAML thresholds see a constant; zero is not missing, so NaN skip does not rescue it |
| binance_funding_rate | Variable; absent in 2018–2019 and most of 2020 | Structural raw-funding branch can fire while normalized-funding gate fails |
| effort_result_ratio | Populated through 2024; entirely missing in 2025–2026 | A skipping gate evaluates in one era and disappears in another |
| vol_shock | Column absent | Long-squeeze gate configured to skip cannot constrain this stored input |
| taker_imbalance | 65.0% nonmissing during 2022 | Fold-level coverage must accompany any flow-based study |
| 4H/1D Wyckoff bullish scores | Populated and variable | Existence verified, not bar-availability timing or interpretive accuracy |

The 533 saved live-feature rows have 510 unique funding_Z values, spanning approximately -2.85 to +2.81; effort_result_ratio and vol_shock are absent. This establishes an archived-input mismatch. It is not yet a complete runtime feature-repair diagnosis or a claim that the current server has the same coverage as those June/July snapshots. Inspect transformations before declaring the effect on final trades.

### E. Minute study: reproduced result, invalidated causal claim

Data: Binance BTCUSDT futures minute history, 2021-01-01–2026-08-31 UTC, 2,979,360 rows; no missing timestamps relative to a complete minute index. This does not certify feed authenticity or execution quality. Source event cache: 7,952 rows. Replay simulation: one position at a time with a **fixed four-hour lockout after entry even when stopped earlier**, 12 bps all-in round-trip cost, stop at sweep low minus 0.15%.

There is no explicit funded-wallet starting equity or margin model in this simulation: $50,000 is **fixed trade notional**, not a validated account requirement. Average initial price-to-stop risk is approximately $189.23 before fees/gap risk. Do not present its dollars as a leveraged account return.

| Reproduction / diagnostic | Taken trades | Total P&L | Realized closed-trade max DD | Status |
|---|---:|---:|---:|---|
| Original close fill, flat $50k | 5,033 | $204,497.72 | -$9,773.67 | Headline reproduced; noncausal selector |
| Original next-open latency code | 5,033 | $215,270.55 | -$8,556.02 | Skips stop checks on entry minute |
| Same next-open assumption, include entry-minute low | 5,033 | $204,558.84 | -$9,478.78 | Fill-bar diagnostic only; selector still noncausal |
| Close-fill plus documented 1.5x touch tilt | 5,033 | $278,361.32 | -$12,416.30 | Different risk budget; not original headline or accepted sizing |

All four use the same cached historical event selection; none is a new validated strategy. Touch tilt raises average initial risk to about $236.86. Realized-close DD omits intratrade mark-to-market drawdown. Funding, spread/impact, gap fills, venue transfer, and actual available capital are not separately modeled. The 7,592-event original claim works out to roughly **3.7 events/day**, not the document's ~22/day.

**Causality counterexample, UTC:**

| Input ends immediately before | Signals in prefix | Full-run signals before same cutoff | Historical entries lost in full run |
|---|---:|---:|---|
| 2021-01-15 | 50 | 48 | Jan 14 12:12 and 18:31 |
| 2021-04-01 | 330 | 327 | Mar 31 19:59, 21:47, 22:58 |

The Jan-14 07:03 pivot is processed first and searches forward to a Jan-15 06:07 sweep. That future sweep sets `last_event_i`. Later-created Jan-14 pivots then have their earlier sweeps rejected because `s - last_event_i < 60` is negative. This is event-order lookahead, not merely an unconfirmed centered pivot. The detector already waits 15 bars after its pivot; that wait does **not** solve the future cooldown mutation.

Impact on total P&L is **unknown** until the event-selection semantics are fixed and causally replayed. The counterexample is enough to withdraw acceptance of the current backtest, not enough to conclude the underlying market pattern has no edge.

## 4. Recommendation and proposed experiment design

**Decision: retain production unchanged; do not tune gates against the present unqualified baselines.** No YAML change is recommended yet, so there is no deployment diff. First separate implementation correctness from empirical strategy usefulness.

The user's approved conceptual target is structure within structure. Three possible approaches:

1. **Recommended: staged research using existing archetypes plus a shared structural context.** Preserve each setup's identity and independently test its entry/management; share timestamps, parent/child ranges, and exposure accounting. Lowest conceptual drift, easiest attribution.
2. A new unified state machine immediately. Closer to a single-eye implementation, but a large change with many interacting choices; prior `one-strategy/idea_lab` versions already failed across eras on older data.
3. A learned meta-selector over all live features. Potentially useful later, but the current small effective live sample and parity gaps cannot support treating it as a reliable oracle.

Suggested research contract, **not yet an implementation specification**:

- **Context:** confirmed higher-timeframe structure, range ID, location, direction and liquidity objectives. HTF context may alter a setup's interpretation rather than universally veto countertrend entries.
- **Setup:** lower-timeframe structure linked to that parent range. Distinguish sweep/reclaim reversal, continuation/retest, and crowded-position unwind rather than averaging their incompatible prerequisites.
- **Trigger:** observable ordered events with explicit `formed_at`, `confirmed_at`, `available_at`, expiry, and invalidation. A 4H close cannot inform minute entries before it closes.
- **Management:** original risk stays immutable for R accounting; stops/targets refer to the same structure as entry. Model entry, stop, scale-out and gap ordering, including mark-to-market and portfolio constraints.
- **Explanation:** every decision records the evidence, unavailable inputs, passed/failed/skipped gates, alternative setups, and what would invalidate the trade. Abstaining is a valid decision.

### Ordered test queue

1. **Minute causal replay contract:** preserve the discovered counterexample; define chronological level/sweep/reclaim handling and concurrent-level dedup before a repair. Require historical decisions to be invariant to future-data changes. Reproduce the old run as a separate reference, never silently overwrite it.
2. **Hourly gate-observability matrix:** every gate × feature × fold × live snapshot: units, availability, missing/constant rate, evaluated/pass/fail/skip count, effective runtime enforcement. Trace funding_Z construction and the 2025 effort/result discontinuity.
3. **Golden-master replay:** freeze code, config, instrument, input hashes and as-of joins. Run the same bars through research and live decision paths. Establish deterministic entry/exit accounting before optimization.
4. **Single-mechanism ablations:** for a predeclared structural condition compare baseline, explicit hard gating, existing soft behavior, and a risk-budget-matched sizing alternative in scratch configurations. Do not assume boosts are better; the old 2/2 success anecdote and later parity revisions are not proof. Change entries and exits in separate experiments.
5. **Time-based validation:** use existing 2018–2022 / 2023–2024 / 2025–2026 splits as historical robustness diagnostics, not untouched evidence. For minute data use available 2021 onward. Register rolling folds, purge overlapping outcome windows, and select parameters only inside training folds. Account for all attempted variants; preserve newly collected forward data after the design is frozen.
6. **Acceptance:** report risk-normalized expectancy, PF, completed n, drawdown including open positions, tail/gap losses, turnover/cost sensitivity, capital exposure, regime stratification, and target-archetype plus full-book deltas. A sample below 30 OOS completed positions is directional only under the project's rules; exceeding 30 alone proves nothing. Require parameter stability and independent forward execution evidence before deployment.

Repeated selection on historical backtests increases false-discovery risk; a profitable-looking best variant is not enough. This is the motivation for tracking every trial and reserving genuinely forward evidence. [Bailey et al., The Probability of Backtest Overfitting](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf).

## 5. Sample size and honest caveats

Live data is paper-traded, selected by a changing shared engine, concentrated in BTC, and not constant risk. The post-cutoff ledger includes earlier configurations and a strong August; it is not a frozen-rule prospective test. A high exit win rate is not completed-position accuracy. A clean causal detector is necessary but does not establish a profitable strategy.

The funding/effort-result observations are on a specific recovered store. No broad claim that every V23 variant or live computation is broken is justified. The minute causality issue is proved; its corrected economic impact is not measured here. No strategy earns the label safe, guaranteed, or consistently profitable from these results.

## 6. What this does not test

No optimized gate, new all-seeing-eye implementation, repaired minute detector, independent holdout, funded account simulation, full execution model, or live deployment. No complete current-server config/commit fingerprint. No all-period live feature ledger. No revalidation of every old Wyckoff/Bojan/SMC detector. Trader attribution is limited to recovered project documents, not a fresh review of every original trader post/video.

## 7. Files and reproducibility

Research additions: this report; `scripts/research/archetype_evidence_audit.py`; `tests/research/test_archetype_evidence_audit.py`; `docs/superpowers/plans/2026-09-09-archetype-evidence-audit.md`. Generated scorecard is under `results/archetype_evidence_2026_09_09/`. Production files untouched.

Inspected input SHA-256 fingerprints:

```text
V23_PARITY_2018_2026: 0e8604d0dacf7435e42e9759fe64aca15bfa1e3d66f14c1d21c142f41d1ab61f
btc_1m_2021_2026:    5b8a4533f70b8ccd0dc533984469e6886ec73ce315d48f150c667ccb97e84035
scalper_events_idx: 8020ed91431648f4e8ed5822e45088b567702439c5f8ca9e830554dae718623b
```

Accounting reproduction from this checkout:

```sh
python3 -m unittest discover -s tests/research -p test_archetype_evidence_audit.py -v
python3 scripts/research/archetype_evidence_audit.py --trades /tmp/bull-live-audit.oGDiLt/trades.json --status /tmp/bull-live-audit.oGDiLt/status.json --archetype-dir configs/champion/archetypes_v14rq --out results/archetype_evidence_2026_09_09
```

### Minute causality reproduction (read-only diagnostic)

This reproduces the recovered fast-event cache, including its bug. It is not a trading implementation. The integer-minute substitution for time lookup is valid for this specific complete minute index.

```python
from pathlib import Path
import bisect
import numpy as np
import pandas as pd

scratch = Path('/private/tmp/claude-501/-Users-rayghandchi-Bull-Machine-Bull-machine-/833eefef-c5b5-45bb-af8d-fd9afb9e129c/scratchpad')
bars = pd.read_parquet(scratch / 'btc_1m_2021_2026.parquet')
cache = pd.read_parquet(scratch / 'scalper_events_idx.parquet')

def legacy_events(frame):
    low, close = frame.low.to_numpy(), frame.close.to_numpy()
    piv = np.flatnonzero((frame.low.rolling(31, center=True).min() == frame.low).to_numpy())
    last, events = -10**9, []
    for k in range(2, len(piv)):
        i, level = piv[k], low[piv[k]]
        j = bisect.bisect_left(piv, i - 1440, 0, k)
        match = (abs(low[piv[j:k]] - level) / level <= .001) & (piv[j:k] <= i - 30)
        if match.sum() < 1:
            continue
        breaks = np.flatnonzero(low[i+15:min(len(frame)-1, i+1440)] < level * .9998)
        if not len(breaks):
            continue
        sweep = i + 15 + breaks[0]
        if sweep - last < 60:
            continue
        reclaim = np.flatnonzero(close[sweep:min(sweep+121, len(frame))] > level)
        if not len(reclaim) or reclaim[0] > 30:
            continue
        events.append(int(sweep + reclaim[0]))
        last = sweep
    return events

full = legacy_events(bars)
assert len(full) == len(cache) == 7952
assert set(full) == set(cache.ridx)
for date in ['2021-01-15', '2021-04-01']:
    end = bars.index.searchsorted(pd.Timestamp(date, tz='UTC'))
    prefix = set(legacy_events(bars.iloc[:end]))
    reference = {i for i in full if i < end}
    print(date, len(prefix), len(reference), [str(bars.index[i]) for i in sorted(prefix-reference)])
    # These assertions document a known bug; they do NOT validate the strategy.
    assert prefix != reference
```

### Standing orders (verbatim from project quant instructions)

- **NEVER turn off bypass_threshold** — data collection mode is required for the foreseeable future
- **NEVER disable any archetype** — all 16 stay enabled to collect maximum live signal data
- **NEVER make production config changes** (bypass, disabled_archetypes, thresholds, archetype YAMLs) without explicit user approval
- **NEVER edit production code/configs directly** — recommendations and diffs only

The quoted historical count of 16 is stale; this audit preserves all **17** currently configured archetypes.
