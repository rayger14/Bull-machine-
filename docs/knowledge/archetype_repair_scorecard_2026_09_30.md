# Archetype repair priorities and shared mechanisms

Started September 30, 2026; verified October 1. Offline research triage on `quant/archetype-evidence-audit`,
HEAD `85923a4`. The user approved the broader discovery-and-repair roadmap.
This report completes its first deliverable: current-code reconciliation,
research priorities, shared mechanisms and data readiness. It does not report
a new strategy backtest or certify any archetype for deployment.

## Recommendation

Keep upside LC as a frozen reference, and spend the next research budget on
directional permission for `trap_within_trend` and a minute-native nested
compression, breakout and first-retest sequence. The initially shortlisted
hourly swept-level repair is parked: the final prior-study reconciliation found
related level-restoration failures. Do not fill that slot with another idea just
to reach three hypotheses.
The [experiment specification](../superpowers/specs/2026-09-30-archetype-repair-discovery-design.md)
defines these precisely. No rule has been enabled or economically scored here.

This is a ranking of testability, reuse and repair scope, not an estimated
ranking of future profitability. LC has the best-developed recent research
baseline, not a proven edge. Other archetypes remain preserved and visible.

## What was inspected

The active local champion selects `configs/champion/archetypes_v14rq/`, not
the similarly named default directory. Parsing its immediate YAML files found
17 named, enabled definitions: 14 long, one short and two neutral; seven hard
gate modes, ten soft modes, and 60 configured gate records. Three additional
example YAML files lack a top-level name and are not archetypes. Old "16"
comments and letter aliases are not the roster authority.

The inspection covered all 17 identity functions and YAML gate definitions,
champion structural overrides, structural bridge, detector gate/cooldown flow,
runner bypass branch, exit initialization, signal observer and isolated replay
boundary. This is the current checkout, not a fingerprint of every historical
server deployment. Selected data columns were profiled directly; no new trade
outcomes were used to choose this ranking.

Primary code: [identity logic](../../engine/archetypes/logic.py),
[champion configuration](../../configs/champion_paper.json),
[structural bridge](../../engine/archetypes/structural_check.py),
[detector](../../engine/archetypes/archetype_instance.py),
[feature producer](../../bin/live/live_feature_computer.py), and
[runner](../../bin/live/v11_shadow_runner.py).
The existing Graphify `ArchetypeLogic` node oriented the identity/bridge trace;
its incomplete research coverage was not treated as proof about current results.

## Ranked scorecard for all 17

H/S below describes YAML mode, not the entire entry decision. All numerical
rules are project implementation choices unless separately sourced. "Next"
means outside this campaign's three-hypothesis budget, not permission to tune.

| Rank | Archetype and role | Operative identity and gates | Missing distinction or dependency | Research decision |
|---|---|---|---|---|
| 1 | `liquidity_compression`, long, H | Climax/absorption or volume plus either RSI extreme; volume Z >=3, RSI extreme, BB width <=.06, chop <=.5. All four gates permit missing-value skip. | Current-bar eligibility does not prove preceding compression or distinguish rebound from expansion. Historical absorption coverage changes. | Frozen upside reference; no new room or confirmation gate. Full-LC and room-filter weaknesses are already documented. |
| 2 | `trap_within_trend`, long, H | Large wick, ADX check, then above-EMA **or extreme 4H fusion**; volume >=0, lower wick >=.25, pivot age <=110 when available. | The fallback score is nondirectional; low strength can qualify as "trend" when price is below EMA. | R1: require the existing above-EMA evidence; isolate removal of the fusion fallback. |
| 3 | `liquidity_sweep`, long, S | Dominant lower wick; liquidity >=.35 and lower wick/range >=.5 gates. | No identified pre-existing level must be breached and reclaimed. A long lower wick need not sweep anything. | R2 parked after prior-study reconciliation: related hourly level repairs already failed. Preserve the explicit prior-24-hour-low proposal, but no new economic run without a materially distinct rationale and approval. |
| 4 | `order_block_retest`, long, S | Close near a prior bullish **or bearish** BOS close within 20 bars; any 1H BOS and fib-time >=.1 soft gates. | No required order-block candle/zone, freshness or direction linkage; live fib anchor semantics need correction before interpretation. | Next: same-zone, direction-correct retest definition. More construction work than R1/R2; four old paper positions do not establish dependability. |
| 5 | `fvg_continuation`, long, H | Any 1H/4H FVG plus recent BOS of either direction; any-BOS/any-FVG gates. | Break, gap and entry direction need not describe the same event or still-active zone. | Next: bind gap and directional break IDs; do not substitute more SMC score. |
| 6 | `wick_trap`, long, H | Wick anomaly, lower wick >=.35 and volume >=0; champion enables a macro-exodus refusal when inputs exist. | Rejection is not an ordered trap/reset at an identified level. Macro coverage and provenance are separate. | Preserve earlier studies; not a new buyer-flow boost campaign. Shares rejection evidence with sweep/TWT, not independent confirmation. |
| 7 | `exhaustion_reversal`, long, H | Champion already sets oversold-only identity; RSI extreme, ATR percentile >=.5 when available, volume >=.3. | Oversold does not prove reversal completion; historical rule versions differ. | Next: level/reclaim timing after exhaustion, not rediscovery of the already-enabled direction repair. |
| 8 | `spring`, long, S | Bullish PTI spring/bear-trap identity by default; PTI >=.1 when available. | The label must retain the parent range, boundary and event lifecycle that produced it. | Next: PTI-to-range lineage audit. Do not re-enable historical bearish UTAD acceptance for long entries. |
| 9 | `confluence_breakout`, long, S | Structural ATR percentile <.30, POC distance <.05, BOMS confirmation by default; YAML has additional volume/context gates and bypass-enforcement opt-out. | A coil near value is not an accepted directional break. Several YAML numbers differ from identity constants; tightening an unused threshold can be inert. | Source of compression/break concepts for R3; no full-book threshold optimization. |
| 10 | `failed_continuation`, long, H | FVG, declining ADX and identity volume Z <1.5; RSI <=55, effort/result <=1.4 and chop <=.25 when available. | Declining ADX is not a failed break/reclaim sequence; effort/result disappears in the later store. | Sequence/data work first. Do not interpret a skipped missing gate as validated effort/result evidence. |
| 11 | `liquidity_vacuum`, long, H | Bearish 1H BOS plus climax/absorption; low liquidity, volume and wick-exhaustion gates. | A downward break and high activity do not themselves establish reversal permission. | Require a distinct reversal thesis before further research; preserve the original identity as comparator. |
| 12 | `retest_cluster`, long, S | Volume/RSI extreme plus permissive fib-cluster predicate; volume, RSI and temporal-score gates. | No specific retested level; absent cluster rejects but null/invalid values may bypass. Temporal score is not a validated time-cycle observation. | Park until object/anchor and missing-value semantics are explicit. |
| 13 | `funding_divergence`, long, S | Negative raw funding or funding-Z extreme; funding/OI/crowding gates mostly skip missing values. | A persistent crowded state lacks entry timing; constant funding Z and source clocks compromise replay. | Data-blocked for economic ranking; retain crowding as a later context hypothesis. |
| 14 | `oi_divergence`, long, S | Structural identity defaults to true unless optional price-extreme switch is set; declining OI, RSI, volume and taker gates. | No mandatory price/OI divergence object; missing/defaulted OI can masquerade as evidence. | Data and identity work before profit comparisons. |
| 15 | `long_squeeze`, short, S | Positive funding/raw or Z extreme; elevated RSI, crowding/OI/vol-shock/context gates. | Historical input gaps plus no short support in the current isolated research wrapper. | Park pending reliable derivatives and separately verified short execution; never relabel as a long dip-buy setup. |
| 16 | `volume_fade_chop`, neutral, S | Low volume and ADX, RSI extreme and effort/result gates. | Generic detector returns no signal for neutral direction; range side and boundary are unspecified. | Definition-blocked, not an observed zero-edge strategy. |
| 17 | `whipsaw`, neutral, S | Upper wick >2 bodies plus weakness or high volume; YAML asks for low volume and weakness. | Generic neutral direction cannot emit; high-volume identity alternative conflicts with low-volume gate. | Definition-blocked. Specify short rejection versus two-sided range fade before testing. |

Ranks 4–12 are an explicitly judgmental queue, not finely measured differences.
Input readiness and repair size break ties; old profit factors do not.

### Effective parameters and runtime boundaries

Champion structural overrides matter. Current values include BOS proximity
`2.9912363698443576` ATR for OBR; sweep wick fraction
`.38184757169438016`; K/H wick fraction `.4910969436357509`;
ER lower RSI `24.528581875434863` with oversold-only enabled; retest-cluster
RSI bounds `32.968023961114554`/`65.68071845610109` and volume Z
`1.2794004516616884`. Do not reproduce the identity using its fallback constants
or similarly named YAML quality thresholds. Exact values remain bound by the
source bundle below, not recommendations to optimize them.

Missing configured skips are excluded from the gate denominator; all skipped
gates return pass. Soft failures reduce fusion. An inner detector floor remains;
the runner's outer bypass is not universal removal of selection. Below that
outer threshold the runner usually enforces failed gates, with CB opting out;
above it a different branch admits the signal. Dedup and cooldown can change
which archetype owns the entry. Independent books alone do not isolate those
upstream effects.

The structural bridge catches exceptions and returns `(True, 'error:...')`.
This source behavior was inspected, not assigned a historical loss count.
The new research eligibility contract must distinguish errors from affirmative
structural evidence, while retaining the unchanged native diagnostic.

Runner exit initialization calls `create_default_exit_config()` and overlays
the champion `exit_logic` dictionary. Per-archetype YAML exit text is not
automatically the operative exit schedule. The current isolated wrapper is
long-only and does not reproduce full native scale-outs/management.

## A concrete cross-archetype representation problem

`LiveFeatureComputer._fusion_scores` uses absolute RSI displacement and
`bullish_BOS or bearish_BOS` when constructing `tf4h_fusion_score`. It then sets
`tf1d_fusion_score` equal to that 4H value. These particular fields measure
neither independent daily agreement nor a signed higher-timeframe direction.
Other actual daily/4H Wyckoff features exist; this is not a claim that the
entire multi-timeframe engine is duplicated.

Three read-only synthetic checks reproduced the consequence using the original
function body and current identity method:

- RSI 70 with bullish 4H BOS and RSI 30 with bearish 4H BOS both produce 0.475
  when ADX=20 and liquidity=.2; only direction was mirrored.
- Daily fusion equals 4H fusion for that calculation.
- RSI 50, ADX=20, liquidity=.2 and no 4H BOS produce approximately .105.
  A large lower-wick TWT fixture below its EMA passes `_check_H` through that
  low-score fallback, using current champion parameters. This is an identity
  witness, not proof that the whole engine would allocate a trade.

Therefore the earlier wording "bearish fusion fallback" needs qualification:
the consumer treats it as directional, but the producer does not encode direction.
R1 tests removing this inference. A successful software repair still needs an
economic comparison; rejection count is not profit.

## Shared mechanisms rather than 17 votes

This map is a research synthesis of the code and stored teachings, not a claim
that the traders prescribed this exact taxonomy. Membership can overlap.

| Mechanism | Archetypes carrying pieces | Shared facts to record | Question a score cannot answer |
|---|---|---|---|
| Rejection and reclaim | Sweep, spring, wick trap, TWT, failed continuation, vacuum, downside LC, ER | Pre-existing level ID, sweep time, close reclaim, expiry, higher-range location | Did the same known level actually fail to break, or is there just a large wick? |
| Expansion and retest | Upside LC, CB, OBR, FVG continuation | Child compression bounds, directional break ID, first retest, gap/zone lifecycle | Is this continuation of the same move, or an unrelated nearby break/gap? |
| Crowding release | Funding divergence, long squeeze, OI divergence | Observed funding/OI/flow with units and arrival clocks, then a price trigger | Has crowded positioning begun to unwind, or has it merely persisted? |
| Range fade and timed location | Volume fade chop, whipsaw, retest cluster | Explicit side, fixed range, repeated touches, confirmed price/time anchors | Where does the range trade cease to apply, and what numerical timing claim is being tested? |

The useful "pattern within the pattern" is a relationship between **location,
sequence and horizon**, not a count of similarly derived indicators. A common
parent/level identifier lets us learn across archetypes without sharing book
capacity or pretending their correlated BTC trades are independent samples.

The [primary-source ledger](trader_primary_source_ledger_2026_09_12.md) supports
level-aware execution and distinct timeframe roles within its stated reading
limits. Numerical windows and thresholds in the new specification are project
hypotheses. The [Fibonacci map](fibonacci_price_time_map_2026_09_12.md) documents
anchor/consumer mismatches; restoring or adding ratios is not part of this budget.
Older [archaeology](founding_knowledge_archaeology_2026_07_17.md) is an idea index,
not current proof that every item marked "FULL" is faithful or that temporal
features are still dormant.

## Data and exposure inventory

Direct metadata and selected-column checks in this turn:

| Input | Verified local scope | Research use and restriction |
|---|---|---|
| Recovered Binance USD-M BTCUSDT minute archive | 2,979,360 rows, Jan 1 2021 through Aug 31 2026 23:59 UTC; exact previously bound hash | Primary common price stream. Derive complete 5m/1H/4H/daily bars; do not mix Coinbase or CME prices into its fill path. Acquisition record is `data/recovered_binance_minute_2026_09_14/recovery.json`. |
| Parent checkout V23 parity store | 74,436 rows, 364 Arrow schema fields, Mar 1 2018 through Aug 30 2026 00:00 UTC; exact audit hash | Legacy forensic comparison, not automatic native-input parity. Unique ordered index has 61 missing hours: 23 in 2024, five in 2025 and five in 2026. |
| Parent checkout derivatives hourly cache | 50,617 rows; Sep 1 2020 through Jun 11 2026 00:00 UTC | Potential source only; same-instrument availability/normalization still needs qualification. |
| Parent checkout macro daily cache | 2,274 rows; Jun 1 2017 through Jun 11 2026 | Date-stamped daily values do not establish publication or historical receipt times. |
| Parent checkout CME minute archive | 2,633,326 rows; Jan 3 2021 23:02 through Aug 18 2026 23:57 UTC | Different instrument, sessions and contracts. Not an extension of Binance history. |
| Saved LC monthly sources | All 31 source-file hashes matched controller receipts; 142 candidates, Jan 2024–Jul 2026 | They save **LC-only candidate projections**, not the complete all-17 hourly opportunity population. They cannot supply the proposed all-17 census without a new export. |

V23 selected-field findings, reconfirmed on the hash-matching file:

- `funding_Z`: 74,436 non-null observations, one value, 0.0.
- `effort_result_ratio` and `absorption_flag`: each 59,901 non-null observations,
  none in 2025 or 2026. This changes evidence, not just sample size.
- `vol_shock`: absent column.
- `alt_basket_ret_4h` and `stables_rot_rising`: each 3,859 non-null values out
  of 5,780 stored 2026 rows; exodus filters cannot be treated as continuously
  observed that year.
- Raw funding, OI changes, BOS/FVG flags and fusion fields do vary. Existence
  and variability are not validation of units, causal alignment or live parity.

Prior exposure is extensive. Both 2024–July 2026 LC and August replication
have been examined; the separate minute sweep history 2021–August 2026 was
already replayed. Older V23/graduate studies also exposed recent eras. There is
no newly certified untouched local interval in this inventory. Chronological
historical blocks can diagnose robustness, but cannot be renamed fresh holdouts.

## Work not to repeat or silently supersede

- The [room-filter comparison](lc_room_validation_results_2026_09_30.md) did
  not justify a mandatory gate. Keep the field; do not tune it to rescue known
  missed winners.
- The old minute headline was noncausal. The corrected equal-low study was
  strongly negative after costs, and negative before fees in aggregate. See
  [timing results](execution_timing_and_parent_recovery_2026_09_10.md).
  R3 is a different, explicitly named minute hypothesis, not that result revived.
- Related **hourly** level repairs also have negative prior evidence. The
  [July level study](unified_strategy_verdict_2026_07_13.md) reports only three
  holdout trades for its wick-trap sweep gate; the
  [September study's origin note](sweep_native_scalper_2026_09_09.md) and saved
  identity-restoration memory report an hourly liquidity-sweep restoration
  starved to 14 cases with negative fresh results. These are different historical
  contracts; the exact September restoration implementation was not recovered
  in this turn, so equivalence to the proposed prior-24-hour-low rule is not
  asserted. Nevertheless, the idea is not fresh enough to justify another run
  merely by changing the window. **R2 is parked before implementation/scoring.**
  The old claim that the remaining proxy therefore has a proven edge is not
  adopted. This changes the shortlist, not the preserved historical reports.
- One-hour prior BB compression was already specified and diagnosed in
  [candidate contracts](candidate_rule_contracts_2026_09_10.md). Do not present
  it as a new LC discovery.
- Old CPCV/"graduate" labels do not override later weakness. The
  [August battery](fresh_data_battery_2026_08_30.md) includes failed fresh-period
  flow boosts and a losing graduate book. Its book-level stand-down finding is
  not an isolated-archetype timing rule or proof that entry repair is pointless.

## Verification and reproducibility

No new market assessments, source-month engine replay, trade-outcome scoring,
network data download, production edit or dependency install occurred. Read-only
probes checked the roster, selected data coverage, three synthetic semantics
witnesses and 31 saved LC source hashes. Additionally, 43 existing focused tests
passed in the final rerun in 2.06 seconds across trader-intent witnesses, signal replay and minute
sweep modules (one existing LibreSSL warning). This is not a repo-wide pass or
a test of unimplemented R1/R2/R3 behavior. No historical all-17 census is claimed.

One bounded design review identified ambiguous R3 rearming/population and five
important contracts concerning cooldown, opportunity IDs, clocks, funding/risk
and advancement. The specification now uses an arm-neutral box census, explicit
source opportunity IDs, independent signal-time cooldown, exact clock/accounting
rules and an ordered decision table. These are controller-checked design fixes;
the original review is not a post-fix implementation or profitability approval.
The controller's final history check then parked R2; the reviewer did not review
that later scope reduction. There are two active hypotheses and no replacement.

Source bundle: 29 files; SHA256
`3166af1baff7121a46535d39c83f11d8b26a58c21b8321acc32dff666924de59`.
Construction: lexicographically sort the following repo-relative paths, concatenate
each path, NUL, its lowercase file SHA256 and newline; hash the resulting UTF-8.
Files are the 17 named non-example immediate champion YAMLs, plus:

```text
configs/champion_paper.json
engine/archetypes/logic.py
engine/archetypes/structural_check.py
engine/archetypes/archetype_instance.py
engine/archetypes/exit_logic.py
engine/integrations/isolated_archetype_engine.py
bin/live/live_feature_computer.py
bin/live/v11_shadow_runner.py
scripts/research/engine_signal_replay.py
scripts/research/minute_sweep_validation.py
scripts/research/causal_parent_ledger.py
scripts/research/conditional_occupancy.py
```

Minute archive SHA256:
`5b8a4533f70b8ccd0dc533984469e6886ec73ce315d48f150c667ccb97e84035`.
V23 store SHA256:
`0e8604d0dacf7435e42e9759fe64aca15bfa1e3d66f14c1d21c142f41d1ab61f`.
Both hashes were freshly recomputed. These data and source receipts are local-only;
GitHub alone does not provide reproduction inputs.
