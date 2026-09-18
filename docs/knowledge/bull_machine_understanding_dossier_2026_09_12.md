# Bull Machine understanding dossier

September 12, 2026 · version 1.0 · research/design document

This dossier organizes the system understanding needed by a future master agent:
what Bull Machine is trying to recognize, how its current implementation differs,
and which questions specialists should answer. It is not a deployed master,
whole-corpus mastery, model training, or a new trading result. Selected local
sources were checked at checkout HEAD `a31043b41faae4bdf087c80f9e7626240b13cf8b`;
that is not a fingerprint of the running server. No production behavior changes.

## 1. Founding thesis and evidence boundaries

The useful organizing thesis is **a local trade must make sense inside an
identified larger structure, complete its own sequence, and have a destination
and invalidation appropriate to its horizon**. This is our synthesis of the
project's intended design, not a recovered universal teacher rule.

Wyckoff supplies the backbone: ranges, phases, tests, springs/upthrusts,
strength/weakness and effort versus result provide a language for a developing
auction rather than isolated indicator readings. Code contains a state machine
and event structures ([events.py](../../engine/wyckoff/events.py#L115)); current
archetype fusion consumes direction-aware 1H/4H/1D scores
([archetype_instance.py](../../engine/archetypes/archetype_instance.py#L384)).
Those are different levels of representation: a high score cannot itself prove
that the right range, event order and target existed.

The other schools retain their identities. SMC contributes breaks, gaps and
retests; the local Bojan/wick notes describe trap and level behavior; liquidity
sweep work emphasizes identified pools and reclaim; derivatives families concern
crowding/unwinds. Moneytaur and Wyckoff_Insider are contemporary attributed
sources, not interchangeable authors or automatic proof of original Wyckoff doctrine.

The newly opened [Moneytaur August 3, 2024 post](https://x.com/Moneytaur_/status/1819660778359644193)
supports waiting at levels, distinguishing execution horizons, and planning a
destination within the relevant range. Its higher-timeframe confirmation language
has a particular opportunity scope; it does not establish an all-archetype veto.
The [access note](moneytaur_primary_sources_2026_09_12.md) records a damaged passage,
excerpt-only sources and access limits. Original authorship does not establish edge.

Maintain five source classes: **primary teaching; attributed secondary notes;
executable code/config; measured experiment; project hypothesis**. Do not promote
comments such as “institutional capitulation” into measured order-flow facts.
Conflicting accounts of HTF sizing remain unresolved. Exact ratios, tolerances,
stops and target formulas require their own provenance. A newly inspected
[Wyckoff_Insider July 27, 2025 post](https://x.com/Wyckoff_Insider/status/1949423128217633091)
explicitly mentions Fibonacci price ratios and time cycles. This corrects any
blanket claim that every number originated in the project: some ratios are
teacher-mentioned, while their selected anchors, timing windows, tolerances and
executable use remain separate project choices. Broad assertions about algorithms
obeying those ratios remain the author's claims, not established market facts.

Two more original sources sharpen the design. Wyckoff_Insider's
[timeframe/journal teaching](https://x.com/Wyckoff_Insider/status/2000223306620846408)
separates HTF analysis, MTF context and LTF execution, with entry-time reasoning
followed by taken/missed trade review. Bojan's
[weekly-wick idea](https://x.com/Bojan_618/status/2000564037780946977) is regime-dependent;
his [qualifying reply](https://x.com/Bojan_618/status/2000564041102901639) warns that
a plausible later destination does not justify entering immediately. Neither
post supplies the engine's numerical gates or proves a profitable setup.

Companion research: the [trader primary-source ledger](trader_primary_source_ledger_2026_09_12.md)
records exactly what was read and what remains missing; the
[Fibonacci price/time map](fibonacci_price_time_map_2026_09_12.md) traces legacy
and current-code formulas, consumers and limitations.

## 2. Why seventeen: implementation catalog, not complete taxonomy

The champion selects `configs/champion/archetypes_v14rq/`
([config](../../configs/champion_paper.json#L14)). Its 17 enabled definitions are
the historical catalog to preserve for comparison; no inspected source establishes
17 as optimal, exhaustive, or a single teacher's taxonomy. Fourteen are long,
`long_squeeze` is short, and two are neutral. The generic detector rejects neutral
directions ([branch](../../engine/archetypes/archetype_instance.py#L877)); enabled
does not mean executable. Extra dormant checks and research candidates do not
silently enlarge or replace this roster.
The champion JSON still contains stale “all 16 archetypes” descriptions
([metadata](../../configs/champion_paper.json#L3)); the verified enabled-YAML roster
is seventeen. That internal documentation mismatch is not a different taxonomy.

The atlas below summarizes native **identity**, not every YAML gate or the final
entry decision. Numeric checks are implementation choices. Detailed gate/source
comparisons remain in the [translation audit](archetype_translation_audit_2026_09_09.md).
Each code link points to the inspected identity function.

| Archetype | Intended family | Current identity anchor | Principal unresolved translation |
|---|---|---|---|
| spring | Wyckoff spring/bear trap | [A](../../engine/archetypes/logic.py#L544): bullish PTI trap by default | Preserve range/level/reclaim identity and lifecycle behind the label. |
| order_block_retest | SMC/HOB retest | [B](../../engine/archetypes/logic.py#L566): near a recent bullish or bearish BOS close | No required order-block candle, zone, freshness or direction linkage in identity. |
| fvg_continuation | Gap continuation | [C](../../engine/archetypes/logic.py#L596): 1H/4H FVG plus recent BOS of either direction | Bind gap, break and long direction to the same structure. |
| failed_continuation | Failed move; effort/result | [D](../../engine/archetypes/logic.py#L627): FVG, falling ADX, no high-volume confirmation | Weakening ADX does not establish failed break and reclaim. |
| liquidity_compression | Climax/absorption reversal | [E](../../engine/archetypes/logic.py#L662): climax/absorption or high volume with two-sided RSI extreme | Current-bar proxies do not prove preceding compression, structural location or long thesis. |
| exhaustion_reversal | Exhaustion reversal | [F](../../engine/archetypes/logic.py#L698): RSI extreme; champion enables oversold-only override | An oversold reading does not prove reversal completion; preserve historical rule versions. |
| liquidity_sweep | Liquidity rejection/reclaim | [G](../../engine/archetypes/logic.py#L720): dominant lower wick | Prior identified level and ordered multi-bar reclaim are not required. |
| trap_within_trend | Wick trap in trend | [H](../../engine/archetypes/logic.py#L758): wick, ADX, EMA or HTF-fusion context | Bearish fusion fallback can qualify a long despite absent upward EMA alignment. |
| wick_trap | Bojan-style wick/trap proxy | [K](../../engine/archetypes/logic.py#L813): wick anomaly; optional exodus refusal | Persistent magnet, trap reset and direction sequence remain distinct concepts. |
| retest_cluster | Exhaustion at temporal cluster | [L](../../engine/archetypes/logic.py#L834): volume/RSI extremes; finite numeric cluster values must be positive | No identified retested level; null/invalid cluster can bypass while absent key rejects. |
| confluence_breakout | Compression/value-area breakout | [M](../../engine/archetypes/logic.py#L866): low ATR percentile, near POC, BOMS confirmation by default | Proximity to value does not alone establish directional opportunity or room. |
| liquidity_vacuum | Capitulation reversal | [S1](../../engine/archetypes/logic.py#L915): bearish BOS plus climax/absorption; configured long | Explain reversal after the break rather than infer it from capitulation. |
| whipsaw | Failed upside break/weakness | [S3](../../engine/archetypes/logic.py#L987): upper wick over twice body plus SOW or climax | Neutral/nonemitting; execution direction and low-volume gate relationship need explicit design. |
| funding_divergence | Crowded-short squeeze | [S4](../../engine/archetypes/logic.py#L1021): negative raw funding or funding-Z extreme | Crowding can persist; observe timed reversal and distinguish source/units. |
| long_squeeze | Crowded-long unwind, short | [S5](../../engine/archetypes/logic.py#L1045): positive raw funding or funding-Z extreme | Crowding state is not the unwind trigger; funding/OI coverage matters. |
| volume_fade_chop | Low-volume range fade | [S8](../../engine/archetypes/logic.py#L1137): low volume and ADX | Neutral/nonemitting; range boundaries, side and invalidation are unspecified. |
| oi_divergence | OI capitulation/price divergence | [S11](../../engine/archetypes/logic.py#L1157): optional price-extreme check; otherwise true | Native identity need not establish structural price/OI divergence; gates carry selection. |

## 3. Fusion: preserve the purpose, expose the decisions

Legacy [FusionEngine](../../engine/fusion.py#L85) explicitly combines domains with
macro context, vetoes and explanations. It supports the recollection of an
all-context assessor, but is not the present entry path.

The inspected active flow is:

`identity → cooldown/gates → technical fusion and inner floor → dedup → runner context/outer branch → allocation/execution`

[Technical fusion](../../engine/archetypes/archetype_instance.py#L245) weights
Wyckoff, liquidity, momentum and SMC, then applies fakeout/PTI and other penalties.
Soft gates scale the result; the default fusion mode retains an inner threshold
([detection](../../engine/archetypes/archetype_instance.py#L822)).
[Dedup](../../engine/integrations/isolated_archetype_engine.py#L585) chooses maximum
fusion per direction; cooldown can arm even for a subsequently discarded signal.

The [runner](../../bin/live/v11_shadow_runner.py#L1120) applies crisis adjustment
and an outer threshold using risk temperature/instability. Champion bypasses that
outer cutoff, but below-threshold signals encounter additional gate enforcement,
with archetype opt-outs. Above-threshold admission takes another branch. Fusion
therefore still affects who enters and which archetype owns a bar. Its confidence
field is a scaled score, not calibrated probability of profit. See the
[fusion audit](fusion_intent_and_revival_audit_2026_09_10.md) for stage-specific limits.

The desired contextual judgment is: which prerequisites hold, which context
supports or opposes this horizon, and what remains unknown? A strong unrelated
factor cannot supply a missing required event. Softer preferences need not all
become vetoes. Historical book outcomes must account for dedup, cooldown and
displaced opportunities before attributing a gain to better detection.

Price/time already has narrow representation:
[live features](../../bin/live/live_feature_computer.py#L2262) compute a geometric
mean of price/time scores, and OBR/retest-cluster consume temporal features.
This is implementation evidence, not proof that Fibonacci timing predicts returns.
Confirmed anchor identity, horizon, units and arrival clocks must accompany a ratio.

The code map exposes two concrete representation problems: `_fib_features`
shifts pivot markers 20 bars for confirmation but then reads prices from the
confirmation rows, not the original pivot rows; the live single-row temporal
score receives no event-age fields, so its Fibonacci-cycle component is a fixed
0.30 fallback. A separate batch flag still varies. These are source-inspected
mismatches, not measured causes of live losses or proof that all timing logic is
inactive. Old notes report useful book-level gate suppression but an unsuccessful
price/time boost; neither result authenticates the feature's intended meaning.

## 4. Two known educational cases

These reuse previously assessed evidence; they are not new blinded observations,
new outcomes or demonstrations of mastery. Values and case IDs were checked against
the existing reports/private inputs. Neither example proves an executed trade.

**Hourly — `hourly-lc-real-001`.** Source hour June 14, 2026, 21:00 UTC;
decision 22:00. Volume-z 4.225450, RSI 71.426069, BB width 0.022398 and chop
0.434101 pass the supplied LC checks. Previous completed-hour width 0.017620
satisfies the separately proposed prior-compression test. Signal prices order
correctly: stop 64,377.5113 < entry 65,280.67 < target 67,020.0868.

An informed explanation separates three things: numerical eligibility; a
directional reversal thesis despite high RSI; and an executable trade with room.
The packet supplies the first, but lacks identified larger structure, field-level
receipt/formula provenance and complete management. “Unresolved” describes that
packet, not the absence of such capabilities everywhere. The nominal 1.925926
reward/risk excludes costs and does not prove target reachability.
[Original case report](assessment_real_case_2026_09_11.md).

**Minute — `minute-first-june-window-4h-n3`.** June 10 sweep 00:36, reclaim
00:37, decision 00:38 UTC. The bound 4H/N3 parent was available June 9 at 17:00.
Its range was 61,150.20–64,179.50; child level 61,519.00 lies below midpoint
62,664.85. Sweep low 61,428.60 breached the child level; reclaim 61,551.60
closed back above it and inside the parent. The parent floor was never breached.

Thus the stipulated structural permission passes without requiring a parent-floor
sweep. Continuity holds under the supplied hourly ledger contract, not proven
intraminute receipt history. Missing execution/receipt evidence keeps the overall
assessment unresolved; the parent high is not automatically a target. This minute
detector is separate research logic, not hourly LC renamed. See
[case report](assessment_minute_case_2026_09_11.md) and
[detector](../../scripts/research/minute_sweep_validation.py#L36).

## 5. Master and specialists: proposed architecture

The master maintains one versioned structural account and reconciles competing
theses. Specialist rulecards preserve setup-specific context, sequence, location,
confirmation, cancellation, invalidation, target/management and unknown states.
Run relevant specialists on a shared snapshot; seventeen roles need not mean
seventeen calls per bar. Higher-timeframe context can produce an expiring
conditional plan ahead of minute triggers.

Persistent memory is source-linked files/records: source claims, code mappings,
experiment results, corrections and hypotheses, each with version and supersession.
Future sessions retrieve them; weights have not thereby learned Bull Machine.
Each assessment pins memory, prompt/model, code/config and evidence versions.
Snapshot fields retain source, units/formula, observed/defaulted/missing state,
event/available/receipt times and dependencies. A hash establishes identity, not
authentic receipt or semantic truth.

Specialists return supporting and opposing evidence, unresolved prerequisites and
a conditional plan. The master resolves cross-timeframe level/horizon conflicts
and shared exposure; a critic challenges contested claims. A bearish parent can
contain a coherent brief rebound. Agent agreement does not create independent
evidence or override deterministic clocks, arithmetic, hard gates and risk limits.
The initial role has no order, sizing, stop or exit authority. The fuller
[agentification design](agentification_design_2026_09_11.md) remains a proposal.

## 6. Three provisional hypotheses, never automatic promotion

| Candidate | Mechanism and required observations | Counterexample / falsifier |
|---|---|---|
| Parent-contained sweep with explicit destination | Refine the existing minute family using a prebound range, identified child pool, sweep/reclaim sequence, predefined destination and net room after costs. Requires same-stream causal parents/levels and executable timing. | Attractive lower-half reclaim facing nearby supply; reject the hypothesis if frozen comparisons do not improve net expectancy over identical sweep candidates after costs/displacement. |
| Structure-anchored price/time retest | Refine retest-cluster/OBR: freeze an observable pivot/zone and timing projection before price revisits it; test interaction of price location and timing rather than time alone. Requires anchor confirmation clocks, projection units, zone lifecycle and confirmation event. | A fit appearing only after moving anchors; no advantage over matched price-only and time-only controls after trial accounting invalidates the claimed interaction. |
| Crowding release with failed structural continuation | Refine funding/OI families: crowded state followed by failed continuation and reclaim/break in the trade direction. Requires timestamped funding/OI, price-level identity, ordered release/confirmation and fixed exit rules. | Crowding persists while trend continues; no incremental value beyond structural confirmation, or gains dependent on revised/missing derivatives data, defeats the proposed mechanism. |

These are project proposals, not recovered teacher systems or claims of novelty
over all prior research. Definitions and tolerances must be frozen before testing;
retain losses, nonselections and all attempted variants. Overlapping timeframes,
parent lineages and BTC trades are correlated. Known cases cannot become pristine
holdouts; model pretraining can also contaminate history. Later forward paper
evidence is needed, with inference latency/cost and deterministic baselines.

## 7. Finite next master assignment

The source-linked system dossier is the first deliverable; do not repeat it as
another generic comprehension milestone. The next bounded assignment is exactly
**two specialist rulecards**: hourly liquidity compression and the separate minute
parent-contained sweep research family. Each card must distinguish native rules
from a proposed faithful trader interpretation, explain required evidence and
hard/soft/unknown conditions, and cite its sources. The minute family is not a
quiet replacement for one of the seventeen champion definitions.

Apply each card to its corresponding known educational case above as a conditional
plan: supporting/opposing evidence, unmet prerequisites, entry trigger or waiting
condition, invalidation, destination and management unknowns. Never manufacture
missing levels to complete a template. Grade source/code/hypothesis separation and
factual correctness, not the known trading outcome. This tests whether the master
can brief useful specialists; it is not fresh market validation or weight training.

After those two artifacts lock, separately freeze a fresh real-case comparison
using the newest [MEMORY](MEMORY.md) protocol. Preserve equal hourly/minute standing,
same-evidence deterministic baselines, rejected opportunities, costs and causal
clocks. The aim is to measure whether agent judgment improves selection, not merely
whether it can narrate a chart. No new runtime, backtest, market call or live
promotion was performed for this dossier.

## Review and verification

The core draft used a requested `gpt-6-astra`/high agent; a parallel code-source
audit and an independent quant reviewer supported root integration. This describes
requested presets, not a verified runtime model snapshot. Review approved local
research handoff after clarifying the permissive cluster predicate and stale
sixteen-archetype metadata. The reviewer independently checked key code paths but
could not reproduce the coordinating agent's native-X reads; the source ledger
retains that access scope. Relative links/line bounds, whitespace and the exact
17-row enabled-YAML roster were checked. No code changed or backtests were rerun.
