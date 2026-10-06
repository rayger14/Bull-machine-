# M1/M2 and Wyckoff concept-fidelity code audit

Date: October 4, 2026. Branch: `quant/archetype-evidence-audit`.
Inspected HEAD: `85923a4501153271b2e2e1d1437dc52ec4cb778e`, with the existing
dirty research worktree preserved. This is an audit, not a strategy repair or
another economic experiment.

## Decision

The repository contains real event detectors and a sequencer, but not one
consistently named, end-to-end verified M1/M2 implementation. There are objective
integration defects as well as incomplete interpretations of the concepts.
Fixing these is necessary before claiming fidelity; it does not establish an edge.

The first repair should address evidence identity, direction and timing in the
currently selected path. Do not enable full M2, change thresholds, rewrite all
17 archetypes, or resurrect legacy modules merely because their names sound
more advanced. No implementation or live configuration was changed by this audit.

Three bounded reviews covered the legacy implementations, current event engine,
and selected-path integration. The controller inspected source and independently
reproduced the principal findings below. Synthetic counterexamples establish
that a behavior is possible, not its frequency or financial impact.

## Scope and meaning of the names

Here M1/M2 refers to the project's schematic terminology, not one-minute/two-minute
bars. Three incompatible usages coexist:

| Location | Meaning of M1 / M2 | Current role |
| --- | --- | --- |
| `engine/wyckoff/events.py` and `tests/test_wyckoff_m2_sequence.py` | Spring-based accumulation / accumulation without a spring | Current event engine; full M2 path defaults off |
| `bull_machine/strategy/wyckoff_m1m2.py` | Scalar spring score / markup score, with a helper also calling M2 markdown | Legacy/offline callers found |
| `bin/live/live_feature_computer.py:1833` | Daily bullish score above 0.1 / daily bearish score above 0.1 | Compatibility flags, not schematic detections |

The selected local route is:

`coinbase_runner -> LiveFeatureComputer -> engine.wyckoff.events`

`v11_shadow_runner -> integrations.IsolatedArchetypeEngine -> ArchetypeInstance -> structural_check/logic`

The selector is `configs/champion_paper.json:14`, pointing to
`configs/champion/archetypes_v14rq/`. This is a local source/config trace, not a
verification of what the remote server is currently running.

Reviewed: the complete current event module; relevant live feature, runner,
fusion, structural and selected-config consumers; the legacy M1/M2 scorer,
v160/v151 caller boundary, advanced/state-machine/analyzer/phase/MTF alternatives;
five focused test modules and related legacy tests. The existing graph supplied
navigation only; its missing/stale coverage was checked against actual source.

Not covered as a fresh full audit: every concept in all 17 archetypes, Fibonacci,
Gann, execution/venue accounting, the entire minute-research stack, deployment
state, an exhaustive trader archive, or historical trade-by-trade attribution.
The recent support-reaction policy is separate from these native detectors.

## Source contract before formulas

The primary reference distinguishes accumulation with and without a spring.
SC/AR/ST help establish a range; a spring breaches established support and
recovers; a test assesses the supply response. Phase C need not contain a spring.
Demand/supply behavior and the role of a move within its range matter. An early
ST can belong to phase A, and a sign of strength can occur during phase B; neither
event name alone determines a phase. Distribution likewise need not include a
UTAD. Reaccumulation need not begin with a new selling climax.
Source: [Wyckoff Analytics — The Wyckoff Method](https://www.wyckoffanalytics.com/wyckoff-method/).

Local provenance: [historical Wyckoff audit](wyckoff_audit.md) and
[October 1 trader source ledger](../../research_notes/Trader%20origins%20and%20archetype%20fidelity/original_trader_sources.md).
The historical audit explicitly corrected its earlier claim that M2 could not
be mechanized and records why the full sequencer was parked. Its old performance
headlines were not revalidated here. The ledger supports separating larger
context from lower-timeframe execution; it does not authenticate every code
threshold or a universal requirement for all indicators to agree.

## A. Selected-path defects and interpretation gaps

Priority below means repair priority for trustworthy evidence, not estimated P&L.

### A1 — High: opposite-direction and proxy evidence can become positive Wyckoff evidence

Source: [ArchetypeInstance](../../engine/archetypes/archetype_instance.py),
lines 384–458; [live features](../../bin/live/live_feature_computer.py), 1955–1980.

The scorer first uses directional evidence. If the desired side is zero, it
falls back to a non-directional maximum even when explicit directional columns
exist and show the opposite side. The comment says this fallback is only for
missing directional columns, but the code tests values, not availability.

Controller witness: all bullish scores zero, bearish scores 0.8 and generic
confidence 0.8 give a **long Wyckoff score of 0.8**. A generic 4H phase score of
1.0 gives **1.0 to both long and short**. The live fallback can populate that
field from bullish EMA alignment with all directional Wyckoff scores zero.
There is no explicit proxy/unavailable flag in that fallback result.

Required invariant: unavailable, neutral, conflicting, and confirmed evidence
must remain distinguishable. EMA alignment is not a detected Wyckoff structure.

### A2 — High: delayed events lose candidate geometry; rejected springs can advance phase

Source: [events](../../engine/wyckoff/events.py), 223–245, 314–336,
1098–1104, 1168–1173, 1239–1245, 1543–1564.

Raw spring/upthrust detectors correctly wait for later confirmation, but emit
only a Boolean/confidence on that later bar. The sequencer then checks that
confirmation candle's low/high against the parent range, not the original
candidate's extreme.

Controller witnesses with a seeded parent and actual raw detector:

- SC support 100; candidate low 98; confirmation low 103: raw spring true,
  sequencer spring false.
- BC resistance 110; candidate high 113; confirmation high 107: raw upthrust
  true, sequencer upthrust false.

A second, independent defect: `ACCUM_SPRING` is assigned outside the specific
event-acceptance branch. Raw Spring A at low 100.5 against SC 100 passes the
1% proximity wrapper but not the actual sweep. The event is false, yet the
state becomes `accum_spring` and phase becomes `C_accum`.

Required invariant: retain candidate time/extreme, confirmation/availability
time, swept level and parent identity. Rejected events cannot change phase.
Do not solve this by backdating information to before confirmation.

### A3 — High: relative-volume and range-lifecycle contracts are inconsistent

Source: [events](../../engine/wyckoff/events.py), 189–215, 296–300,
370–381, 1543–1557.

`sc_volume`/`bc_volume` store a rolling volume z-score. The secondary-test and
invalidation checks multiply that z-score by a ratio, despite describing a
comparison with climax volume. The adapter does not pass raw volume. A ratio
of separately standardized z-scores is not a ratio of traded volume.

Controller witness: SC actual volume 1,000/z-score 3, then a proposed ST actual
volume 5,000/z-score 0: ST accepted; stored SC volume is 3. This isolates the
sequencer and does not claim the entire raw detector accepts every such candle.

Additional lifecycle gaps from source review: ST validation does not itself
check that the test revisits the referenced SC/AR range. Invalidation tests a
close beyond the climax boundary and a z-score ratio before processing new
events. It does not separately encode a temporary excursion, failed recovery,
and sustained acceptance outside the range. Quiet undercuts can be legitimate;
the repair must specify those cases rather than invalidate every breach.

### A4 — High interpretation risk: ambiguous climaxes can reset context; V2 is shadow only

Source: [events](../../engine/wyckoff/events.py), `detect_buying_climax` around
634, sequencer 184–196/290–303, V2 integration 1788–1794;
[V2 tests](../../tests/test_wyckoff_v2_climax.py).

The active BC rule can label a wide, high-volume upward expansion as a buying
climax without established reversal evidence. Any accepted BC starts a new
distribution reference. In the existing deterministic continued-upward-breakout
fixture, a full `detect_all_wyckoff_events` run produced BC=true, SOS=true,
BC_v2=false, and context/phase `distribution / A_distrib` on the spike.

This does not mean strong-closing bars can never be climaxes. It means the
active rule can turn ambiguous bar evidence into a definitive range reset.
V2 adds reversal-aware shadow columns but does not replace the active event
used by the sequencer. Turning V2 on as the new detector requires separate
validation of delayed availability and both positive and negative examples.

### A5 — High interpretation risk: full M2 is disabled and incomplete; its phase can still affect sizing

Source: [events](../../engine/wyckoff/events.py), 258–286, 350–365,
383–402, 1295–1342; [live features](../../bin/live/live_feature_computer.py),
1646–1652; [runner](../../bin/live/v11_shadow_runner.py), 1458–1467;
[selected config](../../configs/champion_paper.json), 303–305.

- Full `sm_m2_path` defaults false. Only the hourly live detector opts into
  `sm_m2_context_only`; the inspected 4H/daily dictionaries do not.
- "Context-only" means no new sequencer event, not no economic effect.
  The selected config enables a 1.25x sizing/capex multiplier for long intents
  labeled `C_accum`. That label also includes spring-based C, not just M2.
  A2's false phase can reach this consumer; actual live occurrence is unmeasured.
- The five M2 tests inject event labels. Their higher-low example at 103 above
  a recent 99 low is 4.04% away, while the raw LPS rule requires less than 3%.
  Controller reproduced raw LPS=false. Passing the sequencer test therefore
  does not prove raw candles can produce the tested sequence.
- `DISTRIB_ST` is declared but never entered. The no-upthrust distribution
  path cannot reach pre-SOW LPSY through ST. The mirror test supplies an UT;
  it is not a test of a no-UT schematic.
- The full accumulation M2 branch also accepts `ACCUM_SPRING`; a distinct
  "without spring" identity is not preserved by that branch alone.
- Phase is a fixed mapping from the latest event state. Controller's
  ST -> LPS_C -> SOS -> BU sequence reports **A -> C -> B -> D**.
  No phase-E state is emitted. This cannot distinguish an early ST or a
  phase-B SOS from analogous events later in a developed structure.

Required change in claims: partial sequence support is not a complete A–E
schematic classifier, and "shadow" does not imply orders/sizing are unaffected.
The module header advertises 18 events; the actual entrypoint implements 13.
The advertised ST_BC, shakeout/terminal-shakeout and explicit markup/markdown
continuation events are not implemented there. Preliminary support/supply and
separate reaccumulation/redistribution hypotheses are also not modeled as
distinct sequencer events/paths.

### A6 — Medium: UT and UTAD are duplicate evidence rewarded as diversity

Source: [events](../../engine/wyckoff/events.py), 1265–1288, 1401, 1700–1707.

UTAD reuses exactly the UT detection Boolean, adding only an optional RSI
confidence bonus. It does not add a distinct distribution-lifecycle condition.
Both columns count as separate event types in the diversity score.

Controller witness with the normal confidence-column set: UT confidence 0.8
alone gives bearish score 0.133333; adding the identical UTAD confidence gives
0.533333. One occurrence has been counted as two kinds of evidence. Neither
score is a calibrated probability of a successful trade.

### A7 — High: candle integrity and historical-context time need explicit contracts

Source: [live resampler](../../bin/live/live_feature_computer.py), 2752–2759;
hourly buffer append 779–781; [runner startup/loop](../../bin/live/coinbase_runner.py),
263, 1381–1398, 3230–3241; [HTF modulation](../../engine/wyckoff/events.py), 1637–1660.

Controller ran the actual resampler function on hours 00, 02, 03, 04, omitting
01. It returned an apparently ordinary 00:00 four-hour candle with three hours
of volume and an 04:00 candle with one hour. There is no completeness/closure
flag or expected-count check. Partial daily bars are deliberately included
at live-feature lines 1799–1805. Provisional observations are not automatically
lookahead; treating them as completed confirmations would be a different claim.

Source trace also exposes a conditional startup duplicate: warmup ingests the
latest completed hour but leaves `last_processed_ts=None`; if the first poll
returns the same completed hour, it is processed again and blindly appended.
This path was inspected, not reproduced through an actual network startup.

Finally, the newest HTF context rescales confidence columns throughout the
lower-timeframe buffer. This can be a current as-of reinterpretation, but is
not a per-event historical as-of join. Earlier-row scores must not be treated
as what was available at those earlier times without a separate causal replay.

### A8 — Structural gap: retained maxima are not a nested market-structure model

Source: [context constructor](../../engine/wyckoff/events.py), 1404–1520;
[live carry-forward](../../bin/live/live_feature_computer.py), 1666–1698,
1821–1827, 1858–1877; [fusion](../../engine/archetypes/archetype_instance.py), 413–433.

Hourly evidence uses a 24-bar maximum; 4H context scans the available buffer;
daily context scans up to 90 bars. The context constructor uses confidence
maxima and net dominance, not the current sequencer's parent-range lifecycle.
Controller supplied an earlier SC confidence 0.9 followed by a current
neutral/no-context state: the constructor still returned accumulation/0.9.

Fusion normalizes over positive same-side timeframe scores. Opposing/zero
timeframes do not supply a parent-child relationship or an explicit contradiction
role. A weighted confidence boost is implemented; a linked daily range -> 4H
setup -> minute entry with distinct invalidation and destination is not
established by these functions. This is a design gap, not a request that every
timeframe agree or that recent evidence always be discarded.

The sequencer also deliberately retains raw spring/UT and SOS/SOW events at
reduced confidence when no parent context exists (`events.py:233–256,328–348`).
Those observations may be useful, but "validated event" then does not mean
"confirmed setup within an established parent range." Keep those claims separate.

A separate sizing consumer at `v11_shadow_runner.py:1379–1387` increases size
1.25x when the 4H bearish score is at least 0.6, without checking trade direction.
This is an explicit historical policy, not proof that bearish structure
confirms a long; its old validation claim was not retested in this audit.

### A9 — Integration gap: archetype labels/configuration are not complete Wyckoff contracts

All 17 selected YAML files were inspected for their Wyckoff consumption:

| Wyckoff fusion weight | Selected archetypes |
| --- | --- |
| 0.60 | spring |
| 0.35 | confluence_breakout, liquidity_vacuum, trap_within_trend |
| 0.30 | order_block_retest, liquidity_sweep |
| 0.25 | failed_continuation, exhaustion_reversal, retest_cluster, whipsaw |
| 0.20 | liquidity_compression, funding_divergence, long_squeeze, wick_trap |
| 0.15 | fvg_continuation, oi_divergence, volume_fade_chop |

Seven configure hard gate mode, ten soft. Direct Wyckoff-related YAML gates are
`distribution_at_resistance` for confluence breakout/OI divergence,
`accumulation_at_support` for long squeeze, and `wyckoff_sow` for whipsaw;
these are in soft-mode configurations. Confluence breakout also opts out of
gate enforcement under the outer bypass. A field named `hard_gates` therefore
does not prove that failed conditions always reject a signal.

Specific boundary findings:

- `integrations/isolated_archetype_engine.py:355` stores YAML `thresholds` as
  `pattern_params`. Its consumer in `archetype_instance.py:220` reads the
  cooling period, not every named pattern threshold. For example,
  `liquidity_sweep.yaml:15` saying `wyckoff_phase: B` is not an enforced phase-B
  contract on this route.
- Native spring identity in `engine/archetypes/logic.py:544` is a PTI/trap
  check, not a complete SC/AR/ST/spring lineage.
- `engine/archetypes/structural_check.py:180` returns success on an exception.
  An unavailable structural check can therefore pass. This is source-verified;
  its live frequency was not measured.
- Outer threshold bypass does not erase every inner fusion use:
  `archetype_instance.py:847` retains a threshold in fusion signal mode.
- The two neutral configurations, whipsaw and volume fade chop, reach an
  explicit `return None` for neutral directions at lines 877–879. Configuration
  presence is not proof that all 17 can issue entries through this route.

These are consumer-contract findings, not a full independent audit of every
archetype's liquidity/SMC/momentum logic. No selected configuration requires
a complete M1 or M2 schematic as its standalone setup contract.

## B. Legacy implementations: real defects, but not the selected live path

The bounded import/caller trace found historical scripts, apps, adapters and
tests. Do not attribute these defects to current live trades without additional
deployment evidence.

| ID | Finding and evidence | Classification |
| --- | --- | --- |
| L1 | `wyckoff_m1m2.py:272` evaluates `(df_htf or df_ltf)`. A nonempty DataFrame raises ambiguous-truth ValueError; broad catch at 292 returns zero/neutral. Controller: M1 0.8 without HTF became 0.0 with HTF. | High, legacy objective bug |
| L2 | Inclusive range at `wyckoff_m1m2.py:58` already includes the candidate high. Strict breakout checks at 173–189 cannot be true for valid OHLC. Controller: actual prior-range breakout=true, both internal breakout tests=false, score still 0.6 from other ingredients. | High, legacy objective bug |
| L3 | `wyckoff_m1m2.py:90` scores position, volume and candle behavior without a required prior-level breach/reclaim. Controller: candidate low equals prior low (no sweep), M1=0.8. Older `wyckoff_phase.py:27` instead scores a close below prior support, also without reclaim. | High, concept mismatch |
| L4 | `v160_enhanced.py:39` adds string `side` to numeric layer scores; inherited `v151_core_trader.py:344` clamps every value numerically. Controller reproduced TypeError on the metadata value. | High, legacy caller defect |
| L5 | M2 is described as markup but side selection can yield short (`wyckoff_m1m2.py:37`; `v160_enhanced.py:363`). Reviewer reproduced both short paths; controller inspected the branches. | Naming/direction inconsistency; not evidence that every short is wrong |
| L6 | Older `modules/wyckoff/state_machine.py:89,294` includes the candidate in its range before outside-range checks. `analyzer.py:34,93` similarly makes its outside-range/phase-E branch unreachable for valid OHLC. `advanced.py:1,93` explicitly remains a scaffold/TODO using range position and averages. | Legacy defects/incompleteness; not a hidden complete replacement |

## Test evidence and what it does not certify

Fresh controller command (no source/test modifications):

```bash
python3 -m pytest -o addopts='' -q \
  tests/test_wyckoff_events.py tests/test_wyckoff_m2_sequence.py \
  tests/test_wyckoff_causality.py tests/test_wyckoff_mtf.py \
  tests/test_wyckoff_v2_climax.py --tb=short
```

Final rerun: **41 passed, 5 failed, 291 warnings, 2.67s, exit 1**.
An earlier controller run and the event reviewer obtained the same pass/fail
counts. This is a focused suite, not a full-repository result.

All five failures are in `tests/test_wyckoff_events.py`:

| Failure | Fixture/implementation disagreement observed |
| --- | --- |
| `test_sc_basic_detection` | Edited high is below the unchanged open; unseeded random baseline also affects range statistics. |
| `test_bc_basic_detection` | Edited low exceeds the unchanged open; intended spike also fails the rolling-range close-position threshold. |
| `test_ar_after_sc` | Rally low is below the intervening lows, violating the detector's no-new-lows condition. |
| `test_st_basic_detection` | The 15-bar window no longer includes the earlier climax; the proposed test is the new rolling low. |
| `test_spring_a_fake_breakdown` | Only part of the prior 20-bar range is raised; the candidate does not satisfy the actual 1.5% breach rule. |

Do not weaken production rules just to make these fixtures green. Make the
OHLC fixtures valid and deterministic, then adjudicate the source contract.

Passing tests also have limits:

- The five M2 tests validate supplied event labels, not raw-candle recognition.
- The four causality tests exercise selected cases; they do not certify the
  live HTF resampler, lifecycle carry-forward or whole integration.
- The nine MTF tests import legacy `bull_machine.modules.wyckoff.mtf_sync`,
  not the current live hierarchical feature path.
- The eleven V2 climax tests concern shadow detectors and selected isolation
  properties, not promotion of V2 into active production behavior.
- Legacy bounded-score/long-or-neutral assertions can pass after a caught
  error produces all-zero evidence.

No new annotated historical chart benchmark, precision/recall estimate,
walk-forward/CPCV study, profit result, or live-loss attribution was produced.

## Minimal reproducible witnesses

Run from the repository root. This diagnostic writes no project files and
does not construct a live runner or submit orders. It demonstrates A1, the
phase-mutation part of A2, and the relative-volume issue in A3:

```python
from engine.archetypes.archetype_instance import ArchetypeInstance
from engine.wyckoff.events import WyckoffStateMachine

obj = object.__new__(ArchetypeInstance)
obj.direction = 'long'
features = {'wyckoff_event_confidence': 0.8}
for prefix in ('', 'tf4h_', 'tf1d_'):
    features[prefix + 'wyckoff_bullish_score'] = 0.0
    features[prefix + 'wyckoff_bearish_score'] = 0.8
print('bearish-only long score:', obj._get_wyckoff_score(features))  # 0.8

def row(low, close, high, z=0.0, volume=1000):
    return dict(open=close, low=low, close=close, high=high,
                volume_z=z, volume=volume)

def parent():
    sm = WyckoffStateMachine({})
    sm.process_bar(0, row(100, 102, 103, 3), {'sc': True})
    sm.process_bar(3, row(105, 109, 110), {'ar': True})
    return sm

sm = parent()
valid, _ = sm.process_bar(6, row(100.5, 103, 104), {'spring_a': True})
print('rejected event / phase:', valid['spring_a'], sm.get_phase_dir())
# False C_accum
sm = parent()
valid, _ = sm.process_bar(6, row(101, 103, 104, 0, 5000), {'st': True})
print('larger raw-volume ST / stored SC:', valid['st'], sm.range_ref.sc_volume)
# True 3
```

The delayed spring/upthrust witnesses in A2 used the actual raw detectors and
a seeded sequencer parent; the climax witness in A4 used the existing V2
fixture through the complete current detector entrypoint. The resampler
witness in A7 executed that method's source in isolation, avoiding live
startup. These boundaries are intentionally narrower than live certification.

## Repair order and exit criteria

1. **Repair evidence correctness, not profitability.** Preserve candidate and
   confirmation identity, reject phase changes from rejected events, prevent
   opposite-side/proxy fallbacks from masquerading as confirmation, and enforce
   candle uniqueness/completeness or explicit provisional status. Add failing
   deterministic regressions first. No parameter search.
2. **Specify and verify complete sequences.** Carry parent range identity and
   lifecycle, raw and relative volume with their units, and explicit uncertain
   phase. Distinguish spring and no-spring paths; distinguish UT from UTAD and
   in-range SOS from range escape. Add raw-OHLC -> event -> phase -> consumer
   positive, negative and ambiguous fixtures, including no-climax
   reaccumulation when that variant is in scope. Do not force every structure
   into one schematic or promote M2 merely to increase counts.
3. **Verify contextual roles and consumers.** Larger structure determines
   applicability; volume/spread supports or challenges the thesis; minute
   structure supplies timing. Test explicit parent/child linkage, as-of data,
   invalidation and destination. Test the selected 17 consumers and sizing
   separately so "shadow" and "bypass" claims match actual effects.
4. **Then test recognition on unseen, outcome-hidden examples.** Label both
   genuine and lookalike setups, report false positives, missed setups,
   uncertainty and reviewer disagreement. Code tests alone cannot authenticate
   a trader's interpretation. Freeze semantics before looking at trade outcomes.
5. **Only then run an isolated economic comparison.** One specified archetype,
   frozen costs/execution, chronological held blocks and overlap controls;
   report rejected candidates and retained winners/losers. No guarantee of
   profitability, and no reuse of exposed results as an untouched holdout.

The next bounded implementation should be item 1, followed by a software review.
Items 2–5 are downstream acceptance gates, not experiments launched by this audit.
Legacy modules should remain explicitly legacy unless a real consumer requires
repair; they are not the priority over the selected path.

## Reproducibility and handoff

Critical inspected file SHA-256 values:

```text
58b6fdbca43b68d0f37638b3e43767a27be1a612140c2ecbe251c5c79c9067fd  engine/wyckoff/events.py
3efad743afb994bd5982b0e68eccca9e7d95323841f81b0035ad071344e89dd9  engine/archetypes/archetype_instance.py
c1e5561c063112c213d48bbf2e40418ecbb5903c4a88d6acbc7cddd331d6edcc  bin/live/live_feature_computer.py
10722d523ef931bc6172b8b67efb69d1dec3f956d4926552aec1f85a36f659c8  bin/live/v11_shadow_runner.py
51c53f678e5fecfa0c87be5be9b89d1f9c5c5fa72017604eea29246baa65b4c0  bull_machine/strategy/wyckoff_m1m2.py
899478d8f2bc06d71ec267cb9b7e23707a15895c0c240dd5d75fbbc4fee76dac  configs/champion_paper.json
```

Only this report and the PROJECT/MEMORY continuity additions were authored by
this audit. Existing unrelated uncommitted files were retained. No engine,
tests, selected config, frozen experiments, or live orders changed. No commit,
push, PR, library install, paid external model experiment or new market-data
download. Session reviewers consume normal agent usage; they are not free
API credits. No audit jobs remain running at handoff.

The focused diagnostics use the existing local Python dependencies, not the
private minute-data archive. Earlier economic studies still require their
local-only inputs; this audit does not make those studies portable by itself.
