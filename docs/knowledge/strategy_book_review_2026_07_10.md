# Strategy Book Coherence Review — All 17 Archetypes (v14rq Live Set)

**Date**: 2026-07-10
**Reviewer**: quant-analyst (rails-on, READ-ONLY audit — no backtests run, no configs touched)
**Scope**: `configs/champion/archetypes_v14rq/*.yaml` (17 archetypes) vs detection code
(`engine/archetypes/logic.py::_check_*`, `archetype_instance.py` gates/fusion/whale penalty,
`structural_check.py` bridge), reconciled against the honest V14 evidence corpus
(champion hunt 06-11, stack verdict 07-08, live forensics 07-02/04, wick_trap CPCV 07-09,
winner-loser forensic 06-28, industry/ES knowledge docs) and the full16 trade log
(`results/champion_v14_risk/comp_full16/full/trade_log.csv`, 1,894 positions) plus
gate-pass-rate measurement on `BTC_1H_FEATURES_V14_FULL_LIVE_PATH.parquet` (73,829 bars).

> **STANDING ORDERS (verbatim, not overridden):** NEVER turn off bypass_threshold — data
> collection mode is required for the foreseeable future. NEVER disable any archetype — all 16
> stay enabled to collect maximum live signal data. NEVER make production config changes
> (bypass, disabled_archetypes, thresholds, archetype YAMLs) without explicit user approval.
> NEVER edit production code/configs directly — recommendations and diffs only.

Everything below is diagnosis + **pre-registered hypothesis candidates**. Nothing is a change.

---

## 1. Executive Summary

- **The book is nominally 17 strategies but functionally ~1.5.** Five archetypes never trade at all
  (2 dead-by-direction, 3 dead-by-data). Of the 12 that trade, at least 7 are variations of the same
  trade — "buy a long-lower-wick/RSI-extreme flush and hope for a bounce" — confirmed by entry-time
  clustering (35–46% of reversal-archetype entries land within 24h of a spring/CB entry). The one
  honestly validated edge is wick_trap_v14rq; almost everything else is either refuted OOS or
  statistically starved.
- **A systemic "wrong-side condition" bug family:** long-only archetypes gate on BIDIRECTIONAL
  extremes. `derived:wick_anomaly` passes on upper-wick-only bars (40.7% of qualifying bars are the
  bearish shape); `derived:rsi_extreme_65` is 53% overbought-side; exhaustion_reversal's identity
  fires more often on RSI>78 (5.3% of bars) than RSI<22 (4.1%); spring's identity accepts `utad`
  (a distribution-TOP pattern, 1,860 detections) alongside `spring` (826). Long archetypes are
  systematically fed bearish-side events.
- **A systemic inert-gate problem on the V14 live-path store:** `instability` (wick_trap),
  `rsi_divergence` (5 archetypes), `effort_result_ratio` (2), `vol_shock` (long_squeeze's prized
  n=14 gate), `bars_since_pivot ≤ 110` (feature maxes at 48) are all missing/degenerate with
  `nan_policy: skip` → silently do nothing. Worse, `wick_lower_ratio ≥ 1.3` (liquidity_sweep) is
  IMPOSSIBLE (feature max = 1.0, scale mismatch) → a permanent −50% soft-gate fusion penalty on
  every LS signal. And **all BOS columns are 0.0 across the entire V14 store**, structurally
  killing fvg_continuation, order_block_retest, and liquidity_vacuum (0 trades each in full16)
  and zeroing the SMC domain's BOS component book-wide.
- **The biggest strategic gap is not a missing archetype, it's a missing dimension:** no archetype
  anchors to a LEVEL, requires acceptance/confirmation (all enter on the signal bar — the Osler
  cascade risk), or expresses the short/range side with a working detector. The book's dominant
  loss mode (long-only bleed in sustained downtrends) is a portfolio-construction fact, not a
  tuning problem.
- **Top-ranked pre-registered hypotheses:** (1) level-anchored wick_trap rebuild, (2) wick_trap
  `trailing_start_r` sweep, (3) CB no-chase identity condition + boost mirror, (4) direction-purity
  fixes (spring/UTAD, ER RSI>78 branch, wick_anomaly side), (5) data-integrity repairs (BOS,
  wick-ratio scale, inert features) **before** any further tuning of affected archetypes.

---

## 2. Methodology

- Read all 17 live YAMLs; diffed against `configs/archetypes/*.yaml` — the sets are identical
  except the two re-quantiled `liquidity_threshold: 0.72 → 0.43` entries (wick_trap,
  trap_within_trend), so this review covers both.
- Read the full detection pipeline: `structural_check.py` (identity bridge; note **oi_divergence
  has NO structural check** — absent from NAME_TO_LETTER), `logic.py::_check_A.._check_S8`
  (identity gates), `archetype_instance.py` (derived features, `_evaluate_gates`, soft-penalty
  math, whale conflict, fusion domains, `direction == 'neutral' → return None`).
- Measured gate pass-rates and feature availability directly on the V14 live-path parquet.
- Aggregated the full16 trade log to position level (1,894 positions, 2018–2024) for per-archetype
  PnL/PF/n and cross-archetype entry-time proximity.
- No new backtests, no WFO — this is a coherence audit; all improvement items are framed as
  hypotheses requiring the standard train+OOS co-move / per-archetype-OOS machinery before any adoption.

Zero-tolerance compliance: no fusion-based filters are proposed anywhere below (Lesson #54); every
filter-shaped idea is framed as detector IDENTITY (structural condition) or as a sizing boost/penalty
with its boost mirror (Rule 8); dead/refuted archetypes are flagged for *funding* review only —
disabling is a Standing-Order/user decision.

---

## 3. Per-Archetype Verdicts

Evidence key: full16 = comp_full16 2018–2024 book run; hunt = v14_champion_hunt standalone;
holdout = 2025-01→2026-06 pristine.

### 3.1 wick_trap (long, gate_mode hard) — COHERENT-ISH, the only validated edge
**Thesis**: panic flush prints an outsized rejection wick; buy the washout.
**Gate audit**: `derived:wick_anomaly` is BIDIRECTIONAL — an upper-wick-only bar (bearish rejection
shape) satisfies it; 40.7% of qualifying bars are that wrong shape, and there is no lower-wick
requirement anywhere in K's identity or gates. `volume_zscore ≥ 0` OK. `instability ≤ 0.45` is
**INERT** (feature absent on V14, nan skip) — the p=0.018 WR claim in its description was never
exercised in any V14 backtest, holdout, or CPCV. Also: `wick_anomaly` passes on 62.5% of ALL bars —
as an "anomaly" it is barely selective; selectivity actually comes from `liquidity_threshold 0.43`
(q90.3) + cooling.
**Evidence**: SUPPORTED — holdout PF 1.43 +$6.8K n=70, CPCV 15/15 positive, worst path +$2,421.
Known bear-year bleed (~$8-9K per sustained bear) is priced in. skip200 rejected (deletes its
washout winners — Rule 9). BE@1R a noop; the +0.41R..1.0R window is unprotected.
**Verdict**: the edge is real but the detector under-specifies the thesis (no level, no side, no
reclaim). CAUTION: the validated CPCV result includes the inert instability gate — "fixing" the
feature name is a BEHAVIOR CHANGE requiring full re-validation, not a cleanup.

### 3.2 trap_within_trend (long, hard) — INCOHERENT trend test, edge V12-artifact
**Thesis**: false breakdown within an established UPtrend.
**Gate audit**: the identity's "trend context" (`_check_H`) passes if `price_above_ema_50` merely
EXISTS as a column — a bar BELOW the EMA-50 counts as "trend context present." So "within trend"
degenerates to ADX ≥ 10 + wick anomaly. `bars_since_pivot ≤ 110` is INERT (feature range 30–48,
always passes; the fusion staleness penalty at >89 also never fires). `wick_lower_ratio ≥ 0.25`
does enforce the correct side (nan_policy fail) — the only direction-pure wick gate in the book.
**Evidence**: REFUTED standalone — hunt: 2023/2024/holdout all negative post-requantile ("V12
artifact, deprioritize"). In the full16 book it shows +$48.6K/PF 1.37/n=237, but that is
dedup-context PnL (it absorbs bars other archetypes would have taken), not a validated claim.
Winner-loser forensic found its one OOS-real entry edge: wins with HIGHER trend_align and FURTHER
from support — i.e., exactly the trend condition the identity fails to test.
**Verdict**: thesis fine, trend test vacuous. H6 below.

### 3.3 liquidity_sweep (long, soft) — COHERENT thesis, BROKEN gate, passenger OOS
**Thesis**: sweep of lows takes liquidity, reclaim follows.
**Gate audit**: identity (`_check_G`) is a decent side-pure sweep-shape check (lower wick >35% of
range AND lower>upper). But `wick_lower_ratio ≥ 1.3` (described "lower wick > body") is
**IMPOSSIBLE on V14** — the live-path feature is wick/range-scaled (max 1.0), not wick/body. In
soft mode that is a permanent 50% fusion penalty on every signal (2 gates, 1 always fails), and
the backtester enforces thresholds (Lesson #43), so this materially suppresses LS expression —
same failure class as the liquidity_score root cause. No reclaim/level/acceptance condition:
enters ON the sweep bar, which Osler 2005 says is inside the cascade.
**Evidence**: WEAK — holdout +$62 over 35 trades (a passenger per stack verdict); V12-era $91K/1.59
was feature-scale inflation.
**Verdict**: the archetype most obviously wounded by the V12→V14 migration; fix the scale (H7)
and re-baseline before judging the thesis.

### 3.4 confluence_breakout (long, soft + bypass_fusion + enforce_gates_under_bypass: false) — INCOHERENT
**Thesis**: volatility coil at the value-area POC resolves into a breakout.
**Gate audit**: identity = ATR pctile <0.30 AND within 5% of POC. **There is no breakout condition
anywhere** — `require_breakout_flag: false`, no level definition, no range-break test. It buys
"quiet near fair value," i.e., nothing has happened yet. It is also the book's known gate-immune
architecture (Rule 10): soft gates + fusion bypass = every gate is decorative. Its
`distribution_at_resistance` block and `volume_zscore ≥ 0.5` gates therefore do nothing.
**Evidence**: REFUTED as an edge — 617 positions (the book's most-traded archetype), PF 1.04,
42% WR; live last-10 3W/7L; loss concentration validated at scale: entries >4% above the 7-day low
= 410 positions, −$32K; near-low entries ≈ breakeven. CB only "works" when it accidentally buys a
washout — i.e., when it behaves like wick_trap.
**Verdict**: name and detector describe different strategies; the empirical winner population
contradicts the breakout thesis outright. H3 below.

### 3.5 exhaustion_reversal (long, hard) — HALF-COHERENT (wrong-side branch)
**Thesis**: climactic move exhausts; fade it. As a LONG-only archetype that means buying
capitulation lows.
**Gate audit**: identity is RSI>78 OR RSI<22 — the >78 branch (5.3% of bars, MORE common than the
<22 branch's 4.1%) takes longs into overbought blow-offs, contradicting the long thesis.
`derived:rsi_extreme_65` repeats the bidirectionality. `atr_percentile ≥ 0.5` is NOT unreachable
(33.7% of bars pass; combined identity+gates ≈ 1.9% of bars) — the "unreachable-ish" suspicion is
cleared. 48h max_hold on hard-gated 2.2x-ATR stops; crisis regime pref 1.5 leans INTO crashes.
**Evidence**: MIXED-to-REFUTED — breadth (n=443 full, +$22.5K PF 1.23) but holdout PF 0.85
negative; bleeds 2018/2022; MFE study: 24% of its losers were +1R first (worst give-back
archetype); the V12-era pair complementarity did not survive V14.
**Verdict**: the RSI>78-long branch is the cleanest single pre-registerable split in the book (H4).

### 3.6 failed_continuation (long, hard) — MOSTLY COHERENT, half its gates are dead
**Thesis**: bearish continuation attempt fails (FVG + fading momentum) → reversal long.
**Gate audit**: identity (FVG + ADX falling + vol_z<1.5) matches the thesis. But of 6 YAML gates:
`effort_result_ratio ≤ 1.4` INERT (feature 100% NaN, skip), `rsi_divergence ≥ 0.1` INERT (feature
missing, skip), `volume_zscore ≤ 3.5` near-vacuous (described as "volume fading" but 3.5σ excludes
almost nothing). Real gates: any_fvg, RSI ≤ 55, chop ≤ 0.25. Cruft: thresholds block has
`rsi_min: 55` AND `rsi_max: 55` simultaneously.
**Evidence**: PARTIAL — full16 +$32.9K PF 1.29 n=227 (book context); no standalone holdout pass on
record; old post-Optuna table showed it both at PF 13.47 and PF 0.35 in different rows (sample
instability).
**Verdict**: coherent core, but its documented quality gates (PF 1.94→2.51 claim on chop) are the
only live ones; the rest is dead weight that will silently activate if the features ever appear —
re-validate then.

### 3.7 spring (long, soft) — DIRECTION-IMPURE identity
**Thesis**: Wyckoff spring — a sweep below accumulation range lows that recovers.
**Gate audit**: identity accepts `tf1h_pti_trap_type ∈ {spring, utad, bull_trap, bear_trap}`. UTAD
is the DISTRIBUTION-TOP mirror pattern — going long on it inverts Wyckoff — and the V14 store
contains 1,860 utad vs 826 spring detections, so the majority of eligible events are the wrong
pattern for a long. Wyckoff/PTI gates are nan-skip but the features exist. Exits (scale-outs at
1/2/3R, 168h) fit a multi-day accumulation thesis. Crisis pref 2.0 is thesis-consistent
(springs happen at panic lows) but risky.
**Evidence**: REFUTED standalone (hunt: PF 1.06, 27% DD, "dead"); full16 book +$21.6K n=295 is
dedup-context. Winner-loser: mild real edge from wyckoff_score/threshold_margin.
**Verdict**: H5 (trap-type purity) is a one-line identity hypothesis with a clean frozen-entry A/B.

### 3.8 retest_cluster (long, soft) — REFUTED; leading retirement-from-funding candidate
**Thesis**: fakeout at a level, then the real move, confirmed by multi-level (time) confluence.
**Gate audit**: identity = vol_z>1.0 + RSI>70|<30 (bidirectional again) + fib_time_cluster>0. The
"cluster" is fib-TIME only — no price level is ever involved despite the name. YAML gates:
`temporal_confluence_score ≥ 0.45` passes 99.6% of bars (vacuous), `rsi_divergence` INERT.
**Evidence**: REFUTED — hunt: full −$0.3K, holdout −$7.6K PF 0.38 ("dead"); full16 book: WORST
archetype, −$22.7K, PF 0.57, 35% WR. It is also the archetype whose hard-mode flip was already
tried and reverted (Lesson #55).
**Verdict**: incoherent (no price level in a "retest" pattern) and empirically refuted on both
stores. Keep logging (Standing Order); user decision on funding.

### 3.9 liquidity_compression (long, hard) — TWO THESES STAPLED, weakly supported
**Thesis (name/description)**: compression before expansion. **Thesis (identity/gates)**: climax
volume at RSI extreme = capitulation. `_check_E` requires volume climax / vol_z>2 at RSI extreme;
gates add vol_z ≥ 1.5 + `bb_width ≤ 0.06` (83.5% pass — weak squeeze test) + rsi_extreme_65
(bidirectional) + chop ≤ 0.5. A "quiet coil" and a "volume climax" are opposite events; the
detector actually trades the climax, the name says coil.
**Evidence**: PARTIAL — full16 +$29.3K PF 1.35 n=178 (2nd best book PnL); the ONLY positive
archetype in the last-30 live forensic (+$442, 3/7 in a −22.8% tape). No standalone holdout run
on record.
**Verdict**: rename-or-refactor honesty problem, but the capitulation detector it actually runs is
one of the book's few net contributors. Deserves a standalone battery before more capital trust.

### 3.10 liquidity_vacuum (long, hard) — COHERENT THESIS, DEAD DETECTOR
**Thesis**: crisis capitulation at panic lows (regime prefs crisis 2.5 / risk_off 1.5 — the book's
only explicitly counter-crisis design, thesis-consistent).
**Gate audit**: identity `_check_S1` requires `tf1h_bos_bearish` — **all-zero on the V14 store** —
plus volume climax OR `absorption_flag` (100% NaN). Result: 0 trades in full16. YAML gates
(liquidity_score ≤ 0.45 = capitulation, vol_z ≥ 0, wick_exhaustion) are fine but unreachable.
**Evidence**: UNTESTED on V14 (V12-era n=12 "inf PF" is meaningless).
**Verdict**: repair-then-baseline (H7). Do not tune anything until BOS exists.

### 3.11 fvg_continuation (long, hard) — DEAD (data), thesis muddled
**Thesis**: continuation through a fair-value gap after a break of structure. Aliases say
"bos_choch_reversal" — continuation and reversal in the same identity crisis.
**Gate audit**: identity requires FVG AND recent BOS (10 bars) → dead, BOS all-zero.
`rsi_divergence` gate INERT. Tightest stop in the book (1.2x ATR) with 5.2x TP — a 4.3:1 shape
that the MFE study (trades top out ~1.4R book-wide) suggests is fantasy geometry.
**Evidence**: UNTESTED on V14 (0 trades); V12-era +$3.8K PF 1.36.
**Verdict**: dead pending BOS repair; on revival, reconcile continuation-vs-reversal identity first.

### 3.12 order_block_retest (long, soft) — DEAD (data) + previously refuted
**Thesis**: price retests an institutional order block after a structure break.
**Gate audit**: identity needs a BOS within 20 bars at ≤1.5 ATR — dead (BOS all-zero). Gates
(wyckoff ≥ 0.05, fib_time_confluence ≥ 0.1, rsi_divergence INERT) are soft and weak; the 2026-03
overhaul already showed tightening kills it entirely (Lesson #55).
**Evidence**: REFUTED when it did trade (PF 0.75, −$2.7K V12-era; MEMORY lists it as a proven
loser) and 0 trades on V14.
**Verdict**: doubly dead. Retirement-from-funding candidate #2 (user decision).

### 3.13 funding_divergence (long, soft) — MOST COHERENT GATE SET, statistically starved
**Thesis**: shorts crowded (negative funding) while OI unwinds = squeeze fuel; buy.
**Gate audit**: the book's best thesis-to-gate mapping — all five gates are concurrent-state
derivatives (funding_Z ≤ −0.5, funding_oi_divergence == 1, ls_ratio ≤ −0.5, oi_price_divergence,
oi_change_4h ≤ 0.05), exactly the Lesson #41-compliant design. Differentiated exits (240h,
4 scale-outs to 3R) match the multi-day squeeze thesis. Minor cruft: thresholds block carries
contradictory `funding_z_max: -0.5` / `funding_z_min: 0.5` leftovers.
**Evidence**: DIRECTIONAL-ONLY — full PF 2.23 but ~4 trades/yr; holdout −$0.3K; full16 n=6,
+$9.9K, PF 11 (small-n, Lesson #3). "Garnish, not a strategy."
**Verdict**: coherent; nothing to fix except acknowledging it cannot carry capital. Leave it
accumulating samples.

### 3.14 long_squeeze (short, soft) — COHERENT THESIS, REFUTED IMPLEMENTATION, book's only short
**Thesis**: overcrowded longs (positive funding, RSI hot, L/S extreme) cascade down.
**Gate audit**: thesis-consistent derivative gates, BUT `vol_shock ≤ 0.10` — the gate derived from
the n=14 winners/losers split — is INERT (feature missing on V14, skip). `accumulation_at_support`
block is a sensible mirror of CB's distribution block. Tight 1.3x stop with 6.0x TP is lottery
geometry.
**Evidence**: REFUTED — historical PF 0.13/−$6.8K; regime gates on it filtered the WINS three
times (Lesson #60b selection asymmetry); full16 n=2, −$2K. Meanwhile the downtrend study says
naive regime-shorting has no edge — so the failure is not "shorts are impossible," it's that this
implementation never validated.
**Verdict**: the book's single short expression is broken while the book's #1 loss mode is
long-only bear bleed. That asymmetry is the strategic hole (see §4), but do NOT expand shorts
without a precise setup (downtrend study's explicit warning).

### 3.15 oi_divergence (long, soft) — HALF-TESTED THESIS
**Thesis**: "hollow" move — price pushes to an extreme while OI declines (positions closing) →
reversal. **The price-extreme half is never tested**: there is no structural check (absent from
NAME_TO_LETTER) and no gate references `price_extreme_lookback/percentile` (they sit unused in
thresholds). What actually trades is "OI falling + RSI ≤ 35 + net selling" — an oversold-plus-
unwind long, not a divergence. Comment block even argues for shorts in bull regimes, but
direction is long-only.
**Evidence**: MIXED — infamous 0/5 live leak and 0/3 in last-30, but full16 n=21 +$3.0K PF 1.36.
Underpowered either way.
**Verdict**: implement the missing half (price-extreme condition) as identity, or admit it's an
oversold-OI-unwind long. n too small for any conclusion; it's the +1 shadow archetype — fine.

### 3.16 whipsaw (neutral) — STRUCTURALLY UNREACHABLE
**Thesis**: fade a false break above range highs (upper-wick rejection + Sign of Weakness) —
which is a SHORT thesis, labeled `direction: neutral`.
**Gate audit**: `ArchetypeInstance.detect()` returns None for `direction == 'neutral'` before
building any signal → whipsaw can NEVER trade (0 trades in full16, confirmed). Its identity
(`_check_S3`: upper wick > 2x body + SOW/vol climax) is actually one of the better-specified
patterns in logic.py — and it's the book's only range-fade with a directional edge story.
**Evidence**: UNTESTED (unreachable).
**Verdict**: `enabled: true` + unreachable = configuration theater. Either give it
`direction: short` (as a pre-registered hypothesis with full validation) or acknowledge it as dead.

### 3.17 volume_fade_chop (neutral) — STRUCTURALLY UNREACHABLE, doubly dead
**Thesis**: fade low-conviction moves in trendless chop.
**Gate audit**: same neutral-direction dead-end; additionally its `adx_14` gate references a
feature that doesn't exist on V14 (`adx` does; nan_policy defaults to FAIL, so it would block even
if direction were fixed).
**Evidence**: UNTESTED (unreachable).
**Verdict**: the book has NO functioning range/mean-reversion expression — both candidates are
unreachable. Gap, not tuning item.

---

## 4. Book-Level Findings

### 4.1 Functional census (17 configured → ~10 trading → 1 validated)
| Status | Archetypes |
|---|---|
| Validated OOS edge | wick_trap_v14rq |
| Trading, partial/mixed support | liquidity_compression, failed_continuation, funding_divergence (starved), oi_divergence (starved) |
| Trading, refuted or ~zero edge | confluence_breakout, retest_cluster, long_squeeze, spring (standalone), trap_within_trend (standalone), exhaustion_reversal (holdout), liquidity_sweep (passenger) |
| Dead — data (BOS all-zero / feature scale) | fvg_continuation, order_block_retest, liquidity_vacuum |
| Dead — direction: neutral | whipsaw, volume_fade_chop |

### 4.2 The book is one trade wearing eight costumes
Entry-time clustering (full16, ±24h): exhaustion_reversal↔spring 46%, failed_continuation↔spring
44%, retest_cluster↔{CB, spring} 43%, trap_within_trend↔spring 42%, liquidity_sweep↔liquidity_
compression 43%. Dedup (best_per_direction) already caps 1 long/bar — verified only 1 bar in 1,893
had two archetypes enter — so the overlap manifests as *serial* re-entries into the same episode
by sibling archetypes on adjacent bars, and as the dedup-reshuffling false-signal machinery
(Rules 7/61). Seven long archetypes share the wick/RSI-extreme/volume-climax reversal family;
CB, nominally a breakout, empirically only earns when it does the same washout-buy. Portfolio
implication: the book's diversification is largely cosmetic; its factor exposure is ~single
(long BTC mean-reversion after flushes), which is exactly why every sustained downtrend produces
the observed correlated bleed (2018, 2021H2, 2022, May–Jun 2026 live).

### 4.3 Gaps
1. **Level dimension**: no archetype conditions on any price level (swing, prior-day extreme,
   pool, POC-as-level). "Retest," "order block," "breakout," and "sweep" archetypes all lack the
   level concept their names imply. This is the single biggest coherence deficit and the
   documented industry-standard gap (wick-trap industry study; ES failed-breakdown knowledge).
2. **Acceptance/confirmation**: every archetype enters on the signal bar. Osler: stop cascades
   run for hours; the reversal edge is AT levels after confirmation. No archetype waits.
3. **Short side**: 14 long, 2 unreachable neutrals, 1 refuted short. Defensive skip is validated
   only as bleed-control for the junk book and is anti-correlated with wick_trap.
4. **Range regime**: both mean-reversion specialists are unreachable.
5. **Data integrity as strategy risk**: BOS all-zero, wick-ratio scale mismatch, and 6+ inert gate
   features mean parts of the book run a DIFFERENT strategy than the YAML claims. Any future
   store rebuild that silently revives these features will change behavior of "validated" configs
   without any config diff — the exact parity failure class already documented twice.

### 4.4 Funding-tier read (recommendation to user; Standing Orders untouched)
- **Fund**: wick_trap_v14rq (satellite-sized; known bear bleed).
- **More data, unfunded/minimal**: liquidity_compression (run its standalone battery),
  funding_divergence, oi_divergence.
- **Retirement-from-funding review**: retest_cluster (refuted both stores), long_squeeze
  (refuted, gate inert), order_block_retest (dead + refuted), confluence_breakout (PF 1.04 on
  n=617 — the book's largest trade-count consumer for ~nothing). All stay ENABLED for logging.
- **Repair before judging**: liquidity_sweep, liquidity_vacuum, fvg_continuation.

---

## 5. Ranked Improvement Hypotheses (pre-registered candidates — NOT changes)

All require: WFO train (2018–22) + holdout co-move (Rule 9), per-archetype OOS reporting (Rule 7),
boost-mirror tested alongside any filter-shaped variant (Rule 8), regime stratification, n≥30 OOS.

**H1 — Level-anchored wick_trap rebuild** (highest ceiling; carried from 06-11, still open).
Sweep-beyond-prior-level + close-back-inside + N-bar acceptance + no-chase distance + structural
stop past the sweep extreme. Evidence: industry study (every surviving implementation is
level-anchored; no wick-ratio gate exists anywhere in the field), ES knowledge, wick_trap already
the best asset. Test: new detector as a PARALLEL archetype config vs wick_trap_v14rq baseline,
full battery + CPCV. Effort: LARGE (new features: swing/pool inventory, prior-day levels).

**H2 — wick_trap `trailing_start_r` sweep {0.5, 0.75, 1.0}** (stack verdict's named next step).
Evidence: the +0.41R..1.0R window is unprotected; trailing dominates BE. Test: existing production
keys, frozen grid, wt_only book, train+holdout co-move. Effort: SMALL.

**H3 — CB no-chase as IDENTITY condition + boost mirror.** Add to `_check_M` (or a derived
feature): entry within X% of the 7-day low, X ∈ {2, 3, 4} pre-registered; mirror variant: 0.5x
sizing penalty when extended (Rule 8 requires testing both). Must be identity/architectural — CB
is gate-immune (Rule 10), YAML gate edits are no-ops. Evidence: −$32K concentration in >4%-above-
low entries at scale; near-low ≈ breakeven. Honest expectation: loss-avoidance, not edge creation
(CB stays weak). Effort: SMALL-MEDIUM.

**H4 — Direction-purity family (three one-liners, tested separately):**
   a) exhaustion_reversal: split results RSI<22 vs RSI>78 branches on existing logs (zero-cost
      diagnosis), then hypothesis "restrict identity to RSI<22 for longs."
   b) spring: identity accepts only `spring` (± `bull_trap`) — drop `utad`/`bear_trap` (69% of
      current eligible events are utad).
   c) wick_anomaly consumers (wick_trap, TWT): require the LOWER wick to be the anomalous one for
      long direction (TWT already half-does this via wick_lower_ratio ≥ 0.25).
Evidence: measured wrong-side shares (40.7% upper-only wick bars; RSI extreme 53% overbought).
Caveat: these REMOVE trades — expect dedup-reshuffling noise; adjudicate per-archetype (Rule 7).
For wick_trap specifically this modifies a CPCV-validated config → full re-validation, one shot.
Effort: SMALL each.

**H5 — TWT real-trend identity**: `has_trend` requires `price_above_ema_50 == True` (not merely
present) or trend_align above threshold; plus the already-flagged OOS-real boost (higher
trend_align + dist_to_support_atr → 1.25–1.5x size). Evidence: winner-loser forensic
(AUC 0.587/0.625 and 0.589/0.619, train/holdout sign-consistent). Effort: SMALL.

**H6 — Repair the dead data, then re-baseline (precondition, not a strategy test):** BOS pipeline
(all-zero → fvg_continuation/OB_retest/liquidity_vacuum dead, SMC domain half-blind);
wick_lower_ratio scale (liquidity_sweep's permanent −50% penalty); `instability`/`rsi_divergence`/
`effort_result_ratio`/`vol_shock`/`adx_14` naming-or-computation. RULE: after repair, re-run
BASELINES for affected archetypes before any tuning — revived gates change behavior with zero
config diff. Effort: MEDIUM (feature-store work, already partly in flight on this branch).

**H7 — Re-register the wick_trap "ride policy"** (from the 07-08 retraction): scale-outs off +
entry-floor stop after intrabar +1R, clean production implementation, one shot. Effort: MEDIUM.

**H8 — Day-type (range-vs-trend) classification as sizing overlay** (ES H3): counter-trend
archetypes ×0.25 on confirmed trend days. Evidence: 2022 + May–Jun 2026 loss anatomy. Effort:
MEDIUM (classifier features exist per 06-11 notes).

**H9 — whipsaw direction: short resurrection** as a NEW pre-registered short hypothesis (precise
setup: upper-wick rejection + SOW at range highs — closer to "a specific setup" than the naive
regime short the downtrend study rejected). Effort: SMALL config + full validation battery.

**H10 — liquidity_compression standalone battery** (it earned book PnL and the only live green,
but has never faced the holdout alone). Effort: SMALL (existing harness).

Explicitly NOT proposed: any fusion-score gate (Lesson #54), any bypass/enable change (Standing
Orders), CB YAML gate tuning (Rule 10 no-op), further ER/retest regime gates (Lesson #60 selection
asymmetry), naive short expansion (downtrend study).

---

## 6. Sample Size & Honest Caveats

- Wrong-side shares (40.7%, 53%) are BAR-level base rates, not trade-level PnL attribution — they
  prove the gates don't test the thesis, not that the wrong-side trades lose. H4's diagnosis step
  (split existing trade logs by side) must precede any identity change.
- full16 per-archetype PnL is book-context (dedup-shaped): TWT +$48.6K and spring +$21.6K there
  do NOT contradict their standalone refutations — both things are true simultaneously.
- Overlap metric (±24h proximity) is a coincidence measure, not correlation of returns; it
  overstates overlap for high-frequency archetypes (CB n=617 is "near" everything).
- Store checks were run on `BTC_1H_FEATURES_V14_FULL_LIVE_PATH.parquet`; the live runner computes
  features independently — live-path values of the inert/broken features may differ (the
  `instability` gate inertness was, however, already confirmed live on 06-24).
- No new backtests were run; every "would improve" statement here is hypothesis-grade by
  construction. Bias toward NO stands: the honest prior from this codebase is that most of these
  will fail validation (filters 0/8 historically; boosts 2/2).

## 7. What This Doesn't Test

- No WFO/CPCV on any hypothesis; no phantom simulations used anywhere.
- Exit-parameter coherence beyond flagged items (per-archetype ATR stop/TP geometry was noted but
  not audited against realized R distributions except via the existing MFE study).
- Whale conflict penalty calibration (reviewed for direction-logic coherence — it is coherent —
  but not for empirical effect size).
- The `production/` JSON subdir and `liquidity_sweep.yaml.gated` variant (not in the live set).
- Live-path parity of each inert feature (only store-side verified, except instability).

## 8. Files

- **Written**: `docs/knowledge/strategy_book_review_2026_07_10.md` (this file ONLY).
- **Read-only**: all 17 v14rq YAMLs, `configs/archetypes/*.yaml`, `engine/archetypes/{logic.py,
  archetype_instance.py, structural_check.py}`, `engine/integrations/isolated_archetype_engine.py`
  (dedup), `results/champion_v14_risk/comp_full16/full/{trade_log.csv, performance_stats.json}`,
  `data/features_mtf/BTC_1H_FEATURES_V14_FULL_LIVE_PATH.parquet`, knowledge docs listed in §2.
- **Production code/configs: UNTOUCHED. No git commits made.**
