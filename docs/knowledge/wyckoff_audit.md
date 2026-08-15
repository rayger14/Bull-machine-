# Wyckoff Detection Audit & Fix (2026-02-20)

## Pre-Fix Audit: 3/14 (21.4%) Hit Rate

| # | Type | Date | Price | Pre-Fix | Root Cause of Miss |
|---|------|------|-------|---------|-------------------|
| 1 | SC | Dec 2018 | $3,200 | MISS | BC fired instead (misclassification) |
| 2 | SOS | Apr 2019 | $5,000 | MISS | Nothing detected |
| 3 | SC | Mar 2020 | $3,800 | HIT | conf=0.643, 11h offset |
| 4 | BC | Apr 2021 | $64,895 | MISS | No upper wick rejection at euphoric top |
| 5 | AR | Apr 2021 | $47,000 | HIT | conf=0.928, 38h offset |
| 6 | SC | May 2021 | $30,000 | MISS | Nothing detected |
| 7 | Spring | Jun 2021 | $29,000 | MISS | Nothing detected |
| 8 | SOS | Aug 2021 | $42,000 | MISS | BC fired instead (misclassification) |
| 9 | BC | Nov 2021 | $69,000 | MISS | Nothing detected |
| 10 | SOW | May 2022 | $35,000 | MISS | SC fired instead (misclassification) |
| 11 | SC | Jun 2022 | $17,500 | MISS | wick_ratio=0.37 < 0.4 threshold |
| 12 | Spring | Nov 2022 | $15,500 | HIT | conf=0.616, via SC (5h offset) |
| 13 | SOS | Jan 2023 | $21,000 | MISS | BC fired instead (misclassification) |
| 14 | Spring | Jan 2024 | $38,500 | MISS | SC/AR fired instead |

## Post-Fix Validation: 12/14 (85.7%) Hit Rate

| # | Type | Date | Price | Post-Fix | Conf | Offset | How Fixed |
|---|------|------|-------|----------|------|--------|-----------|
| 1 | SC | Dec 2018 | $3,200 | **HIT** | 0.660 | 24h | Wick gate removed |
| 2 | SOS | Apr 2019 | $5,000 | **HIT** | 0.471 | 4h | No-context SOS fallback |
| 3 | SC | Mar 2020 | $3,800 | **HIT** | 0.795 | 11h | Already worked |
| 4 | BC | Apr 2021 | $64,895 | **HIT** | 0.867 | 16h | Wick gate removed + close conviction |
| 5 | AR | Apr 2021 | $47,000 | **HIT** | 0.873 | 10h | Already worked |
| 6 | SC | May 2021 | $30,000 | **HIT** | 0.732 | 12h | Wick gate removed |
| 7 | Spring | Jun 2021 | $29,000 | **HIT** | 0.412 | 0h | Spring_B relaxed gates |
| 8 | SOS | Aug 2021 | $42,000 | **HIT** | 0.352 | 20h | No-context SOS fallback |
| 9 | BC | Nov 2021 | $69,000 | **HIT** | 1.000 | 48h | Wick gate removed + close conviction |
| 10 | SOW | May 2022 | $35,000 | MISS | — | — | Still misclassified as SC |
| 11 | SC | Jun 2022 | $17,500 | **HIT** | 0.446 | 21h | Wick gate removed (was 0.37 < 0.4) |
| 12 | Spring | Nov 2022 | $15,500 | MISS | — | — | Lost: SC/BC/AR/ST fire but no Spring |
| 13 | SOS | Jan 2023 | $21,000 | **HIT** | 0.382 | 24h | No-context SOS fallback |
| 14 | Spring | Jan 2024 | $38,500 | **HIT** | 0.340 | 14h | Spring_B relaxed gates |

## 6 Fixes Applied to `engine/wyckoff/events.py`

1. **SC: wick gate → confidence modifier** + 50-bar lookback for range_position
2. **BC: wick gate → confidence modifier** + close conviction for euphoric tops + 50-bar lookback
3. **Spring_A: relaxed defaults** — wick 0.50→0.30, vol_z 0.8→0.5, breakdown 1.5%→1%
4. **UT: relaxed retreat gate** — factor 0.5→0.75, margin 2%→1.5%
5. **UTAD: RSI as confidence modifier** not hard gate, min 70→65
6. **SM: SOS/SOW no-context fallback** with 0.5x confidence penalty

## Event Count Changes (1H timeframe)

| Event | Before | After | Change |
|-------|--------|-------|--------|
| SC | 99 | 832 | +740% |
| BC | 82 | 909 | +1,009% |
| AR | ~50 | 338 | +576% |
| ST | ~200 | 8,330 | +4,065% |
| SOS | 15 | 221 | +1,373% |
| SOW | 11 | 28 | +155% |
| Spring_A | 0 | 12 | NEW |
| Spring_B | ~100 | 107 | ~same |
| UT | 0 | 8 | NEW |
| UTAD | 0 | 8 | NEW |
| LPS | ~5 | 5 | ~same |
| LPSY | ~8 | 8 | ~same |

## Remaining Issues (2 misses)

1. **SOW misclassification (#10)**: SOW (high vol + low range_pos + breakdown) overlaps with SC detection. Would need directional disambiguation (price context: near recent highs → SOW, near lows → SC).
2. **Spring #12 regression**: Was detected pre-fix via SC proxy, now SC/BC/AR/ST all fire but Spring doesn't. The event itself may be too shallow for Spring_A/B thresholds.

## OOS Validation (2026-02-20) — No Overfitting

### Forward Return Analysis
- **SOS**: +0.85% at 48h, +1.72% at 1w — genuine alpha (59.7% positive)
- **Spring_A**: +4.0% at 1w — strong signal (n=12)
- **LPS**: +8.3% at 1w — strongest but tiny sample (n=5)
- **SC/BC**: No predictive power (random noise). Value is as state machine scaffolding.
- New-only detections (post-fix) are NOT worse quality than old ones.

### Additional BTC Events (14 new consensus)
- Strict hit rate: 3/14 (21.4%), with price misses: 6/14 (42.9%)
- Rare event detectors (LPS=5, LPSY=8, UTAD=8 total) essentially non-functional
- Distribution-side events systematically worse than accumulation-side

### ETH Cross-Validation (completely independent asset)
- **5/6 known ETH events detected** without retuning (83.3%)
- COVID SC, ATH BC, LUNA SC, FTX SC, Recovery SOS all caught
- Only miss: Jun 2022 Spring (Spring_A too strict for ETH volume)

### Verdict: No overfitting. Alpha comes from SOS/Spring/LPS.

## Impact on Trading

- **Spring archetype unlocked**: PF=0.86 → PF=1.62 (now profitable, +$14.4K OOS)
- **Confluence_breakout unlocked**: PF=0.93 → PF=1.80 (+$7.8K OOS)
- **Exhaustion_reversal unlocked**: PF=0.89 → PF=1.97 (+$6.6K OOS)
- **order_block_retest improved**: PF=0.72 → PF=3.06 (+$7.8K OOS)
- Core 6 archetypes maintained profitability

## MTF Parity Fix Results (same session, before detector fix)
- Before: tf1d_wyckoff_score=0.5 constant, M1/M2=0, tf4h missing
- After: Real computed values from resampled 4H/1D Wyckoff detection
- Impact: PF 1.28→1.38, Return +44.9%→+78.8%, Sharpe 1.39→1.59

---

## 2026-08-04 — spring_b look-ahead fix + no-context fallback + directional phase (V20)

**Bug (fixed):** `detect_spring_type_b` read future closes (`close.shift(-offset)`,
offset 0..2) with a "shift result back" comment whose shift was never written —
batch runs repainted; live could not reproduce (future bars absent → NaN). Fixed to
the spring_a/UT candidate/confirm pattern: fires on the confirmation bar, causal.
Guard: `tests/test_wyckoff_causality.py` (batch-vs-truncated-walk agreement for all
event columns — failed on old code exactly at wyckoff_spring_b; 4/4 pass post-fix).

**Sparsity root cause (fixed):** the state machine discarded any spring/UT without a
previously validated SC→AR structure, and SC's triple-extreme gate (volume_z>2.5 AND
range_pos<0.2 AND range_z>1.5) almost never validates one. Springs: 9 events / 8.5y.
Fix: extended the sanctioned SOS/SOW no-context fallback (fire at 0.5x confidence,
`sm_no_context_fallback`, default ON) to spring_a/spring_b/UT/UTAD.
Batch counts: spring_a 9→77, spring_b 54→169 (now causal), ut/utad 6→19; all other
detectors bit-unchanged. ~29 springs/year — populated, no explosion.

**Directional phase (new):** `wyckoff_phase_dir` ('C_accum'/'C_distrib'/...) +
`wyckoff_context` exported — the bare letter conflates accumulation/distribution
(both ACCUM_SPRING and DISTRIB_UT map to 'C'; only ~37% of C/D bars were
accumulation-direction). mutual_exclusion was already default-on (2026-06 fix).

**Consensus-14 harness (honesty):** label-level recall 0/14 → 0/14. The curated
events are DAILY-scale structures; on 1H the engine correctly labels e.g. the
2024-08-05 crash low ($49,000) as SC, not Spring. Even V12's columns score only
4/14 today. Harness has a granularity mismatch — needs 1H-scale ground truth.
Event-agnostic: 8/14 windows contain validated events (unchanged), with new
spring/UT detections inside 2 of them.

**V20 store:** `BTC_1H_FEATURES_V20_WYCKOFF.parquet` = V18_ROTATION + wyckoff
family regenerated via live replay (`scripts/rebuild/patch_v20_wyckoff.py`, 6
chunks; store==live parity by construction). Store validator: 8 PASS / 0 FAIL.
NOTE: the v15-pattern stitch dropped object-dtype event bools — meaning V15/V18's
event booleans were V14-era values all along; v20 stitch coerces them properly.

**Champion co-move (V18 vs V20, champion_paper.json, position-level):**
- TRAIN 2018-22: 796→794 pos, PF 1.193→1.192, $98,535→$97,566 (−1.0%, noise), DD −29.8→−30.7%
- HOLDOUT 2025-26: 255→263 pos, PF 1.028→1.114, $2,867→**$12,159** (+$9.3K), DD −16.8→−16.9%
- VERDICT: PASS (train within noise, holdout strongly improved) → fallback ships ON.
- Composition caveat: holdout gain is bleeders bleeding less (liquidity_sweep +$4.9K,
  TWT +$4.1K, LC +$2.7K, FC +$1.8K) while wick_trap gives back −$5.0K and spring
  −$1.9K (fusion/dedup reshuffle from repopulated wyckoff scores). Book-level and
  validated-side net positive; watch wick_trap live share after any deploy.

### 2026-08-04 addendum — Wyckoff Insider's actual practice (user-sourced, public posts)

1. **Springs do NOT always require a prior SC** — valid inside reaccumulation ranges
   (mid-uptrend pauses) without a fresh climax. EXTERNALLY VALIDATES the
   sm_no_context_fallback change. His practice ≈ "defined range + poke-and-recover",
   suggesting a future refinement: range-existence gate (e.g. tf4h_coil / band
   tightness) instead of unconditional no-context fallback.
2. **Volume tells**: he prioritizes structural recovery + confluence (MSS, return-to-
   zone, LPS+Bojan) over strict volume formulas. Entry safety ranking: Spring/Test
   safest > Bojan-low-in-Spring (aggressive, high R:R) > LPS (riskier re-entry).
3. **Timeframe hierarchy (clear)**: HTF (D/W) = range, bias, permission; MTF =
   structure/Bojan/LPS refinement; LTF = trigger. Our fixed 0.5/0.3/0.2 MTF blend in
   _get_wyckoff_score is reasonable but suboptimal vs HTF-permission-first gating.
4. **Model 2 is NOT publicly mechanizable** — named schematic (return-to-zone, LPS+
   Bojan, MSS on LPS, TT targets) with no bar-level public spec. Do NOT attempt to
   rebuild mechanically (PO3-shell lesson). Our tf1d_wyckoff_m2_signal (~26% of bars)
   provenance is fuzzy — treat with skepticism.
5. **Ground truth**: his labels exist only as chart images. Transcribing 20-30
   (spring bar + range + outcome) = highest-leverage addition — would replace the
   granularity-mismatched consensus-14 harness.

### 2026-08-04 addendum 2 — M2 DECODED + first real ground-truth recall test

**CORRECTION to addendum 1 item 4: Model 2 IS publicly mechanizable.** His chart insets
are the canonical **Accumulation Schematic #2 (Bogomazov/Pruden, WyckoffAnalytics)**:
Phase A: PS → SC → AR → ST | Phase B: "ST in Phase B" (undercuts SC support = his
"ST-B") | Phase C: **LPS above support — NO SPRING** | Phase D: LPS → SOS → BU/LPS |
Phase E: markup. M1 = Schematic #1 (spring in phase C). His caption: "Not every
accumulation has a spring. In Schematic #2, smart money absorbs supply more quietly."
He confirms M2 usage with: SMT on the low, return-to-zone, LPS+Bojan, MSS on LPS.
Fractal: same schematic on 15m BTC (8-day range), 3D/4D USDT.D (2-year range), 45m alts.

**Ground truth v0 — transcribed from his BTC M2 chart (posted Jul 03 2026, Bybit 15m):**
PS 06-25 ~$59.0K | SC 06-26 ~$57.9K | AR 06-26 ~$60.8K | ST 06-27 ~$58.3K |
ST-B 07-01 ~$57.7K (undercuts SC) | LPS 07-01 ~$58.25K (Bojan 58,217-58,523) |
SOS 07-02 ~$60.8K | BU/LPS 07-03 ~$59.6K. (Chart images archived from pbs.twimg.com;
more transcribable M2 charts listed in the source thread.)

**Recall test (fixed detector, production _CFG_1H, real Binance klines):**
- 1H bars: ~0/8 — detector effectively blind at 1H to an 8-day structure (windows
  are 15-50 bars; the range is 120+ 1H bars). phase_dir neutral throughout.
- **4H bars: front half CAUGHT** — SC 06-25 conf 0.66 (his SC 06-26), AR conf 0.90,
  ST twice, and phase_dir = A_accum for the ENTIRE structure (correct direction!).
  Back half (ST-B / LPS / SOS): NOTHING.
- **Structural cause of the back-half miss: our sequencer has NO phase-C LPS path** —
  it only allows LPS *after* SOS (BU/LPS). In Schematic #2 the LPS comes BEFORE the
  SOS. M2's signature moment is literally unrepresentable in the current state machine.

**Next-upgrade queue (pre-registered studies, in order of evidence):**
1. **M2 sequencer path**: allow ST-in-B (undercut without invalidation), phase-C LPS
   from AR/ST states, SOS following LPS. All constituent detectors already exist.
2. **HTF-native detection**: his structures live at 4H+ scale; 1H windows are myopic.
   4H already catches phase A/B — supports the HTF-permission hierarchy (addendum 1).
3. Expand ground truth from the remaining archived M2 charts (USDT.D 3D/4D series).

### 2026-08-04 addendum 3 — M2 sequencer path BUILT, champion-gated OFF

Implemented Schematic #2 in the state machine (states ACCUM_LPS_C / DISTRIB_LPSY_C,
phase 'C' direction-aware; phase-B work required before the phase-C LPS; one per
structure — first draft without these constraints exploded LPS 16→2,727 (170x);
constrained: 16→452, ~53/yr). Tests: tests/test_wyckoff_m2_sequence.py (5, incl.
default-off + straight-from-AR rejection). On WI's labeled BTC chart (4H): ONE
phase-C LPS conf 0.93 + sustained C_accum through the full 8-day structure (V21
store: C_accum coverage 2,859→11,147 bars).

**Champion co-move (V21 = V20 + M2, champion_paper.json, position-level):**
| store | train PF / PnL | hold PF / PnL |
|---|---|---|
| V18 (orig) | 1.193 / $98,535 | 1.028 / $2,867 |
| V20 (fallback) | 1.192 / $97,566 | **1.114 / $12,159** |
| V21 (+M2) | **1.219 / $110,609** | 1.066 / $7,001 |

V21 beats V18 in BOTH windows, but vs the validated V20 baseline the M2 increment is
**train +$13.0K / holdout −$5.2K** — the train-up/OOS-down gradient (Rule 9 fail).
**VERDICT: sm_m2_path default OFF** (evidence-gated; code+tests kept, opt-in).
V20 remains the candidate store-of-record; V21_M2 kept as research artifact.

Open question for a future study: C_accum as a CONTEXT/bias feature (sizing boost
when phase_dir=C_accum) rather than through fusion-score reshuffling — the failure
mode here was fusion/dedup reshuffle, not the phase reading itself (which was
CORRECT on WI's chart). Prior: boosts 5/5.

### 2026-08-04 addendum 4 — wyckoff_phase_boost VALIDATED (the correct test)

User correction applied: judging M2 detection through fusion-mediated book PnL was
the wrong instrument (Lesson #54 — fusion is noise). Correct instrument = direct
predictive test + boost channel (boosts 5/5).

**Direct test:** V20-champion LONG entries inside phase-C accumulation: PF 2.07
train / 1.50 hold vs 1.16 / 1.09 outside (n=52/19, co-move both eras). Bar-level
C_accum alone sign-flips across eras → qualifier, not standalone predictor.

**Architecture:** `sm_m2_context_only` SHADOW phase — M2 state machine drives
wyckoff_phase_dir ONLY; no event emission, no state mutation. Guard-verified:
V22_CTX store differs from V20 in ONLY {phase_abc, phase_dir, sequence_position};
all 29 event/score columns bit-identical; C_accum on the same 11,147 bars as full
M2. (Guard also caught + reverted an unintended legacy widening: repeat-BU/LPS
validation had leaked into every score column via replay. Tests were green — only
the byte-level store comparison saw it.)

**Boost 6** (`wyckoff_phase_boost`, 1.25x + scoped capex, ANY long archetype when
phase_dir=='C_accum', default OFF): battery on V22_CTX, champion_paper base:
- CONTROL: boost-OFF bit-identical to V20 champion runs (1738/596 rows) — clean attribution
- TRAIN: PF 1.192→1.208, $97,566→$106,175 (+$8,609), DD −30.7→−29.8 (better)
- HOLD:  PF 1.114→1.120, $12,159→$12,928 (+$769), DD −16.9→−17.4 (−0.5pp)
- Mechanical sanity: hold delta +$769 ≈ 25% × the $3.1K C_accum-entry PnL (exact)
**VERDICT: PASS (co-move both eras; boosts now 6-for-6). Deploy-gated on user.**
Caveats: hold boosted n=19 (<30, flagged); train −2 positions (capex margin
consumption); DD hold +0.5pp minor.

### 2026-08-05 addendum 5 — wyckoff_campaign standalone REJECTED (3rd full-story failure)

S12 "wyckoff_campaign" (C_accum phase + spring/PTI confirmation entry, swing_low_50
structural stop, Smart Exits V2, long-only) built on study/wyckoff-campaign (64f79a7),
full pre-registered protocol. VERDICT: REJECT — TRAIN −$3,825 PF 0.72 (bear years
bleed); OOS-A +$787 (n=7, thin), OOS-B +$4,280 PF 2.27 (n=10); pooled PF 1.07; in-book
ZERO trades (neutral-weight fusion never clears the champion threshold, $0 delta).
94% orthogonal to spring/wick_trap entries — not redundancy; the fresh entries are
simply not good enough alone. FAILURE MECHANISM = exit/stop geometry AGAIN: wide
structural stop → avg loss $1,096 vs avg win $401 with R-ladder-capped winners (same
killer as PO3-orchestration). THE SPLIT VERDICT vs the validated boost: C_accum
context makes EXISTING archetype entries better (boost passed, co-move) but does NOT
make raw spring-event entries inside C_accum a standalone edge. Chapter-awareness is
real; story-ENTRY is solved by existing archetypes; story-EXIT (ride the markup) is
the unsolved discretion gap → exactly WI questions #1 (TT/profit-taking formula),
#2 (failed accumulations), #6 (failed-LPS handling). Do NOT retune exits on this
rejected identity. Full-story mechanization now 0/3.

### 2026-08-05 addendum 6 — WI execution intel (user-sourced): the missing exit organ

Answers to the 7 questions (direct + labeled inference). The prize is his EXECUTION
recipe, which maps 1:1 onto why wyckoff_campaign v1 (and PO3-orch) bled:
1. **Exits (pre-mapped, banked, derisked):** TT defined IN ADVANCE (measured move /
   "2X" projection of the range). TP1 partial (~40% observed), then SL moves UNDER the
   newly-created LPS/structure (derisk to ~1/10), core runner held to full TT. Does
   NOT trail LTF structure once TT is mapped. "Clear invalidation, clear target."
2. **Failed accumulations:** near-zero public post-mortems (survivorship). Invalidation
   tells (inferred): no MSS/acceptance after LPS/spring; close outside range/DL; HTF
   flip. Failure = wait for the next (deeper) structure, not force the thesis.
3. **Which range:** HTF range first ("HTF range until proven otherwise"), at HTF POI /
   discount, with a clean Wyckoff (esp. M2) forming INSIDE it and room to the opposite
   side. Duration matters less than developed phases.
4. **Phase-C confirmation = ACCEPTANCE, not the event bar:** "Market structure break on
   LPS. That is the trade." Waits for closes, sometimes days. (Explains our detector
   firing Jun 26 vs his Jul 1 label.)
5. **Phase-B maturity:** discretionary; more tests = quality; no public bar count.
6. **Failed M2 LPS:** stop-out for the M2 TRADE (stop under the LPS structure — the
   model's invalidation, NOT the deep range low); planned M1 re-entry lower may remain
   if HTF intact. → per-model tight stops, layered demand.
7. **Cross-market confluence (SMT/USDT.D):** conviction TIER, not hard requirement —
   matches our boost framework exactly.
DIAGNOSIS CONFIRMED: v1 campaign died on inverted geometry (deep stop, full size,
R-ladder-capped winners). His geometry: tight model stop, banked partial, derisked
runner to a pre-mapped structural TT. → ONE pre-registered v2 with WI-faithful
execution (acceptance entry, LPS-structure stop, TP1+derisk+runner-to-TT).

### 2026-08-05 addendum 7 — WI batch 2: execution mechanics DECODED (direct evidence)

**#8 selectivity:** no exact number published; "When not trading is the edge"; campaign
trades = few high-conviction swings/year + scalps inside their ranges. Our ~950
trades/5y is confirmed off-scale vs the source methodology.
**#9 PO3 (VERBATIM): "A valid PO3 needs three things: A defined range. An aggressive
break of that range. A Wyckoff inside the manipulation phase."** Never standalone. Our
5x-rejected bare PO3 was structurally invalid by his own definition — diagnosis confirmed.

**Ambiguities RESOLVED (direct quotes):**
1. **Entry = return-to-zone (RTZ) after the MSS-on-LPS confirms** — "Return to zone
   inside the range. LPS + Bojan. Market structure break on LPS. That is the trade."
   MSS confirms; preferred fill is the retest back into the zone, not the breakout close.
2. **Post-TP1 stop = UNDER THE CREATED LPS (structure anchor, NOT breakeven)** — gold
   trade: "TP1 40%. SL movement under created LPS. Position fully derisked 1/10";
   "TP by first supply or LTF bojan high to derisk SL same place not entry". One-step
   structure trail anchored at the key created LPS; not a continuous every-higher-low trail.
3. **TT** = pre-computed measured-move-style projection tied to the range/M2 ("2X"),
   often BEYOND the plain range high/low, precise to decimals (74,384.7) — exact
   formula still not public. Partials happen earlier at supply zones / Bojan highs;
   core runner held to TT.

**From the charts (gold 2H Jul-2026 + BTC 2H Dec-2025):** TP1 anchor = FIRST OPPOSING
SUPPLY ZONE (structural, not an R-multiple). "UNF" (unfinished) W/M/5D candle levels
act as magnets/targets ("M UNF fixed on demand", "UNF 5D 86,147"). Level vocabulary in
active use: MM supply, M1 POI, HPS POI zone, inducement level, weekend $$$ (liquidity),
timeframe-scoped Bojan zones ("3D Bojan + M Bojan + 6M Bojan" confluence at one price).
Implication for v2 exits: TP1 should anchor to the nearest opposing structural zone
(swing_high/supply), stop trails under the post-entry created LPS — both codable.

### 2026-08-05 addendum 8 — WI batch 3 (final): Bojan/UNF decoded to public limit

**Bojan** ("Bojan = multi-timeframe precision"; source influence @Bojan_618 — wick
quality, opens without wicks that get "fixed", early/ugly lows):
- M1 Bojan = highest wick tags major liquidity / MM supply-demand. M2 Bojan = same
  structure connected to mid/HTF range highs/lows.
- HIGHEST CONFLUENCE STACK (verbatim): M2 (third candle) + M1 (fourth candle) +
  unfinished-candle push into liquidity + POC/Wyckoff + 0.5–0.618 fib.
- Dual nature: bar pattern AND persistent timeframe-scoped zone ("6M Bojan low" lives
  for months). Zone centered on the WICK TIP (body/open secondary).
- INVALIDATION: close beyond it — and it CONVERTS to an unfinished candle ("Monthly
  Bojan High invalidated. We now have a monthly unfinished candle.").
- Grammar (observed): Bojan LOWS = demand/entry confluence + SL anchors ("SL under
  45M Bojan low"); Bojan HIGHS = TP anchors / short triggers.
- NO public wick-% rule — our legacy bojan.py 70% threshold is OUR invention, not his.
**UNF (unfinished)**: candle missing a proper wick on one side / incomplete extreme
expected to be revisited ("fixed"). Tracked D+ (W/3W/5D/12M); prioritize HTF + those
aligned with Bojan levels/range extremes. UNF + Bojan highs + TT = his target ladder.
**TT formula**: NOT publicly recoverable (no worked example; decimals imply clean
2-anchor calc). Treat any implementation as hypothesis.
**RTZ**: zone = the LPS/Bojan demand zone or impulse origin; prefers return after MSS
("MS break > Time displacement"); no chase of unreturned displacement; fill depth unquantified.
**Sizing**: "2% risk setups" (matches our 2%); TP1 ~40% at first supply → SL under
created LPS → residual risk ≈ 1/10 of initial ("fully derisked 1/10"); builds
positions, no mechanical grid published.
**Vocab**: DL/DL2 = custom range/deviation tool ("DL2 always overrules negative
FIBS"); inducement = engineered trap level at MM zones; SMT = related-pair divergence
(ETH/BTC, dominance) at swing pivots, legs/TF not fully specified; "$$$" ≈ liquidity
(weak evidence); "6-candle type logic" = older range-construction rules, unrecovered.
**Failed structures**: permanent survivorship gap — compensate by self-labeling
broken ranges from raw history when building ground truth.

### 2026-08-05 addendum 9 — v2 gate CLOSED: WI geometry does not rescue campaign entries

Descriptive replay of all 100 reconstructed v1 episodes under the decoded WI execution
(model stop = event-window low −0.25 ATR; TP1 40% at nearest resistance; runner
derisked to ~1/10R; TT = swing_high_50; 168h cap): TRAIN n=57 PF 0.92 −$5,783;
OOS-A n=25 PF 1.44 +$12,443; OOS-B n=18 PF 0.67 −$6,862. Geometry WORKED mechanically
(avg win/loss flipped 0.37→1.7) but tighter stops cut WR to ~35% → net negative 2/3
windows. DOUBLE-CONFIRMED: the raw C_accum+spring entry has no standalone edge under
either geometry — it was never the exits. wyckoff_campaign v2 NOT built (gate did its
job for ~0 cost). The campaign vision's validated expression remains: existing
archetype entries + phase-context sizing (the boost). Caveat: TP1/TT proxies crude
(dist_to_resistance/swing_high_50); insufficient to overturn a 2-of-3-negative gate.

### 2026-08-05 addendum 10 — Bojan/UNF registry: built, Stage-1 gate REJECTED

Registry (engine/features/bojan_unf_registry.py, branch feat/bojan-unf-registry,
9dc083c): 1D/3D/1W wick-extreme Bojan zones + UNF magnets + close-beyond→UNF
conversion; causal (3-truncation no-repaint verified); sane counts (median 4
concurrent active Bojan-lows). KEPT as research infrastructure.
**Stage-1 gate FAILED**: near-Bojan-low champion long entries TRAIN PF 0.72 vs 1.24
away (−$13.4K, damage in 2019/2021/2022 bear years — falling knives), HOLD PF 1.46
vs 1.07 (bull tailwind) — Rule-9 sign-flip; regime-conditional, not structural.
Target-ladder respect weak (winners stall within 1 ATR of overhead Bojan-high/UNF
only 15%/12%). NO boost built (gate stopped it at descriptive cost).
Noted for a possible future pre-registered study (NOT run): WI never buys a Bojan
low blind — his usage is LPS+Bojan INSIDE an armed accumulation; the faithful test
is Bojan-low ∩ C_accum, but the intersection cohort is likely n<10 → underpowered
today; revisit when live data accumulates. Legacy test_wyckoff_events.py 5 failures
confirmed PRE-EXISTING on main (Feb-2026 recalibration fixture debt, unrelated).

### 2026-08-05 addendum 11 — WI batch 4 (final): HTF mechanics for the all-seeing eye

**States (his operative set — NOT the rigid 4-phase schematic):** IN_RANGE (default:
"HTF range until proven otherwise"), MANIPULATION (wick deviation of/inside range —
explicitly NOT a bias flip), MODEL_FORMING (M1/M2 accum/dist at the extremes),
CONFIRMED_BREAK (real candle-BODY structure beyond the range + acceptance),
TRENDING (post-break + successful retest/LPS). Plus premium/discount location.
Neutral until proven; "reacts to the model that is present."
**Precedence:** HTF governs bias + SIZE; MTF/LTF may authorize only small tactical
scalps before/against HTF confirmation. "No confirmation = no trade" (at size).
**Bias flip rule:** real candle-BODY break beyond the HTF range (wick = manipulation,
never a flip); Monthly/2W breaks weighted highest; acceptance vs rejection decides;
a COMPLETED M2 (LPS + MSS) at the extreme can shift local bias pre-break.
**Permission = DIAL (his own words' shape):** against-HTF → main size ~zero until
confirmation, scalps allowed; full size only after Wyckoff confirmation. Soft gate
for meaningful risk, dial for tactical risk. → matches our boosts-6/6 vs filters-0/9
evidence EXACTLY: the source's architecture is size-tiers, not vetoes.
**Early HTF-read invalidation:** ABSENCE of the expected model at the extreme is
itself information (no M1/M2/LPS/Bojan forming); body-break against; acceptance on
the wrong side; missing confluence (SMT/dominance silent); time-based expiry of the
expected window. "Data points alone are not a trade."

### 2026-08-05 addendum 12 — Eye Phase 2b: corrected tiers REJECTED on untouched fold; eye sizing-dial PARKED

The corrected map (DIP_IN_DEMAND > PULLBACK > EXTENSION) was validated against the
never-touched 2023-24 fold (champion battery, 475 long positions). RESULT: INVERTED —
EXTENSION was the best-powered, best-performing tier (TRENDING/CONFIRMED chase subset
n=57 PF 1.82; full tier n=239 PF 1.45) and DIP the worst (n=12 PF 0.46). Worse: the
ORIGINAL Phase-2 tiers held CORRECTLY on 2023-24 (ALIGNED_CONFIRMED 1.61 > FORMING
1.31) — the inversion that motivated the redesign was itself era-specific. Three eras,
three different orderings: 2018-22 strength-best, 2023-24 strength-best, 2025-26
strength-worst. NO stable cross-era size dial exists in the eye's tier family.
**DECISIONS:** (1) Eye sizing-dial PARKED — no Phase 3, and NO further tier re-slices:
the untouched fold is now spent; any additional redefinition tested on seen data is
self-deception. (2) The eye Phase-1 state machine (causal, validated timelines) is
KEPT as infrastructure. (3) What survives every era remains the narrow mechanism-
backed qualifiers only: the 1H C_accum phase boost (deployed), seller-flow, Bojan-wick,
breadth. **META-LESSON (write it in stone): era-stability is rarer than mechanism-
plausibility — 2-of-3-era signals died on the third era all week. Revisit the eye
when LIVE data (new detection, deployed 2026-08-05) accumulates a fresh honest fold.**

### 2026-08-05 addendum 13 — WI batch 5 (final): all four eye-failure diagnoses CONFIRMED

1. **HTF range = FIXED structural object** (W/M swing-anchored, redrawn ONLY on body-close
   MSS break; premium/discount measured against THAT). Our rolling-40D location feature
   is confirmed mis-built — in trends a rolling range trails price so "premium" degrades
   to "market has been rising" = regime beta = the era-flips we observed.
2. **Extension conflation confirmed**: his premium-side continuation entries REQUIRE a
   completed/completing model (BU/LPS, retest+acceptance, clear invalidation); naked
   strength = the chase he skips. "Deep in premium is where you start prepping longs,
   not chasing fear."
3. **Cross-market is a CORE bias input, not confluence**: "We're using Dominance as the
   main driver... clear PO3... strong directional bias"; "the dominance chart gives the
   blueprint, entries come from a valid Wyckoff on the crypto charts." His USDT.D M2/P03
   read called the exact 2025-26 regime our price-only eye missed.
4. **He does NOT size by regime**: size trigger = completed high-quality model
   (normal 0.5-1%, high-probability HTF+MTF 3-5%). Regime picks WHICH models are
   takeable; no regime size-dial exists. → the eye-as-sizing-dial was un-Wyckoff by
   construction; the correct architecture = a PORTFOLIO OF MODEL DETECTORS, each
   earning its own boost (exactly the 6/6 pattern; C_accum boost = first instance).
5. **One playbook across eras**; context (HTF structure + dominance) changes which
   models are high-probability. Era-instability is adapted to via structure+cross-market
   reads, not system switches.
USER NOTE: founding codebase also carries dormant Gann / time-cycle / price-zone
machinery (relevant to his T7/Gann time-windows + time-based model-failure rule) —
queued for inventory.
**ARCHITECTURE VERDICT (final): eye-as-dial retired permanently. Forward path =
(a) fixed-structural-range location feature (infrastructure fix), (b) dominance-HTF
structure as bias input — buildable NOW from engine/features/stables_rotation.py
deterministic dominance series (a NEW data dimension = honest historical test, not a
spent-fold re-slice), (c) more model detectors each earning boosts.**

### 2026-08-05 addendum 14 — Dominance-HTF structure REJECTED; rot_rising book-wide lead found

**Dominance-structure bias: REJECT** (branch study/dominance-bias, 6e518c0). Decomposed:
(a) the raw dominance DATA is real — co-moves with WI's narrative at 4/6 dated marks
(2022 bear rise +77%, 2023 fall −33%, late-2025 M2 rise +41%, Feb-2026 peak; both
misses at 2021 alt-season tops = our basket ≠ CRYPTOCAP USDT.D); (b) the pre-registered
READER failed — rolling-90d-max boundaries can't read a grinding ratio (94.7% NEUTRAL,
NEUTRAL at all 6 WI marks incl. the flagship M2→TT arc); (c) tier ordering 0/3 on PF
with era sign-flips (WARNING best in train, worst in hold, n≤19 tiers); (d) 91-100%
redundant with stables_rot_rising. No reader re-tune on seen folds (addendum-12
discipline) — a fixed-structural-range reader for dominance awaits a fresh live fold.

**BYPRODUCT LEAD (the real finding): `stables_rot_rising` is 3/3-fold consistent
BOOK-WIDE** — long entries with rot=0 vs rot=1: train PF 1.52 vs 0.81; oosA (untouched
dimension-framing) 2.11 vs 0.74; hold 1.51 vs 0.60. The deployed usage (wick_trap
exodus refusal) uses a NARROW slice of this; the general form — size UP longs when
rotation is NOT rising (money not fleeing to stables) — is boost-shaped, mechanism-
backed, validated-lineage, and descriptively consistent in all three folds. →
CANDIDATE BOOST 7: standard battery (boost on/off, co-move) is the confirmation step.

### 2026-08-06 addendum 15 — WI batch 6 (final): stand-down horizon, dominance reader spec, T7, POC, the 3-5% stack

1. **Stand-down rule**: multi-day dominance rises ALONE are secondary; he stands down on
   dominance STRUCTURE (developing model / MSS / key extreme). Our 3-day rot_rising is
   SHORTER than his horizon → Boost 7 stands on OUR 3/3 empirical evidence, not his
   authority; his structural version = the fresh-fold dominance-reader redesign.
2. **Dominance reader spec (guess #2 prevented)**: same toolkit as price (fixed HTF
   ranges, M2, Bojans, LPS, body-close breaks); W/multi-week/M structure governs bias;
   LTF only for timing. → future reader = fixed-structural-range machine on W/M dominance.
3. **T7/Gann decoded**: T±7 event rhythm (T−14→T−7 accumulate, T−7→T0 front-run,
   T0→T+7 hangover-short; macro events T+45/T+90); numbered windows (T45) carry ~3-day
   grace for the expected manipulation/model to form; expiry without model = invalidation.
   Mechanizable but needs event calendars — queued (ties to dormant Gann machinery).
4. **POC = point of control OF HIS DRAWN RANGE** (not multi-year composite) — a
   confluence/validity magnet inside the range, part of the top stack with 0.5-0.618 fib.
   Buildable as range-scoped volume profile — queued.
5. **The 3-5% risk tier** = HTF model + MTF trigger (LPS/Bojan/MSS+RTZ) + dominance/SMT
   agreement + clear invalidation/target + sensible location. **CONVERGENCE NOTE: our
   multiplicative boost stacking already mirrors this** — a long carrying seller-flow +
   breadth + C_accum (+ rot-calm if validated) ≈ 1.25^3-4 ≈ 2-2.4x base risk ≈ his
   high-probability tier, reached the same way: independent confluences each earning size.
   The architecture converged on his risk model without copying it.
PUBLIC-KNOWLEDGE EXTRACTION NOW COMPLETE (6 batches). Remaining WI-dependent items all
require either fresh live data (dominance reader, eye rematch) or new data infrastructure
(event calendars for T-windows, range-scoped volume profile for POC).

### 2026-08-06 addendum 16 — Stack-depth audit: binary conviction tier CONFIRMED 3/3; ladder rejected

Depth>=2 longs beat depth<=1 in ALL THREE folds: PF 1.78 vs 1.13 (train), 1.91 vs 1.30
(oosA), 2.02 vs 1.01 (hold); WR ~60 vs ~44%; robust to dropping B6 (strengthens oosA
to 2.46). STRICT ladder (0<1<2<3+) FAILS (1/3 folds; 3+ tiers n=10/11/2 noise).
Confounds honest: ~87% of deep-stack = wick_trap; outside wick_trap the depth signal
DIES in holdout (PF 0.41 n=6). So the true claim: "wick_trap flushes with multiple
confirming flow reads = the book's premium trade class" — the WI 3-5% tier exists in
our book but is narrow, not a universal ladder.
**Cap-clipping quantified**: 44% of oosA depth>=2 positions pinned at the $52.5K base
cap despite earning ~1.8x on legacy non-capex B1/B2 — top tier under-expressed.
COLLISION WITH PRIOR EVIDENCE (do not forget): unconditional legacy-capex was already
tested 2026-07-28 and REJECTED (holdout −$1.3K). The audit's narrower hypothesis
(capex legacy multipliers ONLY at depth>=2) is a DIFFERENT, untested claim — queued as
a potential WFO study, NOT actioned; per-archetype OOS + co-move required, and the
depth>=2 holdout cell is n=21 (<30) so it likely needs live data to be decidable.

### 2026-08-06 addendum 17 — Range-POC Stage-1: PASS 3/3 (first new-family pass; battery-gated)

WI's "close above POC = validity" tested on champion longs (rolling-60d range POC,
0.25% bins, causal, no-repaint 3/3): ABOVE-POC PF 1.31/1.44/1.31 vs BELOW 0.69/1.19/
0.73; avgR positive-above / negative-below in ALL THREE folds; n>=30 every cell; WR
gap ~11pts stable. Magnet split (near-POC) inconclusive as pre-registered-exploratory.
CAVEATS (load-bearing): rolling-range proxy may be trend-beta in disguise (the exact
addendum-13 failure mode) → REQUIRED follow-up #1 = ORTHOGONALITY battery (does
above-POC survive controlling for trend/EMA/range_position?); multiplicity caveat
recorded verbatim (Nth family on these folds — 3/3 necessary, not sufficient).
Follow-up order: orthogonality → standard boost battery → someday structural-range POC.
Branch study/range-poc @ 9270bf0. Nothing wired.

### 2026-08-06 addendum 18 — Boost 7 (rotation-calm book-wide) REJECTED: crowd-out, not signal failure

Battery (branch study/rot-calm-boost, 70e0815): TRAIN +$13.4K ✓, OOS-A +$21.9K ✓,
OOS-B −$4.7K PF 1.120→1.069 ✗ → co-move FAIL. First boost to fail a battery.
**The per-trade edge is CONFIRMED — the ACTION failed.** Exact decomposition: resize
channel mechanically perfect (=0.25× boosted-cohort PnL to the dollar, POSITIVE in all
three windows incl. +$6.0K in OOS-B); the kill was CAPEX CROWD-OUT — the boost fires
on ~55-59% of ALL longs (rot=0 ≈ 51% of bars), a near-book-wide 1.25x leverage dial
that consumes shared margin: OOS-B lost 30 profitable would-be positions (+$6.0K)
to gain 19 losers (−$4.7K), reshuffle −$10.7K > resize +$6.0K.
**NEW HOUSE RULE (learned): boosts must be SURGICAL — condition base-rate matters.
Boosts 1-6 fired on tiny cohorts; a >50%-of-book condition is a leverage change in
boost clothing and competes with itself for margin.**
Legitimate follow-ups (mechanism-driven, not fold re-slicing; pre-register fresh):
(a) narrow the cohort: rot=0 ∩ a structural qualifier (e.g. C_accum or discount
range_position) to make it surgical; (b) risk-budget-neutral variant (hold gross
exposure flat). Overlap note: complementary to deployed exodus-refusal (opposite
sides of the same signal; no double-boosting found).

### 2026-08-06 addendum 19 — T-window event study: KNOWLEDGE-ONLY (nothing passes 3/3)

First empirical test of WI's event rhythm (branch study/t-windows, f73883f). ALL gated
results PARTIAL 1-2/3 — the same era-flip signature: FOMC/Conference rhythm holds in
2018-22 + 2025-26 but INVERTS in the 2023-24 bull (hangover out-drifts front-run);
CPI inverts in 2018-22. FOMC T0-day pop is real (+0.80%/24h in 18-22) but that's not
the WI asymmetry. T+45/T+90 "echo" = confounded (median FOMC gap 42d → windows overlap
the NEXT meeting; mostly bull beta). Nothing promoted: the suggestive cell (hangover
down-weight) is filter-shaped (0/9) AND fails the gate AND a front-run boost violates
the addendum-18 surgicality rule (15-46% cohorts). Calendar retained as infrastructure.
HONEST GAP: WI's actual rule is MODEL-CONDITIONAL ("does the expected model form in the
window") — untested; window∩model cohorts are n<10 today → needs live data accumulation.

### 2026-08-06 addendum 20 — Boost 7b (surgical) REJECTED: crowd-out is capex-margin timing, NOT cohort size

Boost 7b (rot=0 ∩ C_accum, branch study/rot-calm-surgical @ c0b0d77): surgicality
PASSED (2.0-2.7% of longs, n=16/13/7) yet REJECT — reshuffle residual still dominates
(train −$8.0K vs resize +$7.2K; OOS-B −$5.2K vs resize +$0.06K; OOS-B fully explained:
7 boosted positions' extra scoped-capex margin knocked out 9 other positions worth
net +$1.8K and admitted 4 losers worth −$3.4K). AND the intersection's per-trade edge
was a TRAIN ARTIFACT (+$28.8K train cohort PnL vs +$245/+$253 OOS).
**META-FINDING (house rule update): crowd-out originates in SCOPED CAP-EXEMPTION
consuming shared margin during concurrency pileups — even a 2.7% cohort triggers it.
Cohort size does not neutralize it; the honest isolator is RISK-BUDGET-NEUTRAL
sizing (flat gross exposure) — untested, and per-trade OOS edge ~$250 means likely
underpowered until live data accrues.** Consistent with the live concurrency-pileup
watch-item (live_emergent_mining_2026_07_21). Rotation-calm thread now fully closed
(book-wide + surgical both rejected); the signal survives only in its deployed narrow
forms (exodus refusal, and descriptively in the depth>=2 tier).

### 2026-08-06 addendum 21 — POC: orthogonality PASS (finding kept), Boost 8 action REJECTED; the action channel is saturated

Part 1 GATE PASS (branch study/poc-orthogonality @ 146c2ba): above-POC separates
within trend strata in ALL THREE folds (dominant stratum PF 1.35/1.61/1.30 above vs
0.74/1.27/0.73 below; agreement with trend features ≤66.5%, phi ≤0.34) — a REAL,
non-redundant structural discriminator (opposite of the eye-location failure). Store
artifact noted: price_above_ema_50 ≡ ema_slope_50>0 (identical columns).
Part 2 REJECT: boost fires on 65-75% of longs (LESS surgical than Boost 7); PF drops
in all three windows; crowd-out residual −$22.9K/−$10.5K/−$9.4K swamps positive
resize (+$26.9K/+$14.9K/+$5.1K); DD worse −4.4/−4.8pp. Same capex-reshuffle killer.
**CAMPAIGN META-PATTERN (now 3 consecutive cases): real per-trade discriminators keep
being found (rotation-calm, depth>=2, above-POC — resize channels ALL positive), but
the CAPEX-BOOST ACTION CHANNEL IS SATURATED — the margin-constrained book cannibalizes
itself at concurrency pileups whenever more size is added, regardless of cohort size.
The validated Boosts 1-6 got through when the book was less boost-loaded; the marginal
capex boost now costs more in crowd-out than it earns in resize. REMAINING LEVERS:
(a) risk-budget-neutral sizing mechanics (re-weight within flat gross — an ENGINE
change, the one untested structural lever; would allow honest retests of all three
shelved discriminators), (b) surgical intersections (mostly n<10 now), (c) live-data
accumulation. Validated inventory unchanged: above-POC + depth>=2 + rotation-calm are
SHELVED DISCRIMINATORS awaiting an action channel that can express them.**

### 2026-08-06 addendum 22 — USER CORRECTION VALIDATED: unconstrained re-battery rescues Boosts 7 & 8

User called out that margin crowd-out is a paper-wallet artifact contaminating edge
measurement — CONFIRMED (branch study/unconstrained-boosts @ b52ade3): the $100K OFF
arms had 29/6/18 margin rejections; at $2M (dollar-identical positions) zero rejections,
position set invariant, residual ≈ $0 → pure signal measurement.
**Boost 7 rotation_calm = VALIDATED SIGNAL**: ΔPnL +$34.4K/+$29.5K/+$8.1K, ΔPF
+0.033/+0.067/+0.050 (3/3); cohort PF 1.46/1.94/1.52 beats book 1.19/1.37/1.11 (3/3).
**Boost 8 poc_validity = VALIDATED SIGNAL (weaker)**: +$32.8K/+$19.7K/+$5.7K, ΔPF+ 3/3,
cohort above book 3/3; caveats: rolling-range proxy, 68-76% cohort, MaxDD slightly worse.
**Boost 7b = rejected outright** (crowd-out workaround now obsolete; n=10-16 noise).
PRIOR "REJECTS" RETRACTED as capital-competition artifacts. METHOD RULE (permanent):
edge discovery runs on a NON-BINDING wallet; finite-capital expression is a separate
engineering question. DEPLOYMENT GATE REMAINING: risk-budget-neutral allocation study
(shrink base risk so boosted positions fit the margin envelope at $100K) before any
production wiring — naive capex deployment of these signals demonstrably self-cannibalizes.

### 2026-08-06 addendum 23 — POC live forensics INVERTS the premise: Boost 8 deploy HALTED

User-requested pre-ship forensics on 387 LIVE trades (Feb-Aug 2026, server state.json,
kline alignment 0.05% median): **above-POC longs were the LOSING cohort live** (n=188,
PF 0.56, −$24.7K) while below-POC ≈ breakeven (n=182, PF 1.01, +$0.4K) — the OPPOSITE
of the 3/3 historical validation (above 1.31/1.44/1.31 vs below 0.69/1.19/0.73).
Boost-8 counterfactual on the live sample: **−$6.2K** (would have amplified losers).
Exit counterfactuals: magnet-target headline +$20.8K is a hold-168h-longer LOOK-AHEAD
artifact — the runnable version is −$508; invalidation-stop +$1.8K but WR down and 7
winners→losers. 12 losers→winners exist only under the look-ahead version.
Confounds noted: one regime; starved-wallet era; oi_divergence above-POC leak −$6.3K
(known leak) inside the losing cohort. STILL: freshest data contradicts the premise =
the era-instability signature (addendum 12) landing on POC. **DECISION: Boost 8 stays
OFF/unshipped. Optional middle path: merge the FEATURE (computed+logged live, flag off)
so forward live data adjudicates the above/below cohorts with the healthy wallet.**
The parity-proven build (feat/live-poc-boost8) remains ready either way.

### 2026-08-06 addendum 24 — POC inversion diagnosed: NOT the entry — a book-wide LIVE reward:risk collapse

Deep diagnostic (370 live longs + history POC recompute) REFUTED both my hypotheses:
- Trend-proxy contamination: REFUTED — above-POC beat below-POC within NON-uptrend bars
  in all 3 historical folds (train 1.16 vs 0.64, oosA 1.13 vs 0.91, hold 1.23 vs 0.73).
- Context-flip (distribution inverts POC): REFUTED — in historical markdown/distribution
  above-POC still WON (markdown PF 1.24 vs 0.80; 4H-distribution 1.36 vs 0.75). Live
  inversion reproduces in NO historical bear slice.
**ACTUAL CAUSE (mechanically dominant): payoff-geometry / RR collapse in the LIVE book.**
Live above-POC avg-win/avg-loss |ratio| = 0.35 (win $270 / loss −$778) vs HISTORY 1.46
(win $1,622 / loss −$1,110). Live WIN RATE actually ROSE (61.7% vs 47.9%) — but every
live above-POC loser (72/72) is a FULL stop_loss while winners bank tiny 0.5R/1R
scale-outs. A 4x RR collapse, size-invariant, present in below-POC too (W/L 0.73) →
BOOK-WIDE, not POC-specific. oi_divergence leak = minor (~26%, −$6.3K).
**CRITICAL CONFOUND the diagnostic flagged:** "history won" = BACKTESTER exits;
"live lost" = PAPER RUNNER exits + real fills. The RR gap is partly a SYSTEM (exit-engine)
difference, NOT purely regime. THIS CONFLICTS with the 2026-07 "exit-capture gap was a
measurement artifact / live matches backtest" correction (commit a134fdd) — that audit
may have sampled a benign period; this markdown sample shows a large gap.
**DECISIVE NEXT TEST (running): replay the 370 live entries through the BACKTESTER's
Smart-Exits-V2 on the real kline path — same entries, only exit rules differ. RR recovers
→ live exit engine is broken (book-wide, fixable, biggest find of campaign). RR stays
collapsed → regime (markdown full-stops everything; wait it out).**
POC decision UNCHANGED: not shipping — but the reason is the RR problem, not a dead entry
signal (POC entry edge survives every historical control).

### 2026-08-06 addendum 25 — CORRECTION of addendum 24: the "RR collapse" was a scale-out-chunk artifact; exits are FINE

Decisive replay REFUTES addendum 24. The 0.35 avg-win/avg-loss "collapse" was produced
by counting each SCALE-OUT CHUNK as a trade (small partial-profit rows vs full-size
stops) — the SAME chunk-vs-position artifact as PO3's fake 80% WR (Lesson #19).
CORRECT position-level metrics (aggregate by position_id):
- LIVE long |W/L| = **1.14** (avg-win $886 / avg-loss $778) — healthy, NOT collapsed.
- Chunk-|W/L| ≈ 0.4 in EVERY regime, INCLUDING the +$36K 2024 year → the chunk metric
  is meaningless as a health signal; I mis-read it.
- Real BACKTESTER over the SAME 2026 markdown: position |W/L| 1.12, PF 0.97 ≈ live 1.14.
- Exit engine is SHARED CODE (coinbase_runner → V11ShadowRunner → exit_logic.py); grep-
  confirmed. Replaying backtester exits on identical live entries does NOT recover RR
  (lands 0.85, below live's 1.14). NO exit-engine bug exists. Fills neutral (live stops
  −2.9bps ≈ 3bps model; scale-outs +32bps favorable).
**TRUE cause of the −$24.3K live long loss = REGIME × ENTRY BREADTH:** 6-month markdown
compresses winner follow-through for everyone (backtester ~breakeven over the overlap),
AND the full junk book admits more marginal entries that full-stop (55% live vs 43%
gated backtester) + live ran ~2 months deeper into the drawdown than the store cutoff.
The only lever = tighten entry selection, which COLLIDES with the standing junk-book-
full-for-data decision (2026-07-23) → NOT proposed. This resolves a134fdd in the correct
direction: live DOES match backtest at the position level. MY addendum-24 escalation was
the error; the adversarial re-check caught it. POC decision unchanged (parked; entry
signal survives historical controls, live is one hostile regime + the chunk artifact).

### 2026-08-06 addendum 26 — Backtester trust audit: engine SOUND, one real entry-cost accounting bug

Independent differential test (hand-rolled numpy referee, vectorbt/backtesting.py not
importable): per-trade PnL reconciles **0.00%** to an independent implementation. PASS on
look-ahead (signal-at-close/fill-at-close, no future bars), same-bar round-trip guard,
scale-out-from-original-qty, margin↔equity reconciliation, float guards, exit-chain
preemption, and **intrabar stop-first PESSIMISM** (stops fire on wicks + fill at stop
level; targets need a close — conservative, no target-first inflation). Relative rankings
trustworthy.
**BUG FOUND (Finding 1, entry-cost path only, differential-confirmed — NOT a metric
artifact this time):** (1) entry COMMISSION is deducted from cash but omitted from
`trade.pnl` → headline `total_pnl` OPTIMISTIC ~2bps/trade (~$7-15K / 5-10% of the $132K
production figure). (2) entry SLIPPAGE double-charged: in the fill price AND as a separate
never-returned cash term → equity-curve/MaxDD PESSIMISTIC ~3bps/trade. Headline PnL and
equity-curve PnL are on inconsistent bases. Minimal fix (NOT applied): add entry-commission
term to `_close_position` pnl; drop the explicit slippage cash term from margin_cost (already
in fill price). CHECK: does live v11_shadow_runner share the pattern (parity)? Re-run the
production floors after. Regression harness kept: scratchpad/bt_audit/differential_test.py.
WORKFLOW VERDICT (answers user's vectorbt question): keep our engine as SOURCE OF TRUTH
(it carries live-parity + fusion/dedup/margin/Smart-Exits that libraries can't model); use
a clean library/hand-rolled backtester as a FAST IDEA-SCREENER for simple standalone
signals before engine integration → this seeds the "idea-lab" for testing extracted
knowledge (PO3, POC, all-seeing-eye, Bojan, WI exits) WITHOUT fusion/dedup/crowd-out noise.

### 2026-08-07 addendum 45 — Full-stack M2 unified (user-approved): REJECT as edge; RTZ filter is a real keeper; strict M2 is architecturally inert

User chose "full stack together" off the add.44 critique (M2/LPS + RTZ filter + time-validity +
USDT.D dominance reader). Pre-flight found 2 of 4 already built: dominance reader = CONFIRMED
REJECT (commit 6e518c0: 0/3 tier ordering, redundant with rot_rising — NOT rebuilt); the M2 HTF
state machine already exists (engine/features/eye_state.py @81c9b1f, causal no-repaint 4/4). The
eye GATE on champion had failed (CONFIRMED_BREAK/TRENDING tier INVERTS OOS: PF 1.74 train→0.47
hold; only ALIGNED_FORMING stable both eras) → M2 built to enter the LPS RTZ pullback into
ALIGNED_FORMING, NOT the extension. Branch study/unified-m2, pre-registered (910e0a6) before
measurement, self-test parity 0.00%, no threshold-fishing.

VERDICT: REJECT as edge (nothing ships). Three results:
1. STRICT M2 = 0 TRADES every era + live. ARCHITECTURAL, not tunable: in the eye machine
   `bull & MODEL_FORMING` = 3,263 bars but `+ recent bull CONFIRMED_BREAK (≤360b)` = 0 bars — the
   two states are TEMPORALLY DISJOINT by construction (MODEL_FORMING only marks fresh low-third
   accumulation, never a post-break retest). The M2 door literally cannot fire as designed. Fix
   would require a NEW eye state (explicit RETEST_HOLDING/LPS emitted after a confirmed break) —
   state-machine work, must NOT be fished on spent folds.
2. RTZ FILTER genuinely improves M1 (FIRST real structural win in a while): multi-era 2/3 vs v1's
   1/3 — TRAIN PF 2.02→2.38, OOS-A 0.73→1.11 (flips to PASS), OOS-B 0.15→0.29 (still fails);
   live-forward −$822→+$2,605 (4 trades). BUT M1 still 81% below-EMA200 pooled → DNA unchanged,
   just cleaner. Not sufficient alone (OOS-B markdown still loses).
3. M2-BROAD diagnostic (relaxed) points the RIGHT way: candidate context 83% ABOVE-EMA200,
   profitable where it fired (TRAIN 100% above, OOS-A 50%), 0 in OOS-B markdown (stand-down) —
   but only 7 trades pooled, 0 in OOS-B. Directional only (n<30), NOT proof.

Stand-down CONFIRMED (M2 took 0 in every markdown incl. live) but that's not edge. Champion actual
same live window: −$24,568 PF 0.72 (both bled; everything long-biased loses in markdown). Root
cause UNCHANGED from add.44: the missing organ is a regime stand-down DIAL, not another entry door.
KEEPERS: RTZ filter (real, partial); the M2-broad direction (above-EMA200 + markdown stand-down)
is what we want but too rare. NEXT LEVERS (both need FRESH data — folds spent): (a) new
RETEST_HOLDING eye state to make strict M2 fireable; (b) the regime stand-down dial (needs CMI
regime block materialized). Neither validatable on spent folds — forward-only.

### 2026-08-07 addendum 47 — Cross-asset (SPX 1990-2026): spring dip-buyer failure is INTRINSIC, not BTC. The trend-CONTINUATION door is the real cross-asset edge candidate.

User asked: test the long "all-seeing eye" model on SPX/MES — is the failure a BTC thing (Wyckoff is
an equities method; traders use it on BTC)? Fetched Yahoo ^GSPC: SPX_1D 9,216 bars 1990-2026 (dotcom
−49%, GFC −57%, COVID −34%, 2022 −25%), SPX_1H 5,090 bars 2023-26. Ported the LONG v2 model with the
SAME BTC-tuned params (NO retune; only timeframe H→D, struct range D→W), crypto inputs dropped
(stables→0, regime=price half only). Branch study/xasset-spx, self-test parity 0.00%.

VERDICT: the M1 SPRING DIP-BUYER's failure is INTRINSIC, cross-asset.
  - SPX hourly (bull regime that SHOULD flatter a dip-buyer): PF 0.56, −$3,208, 90% below-EMA200 —
    lost even in a bull.
  - SPX daily per-regime: the ONLY positive regime is the 1990s bull (+$8,219); aggregate PF 1.11 is
    the secular-uptrend/survivorship CONFOUND, not survival.
  - DISCRIMINATOR: pooled SPX BEARS n=4, WR 0%, ALL 4 stopped, avgR −1.02 — identical death signature
    to BTC OOS-B (5/5 stops). Same falling-knife DNA, second asset class. Where the spring fires in an
    equity bear it dies. (COVID/2022 fired 0: regime permission was OPEN 97-99% of bear bars — NO
    markdown protection; only the RTZ filter's sequence-timing incidentally blocked the fast/grinding
    bears. When a bear is slow enough for a spring to complete (dotcom, GFC), RTZ passes it and it dies.)
  => STOP trying to fix the spring dip-buyer. It is intrinsically markdown-fragile on every asset.

THE REAL FINDING (section 5, constructive): the PRICE-ONLY trend-CONTINUATION door — bull
CONFIRMED_BREAK → LPS pullback that HOLDS (RTZ), needs NO wyckoff store, NO distribution detection —
on daily SPX: n=24, PF 3.40, +$16,885, 92% ABOVE-EMA200, and STANDS DOWN in bears (dotcom/GFC/COVID
0 fires, 2022 1 fire). OPPOSITE character to the dip-buyer: buys STRENGTH after the break, self-
regime-filters (requires an up-break to fire → cannot fire deep in a bear). This is CONSISTENT with
BTC's M2-broad (add.45: 83% above-EMA200, profitable where fired, stood down in markdowns, but only
n=7 — starved on BTC, had room on SPX). Two asset classes now show the same self-filtering above-
EMA200 continuation character.

SYNTHESIS: the campaign's dip-buy/"spring at discount" thesis (PO3, unified M1, unified M2-strict,
short mirror) is CLOSED. The edge that keeps surfacing is its behavioral OPPOSITE: BUY THE PULLBACK
AFTER THE BREAK (trend continuation, above EMA200, self-regime-filtering) — the resilience the user
wants comes not from shorting bears but from a door that simply DOESN'T FIRE in bears and rides bull
continuations. NEXT (separate study, user decision): validate the trend-continuation door properly
(more trades, both assets, CPCV, forward). CAVEATS: still small n (7 BTC / 24 SPX), directional not
conclusive; SPX has survivorship confound but the bear stand-down (0 fires) shows it's SELECTIVE, not
just riding the uptrend; folds semi-spent.

### 2026-08-08 addendum 48 — Trend-continuation door: cross-asset validated on edge (4/4 assets, identical params, incl. uncorrelated Gold) but NOT on bear stand-down; too RARE to prove on history → forward-collection only

Validated the price-only breakout-retest door ("buy the pullback after the break") standalone on 4
assets with IDENTICAL BTC-tuned params, no per-asset tuning, pre-registered pass rule. Branch
study/trend-continuation (trend_continuation_door.py, run_trend_continuation.py). Self-test 0.00%/asset.
Note: eye_state computed OHLCV-only on EVERY asset incl. BTC (wyckoff cols dropped) → stricter than
the add.45 BTC flicker (n=7); BTC door now n=9, character-identical across assets. No volume used.

RESULTS (headline struct/flat):
  BTC   n=9  PF 2.56  +$3,740  100% above-EMA200  bear 3/3 clean  CPCV frac>1 80%
  SPX   n=25 PF 3.53  +$17,994 92% above          bear 3/4        CPCV 93%
  NDX   n=34 PF 2.82  +$16,336 97% above          bear 2/4        CPCV 100%
  GOLD  n=32 PF 3.51  +$21,275 97% above          bear 1/3        CPCV 93%   (KEY uncorrelated test)
  SPX1H n=8  PF 1.87  +$2,059  88% above (bonus, single bull)

VERDICT vs pre-registered rule (PF>=1.5 on >=3/4 AND >=80% bear stand-down AND CPCV mean>1):
  C1 edge: PASS 4/4 (incl. uncorrelated Gold 3.51) — identical params, real structural signal not a fit.
  C2 bear stand-down: FAIL/qualified — 9/14 = 64% strict zero-fire (86% lenient). The self-filter is
     REAL (breakout requirement blocks deep-bear entries) but LEAKS: bear-market-RALLY breakouts fire
     and EVERY bear fire LOSES; worst on Gold (bear pooled PF 0.10, −$2,999). 8 bear trades, all-asset,
     all net losers.
  C3 CPCV: PASS directionally (frac PF>1 80-100%) but n/fold 3-11 → not conclusive. (Mean PF inflated by
     zero-loss folds — ignore; use frac>1.)
  => NOT fully validated (fails C2). BUT a genuine cross-asset-consistent, 92-100%-above-EMA200,
     positive-expectancy REAL-BUT-RARE candidate; behavioral OPPOSITE of the dead spring (which dies on
     Gold −$1,596 and SPX-1H 0.56). Forward test BTC 2026-02→06-10: ZERO fires, clean markdown stand-down.

BINDING CONSTRAINT = frequency: n=9-34/asset over 8-36 yrs (~1-4 trades/yr) → HISTORY CANNOT PROVE IT.
Per the rule, next step = FORWARD PAPER-COLLECTION, not deploy. THE ONE DEFECT (bear-rally breakout
leak, worst on Gold) is EXACTLY what a causal BEAR-REGIME OVERLAY would plug → converges with the live
market-state-detection research thread (deep-research running). Plan: let research land → design regime
overlay to plug the Gold-style leak → then decide forward-collection of the COMPLETE door. Watch-item:
Gold-style bear-market-rally breakouts. This is the campaign's best surviving edge candidate.

### 2026-08-08 addendum 49 — Research verdict: "outsource Wyckoff detection" = NO (and it's the wrong target); the validated answer is a COARSE CAUSAL REGIME gate — which is exactly the door's missing overlay

Deep-research sweep (101 agents, 19 sources, 81 verified claims; synthesis step returned a corrupt
"test" stub — recovered manually from journal.jsonl). Answer to "can we buy/build/LLM better Wyckoff
phase detection": decisive.

(1) TURNKEY: NO reliable Wyckoff-phase API exists. Closest: getregime.com = coarse bull/bear/chop
REST API, RULES-based composite (SMA/funding/fear-greed/dominance/stables/DXY), no causal guarantee,
$0-149/mo — i.e. a hosted version of signals we ALREADY have, no better. smart-money-concepts pip pkg
(1.9k★) = SMC primitives (BOS/CHoCH/order-blocks/FVG) NOT Wyckoff phases — overlaps our structural range.
(2) ML/HMM: the quant standard, but detects VOLATILITY/TREND regimes (bull/bear/range/calm), NOT
accumulation/distribution. Failure modes (all confirmed): smoothed/Viterbi labels LOOK-AHEAD (only
FILTERED past-only is causal-safe); labels REPAINT on refit; non-stationarity needs retraining;
lag is IRREDUCIBLE (filtered bull→bear median 2-3d, up to 7d — a Bayesian limit); causal rolling HMM
agrees w/ offline only 56.6%, labels 30.7% bear vs 13.5%; regime-aware ML often FAILS Deflated-Sharpe
OOS. UPSIDE (key): as a RISK FILTER a lagging filtered HMM cuts DD Buy&Hold −40.1%→−17.1% vs
Oracle −16.8% — ~95% of perfect-foresight DD protection despite lag.
(3) LLM/VISION: NOT reliable — VLMs only work in persistent trends, poor in ranges (where you need it),
biased, no proven candlestick comprehension; multimodal LLMs underperform analysts; the one benchmark
(FinMR) was WITHDRAWN; a chart-CNN got 0.892 AUC but on 500 samples w/ look-ahead labels — fatal at our
14-label scarcity.
(4) DESKS: use COARSE regime detection, NOT Wyckoff phases — HMM on returns/vol, OR simple rules
(price>200EMA=bull; ADX<20 range/>25 trend; ATR percentile=vol), to GATE strategies asymmetrically
(block entries in bad regime, allow exits). Nobody serious uses fine Wyckoff phase labeling.
(5) DISTRIBUTION: genuinely harder AND partly structural for crypto — BTC's real top was covert whale
distribution (split across exchanges to evade detection), textbook signals ABSENT; post-ETF changes
degraded classic signals. CONFIRMS add.46 externally: our 40x starvation is partly real, not just a bug.
(6) RECOMMENDATION: BUY=no, LLM=no, fine-phase ML=no (label-starved + wrong target). BUILD a coarse
CAUSAL regime gate — RULES-based preferred (200EMA+ADX+ATR: causal-by-construction, no repaint, no
training data) over HMM (whose ~95%-oracle DD benefit isn't worth its lag+repaint+retrain overhead for
our case). Use as a RISK GATE not a return-timer. Avoid: smoothed/look-ahead labels (#1 trap), repaint,
ignoring lag, no-retrain drift.

PUNCHLINE FOR US: the user's premise ("if we knew Wyckoff phases exactly, issues solved") is inverted by
the evidence — fine phases are un-buyable, un-trainable (label-starved), un-LLM-able, and genuinely
absent at crypto tops; AND desks don't use them. The tractable + validated answer is a COARSE CAUSAL
regime gate — which is EXACTLY the overlay the add.48 trend-continuation door needs to plug its one
defect (bear-market-RALLY breakout leak, worst on Gold). The door already keys off EMA-200 (92-100% of
entries above it); the leak is short-term pops above structure inside a bigger bear → a HIGHER-TIMEFRAME
regime gate (weekly 200EMA / slow bear flag) plugs it. Rules-based, no new detection tech. Next decision
(user): build the coarse regime overlay onto the door → then forward-collect the complete system.

### 2026-08-09 addendum 50 — Regime-overlay REUSE test: the cheap owned flag does NOT robustly plug the door's leak; the real CMI labels are UNMATERIALIZED (all-NaN in study store). Correction + honest null.

Tested whether a coarse causal bear flag we ALREADY own plugs the add.48 trend-continuation door's
bear-rally leak. Inline (no agent). idea_lab/overlay_regime_test.py on study/trend-continuation.
Pre-registered flags (no fishing): ema200-falling over K∈{20,60} daily bars (targets "price ABOVE
ema200 but medium trend DOWN" = bear rally) on all 4 assets; existing CMI regime_label (BTC).

CORRECTION to add.49/prior turn: I claimed the CMI regime signal was "materialized in the store." WRONG.
regime_label / regime_risk_off_prob / regime_risk_on_prob / regime_crisis_prob / regime_confidence are
ALL 0% populated (all-NaN) in V22_CTX. The CMI RegimeService (engine/context/regime_service.py) exists
as code and runs live/in the fusion path, but its output was never written to the offline study store.
So FLAG-STORE could not be tested (labels all NaN) — the "thing we own" is not usable offline as-is.

EMA-SLOPE proxy result — MIXED / not a robust fix (do NOT deploy, do NOT fish K):
  SPX  : WORKS. SLOPE-20 removed 4 losers (PF 0.35, −$2,122 incl the 1 bear leaker) → KEPT n=21
         PF 3.53→6.22, +$17,994→+$20,116; bear leak −$1,140 → $0. Clean.
  NDX  : FAILS. removed 3 WINNERS (+$3,271), leak UNCHANGED (−$2,164). Net harmful.
  GOLD : FAILS (worst-leak asset). SLOPE-20 removed 2 winners (+$1,436), leak −$2,999 UNCHANGED;
         SLOPE-60 made leak WORSE. The Gold bear-rally losers fired while the 200-EMA was still
         RISING (fast rips) → a lagging trend-slope flag structurally cannot catch them.
  BTC  : no leak (0 bear fires); flag removed net-positive trades → mild harm, N/A.
So a naive EMA-slope regime gate is ASSET-INCONSISTENT: fixes SPX, hurts NDX/Gold. Not a universal plug.

PERSPECTIVE (important): the door is net-positive INCLUDING the leak — leak is ~7-14% of gross
(−$1.1K to −$3K) vs door gross +$16-21K/asset; headline PF 2.56-3.53 already contains these losers. The
leak is a MODEST DRAG, not a survival threat. The door already stands down in DEEP bears; only fast
bear-RALLY breakouts leak.

HONEST VERDICT: "do we already have a regime detector?" = YES (CMI RegimeService), but (a) its labels
are NOT materialized in the study data, and (b) a cheap proxy for it does NOT robustly fix the specific
leak. Properly answering "does our real regime detector plug the leak" requires MATERIALIZING the CMI
labels over history (run the live service across the store) — and per discipline that should be a
FORWARD evaluation, not fit on spent folds. Recommendation: do NOT over-engineer a regime overlay on
this data; the door survives the leak; treat the bear-rally leak as a forward-watch item; if we
forward-collect the door, log the live CMI regime_label alongside and evaluate the overlay on fresh
fires. Not fishing K to force Gold — that would be overfitting a −$3K tail on 26 years.

### 2026-08-11 addendum 54 — THE ONE STRATEGY assembled + validated on FRESH un-mined crypto: door PORTABILITY CONFIRMED (basket PF 2.50), boost stack partly inert, forward-collection is the only remaining proof
(addenda 51–53 live on branch study/htf-ltf-expansion: 51 BOMS-direction fix, 52 promising HTF-state
pulse, 53 expansion REJECT + verbatim gem recovery + ob_quality/eq_magnet weak 3/3 sizing pulse.)

Assembled the campaign's surviving synthesis into ONE named strategy (docs/knowledge/ONE_STRATEGY.md)
= WI's M2 continuation mechanized: weekly-anchored CONFIRMED up-break (PERMISSION, stand-down by
construction) → daily retest/LPS-hold (ENTRY) → conviction sizing (boosts, never gates) → banked-and-
derisked exits (TP1 40% at range high → BE → 60% runner to measured move). Params IDENTICAL to add.48,
no retune. Validated on FRESH, never-touched daily crypto (Coinbase spot — Yahoo v8 hard-429'd this IP,
Stooq JS-gated, Binance 451; vendor substituted, assets unchanged). Branch study/one-strategy off
study/trend-continuation; idea_lab/run_one_strategy.py + fetch_fresh_crypto.py. Self-test parity 0.00%.

RESULTS (headline struct/flat, rmult=1.0, identical params; 9 fresh assets + BTC ref):
  PF>=1.5 on 6/9 fresh (ETH 1.93 n12, SOL inf n3, LTC 3.26 n9, DOT inf n1, AVAX inf n1, LINK 9.89 n5);
  the 3 fails all n<5 (XRP 0.08 — but ~2.5yr Coinbase-suspension DATA GAP, artifact; ADA 0.28 n3;
  DOGE 0.00 n2). n>=5 subset: 3/3 pass. BTC ref PF 2.42 n10. Pooled above-EMA200 88%. Pooled BEAR-
  window fires n=8 PF 0.58 −$1,610 (the add.48 leak: net-losing minority, ~6% of gross, reproduced
  EXACTLY). Dead-spring baseline stayed the opposite (DOT −$7,949 / AVAX −$2,991 catastrophic).
  BASKET (fixed 1% risk): 40 raw trades → 32 independent episodes (±5d cross-asset), episode PF 2.68 /
  basket PF 2.50 / MaxDD −5.79% / ~4.4 trades/basket-yr; CPCV K6m2 meanPF 2.84, frac>1 100%, >=1.5 87%.

VERDICT vs pre-registered rule (majority PF>=1.5 AND agg episode PF>=1.5 AND bear-standdown consistent
AND no catastrophic n>=5): PASS on all four → PORTABILITY CONFIRMED. The door is a real cross-asset
edge, not a BTC/SPX/Gold fit.
HONESTY (binding): (a) cryptos are one macro factor — 40→32 episodes; INDEPENDENT evidence for the door
stays SPX/NDX/Gold; fresh cryptos test PORTABILITY + build the tradeable basket, nothing more.
(b) Rarity binds — only 3 assets reach n>=5; "inf" PFs are n<=3, directional; the BASKET is the unit of
evidence. (c) Boost stack partly INERT: eq_magnet at verbatim 0.1% tol shows ZERO ≥3-pivot clusters on
daily crypto (original was a 1H/intrabar store proxy) — NOT loosened (=fishing). ob_quality needs the
full 5-comp HOB pipeline — NOT computed (the add.53 quality-axis study, separate). So the live-computable
stack = fib-time ×1.25 only, mildly net-+ (~+12% basket PnL), a sizing tilt not an edge. (d) History
cannot prove a ~1–4/yr signal — the ONLY remaining honest validation is FORWARD paper-collection of the
basket. DEPLOYMENT PROPOSAL (not deployed): separate daily-cadence runner (NOT an 18th archetype — keeps
it off fusion/dedup/CMI), Coinbase INTX perp basket (BNB excluded — not on Coinbase), bear-rally leak as
a LOGGED flag never a filter. NOTHING SHIPPED; needs explicit user go per standing rules.

### 2026-08-11 addendum 55 — Fractal probe: the door does NOT scale down. 4H version = same trade count, edge gone (third death of "go faster"). WI's extra cadence = small tactical scalps, not more at-size trades.

User asked whether MTF/LTF sniper entries could raise the door's 4-5/yr cadence ("WI takes more trades
than that"). Two prior LTF attempts died (add.52-53 expansion; order_block_retest PF 0.75) but those used
DIFFERENT trigger definitions — the EXACT door geometry fractally scaled (4H exec, daily N=5 reanchor,
identical bar-unit params) was untested. Ran it (idea_lab/probe_fractal_4h.py, BTC 8.4y):
  DAILY (ref): n=9 (1.1/yr) WR 67% PF 2.56, OOS-A 2.84, 0 bear fires, 100% above-EMA200.
  4H FRACTAL:  n=8 (0.9/yr) WR 50% PF 0.93 (−$304), OOS-A COLLAPSES to 0.25 (−$2,341), 1 losing bear fire.
TWO decisive facts: (1) one TF down produced NO additional trades — the clean break→hold pattern is just
as rare at 4H; lower TFs add noise, not opportunity; (2) what it did produce was train-good (2.96) /
OOS-dead (0.25) — the same speed-up death signature, third confirmation. THE EDGE LIVES AT THE
DAILY/WEEKLY SCALE, PERIOD. Do not revisit sub-daily variants of the door.
DOCTRINE ANSWER to the cadence question: WI's own precedence rule ("HTF governs bias + SIZE; MTF/LTF may
authorize only small tactical scalps; no confirmation = no trade AT SIZE") — his extra trades are SMALL
scalps around rare at-size cores. Our door = the at-size book, faithfully. Honest frequency levers that
remain: BREADTH (the 10-asset basket ≈ 4-5 at-size trades/yr, add.54) and — if ever — a tactical small-
size layer would need its OWN validated edge, which three tests now say does not exist in our detector
set at sub-daily scale.

### 2026-08-12 addendum 57 — WI batch-6 answers: CAMPAIGN topology (1-3 sized entries per model, bank→re-enter) + the GANN TIME layer (verbatim mechanics). Our cadence gap = entries-per-model, NOT range scale.

User relayed WI's answers to the cadence questions (add.56 aftermath). VERBATIM knowledge, recorded:

Q1/Q2 CAMPAIGN MANAGEMENT: he does NOT buy once and sit. Active management around a core: banks
partials (TP1 40% — matches our spec), moves stop under the NEWLY CREATED LPS/Bojan (structure-event
trail, not pivot-ATR trail), fully derisks to ~1/10 (not our 60% runner), takes LATER LPS/RTZ/Bojan
entries INSIDE the same larger structure. Typically 1-3 (sometimes more) SIZED entries per major
model: M1 spring/test primary + LPS re-entry; M2 LPS primary (can be multiple) + BU. => OUR
SINGLE-ENTRY-PER-MODEL IS THE CADENCE GAP, not the range scale.
Q6/Q7 CADENCE GROUND TRUTH: no published counts; one week "BTC gave 6 clean setups, took 3";
at-size HTF campaigns = few (high-single to low-teens/yr INCLUSIVE of the 1-3 entries each);
major completed HTF models = 2-5 quality/yr (multi-week to multi-month ranges, 2W structure,
monthly levels, range kept "until proven otherwise" by BODY structure break). Our ~1 event/yr
detector is "right ballpark" (low end); the multiplier is ENTRIES PER MODEL + re-entries after
banking + a tactical MTF layer (which for OUR detectors is proven dead, add.52-56 — do not revisit).

GANN TIME LAYER (never captured before; core confluence alongside Wyckoff/Bojan/dominance):
- Philosophy: "Gann timed. Wyckoff mapped." Time says WHEN turns are likely; never trade Gann alone;
  react only when Gann window + Wyckoff confirmation align.
- Counts: 90/180/360/540/720/1080/1440 DAYS from major highs/lows or the HALVING; 144 periods
  (days/hours/weeks) is the recurring hidden number; turns land within ±1-3 bars of the count.
- Halving vibrations: +180d minor pivot, +360d expansion mark, +720d 2-yr crest / DISTRIBUTION
  window, +1440d next-halving reset.
- STRONGEST RULES (his evidence tiers): "Don't be in a swing long in final Gann time window" (Very
  High); "Full exit trigger all swing longs (Wyckoff + Gann time window confirms)" (High); green
  candles w/o confirmation inside a Gann window = trap; re-accumulation can EXTEND a cycle; danger
  zones are calendar windows (e.g. 0818→1027, Week 43 symmetry).
- Geometry (secondary): 1x1, 1x2 angles to start; Square of 144; 144° arcs. NOT published: exact
  square construction, exact count start-point selection (discretionary), full mechanical ruleset.
- USABLE MECHANICAL TAKEAWAYS: overlay harmonic counts on major pivots + halving; final/danger
  windows = stand-down/exit for swing longs WHEN STRUCTURE CONFIRMS; 144/90/180/360/720 counts as
  entry CONFLUENCE (stacks with our validated fib-time layer — a NEW time family, never tested).

IMPLICATIONS FOR ONE_STRATEGY: (1) the 168d passive hold + 60% runner are OUR placeholders — WI's
real topology is bank 40% → trail under each new LPS → derisk to 1/10 → RE-ENTER at next LPS/RTZ
within the live campaign; (2) the Gann danger-window exit is a TIME-based stand-down we never had —
potentially the honest answer to both the bear-rally leak AND the "regime stand-down dial" gap;
(3) cadence fix = campaign re-entries (validated regime, same scale), NOT smaller timeframes.
NEXT: pre-registered CAMPAIGN-v2 + GANN study from these verbatim specs (knowledge injection, not
fishing). Note exit-craft caution (add.56: our Moneytaur-trail test was harmful) — WI's LPS-event
trail + re-entry topology is a DIFFERENT system (single-position testing couldn't express it).

### 2026-08-12 addendum 58 — CAMPAIGN-v2 + GANN study: WI's own topology UNDERPERFORMS the simple placeholder on his own signal. Cadence goes DOWN not up; Gann tiers inert. Adopt NEITHER. (add.44–56 skepticism, 7th confirmation)

Pre-registered study of WI's batch-6 verbatim specs (add.57): CAMPAIGN-v2 management topology
+ the GANN time layer. Branch study/campaign-v2 (worktree-isolated, off study/one-strategy).
Knowledge-injection of HIS specs (legit); every interpretation flagged OURS and fixed BEFORE
measuring; NO grids. Files: idea_lab/campaign_backtester.py (multi-entry engine, referee parity
0.00% on the 10-spec bt_audit self-test — the extension did not disturb cost/fill discipline),
gann_time.py (harmonic-count engine; causality PASS, 3-point truncation no-repaint PASS, 0
mismatches), campaign_strategy.py (CampaignV2Door — door ENTRIES/STOPS unchanged, add.57
management), run_campaign_v2.py. Costs 2bps+3bps, 1% risk, $100k. 12 markets (10 crypto incl.
BTC ref + GOLD + NDX); SPX = DATA OUTAGE this run (Yahoo v8 429 + Stooq JS-gated, exactly the
add.54 situation) — non-crypto independent evidence = GOLD (uncorrelated) + NDX.

GANN ENGINE (fresh build; founding engine/temporal/gann.py = Fib 21/34/55/89 w/ 30d cap, and
gann_cycles.py = ACF 30/60/90 + Square-of-9 PRICE — NEITHER implements WI's count-from-anchor
scheme; halving there 2024-04-20 vs verbatim 2024-04-19). Anchors = weekly N=5 pivots + halvings
(2016-07-09/2020-05-11/2024-04-19); counts 90/180/360/540/720/1080/1440d + 144d + 144w; ±3d.

PART 1 — v1 (add.54: 1 entry, 60% runner, 168d) vs CAMPAIGN-v2 (bank40→structure-event trail→
derisk to 1/10→10% runner to campaign death; up to E=3 higher-break re-entries when FLAT):
  BASKET (fixed 1% risk, additive, 35.2yr span):
    v1: PF 2.84  PnL $62,013  MaxDD -4.11%  n=116  cadence 3.30 tr/yr
    v2: PF 2.96  PnL $57,120  MaxDD -3.62%  n= 88  cadence 2.50 tr/yr
    CADENCE MULTIPLIER v2/v1 = 0.76x  (i.e. FEWER trades, not more).
  PAIRED campaign comparison (85 campaigns, same campaigns two managements):
    v2>v1 in 39/85; mean ΔPnL/campaign = -$58; bootstrap 95% CI [-$266, +$159] (STRADDLES 0);
    total ΔPnL -$4,892. => WI's management is statistically INDISTINGUISHABLE from the placeholder.
  PRE-REGISTERED VERDICT: C1 PF PASS (2.96 ≥ 2.54); C2 PnL FAIL (57.1k < 62.0k); C3 cadence FAIL
    (0.76x < 1.5x); C4 MaxDD PASS (-3.62% better than -6.16%). => CAMPAIGN-v2 FAILS the rule.

THE CADENCE ANSWER (the whole point of the study): WI's topology does NOT raise our cadence — it
LOWERS it. TWO mechanisms, both decisive:
  (1) entries-per-campaign ≈ 1.00 everywhere. Across 12 assets / 85 campaigns only ~3 second-entries
      EVER fired (BTC 8e/7c, LTC 7e/6c, NDX 23e/22c; all others exactly 1.00). WI's "1-3 sized
      entries per model" simply does not occur on our detector: a NEW HIGHER confirmed break inside a
      live campaign while flat is as rare as the primary door itself. This is add.55's death-of-
      "go-faster" AGAIN — our detector emits ~1 at-size event per campaign, PERIOD.
  (2) the campaign HOLDS one position longer (runner rides to campaign death, no 168d cap), so the
      one-position scanner SKIPS the later door fires that v1 booked as separate trades. WI's
      "keep the range until proven otherwise" ABSORBS v1's independent re-fires into single long holds
      → 88 trades vs 116. His topology trades LESS often at size on our signal, not more.
  Net: the cadence gap WI attributed to "entries per model" is NOT closable on our detector set. The
  only honest cadence lever remains BREADTH (the basket), exactly as add.54/55 concluded.

PART 2 — GANN TIME LAYER:
  ENGINE HONESTY: the VERBATIM anchor set (every weekly N=5 pivot high AND low + halvings, 9 counts,
    ±3d) SATURATES the calendar — entry-window coverage 50–65% per asset. WI's "major" pivots are
    DISCRETIONARY and FEW; mechanizing them as every N=5 fractal pivot over-produces anchors. A
    ~55%-coverage "confluence tier" is nearly useless as a discriminator by construction. (Danger
    tier — high-pivots only, 360+/halving-720+ — is tighter at 14–24%.)
  G-ENTRY split (in- vs out-window mean R): in-window is LOWER, not higher, on BOTH topologies —
    v1 in +0.491 (n72) vs out +0.606 (n44), Δ -0.115 CI [-0.645,+0.377]; v2 in +0.601 (n53) vs out
    +0.721 (n35), Δ -0.120 CI [-0.822,+0.526]. Both CIs straddle 0. => G-ENTRY FAILS (no edge; if
    anything mildly negative). NO conviction-sizing candidate.
  G-EXIT: (a) on CAMPAIGN-v2 it is ARCHITECTURALLY INERT — its pre-registered structure-confirmation
    clause (close < campaign floor) IS the campaign's own D2 body-break death, so v2 and v2+gexit are
    bit-identical (PF 2.96 / PnL $57,120 / MaxDD -3.62%, n=88 both). (b) on the v1 door (Part-1
    winner) — paired per-trade re-sim (add.56 method): G-EXIT altered 1/116 trades, mean ΔR +0.000
    (CI [+0.000,+0.001]), sumR 62.0→62.0, maxR 5.63→5.63, runner-tail sumR(≥3) 11.1→11.1. The
    Gann-danger-window + (close<created_LPS) conjunction almost never co-occurs while positioned
    (v1 is already stopped by the time price closes below its entry LPS). => G-EXIT is a NO-OP on our
    signal; it does not help the two feared soft spots because in this data those soft spots aren't
    biting: the below-EMA200 "bear-rally leak" is net +12.7R (n=6) and campaign-death exits are net
    +36.7R (n=19) — the runner tends to bank campaign deaths in PROFIT, not give-back.

VERDICTS vs pre-registered rules: CAMPAIGN-v2 FAILS (cadence+PnL). G-ENTRY FAILS. G-EXIT records as
INERT (redundant on v2, no-op on v1) — the "PASS" on v1 is vacuous (zero effect). ADOPT NEITHER.

RECOMMENDATION for ONE_STRATEGY.md: KEEP v1 (single entry, banked-and-derisked struct exits) as the
headline door. Do NOT adopt WI's campaign topology on our detector — it trades less often at size and
does not beat the simple placeholder (per-campaign CI straddles 0). Do NOT adopt Gann tiers (entry
confluence has no edge and saturates; the danger-window exit is subsumed by our own structure-break
death). This is the SEVENTH confirmation of the add.44–56 skepticism (fusion, dip-buying, speed-up,
LTF-generator, LTF-executor, exit-toolkit, and now campaign-topology + Gann all fail to add edge on
the door's own signal). The door's edge lives at the DAILY/WEEKLY scale as a RARE (~1/campaign,
~4–5/basket-yr) at-size signal, and BREADTH is the only honest frequency lever.

WHAT THIS DOES NOT TEST (the one real caveat): PYRAMIDING-WHILE-POSITIONED. WI adds to a WINNING
position while still holding; our engine is one-position / re-enter-only-when-flat (faithful to the
task's primary "only when FLAT" reading), so concurrent adds are UNEXPRESSED — they need a multi-
position engine (not cheap, deferred). If any part of WI's cadence claim survives our detector it is
there, not in flat re-entries. Also flagged: "newly created higher structure" was pre-registered as
higher break_level; a looser reading (higher LPS low) might fire more 2nd entries — a follow-up, NOT
a retrofit (changing it now = fishing). And per standing rule, even a PASS would require FORWARD
paper proof before adoption; this study produced no pass to forward. Referee parity 0.00%; Gann
causality + no-repaint PASS. Nothing shipped; worktree removed, branch study/campaign-v2 pushed.

#### addendum 58b — GANN RECONCILIATION vs the existing engine/temporal stack (coordinator directive)

Inventory-and-reconcile of the dormant/partially-live Gann code before finalizing (the founding
temporal stack is larger than the brief implied: engine/temporal ≈ 2,707 ln; gann_cycles.py header
explicitly cites @Wyckoff_Insider + @ZeroIKA → founding-knowledge-grade). Read gann.py, gann_cycles.py,
temporal_confluence.py, temporal_fusion.py, tpi.py, cycles.py.

RECONCILIATION TABLE (existing vs WI batch-6 add.57):
  | dimension    | EXISTING engine/temporal (founding)                    | WI batch-6 (add.57 / this study)          |
  | counts       | Gann vibrations [9,21,36,45,72,90,144] (BAR counts)    | [90,180,360,540,720,1080,1440]d +144d+144w |
  | tolerance    | ±3 BARS                                                | ±3 DAYS                                    |
  | anchors      | RECENT swings / bars_since_* Wyckoff events / roll-20  | MAJOR weekly-N5 pivots + HALVING dates     |
  | horizon      | max_projection ≈ 30 DAYS (tpi max_cycle_bars=720h)     | up to 1440 DAYS (4-yr halving cycle)       |
  | grid         | 1H feature store                                       | DAILY                                      |
  | halving use  | ONLY log-premium norm (days/1460) + thermo 144 blk/day; date 2024-04-20 | TIME ANCHOR for turn windows; date 2024-04-19 |
  | live status  | temporal_confluence_score PARTIALLY LIVE (min-gate on retest_cluster, backtest_composite.py:361-362) | study-only, nothing wired |

FAMILY GAP (confirmed FINDING): the founding-era Gann was implemented as SHORT-CYCLE BAR STATISTICS
(≤30-day vibrations off recent swings, on 1H), and NEVER implemented WI's LONG-CALENDAR-COUNT-from-
MAJOR-ANCHOR doctrine (90–1440 DAYS from major pivots + the HALVING). The only shared numbers are 90
and 144 — but in different UNITS (bars vs days) and off different ANCHORS (recent swing vs major
pivot/halving), so even those are different features. Halving-as-a-Gann-time-ANCHOR is genuinely new
(existing code uses the halving only for a log-premium normalizer and a hashrate floor, not as a turn-
window origin, and with the wrong date 2024-04-20 vs verbatim 2024-04-19). So gann_time.py is correctly
a FRESH build; nothing verbatim-matching existed to reuse.

ORTHOGONALITY vs the VALIDATED fib-time layer (the layer the door already sizes on, ×1.25): measured
phi(entry_window, fib_time>0) across all bars = +0.01 (BTC), +0.02 (GOLD), −0.03 (NDX); lift ≈ 1.00.
=> the new daily Gann window is STATISTICALLY INDEPENDENT of fib-time (co-occurs at chance), i.e. NOT
redundant with fib-time and NOT redundant with the live short-bar gann_time_cluster (different
units/anchors entirely). BUT independence does not rescue it: G-ENTRY already showed in-window mean R
≤ out-window (add.58), so the new layer is an orthogonal-BUT-worthless time family — it adds an
independent signal that carries no predictive value on the door's fires. Independent ≠ useful.
CONCLUSION UNCHANGED: adopt neither the campaign topology nor any Gann tier; and do not extend the
live temporal_confluence_score with the long-count family on this evidence (no edge, and it would only
add an orthogonal-but-inert feature). Record the family gap as founding-era doctrine coverage, not as
a to-build.

### 2026-08-13 addendum 59 — FINAL-round corrected re-test of add.58's two self-flagged defects (MAJOR-anchor Gann + gentler LPS re-entry): BOTH re-tests FAIL. Both hypotheses CLOSED permanently. 8th confirmation.

add.58 REJECTED Campaign-v2 + the Gann layer but flagged two of its OWN implementation defects as
legitimate round-2 follow-ups. This is that round-2 (ONE pre-registered corrected definition each, NO
grids, FINAL — fail again ⇒ close forever). Branch study/major-anchors off study/campaign-v2, worktree-
isolated. Files: gann_time.py (`major_only` trailing-365-extreme anchor filter, add.59 D1), campaign_
strategy.py (`CampaignV2bDoor` gentler LPS-hold re-entry, add.59 D2), run_major_anchors.py. Costs
2bps+3bps, 1% risk, $100k, 12 markets (10 fresh crypto incl. BTC ref + GOLD + NDX re-fetched via
yfinance; SPX still gated per add.54/58). Referee parity 0.00% (10/10 textbook), Gann causality +
3-point no-repaint PASS on the MAJOR anchors.

PART A — MAJOR-anchor Gann (D1 fix = "anchors saturated: it used EVERY weekly N=5 pivot → 50-65%
coverage"). Pre-registered anchor: a weekly N=5 CONFIRMED pivot that at FORMATION is ALSO the trailing-
365-calendar-day extreme (highest high / lowest low), computed causally from data ≤ formation ≤ the
N=5 confirm date (no repaint) + crypto halvings. Counts/tol UNCHANGED from add.58.
  ANCHOR CENSUS (the filter works, and picks the RIGHT pivots): major filter cut anchors ~55-63%
    (BTC 67→28, GOLD 160→48, NDX 153→59). BTC's 28 = the real cycle turns: 2017-12-17 (~$20k top),
    2021-04-18 + 2021-11-14 (double top), 2020-03-15 (COVID low), 2018-12-16 + 2022-11-27 (cycle
    bottoms) + 3 halvings — exactly WI's "handful of cycle-scale anchors."
  SATURATION GUARD (pre-registered: meaningful only if entry-window coverage < 25%; else STOP + close):
    coverage HALVED (~57%→~25%) but did NOT clear the guard. Basket mean 25.3%, median 25.4%, range
    [19.7, 31.5]; only 6/12 assets < 25%; the FLAGSHIPS all SATURATE — BTC 31.5%, GOLD 26.6%, NDX 31.2%
    (the two longest-history, most-liquid, non-crypto INDEPENDENT anchors). ⇒ GUARD NOT CLEARED → STOP
    Part A per pre-registration. STRUCTURAL REASON (decisive): 9 counts × horizons up to 1440 DAYS ×
    ±3d means even ~28 anchors blanket ≥25% of the calendar — the long-count doctrine intrinsically
    saturates. "Major" CANNOT be mechanized causally without discretion. Hypothesis CLOSED.
  DIAGNOSTIC (non-dispositive, run anyway to prove no hidden edge was discarded): even setting the
    guard aside, G-ENTRY on the v1 door with MAJOR windows = in-window +0.472 (n31) vs out +0.514
    (n75), Δ −0.042 CI[−0.530,+0.436] (straddles 0, mildly NEGATIVE — SAME sign as add.58's every-pivot
    result). G-EXIT (major danger tier) altered 0/106 v1 trades — INERT, identical to add.58. No latent
    edge exists to rescue with a non-saturating redesign.

PART B — gentler LPS re-entry (D2 fix = "second-entry too strict: required a new HIGHER break_level +
fresh door fire → entries/campaign ≈ 1.00"). Pre-registered trigger (CampaignV2bDoor): inside a LIVE
campaign while FLAT, a NEW confirmed N=10 fractal swing low forms HIGHER than the previous campaign
floor, price retests the nearer of {new LPS low, prior break_level} (low ≤ level+0.5·ATR) and CLOSES
back above it → entry 2 (cap E=3, dedup 3 bars). All post-entry management identical to add.58 v2.
  THE MECHANISM FIRED THIS TIME (the D2 fix is mechanically real): 8 gentler re-entries across 82
    campaigns vs only ~3 EVER in add.58's strict definition. entries/campaign rose on several assets
    (BTC 1.29, ADA 1.33, ETH 1.20, LINK 1.20). So the defect WAS the strictness — corrected, it fires
    ~2.7×.
  BUT IT STILL FAILS, for add.58's mechanism-2 reason: basket entries/campaign only ≈ 1.10 and cadence
    multiplier v2b/v1 = 0.85× (STILL FEWER trades than v1). Holding one campaign position longer
    ABSORBS v1's independent later door re-fires faster than the 8 gentle re-entries ADD them. Basket
    (fixed 1% risk, 25.2yr span): v1 PF 2.61 / $53,182 / DD −4.20% / n106 vs v2b PF 2.44 / $50,777 /
    DD −4.47% / n90. Paired campaigns 38/82 v2b>v1, mean ΔPnL −$25/campaign, bootstrap CI [−$247,+$211]
    (STRADDLES 0 — statistically indistinguishable from the placeholder, exactly like add.58).
  PRE-REGISTERED VERDICT: C1 PF PASS (2.44 ≥ 2.31); C2 PnL FAIL (50.8k < 53.2k); C3 cadence FAIL
    (0.85× < 1.5×); C4 MaxDD PASS. ⇒ CAMPAIGN-v2b FAILS. Hypothesis CLOSED.

FINAL DISPOSITION: BOTH hypotheses CLOSED PERMANENTLY (round 2 was the last per the multiplicity
contract). The Gann long-count layer adds no edge and cannot be mechanized without discretion; WI's
extra "1-3 entries per model" cadence is NOT recoverable on our detector — even the gentler LPS-hold
re-entry that DOES fire trades LESS often at size, not more. This is the 8th consecutive confirmation
of the add.44-58 skepticism (fusion, dip-buying, speed-up, LTF-generator, LTF-executor, exit-toolkit,
campaign-topology+Gann, and now major-anchor-Gann + gentler-re-entry) — the add-on base rate is now
0-for-9. UNCHANGED doctrine: the door's edge is a RARE (~1/campaign, ~4-5/basket-yr) at-size signal at
the DAILY/WEEKLY scale; BREADTH (the basket) is the only honest cadence lever; forward paper-collection
is the only remaining validation. Nothing shipped; production untouched; worktree removed; branch
study/major-anchors pushed.

### 2026-08-13 addendum 62 — WIDE-BASKET validation: the door is a TRENDING-ASSET edge (crypto+equity+single-stock+metal PASS; FX+energy REFUTE) and a 25-market forward basket reaches a verdict in ~2yr not ~7yr. Forward test PRE-REGISTERED.

Purpose (from add.60): at ~4.5 basket-trades/yr the M3 forward finish line is ~7 YEARS out. Widen the
universe to reach 10-15+ trades/yr → verdict in ~2yr. Validate the IDENTICAL add.48 door on the widest
honest universe and lock the forward basket. Branch study/wide-basket off study/major-anchors, worktree-
isolated. Files: idea_lab/run_wide_basket.py, fetch_wide.py, wide_basket_PREREGISTRATION.txt (universe +
verdict rules frozen BEFORE any fetch), docs/knowledge/FORWARD_TEST_SPEC.md. STUDY ONLY; production
untouched. NO per-asset tuning; ran ALL, reported ALL — a failing family is a FINDING, not a drop.

UNIVERSE: 29 markets = 13 reproduced (10 crypto cached + GOLD/NDX cached) + 16 NEW + SPX-retry, fetched
via yfinance (worked add.59). REFEREE PARITY: 17/17 new assets self-test 0.00% (textbook referee).
REPRODUCTION check (un-tuned, cached/refetched): SPX ^GSPC now UN-GATED reproduces add.48 EXACTLY
(PF 3.53 +$17,994 n25); GOLD 3.51 +$21,275; BTC 2.42; NDX 1.85 (= add.60's yfinance refetch, not
add.48's older 2.82 fetch); crypto basket PFs identical to add.54. The door is stable across vendor/date.

VOLUME-INDEPENDENCE (now load-bearing): FX majors have 100% zero-volume in yfinance; the door FIRED
NORMALLY on all four (n=12-20) with 0.00% referee parity → PROOF the headline path never gates on volume
(code: the only volume reader `build_structural_poc` is orthogonal and uncalled by `build_daily_sensors`).
DAX/N225/FTSE (25-34% zero-vol index noise) also clean. Confirms the add.48 volume-independence claim.

PART 1 — PER-ASSET (headline struct/flat, identical params). PF | PnL(1%R) | n | above-EMA200:
  crypto:        BTC 2.42/$4.5k/10/90%  ETH 1.93/$4.8k/12  SOL inf/$3.8k/3  LTC 3.26/$7.8k/9
                 XRP 0.08/-$2.7k/4 (Coinbase-suspension DATA GAP artifact, add.54) ADA 0.28/n3 DOGE 0.00/n2
                 DOT inf/n1 AVAX inf/n1 LINK 9.89/$8.9k/5   [n<5 fails are all directional]
  equity-index:  SPX 3.53/$18.0k/25  NDX 1.85/$6.6k/24  DJI 1.97/$8.0k/23  RUT 1.83/$8.3k/26
                 N225 1.59/$7.0k/31  DAX 1.25/$2.8k/30  FTSE 2.87/$6.0k/16   ALL POSITIVE, 87-100% above
  single-stock:  AAPL 3.17/$17.5k/27  MSFT 2.14/$19.2k/34  NVDA 3.92/$18.8k/24   ALL STRONG (new microstructure)
  metal:         GOLD 3.51/$21.3k/32  SILVER 1.70/$9.3k/31  COPPER 2.61/$18.3k/30  PLATINUM 1.48/$5.6k/29  ALL POS
  energy:        OIL 0.59/-$3.6k/13   NEGATIVE (small n, directional)
  fx:            USDJPY 1.50/$3.3k/20 (carry-trending)  EURUSD 0.65/-$2.4k  GBPUSD 0.25/-$6.3k  AUDUSD 0.26/-$9.0k

PART 2 — FAMILY VERDICTS (pooled, fixed 1% risk):
  crypto        n50  PF 2.50  +$27.2k  meanR 0.543  88% above   SUPPORTS (portability, add.54 reproduced)
  equity-index  n175 PF 1.97  +$55.1k  meanR 0.315  94% above   SUPPORTS (7 indices, 1990-2026)
  single-stock  n85  PF 2.88  +$52.3k  meanR 0.616  95% above   SUPPORTS (different microstructure — key new test)
  metal         n122 PF 2.25  +$51.5k  meanR 0.422  89% above   SUPPORTS (incl. uncorrelated Gold)
  energy        n13  PF 0.59  -$3.6k   meanR -0.274 100% above  REFUTES (weak/negative, n=13 directional)
  fx            n63  PF 0.57  -$14.8k  meanR -0.235 90% above   REFUTES (3/4 majors negative; mean-reverting)
  DOMAIN VERDICT: the door's "structural breakout-retest is universal" thesis holds for TRENDING asset
  classes (crypto, equities, single-stocks, metals) and FAILS on MEAN-REVERTING FX majors + thin OIL.
  The single trending FX pair (USDJPY carry) is positive — consistent with "door needs a trend," not a
  refutation of the mechanism. This BOUNDS the edge's domain honestly: it is a trend edge, not an all-
  market edge. (fx/energy fire despite high above-EMA200 share → EMA200 alone does not capture the
  choppy-but-drifting-up FX structure; the door's break-retest still whipsaws there.)

WIDE BASKET (all 29 pooled, fixed 1% risk): 508 raw trades → 372 episodes (±5d), 36.4yr span.
  cadence 14.0 raw/yr (RECENT-ERA 2021-26 all-live: 24.2/yr) → REACHES the 10-15 target.
  basket PF 1.89  +$167.8k  MaxDD -6.69%  meanR +0.330 (sd 1.321, boot 95% CI [+0.220,+0.447]).
  K-block K6m2: frac PF>1 100%, frac>=1.5 80%. (Diluted by the fx/energy losers.)
PASS-ONLY FORWARD BASKET (drop fx+energy = 25 markets, the forward universe):
  n432  PF 2.29  +$186.1k  MaxDD -6.02%  meanR +0.431  boot 95% CI [+0.305,+0.561] (CI-lo well >0).
  recent-era cadence (2021-26 all live): FULL 24.2/yr, PASS 20.0/yr → n>=30 verdict in ~1.5-2.5yr.

DEFLATION (add.60 M3, honest): forward expectation ~ HALF in-sample. PASS-basket meanR +0.431 → deflated
  ~+0.22R; PF 2.29 → deflated forward PF ~1.6-1.7. Clears the CONFIRM 1.5 floor but only MARGINALLY —
  the forward test is a real test. Family selection (dropping fx/energy on the SAME history) adds forking;
  the forward data re-adjudicates the 4-family domain claim, which is therefore a forward hypothesis, not
  a settled result. NOT a 2023-24 artifact (positive across 1990-2026 on equities/metals/single-stocks;
  Gold + AAPL/MSFT + SPX carry the long-history independent evidence).

PART 3 — PRE-REGISTERED FORWARD TEST (docs/knowledge/FORWARD_TEST_SPEC.md, LOCKED):
  Unit = 25-market PASS basket at fixed 1% risk. CONFIRM iff forward PF>=1.5 AND meanR boot CI-lo>0 at
  n>=30; REFUTE iff PF<1.0 with CI-hi<0; else keep collecting. Expected 12-20 trades/yr → finish line
  ~2yr (vs add.60's ~7yr — the study's goal, achieved). Minimal runner = SEPARATE daily-close process
  (one run/day after UTC close), NOT the 1H archetype engine, off fusion/dedup/CMI. Per-fire flags logged
  (never gated): CMI_regime_label, bear_window_flag, above_ema200, family, episode_id — enabling a later
  PAIRED-channel bear-overlay study on REAL forward fires (the add.48/49/50 open thread; the door's one
  missing organ is a causal bear-regime gate to plug the bear-rally-breakout leak). fx+energy kept as a
  LOGGED shadow book so the domain boundary stays under live test. SPEC ONLY — nothing deployed; needs
  explicit user go per standing rules.

BOTTOM LINE: the trend-continuation door is a GENUINE cross-family trend edge (4/6 families, incl. the
uncorrelated Gold and independent single-stocks and 7 equity indices back to 1990) — NOT a BTC/SPX fit and
NOT a 2023-24 artifact — but its domain is TRENDING markets; it does NOT generalize to mean-reverting FX
or thin energy, and that boundary is now measured, not assumed. Widening to 25 markets converts the
un-adjudicable ~7yr forward test into a ~2yr one. Deflated forward PF ~1.6-1.7 sits just above the
CONFIRM floor, so the forward collection is decisive, not ceremonial. Nothing shipped; production
untouched; worktree removed; branch study/wide-basket pushed.

### 2026-08-14 addendum 64 — BREADTH-AT-SCALE: the door's EDGE holds on 476 S&P 500 names (not mega-cap-specific, not a 2018-24 fit) but the owner's DAILY-cadence requirement CANNOT be met honestly — long holds + hard cross-name clustering + a 15% DD ceiling are in irreducible tension. VERDICT: 4/5 PASS, cadence FAILS.

Purpose (owner's core requirement = "many trades that are good"): the FROZEN add.48 door fires ~0.7-0.9/yr
per instrument and cannot be sped up (add.55/63). The FREE parameter is instrument COUNT. 25 markets -> ~20
tr/yr; test whether ~300 names -> ~200+/yr = "a good trade nearly every day," AND characterize the actual
PORTFOLIO. Branch study/breadth-300 off study/wide-basket, worktree-isolated. Files: idea_lab/
fetch_breadth300.py, run_breadth300.py, breadth300_PREREGISTRATION.txt. STUDY ONLY; production untouched.

UNIVERSE (pre-registered, mechanical): ALL 503 current S&P 500 constituents (Wikipedia 2026-08-14) with
>=8yr adjusted daily history + integrity pass. 501 fetched (2 mangled tickers dropped), 476 pass (25
excluded = all <8yr recent IPOs: ABNB/COIN/CRWD/GEHC/etc.; ZERO integrity failures). Financials KEPT
(run-ALL doctrine). auto_adjust=True (split/div adjusted — multiplicative => door's scale-invariant signals
UNCHANGED & causally safe; a documented improvement over add.62's raw fetch: AAPL/MSFT/NVDA reproduce
3.12/2.52/3.80 vs add.62 3.17/2.14/3.92, MSFT lifts on div-adjust). REFEREE PARITY 10/10 self-test 0.00%.

SURVIVORSHIP (binding, stated plainly): current-constituent lists inflate history — today's members are the
WINNERS that survived. Every in-sample PF here is upper-bounded. Mitigations run: (a) primary read on recent
2018-2026 AND full window; (b) mega-cap-trio vs broad; (c) the FORWARD test is the only survivorship-immune
arbiter. add.60 half-life deflation: recent PF 1.75 -> forward ~1.4 (may sit BELOW the 1.5 CONFIRM floor).

PART 1 — PER-NAME (476 names, 8,601 trades, 36yr, headline struct/flat identical params):
  median n 18/name; median PF 2.10 (mean 2.95); 92.4% of names PF>1, 73.9% PF>1.5; 92.6% of names
  net-profitable; median PnL/name +$5,976. Worst decile (47 names) sum only -$70k (mean -$1.5k/name = shallow
  losers). PF histogram unimodal-right: mode 2-3 (132 names), only 35/476 names PF<1. The edge is BROAD, not
  carried by a handful.

PART 2 — POOLED (fixed 1% risk = $1000/trade):
  FULL 1990-2026 : n8601  PF 2.21  meanR +0.401  boot95%CI [+0.371,+0.431]  (positive every decade)
  RECENT 2018-26 : n2622  PF 1.75  meanR +0.279  boot95%CI [+0.226,+0.334]  (survivorship-primary)
  Every SECTOR PF>1.5 (Materials weakest 1.68, Utilities strongest 2.71) — no sector carries it.
  SURVIVORSHIP CHECK (b): mega-cap trio AAPL/MSFT/NVDA PF 3.06 vs broad-rest PF 2.20. Breadth DILUTES the
  mega-cap success but the broad universe stays STRONGLY positive => the door's stock edge is NOT mega-cap-
  specific. This is the key affirmative finding: the edge SCALES.

PART 3 — THE PORTFOLIO REALITY (the decisive, un-flattering part):
  CLUSTERING: 8,601 raw entries -> only 504 independent episodes (+-5d); recent 2,622 raw -> 103 episodes =
    12.0/yr. Equities pile into the SAME index breakouts. The "304 raw/yr" is ~25x inflated by correlation;
    the honest independent-signal rate is ~12/yr.
  HOLD TIME: mean 161 / median 185 CALENDAR DAYS (~6 months — 8-mo max_hold + measured-move runners).
  CONCURRENCY (uncapped, 2018-26 daily grid): peak 237, median 140 simultaneous = 237%/140% of equity at
    risk at 1% each. The uncapped book is a FICTION (recent uncapped MaxDD -89.9%). The cap is an OPERATIONAL
    NECESSITY, not a filter.
  CAPPED PORTFOLIO (pre-registered P1 max 20 total + P2 max 5/sector, FCFS): accepts 1,763/8,601 (skips 6,728
    to the total cap). PF 2.27, PnL $833k, meanR +0.473, MaxDD -12.58%, median concurrency 20 (book ~ALWAYS
    FULL). Cadence 53/yr recent = ~1/week. Monthly (104 mo): mean 4.4 entries/mo, median 3, 41% of months
    >=5, only 4% zero; last-12-mo 50; last-month 8.
  CAP-LEVEL SENSITIVITY (the money table — cadence vs DD are in direct tension because holds are long &
    correlated): cap10 28/yr/-9.5%/PF2.23 | cap20 53/yr/-12.6%/2.27 | cap30 77/yr/-16.8%/2.13 |
    cap40 95/yr/-22.5%/2.04 | cap60+ plateaus 111/yr/-22.5% (raw supply + 5/sector exhausted). To reach the
    owner's >=100/yr you must run ~50-60 slots = 50-60% gross exposure AND eat MaxDD ~-22% (BREACHES the 15%
    ceiling). You CANNOT have both >=100 risk-capped entries/yr AND <=15% DD from this door.
  BEAR BEHAVIOR: breadth AMPLIFIES the add.48 bear-rally leak — some name is always breaking out, so the
    self-regime-filter weakens in aggregate: COVID 60 fires PF 0.54 (-$20k), 2022 171 fires PF 0.83 (-$16k),
    2018Q4 43 fires PF 0.99 (~flat). Capped MaxDD through 2021-22 -15.45%. The missing causal bear-regime
    gate (add.48/49/50 open thread) matters MORE at universe scale, not less.

PART 4 — VERDICT vs PRE-REGISTERED RULE (all 5 required):
  V1 recent PF>=1.5 & CI-lo>0 & n>=300 : 1.75 / +0.226 / 2622  PASS
  V2 median per-name PF>1              : 2.10                  PASS
  V3 >=55% names PF>1                  : 92.4%                 PASS
  V4 capped MaxDD<=15%                 : -12.58%               PASS
  V5 cadence>=100/yr (capped)          : 53.0/yr               FAIL
  => BREADTH-AT-SCALE does NOT fully validate (fails V5). The EDGE scales cleanly (V1-V4 strong, survivorship-
  caveated, NOT mega-cap-specific); the owner's DAILY-frequency requirement does NOT, and the failure is
  STRUCTURAL, not fixable by more names: long ~6-mo holds + ~12 independent episodes/yr + a sane DD ceiling
  bound the risk-capped intake to ~50-65/yr. More names past ~476 add correlated, not independent, signal.

BOTTOM LINE for the owner (vivid + honest): the door is a GENUINE, broad trend edge — 476 survivor names,
median PF 2.10, 92% profitable, every sector, back to 1990, and NOT just AAPL/MSFT/NVDA. But "a good trade
nearly every day" is not what this delivers: the lived experience is a book that is ALWAYS ~20 positions full,
taking ~1 NEW trade/week (~4-5/month), each held ~6 months, clustered into ~12 real market episodes/yr. To
manufacture 100+ entries/yr you'd trade correlated duplicates at 50%+ gross exposure and a 22% drawdown — the
frequency and the safety cannot both be had. The honest recommendation is BREADTH AS DIVERSIFICATION (many
names to smooth the equity curve and keep the book working), NOT breadth as a cadence multiplier. FORWARD
PROPOSAL (amend, not lock): keep the pre-registered 25-market wide basket (add.62) as the forward unit; ADD a
capped S&P 500 stock sleeve (cap 20 total / 5 per sector, 1% risk) as a SECOND logged forward channel with the
survivorship caveat prominent and the same per-fire flags (CMI_regime_label, bear_window_flag, above_ema200,
sector, episode_id) — its value is DD-smoothing + faster accumulation of independent bear-leak fires for the
overlay study, expressly NOT a claim of daily cadence. Nothing shipped; production untouched; needs explicit
user go per standing rules; worktree removed; branch study/breadth-300 pushed.

### 2026-08-15 addendum 65 — WI BATCH-7 CONFLUENCE TEST: WI's core claim ("the naked retest underperforms the confluent retest") is REFUTED on the powered universe. All 6 confluence specs CLOSE (0/6); the NAKED retest is the BEST tier. Add-on base rate now 0-for-18.

The first POWERED test of Wyckoff-Insider's confluence-stack critique of THE DOOR. WI reviewed our exact
mechanized rulebook and said the door "fires on structure alone — I almost never take the pure retest
without additional confluence." Until now untestable (n~106). Now decidable: PRIMARY = 476 S&P-500 names
recent 2018-26 (n=2,622 fires), SECONDARY = full-history stocks (n=8,601) + the 25-market wide basket
(trending-pass n=432), via the add.60/61-verified PAIRED size-multiplier ΔR machinery (MDE 0.02-0.05R).
Branch study/wi-batch7 off study/breadth-300, worktree-isolated. Files: idea_lab/run_wi_batch7.py,
check_n5_truncation.py. THE DOOR IS FROZEN — all six specs are OVERLAYS computed POST-HOC from the SAME
causal sensor arrays the door used (OHLCV, atr_14, eye_state/dir, range_upper_1d) + the door's own logged
break_level; the break BAR is recomputed deterministically from the eye arrays and verified to reproduce
the door's break_level EXACTLY (26/26 AAPL) → zero door mutation. STUDY ONLY; production untouched.
REFEREE PARITY 10/10 self-test 0.00%. N5 (a NEW rolling sensor) 3-point truncation check: 594 checks,
0 mismatches → causal / no-repaint.

CLOSED-LIST ACKNOWLEDGMENT (WI's batch-7 suggestions overlapping already-adjudicated axes — NOT re-tested):
  Gann/time windows (add.58/59/61 FINAL, closed) · dominance/USDT.D structure (add.14/46 REJECT; stables
  flag already in the door's R0) · multi-TF/LTF entries (add.52/53/55/63 FINAL) · shorts/springs (add.44-47
  closed; dip-buyer is DETECTED-HARM) · ob_quality / eq_magnet (add.61 H2/H3 FINAL CLOSE). WI's overlaps
  with these were acknowledged as decided, not reopened.

THE SIX NEW SPECS (pre-registered exact defs; WI verbatim where given, OURS where interpolated). Overlay =
1.25x sizing ΔR=0.25·R·flag (entry-quality N1/N2a/N2b/N4) or removal/inverse-size veto (N5/N6):
  N1 FRESHNESS  bars break→retest <= 10 (WI "8-12" midpoint, OURS)
  N2a SWEEP     entry/i-1 low undercuts break_level by 0.1-0.4 ATR then closes back above (WI verbatim)
  N2b ANATOMY   entry lower-wick >= 0.55 range AND close in top 40% (WI "55-60", 0.55; never-tested exact rule)
  N3 ACCEPTANCE higher low within 2 bars after retest (FORWARD info → delayed-entry diagnostic, NOT a sizing overlay)
  N4 ORIGIN     break_level in LOWER 60% of trailing-365d span at break (WI "discount/EQ not premium"; 0.60 OURS)
  N5 CHOP-VETO  >=3 failed breakouts (confirmed bull break whose close drops back below level within 15 bars,
                resolution<=entry) in trailing 90d = "chop regime" (WI's highest-interest, aimed at the bear leak)
  N6 CLIMAX-VETO atr14 at break > 1.8x its 90-bar median = blow-off skip (1.8 OURS)

PER-SPEC RESULTS (paired ΔR; PRIMARY n=2,622, pooled PF 1.75 meanR +0.279). cover | meanR_flag | meanR_unfl
| PF_flag | ΔR_CI-lo | C2(flag>=unfl) | secondary-consistency | VERDICT:
  N1 fresh    49.3%  0.215  0.342  1.66  +0.020  C2-FAIL  full+wide agree flag WORSE   CLOSE (false pass)
  N2a sweep   27.8%  0.279  0.279  1.80  +0.013  C2-FAIL  full flag-worse, wide flag-better (inconsist) CLOSE
  N2b anatomy 10.6%  0.280  0.279  1.71  +0.003  C2 +0.001R  wide INVERTS (0.284<0.451)  CLOSE (false pass)
  N4 origin   20.7%  0.241  0.289  1.69  +0.008  C2-FAIL  full+wide agree flag WORSE   CLOSE (false pass)
  N5 chop     2.4%   0.473  (kept 0.274)  removalΔ -0.005  cover DEGENERATE(<10%)  sign flips prim/wide  CLOSE
  N6 climax   2.4%   0.628  (kept 0.271)  removalΔ -0.009  cover DEGENERATE(<10%)      CLOSE

THE DECISIVE READ — the mechanical pre-registered rule (P1 CI-lo>0 + sign-holds-secondary + coverage) flags
ONLY N2b as "ADOPT*", but that is a TEXTBOOK add.61-H2 FALSE PASS: C2 margin is +0.001R (economically nil),
coverage sits ON the 10% degenerate boundary, and the wide-secondary INVERTS the selection (flagged 0.284 <
unflagged 0.451). Per the add.61 BINDING standard (on a 55-63%-WR net-positive book, C1/P1 is a weak bar
almost any positive-catching flag clears; the deciding criterion is C2 + family-consistency + monotonicity),
N2b CLOSES. Net: 0-for-6.

THE COMBINED CONFLUENCE SCORE = the cleanest refutation of WI. score = #{N1,N2a,N2b,N4} passing; meanR by level:
  PRIMARY:  s0 0.364(PF1.86 n735) · s1 0.259 · s2 0.221 · s3 0.267 · s4 0.264(n11)   NOT monotonic
  FULL:     s0 0.412 · s1 0.422 · s2 0.360 · s3 0.362 · s4 0.549(n30)                 NOT monotonic
  WIDE:     s0 0.536(PF2.55) · s1 0.452 · s2 0.294 · s3 0.343 · s4 —                   NOT monotonic
  In every set the NAKED retest (score 0 = zero confluence) has the HIGHEST or tied-highest meanR, and meanR
  is FLAT-to-DECREASING as confluence stacks. The sized-portfolio (1/1.25/1.5) "beats flat" on CI-lo>0 —
  but that is again the net-positive-book artifact (up-sizing any positive trades raises PnL); the pre-
  registered combined-adopt rule REQUIRES monotonicity, which FAILS on all three sets. Combined score: CLOSE.

FRESHNESS DISTRIBUTION DIAGNOSTIC (doubles as the answer to "is the 360-bar validity window admitting weak
stale retests?"): bars-to-retest p50=11, p90=239, max 360 (window fully used). meanR by staleness (PRIMARY):
  0-10: 0.215 (PF1.66) · 11-30: 0.314 · 31-90: 0.299 · 91-180: 0.356 · 181-360: 0.369 (PF1.81).
  The FRESHEST retests are the WORST, not the best — staler retests hold up fine (even 181-360 bars is
  net-positive PF 1.81). So (a) WI's freshness intuition is inverted here, and (b) the 360-bar window is NOT
  admitting weak stale retests — the tail is healthy; no case to tighten it. (Full-history mirrors: 0-10
  bucket 0.395 ~ overall 0.401.)

N5 BEAR/CHOP-WINDOW EFFECT (the spec WI cared most about — aimed at the add.48/64 bear-rally leak). On the
full-history stock pool the chop-veto MISSES its target: COVID 60 fires (chop-flag on only 1) chop meanR
-1.003; but 2018Q4 (chop +0.247 vs calm -0.011) and 2022 (chop +0.307 vs calm -0.100) have the chop-flagged
trades OUTPERFORMING; 2026-YTD chop -0.336 vs calm +0.259. Coverage in every bear is 1-3% (n_flag 1-6). The
chop-count veto does NOT cleanly identify the bear-rally losers and does NOT plug the leak; on the wide
basket its removal-delta is +0.029 (CI-lo>0) but that reverses the stock sign and rests on ~22 flagged
trades. The door's missing organ remains a causal bear-REGIME gate (add.48/49/50), NOT a chop-count veto.

N3 ACCEPTANCE — the one honest standout, but NOT adoptable here. cover 75%, meanR_flag +0.342 vs unflag
+0.090 (PRIMARY; full +0.479 vs +0.162; wide +0.503 vs +0.149) — a real +0.25-0.35R separation. BUT it is
FORWARD price action (a higher low in the next 2 bars), i.e. nearly tautological (trades that don't
immediately make new lows win more), and it is a DELAYED-ENTRY variant, not a causal sizing overlay. The
entry-shift re-sim (enter at bar i+2 with a recomputed stop, re-scan exits) is NOT cheap on the frozen
single-position backtester and was DEFERRED. N3 is the single pre-registered watch-item for a future round
IF a proper delayed-entry re-sim is built; it is not adopted now (look-ahead as a sizing flag).

VERDICT vs pre-registered pass rules: N1 CLOSE, N2a CLOSE, N2b CLOSE (false pass), N3 watch-item (deferred),
N4 CLOSE, N5 CLOSE, N6 CLOSE. 0-for-6. This is the 10th consecutive add-on round to reject; add-on base rate
0-for-18 (was 0-for-12 at add.61). The corrected POWERFUL paired gauge (MDE 0.02-0.05R) — not the blind
in/out split — delivered these nulls at n=2,622-8,601, so they are DETECTED nulls, not underpowered ones.

WI's CORE CLAIM ON TRIAL — "the naked retest underperforms the confluent retest": the data says the OPPOSITE.
On 8,601 fires the naked retest is as good as or BETTER than every confluence tier, and meanR does not rise
with stacked confluence. WI's edge is discretionary chart-reading that does not survive mechanization on this
door's fires — exactly the add.44-61 pattern (his geometry underperforms the simple placeholder on his own
signal). The honest answer to give back to WI: on THIS mechanized door, additional confluence does not select
better trades; the structural break-retest is already the whole edge.

WHAT AMENDS ONE_STRATEGY / FORWARD_TEST_SPEC: NOTHING. No spec adopts. The frozen door and the pre-registered
25-market forward basket (add.62) + capped S&P-500 sleeve (add.64) stand unchanged. The two open threads are
UNCHANGED: (1) a causal bear-REGIME gate for the bear-rally leak (add.48/49/50) — N5 does not substitute for
it; (2) forward paper-collection to the finish line. N3 acceptance added as a deferred future-round watch-item
(needs a delayed-entry re-sim, not a sizing overlay). Nothing shipped; production untouched; worktree removed;
branch study/wi-batch7 pushed.
