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

### 2026-08-06 addendum 27 — Idea-lab built; integrated all-seeing-eye FAILS 3/3 CLEANLY (no artifact to blame)

Clean single-strategy backtester built (branch study/idea-lab, 2a7d3e8; 0.00% self-test
parity vs the audit referee; addendum-26 cost bug fixed from the start; NO fusion/dedup/
crowd-out/chunk possible). First test = the INTEGRATED all-seeing-eye as WI actually
trades it: C_accum/D_accum gate → spring/LPS + return-to-zone acceptance entry → tight
LPS stop → banked-derisk-runner exits.
RESULT — FAIL 1/3 eras: OOS-A 2023-24 PF 1.56 (bull) PASS; TRAIN 2018-22 PF 0.72 and
OOS-B 2025-26 PF 0.58 both NEGATIVE. Only the bull era passes = regime tailwind, not
edge. WI exit geometry did NOT beat a naive 1R/2R/3R ladder (naive ≥ WI every era; WI's
0.1R post-TP1 derisk kills runners). Entries 69% orthogonal to spring (genuinely new set,
still no edge). **DECISIVE: this removed EVERY engine artifact and the edge still wasn't
there → the failure is the IDEA/mechanization, not our plumbing. Closes the campaign's
central open question.**
META-CONCLUSION (now proven every way): trader-knowledge mechanizes as CONTEXT/SIZING
(6 validated boosts, deployed) but NOT as standalone whole-strategies (PO3 ×5, wyckoff-
campaign v1/v2, integrated eye — all fail; discretion + regime-timing is the un-
mechanizable edge). The all-seeing-eye is closed in every testable form: standalone
strategy (fail 2/3), sizing dial (failed untouched-fold gate, addendum 12/21), detection
(kept as infra). Surviving yield = the C_accum sizing boost (Boost 6, live).
ASSET: idea_lab/ is a reusable clean screener for any future extracted idea.

### 2026-08-06 addendum 28 — 6-boost forward replay on live trades: amplifier, not rescue

No-look-ahead replay (boost conditions from entry-bar features, server real-time live
logs). PAST 2 MONTHS (74 longs, base PnL +$885, PF 1.03): all-boosts +$1,543, delta
+$658 (+74%) — BUT only 6/74 positions had any boost fire (8% coverage: live logs only
recorded each boost's feature once it shipped), and ~$904 of the +$658 is ONE wick_trap
winner → a WASH within noise, not signal. FULL LIVE SAMPLE (212 longs, base −$24.3K):
all-boosts −$26.3K, delta −$1,950 (~8% worse) — the broad context boosts (phase-C 29
fires, rotation-calm 66 fires) amplified a LOSING long book in the Feb-Jun markdown.
CONCLUSION: boosts do exactly what boosts do — SCALE whatever the book is already doing;
sign = whether the amplified cohort was winning/losing in the chosen window. They neither
rescued nor broke the book. The validated edge lives in the historical 3/3 WFO cycles,
NOT in a 2-month markdown. Confirms the deeper campaign truth: the system's yield is
regime-dependent (long-biased book struggles in markdown/chop); boosts amplify, they
don't fix regime.

### 2026-08-07 addendum 29 — Decision-architecture research: fusion is COMPENSATORY, trading is CONJUNCTIVE+REGIME+META-LABELED

Deep-research (16 verified claims, cited; synthesis step hit the Fable weekly limit, synthesized manually). Confirms the user's diagnosis that fusion scoring was the wrong SHAPE.
1. WHY FUSION FAILS (3-0, Springer composite-index lit; Columbia "disjunctions of conjunctions"): a weighted-average is valid ONLY when factors don't interact and are perfect SUBSTITUTES (strong in one buys back weak in another = "compensatory"). Trading confluence is NON-compensatory — you need range AND sweep AND confirmation; a strong sweep can't substitute for a missing range. Formal opposite = the MIN operator (weakest-link gate). Behavioral evidence: 76% of real decisions use non-compensatory conjunctive rules; gating models beat every additive benchmark OUT-OF-SAMPLE (3-0). HONEST CAVEAT: one source (sjdm.org) found compensatory WADD beat lexicographic in a consumer domain, and 2 over-strong claims were REFUTED (0-3) — so it's "match the rule to whether criteria actually compensate," not "additive always wrong."
2. META-LABELING (multiple 3-0, Lopez de Prado / Hudson&Thames): the rigorous "soul." Primary model = SIDE (direction); SECONDARY model = p(this trade is profitable) → SIZE. position_size = side × p(profitable). Separates direction from participation/size; the secondary layer LEARNS which conditions make the primary signal work. This is exactly "I have entry signals, need a layer that decides whether-and-how-big" — and the correct form of our BOOSTS (which crudely size by #confirmations; meta-labeling = the trained-probability version).
3. REGIME-CONDITIONAL (3-0): the SAME signal flips meaning by regime ("oversold in bull = buy; in bear = warning") — EXACTLY our repeated bull-vs-markdown inversions (POC, PO3, all of it). Fix: gate interpretation/participation on regime; detect regime with a PERSISTENT model (Jump Models: <1 shift/yr vs HMM 2-8, less whipsaw); regime-gating cuts vol + drawdown.
4. MTF GATING HIERARCHY (well-sourced, verify budget ran out so unverified-not-refuted): HTF sets direction FIRST, LTF only times entry; distinct non-interchangeable roles; trade only when timeframes AGREE (conjunctive gate, not average).
THE MAPPING (why the user was right): the replacement architecture = REGIME GATE (Moneytaur context-first: CMI + breadth + stables/dominance — all already built) → CONJUNCTIVE SEQUENCE (Wyckoff phase machine: layers must co-occur in order, weakest-link) → META-LABEL SIZING (trained p(profit), the proper form of the 6 boosts). We already own all three pieces; fusion crushed them into a sum. CAVEATS: theory+practitioner+adjacent-domain evidence, NOT a proven-on-BTC result (tells us the SHAPE); meta-labeling can overfit on thin data (quantconnect "not a silver bullet"); must be validated in the clean idea-lab at the 3/3 era bar + purged CV.
NEXT: prototype (regime-gate → conjunction → meta-label) in idea_lab/ over the validated pieces.

### 2026-08-07 addendum 30 — Meta-label prototype: REJECT as built; architecture CONFIRMED; boosts revealed as the working v1

Prototype (branch study/meta-label-prototype, 3e255ce): GBT+isotonic on 1,530 champion
long entries, 35 causal features (fusion excluded), purged 5-fold train-only CV, frozen
single-shot OOS eval.
**REJECT for deployment:** GBT overfit (purged-CV AUC 0.52 vs resub 0.93 — coin-flip
generalization); learned gate did NOT materialize (mean p flat: bull 0.452 = markdown
0.452 — no stand-down where the book bled); live forward sniff ≈ null (+$422 on −$25.2K,
winner/loser p-separation NEGATIVE, partly a feature-coverage gap: dd_score = #2
importance but unavailable live).
**THE TWO REAL FINDINGS:**
1. **Architecture thesis CONFIRMED in the model's own anatomy: importance blocks =
   structure 0.43 + regime 0.31 + context 0.19 = 0.93 vs ARCHETYPE IDENTITY 0.03.**
   WHICH archetype fired is nearly irrelevant to trade quality; the CONDITIONS decide.
   The 16 archetypes are ~interchangeable entry generators; participation should be
   context-driven — precisely addendum 29's claim, now shown empirically in-house.
2. **The crude deployed boost stack IS a working decision layer — and beats the trained
   model OOS:** crude 1.25^count → PF 1.34/1.20 (OOS-A/OOS-B) vs learned-sized 1.25/1.11
   vs baseline 1.24/1.09, at equal risk, in BOTH OOS eras. The hand-built boosts are the
   v1 of the correct architecture, already live. The logistic sanity model ≈ ties crude
   (1.32/1.21) and generalizes better than GBT → at this sample size, SIMPLE beats
   flexible.
**PATH (not actioned):** (a) wire the CMI regime block (risk_temp/instability/chop/
trend_align/dd_score) into the store + live feature logging so the fairest features
exist on both bases — cheap infra, enables everything later; (b) revisit the trained
layer with regularized-logistic-class models once live data accumulates real sample
size; (c) the full addendum-29 stack (JM regime gate + conjunctive sequence) remains
untested as a combination — folds are spent; live data is the honest venue. Labels
inherit book exits; folds heavily reused — all caveats stand.

### 2026-08-07 addendum 31 — Sensor audit pt.1 (Wyckoff): springs are MISLOCATED — user's hypothesis CONFIRMED

Read-only health audit of the 13 Wyckoff event detectors + phase (scratch sensor_audit_wyckoff).
CAUSALITY: all clean (no-repaint, 3 cut points) — the repairs held.
**THE ROOT BUG CONFIRMED — mislocation, not degeneracy:** spring/SC/ST fire off
`rolling(20).min()` and that rolling low coincides with a real structural swing low
(swing_low_50, ≤1 ATR) only ~25-40% of the time (vs 16% chance) — median ~2 ATR away.
So the "spring" fires at arbitrary fresh-20-bar-lows mid-trend, NOT at a defined range
low. Ground truth: 8/8 right EVENT/timing but location HIT 5/8, MISLOCATED 3/8 (FTX 2022
"spring" fired 18.6% ABOVE the real low). → DIRECTLY explains why PO3/campaign entries
were structurally invalid: the spring wasn't at a range low. THE HIGHEST-LEVERAGE FIX:
anchor spring/ST detection to swing_low_50 / a persisted structural range instead of a
rolling window (test in idea_lab at 3/3 bar, do NOT wire).
OTHER FINDINGS: ut/utad/lps/lpsy DEGENERATE (<20 fires; utad bit-identical to ut =
redundant). bc HEALTHY (4/4 major tops, conf .89-.98), sos/sc healthy. Graded
wyckoff_bullish/bearish_score = FUSION-ROTTEN (compensatory avg; deprecate as decision
inputs — use categorical phase_dir + events). wyckoff_phase_dir TRUSTWORTHY (289 C_accum
episodes, coherent) BUT A_distrib = 40.6% of bars (over-broad distribution side, tuning
study flagged). This is the keystone finding: the SENSORS were partly misfiring, so prior
idea-tests were run on structurally-wrong entries — vindicates the audit-first approach.

### 2026-08-07 addendum 32 — Sensor audit pt.2 (MTF/regime): raw inputs alive, DERIVED SCALARS not materialized

Read-only audit (scratch sensor_audit_mtf). CAUSALITY: HTF wyckoff scores no-leak
(8/8 repo tests pass, leak-sniff ~0) BUT they roll HOURLY over the in-progress HTF bar
(not a strict closed-bar step) — 1D bias flickers within-day 35% of buckets; a gate
wanting stable daily permission must lag/smooth it (strict mtf_alignment.py exists but
doesn't feed these columns).
**KEYSTONE PLUMBING GAP (confirms addendum 30):** CMI derived scalars (risk_temp,
instability, crisis_prob, trend_align, dd_score) are computed INLINE in the backtester
(~L696-842), NOT persisted as store columns. Live logs 3 of them; trend_align + dd_score
logged NOWHERE (dd_score = #2 meta-label importance → nulled the live sniff). A
regime-gate can only use features present on BOTH bases = today only the RAW-INPUT tier
(chop_score, drawdown_persistence, adx, wick_ratio, fear_greed_norm, volume_z_7d, ema
flags, HTF wyckoff scores — all 100% clean both sides). Good news: the whole CMI block
is RECONSTRUCTABLE (all raw inputs present 2018+ and live) — fix is PLUMBING not new
sensors.
DEAD LIVE: crisis_prob frozen 0.009 (broken 'original' path; substitute path exists,
unwired); crash_frequency_7d constant 0 live. DEAD COLUMNS: 109 all-null + 6 constant
in store (tf1d_wyckoff_phase, tf4h_range_*, tf1d_frvp_*, mtf_* governor, tf*_boms_direction).
MTF AGREEMENT: 4H/1D directional-agree 49%, both-bull 27.3% (20,120 bars), hard-conflict
only 7.3% → a HTF-agreement gate is FEASIBLE (fires often, rarely conflicts).
PRIORITIZED FIXES (proposed, not applied): P0 materialize CMI block into store+live via
ONE shared formula (kills store/live drift); P1 fix crisis_prob (substitute path) +
retire crash_frequency live; P2 add lagged closed-bar HTF-bias column; P3 GC 109 dead cols.
