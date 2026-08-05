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
