# Live Forensic — Last 50 Paper Positions (per-archetype winners vs losers)

**Date**: 2026-07-10
**Question**: for each archetype, what do its winners have in common vs its losers — what SPECIFICALLY separates them?
**Data**: `results/live_audit_2026_07_10/trades.json` (292 legs → 119 real positions + 68 legacy pre-position_id legs; last 50 positions by final exit time).
**Window**: entries 2026-05-15 → 2026-07-08, exits through 2026-07-10. **Net -$11,187** (17W +$18,518 / 33L -$29,705, 34% WR). All 50 LONG.
**Prior work extended (not re-derived)**: [[live_trade_forensic_2026_07_02]] (last-30: entry ≈ random, fusion inverted, loss = long-into-bear) and [[cb_forensic_2026_07_04]] (CB no-chase hypothesis — formally checked OOS here, Section C).

## STANDING ORDERS (reproduced verbatim, not overridden)
- **NEVER turn off bypass_threshold** — data collection mode is required for the foreseeable future.
- **NEVER disable any archetype** — all 16 stay enabled to collect maximum live signal data.
- **NEVER make production config changes** (bypass, disabled_archetypes, thresholds, archetype YAMLs) without explicit user approval.
- **NEVER edit production code/configs directly** — recommendations and diffs only.

---

## 1. Executive Summary

1. **Third consecutive window where nothing at entry separates winners from losers.** The sign of BTC's move after entry explains 47/50 outcomes: P(win | BTC rose entry→exit) = **15/15 = 100%**, P(win | BTC fell) = **2/35 = 6%**. Every "discriminator" with AUC > 0.6 is either post-entry (reverse-causal) or a calendar/phase proxy.
2. **Fusion inversion replicates a third time.** The 10 positions that cleared the enforced dynamic threshold: 1W/9L, **-$7,623** (fresh-since-07-02 subset: 1W/4L, -$1,737). The 40 bypass-only positions hold all other 16 winners. Reinforces Standing Order (bypass ON) and Rule 1 — no fusion gates, ever.
3. **The CB no-chase pattern (07-04) FAILS its pre-registered OOS check.** Extended (>4% above 7-day low) CB entries had the *highest* WR (43%) and mild avg damage (-$216); the near-low bucket stayed ~breakeven (-$42 avg, consistent with the weak half of the prior finding). The damage-concentration claim does not replicate; `pct_above_7d_low` is largely a market-phase proxy. Downgrade — do not propose a no-chase gate.
4. **The equity recovery is real but is market phase, not improved selection.** Closed-trade equity: trough **$73,343 on 06-30** → **$80,497 on 07-10** (+$7,153 over 9 positions; user's $71K→$79K dashboard figure is the same shape, mark-to-market). Drivers: liquidity_compression +$4,125 (3W/1L) and confluence_breakout +$3,538 (3W/0L). Post-BTC-low (06-25) entries: 58% WR vs 26% pre-low — exactly what a long-only book does when the tape turns.
5. **Actionable: nothing new is deployable from this window.** The only structurally supported lever remains the already-queued downtrend trade-SKIP ([[downtrend_study_2026_07_02]]): the 38 pre-low entries lost -$15.1K, the 12 post-low entries made +$3.9K. One weak hypothesis-grade candidate is pre-registered in Section E (basing-confluence sizing boost, n=19, one phase — explicitly fragile).

---

## 2. Methodology

- **Positions**: grouped 292 legs by `position_id` (net_pnl = Σ pnl_usd per position; entry state from first leg; 68 legacy Feb-era legs without position_id excluded from the last-50 by exit time — window starts 2026-05-15). Last 50 by final exit time.
- **Price path**: V14 live-path store OHLC (2018→2026-06-10) spliced with `data/cache/binance_vision/klines/BTCUSDT_1h.parquet` (2026-06-01→2026-07-09), seam-checked. MFE/MAE in R (R = |entry − stop|, long book) over entry→exit bars. **Cross-validated exactly** against `results/live_audit_2026_07_02/last30_enriched.csv` on all overlapping positions (bit-identical mfe/mae).
- **2 truncated paths flagged** (exits after 2026-07-09 23:00, klines end): `long_confluence_breakout_1783544400` (+$320) and `long_liquidity_compression_1783519200` (+$1,611). Their PnL is real (logged fills); their MFE/MAE understate.
- **Level context**: `engine/features/level_features.py::compute_level_features` on the spliced OHLC from 2026-02 (≫500-bar warmup): day_type, level_quality_low, dist_to_support_atr, acceptance, sweep events at entry bars.
- **Live feature log**: `results/coinbase_paper/live_features/*.jsonl` (starts 2026-06-18) joined at entry timestamps — covers 19/50 positions (11W/8L).
- **Separation metric**: rank-AUC (0.5 = random), winner-median vs loser-median, same harness as `scripts/champion/winner_loser_forensic.py`.
- **No WFO/CPCV** — this is a diagnostic of realized live fills, not a validated change. No phantom outcomes used for any claim.
- Working files staged in `/tmp/live50/` (outside repo). Repo untouched except this doc.

---

## 3. Findings

### 3.0 Book shape

| Archetype | n | W | WR | Net $ |
|---|---:|---:|---:|---:|
| confluence_breakout | 20 | 7 | 35% | -4,100 |
| liquidity_compression | 13 | 5 | 38% | +866 |
| oi_divergence | 5 | 1 | 20% | -1,705 |
| exhaustion_reversal | 4 | 2 | 50% | -836 |
| retest_cluster | 3 | 0 | 0% | -2,712 |
| trap_within_trend | 2 | 1 | 50% | -510 |
| failed_continuation / spring | 1+1 | 0 | 0% | -2,781 |
| liquidity_sweep | 1 | 1 | 100% | +591 |

Only **CB (n=20)** and **LC (n=13)** clear the n≥6 bar. Everything else is anecdote-tier. wick_trap still absent (live-path lockout, [[liquidity_score_root_cause_2026_06_10]]); the single liquidity_sweep trade won (+$591) — first live appearance in these forensics.

Loss-mode taxonomy (extends last-30): **straight-to-stop 26 positions, -$25,217** (the book, again); runner-wins 11, +$13,864; small-wins 6, +$4,654; **reached-1R-then-lost 7, -$4,488** (grew from 3/-$1,850 in last-30; 4 of 7 are fresh; still second-order vs straight-to-stop).

### 3.A Per-archetype winner-vs-loser tables

**ALL last-50 (17W/33L)** — winner-median vs loser-median (AUC):

| Discriminator | Win med | Loss med | AUC | vs earlier windows |
|---|---:|---:|---:|---|
| fusion | 0.234 | 0.235 | 0.50 | consistent (dead; enforced bucket inverted, see 3.0/Exec #2) |
| crisis_prob | 0.045 | 0.009 | 0.68 | consistent with last-30's 0.72 — **timing confound** (corr(crisis, entry_day) = +0.39; crisis rose into the June low, when longs started working) |
| instability | 0.372 | 0.373 | 0.50 | consistent (dead) |
| risk_temp | 0.251 | 0.438 | 0.29 | new axis, but **pure time proxy**: corr(risk_temp, entry_day) = **-0.75** (fell as market fell) |
| threshold_margin | -0.142 | -0.095 | 0.38 | consistent (inverted — winners further BELOW threshold) |
| %-above-7d-low | 1.61 | 1.47 | 0.53 | **no signal at book level**; per-archetype signs FLIP (see below) |
| prior-72h return | -2.69 | -1.45 | 0.43 | consistent with 07-04 "no separation at scale" |
| level_quality_low | 0.75 | 0.75 | 0.65 | medians identical; tail-driven AUC, weak, no prior support |
| day_type | mostly 0 both | — | — | no signal (winners 5/17 on breakdown days vs losers 6/33) |
| hour-of-day | 14 | 12 | 0.66 | noise-tier (see CB) |
| hold duration | 48h | 14h | 0.83 | consistent — reverse-causal (winners run) |
| MFE reached 1R | P(win\|reached)=0.70 (n=23) | P(win\|not)=0.04 (n=27) | 0.95 | consistent — reverse-causal outcome, exit-side lever only |
| BTC fwd return entry→exit | +1.62% | -1.60% | 0.89 | **the actual cause** (post-entry, unforeseeable at entry) |

`sweep_low_event` at entry: **0/17 winners, 0/33 losers** — the live book never enters on a level-sweep bar. No discrimination possible, but it confirms the validated sweep-style entries (wick_trap family) are entirely absent from live.

**confluence_breakout (n=20, 7W/13L, -$4,100)**

| Discriminator | Win med | Loss med | AUC | Read |
|---|---:|---:|---:|---|
| fusion | 0.184 | 0.153 | 0.45 | dead (07-04: "zero separation" — consistent) |
| %-above-7d-low | 1.73 | 1.87 | 0.50 | **no separation — see Section C (pre-registered check FAILS)** |
| prior-72h return | -2.69 | -1.29 | 0.34 | winners bought *bigger* declines — inverse of "chasing"; unstable, phase-driven |
| risk_temp | 0.246 | 0.475 | 0.35 | time proxy (as above) |
| crisis | 0.045 | 0.045 | 0.70 | medians identical, tail-driven; confound |
| hour-of-day | 13 | 5 | 0.74 | winners [1,12,12,13,16,21,22], losers cluster 01-07 UTC. **Noise-tier**: n=20, no mechanism, no prior support, multiple-comparison artifact until it survives a backtest check |
| hold duration | 56h | 10h | 0.84 | reverse-causal |
| BTC fwd return | +1.68% | -1.60% | **1.00** | CB is a pure market-direction coin flip: every winner had BTC rise after entry, every loser had it fall |

**liquidity_compression (n=13, 5W/8L, +$866)**

| Discriminator | Win med | Loss med | AUC | Read |
|---|---:|---:|---:|---|
| crisis | 0.113 | 0.009 | 0.97 | **calendar split, not a signal**: all 5 winners entered 06-11 onward (crisis regime elevated near the low); 6/8 losers entered pre-06-11. Deploying this = curve-fitting the June bottom |
| risk_temp | 0.251 | 0.439 | 0.15 | same time proxy |
| %-above-7d-low | 4.35 | 0.48 | 0.62 | **FLIPS the CB no-chase sign** — LC winners were MORE extended (post-low rally entries: 06-11 +7.6%, 07-02 +6.7%, 07-08 +4.4%). Confirms extension = phase proxy |
| threshold_margin | -0.142 | +0.008 | 0.25 | inverted again — LC losers were the ones nearer/over threshold |
| fusion | 0.390 | 0.420 | 0.40 | dead/inverted |
| instability | 0.417 | 0.332 | 0.68 | weak, no prior support, medians close |
| hold duration | 67h | 16h | 0.85 | reverse-causal; every LC winner shows 0.5R/1R(/2R) scale-outs, every loser is a bare stop_loss |
| MFE reached 1R | P(win\|reached)=0.71 (n=7) | P(win\|not)=0.00 (n=6) | 0.89 | reverse-causal |

**Honest per-archetype answer: nothing entry-observable separates CB or LC winners from their losers that is not a market-phase/calendar proxy.** The separation lives entirely in what BTC did next.

Small-n rows (anecdote): oi_divergence now 1W/4L live (-$1,705; cumulative live record ~2/13 — the known leak persists); retest_cluster 0/3 (-$2,712); exhaustion_reversal 2/4 (-$836).

### 3.B Market-regime overlay

BTC over the window: 81,519 (05-15) → **low 58,290 on 06-25** → 63,230 (07-09); **-22.4%** peak-to-end with a ~+8.5% basing/rally off the low.

| Phase (by entry) | n | WR | Net $ | med BTC fwd ret |
|---|---:|---:|---:|---:|
| Decline (entry < 06-25 low) | 38 | 26% | -15,109 | -1.25% |
| Basing/rally (entry ≥ 06-25) | 12 | **58%** | **+3,921** | +1.13% |

- **Yes, recent trades are winning more — because BTC stopped falling.** P(win | BTC rose after entry) = 100%, P(win | BTC fell) = 6%. The WR doubling post-low is the same coin with the tape's thumb on it, not better selection (entry features unchanged across phases).
- **Equity verification**: closed-trade equity (start $100K, all 187 positions) troughs at **$73,343 on 06-30** and ends **$80,497 on 07-10** — consistent with the user's $71K→$79K (dashboard marks open positions; shape identical).
- **Recovery attribution (+$7,153, 9 positions since the 06-30 trough)**: liquidity_compression **+$4,125** (3W/1L — the 06-30, 07-02, 07-08 compression dip-buys that scaled out through 1R-2R), confluence_breakout **+$3,538** (3W/0L incl. the +$2,780 06-30 entry at the exact retest of the low), trap_within_trend -$510 (1W/1L).

### 3.C Pre-registered OOS check: CB no-chase (07-04 hypothesis) — **FAILS to replicate**

Hypothesis on file (cb_forensic_2026_07_04): CB dollar damage concentrates in entries >4% above the 7-day low; near-low entries ~breakeven. Fresh window, CB n=20:

| Bucket (%-above-7d-low) | n | WR | Avg $ | Total $ |
|---|---:|---:|---:|---:|
| <2% (near-low) | 11 | 36% | -42 | -465 |
| 2-4% | 2 | 0% | -1,063 | -2,125 |
| >4% (extended, "chase") | 7 | **43%** | -216 | -1,510 |

- The **near-low ≈ breakeven** half replicates (-$42/trade — same as the +$8/trade at scale).
- The **damage-concentration half does NOT replicate**: the extended bucket had the highest WR and the smallest per-trade loss of the losing buckets. At n=7 this is not a refutation of the 771-position scale finding, but it is a failed confirmation on genuinely new data.
- Mechanism for the instability: extension is a **phase proxy**. All-archetype version inverts outright (>4%: n=13, 54% WR, **+$2,199**, driven by post-low LC/CB rally entries; <2%: n=29, 31% WR, -$7,477 — "near the 7d low" during a decline just means *still falling*). A feature whose sign depends on whether the market is declining or basing is not a deployable filter.
- **Formal disposition: downgrade the no-chase hypothesis from "config-candidate" to "phase-conditional observation."** Do not propose an extension gate. (This is also what Rule 8's filter accept-rate history predicts.)

### 3.D New commonalities hunt (honest)

- **Cross-archetype consistent-sign candidates** in this window: high crisis / low risk_temp at entry. Both are **disqualified as calendar proxies** (corr with entry day +0.39 / -0.75; the "signal" is "entered after mid-June"). Not deployable; would curve-fit one bottom.
- **Live-feature slice** (n=19 positions ≥ 06-18, 11W/8L — basing/rally phase only), winner-median vs loser-median: adx **27.9 vs 17.8**, chop_score **0.44 vs 0.65**, range_position_20 **0.20 vs 0.49**, rsi_14 38 vs 48, volume_zscore 1.6 vs 1.3, bb_width 0.035 vs 0.020, close_vs_sma_200 -1.7% vs -2.2%. Signs are consistent with the backtest-validated discriminator set (adx/volume_z up, chop down — Lesson-#54-era structural features), which is why it earns pre-registration and nothing more. Composite read: *winners bought low-in-range pullbacks while trend energy (ADX) was building* — i.e., the basing-phase dip-buy. **n=19, one phase, one instrument: directional only.**
- **reached-1R-then-lost** grew to 7/-$4,488 (from 3/-$1,850) — feeds the already-registered `trailing_start_r` sweep (0.5/0.75/1.0, [[stack_validation_verdict_2026_07_08]]); not a new hypothesis. Note the 07-04 LC trade: MFE 1.53R, scaled 0.5R+1R, net only -$119 — existing scale-outs already blunt this mode.
- **hour-of-day** (CB winners afternoon/evening UTC, losers 01-07 UTC): flagged as **noise-tier** — no mechanism, no prior window support, n=20, born of an 11-feature scan (multiple comparisons).
- **Real pattern vs n<10 noise, explicitly**: the only claims in this report that clear "real" are (i) fusion/threshold inversion (3rd window, n=10 enforced all-but-one losers), (ii) outcome = BTC direction after entry (n=50, 47/50), (iii) straight-to-stop as the dominant loss mode (26/33 losers). Everything else — crisis, risk_temp, extension, hour, level_quality, instability, the live-feature composite — is n<20-per-cell and/or phase-confounded.

---

## 4. Recommendation

**Keep current — no config or code change proposed from this window.** Specifically:

1. **REJECT (do not propose)**: any fusion/threshold gate (inverted 3rd time — Rule 1/Lesson #54); a CB no-chase extension gate (failed its own pre-registered OOS check, Section C); crisis/risk_temp entry filters (calendar proxies); hour-of-day filters (noise-tier).
2. **REAFFIRM the queued lever**: the downtrend trade-SKIP (price < 200-day, [[downtrend_study_2026_07_02]]) is the only mechanism that addresses the actual loss structure here (38 decline-phase longs, -$15.1K). It already has offline WFO support; it needs user approval, not more diagnosis.
3. **PRE-REGISTER (hypothesis-grade, per Rule 8 as a BOOST)**: *basing-confluence sizing boost* — amplify size X ∈ {1.0, 1.25, 1.5} when `range_position_20 < 0.35` AND `adx > 25` AND `chop_score < 0.5` at long entry. Source: n=19 live slice, one phase. Must pass full WFO on the V14 store with train/OOS same-sign (Rule 9) and per-archetype OOS PnL non-regression (Rule 7) before it is anything more than a note. Prior for success: modest (boosts 2/2 accepted historically, but this composite describes one market phase).
4. **Exit-side**: fold the grown reached-1R-then-lost line (-$4.5K) into the existing `trailing_start_r` sweep — no new study.

Nothing here is applied; user decides.

---

## 5. Sample Size & Honest Caveats

- **n=50 positions / 17 winners; CB 7W/13L, LC 5W/8L; every other archetype n<6.** All medians and AUCs are directional, not statistically separable. The LC crisis AUC of 0.97 looks spectacular and is a calendar artifact — treat it as the canonical example of why n=13 AUCs must not be deployed.
- **One instrument, one 8-week window, one direction (all long), one regime arc (decline → basing).** The post-low WR of 58% will not survive the next leg down; the pre-low WR of 26% would look equally unfair in a bull.
- The last-50 window **contains** the last-30 window (overlap 30 positions); only 20 positions are fresh evidence (8W/12L, -$2,969). Cross-window "consistency" statements are partially self-referential except where the fresh subset is quoted (fusion inversion: fresh enforced 1W/4L, -$1,737 — holds).
- 2 positions have truncated price paths (exits after 07-09 23:00 klines end) — flagged; PnL real, MFE/MAE understated.
- MFE/MAE from real 1H klines on real paper fills; no phantom outcomes anywhere. Splice seam (V14→binance_vision at 06-01) verified continuous.
- "Would this book have worked in a 2022-style bear?" **No** — same long-only-into-decline structure as the pre-low segment. The recovery segment is not evidence of edge; it is evidence of beta.

## 6. What This Doesn't Test

- No WFO/CPCV — nothing here validates a change; the Section E boost candidate requires a full walk-forward study before recommendation.
- Does not test the short side, the 200-day skip's live behavior, or the live-path lockout fix (wick_trap et al. still absent — their absence itself is re-confirmed, not diagnosed here).
- Does not decompose CB's gate-immune architecture (Rule 10) or dedup interactions — the last-50 book composition is downstream of both.
- Level features beyond the entry bar (e.g., acceptance sequences post-entry) untested; the live feature log covers only 19/50 entries.

## 7. Files

- **Read**: `results/live_audit_2026_07_10/trades.json`, `results/live_audit_2026_07_02/last30_enriched.csv`, `data/features_mtf/BTC_1H_FEATURES_V14_FULL_LIVE_PATH.parquet`, `data/cache/binance_vision/klines/BTCUSDT_1h.parquet`, `results/coinbase_paper/live_features/2026-0{6,7}.jsonl`, `engine/features/level_features.py`, prior knowledge docs.
- **Written**: this file only. Working CSVs staged in `/tmp/live50/` (outside repo). **No production code or config modified. No commits made.**

## Cross-references
[[live_trade_forensic_2026_07_02]] · [[cb_forensic_2026_07_04]] (no-chase — now downgraded) · [[winner_loser_forensic_2026_06_28]] · [[downtrend_study_2026_07_02]] (the standing lever) · [[stack_validation_verdict_2026_07_08]] (trailing_start_r sweep) · [[liquidity_score_root_cause_2026_06_10]] (wick_trap lockout, still visible here)
