# Live Trade Forensic — Last 30 Paper Positions ("Is it random?")

**Date**: 2026-07-02
**Owner question**: "Even the good archetypes seem random — win big once, lose big the next. Why did some work and some not?"
**Data**: `results/live_audit_2026_07_02/last30_enriched.csv` (30 positions), `trades.json` (272 legs).
**Window**: 2026-05-24 → 2026-07-01. **Net: -$10,362** (9 winners +$9,683 / 21 losers -$20,045).
**Method**: entry-feature separation (rank-AUC), per-position mode bucketing, regime split, fusion-vs-enforced-threshold reconstruction. No production code/config touched.

## STANDING ORDERS (reproduced verbatim, not overridden)
- **NEVER turn off bypass_threshold** — data collection mode is required for the foreseeable future.
- **NEVER disable any archetype** — all 16 stay enabled to collect maximum live signal data.
- **NEVER make production config changes** (bypass, disabled_archetypes, thresholds, archetype YAMLs) without explicit user approval.
- **NEVER edit production code/configs directly** — recommendations and diffs only.

---

## Executive summary
1. **The single dominant cause is direction, not signal quality.** All 30 positions are **LONG**; BTC fell **-22.8%** across the entry window (entry prices 77,003 → 59,447). 18 of 21 losers went essentially **straight to their stop** (median loser MFE 0.45R, MAE ~1.0R) — they never went green. This is a long-only book bleeding in a sustained downtrend, not a collection of random archetype outcomes.
2. **Win/loss is genuinely unpredictable from entry-time features** — every entry variable is a coin flip or worse (fusion AUC **0.476, inverted**; instability 0.53; the "size" signal is an artifact, see #4). The outcome was decided by **post-entry price action** (did BTC bounce or keep falling) that a long signal could not foresee. This is Lesson #54 confirmed live.
3. **Fusion is worse than useless here — it is inverted.** The **5 trades that CLEAR the enforced dynamic threshold** (fusion ≥ threshold_at_entry) were **ALL losers, -$5,886**, and include the biggest single-trade losses (spring -$1,096, failed_continuation -$1,685, exhaustion_reversal -$1,423). Enforcing the threshold would have removed **all 9 winners** and kept 5 losers. Do **not** add fusion gates.
4. **"Winners are small-size" is an accounting artifact — REFUTED.** Total notional is a near-constant **~$52,500** for essentially every position. The enriched `size_usd` column is the **first-leg tranche**: winners are multi-leg (scaled in as they ran, small first leg), losers are 1-leg stop-outs (full first leg). Sizing is NOT inverted and NOT the lever — it collapses into the duration/reached-1R (reverse-causal) axis.
5. **The only real levers are the ones already on file: direction/regime and exits — neither is fixable by tuning triggers or fusion.**

---

## Findings

### A. Entry-time separation (is it predictable at entry?) — NO
Rank-AUC for predicting WIN (0.5 = random). n=9 winners / 21 losers.

| Feature (known at entry) | AUC | Read |
|---|---|---|
| fusion | **0.476** | inverted, no signal (Lesson #54 live) |
| instability | 0.534 | ~random |
| crisis | 0.720 | **confound** — winners cluster mid/late-June basing; timing proxy, not causal, n=9 |
| size_usd (first-leg) | 0.034 | **artifact** of scale-in (reverse-causal), not an entry signal |
| *(post-entry, for contrast)* mfe_R | 0.968 | reverse-causal (a trade that runs = a winner) |
| *(post-entry)* hold_h | 0.884 | reverse-causal (winners run, losers stop fast) |

`P(win | reached_1R) = 0.75` vs `P(win | never reached 1R) = 0.00`. Reaching +1R is near-perfectly separating — but it is an *outcome*, not an entry feature. **There is no deployable entry discriminator in this sample.**

### B. Failure/success modes (where the dollars are)
| Mode | n | Net $ |
|---|---:|---:|
| 2 — Long, straight to stop (MAE≥0.85R, MFE<1.0R, never green) | 18 | **-18,195** |
| 3 — Reached +1R then reversed to a loss | 3 | -1,850 |
| 4 — Genuine winner that ran (MFE ≥ 1.5R) | 6 | +7,869 |
| 5 — Small winner (MFE < 1.5R) | 3 | +1,814 |

The book is ~one mode: **18 longs marched straight to the stop.** The 6 runners (up to 2.85R) prove the triggers *can* catch moves — when the market bounces. Only 3 trades fit the "gave-back-a-winner" pattern (the BE@1R target from the MFE study); it is a minor line item here.

### C. Regime split — the label does NOT cleanly explain it
| Regime | n | WR | Net $ | med MFE | med MAE |
|---|---:|---:|---:|---:|---:|
| bull | 1 | 0% | -581 | 1.32R | 0.90R |
| neutral | 10 | 20% | -4,230 | 0.74R | 1.06R |
| bear | 19 | **37%** | -5,552 | 0.68R | 1.02R |

Counter-intuitively **bear had the higher win rate** (the late-June basing/bounce longs won). The CMI regime label lags the tape; the real driver is the raw **-22.8% BTC drift**, which is directional, not captured by the bull/neutral/bear label. "Long into bear" is directionally true as a story but the *label* is not the discriminator — direction is.

### D. Bypass / junk attribution — enforcing the threshold makes it WORSE
Reconstructed each position's `fusion` vs its logged `threshold_at_entry` (dynamic threshold, base 0.18 + regime/instability terms).

| Bucket | n | Net $ | Winners | Losers |
|---|---:|---:|---:|---:|
| Would trade under enforcement (fusion ≥ thr) | 5 | **-5,886** | 0 | 5 |
| Bypass-only (fusion < thr) | 25 | -4,476 | **9** | 16 |

**Enforcing the fusion threshold does not save money — it deletes every winner and retains the 5 worst losers.** The owner's intuition that "junk in bypass caused the loss" is only half right: yes almost all trades are sub-threshold, but the *sub-threshold bucket is where all 9 winners live*, and the *above-threshold bucket is 100% losses*. The config note calling bypass "catastrophic" is a misread of this same inversion — the problem is a long-only book in a downtrend, not the bypass switch. **This reinforces the Standing Order to leave bypass ON**: the fusion gate has no positive selection value to restore.

### Per-archetype
| Archetype | n | W | WR | Net $ | med fusion |
|---|---:|---:|---:|---:|---:|
| confluence_breakout | 13 | 4 | 31% | -4,726 | 0.15 |
| failed_continuation | 1 | 0 | 0% | -1,685 | 0.39 |
| oi_divergence | 3 | 0 | 0% | -1,608 | 0.17 |
| spring | 1 | 0 | 0% | -1,096 | 0.53 |
| retest_cluster | 1 | 0 | 0% | -852 | 0.22 |
| exhaustion_reversal | 4 | 2 | 50% | -836 | 0.31 |
| liquidity_compression | 7 | 3 | 43% | **+442** | 0.34 |

Note the two validated "good" archetypes (**wick_trap, liquidity_sweep**) do **not appear** — they barely fire on live-path features (see `liquidity_score_root_cause_2026_06_10`, `holdout_verdict_2026_06_10`). The live book is dominated by confluence_breakout (a soft+bypass gate-immune archetype, Rule 10) and the proven losers. oi_divergence 0/3 is consistent with its known 0/5 leak. Single-position archetypes (spring, retest, failed_continuation) are n=1 — anecdote, not evidence.

---

## E. VERDICT — "Is it random?"
**From the entry's perspective: effectively yes — win/loss is not predictable from any logged entry feature (all AUC ≈ 0.5, fusion inverted).** But the *portfolio* result is not random noise — it is a **structurally negative bet**: a long-only book in a -22.8% BTC decline has ~30% WR by construction, and the outcome of each trade was set by whether BTC happened to bounce after entry (post-entry price action the long signal could not foresee). The 9 winners were the trades that caught local bounces (mostly the mid/late-June basing). **These trades did not have a per-trade entry edge the system could see in advance; the edge (if any) lives in direction/regime selection and in trade management, not in the trigger or the fusion score.** This is the same conclusion as the 2026-06-28 winner-loser forensic, now reproduced on live money.

## F. Actionable — the ACTUAL lever vs the things that don't work
**Do NOT work (evidence in this sample):**
- Fusion filter / raising the threshold — inverted here, deletes all winners (Rule 7/Lesson #54). Refused.
- Per-archetype fusion gates — same inversion; also confluence_breakout is soft+bypass gate-immune (Rule 10).
- Sizing changes — notional is already constant; the size signal is an artifact.
- Disabling archetypes / flipping bypass — Standing Orders, and the data shows enforcement wouldn't help.

**Candidate levers (hypotheses only — NOT recommended off n=30):**
1. **Direction / regime trade-SKIP** (highest leverage). The loss is one bet: long-only into a confirmed downtrend. The already-queued *regime kill-switch / trade-skip in confirmed downtrend* (`sizing_studies_verdict_2026_06_16`: sizing can't flip a bear year; size=0 is mis-accounted so must be a genuine skip) is the structurally correct response. Must be validated with WFO + regime stratification before any change — not deployable from this window.
2. **Exits** — minor here: only 3 trades (-$1,850) fit "reached +1R then reversed" (the BE@1R / wick_trap breakeven finding). 18/21 losers never reached +1R, so a breakeven stop cannot help them. Winner-capture is not the bottleneck in a downtrend; direction is.
3. **Fix the live-path so the validated edges (wick_trap, liquidity_sweep) actually fire** — they are structurally locked out on live features and are absent from all 30 (`liquidity_score_root_cause_2026_06_10`). Restoring them is the feature-store rebuild already in flight, not a tuning change.

**Bottom line for "win more":** you cannot make these archetypes win more by tuning triggers or fusion — the trigger is ~random at entry and fusion is inverted. The only honest levers are (a) don't take the whole book long into a confirmed downtrend (regime skip, needs WFO) and (b) let the live-path express the two archetypes that actually have an edge. Both are structural, both are already on the roadmap, neither is a fusion/sizing knob.

---

## Sample size & honest caveats
- **n=30 positions / 9 winners — directional only, NOT statistically separable.** Every AUC/split is suggestive, not conclusive.
- **One instrument, one 5-week window, one direction (all long), one regime episode (a BTC downtrend).** Nothing here generalizes to a bull regime; a long-only book would look great in the mirror-image window. Cannot claim "the archetypes are bad" — the *bet* was bad.
- **crisis AUC 0.72 is a timing confound, not a signal** — winners cluster in the basing phase; using it as a filter would be curve-fitting the one bounce (and would violate the no-fusion/anti-overfit rails anyway).
- These are **real live paper fills** (not phantom simulations); MFE/MAE computed from real 1H klines. No phantom outcomes were used for any exit/entry claim.
- "Would this book have worked in a 2022-style bear?" **No** — same long-only-into-decline structure. "In a bull?" Likely yes, but untested here.

## What this does NOT test
- No WFO/CPCV — this is a diagnostic of realized live trades, not a validated change. Any lever above (regime skip especially) requires a proper walk-forward with train/OOS co-movement and per-archetype OOS PnL before recommendation.
- Does not test short-side capability (the mirror fix for a long-only bear problem) — deferred per `sizing_studies_verdict_2026_06_16`.
- Does not isolate confluence_breakout's gate-immune architecture (Rule 10) — separate study.
- Does not re-derive the dynamic threshold from scratch; uses the logged `threshold_at_entry` per leg.

## Files
- Read: `results/live_audit_2026_07_02/{last30_enriched.csv, trades.json}`, `configs/champion_paper.json`, `docs/knowledge/{MEMORY.md, winner_loser_forensic_2026_06_28.md, mfe_analysis_2026_06_28.md}`.
- Written: this file only. **No production code or config modified.**

## Cross-references
[[winner_loser_forensic_2026_06_28]] (entry AUC≈0.5, duration reverse-causal — same conclusion on backtest) ·
[[mfe_analysis_2026_06_28]] (BE@1R only helps the 14% that go green then reverse) ·
[[sizing_studies_verdict_2026_06_16]] (sizing can't flip a bear year; regime trade-skip queued) ·
[[liquidity_score_root_cause_2026_06_10]] (wick_trap/liquidity_sweep structurally locked out live) ·
[[risk_overlay_verdict_2026_06_27]] (cut-the-book; bears bleed long-only)
