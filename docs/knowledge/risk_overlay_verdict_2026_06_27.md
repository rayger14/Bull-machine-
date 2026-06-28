# Risk/Exit Overlay Study — Adjudication Verdict (2026-06-27)

**Analyst:** quant-analyst (rail mode)
**Artifacts:** `results/champion_v14_risk/*/scorecard.json` + `*/trade_log.csv`
**Harness:** `scripts/champion/risk_overlay_study.py`
**Store:** `data/features_mtf/BTC_1H_FEATURES_V14_FULL_LIVE_PATH.parquet` (V14 honest core, live-path)
**Costs:** 2bps commission + 3bps slippage, $100K, thresholds enforced (bypass=False in scratch config)
**Windows:** per-year 2018-2024, wfo_train 2018-2022, pristine holdout 2025-03 -> 2026-05

> Standing Orders reproduced verbatim (do not override):
> - **NEVER turn off bypass_threshold** in production — data collection mode is required.
> - **NEVER disable any archetype** in production — all 16 stay enabled to collect live signal data.
> - **NEVER make production config/code changes** without explicit user approval. This memo recommends; it does not apply.

Production configs/code were **not** touched. Study scratch configs live in `configs/champion/risk_*.json` (untracked); the sizing/stop overlay is a runtime monkeypatch of `StandaloneBacktestEngine._open_position`, not a code edit.

---

## 1. Executive Summary

- **"Cut the book" is CONFIRMED and robust as portfolio composition — and it is NOT a Rule-1 fusion filter.** Removing archetypes by their realized OOS track record is legitimate sleeve selection, not a predictive per-trade gate. comp_core2 wins the holdout (+$6,826, PF 1.31 at position level) versus full16's -$14,366 (PF 0.91, 51% full-period MaxDD).
- **BUT the headline n is inflated 2.3x.** The scorecard counts scale-out *legs* as trades (105). The true holdout sample is **45 positions**. The edge-carrying leg (wick_trap) is **30 positions** — exactly at the n>=30 floor. This is **directional, not conclusive**.
- **The +$6,826 is a one-archetype result.** wick_trap = +$6,764 (PF 1.43); liquidity_sweep = **+$62 (PF 1.01) — a breakeven diversifier, not a contributor.** comp_core2 is effectively a single-archetype bet on wick_trap plus a flat second leg.
- **Wider stops are an OVERFIT TRAP — confirmed, and not a sample artifact.** It is a *paired* sweep on the same entry signals: in-sample full PF rises monotonically 1.41 -> 1.65 -> 1.86 (stop 1.0x -> 1.3x -> 1.5x) while holdout PF falls monotonically 1.31 -> 1.01 -> 0.91. IS-up / OOS-down on a controlled one-parameter sweep is the textbook overfit signature. **Reject wider stops. Reject bear-sizing (holdout -$1,953).**
- **Deploying comp_core2 collides head-on with Standing Order #2 (never disable archetypes). This is an ESCALATION, not an autonomous recommendation.** The reconcilable form keeps all 16 enabled for signal logging and applies a capital-allocation overlay — itself a production change requiring user approval.

---

## 2. Methodology

- Train: wfo_train 2018-2022. Holdout: 2025-03 -> 2026-05 (pristine, untouched by tuning). Plus per-year 2018-2024.
- **Granularity correction (material):** the trade_log logs each exit leg (scale-out at 0.5R/1.0R/2.0R, time exit, stop) as a row. Adjudication metrics below are recomputed at the **position level** (`groupby position_id`), which is the honest trade count. Scorecard `trades=` values are leg counts and overstate n ~2.3x for core variants.
- Real backtest throughout (`StandaloneBacktestEngine`). No phantom outcomes (Rule 6 clean).
- Sizing is risk-based (notional = risk$/stop_distance), so the stop-width overlay is risk-neutral — it changes stop-out *frequency*, not per-stop $. The user's framing is correct.

---

## 3. Findings

### 3A. Holdout at position level (corrected n)

| variant | legs | **positions** | holdout PnL | PF | WR | long share |
|---|---|---|---|---|---|---|
| comp_core2 | 105 | **45** | **+6,826** | **1.31** | 56% | 45/45 |
| risk_core2_stop130 (1.3x) | 93 | 43 | +256 | 1.01 | 51% | 43/43 |
| risk_core2_stop150 (1.5x) | 77 | 37 | -2,478 | 0.91 | 51% | 37/37 |
| risk_core2_bear50_stop130 | 93 | 43 | -1,953 | 0.93 | 51% | 43/43 |
| comp_core4 | 161 | 70 | +4,164 | 1.13 | 54% | 70/70 |
| risk_core4_stop130 | 143 | 65 | +1,565 | 1.04 | 54% | 65/65 |
| comp_full16 | 695 | 355 | -14,366 | 0.91 | 43% | 354/355 |

Every variant is ~100% long in the holdout. None addresses long-only-in-bear.

### 3B. comp_core2 holdout — per-archetype split (Task C)

| archetype | positions | PnL | PF | WR |
|---|---|---|---|---|
| **wick_trap** | 30 | **+6,764** | 1.43 | 63% |
| liquidity_sweep | 15 | +62 | 1.01 | 40% |
| **portfolio** | 45 | +6,826 | 1.31 | 56% |

The +$6,826 is carried entirely by wick_trap. liquidity_sweep is breakeven. **Not robust across two legs — it is one leg + a flat diversifier.**

### 3C. Stop-width paired sweep — IS/OOS reversal (Task B / Read #2)

| stop f | full(IS) PnL | full PF | full MaxDD | holdout PF | holdout PnL | holdout pos |
|---|---|---|---|---|---|---|
| 1.0x (comp_core2) | 63,577 | 1.41 | 17.0% | **1.31** | **+6,826** | 45 |
| 1.3x (stop130) | 101,800 | 1.65 | 10.9% | 1.01 | +256 | 43 |
| 1.5x (stop150) | 126,466 | 1.86 | 10.2% | 0.91 | -2,478 | 37 |

Monotone IS improvement, monotone OOS degradation. Same entries; only stop width varies. **Not a sample-size artifact** — it is a controlled paired comparison and the reversal is unambiguous. The IS MaxDD/PF "improvement" is fitting 2018-2024 noise.

### 3D. Rule-7 dedup-reshuffle test (isolated vs in-book)

Comparing the two surviving archetypes inside full16 vs isolated in core2:

| archetype | full16 (holdout) | core2 (holdout) |
|---|---|---|
| wick_trap | 20 pos, +8,072, PF 1.72 | 30 pos, +6,764, PF 1.43 |
| liquidity_sweep | 6 pos, -1,685, PF 0.46 | 15 pos, +62, PF 1.01 |

**Interpretation:** isolating wick_trap *adds 50% more trades but lowers per-trade quality* (PF 1.72 -> 1.43, PnL -16%) — that is dedup feeding it the bars formerly routed to the removed sleeves. This is the dedup fingerprint Rule 7 warns about. **However, it does NOT trip the Rule-7 false-win auto-reject**, because the portfolio win is *real avoided loss*, not reshuffled accounting: the removed sleeves carried genuine realized holdout losses (confluence_breakout alone -$15,978; trap_within_trend -$4,860; retest_cluster -$3,973). There is no "elsewhere" hiding the loss — the loss is simply not taken. **Caveat that survives:** the deployable wick_trap is the diluted PF~1.43 version, not the in-book PF 1.72. Expectations must use 1.43.

### 3E. Regime context of the holdout (caveat)

comp_core2 holdout entry regimes: risk_on 28, neutral 8, risk_off 7, crisis 2 (of 45). **62% risk-on-skewed.** The holdout is *not* a bear test. The positive holdout demonstrates risk-on/mixed survival, not bear survival. The bear *years* (2018 -$8,058; 2021 -$10,279; 2022 -$8,667) are all negative — single-fold, one-regime evidence -> **needs CPCV before any deploy.**

---

## 4. Verdict on each of the analyst's five reads

1. **"Cut the book is confirmed and robust."** PARTIALLY CONFIRMED. Confirmed it beats full16 on every risk-adjusted/OOS basis and is *legitimate composition, not a Rule-1 fusion filter and not a predictive gate*. Refuted "robust" in the strong sense: real OOS n=45 positions (not 105), edge carried by a single archetype (wick_trap), liquidity_sweep breakeven, wick_trap diluted to PF 1.43 by isolation. Robust *directionally*, not *statistically conclusive*.
2. **"Wider stops are an overfit trap; reject."** CONFIRMED. Verified at position level; monotone IS-up/OOS-down on a paired sweep; not a sample-size artifact. Reject.
3. **"Regime bear-sizing degrades the holdout; reject."** CONFIRMED (holdout -$1,953, PF 0.93). Reject.
4. **"core2 does not fix long-only-in-bear."** CONFIRMED. 45/45 long in holdout; all three bear years negative; holdout positive only because it is risk-on-skewed. Only a short side or a flat-in-bear kill-switch addresses bears; nothing in this grid does.
5. **"Risk>entries only partially supported; winner-capture untested."** AGREE. Stop-WIDTH overfits and bear-sizing fails, so the *tested* risk overlays are not the edge. The exit-reason ledger shows the real structure: portfolio PnL = scale-out/time-exit winners (+~$21.0K of legs) minus the runner stop_loss bleed (-$14.2K). The winner-capture / runner-stop asymmetry is exactly the untested lever — and it requires exit_logic code changes, so it is the correct follow-up.

---

## 5. Rule-by-rule adjudication of the recommended change (deploy comp_core2)

| Rule | Verdict |
|---|---|
| **#1 no fusion filters** | PASS. "Cut the book" selects sleeves by realized OOS PnL, not by fusion/domain score. Not a per-trade predictive gate. No violation. |
| **#2 never disable archetypes** | **VIOLATION / ESCALATE.** Deploying core2 means disabling 14 archetypes in production. Directly conflicts with the Standing Order. Cannot be applied autonomously. |
| **#3 never change bypass_threshold** | PASS for deploy (no prod bypass change). Note parity gap: study ran bypass=False; live runs bypass=True. Backtest realism is correct; live numbers will differ. |
| **#4 recommendations only** | PASS. Nothing applied. |
| **#5 WFO mandatory** | PASS. wfo_train 2018-22 + pristine 2025-26 holdout + per-year folds present. |
| **#6 no phantom outcomes** | PASS. Real backtest, real positions. |
| **#7 system-up + target-down false signal** | PASS (with caveat). System win = real avoided losses, not dedup accounting. wick_trap PnL dips -16% under isolation (dedup dilution) but trade count +50% and net portfolio gain is genuine. Use the diluted PF 1.43 for expectations. |
| **#8 filter->boost reframe** | N/A in spirit. This is composition, not a new gate. No boost equivalent required. |
| **#9 train AND OOS co-move** | comp_core2 wfo_train PF 1.48 / holdout PF 1.31 — same direction, gap < 30%. PASS. (Stop variants FAIL #9: train up, OOS down — reject, consistent with Read #2.) |
| **#10 gate-immune architecture** | N/A — no parameter tightening proposed. |

**Net:** the change is methodologically clean on Rules 1, 5, 6, 9 but is blocked by Standing Order #2. It is an **escalation**.

---

## 6. Recommendation

**Do NOT deploy comp_core2 as a production config change (it disables 14 archetypes — Standing Order #2).** Escalate to the user.

**The composition finding is real and worth acting on**, but the deployable form must reconcile with the data-collection mandate:
- **Keep all 16 archetypes ENABLED** for signal generation/logging (preserves the Standing-Order intent: maximum live signal data).
- **Apply a capital-allocation overlay** that routes live risk only to `wick_trap_v14rq` + `liquidity_sweep`, sizing the other 14 to zero notional while still logging their phantom signals.
- This separates *data collection* (16 sleeves logged) from *capital allocation* (2 sleeves funded). It is a new allocation-layer capability and is itself a production change requiring explicit user approval.

**Reject** all risk overlays tested: stop-width 1.3x/1.5x (overfit, Rule 9 fail) and bear-sizing 0.5x (holdout negative).

### Honest expected behavior of allocate-to-core2
- ~PF 1.3 in risk-on/mixed regimes; **continues to bleed every bear year** — cutting the book stops trading junk, it does not fix long-only-in-bear.
- Edge concentrated in one archetype (wick_trap, n=30 OOS = directional only); liquidity_sweep is a flat diversifier that lowers variance without adding return.
- Deployable wick_trap expectation is PF ~1.43 (isolation-diluted), not the 1.72 it shows inside the full book.
- Single risk-on-skewed fold -> **must pass CPCV before funding real capital.**

### Single best follow-up study
**REAL-backtest exit_logic winner-capture / scale-out asymmetry, WFO.** The exit ledger localizes the entire problem: winners are captured by the 0.5R/1.0R/2.0R scale-out ladder (+~$21K of legs) while the runner stop_loss leg bleeds -$14.2K. Test, in a scratch branch with modified `engine/archetypes/exit_logic.py`:
- the scale-out R-ladder (e.g., earlier/heavier locking vs letting runners run), and
- post-scale runner-stop tightening (move stop to breakeven/+0.5R after first scale-out).

Constraints: real backtest only (Rule 6 forbids phantom exit sims); position-level metrics; train 2018-22 / pristine 2025-26 holdout; per-archetype PnL reported; **train AND OOS must co-move (Rule 9)**; do not edit production exit_logic — scratch only. This is the one candidate in the study that could yield a genuine *exit* edge rather than a stop-width overfit.

---

## 7. Sample Size & Honest Caveats

- comp_core2 real OOS n = **45 positions** (scorecard's 105 = scale-out legs). Edge-carrying leg = **30 wick_trap positions** -> directional, not statistically separable.
- The stop-width OOS degradation is a **paired** sweep -> the *direction* is trustworthy even though absolute n (~37-45) is small; the IS-up/OOS-down reversal is the overfit signal regardless of a formal significance test.
- Holdout is **62% risk-on** -> not a bear test. "Would core2 have worked in a 2022-style bear? **No** — the 2022 year fold is -$8,667, PF 0.11."
- All numbers are real backtest (no phantom). Live will differ: study used thresholds-enforced (bypass=False); live runs bypass=True.
- liquidity_sweep contributes essentially zero OOS PnL — calling core2 a "2-archetype" portfolio overstates its diversification.

## 8. What This Doesn't Test

- A short side or flat-in-bear kill-switch (the only levers that address bear bleed — untested here).
- Exit_logic R-ladder / runner-stop restructuring (Section 6 follow-up — requires code changes).
- CPCV / multiple folds — this is a single 2025-26 fold plus per-year diagnostics.
- Live parity under bypass=True signal flooding.
- Whether a capital-allocation overlay (16 logged / 2 funded) is implementable in the live runner.

## 9. Files

- Read-only: `results/champion_v14_risk/*/{scorecard.json,trade_log.csv,performance_stats.json}`, `scripts/champion/risk_overlay_study.py`, `scripts/champion/run_battery.py`.
- Written: this memo (`docs/knowledge/risk_overlay_verdict_2026_06_27.md`).
- Production configs/code: **untouched.** Scratch configs (`configs/champion/risk_*.json`) are untracked study artifacts; overlay is a runtime monkeypatch.
