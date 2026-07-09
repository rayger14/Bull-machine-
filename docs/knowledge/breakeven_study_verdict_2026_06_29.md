# Breakeven-Stop-After-1R Study — Adversarial Verdict (2026-06-29)

> **RETRACTED 2026-07-08 — implementation artifact.** The monkey-patch mutated `pos.stop_loss` directly, zeroing the R denominator (`_calculate_unrealized_r` stop_distance → 0) from the next bar, which silently DISABLED scale-outs and trailing for every post-BE trade. The measured "BE gains" were an accidental different exit policy (ride-to-target with entry-floor stop), not breakeven protection: be10_b0 max_win 2.5x baseline ($4,595→$11,427), WR 77.5%→61.3% — impossible for a pure stop-ratchet. The honest production implementation (`_apply_breakeven_stop`, adapter rebuilt per bar, R denominator intact) changes **1 trade / +$15** across ~337 wick_trap trades: trailing activation (+1.0R, 2 ATR ≈ entry+2.9 ATR) dominates BE@1R by construction. Section 1's "NOT an accounting artifact" claim conflated internal PF consistency with mechanism validity; the section 2/7 "different trade populations" caveat was the entire result. Applies equally to `breakeven_fullsystem.py` (same patch) — the "core2+BE deployable" result is retracted too. See `stack_validation_verdict_2026_07_08.md`.

**Adjudicator:** quant-analyst (rails-on)
**Artifacts:** `results/champion_v14_breakeven/<arch>/<variant>/{scorecard,performance_stats}.json`
**Harness:** `scripts/champion/breakeven_study.py` (V14 store, monkey-patched `_check_all_exits`)
**Grid:** baseline | be10_b0 (1.0R, buf0) | be10_b10 (1.0R, +0.1R) | be15_b0 (1.5R, buf0)
**Archetypes:** exhaustion_reversal, wick_trap (v14rq), liquidity_sweep, spring
**Windows:** per-year 2018-2024, wfo_train 2018-22, holdout 2025-01..2026-06

> **STANDING ORDERS (verbatim):** NEVER turn off bypass_threshold (data-collection mode mandatory). NEVER disable any archetype. NEVER make production config/code changes without explicit user approval. NEVER edit production code/configs directly — recommendations and diffs only. This study touches **exit behavior** → any deployment requires a per-archetype exit-config + explicit user approval.

---

## 1. Executive Summary

- **The PF gains are NOT an accounting artifact.** Reported PF reconciles to recomputed `gross_profit/gross_loss` with **err = 0.00000** on every full-window cell. `total_trades = len(self.trades)` counts every closed position; breakeven scratches fall into the `pnl <= 0` (losers) bucket and contribute ≈0 to the denominator because their PnL genuinely is ≈0. **Hypothesis B (scratches excluded → PF inflated) is REFUTED.** PnL (the un-gameable bedrock) genuinely rises in-backtest.
- **This is a REAL backtest, not a phantom-exit sim (Rule 6 cleared).** The patch runs the actual engine, fills at the moved stop, processes current-bar exits BEFORE raising the stop (next-bar application), so there is no intrabar lookahead. Rule 6 forbids *phantom* exit recommendations; this is a real backtested exit change, so it is admissible — subject to the rest of the rails.
- **The trade-count drop is real simulation dynamics, not exclusion — and it is a confound, not a feature.** be10_b0 is NOT a clean A/B against baseline (same trades, tighter stop). Earlier exits change *when the archetype is flat*, which changes *which later signals are taken*. Proof: a trivial +0.1R buffer (be10_b10) **inverts** the result to worse-than-baseline and **balloons** counts (spring 634→879). be10_b0 vs baseline compares two **different trade populations**.
- **Only wick_trap shows a mechanistically clean signal.** wick_trap full: gross profit ~flat (155.8k→153.9k, −1.3%), gross loss cut (106.7k→92.8k, −13.1%) — textbook "convert give-back losers to scratches," broad across years, holdout n=54. **liquidity_sweep and spring fail the skeptic checks** (entry-reshuffle gross-profit anomaly, 2-window concentration, sub-30 holdout, marginal/in-sample-losing base).
- **Recommendation: DO NOT deploy any variant to live now.** Advance **wick_trap be10_b0 only** to CPCV + full multi-archetype re-run as a candidate; HOLD liquidity_sweep and spring (directional only). Prior exit-logic changes failed live (V12 invalidation −$54K) — the bar is high and not yet cleared.

---

## 2. Methodology / Rule Audit (Task A)

| Rule | Verdict |
|------|---------|
| **Rule 6 — phantom exits forbidden** | CLEARED. Real backtest via `_check_all_exits` patch; `orig()` runs first (current-bar exits at old stop), stop raised only afterward → applies next bar → no intrabar lookahead. Not a phantom sim. |
| **Rule 9 — train AND OOS co-move** | PASS for wick_trap / liquidity_sweep / spring on PF and PnL. wick train 1.37→1.56 & holdout 1.43→1.67; liq train 1.75→2.70 & holdout 1.10→2.39; spring train 0.97→1.17 & holdout 1.00→1.18. exhaustion FAILS (train 1.18→1.03) → correctly excluded. |
| **Rule 7 — system-up/target-down false signal** | NOT triggered *here* (isolated single-archetype mode ⇒ target PnL = system PnL, and target improves). BUT the failure mode Rule 7 guards against (dedup routing across archetypes) is **completely untested** — this study disables the other 15 archetypes. In live, BE exit-timing interacts with cross-archetype dedup. Must be re-validated in full-system mode before deploy. |
| **Rule 8 — filter→boost** | Not strictly applicable (this is exit management, not an entry gate). But the *effect signature* (trade-count drop, WR collapse) resembles the over-tightening / "cut" failure mode that filters exhibit. Treat with filter-level skepticism, not boost-level optimism. |
| **Standing Orders / Rule 4** | No production change made. Study config sets `bypass_threshold=False` only in the **scratch** config (`cfg_for`), not production. Deployment would require new per-archetype exit params + explicit approval. |

**Anti-overfit guards:** Train/holdout PF gaps acceptable (wick holdout 1.67 > train 1.56; liq gap ~11%; spring consistent ~1.17/1.18). OOS count drops < 50% (wick −23%, liq −24%, spring −43%). **liquidity_sweep holdout n=29 < 30 → directional only, not statistically separable.**

---

## 3. Findings (Task B — accounting integrity)

**PF reconciliation (full window), reported vs recomputed incl. scratches — max err = 0.00000:**

| arch | variant | tt | W | L | grossP | grossL | PF |
|------|---------|----|----|----|--------|--------|-----|
| wick_trap | baseline | 329 | 255 | 74 | 155,819 | 106,746 | 1.46 |
| wick_trap | be10_b0 | 266 | 163 | 103 | 153,857 | 92,768 | 1.66 |
| liquidity_sweep | baseline | 172 | 135 | 37 | 65,964 | 49,625 | 1.33 |
| liquidity_sweep | be10_b0 | 118 | 67 | 51 | 82,431 | 42,106 | 1.96 |
| spring | baseline | 634 | 413 | 221 | 221,632 | 209,288 | 1.06 |
| spring | be10_b0 | 393 | 116 | 277 | 208,397 | 156,843 | 1.33 |

Key reads:
- **Scratches are counted, not excluded.** Under BE, the **losers** bucket *grows* (wick 74→103, spring 221→277) — that is where the breakeven exits land (a BE exit still pays commission+slippage ⇒ slightly negative). If the harness were hiding scratches, the loser count would shrink. It rises. PF rises *despite* more losers, because the surviving losers are tiny.
- **PnL (un-gameable) genuinely rises:** wick +$12.0K, liq +$24.0K, spring +$39.2K (full). The improvement is therefore real *within this backtest*, not an artifact.
- **Gross-profit decomposition exposes the real mechanism — and the fraud risk:**
  - **wick_trap (clean):** grossP −1.3%, grossL −13.1% → pure loss-cut. This is the BE hypothesis working as intended.
  - **liquidity_sweep (reshuffle):** grossP **+25.0%** (66k→82k), grossL −15.1%. Tightening stops cannot *increase* gross profit — the only explanation is a **changed trade population catching bigger winners**. Smoking gun: liq **2024** grossL ~flat (15,801→15,768) but grossP **doubled** (7,118→16,874). BE cut *no* losses in 2024; it simply caught different/bigger winners. That is reshuffle luck, not the BE mechanism.
  - **spring (mixed):** grossP −6.0%, grossL −25.1% (mostly loss-cut), but in the gain years grossP *rose* (2020 54.2k→60.9k, 2024 34.0k→41.6k) → partial reshuffle.
  - **exhaustion_reversal (no edge):** grossP −9.7%, grossL −9.0% — both shrink proportionally, PF flat (1.14→1.13). Correctly excluded.

---

## 4. Robustness / Concentration (Task C)

**Per-year be10_b0 − baseline ΔPnL:**

| year | wick_trap | liquidity_sweep | spring |
|------|-----------|-----------------|--------|
| 2018 | +1.0K | 0 (no trades) | +2.9K (still −12.5K abs) |
| 2019 | +10.9K | +3.3K | +1.6K |
| 2020 | **−4.2K** (gave up upside) | **+12.1K** | **+15.9K** |
| 2021 | +0.8K | +0.1K | +2.8K (still −1.0K abs) |
| 2022 (bear) | −0.3K | 0 | −0.1K (still −6.2K abs) |
| 2023 | −0.5K | −1.3K | +3.0K |
| 2024 | +6.2K | **+9.8K** | **+10.8K** |
| holdout | +1.1K (n=54) | +3.0K (n=29) | +2.5K (n=42) |

- **wick_trap — broadest support.** Positive in most years, modest holdout lift, bears unchanged. Costs some 2020/2023 extreme-bull upside (gives back the tail you would have ridden). The only candidate with distributed, mechanistically-clean gains.
- **liquidity_sweep — NOT robust.** 91% of the +$24K full gain is 2020 (+12.1K) + 2024 (+9.8K). 2023 negative. Holdout **n=29 < 30 (directional only)**. Worst gross-profit-reshuffle anomaly. Two windows carrying the result + sub-threshold OOS.
- **spring — too good to be true.** Baseline spring is **a losing archetype in-sample** (train PF 0.97). BE "rescues" it by cutting count 38% and collapsing WR 65%→**29.5%** (winners 413→116). Gains concentrated 2020+2024; still bleeds 2018 (−12.5K), 2021, 2022. This is the exact signature of an in-sample exit rescue of a marginal archetype — the family that produced the V12 invalidation −$54K failure. High suspicion.

---

## 5. Why buffer-0 > +0.1R and trigger 1.0 > 1.5 (Task D)

**Full-window PF / trade-count across the grid:**

| arch | baseline | be10_b0 | be10_b10 (+0.1R) | be15_b0 (1.5R) |
|------|----------|---------|------------------|----------------|
| wick_trap | 1.46 / 329 | **1.66 / 266** | 1.40 / 405 | 1.45 / 285 |
| liquidity_sweep | 1.33 / 172 | **1.96 / 118** | 1.21 / 216 | 1.62 / 142 |
| spring | 1.06 / 634 | **1.33 / 393** | 1.04 / 879 | 1.18 / 454 |

- **The buffer axis is non-monotone and chaotic.** A +0.1R buffer makes PF **worse than baseline** for ALL three archetypes and **balloons** trade count (spring 634→**879**, wick 329→405). A robust effect degrades gracefully along the parameter axis; here a 0.1R nudge flips best→worst-than-baseline. That is **grid noise / path-dependent re-entry churn**, not a stable optimum.
- **Mechanistic story:** buffer 0 = stop exactly at entry; when hit you exit at ~0 *with price back at entry*, where no fresh setup exists ⇒ no immediate re-entry. Buffer +0.1R = stop *above* entry, triggered on a smaller pullback *while price is still near the highs* where setups re-fire ⇒ a **re-entry cascade** that re-risks capital and adds fee drag. This explains the count explosion and PF degradation.
- **Trigger 1.0 > 1.5 is roughly monotone** (be15_b0 sits between baseline and be10_b0), which is more believable. But because the *buffer* axis is chaotic, the precise (1.0, 0.0) "optimum" is the corner of a noisy grid, **not a validated sweet spot**. A robust deployment needs a stable plateau across (trigger, buffer), which this grid does not demonstrate.

---

## 6. Recommendation (Task E)

**KEEP CURRENT EXITS. Do not deploy be10_b0 (or any variant) to live now.**

Advancement, in order of merit:

1. **wick_trap be10_b0 — CONDITIONAL CANDIDATE (not deploy).** Only case with a clean loss-cut mechanism (grossP intact, grossL −13%), distributed across years, train+holdout co-move, holdout n=54. Advance to validation; do not wire to live until it clears section 7.
2. **liquidity_sweep be10_b0 — HOLD (directional only).** Gain is reshuffle-driven (grossP +25%, 2024 losses uncut) and 2-window concentrated; holdout n=29 < 30. Not separable from noise.
3. **spring be10_b0 — HOLD (high suspicion).** Rescues an in-sample-losing archetype by radical character change (WR→29.5%, count −38%). Matches the V12 in-sample exit-rescue failure pattern.
4. **exhaustion_reversal — REJECT.** Fails Rule 9 (train 1.18→1.03). Confirmed exclude despite the MFE 24% give-back motivation.

**If/when wick_trap is approved by the user, wiring (recommendation only — NOT applied):**
- Add per-archetype exit params to `configs/archetypes/wick_trap.yaml` (scratch copy first), e.g.:
  ```yaml
  exits:
    breakeven_trigger_R: 1.0   # raise stop once high reaches entry + 1.0R
    breakeven_buffer_R: 0.0    # move stop to exactly entry
  ```
- Consume them in `engine/archetypes/exit_logic.py` with the **same no-lookahead ordering** as the harness: evaluate current-bar exits at the existing stop FIRST, then raise the stop for subsequent bars. Track `R0` (entry-to-initial-stop) and a one-shot "moved" flag per position.
- Requires explicit user approval (Standing Order: exit-behavior + config change).

**Honest expected effect (wick_trap, if it survives validation):** modest PF lift (~1.46→1.66 in this backtest), driven by converting give-back losers to scratches; **bears still bleed** (2018/2022 ~unchanged); **gives up some extreme-bull tail** (2020 −$4.2K, 2023 −$0.5K) because winners that dip to entry after +1R get cut before the big run; **higher variance / lower win rate** by design (more scratches).

---

## 7. What This Doesn't Test / Required Validation Before Live

1. **CPCV across folds** — single WFO split + per-year only. The chaotic buffer axis (section 5) mandates combinatorial purged CV before any deploy, and confirmation of a stable (trigger, buffer) plateau, not a corner optimum.
2. **Full multi-archetype mode** — study is **isolated single-archetype** (other 15 disabled). In live, BE exit-timing interacts with cross-archetype **dedup**; Rule 7's failure mode (system-up while target-down via routing) is live and **entirely untested here**. Re-run the candidate in the full system and report target-archetype PnL/PF/count separately from system metrics.
3. **Trade-population confound** — be10_b0 vs baseline are different trade sets (count moves both directions across the grid). A cleaner test would freeze the entry set and vary only the stop, to isolate the exit effect from re-entry reshuffling.
4. **Reshuffle vs mechanism** — for liquidity_sweep/spring, attribute gains to grossL reduction (real BE) vs grossP increase (reshuffle luck) before trusting them.
5. **Live parity** — fills at the moved stop assume backtest stop-fill behavior; validate against live slippage at breakeven levels.

---

## 8. Files

- **Read-only analysis** of `results/champion_v14_breakeven/**` and `scripts/champion/breakeven_study.py`; engine logic at `bin/backtest_v11_standalone.py:1846-1914` (`get_performance_stats`).
- **This report:** `docs/knowledge/breakeven_study_verdict_2026_06_29.md`.
- **Production code/configs: UNTOUCHED.** No production change made or applied.
