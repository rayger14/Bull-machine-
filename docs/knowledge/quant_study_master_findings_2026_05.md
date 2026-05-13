# Master Findings: 5-Study Quant Investigation (April-May 2026)

**Period**: 2026-05-04 to 2026-05-13
**Question that started this**: "Why are CB and LC dominating live trades while other archetypes stay silent?"
**Studies run**: 4 gate ablations (failed_continuation, retest_cluster, order_block_retest, spring) + 1 dedup fairness investigation

## TL;DR

After 5 sequential studies, the answer to "why don't other archetypes trade" is **not** what we expected:

- **Gates are not the problem.** Loosening YAML gates produces nearly zero system-level PnL improvement.
- **Dedup is not the problem either.** No alternative dedup mode beat the status quo on OOS PF.
- **The real bottleneck**: fusion-weight asymmetry between near-identical archetypes (e.g., `trap_within_trend` is functionally similar to `wick_trap` but scores lower → loses every dedup → never trades).
- **Implication**: keep current production config, target the next study at per-archetype fusion-weight rebalancing.

## Study-by-Study Summary

### 1. `failed_continuation` (chop gate is real, but +1% system impact)
| Metric | Baseline | chop_score 0.25 → 0.80 |
|--------|---------:|----------------------:|
| OOS FC trades | 0 | 21 |
| OOS FC PF | — | 4.77 |
| OOS FC WR | — | 95.2% |
| **System PnL Δ** | — | **+$656 (+1.1%)** |

Chop ceiling **is** the bottleneck. Loosening unlocks high-quality FC trades. But system PnL barely improves because new trades cannibalize other archetypes via dedup. **Recommendation: don't ship standalone.**

### 2. `retest_cluster` (gates not the bottleneck — dedup steals 99.8% of fires)
- L wins dedup only **1 of 422 times** in baseline (0.24%)
- Maximally permissive variant inflates L candidate bars 13× → still 0 OOS trades
- Root cause located in `engine/integrations/isolated_archetype_engine.py:610-629` (best_per_direction = pure max fusion)
- **Recommendation: don't touch L's YAML. Fix dedup or fusion weights instead.**

### 3. `order_block_retest` (33 variants, zero OBR trades in any of them)
- Sweep covered: `base_threshold`, `bos_atr_B`, `fib_time_cluster`, `fusion_threshold`, gate-drops, gate_mode
- **Every single variant produces 0 OBR trades** in OOS
- Most likely cause: identity gate `_check_B` rarely fires AND/OR dedup steals it 100% of the time
- **Recommendation: don't touch OBR's YAML. Diagnose identity-firing rate first.**

### 4. `spring` (partial — 6 of ~15 variants; same null result)
- Agent stopped early due to time budget
- Tested variants: `wyc_005`, `wyc_010` — both produced 0 spring trades
- Consistent with the OBR / retest_cluster pattern
- **Recommendation: don't pursue further gate ablation on spring.**

### 5. Dedup fairness investigation (no alternative beats status quo)

936 dedup-loss events analyzed across 2020-2024. Three archetypes do 88% of blocking:
| Blocker | % of blocks |
|---------|------------:|
| wick_trap | 65% |
| liquidity_sweep | 13% |
| oi_divergence | 10% |

**Biggest victim**: `trap_within_trend` — 478 losses, 96% to `wick_trap`. Phantom outcomes look profitable (PF 1.42 paper, +96 bp/trade).

Six dedup modes tested on WFO:

| Mode | Train PF | Test PF | Test PnL | Sharpe | Verdict |
|------|---------:|--------:|---------:|-------:|---------|
| **status_quo** | **1.55** | **1.79** | **$57,837** | **1.67** | **Winner** |
| pass_through | 1.48 | 1.78 | $64,800 | 1.70 | +12% PnL but train MDD worsens to -16.7% |
| hybrid_rr_fusion | 1.43 | 1.73 | $58,448 | 1.64 | redistributes, no net alpha |
| round_robin | 1.40 | 1.69 | $56,042 | 1.57 | unblocked underdogs LOST $5,821 |
| unique_sl_zone | 1.49 | 1.79 | $57,837 | 1.67 | degenerates to baseline |
| normalized | 1.25 | 1.65 | $53,775 | 1.52 | **FAIL: 32% PF gap (overfit)** |

**Critical insight**: phantom outcomes LIED. Round-robin correctly unblocked underdogs (`trap_within_trend` 3→11 trades) — but those unblocked trades lost money. Dedup doesn't just unblock; it also reroutes capital among winners. **Recommendation: keep status quo.**

## What we learned about the system

1. **The 16-archetype design is sound.** Most are correctly silent in compressed BTC — they're built for *different regimes* (trend continuation, exhaustion, sweeps). Their silence isn't a bug.

2. **CB/LC dominance is regime-correct.** When BTC stops compressing, other archetypes will activate. We saw this in 2020-2022 historical data where `liquidity_sweep`, `retest_cluster`, `wick_trap` were the top earners.

3. **Phantom outcomes are unreliable.** Simulated "what would have happened if we'd taken this trade" can show profitable phantom PF (1.42) that completely fails to materialize in real backtest. Dedup interacts with capital flow in ways phantom analysis misses.

4. **The real lever is fusion weights.** `trap_within_trend` and `wick_trap` fire on near-identical patterns (wick anomaly + trend context). The fact that wick_trap wins dedup 96% of the time means their fusion weights aren't calibrated against each other — wick_trap's weights produce systematically higher scores on the same bar.

5. **Standing orders held up.** Throughout 5 studies, no production config was modified. All investigation work lives in scratch worktrees + result CSVs. Confidence in any future production change can be high because the baseline was preserved.

## Branches still alive (reference data)

| Branch | Worktree | What's there |
|--------|----------|--------------|
| `quant/failed-continuation-gate-ablation` | `agent-a33d9c4a7132fba8a` | 17 variants, full report |
| `chore/retest-cluster-gate-ablation` (committed `215ad26`) | `agent-ad767be3c6c1f78f9` | 15 variants + dedup root-cause |
| `quant/order-block-retest-gate-ablation` | `agent-adb22984d5837ab30` | 33 variants, null result |
| `quant/spring-gate-ablation` | `agent-a1b7352a8eba4b88b` | 6 variants partial |
| `quant/dedup-fairness-investigation` | `agent-a624102143669ea51` | 6 dedup modes + full WFO |

## Recommended next steps (in order)

1. **Fusion-weight rebalancing study** (highest expected EV)
   - Focus on `trap_within_trend` vs `wick_trap` first — they're the most-blocked vs most-blocking pair
   - Use Optuna with `/optuna-optimize` skill enforced (WFO + CPCV)
   - Goal: find fusion-weight values that make TWT score competitively with wick_trap on shared bars without breaking either's individual edge
2. **Identity-gate firing-rate instrumentation**
   - For `order_block_retest` and `spring`, log how often `_check_B` and `_check_A` return true on 2023-2024 bars
   - If <1% of bars, the bottleneck is the identity-gate definition, not the YAML
3. **Regime-stratified ablation** (lower priority)
   - The 5 studies all used 2020-2024 WFO. Stratify by regime (bull/bear/neutral/crisis) before any future config change
   - 2023-2024 is bull-skewed — findings may not generalize to bear/crisis
4. **CB Q0 floor** (still open from May 4 audit)
   - Adding `fusion_threshold: 0.20` floor to CB would block the 5 lowest-fusion CB trades
   - Q0 backtest showed −$1,712 across 3 trades — saves capital with high confidence
   - Smallest risk, simplest change, but requires explicit user approval (Standing Order territory)

## Methodology notes for future studies

- **WFO is mandatory** — never trust full-range optimization
- **Phantom outcomes can lie** — always validate with real backtest before recommending a change
- **30% train/test PF gap = automatic reject** — Phase 1 anti-overfit guard
- **Worktrees + scratch result dirs > forking the backtester** — every study here preserved the production codebase
- **Small archetype-level wins ≠ system-level wins** — always measure system PnL Δ, not just archetype Δ
- **Honest sample-size caveats matter** — n=21 OOS is suggestive, not conclusive

## Closing observation

What started as "why aren't other archetypes firing?" became a 5-study journey that mostly **validated the current production config**. We avoided 3-4 well-intentioned but counterproductive changes:
- Gate loosening that would have cannibalized winners
- Dedup modes that would have unblocked unprofitable trades
- Per-archetype tuning that would have overfit to the bull regime

The system as currently configured is doing approximately the right thing for the current regime. The remaining alpha lives in fusion-weight calibration, not in gate or dedup changes.
