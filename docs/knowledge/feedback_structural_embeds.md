---
name: structural_embed_experiments
description: Hard and soft gate embedding of momentum features into structural checks — regression findings
type: feedback
---

Adding momentum/volatility features (ema_slope, bb_width, effort_result_ratio) as gates in structural checks or YAML REGRESSES performance, even as soft gates.

**Why:** Wick traps and liquidity sweeps catch REVERSAL points where momentum has just turned negative. Adding `ema_slope >= 0` filters out the most valuable turning-point trades. The diagnostic shows these features are predictive (d=0.87 for ema_slope_21 on wick_trap winners), but they're predictive of WHICH wins are biggest, not of which signals should be blocked. Big winners happen at trend reversals where momentum is transitioning.

**Results tested:**
- Hard structural check (logic.py): PF 1.72→1.65, PnL $198K→$180K (wick_trap lost $25K from 50 filtered trades)
- Soft YAML gates: PF 1.72→1.58, PnL $198K→$126K (even worse — hard-mode archetypes treat soft gates as hard vetoes)

**How to apply:**
1. NEVER add ema_slope/momentum gates to wick_trap or liquidity_sweep — they catch reversals
2. If gate_mode is `hard`, ALL gates are hard vetoes regardless of nan_policy
3. The right place for momentum features is position SIZING or exit logic, not entry filtering
4. Optuna-optimized thresholds already capture the momentum edge through dynamic threshold calibration
