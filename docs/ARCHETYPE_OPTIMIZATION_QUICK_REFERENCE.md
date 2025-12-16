# Archetype Optimization Quick Reference

**Quick lookup table for implementing archetype-specific threshold optimization.**

---

## Critical Parameters by Archetype

### Trap Within Trend (A/G/K) - HIGHEST PRIORITY

```json
{
  "fusion_threshold": [0.40, 0.55],
  "min_liquidity": [0.12, 0.20],
  "adx_min": [25, 35],
  "wick_against_trend": [0.30, 0.50],
  "archetype_weight": [2.0, 4.0]
}
```

**Target:** PF 2.5-4.0, DD ≤15%, 120+ trades/year
**Trials:** 250 (150 coarse + 100 fine)
**Runtime:** ~8 hours

---

### Order Block Retest (B/H/L)

```json
{
  "fusion_threshold": [0.30, 0.45],
  "min_liquidity": [0.15, 0.25],
  "boms_strength_min": [0.30, 0.50],
  "wyckoff_score_min": [0.30, 0.50],
  "archetype_weight": [1.5, 3.0]
}
```

**Target:** PF 2.2-3.8, DD ≤18%, 80+ trades/year
**Trials:** 200 (120 coarse + 80 fine)
**Runtime:** ~6 hours

---

### Long Squeeze (S5) - CORRECTED LOGIC

```json
{
  "fusion_threshold": [0.34, 0.46],
  "min_liquidity": [0.08, 0.18],
  "funding_z_min": [1.2, 2.0],
  "rsi_min": [68, 78],
  "oi_change_min": [0.05, 0.15],
  "archetype_weight": [2.0, 3.5]
}
```

**Target:** PF 2.0-3.0, DD ≤22%, 40+ trades/year
**Trials:** 160 (100 coarse + 60 fine)
**Runtime:** ~5 hours
**CRITICAL:** Positive funding → LONG squeeze DOWN (not short squeeze up)

---

## Regime Routing Quick Lookup

| Archetype | risk_on | neutral | risk_off | crisis |
|-----------|---------|---------|----------|--------|
| **Trap Within Trend (A/G/K)** | 1.2x | 1.0x | 0.3x | 0.0x |
| **Order Block (B/H/L)** | 1.1x | 1.0x | 0.5x | 0.2x |
| **BOS/CHoCH (C)** | 1.3x | 1.0x | 0.4x | 0.1x |
| **Long Squeeze (S5)** | 0.3x | 0.8x | 2.0x | 2.5x |
| **Failed Rally (S2)** | **DISABLED** | **DISABLED** | **DISABLED** | **DISABLED** |

---

## Optimization Schedule (Parallel Execution)

### Group 1: Critical (8 workers, 14 hours)
1. **A/G/K (Trap Within Trend):** 250 trials
2. **B/H/L (Order Block Retest):** 200 trials

### Group 2: High Priority (4 workers, 10 hours)
3. **C (BOS/CHoCH):** 160 trials
4. **S5 (Long Squeeze):** 160 trials

### Group 3: Medium (2 workers, 8 hours)
5. **F (Expansion Exhaustion):** 120 trials
6. **M (Re-Accumulate):** 120 trials

### Group 4: Low (1 worker, 9 hours)
7. **D (Failed Continuation):** 90 trials
8. **S4 (Distribution Climax):** 90 trials (Phase 2)
9. **S1 (Breakdown):** 90 trials (BLOCKED - feature backfill)

**Total:** 1,280 trials, ~14 hours (8-core parallel)

---

## Multi-Objective Scoring Formula

```python
score = (
    (pf / target_pf) ** 1.5 * 0.40 +        # 40% PF weight
    (sharpe / 1.5) * 0.25 +                 # 25% Sharpe weight
    regime_multiplier * 0.15                # 15% regime alignment
) - (
    dd_penalty * 0.12 +                     # 12% DD penalty
    trade_count_penalty * 0.08              # 8% trade frequency penalty
)
```

**Hard Constraints (Immediate Pruning):**
- PF < 1.0 → reject
- DD > 35% → reject
- Trade count < 5 → reject

---

## Multi-Fidelity Validation Thresholds

| Level | Period | Prune If | Keep |
|-------|--------|----------|------|
| **L1 (1mo)** | 2024-09 | PF < 1.0 | Top 50% |
| **L2 (3mo)** | 2024-07 to 2024-09 | PF < 1.3 OR trades < 10 | Top 20% |
| **L3 (9mo)** | 2024-01 to 2024-09 | Full scoring | Top 5 |
| **L4 (OOS)** | 2024-10 to 2024-12 | Degradation > 30% | Best 1 |

---

## Production Deployment Checklist

### Minimum Viable Criteria (ALL must pass)
- [ ] OOS PF ≥ 1.5
- [ ] OOS Max DD ≤ 20%
- [ ] PF degradation (IS→OOS) < 30%
- [ ] Minimum 20 trades in OOS
- [ ] No >5 consecutive losses in OOS

### Bull Archetype Additional Criteria
- [ ] Risk-on regime PF ≥ 2.0
- [ ] Crisis regime properly muted (PF < 1.0 OR trades < 5)

### Bear Archetype Additional Criteria
- [ ] Risk-off regime PF ≥ 1.8
- [ ] Risk-on regime properly suppressed (PF < 1.2 OR trades < 5)

---

## Overfit Detection Flags

1. **IS/OOS Divergence:** IS PF > 3.0 AND OOS PF < 1.5 → REJECT
2. **Regime Instability:** PF variance across regimes > 2.0 → REJECT
3. **Boundary Parameters:** 3+ params at min/max bounds → REJECT
4. **Statistical Insignificance:** Bootstrap 95% CI lower bound < 1.2 → REJECT

---

## Feature Dependencies (Blockers)

| Archetype | Blocked By | Status | Resolution |
|-----------|------------|--------|------------|
| **S1 (Breakdown)** | `liquidity_score` missing | BLOCKED | Run `bin/backfill_liquidity_score.py` |
| **S2 (Failed Rally)** | PF 0.48 after 157 trials | **PERMANENTLY DISABLED** | Remove from all configs |

---

## Optuna Configuration Template

```python
study = optuna.create_study(
    direction="maximize",
    sampler=TPESampler(
        seed=42,
        multivariate=True,
        n_startup_trials=20
    ),
    pruner=HyperbandPruner(
        min_resource=1,
        max_resource=9,
        reduction_factor=3
    )
)

# Parameter definition example (Trap Within Trend)
trial.suggest_float("fusion_threshold", 0.40, 0.55, step=0.01)
trial.suggest_float("min_liquidity", 0.12, 0.20, step=0.01)
trial.suggest_int("adx_min", 25, 35, step=1)
trial.suggest_float("wick_against_trend", 0.30, 0.50, step=0.02)
trial.suggest_float("archetype_weight", 2.0, 4.0, step=0.1)
```

---

## Implementation Priority Order

1. **Week 1:** Infrastructure + A/G/K optimization
2. **Week 2:** B/H/L + C optimization
3. **Week 3:** S5 + F + M optimization
4. **Week 4:** D + S4 optimization (if unblocked)
5. **Week 5:** Config unification + validation
6. **Week 6:** Shadow mode deployment
7. **Week 7-10:** Phased production rollout

---

## Emergency Disable Procedure

**Instant disable via config (no code changes):**
```json
{
  "archetypes": {
    "enable_A": false,
    "enable_B": false,
    "enable_S5": false
  }
}
```

**Archetype-specific muting:**
```json
{
  "archetypes": {
    "routing": {
      "risk_off": {
        "weights": {
          "trap_within_trend": 0.0,
          "long_squeeze": 0.0
        }
      }
    }
  }
}
```

---

**Document Version:** 1.0
**Last Updated:** 2025-11-17
**Companion:** See `ARCHETYPE_OPTIMIZATION_REQUIREMENTS_SPEC.md` for full details
