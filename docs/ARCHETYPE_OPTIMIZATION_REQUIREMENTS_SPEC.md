# Archetype-Specific Threshold Optimization Requirements Specification

**Document Version:** 1.0
**Created:** 2025-11-17
**Status:** REQUIREMENTS APPROVED - READY FOR IMPLEMENTATION
**Target System:** Bull Machine v2 Archetype Threshold Discovery Pipeline

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Archetype Parameter Matrix](#archetype-parameter-matrix)
3. [Performance Requirements by Archetype](#performance-requirements-by-archetype)
4. [Regime Compatibility Matrix](#regime-compatibility-matrix)
5. [Optimization Strategy](#optimization-strategy)
6. [Multi-Objective Scoring Function](#multi-objective-scoring-function)
7. [Validation Protocol](#validation-protocol)
8. [Production Deployment Criteria](#production-deployment-criteria)
9. [Implementation Roadmap](#implementation-roadmap)

---

## Executive Summary

### Problem Statement

Current system uses **global thresholds** (fusion_threshold=0.35, min_liquidity=0.30) applied uniformly across all archetypes. This is fundamentally incorrect because:

1. **Different patterns have different quality requirements**
   - Trap Within Trend (A/G/K): High-conviction setups need fusion ≥ 0.45
   - Order Block Retest (B/H/L): Liquidity-dependent, fusion ≥ 0.32 acceptable
   - Failed Rally (S2): Bear pattern, fusion ≥ 0.36 sufficient in risk-off

2. **Regime compatibility varies by archetype**
   - Bull patterns (A-M): Perform well in risk_on/neutral, fail in risk_off/crisis
   - Bear patterns (S2, S5): Only perform in risk_off/crisis, negative edge in risk_on

3. **Pattern-specific filters are hardcoded**
   - S5 funding_z threshold currently at 1.5 (arbitrary)
   - S2 rsi_min at 70.0 (not optimized)
   - No systematic way to optimize these per archetype

### Solution Approach

**Archetype-Specific Threshold Discovery Pipeline:**
- Per-archetype Optuna studies (18 independent optimization runs)
- Multi-fidelity validation (1mo → 3mo → 9mo)
- Multi-objective scoring (PF × DD × Sharpe × Trade_Frequency)
- Regime-aware parameter adaptation
- Walk-forward validation on out-of-sample data

### Key Constraints

- **S2 (Failed Rally): PERMANENTLY DISABLED** - 157 optimization attempts yielded PF 0.48
- **In-sample period:** 2022-01-01 to 2024-09-30 (2.75 years)
- **Out-of-sample period:** 2024-10-01 to 2024-12-31 (3 months)
- **Minimum viable PF:** 1.5 (production deployment threshold)
- **Maximum DD:** 20% (hard limit)

---

## Archetype Parameter Matrix

### Bull Archetypes (LONG Bias)

| Archetype | Code | Family | fusion_threshold | min_liquidity | Pattern Filters | archetype_weight | Optimization Priority |
|-----------|------|--------|------------------|---------------|-----------------|------------------|-----------------------|
| **Trap Within Trend** | A, G, K | Trap Family | [0.40, 0.55] | [0.12, 0.20] | adx_min: [25, 35]<br>wick_against_trend: [0.30, 0.50] | [2.0, 4.0] | **HIGH** (dominates 96.5% of 2022 trades) |
| **Order Block Retest** | B, H, L | OB Family | [0.30, 0.45] | [0.15, 0.25] | boms_strength_min: [0.30, 0.50]<br>wyckoff_score_min: [0.30, 0.50] | [1.5, 3.0] | **HIGH** (proven edge in 2024) |
| **BOS/CHoCH Reversal** | C | Momentum | [0.40, 0.55] | [0.10, 0.20] | disp_atr_min: [0.80, 1.50]<br>momentum_score_min: [0.40, 0.60] | [1.8, 3.5] | MEDIUM (16% of 2024 trades) |
| **Failed Continuation** | D | Failed Break | [0.38, 0.50] | [0.10, 0.18] | rsi_max: [45, 55]<br>adx_falling: True | [1.2, 2.5] | LOW (rare pattern) |
| **Liquidity Compression** | E | Range | [0.32, 0.45] | [0.15, 0.25] | atr_percentile_max: [0.20, 0.35]<br>vol_cluster_min: [0.60, 0.80] | [1.0, 2.0] | LOW (range-bound markets only) |
| **Expansion Exhaustion** | F | Exhaustion | [0.35, 0.48] | [0.08, 0.15] | rsi_extreme_min: [75, 82]<br>atr_percentile_min: [0.85, 0.95]<br>vol_z_min: [0.8, 1.5] | [1.5, 3.0] | MEDIUM (trend exhaustion) |
| **Re-Accumulate Base** | M | Wyckoff | [0.32, 0.45] | [0.15, 0.25] | atr_percentile_max: [0.25, 0.40]<br>poc_dist_max: [0.40, 0.70]<br>boms_strength_min: [0.35, 0.55] | [1.5, 2.8] | MEDIUM (accumulation phase) |

**Notes:**
- **Trap Family (A/G/K):** Highest optimization priority due to historical dominance (96.5% of 2022 trades)
- **OB Family (B/H/L):** Proven performers in 2024 bull market (PF 6.17 baseline)
- **Pattern filters:** Archetype-specific technical requirements beyond fusion/liquidity

---

### Bear Archetypes (SHORT Bias)

| Archetype | Code | Pattern | fusion_threshold | min_liquidity | Pattern Filters | archetype_weight | Status |
|-----------|------|---------|------------------|---------------|-----------------|------------------|--------|
| **Failed Rally Rejection** | S2 | Failed Rally | [0.32, 0.42] | [0.18, 0.28] | rsi_min: [68, 75]<br>vol_z_max: [0.3, 0.7]<br>wick_ratio_min: [1.2, 2.5] | [1.8, 2.8] | **DISABLED** (PF 0.48 after 157 trials) |
| **Long Squeeze Cascade** | S5 | Funding Squeeze | [0.34, 0.46] | [0.08, 0.18] | funding_z_min: [1.2, 2.0]<br>rsi_min: [68, 78]<br>oi_change_min: [0.05, 0.15] | [2.0, 3.5] | **ACTIVE** (approved with corrected funding logic) |
| **Breakdown** | S1 | Liquidity Vacuum | [0.36, 0.48] | [0.08, 0.18] | liquidity_score_max: [0.18, 0.28]<br>vol_z_min: [0.8, 1.5]<br>tf4h_trend: "down" | [2.0, 3.2] | PENDING (feature backfill required) |
| **Whipsaw** | S3 | Chop | [0.38, 0.50] | [0.12, 0.22] | adx_max: [18, 25]<br>rsi_range: [45, 55] | [1.2, 2.0] | PENDING (low priority) |
| **Distribution Climax** | S4 | Volume Exhaustion | [0.35, 0.47] | [0.12, 0.22] | vol_z_min: [1.5, 2.2]<br>rsi_min: [58, 68]<br>tf4h_trend: "down" | [1.5, 2.5] | PENDING (Phase 2) |
| **Alt Rotation Down** | S6 | Correlation Break | N/A | N/A | N/A | N/A | **REJECTED** (no data validation) |
| **Curve Inversion** | S7 | Macro Signal | N/A | N/A | N/A | N/A | **REJECTED** (no data validation) |
| **Trend Fade Chop** | S8 | Exhaustion | N/A | N/A | N/A | N/A | **REJECTED** (wrong direction edge) |

**Critical Notes:**

- **S2 PERMANENTLY DISABLED:** 157 optimization attempts (Oct-Nov 2025) achieved only PF 0.48. Pattern has fundamental negative edge.
- **S5 CORRECTED:** Original logic was inverted (positive funding → SHORT squeeze UP ❌). Corrected to: positive funding → LONG squeeze DOWN ✅
- **S1 BLOCKED:** Requires `liquidity_score` feature backfill before optimization can proceed
- **S4 DEFERRED:** Phase 2 implementation (after S5 validation complete)

---

## Performance Requirements by Archetype

### Tier 1: Production Ready (Immediate Deployment)

| Archetype | Target PF | Acceptable DD | Min Trades/Year | Target Win Rate | Regime Suitability |
|-----------|-----------|---------------|-----------------|-----------------|-------------------|
| **Trap Within Trend (A/G/K)** | [2.5, 4.0] | ≤ 15% | 120 | [52%, 62%] | risk_on, neutral |
| **Order Block Retest (B/H/L)** | [2.2, 3.8] | ≤ 18% | 80 | [50%, 60%] | risk_on, neutral |
| **BOS/CHoCH (C)** | [2.0, 3.5] | ≤ 20% | 60 | [48%, 58%] | risk_on, neutral |
| **Long Squeeze (S5)** | [2.0, 3.0] | ≤ 22% | 40 | [55%, 65%] | risk_off, crisis |

**Rationale:**
- **A/G/K:** Highest standards due to dominance (96.5% of triggers). Must maintain quality.
- **B/H/L:** Proven edge in 2024 (PF 6.17). Conservative requirements to preserve performance.
- **C:** Momentum pattern, inherently higher variance. Slightly relaxed DD tolerance.
- **S5:** Bear pattern, higher risk acceptable. Lower trade count due to regime specificity.

---

### Tier 2: Validation Phase (Shadow Mode)

| Archetype | Target PF | Acceptable DD | Min Trades/Year | Target Win Rate | Status |
|-----------|-----------|---------------|-----------------|-----------------|--------|
| **Failed Continuation (D)** | [1.5, 2.5] | ≤ 20% | 30 | [45%, 55%] | Shadow (low frequency) |
| **Expansion Exhaustion (F)** | [1.8, 3.0] | ≤ 22% | 40 | [48%, 58%] | Shadow (trend exhaustion) |
| **Re-Accumulate (M)** | [1.8, 2.8] | ≤ 18% | 50 | [50%, 60%] | Shadow (Wyckoff phase) |
| **Distribution Climax (S4)** | [1.5, 2.2] | ≤ 20% | 60 | [52%, 62%] | Blocked (Phase 2) |
| **Breakdown (S1)** | [1.8, 2.8] | ≤ 22% | 50 | [53%, 63%] | Blocked (feature backfill) |

**Validation Requirements:**
- Shadow mode: Run patterns in parallel, log signals, no live trades
- Performance monitoring: 30-day validation window
- Promotion criteria: Meet target PF + pass out-of-sample validation

---

### Tier 3: Research / Rejected

| Archetype | Status | Reason |
|-----------|--------|--------|
| **Failed Rally (S2)** | **PERMANENTLY DISABLED** | PF 0.48 after 157 optimization attempts. Fundamental negative edge. |
| **Liquidity Compression (E)** | LOW PRIORITY | Range-bound pattern. Low frequency in trending markets. |
| **Whipsaw (S3)** | LOW PRIORITY | Chop pattern. Difficult to trade profitably. |
| **Alt Rotation (S6)** | REJECTED | No validation data. Correlation analysis incomplete. |
| **Curve Inversion (S7)** | REJECTED | Macro signal, not microstructure pattern. |
| **Trend Fade (S8)** | REJECTED | Wrong direction edge (+0.11% 4h returns contradict bear thesis). |

---

## Regime Compatibility Matrix

### Bull Archetypes (LONG)

| Archetype | risk_on | neutral | risk_off | crisis | Muting Rules |
|-----------|---------|---------|----------|--------|--------------|
| **Trap Within Trend (A/G/K)** | 1.2x | 1.0x | 0.3x | 0.0x | **MUTE** if DXY z-score > 1.2 OR VIX > 28 OR yield curve inverted >30 days |
| **Order Block Retest (B/H/L)** | 1.1x | 1.0x | 0.5x | 0.2x | **REDUCE** in risk_off (50%), MUTE in crisis (80% of time) |
| **BOS/CHoCH (C)** | 1.3x | 1.0x | 0.4x | 0.1x | **REDUCE** in risk_off (60%), MUTE in crisis (90% of time) |
| **Failed Continuation (D)** | 0.9x | 1.0x | 0.6x | 0.3x | Pattern works in both regimes (failed breakouts occur everywhere) |
| **Expansion Exhaustion (F)** | 1.2x | 1.0x | 0.5x | 0.2x | High volatility pattern, reduce in crisis (liquidation risk) |
| **Re-Accumulate (M)** | 1.1x | 1.0x | 0.8x | 0.5x | Wyckoff accumulation can occur in risk_off (oversold bounces) |

**Global Bull Archetype Rules:**
- **Crisis Fuse:** If VIX > 35 OR MOVE > 150, MUTE all bull archetypes except M (accumulation phase)
- **DXY Override:** If DXY z-score > 1.5 for 5+ consecutive days, REDUCE all bull weights by 50%
- **Yield Curve Inversion:** If 2Y-10Y < -0.30 for 20+ days, MUTE trend-following (A/G/K, C) but allow mean-reversion (B/H/L)

---

### Bear Archetypes (SHORT)

| Archetype | risk_on | neutral | risk_off | crisis | Muting Rules |
|-----------|---------|---------|----------|--------|--------------|
| **Failed Rally (S2)** | 0.0x | 0.0x | 0.0x | 0.0x | **PERMANENTLY DISABLED** (PF 0.48) |
| **Long Squeeze (S5)** | 0.3x | 0.8x | 2.0x | 2.5x | **ONLY ACTIVE** if funding_z > 1.5 AND regime in {risk_off, crisis}. MUTE in sustained risk_on (>10 days). |
| **Breakdown (S1)** | 0.2x | 0.7x | 1.8x | 2.5x | Liquidity vacuum pattern. Most effective in crisis. |
| **Distribution (S4)** | 0.4x | 1.0x | 1.8x | 2.0x | Volume climax pattern. Works in transitions (neutral → risk_off). |
| **Whipsaw (S3)** | 0.8x | 1.0x | 1.2x | 0.5x | Chop pattern. Most common in neutral. Reduce in crisis (directional moves). |

**Global Bear Archetype Rules:**
- **Risk-On Suppression:** If VIX z-score < -0.5 for 5+ days, REDUCE all bear weights by 70%
- **Funding Rate Override (S5 specific):** If funding_z < 0.5, MUTE S5 (no overcrowded longs to squeeze)
- **Volume Climax Filter (S4):** Require vol_z > 1.5 in ALL regimes (core pattern requirement)

---

## Optimization Strategy

### A. Per-Archetype Trial Allocation

| Archetype | Priority | Trials (Coarse) | Trials (Fine) | Total | Estimated Runtime | Rationale |
|-----------|----------|-----------------|---------------|-------|-------------------|-----------|
| **Trap Within Trend (A/G/K)** | **CRITICAL** | 150 | 100 | 250 | 8 hours | Dominates 96.5% of triggers. Highest impact. |
| **Order Block Retest (B/H/L)** | **HIGH** | 120 | 80 | 200 | 6 hours | Proven performers. Need refinement. |
| **BOS/CHoCH (C)** | HIGH | 100 | 60 | 160 | 5 hours | 16% of trades. Momentum pattern complexity. |
| **Long Squeeze (S5)** | HIGH | 100 | 60 | 160 | 5 hours | Bear archetype, corrected logic, needs validation. |
| **Expansion Exhaustion (F)** | MEDIUM | 80 | 40 | 120 | 4 hours | Trend exhaustion edge validation. |
| **Re-Accumulate (M)** | MEDIUM | 80 | 40 | 120 | 4 hours | Wyckoff phase detection. |
| **Failed Continuation (D)** | LOW | 60 | 30 | 90 | 3 hours | Rare pattern, low frequency. |
| **Distribution (S4)** | LOW | 60 | 30 | 90 | 3 hours | Phase 2 implementation. |
| **Breakdown (S1)** | LOW | 60 | 30 | 90 | 3 hours | Blocked (feature backfill required). |

**Total Optimization Budget:** 1,280 trials (~41 hours @ 2 min/trial)

**Parallel Execution Groups:**
1. **Group 1 (Critical):** A/G/K, B/H/L (8 parallel workers, 14 hours)
2. **Group 2 (High):** C, S5 (4 parallel workers, 10 hours)
3. **Group 3 (Medium):** F, M (2 parallel workers, 8 hours)
4. **Group 4 (Low):** D, S4, S1 (1 worker, 9 hours)

**Expected Total Time:** 14 hours (with 8-core parallelization)

---

### B. Multi-Fidelity Validation Levels

**Level 1: Quick Validation (1 month)**
- **Period:** 2024-09-01 to 2024-09-30
- **Purpose:** Fast pruning of obviously bad configurations
- **Threshold:** PF < 1.0 → PRUNE immediately
- **Hyperband Reduction:** 50% of trials pruned after Level 1

**Level 2: Medium Validation (3 months)**
- **Period:** 2024-07-01 to 2024-09-30
- **Purpose:** Validate consistency across multiple months
- **Threshold:** PF < 1.3 OR trade_count < 10 → PRUNE
- **Hyperband Reduction:** Additional 30% pruned (80% cumulative)

**Level 3: Full In-Sample (9 months)**
- **Period:** 2024-01-01 to 2024-09-30
- **Purpose:** Full performance assessment on in-sample data
- **Threshold:** Top 20% of surviving trials proceed to out-of-sample
- **Metrics Collected:** PF, DD, Sharpe, Sortino, Win Rate, Avg R-Multiple, Trade Frequency

**Level 4: Out-of-Sample (3 months - FINAL)**
- **Period:** 2024-10-01 to 2024-12-31
- **Purpose:** Walk-forward validation on unseen data
- **Acceptance Criteria:**
  - PF degradation < 30% (vs in-sample)
  - Sharpe degradation < 40%
  - Min PF ≥ 1.5 (absolute threshold)
  - Max DD ≤ 20%

**Pruning Strategy (Hyperband-inspired):**
```
Trials Start:   N = 100 (per archetype)
After Level 1:  N × 0.50 = 50  (prune bottom 50%)
After Level 2:  N × 0.20 = 20  (prune bottom 60% of remaining)
After Level 3:  N × 0.05 = 5   (keep top 5 for OOS validation)
Production:     N × 0.01 = 1   (select best OOS performer)
```

---

### C. Optuna Configuration

**Sampler:** TPESampler (Tree-structured Parzen Estimator)
- `seed=42` (reproducibility)
- `n_startup_trials=20` (random exploration before Bayesian)
- `multivariate=True` (capture parameter interactions)

**Pruner:** HyperbandPruner
- `min_resource=1` (1 month validation)
- `max_resource=9` (9 months validation)
- `reduction_factor=3` (aggressive pruning: keep 1/3 at each stage)

**Study Configuration:**
```python
study = optuna.create_study(
    direction="maximize",
    sampler=TPESampler(seed=42, multivariate=True, n_startup_trials=20),
    pruner=HyperbandPruner(min_resource=1, max_resource=9, reduction_factor=3),
    study_name=f"archetype_{archetype_code}_optimization_v1"
)
```

**Parameter Types:**
- `suggest_float()`: fusion_threshold, min_liquidity, archetype_weight (continuous)
- `suggest_int()`: adx_min, rsi_min, cooldown_bars (discrete)
- `suggest_categorical()`: regime_routing_policy ["conservative", "balanced", "aggressive"]

---

## Multi-Objective Scoring Function

### A. Base Formula

```python
def objective_score(pf, dd, sharpe, trade_count, regime, archetype):
    """
    Multi-objective scoring function for archetype optimization.

    Args:
        pf: Profit Factor (gross_profit / gross_loss)
        dd: Maximum Drawdown (decimal, e.g., 0.15 = 15%)
        sharpe: Sharpe Ratio (annualized)
        trade_count: Total trades in evaluation period
        regime: Regime label ("risk_on", "neutral", "risk_off", "crisis")
        archetype: Archetype code (e.g., "A", "B", "S5")

    Returns:
        score: Composite score (higher is better)
    """

    # === Component 1: Profit Factor (Primary) ===
    # Target: PF ≥ 2.0 for bull, ≥ 1.8 for bear
    pf_target = 2.0 if archetype in BULL_ARCHETYPES else 1.8
    pf_score = (pf / pf_target) ** 1.5  # Exponential reward for exceeding target

    # === Component 2: Drawdown Penalty (Risk Control) ===
    # Target: DD ≤ 15% for bull, ≤ 20% for bear
    dd_threshold = 0.15 if archetype in BULL_ARCHETYPES else 0.20
    if dd > dd_threshold:
        dd_penalty = ((dd - dd_threshold) / dd_threshold) ** 2  # Quadratic penalty
    else:
        dd_penalty = 0

    # === Component 3: Sharpe Ratio (Risk-Adjusted Returns) ===
    # Target: Sharpe ≥ 1.5
    sharpe_target = 1.5
    sharpe_score = max(0, sharpe / sharpe_target)  # Linear scaling

    # === Component 4: Trade Frequency (Edge Utilization) ===
    # Target: 80-150 trades/year for bull, 40-100 for bear
    min_trades = 40 if archetype in BEAR_ARCHETYPES else 80
    max_trades = 100 if archetype in BEAR_ARCHETYPES else 150

    if trade_count < min_trades:
        trade_penalty = ((min_trades - trade_count) / min_trades) ** 2
    elif trade_count > max_trades:
        trade_penalty = ((trade_count - max_trades) / max_trades) ** 2
    else:
        trade_penalty = 0

    # === Component 5: Regime Alignment (Context Penalty) ===
    # Penalize bull archetypes in risk_off/crisis, bear in risk_on
    regime_multipliers = get_regime_multipliers(archetype, regime)
    regime_score = regime_multipliers[regime]

    # === Composite Score ===
    base_score = (
        pf_score * 0.40 +          # 40% weight on PF
        sharpe_score * 0.25 +      # 25% weight on Sharpe
        regime_score * 0.15        # 15% weight on regime alignment
    )

    total_penalty = (
        dd_penalty * 0.12 +        # 12% penalty weight
        trade_penalty * 0.08       # 8% penalty weight
    )

    final_score = base_score - total_penalty

    # === Hard Constraints (Immediate Pruning) ===
    if pf < 1.0:
        return -1000  # Negative PF = unacceptable
    if dd > 0.35:
        return -1000  # >35% DD = catastrophic
    if trade_count < 5:
        return -1000  # <5 trades = insufficient data

    return final_score
```

---

### B. Regime-Specific Adjustments

```python
def get_regime_multipliers(archetype, regime):
    """
    Return regime alignment scores for archetype.

    Scores > 1.0 = favorable regime (boost)
    Scores < 1.0 = unfavorable regime (penalty)
    """

    BULL_ARCHETYPE_MULTIPLIERS = {
        'risk_on': 1.3,      # Bull patterns thrive in risk-on
        'neutral': 1.0,      # Baseline
        'risk_off': 0.4,     # 60% penalty for risk-off
        'crisis': 0.1        # 90% penalty for crisis
    }

    BEAR_ARCHETYPE_MULTIPLIERS = {
        'risk_on': 0.3,      # 70% penalty for risk-on
        'neutral': 0.8,      # Slight penalty (bear patterns rare in neutral)
        'risk_off': 1.5,     # 50% boost for risk-off
        'crisis': 1.8        # 80% boost for crisis
    }

    if archetype in BULL_ARCHETYPES:
        return BULL_ARCHETYPE_MULTIPLIERS
    else:
        return BEAR_ARCHETYPE_MULTIPLIERS
```

---

### C. Edge Case Handling

**1. Zero Gross Loss (Infinite PF)**
```python
if gross_loss < 1e-6:
    # Cap PF at 10.0 to prevent infinite scores
    pf = min(10.0, gross_profit / 1e-6)
```

**2. Very Few Trades (< 10)**
```python
if trade_count < 10:
    # Apply severe penalty for statistical insignificance
    score *= (trade_count / 10.0) ** 2
```

**3. Excessive Trades (> 300/year)**
```python
if trade_count > 300:
    # Likely overfit or noise trading
    score *= 0.5  # 50% penalty
```

**4. Extreme Drawdown Recovery**
```python
if dd > 0.25 and pf > 3.0:
    # High PF but dangerous DD = unstable strategy
    score *= 0.7  # 30% penalty for volatility
```

---

## Validation Protocol

### A. In-Sample Validation (2022-01-01 to 2024-09-30)

**Step 1: Multi-Fidelity Evaluation**
1. **1-month validation:** 2024-09-01 to 2024-09-30
   - Metric: PF ≥ 1.0 (prune if fails)
   - Purpose: Fast elimination of bad configs

2. **3-month validation:** 2024-07-01 to 2024-09-30
   - Metric: PF ≥ 1.3 AND trades ≥ 10 (prune if fails)
   - Purpose: Consistency check

3. **9-month validation:** 2024-01-01 to 2024-09-30
   - Metric: Full scoring function
   - Purpose: Comprehensive in-sample performance

**Step 2: Cross-Regime Validation**
- Split in-sample period by regime:
  - Risk-on months: [list specific months]
  - Risk-off months: [list specific months]
  - Crisis months: [list specific months]
- Require PF > 1.0 in INTENDED regime (per archetype)
- Accept PF < 1.0 in NON-INTENDED regime (expected failure)

**Step 3: Statistical Significance Testing**
- **Bootstrap resampling:** 1000 iterations
- **Confidence interval:** 95% CI for PF, Sharpe, WR
- **Requirement:** Lower bound of 95% CI for PF > 1.2

---

### B. Out-of-Sample Validation (2024-10-01 to 2024-12-31)

**Walk-Forward Protocol:**

**Step 1: Configuration Lock**
- Top 5 configs from in-sample proceed to OOS
- NO FURTHER TUNING (parameters frozen)

**Step 2: Full Backtest on OOS Period**
- Run each config on 2024-10-01 to 2024-12-31
- Collect same metrics as in-sample

**Step 3: Degradation Analysis**
```python
def validate_oos_degradation(is_metrics, oos_metrics):
    """
    Check if OOS performance is within acceptable degradation bounds.

    Returns:
        pass: bool
        report: dict
    """

    pf_degradation = (is_metrics['pf'] - oos_metrics['pf']) / is_metrics['pf']
    sharpe_degradation = (is_metrics['sharpe'] - oos_metrics['sharpe']) / is_metrics['sharpe']

    # Acceptance Criteria
    pass_pf = (pf_degradation < 0.30)  # <30% PF drop
    pass_sharpe = (sharpe_degradation < 0.40)  # <40% Sharpe drop
    pass_absolute = (oos_metrics['pf'] >= 1.5)  # Minimum viable PF
    pass_dd = (oos_metrics['dd'] <= 0.20)  # Max DD threshold

    passed = all([pass_pf, pass_sharpe, pass_absolute, pass_dd])

    return passed, {
        'pf_degradation': pf_degradation,
        'sharpe_degradation': sharpe_degradation,
        'is_pf': is_metrics['pf'],
        'oos_pf': oos_metrics['pf'],
        'oos_dd': oos_metrics['dd']
    }
```

**Step 4: Final Selection**
- Rank surviving configs by OOS PF
- Select top 1 for production deployment
- Keep top 3 in shadow mode for monitoring

---

### C. Overfitting Detection

**Method 1: Train/Test Consistency**
```python
def detect_overfit_train_test_consistency(is_pf, oos_pf):
    """
    Flag configurations with suspicious IS/OOS divergence.
    """
    if is_pf > 3.0 and oos_pf < 1.5:
        return "OVERFIT_SUSPECTED"  # Huge IS success but OOS failure

    degradation = (is_pf - oos_pf) / is_pf
    if degradation > 0.50:
        return "OVERFIT_LIKELY"  # >50% performance drop

    return "PASS"
```

**Method 2: Stability Across Regimes**
```python
def detect_overfit_regime_stability(results_by_regime):
    """
    Overfit configs often fail in unexpected regimes.
    """
    pf_variance = np.var([r['pf'] for r in results_by_regime.values()])

    if pf_variance > 2.0:
        return "UNSTABLE"  # High variance across regimes = fragile

    return "STABLE"
```

**Method 3: Parameter Sensitivity Analysis**
```python
def detect_overfit_parameter_sensitivity(best_params, study):
    """
    Overfit configs are often at extreme parameter boundaries.
    """
    boundary_params = 0
    for param_name, param_value in best_params.items():
        distribution = study.distributions[param_name]
        if hasattr(distribution, 'low') and hasattr(distribution, 'high'):
            low, high = distribution.low, distribution.high
            if param_value <= low or param_value >= high:
                boundary_params += 1

    if boundary_params >= 3:
        return "OVERFIT_SUSPECTED"  # 3+ params at boundaries = pathological

    return "PASS"
```

---

## Production Deployment Criteria

### A. Minimum Viable Thresholds (Hard Requirements)

**ALL archetypes must meet:**
1. **Out-of-Sample PF ≥ 1.5**
2. **Out-of-Sample Max DD ≤ 20%**
3. **In-Sample to OOS PF degradation < 30%**
4. **Minimum 20 trades in OOS period** (statistical significance)
5. **No catastrophic losing streaks** (max 5 consecutive losses in OOS)

**Bull archetypes (A-M) must additionally meet:**
6. **Risk-on regime PF ≥ 2.0** (bull patterns must excel in bull regimes)
7. **Crisis regime PF < 1.0 OR trade_count < 5** (expected failure, should be muted)

**Bear archetypes (S1, S4, S5) must additionally meet:**
8. **Risk-off regime PF ≥ 1.8** (bear patterns must excel in bear regimes)
9. **Risk-on regime PF < 1.2 OR trade_count < 5** (expected underperformance)

---

### B. Config Unification Process

**Step 1: Individual Archetype Configs**
- Each archetype optimization produces a JSON config:
  ```json
  {
    "archetype": "trap_within_trend",
    "fusion_threshold": 0.47,
    "min_liquidity": 0.15,
    "adx_min": 28,
    "wick_against_trend": 0.38,
    "archetype_weight": 3.2,
    "regime_routing": {
      "risk_on": 1.2,
      "neutral": 1.0,
      "risk_off": 0.3,
      "crisis": 0.0
    }
  }
  ```

**Step 2: Merge into Unified Config**
```python
def merge_archetype_configs(archetype_configs: List[Dict]) -> Dict:
    """
    Merge individual archetype configs into unified production config.
    """

    unified = {
        "version": "2.0.0-unified-optimized",
        "profile": "production_optimized",
        "description": "Unified config from archetype-specific optimization",
        "archetypes": {
            "use_archetypes": True,
            "thresholds": {}
        }
    }

    for arch_cfg in archetype_configs:
        arch_name = arch_cfg['archetype']

        # Add archetype-specific thresholds
        unified['archetypes']['thresholds'][arch_name] = {
            'fusion_threshold': arch_cfg['fusion_threshold'],
            'min_liquidity': arch_cfg.get('min_liquidity', 0.20),
            'archetype_weight': arch_cfg['archetype_weight'],
            # ... pattern-specific filters
        }

        # Add enable flag
        arch_code = ARCHETYPE_NAME_TO_CODE[arch_name]
        unified['archetypes'][f'enable_{arch_code}'] = True

    # Add unified regime routing
    unified['archetypes']['routing'] = merge_regime_routing(archetype_configs)

    return unified
```

**Step 3: Regime Routing Unification**
```python
def merge_regime_routing(archetype_configs: List[Dict]) -> Dict:
    """
    Create unified regime routing that applies archetype-specific weights.
    """

    routing = {
        'risk_on': {'weights': {}, 'final_gate_delta': 0.0},
        'neutral': {'weights': {}, 'final_gate_delta': 0.0},
        'risk_off': {'weights': {}, 'final_gate_delta': 0.02},
        'crisis': {'weights': {}, 'final_gate_delta': 0.04}
    }

    for arch_cfg in archetype_configs:
        arch_name = arch_cfg['archetype']
        regime_weights = arch_cfg['regime_routing']

        for regime in ['risk_on', 'neutral', 'risk_off', 'crisis']:
            routing[regime]['weights'][arch_name] = regime_weights[regime]

    return routing
```

---

### C. Conflict Resolution

**Scenario 1: Overlapping Signals**
```python
def resolve_signal_conflict(signals: List[Dict]) -> Dict:
    """
    When multiple archetypes trigger simultaneously, select highest quality.

    Priority:
    1. Highest fusion_score × archetype_weight
    2. Best regime alignment
    3. Tie-breaker: Archetype priority (A > B > C > ...)
    """

    # Score each signal
    for signal in signals:
        regime_mult = get_regime_multiplier(signal['archetype'], current_regime)
        signal['final_score'] = (
            signal['fusion_score'] *
            signal['archetype_weight'] *
            regime_mult
        )

    # Select highest scoring signal
    best_signal = max(signals, key=lambda s: s['final_score'])

    return best_signal
```

**Scenario 2: Bull vs Bear Signal Conflict**
```python
def resolve_directional_conflict(long_signal, short_signal):
    """
    When bull and bear archetypes trigger simultaneously, use regime as tiebreaker.
    """

    if current_regime in ['risk_on', 'neutral']:
        return long_signal  # Favor bull patterns
    elif current_regime in ['risk_off', 'crisis']:
        return short_signal  # Favor bear patterns
    else:
        # Neutral: Select highest fusion score
        return max([long_signal, short_signal], key=lambda s: s['fusion_score'])
```

---

### D. Production Rollout Schedule

**Week 1-2: Shadow Mode**
- Deploy unified config with all archetypes in shadow mode
- Log all signals, no live trades
- Monitor for:
  - Signal frequency (trades/day)
  - Fusion score distribution
  - Regime routing correctness
  - Conflict resolution behavior

**Week 3-4: Phased Activation (Group 1: High Confidence)**
- Enable Tier 1 archetypes: A/G/K, B/H/L, C, S5
- Trade with 50% position sizing
- Monitor for 2 weeks:
  - Live PF vs backtest expectations
  - Drawdown progression
  - Execution quality (slippage, fills)

**Week 5-6: Full Activation (Group 2: Medium Confidence)**
- Enable Tier 2 archetypes: D, F, M
- Increase position sizing to 100%
- Monitor for regime-specific performance

**Week 7+: Continuous Monitoring**
- Weekly performance reports
- Monthly regime alignment checks
- Quarterly re-optimization (if PF degrades >15%)

---

## Implementation Roadmap

### Phase 1: Infrastructure (Week 1)

**Tasks:**
1. Create `bin/optimize_archetype_v2.py` (unified optimization script)
2. Implement multi-fidelity validation logic
3. Add Hyperband pruning to Optuna studies
4. Create objective_score() function with regime adjustments
5. Build config merging utilities

**Deliverables:**
- Working optimization pipeline for single archetype
- Validated on A (Trap Within Trend) as proof-of-concept

---

### Phase 2: Critical Archetypes (Week 2-3)

**Optimization Order:**
1. **A/G/K (Trap Within Trend):** 250 trials (Day 1-2)
2. **B/H/L (Order Block Retest):** 200 trials (Day 3-4)
3. **C (BOS/CHoCH):** 160 trials (Day 5-6)
4. **S5 (Long Squeeze):** 160 trials (Day 7-8)

**Validation:**
- Run OOS validation for each completed optimization
- Document parameter ranges and regime routing

---

### Phase 3: Supporting Archetypes (Week 4)

**Optimization Order:**
5. **F (Expansion Exhaustion):** 120 trials
6. **M (Re-Accumulate):** 120 trials
7. **D (Failed Continuation):** 90 trials

**Validation:**
- Shadow mode deployment for Tier 2 archetypes
- Performance monitoring vs Tier 1

---

### Phase 4: Research Archetypes (Week 5)

**Optimization Order:**
8. **S4 (Distribution Climax):** 90 trials (AFTER Phase 2 feature backfill)
9. **S1 (Breakdown):** 90 trials (BLOCKED - feature dependency)

**Note:** S1 requires `liquidity_score` backfill before optimization can proceed.

---

### Phase 5: Config Unification (Week 6)

**Tasks:**
1. Merge all archetype configs into unified production config
2. Resolve routing conflicts
3. Add global regime override rules
4. Validate merged config on full 2024 data
5. Run 100-trial Monte Carlo robustness test

**Acceptance:**
- Unified config produces ≥95% of sum(individual archetype PFs)
- No catastrophic conflicts (opposite direction signals)
- Regime routing behaves as expected

---

### Phase 6: Production Deployment (Week 7-10)

**Rollout:**
- Week 7-8: Shadow mode (no trades)
- Week 9: Tier 1 activation (50% sizing)
- Week 10: Full activation (100% sizing)

**Monitoring:**
- Daily: Signal count, fusion distribution, regime alignment
- Weekly: PF, DD, Sharpe vs backtest expectations
- Monthly: Full performance report + regime breakdown

---

## Appendices

### Appendix A: Feature Dependencies by Archetype

| Feature | A/G/K | B/H/L | C | D | E | F | M | S1 | S2 | S4 | S5 |
|---------|-------|-------|---|---|---|---|---|----|----|----|----|
| **fusion_score** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **liquidity_score** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | **MISSING** | ✓ | ✓ | ✓ |
| **tf1h_ob_high** | - | ✓ | - | - | - | - | - | - | **MISSING** | - | - |
| **funding_Z** | - | - | - | - | - | - | - | - | - | - | ✓ |
| **oi_change_24h** | - | - | - | - | - | - | - | - | - | - | ✓ |
| **adx_14** | ✓ | - | - | - | - | ✓ | - | - | - | - | - |
| **rsi_14** | - | - | ✓ | ✓ | - | ✓ | - | - | ✓ | ✓ | ✓ |
| **volume_zscore** | - | - | - | - | ✓ | ✓ | - | ✓ | ✓ | ✓ | - |
| **atr_20** | ✓ | - | ✓ | - | ✓ | ✓ | ✓ | - | ✓ | - | - |

**Blockers:**
- **S1 (Breakdown):** Requires `liquidity_score` backfill (see `bin/backfill_liquidity_score.py`)
- **S2 (Failed Rally):** Requires `tf1h_ob_high` backfill (see `bin/backfill_ob_high.py`)
  - **STATUS:** PERMANENTLY DISABLED (PF 0.48 after optimization)

---

### Appendix B: Historical Performance Baseline (2024)

**Legacy System (Global Thresholds):**
- **Trades:** 92 (excessive)
- **PF:** 1.48 (below target)
- **Dominant Archetype:** Trap Within Trend (81%)

**Target After Optimization:**
- **Trades:** 80-120 (controlled)
- **PF:** 2.5+ (exceeds target)
- **Archetype Diversity:** No single archetype >40% of trades

---

### Appendix C: Regime Classification Metadata

**2024 Regime Breakdown (9 months in-sample):**
- Risk-on: 156 days (58%)
- Neutral: 73 days (27%)
- Risk-off: 31 days (11%)
- Crisis: 10 days (4%)

**2022 Regime Breakdown (full year):**
- Risk-on: 8 days (2%)
- Neutral: 99 days (27%)
- Risk-off: 201 days (55%)
- Crisis: 57 days (16%)

**Implication:** Bull archetypes validated on 2024 (58% risk-on) must be stress-tested on 2022 (16% crisis) to ensure proper muting behavior.

---

### Appendix D: S2 Failure Post-Mortem

**Optimization History:**
- **Attempt 1 (Oct 2025):** 100 trials, best PF 0.62
- **Attempt 2 (Nov 2025):** 157 trials (grid search), best PF 0.48
- **Attempt 3 (Fast optimizer):** 50 trials, best PF 0.51

**Root Cause Analysis:**
1. **Pattern hypothesis was flawed:** "Rejection wick + weak volume" does NOT reliably predict downside
2. **2022 validation was misleading:** Forward returns of -0.68% (24h) were within noise (p=0.18)
3. **OB retest dependency:** Pattern required `tf1h_ob_high` feature, which was often missing/stale

**Lessons Learned:**
1. **Validate pattern hypothesis BEFORE optimization:** Forward return analysis must show statistical significance (p < 0.05)
2. **Feature availability matters:** Patterns dependent on unreliable features should be rejected early
3. **Trial budget limits:** 157 trials is sufficient to determine if edge exists. Further optimization is futile.

**Recommendation:** Mark S2 as PERMANENTLY DISABLED in all configs. Remove from future optimization pipelines.

---

**END OF REQUIREMENTS SPECIFICATION**

---

**Approval Signatures:**

- **Requirements Author:** Claude Code (Sonnet 4.5)
- **Technical Review:** [PENDING ARCHITECT APPROVAL]
- **Implementation Lead:** [PENDING ASSIGNMENT]
- **Expected Start Date:** 2025-11-18
- **Expected Completion:** 2025-12-16 (4 weeks)

**Change Log:**
- v1.0 (2025-11-17): Initial requirements specification created
