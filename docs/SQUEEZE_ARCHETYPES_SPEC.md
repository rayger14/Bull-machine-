# Squeeze Archetypes Requirements Specification

**Phase**: Phase 3 - Archetypes, Regimes & Fusion Cleanup
**Status**: Requirements Definition
**Target**: MVP Production-Ready Squeeze Patterns

---

## Executive Summary

Define production requirements for squeeze-based archetypes (S1, S3, S4) with clear acceptance criteria. S2 is LOCKED OFF due to proven negative edge. All squeeze archetypes must demonstrate PF ≥ 1.2 and WR ≥ 35% on isolated backtests across 2022-2024 or face deprecation.

---

## 1. S2 Status: LOCKED OFF

### 1.1 Decision Record

**Archetype**: S2 - Failed Rally (Rejection)
**Status**: PERMANENTLY DISABLED
**Rationale**: Fundamental pattern failure after exhaustive optimization
**Evidence**: PF 0.48 after optimization (target: PF ≥ 1.2)

### 1.2 Implementation

**Config Setting**:
```json
{
  "archetypes": {
    "enable_S2": false,
    "_comment_S2": "Failed Rally DISABLED - PF 0.48 after optimization, pattern fundamentally broken. See docs/decisions/S2_DISABLE_DECISION.md for analysis."
  }
}
```

**Location**: All production configs (`configs/mvp/*.json`, `configs/regime/*.json`)

**Documentation**:
- Existing: `docs/decisions/S2_DISABLE_DECISION.md`
- Archive pattern definition in: `docs/archive/2024-q4/S2_PATTERN_AUTOPSY.md`

**Code Actions**:
- Keep detection function `_check_S2()` in `engine/archetypes/logic.py` (line 1026) for historical reference
- Add warning log if accidentally enabled: `logger.warning("S2 is deprecated - see S2_DISABLE_DECISION.md")`
- Remove from archetype registry? **NO** - Keep for backward compatibility with old configs

### 1.3 Lessons Learned

**Why S2 Failed**:
1. Resistance rejection signals are inherently noisy (false positives)
2. RSI divergence detection unreliable on 1H timeframe
3. Volume fade metric conflicts with institutional absorption patterns
4. Regime tuning couldn't rescue fundamental pattern weakness

**Implications for S1/S3/S4**:
- Require volume CONFIRMATION, not volume divergence
- Use multi-timeframe validation (1H + 4H alignment)
- Demand clear structural breaks (BOS, not just wicks)
- Test across all three regime periods (2022/2023/2024)

---

## 2. S1 Requirements: Liquidity Sweep Reversal

### 2.1 Pattern Definition

**Name**: S1 - Liquidity Sweep Reversal
**Concept**: Hunt stops below support → immediate reversal (smart money trap)
**Direction**: Long (counter-trend reversal)
**Regimes**: Bull, Neutral (NOT bear - too risky in sustained downtrends)

### 2.2 Required Features

**Critical Dependencies** (MUST exist in feature store):

| Feature | Source | Validation | Fallback |
|---------|--------|------------|----------|
| `liquidity_score` | Phase 1 backfill | Check `bin/backfill_liquidity_score.py` completed | BLOCK deployment |
| `liquidity_void_detected` | SMC engine | `engine/smc/smc_engine.py` | Compute on-the-fly |
| `tf1h_pti_score` | Phase 2 PTI | `engine/pti/pti_engine.py` | BLOCK deployment |
| `tf1h_fakeout_intensity` | Existing | Feature store | Fallback: 0.0 |
| `tf1d_wyckoff_phase` | Existing | Wyckoff engine | Fallback: 'transition' |

**Optional Enhancers** (boost fusion if present):
- `liquidity_sweep_strength` - If missing, derive from `liquidity_void_detected` + volume spike
- `tf4h_bos_bearish` - Multi-timeframe downtrend confirmation
- Fibonacci cluster proximity (from Phase 2 temporal features)

### 2.3 Hard Conditions (ALL must pass)

```python
def _check_S1(row, prev_row, df, index, fusion_score) -> bool:
    """
    S1 - Liquidity Sweep Reversal

    Entry criteria:
    1. Liquidity sweep occurred (price touched void/sweep level)
    2. Reversal pattern (wick rejection, volume confirmation)
    3. PTI trap signal (pti_score > 0.5)
    4. Fusion score >= min_threshold
    5. NOT in bear regime (veto in risk_off/crisis)
    """

    # 1. Liquidity sweep detection
    liquidity_void = row.get('liquidity_void_detected', False)
    if not liquidity_void:
        # Fallback: Check if price touched low liquidity area
        liquidity_score = row.get('liquidity_score', 0.5)
        if liquidity_score >= 0.20:  # NOT in void
            return False

    # 2. Reversal pattern (wick rejection + volume)
    close = row.get('close', 0.0)
    open_price = row.get('open', close)
    high = row.get('high', close)
    low = row.get('low', close)

    body = abs(close - open_price)
    lower_wick = min(close, open_price) - low

    # Require significant lower wick (hunt below support)
    if lower_wick < 2.0 * body:
        return False

    # Volume confirmation (spike on reversal bar)
    volume_z = row.get('volume_zscore', 0.0)
    if volume_z < 1.0:
        return False

    # 3. PTI trap signal
    pti_score = row.get('tf1h_pti_score', 0.0)
    if pti_score < 0.5:
        return False

    # 4. Fusion threshold
    if fusion_score < thresh_S1.get('fusion', 0.38):
        return False

    return True
```

### 2.4 Soft Conditions (Fusion Boosters)

Apply AFTER hard conditions pass, BEFORE final fusion threshold:

```python
# Wyckoff context boost (+0.05 if Spring or LPS)
wyckoff_phase = row.get('tf1d_wyckoff_phase', 'transition')
if wyckoff_phase in ['spring', 'accumulation', 'phase_c', 'phase_d']:
    fusion_score += 0.05

# Order block support (+0.05 if near OB)
near_ob = check_ob_proximity(row, df, atr_threshold=1.0)
if near_ob:
    fusion_score += 0.05

# Fib time cluster (+0.03 if within cluster)
fib_cluster = row.get('tf1h_fib_cluster_proximity', 0.0)
if fib_cluster > 0.7:
    fusion_score += 0.03
```

### 2.5 Regime Routing

**Regime-Specific Adjustments**:

| Regime | Weight Multiplier | Fusion Floor | Notes |
|--------|------------------|--------------|-------|
| Risk On | 1.2x | 0.35 | Favorable - trend reversals work |
| Neutral | 1.0x | 0.38 | Standard - balanced conditions |
| Risk Off | 0.6x | 0.45 | Risky - require higher conviction |
| Crisis | VETO | N/A | Hard block - no counter-trend in panic |

### 2.6 Target Metrics

**Acceptance Criteria** (isolated backtest, 2022-2024):
- **PF ≥ 1.2** (minimum viable edge)
- **Win Rate ≥ 45%** (reversal patterns need higher WR)
- **Trade Count ≥ 10** (per year, sufficient sample)
- **Avg R-multiple ≥ 1.5** (risk/reward validation)

**If ANY metric fails** → Deprecate S1 and document why

---

## 3. S3 Requirements: Failed Rally (Whipsaw)

### 3.1 Decision Point: Keep or Deprecate?

**Current Status**: UNDEFINED - No historical validation
**Critical Question**: Does S3 provide unique edge distinct from S2?

**Decision Matrix**:

| Scenario | Action | Rationale |
|----------|--------|-----------|
| PF ≥ 1.2, unique triggers | KEEP | Viable pattern |
| PF < 1.2 OR overlaps S2 | DEPRECATE | Redundant/broken |
| Insufficient sample (< 10 trades/year) | DEPRECATE | Unstable |

### 3.2 Pattern Definition (IF KEEPING)

**Name**: S3 - Whipsaw Rejection
**Concept**: False breakout above resistance → immediate reversal
**Direction**: Short (fade the breakout)
**Regimes**: Neutral, Early Bear (NOT bull - counter-productive)

**Distinction from S2**:
- S2: Resistance TEST with divergence (gradual rejection)
- S3: Resistance BREAK with whipsaw (violent reversal)

### 3.3 Required Features (IF KEEPING)

**Critical Dependencies**:

| Feature | Requirement | Validation |
|---------|-------------|------------|
| Wick anomaly | Upper wick > 2.5x body | Compute from OHLC |
| Volume fade | Volume < 0.5x average | Use `volume_zscore` |
| MTF downtrend | `tf4h_bos_bearish` == True | SMC feature |
| Fusion score | >= 0.35 | Standard threshold |

**Trigger Conditions**:
```python
def _check_S3(row, prev_row, df, index, fusion_score) -> bool:
    """
    S3 - Whipsaw Rejection

    Entry criteria:
    1. Upper wick > 2.5x body (false break above resistance)
    2. Volume fade (no conviction on breakout)
    3. MTF downtrend confirmation (4H bearish structure)
    4. Fusion score >= 0.35
    """

    # 1. Wick anomaly calculation
    close = row.get('close', 0.0)
    open_price = row.get('open', close)
    high = row.get('high', close)

    body = abs(close - open_price)
    upper_wick = high - max(close, open_price)

    if body == 0 or upper_wick < 2.5 * body:
        return False

    # 2. Volume fade check
    vol_z = row.get('volume_zscore', 0.0)
    if vol_z >= 0.5:
        return False

    # 3. MTF downtrend
    tf4h_bos_bearish = row.get('tf4h_bos_bearish', False)
    if not tf4h_bos_bearish:
        # Fallback: Check 4H fusion trend
        tf4h_fusion = row.get('tf4h_fusion_score', 0.5)
        if tf4h_fusion >= 0.5:  # Bullish/neutral
            return False

    # 4. Fusion threshold
    if fusion_score < 0.35:
        return False

    return True
```

### 3.4 Target Metrics (IF KEEPING)

Same as S1:
- PF ≥ 1.2
- WR ≥ 40% (short patterns have lower WR tolerance)
- Trade Count ≥ 10/year
- Avg R-multiple ≥ 1.5

**If fails any metric** → DEPRECATE immediately

### 3.5 Deprecation Path (IF REMOVING)

**Actions**:
```json
{
  "enable_S3": false,
  "_comment_S3": "Whipsaw Rejection DEPRECATED - [reason: overlaps S2 / insufficient edge / low sample]. See docs/decisions/S3_DEPRECATION.md"
}
```

**Documentation**:
- Create `docs/decisions/S3_DEPRECATION.md`
- Archive pattern definition
- Remove from future optimization runs

---

## 4. S4 Requirements: Momentum Reversal (Distribution)

### 4.1 Pattern Definition

**Name**: S4 - Momentum Reversal
**Concept**: Exhaustion climax → distribution begins
**Direction**: Short (fade the blow-off)
**Regimes**: Bull, Neutral (catches tops, NOT for bear grinding)

### 4.1 Required Features

**Critical Dependencies**:

| Feature | Source | Validation | Fallback |
|---------|--------|------------|----------|
| `momentum_exhaustion_score` | Derive from RSI/MACD | Compute if missing | REQUIRED |
| `liquidity_score` | Phase 1 backfill | Check populated | BLOCK |
| `tf1h_pti_score` | Phase 2 PTI | Feature store | BLOCK |
| `tf1d_wyckoff_phase` | Wyckoff engine | Existing | Fallback: 'transition' |

**Momentum Exhaustion Calculation** (if missing):
```python
def calculate_momentum_exhaustion(row, prev_row) -> float:
    """
    Compute momentum exhaustion score [0-1]

    Factors:
    - RSI > 70 (overbought)
    - MACD histogram declining
    - Volume spike with price stall
    """
    score = 0.0

    # RSI component (0-0.4 range)
    rsi = row.get('rsi_14', 50.0)
    if rsi > 70:
        score += (rsi - 70) / 30.0 * 0.4  # 70-100 → 0-0.4

    # MACD divergence (0-0.3 range)
    if prev_row is not None:
        macd_curr = row.get('macd_histogram', 0.0)
        macd_prev = prev_row.get('macd_histogram', 0.0)
        if macd_curr < macd_prev:  # Declining
            score += 0.3

    # Volume spike + price stall (0-0.3 range)
    vol_z = row.get('volume_zscore', 0.0)
    close_curr = row.get('close', 0.0)
    close_prev = prev_row.get('close', close_curr) if prev_row else close_curr

    if vol_z > 1.5 and abs(close_curr - close_prev) < 0.01 * close_curr:
        score += 0.3  # High volume but no progress

    return min(score, 1.0)
```

### 4.3 Hard Conditions

```python
def _check_S4(row, prev_row, df, index, fusion_score) -> bool:
    """
    S4 - Momentum Reversal (Distribution)

    Entry criteria:
    1. Momentum exhausted (score > 0.6)
    2. Liquidity drying up (score < 0.3)
    3. PTI elevated (trap forming, score > 0.6)
    4. Fusion score >= 0.37
    """

    # 1. Momentum exhaustion
    momentum_exhaustion = row.get('momentum_exhaustion_score')
    if momentum_exhaustion is None:
        momentum_exhaustion = calculate_momentum_exhaustion(row, prev_row)

    if momentum_exhaustion < 0.6:
        return False

    # 2. Liquidity drop-off
    liquidity_score = row.get('liquidity_score', 0.5)
    if liquidity_score >= 0.3:
        return False

    # 3. PTI trap signal
    pti_score = row.get('tf1h_pti_score', 0.0)
    if pti_score < 0.6:
        return False

    # 4. Fusion threshold
    if fusion_score < 0.37:
        return False

    return True
```

### 4.4 Soft Conditions (Fusion Boosters)

```python
# Wyckoff distribution phase boost (+0.05)
wyckoff_phase = row.get('tf1d_wyckoff_phase', 'transition')
if wyckoff_phase in ['distribution', 'phase_b', 'phase_c', 'upthrust']:
    fusion_score += 0.05

# RSI extreme boost (+0.03 if RSI > 75)
rsi = row.get('rsi_14', 50.0)
if rsi > 75:
    fusion_score += 0.03

# Multi-timeframe alignment (+0.05 if 4H also exhausted)
tf4h_rsi = row.get('tf4h_rsi_14', 50.0)
if tf4h_rsi > 70:
    fusion_score += 0.05
```

### 4.5 Regime Routing

| Regime | Weight Multiplier | Fusion Floor | Notes |
|--------|------------------|--------------|-------|
| Risk On | 1.3x | 0.35 | Best - catches bull market tops |
| Neutral | 1.0x | 0.37 | Standard |
| Risk Off | 0.7x | 0.42 | Less effective - already bearish |
| Crisis | 0.5x | 0.50 | Risky - volatility skews signals |

### 4.6 Target Metrics

**Acceptance Criteria**:
- PF ≥ 1.2
- WR ≥ 40%
- Trade Count ≥ 10/year
- Avg R-multiple ≥ 1.5

---

## 5. Testing & Validation Plan

### 5.1 Isolated Backtest Protocol

**Objective**: Test each archetype in isolation to measure true edge

**Configuration**:
```json
{
  "archetypes": {
    "use_archetypes": true,
    "enable_S1": true,  // Test S1 only
    "enable_S2": false,
    "enable_S3": false,
    "enable_S4": false,
    // Disable all other archetypes (A-M)
    "enable_A": false,
    "enable_B": false,
    // ... etc
  }
}
```

**Test Periods**:
1. **2022** (bear market): Validate S1/S3/S4 don't over-trade in sustained downtrends
2. **2023** (neutral/recovery): Validate balanced performance
3. **2024** (bull market): Validate S4 catches tops, S1 catches dips

**Metrics Collection**:
```bash
python bin/backtest_knowledge_v2.py \
  --config configs/test/s1_isolated.json \
  --start 2022-01-01 \
  --end 2024-12-31 \
  --output results/squeeze/s1_isolated.json
```

### 5.2 Accept/Reject Criteria

**Per-Archetype Decision Tree**:

```
For each S1, S3, S4:
├─ PF >= 1.2?
│  ├─ YES → Continue
│  └─ NO → REJECT (negative/neutral edge)
│
├─ WR >= 35% (40% for shorts)?
│  ├─ YES → Continue
│  └─ NO → REJECT (unreliable signals)
│
├─ Trades >= 10/year?
│  ├─ YES → Continue
│  └─ NO → REJECT (insufficient sample)
│
├─ Avg R-multiple >= 1.5?
│  ├─ YES → ACCEPT ✅
│  └─ NO → REJECT (poor risk/reward)
```

**Rejection Actions**:
1. Set `enable_SX: false` in all production configs
2. Create `docs/decisions/SX_REJECTION.md` with autopsy
3. Archive pattern definition in `docs/archive/`
4. Remove from future optimization runs

### 5.3 Feature Dependency Validation

**Pre-Deployment Checklist**:

```bash
# 1. Validate liquidity_score populated
python -c "
import pandas as pd
df = pd.read_parquet('data/BTC_1H_features.parquet')
assert 'liquidity_score' in df.columns
assert df['liquidity_score'].notna().sum() > 0.95 * len(df)
print('✅ liquidity_score validated')
"

# 2. Validate PTI scores
python -c "
import pandas as pd
df = pd.read_parquet('data/BTC_1H_features.parquet')
assert 'tf1h_pti_score' in df.columns
assert df['tf1h_pti_score'].notna().sum() > 0.90 * len(df)
print('✅ pti_score validated')
"

# 3. Validate Wyckoff phases
python -c "
import pandas as pd
df = pd.read_parquet('data/BTC_1D_features.parquet')
assert 'tf1d_wyckoff_phase' in df.columns
print('✅ wyckoff_phase validated')
"
```

**If validation fails** → BLOCK squeeze archetype deployment

---

## 6. Feature Dependency Matrix

**Summary Table**:

| Archetype | Critical Features (MUST have) | Optional Features (boost) | Blockers |
|-----------|------------------------------|---------------------------|----------|
| **S1** | liquidity_score, liquidity_void_detected, tf1h_pti_score | liquidity_sweep_strength, tf1d_wyckoff_phase, fib_cluster | Phase 1 backfill incomplete |
| **S2** | N/A - DISABLED | N/A | Permanently locked off |
| **S3** | wick_anomaly (OHLC), volume_zscore, tf4h_bos_bearish | N/A | Decision pending: keep or deprecate |
| **S4** | momentum_exhaustion_score, liquidity_score, tf1h_pti_score | tf1d_wyckoff_phase, tf4h_rsi_14 | Phase 1/2 incomplete |

---

## 7. Implementation Checklist

### 7.1 Code Changes

- [ ] Update `engine/archetypes/logic.py`:
  - [ ] Enhance `_check_S1()` with new conditions (lines 981-1024)
  - [ ] Add `calculate_momentum_exhaustion()` helper
  - [ ] Update `_check_S4()` to use momentum_exhaustion (lines 1119-1153)
  - [ ] Add S3 deprecation warning if enabled

- [ ] Create `engine/archetypes/squeeze_helpers.py`:
  - [ ] `calculate_momentum_exhaustion(row, prev_row) -> float`
  - [ ] `calculate_liquidity_sweep_strength(row, df, lookback=10) -> float`
  - [ ] `check_ob_proximity(row, df, atr_threshold=1.0) -> bool`

- [ ] Update `engine/archetypes/threshold_policy.py`:
  - [ ] Add regime-specific thresholds for S1, S3, S4
  - [ ] Implement fusion boosters logic

### 7.2 Configuration Files

- [ ] Update all production configs:
  - [ ] `configs/mvp/mvp_bull_market_v1.json`
  - [ ] `configs/mvp/mvp_bear_market_v1.json`
  - [ ] `configs/mvp/mvp_regime_routed_production.json`

- [ ] Create test configs:
  - [ ] `configs/test/s1_isolated.json`
  - [ ] `configs/test/s3_isolated.json` (if keeping)
  - [ ] `configs/test/s4_isolated.json`

### 7.3 Documentation

- [ ] Create decision docs:
  - [ ] `docs/decisions/S3_DECISION.md` (keep or deprecate)
  - [ ] Update `docs/decisions/S2_DISABLE_DECISION.md` (add to all configs)

- [ ] Create validation reports:
  - [ ] `results/squeeze/s1_validation_report.md`
  - [ ] `results/squeeze/s3_validation_report.md` (if keeping)
  - [ ] `results/squeeze/s4_validation_report.md`

### 7.4 Testing

- [ ] Run isolated backtests (S1, S3, S4)
- [ ] Validate feature dependencies
- [ ] Accept/reject each archetype based on metrics
- [ ] Document results in `results/squeeze/`

---

## 8. Decision Log

| Date | Archetype | Decision | Rationale | Responsible |
|------|-----------|----------|-----------|-------------|
| 2024-11-XX | S2 | LOCKED OFF | PF 0.48, fundamentally broken | Requirements Analyst |
| 2024-11-XX | S1 | TBD | Pending validation backtest | Phase 3 Team |
| 2024-11-XX | S3 | TBD | Decision: keep or deprecate | Phase 3 Team |
| 2024-11-XX | S4 | TBD | Pending validation backtest | Phase 3 Team |

---

## Appendix A: S2 Autopsy (Reference)

**Pattern**: Failed Rally (Resistance Rejection)
**Hypothesis**: Fade resistance tests with RSI divergence
**Result**: PF 0.48 (target: 1.2) after optimization

**Root Causes**:
1. RSI divergence too noisy on 1H timeframe
2. Volume fade conflicts with institutional absorption
3. Resistance "test" vs "break" ambiguous without HTF context
4. Regime tuning couldn't overcome signal weakness

**Recommendation**: PERMANENT DISABLE - do not resurrect

---

## Appendix B: Regime Routing Weights

**Base Archetype Weights** (before regime adjustment):

| Archetype | Risk On | Neutral | Risk Off | Crisis |
|-----------|---------|---------|----------|--------|
| S1 (Sweep) | 1.2x | 1.0x | 0.6x | VETO |
| S3 (Whipsaw) | 0.8x | 1.0x | 1.1x | VETO |
| S4 (Momentum) | 1.3x | 1.0x | 0.7x | 0.5x |

**Fusion Thresholds** (regime-adjusted):

| Archetype | Risk On | Neutral | Risk Off | Crisis |
|-----------|---------|---------|----------|--------|
| S1 | 0.35 | 0.38 | 0.45 | N/A |
| S3 | 0.33 | 0.35 | 0.32 | N/A |
| S4 | 0.35 | 0.37 | 0.42 | 0.50 |

---

**End of Specification**
