# 3-Archetype Entry System - Implementation Plan

**Status**: Ready to implement
**Created**: 2025-10-22
**Goal**: Increase trade volume from 20 to 50-80 trades/year while maintaining 65% WR and 5.40 PF quality

---

## Problem Statement

Current system uses single fusion threshold (0.374) as one-size-fits-all door:
- **Result**: Only 20 trades in 2024 (should be 50-80)
- **Root cause**: Different market contexts (springs, retests, continuations) need different thresholds
- **Evidence**: Exit optimizer v2 produced 200 identical trials → entry is the bottleneck

---

## Solution: 3-Archetype Entry Doors

Split entries into context-specific archetypes based on Wyckoff/ZeroIka/Moneytaur methodologies:

### Archetype A: Trap Reversal (Bojan-style)
**Market Context**: Phase C spring / UTAD + trap-reset pattern
**Conditions**:
- PTI trap detected (tf1h_pti_trap_type is not None)
- Strong displacement (tf4h_boms_displacement ≥ 1.25× ATR)
- Price closes back into range (flip-close pattern)

**Gates**:
- `pti_score >= 0.65` (strong trap signal)
- FRVP position in discount/premium zone
- No additional gates required (PTI is primary)

**Threshold**: 0.32-0.34 (LOWER than default - traps are low-fusion but high-edge)
**Risk Sizing**: 0.75× normal size (reduced risk due to uncertainty)

---

### Archetype B: OB/pHOB Retest (ZeroIka refinement)
**Market Context**: BOS, return to unmitigated order block, HTF alignment
**Conditions**:
- BOS detected (tf1h_bos_bullish or tf1h_bos_bearish)
- Strong liquidity sweep (tf1d_boms_strength ≥ 0.68)
- Wyckoff bias agrees (wyckoff_score and direction align)

**Gates**:
- `liquidity_score >= 0.68` (from fusion calculation)
- `wyckoff_score >= 0.50` AND direction agrees
- FRVP positioning near value area (optional)

**Threshold**: 0.36-0.38 (slightly below default 0.374)
**Risk Sizing**: 1.0× normal size (standard risk)

---

### Archetype C: FVG/Breaker Continuation (Moneytaur)
**Market Context**: Strong displacement, ATR-gated FVG, fresh BOS
**Conditions**:
- FVG present (tf1h_fvg_present or tf4h_fvg_present)
- Strong displacement (tf4h_boms_displacement ≥ 1.5)
- Fresh BOS (tf1h_bos_bullish/bearish within last 3-5 bars)

**Gates**:
- `liquidity_score >= 0.72` (stricter than archetype B)
- `momentum_score >= 0.55` (from fusion calculation)
- **Plus-One gate**: `tf4h_fusion_score >= 0.62` (enables 1.25× sizing)

**Threshold**: 0.40-0.44 (HIGHER than default - high-conviction setups only)
**Risk Sizing**:
- Base: 1.0× normal size
- Plus-One: 1.25× when tf4h_fusion >= 0.62 AND SMT confirms

---

## Implementation Steps

### Step 1: Add Archetype Detection Method (NEW)

Add new method to `KnowledgeBacktestEngine` class (after `calculate_fusion_score`):

```python
def classify_entry_archetype(self, row: pd.Series, context: Dict) -> Optional[Tuple[str, float, float]]:
    """
    Classify entry opportunity into one of 3 archetypes.

    Returns:
        (archetype_name, threshold, size_multiplier) or None if no archetype matches
    """
    # Archetype A: Trap Reversal
    pti_trap = row.get('tf1h_pti_trap_type', None)
    pti_score = context.get('pti_score', 0.0)
    boms_disp = row.get('tf4h_boms_displacement', 0.0)
    atr = row.get('atr_14', row['close'] * 0.02)

    if pti_trap is not None and pti_score >= 0.65 and boms_disp >= (1.25 * atr):
        # Check for flip-close (price returned to range)
        frvp_pos = context.get('frvp_poc_position', 'middle')
        if frvp_pos in ['at_poc', 'middle']:
            return ("trap_reversal", 0.33, 0.75)  # Low threshold, reduced size

    # Archetype B: OB/pHOB Retest
    bos_bull = row.get('tf1h_bos_bullish', False)
    bos_bear = row.get('tf1h_bos_bearish', False)
    boms_strength = row.get('tf1d_boms_strength', 0.0)
    liq_score = context.get('liquidity_score', 0.0)
    wyc_score = context.get('wyckoff_score', 0.0)

    if (bos_bull or bos_bear) and boms_strength >= 0.68:
        if liq_score >= 0.68 and wyc_score >= 0.50:
            return ("ob_retest", 0.37, 1.0)  # Mid threshold, normal size

    # Archetype C: FVG/Breaker Continuation
    fvg_1h = row.get('tf1h_fvg_present', False)
    fvg_4h = row.get('tf4h_fvg_present', False)
    mom_score = context.get('momentum_score', 0.0)
    tf4h_fusion = row.get('tf4h_fusion_score', 0.0)

    if (fvg_1h or fvg_4h) and boms_disp >= (1.5 * atr):
        if liq_score >= 0.72 and mom_score >= 0.55:
            # Check for Plus-One sizing
            size_mult = 1.25 if tf4h_fusion >= 0.62 else 1.0
            return ("fvg_continuation", 0.42, size_mult)  # High threshold, variable size

    return None  # No archetype matched
```

### Step 2: Modify check_entry_conditions() Method

Replace current tiered logic (lines 349-393) with archetype-first approach:

```python
def check_entry_conditions(self, row: pd.Series, fusion_score: float, context: Dict) -> Optional[Tuple[str, float]]:
    """
    Check if entry conditions are met using 3-archetype classification.

    Returns:
        (entry_type, entry_price) or None
    """
    # PHASE 1: Try archetype classification first
    archetype_result = self.classify_entry_archetype(row, context)

    if archetype_result:
        archetype_name, threshold, size_mult = archetype_result

        # Check if fusion score meets archetype-specific threshold
        if fusion_score >= threshold:
            # Store archetype info for position sizing later
            context['entry_archetype'] = archetype_name
            context['archetype_size_mult'] = size_mult

            # Check macro filter (crisis veto applies to all archetypes)
            if context.get('macro_regime') == 'crisis':
                return None

            logger.info(f"ARCHETYPE ENTRY: {archetype_name} | fusion={fusion_score:.3f} >= {threshold:.3f} | size_mult={size_mult:.2f}x")
            return (f"archetype_{archetype_name}", row['close'])

    # PHASE 2: Fallback to legacy tiered system (safety net)
    # Only trigger if fusion score is exceptionally high (> 0.45)
    if fusion_score >= 0.45:
        # Ultra-high conviction fallback
        if context.get('macro_regime') not in ['crisis']:
            logger.info(f"LEGACY TIER1 ENTRY: fusion={fusion_score:.3f} (no archetype matched)")
            context['entry_archetype'] = 'legacy_tier1'
            context['archetype_size_mult'] = 1.0
            return ("tier1_market", row['close'])

    return None  # No entry
```

### Step 3: Modify calculate_position_size() to Use Archetype Sizing

Update position sizing method (lines 311-347) to use archetype multiplier:

```python
def calculate_position_size(self, row: pd.Series, fusion_score: float, context: Dict = None) -> float:
    """
    Calculate position size using ATR-based risk management + archetype sizing.
    """
    if self.params.position_size_method == "fixed":
        return self.equity * 0.95

    # ATR-based sizing
    atr = row.get('atr_14', row['close'] * 0.02)
    stop_distance = atr * self.params.atr_stop_mult
    risk_dollars = self.equity * self.params.max_risk_pct
    position_size = risk_dollars / (stop_distance / row['close'])

    # Volatility scaling
    if self.params.volatility_scaling:
        vix_level = row.get('macro_vix_level', 'medium')
        vix_scaling = {'low': 1.0, 'medium': 0.8, 'high': 0.5, 'extreme': 0.25}
        position_size *= vix_scaling.get(vix_level, 0.8)

    # Confidence scaling (baseline: 0.5 to 1.0)
    confidence_mult = 0.5 + (fusion_score * 0.5)
    position_size *= confidence_mult

    # ARCHETYPE SIZING MULTIPLIER (NEW)
    if context and 'archetype_size_mult' in context:
        arch_mult = context['archetype_size_mult']
        position_size *= arch_mult
        logger.info(f"SIZING: Archetype {context.get('entry_archetype')} × {arch_mult:.2f} → ${position_size:,.0f}")

    # Cap at 95% of equity
    position_size = min(position_size, self.equity * 0.95)

    return position_size
```

### Step 4: Update Phase 4 Re-Entry Parameters

Modify environment variable defaults (lines 181-184):

```python
# Phase 4: Re-Entry Parameters (ENABLE with sane gates)
self.reentry_confluence_threshold = int(os.getenv('EXIT_REENTRY_CONF', '2'))  # Changed from 3 to 2
self.reentry_window_btc_eth = int(os.getenv('EXIT_REENTRY_WINDOW', '7'))
self.reentry_fusion_delta = float(os.getenv('EXIT_REENTRY_DELTA', '0.045'))  # Changed from 0.05
self.reentry_4h_fusion_min = float(os.getenv('GATE5_4H_FUSION_MIN', '0.60'))  # NEW
self.reentry_vol_z_min = float(os.getenv('GATE5_VOL_Z_MIN', '0.35'))  # NEW
self.reentry_cooldown_bars = int(os.getenv('REENTRY_COOLDOWN', '4'))  # NEW
```

### Step 5: Loosen Exit Parameters (Assist Mode)

Update exit parameter defaults (lines 169-177):

```python
# Phase 2: Pattern Exit Parameters (LOOSENED from governor to assist)
self.pattern_confluence_threshold = int(os.getenv('EXIT_PATTERN_CONFLUENCE', '2'))  # Changed from 3

# Phase 2: Structure Exit Parameters (LOOSENED)
self.structure_min_hold_bars = int(os.getenv('EXIT_STRUCT_MIN_HOLD', '12'))  # Changed from 20
self.structure_rsi_long_threshold = int(os.getenv('EXIT_STRUCT_RSI_LONG', '25'))  # Keep same
self.structure_rsi_short_threshold = int(os.getenv('EXIT_STRUCT_RSI_SHORT', '75'))  # Keep same
self.structure_vol_zscore_min = float(os.getenv('EXIT_STRUCT_VOL_Z', '0.35'))  # Changed from 0.5
```

### Step 6: Match Baseline Risk Model

Update risk parameters in KnowledgeParams (lines 83-91):

```python
# Position sizing (MATCH BASELINE)
max_risk_pct: float = 0.01                # Changed to 1% (was 2%)
atr_stop_mult: float = 2.5                # Keep 2.5× ATR
position_size_method: str = "atr"         # Keep ATR-based
volatility_scaling: bool = True           # Keep VIX scaling
max_concurrent_positions: int = 2         # NEW: Max 2 positions
```

---

## Expected Results

After implementing archetype system:

| Metric | Current (Single Threshold) | Target (3-Archetype) |
|--------|---------------------------|----------------------|
| **Trades/Year** | 20 | 50-80 |
| **Win Rate** | 65% | 55-65% (may decrease slightly) |
| **Profit Factor** | 5.40 | 3.0-5.0 (more trades = more variance) |
| **Total PNL** | $584 | $2,000-$5,000 (volume increase) |
| **Max DD** | 0.0% (!) | 5-10% (realistic with more trades) |

---

## Testing Plan

1. **Implement changes** to bin/backtest_knowledge_v2.py
2. **Test on BTC 2024** full-year with new archetype system
3. **Compare**:
   - Old: 20 trades, 65% WR, $584 PNL
   - New: Expected 50-80 trades, 55-65% WR, $2K-$5K PNL
4. **Log archetype distribution**:
   - How many Trap Reversals? (Archetype A)
   - How many OB Retests? (Archetype B)
   - How many FVG Continuations? (Archetype C)

---

## Files to Modify

1. **bin/backtest_knowledge_v2.py**:
   - Add `classify_entry_archetype()` method (after line 309)
   - Replace `check_entry_conditions()` method (lines 349-393)
   - Update `calculate_position_size()` to use archetype multiplier (lines 311-347)
   - Update exit parameter defaults (lines 169-184)
   - Update risk parameters in KnowledgeParams (lines 83-91)

---

## Next Steps

After implementation:
1. Run backtest with archetype logging enabled
2. Analyze archetype distribution and win rates per archetype
3. Fine-tune thresholds based on results
4. Consider adding USDT.D and OI suppressors (stretch goal)

---

**Status**: Plan complete, ready to implement.
