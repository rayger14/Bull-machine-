# Enhanced Exit Strategies Design - Bull Machine v2.1

Generated: 2025-10-19

## Executive Summary

Building on the current solid foundation (31 trades, $5,715 PNL for BTC, 54.8% win rate), we're adding **trader knowledge gaps** identified from Moneytaur, Zeroika, and Wyckoff Insider methodologies to push win rates from ~55% to **65-70%** while maintaining risk controls.

**Current Strengths:**
- ✅ Risk-first hierarchy (SL → Partials → Trailing → Signal checks → Fallbacks)
- ✅ Dynamic adaptation (PTI/macro regime awareness)
- ✅ Confluence-driven (fusion scoring + knowledge domains)

**Identified Gaps (from semantic analysis of trader posts):**
1. **Structure Invalidation**: No OB/BB/FVG break detection
2. **Pattern-Triggered Exits**: Missing H&S, double tops, distribution anomalies
3. **Dynamic Trailing**: Fixed +1R threshold, not regime-adaptive
4. **Re-Entry Logic**: Exits are final, no pullback re-adds
5. **Edge Case Handling**: No wick hunt/spoof/manipulation detection

---

## Design Principles

### Keep Everything Canonical
- All enhancements must integrate cleanly into existing `check_exit_conditions()` hierarchy
- No breaking changes to current exit order (maintain backward compatibility)
- New checks insert logically between existing layers
- Feature store schema extensions (not replacements)

### Trader Knowledge Integration
- **Moneytaur**: Liquidity-focused trails, partial locks, re-entry on pullbacks
- **Zeroika**: Structure invalidations, manual TSL adjustments, volatility-aware widening
- **Wyckoff Insider**: Phase-based rule breaks, trap detection, geometric failures

### ML-Discoverable Patterns
- All new exit logic must be featurizable for PyTorch learning
- Embeddings for "exit wisdom" (e.g., optimal trail width predictor)
- Anomaly detection for rare invalidations (e.g., spoofing, glitches)

---

## Enhanced Exit Hierarchy (New Order)

```python
def check_exit_conditions_v2(self, row: pd.Series, trade: Trade) -> Optional[Tuple[str, float]]:
    """
    Enhanced exit logic with trader knowledge gaps filled.

    New checks inserted between existing layers (marked with 🆕).
    """

    # 1. Stop loss hit (UNCHANGED - first line of defense)
    if self._check_stop_loss(row, trade):
        return ("stop_loss", self._calculate_stop_price(trade))

    # 2. Partial exits (UNCHANGED - lock gains early)
    self._process_partial_exits(row, trade)

    # 🆕 2a. Structure Invalidation Exit (NEW - Zeroika/Moneytaur)
    #   Check BEFORE trailing to catch hard breaks early
    if self._check_structure_invalidation(row, trade):
        return ("structure_invalidated", row['close'])

    # 3. Trailing stop (ENHANCED - now regime-adaptive)
    if self._check_trailing_stop_v2(row, trade):  # 🆕 v2 = dynamic
        return ("trailing_stop", self._calculate_trailing_price(trade))

    # 🆕 3a. Pattern-Triggered Exit (NEW - Zeroika distribution signals)
    #   After trailing but before signal checks (visual patterns → early warning)
    if self._check_distribution_patterns(row, trade):
        return ("pattern_exit", row['close'])

    # 4. Fusion score drops (UNCHANGED - signal neutralized)
    fusion_score, context = self.compute_advanced_fusion_score(row)
    if fusion_score < self.params.tier3_threshold:
        # 🆕 Check for re-entry eligibility before finalizing exit
        if self._mark_reentry_candidate(row, trade, "neutralized"):
            pass  # Flag set, will check later
        return ("signal_neutralized", row['close'])

    # 5. PTI reversal detected (UNCHANGED)
    if context.get('pti_score', 0.0) > 0.6:
        if self._mark_reentry_candidate(row, trade, "pti_reversal"):
            pass
        return ("pti_reversal", row['close'])

    # 6. Macro regime flip (UNCHANGED)
    if context.get('macro_regime') == 'crisis':
        return ("macro_crisis", row['close'])

    # 🆕 6a. Edge Case Handling (NEW - Zeroika wick hunts, spoofing)
    if self._check_manipulation_exit(row, trade):
        return ("manipulation_detected", row['close'])

    # 7. Max holding period (UNCHANGED - adaptive or fixed)
    bars_held = (row.name - trade.entry_time).total_seconds() / 3600
    if self.params.adaptive_max_hold:
        max_hold_adjusted = self._compute_adaptive_max_hold(context, row, trade)
    else:
        max_hold_adjusted = self.params.max_hold_bars

    if bars_held >= max_hold_adjusted:
        return ("max_hold", row['close'])

    # 8. MTF conflict (UNCHANGED)
    mtf_conflict = row.get('mtf_conflict_score', 0.0)
    if mtf_conflict > 0.7:
        return ("mtf_conflict", row['close'])

    return None
```

---

## New Exit Mechanisms (Detailed)

### 1. Structure Invalidation Exit (🆕 Priority 2a)

**Gap Filled:** Zeroika's "exit on OB/BB/FVG melt", Moneytaur's "trail below key levels"

**Logic:**
```python
def _check_structure_invalidation(self, row: pd.Series, trade: Trade) -> bool:
    """
    Exit if critical support/resistance structures break.

    Checks (in order):
    1. Order Block (OB) invalidation: BOS close below refined OB low (longs)
    2. Breaker Block (BB) penetration: Full body through BB without recovery
    3. Fair Value Gap (FVG) fill: Price melts through FVG with momentum

    Returns:
        True if any structure invalidated (exit immediately)
    """
    # Get structure levels from feature store
    ob_low = row.get('tf1h_ob_low', None)  # From SMC detector
    bb_low = row.get('tf1h_bb_low', None)  # From HOB detector
    fvg_low = row.get('tf1h_fvg_low', None)  # From FVG detector

    current_close = row['close']

    # Long trade checks
    if trade.direction == 1:
        # OB invalidation: Close below OB with BOS confirmation
        if ob_low and current_close < ob_low:
            bos_confirmed = row.get('tf1h_bos_bearish', False)
            if bos_confirmed:
                logger.info(f"Structure invalidation: OB broken at {current_close:.2f} < {ob_low:.2f}")
                return True

        # BB penetration: Full body below BB (no wick-only)
        if bb_low and current_close < bb_low:
            body_penetration = (row['open'] + row['close']) / 2 < bb_low
            if body_penetration:
                logger.info(f"Structure invalidation: BB penetrated at {current_close:.2f}")
                return True

        # FVG melt: Price through FVG with momentum (RSI < 40)
        if fvg_low and current_close < fvg_low:
            rsi = row.get('rsi_14', 50)
            if rsi < 40:  # Momentum confirmation
                logger.info(f"Structure invalidation: FVG melted with momentum")
                return True

    # Short trade checks (inverse logic)
    else:
        ob_high = row.get('tf1h_ob_high', None)
        bb_high = row.get('tf1h_bb_high', None)
        fvg_high = row.get('tf1h_fvg_high', None)

        if ob_high and current_close > ob_high and row.get('tf1h_bos_bullish', False):
            return True
        if bb_high and current_close > bb_high and (row['open'] + row['close']) / 2 > bb_high:
            return True
        if fvg_high and current_close > fvg_high and row.get('rsi_14', 50) > 60:
            return True

    return False
```

**Feature Store Requirements:**
- `tf1h_ob_low`, `tf1h_ob_high`: Refined order block levels (from `smc_hob_v2.py`)
- `tf1h_bb_low`, `tf1h_bb_high`: Breaker block levels
- `tf1h_fvg_low`, `tf1h_fvg_high`: FVG boundaries
- `tf1h_bos_bearish`, `tf1h_bos_bullish`: Break of structure flags

**Backtesting Hypothesis:** Reduces max DD by 5-10% by cutting losers earlier when structure fails.

---

### 2. Pattern-Triggered Exits (🆕 Priority 3a)

**Gap Filled:** Zeroika's "H&S, double tops as TP cues", Moneytaur's "exit on bait/narrative dismantle"

**Logic:**
```python
def _check_distribution_patterns(self, row: pd.Series, trade: Trade) -> bool:
    """
    Exit on proactive distribution signals (before full reversal).

    Patterns (semantic from posts):
    1. Head & Shoulders: Via Wyckoff phase sequence (markup → distribution)
    2. Double Top/Bottom: Price rejects same level twice with volume divergence
    3. Prolonged Pump (Bait): Extended move without pullbacks + squiggle divergence
    4. Spike/Shadow Anomalies: Large wicks in imbalances (Zeroika: "intent signals")

    Returns:
        True if pattern detected (exit proactively)
    """
    # 1. Head & Shoulders via Wyckoff phases
    wyckoff_phase = row.get('tf1d_wyckoff_phase', 'neutral')
    prev_phase = trade.metadata.get('entry_wyckoff_phase', 'neutral')

    # Transition from markup → distribution = HS-like top
    if trade.direction == 1 and prev_phase == 'markup' and wyckoff_phase == 'distribution':
        logger.info(f"Pattern exit: Wyckoff markup→distribution transition")
        return True

    # 2. Double top detection (simple heuristic)
    #    Check if price rejected same high twice in last 20 bars
    if trade.direction == 1:
        recent_highs = self._get_recent_highs(row, lookback=20)
        if len(recent_highs) >= 2:
            # Two highs within 1% of each other with volume divergence
            if abs(recent_highs[-1] - recent_highs[-2]) / recent_highs[-1] < 0.01:
                vol_divergence = row.get('volume', 0) < trade.metadata.get('entry_volume', 0) * 0.7
                if vol_divergence:
                    logger.info(f"Pattern exit: Double top with volume divergence")
                    return True

    # 3. Prolonged pump (bait) - Moneytaur style
    #    No pullback > 5% in last 10 bars + squiggle overbought
    bars_since_entry = (row.name - trade.entry_time).total_seconds() / 3600
    if bars_since_entry > 10:
        pullback_detected = self._check_pullback_occurred(row, lookback=10, threshold=0.05)
        squiggle_overbought = row.get('tf4h_squiggle_confidence', 0.5) > 0.85

        if not pullback_detected and squiggle_overbought:
            logger.info(f"Pattern exit: Prolonged pump without pullback (bait)")
            return True

    # 4. Spike/Shadow anomalies (Zeroika wick hunts)
    wick_size = abs(row['high'] - row['low']) - abs(row['close'] - row['open'])
    body_size = abs(row['close'] - row['open'])

    if wick_size > body_size * 3:  # Wick 3x body size
        # In FVG or imbalance zone
        in_fvg = row.get('tf1h_fvg_present', False)
        if in_fvg:
            logger.info(f"Pattern exit: Large wick in FVG (wick hunt signal)")
            return True

    return False
```

**Helper Methods:**
```python
def _get_recent_highs(self, row: pd.Series, lookback: int) -> List[float]:
    """Get recent swing highs from lookback window."""
    # Access DataFrame window via self.df
    idx = self.df.index.get_loc(row.name)
    window = self.df.iloc[max(0, idx-lookback):idx+1]
    # Simple peak detection (local maxima)
    highs = []
    for i in range(1, len(window)-1):
        if window.iloc[i]['high'] > window.iloc[i-1]['high'] and window.iloc[i]['high'] > window.iloc[i+1]['high']:
            highs.append(window.iloc[i]['high'])
    return highs

def _check_pullback_occurred(self, row: pd.Series, lookback: int, threshold: float) -> bool:
    """Check if price pulled back by threshold% in lookback window."""
    idx = self.df.index.get_loc(row.name)
    window = self.df.iloc[max(0, idx-lookback):idx+1]
    peak = window['high'].max()
    trough = window['low'].min()
    pullback_pct = (peak - trough) / peak
    return pullback_pct >= threshold
```

**Feature Store Requirements:**
- Existing: `tf1d_wyckoff_phase`, `tf4h_squiggle_confidence`, `tf1h_fvg_present`
- New: `volume` column (add to feature stores if missing)

**Backtesting Hypothesis:** Captures 5-10% more profits by exiting at distribution tops instead of riding full reversals.

---

### 3. Dynamic Trailing Stop v2 (🆕 Enhanced)

**Gap Filled:** Zeroika's "widen in chop, tighten in trends", Moneytaur's "adaptive trails on volatility"

**Logic:**
```python
def _check_trailing_stop_v2(self, row: pd.Series, trade: Trade) -> bool:
    """
    Dynamic trailing stop with regime-adaptive thresholds.

    Enhancements over v1:
    - Widen trail in chop/sideways (prevent whipsaw)
    - Tighten trail in strong trends (lock gains faster)
    - Adjust based on FRVP distance (structure-aware)

    Returns:
        True if trailing stop hit
    """
    # Only activate trailing after +1R profit
    pnl_r = self._calculate_pnl_r(row, trade)
    if pnl_r <= 1.0 or not self.params.use_smart_exits:
        return False

    # Get current ATR and regime
    atr = row.get('atr_14', trade.atr_at_entry)
    regime = row.get('macro_regime', 'neutral')  # From macro detector
    volatility = row.get('volatility_percentile', 50)  # 0-100 rank

    # Base trailing multiplier (from params)
    base_mult = self.params.trailing_atr_mult  # e.g., 2.0

    # 🆕 Regime adjustments (Zeroika style)
    if regime == 'risk_on' and volatility < 30:
        # Strong trend, low vol → tighten trail to lock gains
        adjusted_mult = base_mult * 0.7  # e.g., 2.0 → 1.4 ATR
        logger.debug(f"Trailing tightened: risk_on regime")

    elif regime == 'chop' or volatility > 70:
        # Sideways/choppy → widen trail to avoid whipsaw
        adjusted_mult = base_mult * 1.5  # e.g., 2.0 → 3.0 ATR
        logger.debug(f"Trailing widened: chop regime")

    else:
        # Neutral → use base
        adjusted_mult = base_mult

    # 🆕 FRVP distance adjustment (Moneytaur: "trail below key levels")
    frvp_poc = row.get('frvp_poc', None)  # Point of Control from volume profile
    if frvp_poc:
        distance_to_poc = abs(row['close'] - frvp_poc) / row['close']
        if distance_to_poc < 0.02:  # Within 2% of POC (key level)
            # Widen trail to avoid stop hunt at POC
            adjusted_mult *= 1.3
            logger.debug(f"Trailing widened: near POC at {frvp_poc:.2f}")

    # Calculate trailing stop with adjusted multiplier
    trailing_stop = trade.entry_price + (trade.peak_profit - adjusted_mult * atr) * trade.direction

    # Check if hit
    current_price = row['close']
    if trade.direction == 1:  # Long
        if current_price <= trailing_stop:
            logger.info(f"Dynamic trailing stop hit: {current_price:.2f} <= {trailing_stop:.2f} (mult={adjusted_mult:.2f})")
            return True
    else:  # Short
        if current_price >= trailing_stop:
            return True

    return False
```

**Feature Store Requirements:**
- Existing: `macro_regime`, `atr_14`, `frvp_poc`
- New: `volatility_percentile` (rolling 50-bar percentile rank of ATR)

**Backtesting Hypothesis:** Reduces whipsaw losses by 10-15% in choppy markets while maintaining trend capture.

---

### 4. Re-Entry Logic (🆕 New Component)

**Gap Filled:** Moneytaur's "re-add on pullbacks", Zeroika's "re-enter if conditions favorable post-stop"

**Logic:**
```python
def _mark_reentry_candidate(self, row: pd.Series, trade: Trade, exit_reason: str) -> bool:
    """
    Mark trade for potential re-entry after certain exit types.

    Re-entry conditions (Moneytaur/Zeroika):
    - Exit was neutralized or PTI reversal (not stop loss!)
    - Pullback to refined level within 5 bars
    - Squiggle confirms direction
    - Fusion score recovers above tier2

    Stores re-entry flag in trade metadata for later processing.

    Returns:
        True if re-entry candidate (flag set)
    """
    # Only mark re-entry for soft exits (not hard stops)
    if exit_reason not in ['neutralized', 'pti_reversal', 'pattern_exit']:
        return False

    # Check if structure still intact (OB/BB not broken)
    if self._check_structure_invalidation(row, trade):
        return False  # Structure broken, no re-entry

    # Mark as candidate with entry conditions
    reentry_window = 5  # bars
    reentry_level = row['close'] * 0.98 if trade.direction == 1 else row['close'] * 1.02  # 2% pullback

    trade.metadata['reentry_candidate'] = True
    trade.metadata['reentry_window_end'] = row.name + pd.Timedelta(hours=reentry_window)
    trade.metadata['reentry_level'] = reentry_level
    trade.metadata['reentry_original_fusion'] = trade.entry_fusion_score

    logger.info(f"Re-entry candidate marked: {exit_reason}, level={reentry_level:.2f}, window={reentry_window}h")
    return True


def check_reentry_conditions(self, row: pd.Series, closed_trades: List[Trade]) -> Optional[Trade]:
    """
    Check if any recently closed trades qualify for re-entry.

    Called on each bar after exits processed.

    Returns:
        Trade object if re-entry triggered, else None
    """
    for trade in closed_trades:
        if not trade.metadata.get('reentry_candidate', False):
            continue

        # Check if still within window
        if row.name > trade.metadata['reentry_window_end']:
            trade.metadata['reentry_candidate'] = False  # Expired
            continue

        # Check if price pulled back to level
        current_price = row['close']
        reentry_level = trade.metadata['reentry_level']

        if trade.direction == 1:
            pullback_hit = current_price <= reentry_level
        else:
            pullback_hit = current_price >= reentry_level

        if not pullback_hit:
            continue

        # Check if conditions recovered
        fusion_score, context = self.compute_advanced_fusion_score(row)
        squiggle_confirms = context.get('momentum_score', 0.5) > 0.6

        if fusion_score >= self.params.tier2_threshold and squiggle_confirms:
            logger.info(f"RE-ENTRY triggered: fusion={fusion_score:.2f}, level={reentry_level:.2f}")

            # Clone trade with new entry
            reentry_trade = Trade(
                entry_time=row.name,
                entry_price=current_price,
                direction=trade.direction,
                position_size=self._calculate_position_size(row, fusion_score),
                initial_stop=self._calculate_stop_loss(row, trade.direction),
                atr_at_entry=row.get('atr_14', 0),
                entry_reason=f"reentry_after_{trade.exit_reason}",
                entry_fusion_score=fusion_score,
                wyckoff_phase=row.get('tf1d_wyckoff_phase', 'neutral')
            )

            # Clear re-entry flag
            trade.metadata['reentry_candidate'] = False

            return reentry_trade

    return None
```

**Integration Point:**
```python
# In backtest run() loop:
for idx, row in self.df.iterrows():
    # ... existing exit checks ...

    # 🆕 After processing exits, check for re-entries
    if closed_trades_this_bar:
        reentry_trade = self.check_reentry_conditions(row, closed_trades_this_bar)
        if reentry_trade:
            self.active_trades.append(reentry_trade)
```

**Backtesting Hypothesis:** Increases win rate by 3-5% by recapturing setups that temporarily weakened.

---

### 5. Edge Case Handling (🆕 Priority 6a)

**Gap Filled:** Zeroika's "wicks as intent, exit on SSL/BSL fails", spoofing/glitch detection

**Logic:**
```python
def _check_manipulation_exit(self, row: pd.Series, trade: Trade) -> bool:
    """
    Exit on detected manipulation, spoofing, or market anomalies.

    Signals (from Zeroika posts + wisdom):
    1. Wick hunts: Large wick at SSL/BSL without followthrough
    2. Spoofing: Volume spike with no price movement
    3. Glitches: Price gaps > 5% in single bar (crypto flash crashes)
    4. Failed sweeps: Liquidity sweep that immediately reverses

    Returns:
        True if manipulation detected
    """
    # 1. Wick hunt detection
    wick_upper = row['high'] - max(row['open'], row['close'])
    wick_lower = min(row['open'], row['close']) - row['low']
    body_size = abs(row['close'] - row['open'])

    # Large wick (3x body) at key level with reversal
    if trade.direction == 1 and wick_lower > body_size * 3:
        # Check if at SSL (Sell Side Liquidity)
        ssl_level = row.get('tf1h_ssl_level', None)  # From liquidity detector
        if ssl_level and abs(row['low'] - ssl_level) / ssl_level < 0.005:  # Within 0.5%
            # Immediate reversal (close > open after wick down)
            if row['close'] > row['open']:
                logger.warning(f"Manipulation exit: Wick hunt at SSL {ssl_level:.2f}")
                return True

    # 2. Spoofing: Volume spike without price movement
    volume = row.get('volume', 0)
    avg_volume = row.get('volume_sma_20', volume)  # 20-bar avg
    price_change = abs(row['close'] - row['open']) / row['open']

    if volume > avg_volume * 5 and price_change < 0.002:  # 5x vol spike, <0.2% move
        logger.warning(f"Manipulation exit: Volume spike without movement")
        return True

    # 3. Flash crash / glitch
    bar_change = abs(row['close'] - row['open']) / row['open']
    if bar_change > 0.05:  # >5% single-bar move
        # Check if followed by immediate reversal (next bar, if available)
        # This requires lookahead, so mark as potential and check next bar
        trade.metadata['flash_crash_flagged'] = True
        logger.warning(f"Flash crash flagged: {bar_change*100:.1f}% move")
        # Don't exit yet, wait for confirmation

    # 4. Failed sweep: SSL/BSL sweep that reversed
    sweep_occurred = row.get('tf1h_liquidity_sweep', False)  # From SMC detector
    if sweep_occurred:
        # Check if price closed back above SSL (for longs)
        if trade.direction == 1:
            ssl_swept = row.get('tf1h_ssl_swept', False)
            if ssl_swept and row['close'] < trade.entry_price * 0.98:  # 2% below entry
                logger.warning(f"Manipulation exit: Failed SSL sweep")
                return True

    return False
```

**Feature Store Requirements:**
- New: `tf1h_ssl_level`, `tf1h_bsl_level` (from liquidity detector)
- New: `tf1h_liquidity_sweep`, `tf1h_ssl_swept`, `tf1h_bsl_swept` (flags)
- New: `volume_sma_20` (rolling average)

**Backtesting Hypothesis:** Prevents 1-2% of catastrophic losses from flash crashes or manipulation.

---

## Feature Store Schema Extensions

### Required New Columns (add to `mtf_feature_builder_v2.py`)

```python
# In build_1h_features():

# Structure levels (from SMC/HOB detectors)
df['tf1h_ob_low'] = ...  # From detect_order_blocks()
df['tf1h_ob_high'] = ...
df['tf1h_bb_low'] = ...  # From detect_breaker_blocks()
df['tf1h_bb_high'] = ...
df['tf1h_fvg_low'] = ...  # From detect_fvg()
df['tf1h_fvg_high'] = ...
df['tf1h_bos_bearish'] = ...  # BOS flags
df['tf1h_bos_bullish'] = ...

# Liquidity levels (from liquidity detector)
df['tf1h_ssl_level'] = ...  # Sell Side Liquidity
df['tf1h_bsl_level'] = ...  # Buy Side Liquidity
df['tf1h_liquidity_sweep'] = ...  # Sweep occurred flag
df['tf1h_ssl_swept'] = ...
df['tf1h_bsl_swept'] = ...

# Volume metrics
df['volume'] = ...  # Raw volume (if not already present)
df['volume_sma_20'] = df['volume'].rolling(20).mean()

# Volatility percentile
df['volatility_percentile'] = df['atr_14'].rolling(50).apply(
    lambda x: (x.iloc[-1] > x).sum() / len(x) * 100
)
```

---

## Implementation Plan

### Phase 1: Structure Invalidation (Week 1)
1. Add OB/BB/FVG levels to feature stores
2. Implement `_check_structure_invalidation()` method
3. Insert check at priority 2a in `check_exit_conditions_v2()`
4. Backtest on BTC 2024: Target 5-10% DD reduction

### Phase 2: Pattern Triggers (Week 2)
1. Implement helper methods (`_get_recent_highs`, `_check_pullback_occurred`)
2. Add `_check_distribution_patterns()` method
3. Insert at priority 3a
4. Backtest on ETH 2024: Target 5-10% profit capture improvement

### Phase 3: Dynamic Trailing (Week 3)
1. Add `volatility_percentile` to feature stores
2. Implement `_check_trailing_stop_v2()` method
3. Replace v1 trailing check
4. Backtest on all assets: Target 10-15% whipsaw reduction

### Phase 4: Re-Entry Logic (Week 4)
1. Implement `_mark_reentry_candidate()` and `check_reentry_conditions()`
2. Add re-entry processing to backtest loop
3. Backtest: Target 3-5% win rate increase

### Phase 5: Edge Cases (Week 5)
1. Add liquidity sweep features to stores
2. Implement `_check_manipulation_exit()`
3. Insert at priority 6a
4. Backtest: Target 1-2% loss prevention

### Phase 6: ML Discovery (Week 6+)
1. Train PyTorch transformer on trader post embeddings for "exit wisdom"
2. Learn optimal trail width predictor (input: regime, volatility, FRVP)
3. Anomaly detector for rare invalidations
4. A/B test ML-suggested exits vs. rule-based

---

## Testing & Validation

### Backtest Acceptance Gates
- **PNL Improvement**: ≥10% vs. current baseline (BTC: $5,715 → $6,287+)
- **Win Rate**: 55% → 65-70%
- **Max DD Reduction**: ≥5% (current ~0% for BTC, maintain or improve)
- **Profit Factor**: Maintain ≥2.0
- **Trade Count**: Allow ±20% variance (some exits earlier = fewer trades ok)

### A/B Test Matrix
| Variant | Structure | Patterns | Dynamic Trail | Re-Entry | Edge Cases |
|---------|-----------|----------|---------------|----------|------------|
| Baseline (v2.0) | ❌ | ❌ | ❌ | ❌ | ❌ |
| A: Structure Only | ✅ | ❌ | ❌ | ❌ | ❌ |
| B: Structure + Patterns | ✅ | ✅ | ❌ | ❌ | ❌ |
| C: Full (v2.1) | ✅ | ✅ | ✅ | ✅ | ✅ |

Run on 2024 data for BTC/ETH/SPY. Compare PF, Sharpe, DD.

### Overfit Prevention
- Validate on 2023 data (out-of-sample)
- If 2023 PF drops >20% vs. 2024, gate via min PF threshold
- Use walk-forward optimization (train on 2024-H1, test on 2024-H2)

---

## ML Integration (PyTorch)

### Exit Wisdom Transformer
```python
# Pseudo-code for ML-enhanced exits
class ExitWisdomModel(nn.Module):
    """
    Learn optimal exit timing from trader knowledge embeddings.

    Inputs:
    - Fusion score history (last 10 bars)
    - Regime features (wyckoff_phase, macro, pti)
    - Structure proximity (distance to OB/BB/FVG)
    - Volatility metrics (ATR percentile, regime vol)

    Output:
    - Exit confidence score [0, 1] (>0.7 = exit now)
    - Optimal trail width multiplier [0.5, 3.0]
    """
    def forward(self, features):
        # Transformer encoder on time series
        embedded = self.embedding(features)
        encoded = self.transformer(embedded)

        # Dual heads
        exit_conf = self.exit_head(encoded)  # Sigmoid
        trail_mult = self.trail_head(encoded)  # Linear + ReLU

        return exit_conf, trail_mult


# Training data: Historical trades with "perfect" exits (hindsight TP levels)
# Loss: Maximize reward = (actual_exit_pnl / perfect_exit_pnl)
```

### Anomaly Detection (for manipulation exits)
```python
# Isolation Forest or VAE for rare events
from sklearn.ensemble import IsolationForest

# Train on normal price action (no wicks, stable vol)
# Predict anomalies: wick hunts, spoofing, glitches
# If anomaly score > threshold → trigger manipulation exit
```

---

## Success Metrics (Definition of Done)

Enhanced exits (v2.1) PASS when:
1. ✅ Backtest PNL improves ≥10% on 2024 data (all assets)
2. ✅ Win rate increases to 65-70% (from ~55%)
3. ✅ Max DD reduces or maintains current levels
4. ✅ Profit Factor ≥2.0 maintained
5. ✅ Out-of-sample (2023) validation shows <20% PF degradation
6. ✅ All new exit mechanisms fire correctly in logs (structure, patterns, dynamic trail, re-entry, edge cases)
7. ✅ ML wisdom model shows ≥5% improvement over rule-based in A/B test

---

## Next Steps (After Design Approval)

1. **Review & Approve**: Get user confirmation on design
2. **Feature Store Updates**: Add new columns (OB/BB/FVG levels, liquidity, volume metrics)
3. **Implement Phase 1**: Structure invalidation exit
4. **Backtest & Iterate**: Test each phase incrementally
5. **ML Training**: Collect trader post embeddings, train wisdom model
6. **Deploy v2.1**: Update replay runner with enhanced exits
7. **Shadow-Live Test**: Validate in shadow mode before capital deployment

---

**Status**: Design complete, awaiting approval to proceed with implementation.
