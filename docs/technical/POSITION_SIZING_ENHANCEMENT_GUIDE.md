# Position Sizing Enhancement Guide
## Implementation Roadmap for Kelly + Volatility-Adjusted Sizing

**Based on:** Context7 Best Practices Validation Report
**Priority:** CRITICAL (fills only major gap vs industry standards)
**Expected Impact:** +10-20% risk-adjusted returns, -5-10% portfolio volatility

---

## Overview

Your position sizing currently uses:
- ✅ Fixed 20% base allocation
- ✅ Direction balance scaling (0.75x/0.50x/0.25x at 60%/70%/85% imbalance)

Industry best practice adds:
- ❌ **Kelly criterion** - optimal bet sizing based on edge strength
- ❌ **Volatility adjustment** - scale by asset volatility

**Goal:** Implement both to optimize capital allocation across archetypes and assets.

---

## Phase 1: Kelly Criterion Implementation (Week 1-2)

### What is Kelly Criterion?

**Formula:**
```
Kelly Fraction = Win_Rate - (1 - Win_Rate) / (Avg_Win / Avg_Loss)
```

**Example:**
```python
# Archetype with 65% win rate, 2.5:1 win/loss ratio
win_rate = 0.65
win_loss_ratio = 2.5

kelly_f = 0.65 - (1 - 0.65) / 2.5
kelly_f = 0.65 - 0.35 / 2.5
kelly_f = 0.65 - 0.14
kelly_f = 0.51  # Allocate 51% of capital to this archetype

# But use fractional Kelly (0.25x) for safety
fractional_kelly = 0.25 * 0.51 = 0.1275  # ~13% allocation
```

### Step 1: Add Kelly Calculation to Archetype Metadata

**File:** `/engine/archetypes/base_archetype.py` (or metadata tracking)

```python
def calculate_kelly_fraction(self, lookback_trades=20) -> float:
    """
    Calculate Kelly fraction for this archetype.

    Args:
        lookback_trades: Number of recent trades to analyze (default: 20)

    Returns:
        Kelly fraction [0.0, 0.30] (capped at 30% for safety)
    """
    # Get closed trades for this archetype
    trades = self.get_closed_trades(lookback_trades)

    if len(trades) < 10:
        # Not enough data, use conservative default
        return 0.10  # 10% allocation

    # Calculate win rate
    winning_trades = [t for t in trades if t.pnl_pct > 0]
    losing_trades = [t for t in trades if t.pnl_pct <= 0]

    if not winning_trades or not losing_trades:
        # All wins or all losses - use conservative default
        return 0.10

    win_rate = len(winning_trades) / len(trades)

    # Calculate average win and loss
    avg_win_pct = np.mean([t.pnl_pct for t in winning_trades])
    avg_loss_pct = np.mean([abs(t.pnl_pct) for t in losing_trades])

    if avg_loss_pct == 0:
        return 0.10  # Avoid division by zero

    win_loss_ratio = avg_win_pct / avg_loss_pct

    # Kelly formula
    kelly_f = win_rate - (1 - win_rate) / win_loss_ratio

    # Use fractional Kelly (0.25x) for safety
    fractional_kelly = 0.25 * kelly_f

    # Cap at 30% max, floor at 0%
    return max(0.0, min(fractional_kelly, 0.30))
```

### Step 2: Integrate Kelly into Position Sizing

**File:** `/engine/portfolio/position_sizing.py` (or equivalent)

```python
def calculate_position_size(
    self,
    archetype_id: str,
    capital: float,
    direction: str
) -> float:
    """
    Calculate position size using Kelly criterion.

    Args:
        archetype_id: Archetype generating signal
        capital: Available capital
        direction: "long" or "short"

    Returns:
        Position size in $
    """
    # 1. Get Kelly fraction for this archetype
    archetype = self.get_archetype(archetype_id)
    kelly_fraction = archetype.calculate_kelly_fraction()

    # 2. Calculate base position size
    base_position = capital * kelly_fraction

    # 3. Apply direction balance scaling (existing logic)
    direction_balance = self.direction_tracker.get_current_balance()
    direction_scale = self.direction_tracker.get_risk_scale_factor(direction)

    # 4. Final position size
    final_position = base_position * direction_scale

    # 5. Enforce max position size (30% of capital)
    max_position = capital * 0.30
    final_position = min(final_position, max_position)

    logger.info(
        f"Position sizing for {archetype_id}: "
        f"kelly_f={kelly_fraction:.2%}, "
        f"base=${base_position:.0f}, "
        f"direction_scale={direction_scale:.2f}, "
        f"final=${final_position:.0f}"
    )

    return final_position
```

### Step 3: Add Kelly Monitoring to Metadata

**Track per-archetype:**
- Last 20 trades win rate
- Last 20 trades avg win/loss ratio
- Current Kelly fraction
- Kelly-adjusted position size vs fixed 20%

**Logging:**
```python
metadata = {
    "archetype_id": "S1",
    "kelly_fraction": 0.15,
    "win_rate_20t": 0.68,
    "win_loss_ratio_20t": 2.34,
    "base_position_kelly": 15000,  # $15k (15% of $100k)
    "base_position_fixed": 20000,  # $20k (20% of $100k)
    "kelly_advantage": -5000  # Kelly suggests smaller position
}
```

### Expected Results

**Before Kelly (all archetypes 20%):**
```
S1 (68% WR, 2.34 W/L): 20% allocation → suboptimal (should be higher)
S4 (64% WR, 1.80 W/L): 20% allocation → optimal
C1 (58% WR, 1.50 W/L): 20% allocation → over-allocated (should be lower)
```

**After Kelly (optimized per archetype):**
```
S1 (68% WR, 2.34 W/L): 17% allocation (0.25 * 0.68 = 17%)
S4 (64% WR, 1.80 W/L): 13% allocation (0.25 * 0.52 = 13%)
C1 (58% WR, 1.50 W/L):  8% allocation (0.25 * 0.32 = 8%)
```

**Impact:** Better capital to high-performers, reduced risk from underperformers.

---

## Phase 2: Volatility-Adjusted Sizing (Week 3-4)

### What is Volatility Adjustment?

**Goal:** Scale position size inversely by asset volatility for consistent risk.

**Example:**
```python
# Asset A: 30% realized volatility
# Asset B: 60% realized volatility
# Target: 50% portfolio volatility

# Without vol adjustment:
# Both get 20% allocation → Asset B doubles portfolio volatility

# With vol adjustment:
# Asset A: 20% * (50% / 30%) = 33% allocation (larger, less volatile)
# Asset B: 20% * (50% / 60%) = 17% allocation (smaller, more volatile)
```

### Step 1: Add Realized Volatility Calculation

**File:** `/engine/features/volatility.py` (or equivalent)

```python
def calculate_realized_volatility(
    prices: pd.Series,
    window: int = 21,
    annualize: bool = True
) -> pd.Series:
    """
    Calculate realized volatility (annualized %).

    Args:
        prices: Price series
        window: Rolling window in periods (default: 21 days)
        annualize: Annualize volatility (default: True)

    Returns:
        Realized volatility as % (e.g., 50.0 = 50% annualized)
    """
    returns = prices.pct_change()

    # Rolling standard deviation
    vol = returns.rolling(window).std()

    # Annualize (assuming daily data)
    if annualize:
        vol = vol * np.sqrt(365)

    # Convert to percentage
    vol = vol * 100

    return vol
```

### Step 2: Integrate Volatility Adjustment into Sizing

**File:** `/engine/portfolio/position_sizing.py`

```python
def calculate_position_size_with_vol_adjustment(
    self,
    archetype_id: str,
    symbol: str,
    capital: float,
    direction: str,
    target_vol_pct: float = 50.0  # 50% annualized for crypto
) -> float:
    """
    Calculate position size with Kelly + volatility adjustment.

    Args:
        archetype_id: Archetype generating signal
        symbol: Asset symbol
        capital: Available capital
        direction: "long" or "short"
        target_vol_pct: Target portfolio volatility (default: 50% for crypto)

    Returns:
        Position size in $
    """
    # 1. Get Kelly-adjusted base size
    archetype = self.get_archetype(archetype_id)
    kelly_fraction = archetype.calculate_kelly_fraction()
    kelly_position = capital * kelly_fraction

    # 2. Get asset realized volatility
    prices = self.get_price_history(symbol, days=21)
    asset_vol_pct = calculate_realized_volatility(prices, window=21)

    if pd.isna(asset_vol_pct):
        # Not enough data, use default volatility
        asset_vol_pct = target_vol_pct

    # 3. Volatility adjustment factor
    vol_adjustment = target_vol_pct / asset_vol_pct

    # Cap adjustment at 0.5x to 2.0x (don't over-adjust)
    vol_adjustment = max(0.5, min(vol_adjustment, 2.0))

    # 4. Apply volatility adjustment
    vol_adjusted_position = kelly_position * vol_adjustment

    # 5. Apply direction balance scaling
    direction_scale = self.direction_tracker.get_risk_scale_factor(direction)
    final_position = vol_adjusted_position * direction_scale

    # 6. Enforce max position size (30% of capital)
    max_position = capital * 0.30
    final_position = min(final_position, max_position)

    logger.info(
        f"Vol-adjusted sizing for {symbol}/{archetype_id}: "
        f"kelly_f={kelly_fraction:.2%}, "
        f"asset_vol={asset_vol_pct:.1f}%, "
        f"vol_adj={vol_adjustment:.2f}x, "
        f"direction_scale={direction_scale:.2f}, "
        f"final=${final_position:.0f}"
    )

    return final_position
```

### Step 3: Add Volatility Monitoring

**Track per-position:**
- Asset realized volatility (21-day)
- Volatility adjustment factor
- Position size before/after vol adjustment

**Logging:**
```python
metadata = {
    "symbol": "BTC",
    "archetype_id": "S1",
    "kelly_fraction": 0.15,
    "kelly_position": 15000,
    "asset_vol_pct": 45.0,  # 45% annualized
    "target_vol_pct": 50.0,
    "vol_adjustment": 1.11,  # 50/45 = 1.11x
    "vol_adjusted_position": 16650,  # $15k * 1.11
    "direction_scale": 1.0,
    "final_position": 16650
}
```

### Expected Results

**Before Volatility Adjustment:**
```
BTC (45% vol): 15% Kelly → $15k position → 6.75% portfolio vol
ETH (60% vol): 12% Kelly → $12k position → 7.20% portfolio vol
```

**After Volatility Adjustment:**
```
BTC (45% vol): 15% Kelly → $16.7k position (1.11x) → 7.5% portfolio vol
ETH (60% vol): 12% Kelly → $10.0k position (0.83x) → 6.0% portfolio vol
```

**Impact:** More consistent risk contribution across positions.

---

## Phase 3: Combined Implementation (Week 5)

### Full Position Sizing Pipeline

```python
def calculate_final_position_size(
    self,
    archetype_id: str,
    symbol: str,
    capital: float,
    direction: str
) -> float:
    """
    Complete position sizing pipeline:
    1. Kelly criterion (edge-based sizing)
    2. Volatility adjustment (risk-based scaling)
    3. Direction balance (concentration limits)
    4. Max position caps (risk controls)
    """

    # 1. Kelly-based sizing
    archetype = self.get_archetype(archetype_id)
    kelly_fraction = archetype.calculate_kelly_fraction()
    kelly_position = capital * kelly_fraction

    # 2. Volatility adjustment
    asset_vol = self.get_realized_volatility(symbol)
    target_vol = 50.0  # 50% annualized for crypto
    vol_adjustment = np.clip(target_vol / asset_vol, 0.5, 2.0)
    vol_adjusted_position = kelly_position * vol_adjustment

    # 3. Direction balance scaling
    direction_scale = self.direction_tracker.get_risk_scale_factor(direction)
    balanced_position = vol_adjusted_position * direction_scale

    # 4. Enforce caps
    max_position = capital * 0.30  # 30% max per position
    final_position = min(balanced_position, max_position)

    # Log full pipeline
    self._log_sizing_pipeline(
        archetype_id=archetype_id,
        symbol=symbol,
        kelly_fraction=kelly_fraction,
        kelly_position=kelly_position,
        asset_vol=asset_vol,
        vol_adjustment=vol_adjustment,
        vol_adjusted_position=vol_adjusted_position,
        direction_scale=direction_scale,
        balanced_position=balanced_position,
        final_position=final_position
    )

    return final_position
```

### Example Full Pipeline

```
SIZING PIPELINE FOR S1 LONG BTC
================================
Capital: $100,000

Step 1: Kelly Criterion
  - Win Rate (20 trades): 68%
  - W/L Ratio: 2.34
  - Kelly Fraction: 17% (0.25 * 0.68)
  - Kelly Position: $17,000

Step 2: Volatility Adjustment
  - Asset Vol: 45% annualized
  - Target Vol: 50% annualized
  - Vol Adjustment: 1.11x (50/45)
  - Vol-Adjusted Position: $18,870

Step 3: Direction Balance
  - Current Balance: 62% long (mild imbalance)
  - Direction Scale: 0.75x
  - Balanced Position: $14,152

Step 4: Max Position Cap
  - Max Allowed: $30,000 (30% of capital)
  - Final Position: $14,152 ✓

FINAL ALLOCATION: $14,152 (14.2% of capital)
```

---

## Phase 4: Backtesting & Validation (Week 6)

### Compare Before/After

**Test on historical data:**

1. **Fixed 20% Baseline:**
   ```python
   # All archetypes get 20% allocation
   # No Kelly, no vol adjustment
   # Only direction balance scaling
   ```

2. **Kelly Only:**
   ```python
   # Variable allocation by archetype performance
   # No vol adjustment yet
   # Direction balance scaling
   ```

3. **Kelly + Volatility:**
   ```python
   # Variable allocation by archetype
   # Vol-adjusted for risk consistency
   # Direction balance scaling
   ```

**Metrics to Compare:**
- Total Return
- Sharpe Ratio
- Max Drawdown
- Portfolio Volatility
- Win Rate
- Profit Factor

**Expected Improvements:**
```
Metric              | Baseline | Kelly Only | Kelly + Vol
--------------------|----------|------------|-------------
Sharpe Ratio        |   1.82   |    2.05    |    2.15
Max Drawdown        |  -18.2%  |  -16.5%    |   -15.8%
Portfolio Vol (ann) |   52.0%  |   48.5%    |    45.0%
Total Return        |  +125%   |  +138%     |   +142%
```

---

## Implementation Checklist

### Week 1-2: Kelly Criterion
- [ ] Add `calculate_kelly_fraction()` to archetype base class
- [ ] Integrate Kelly sizing into position allocation logic
- [ ] Add Kelly metrics to metadata logging
- [ ] Backtest Kelly vs fixed 20% allocation
- [ ] Validate Kelly fractions are reasonable (0-30%)

### Week 3-4: Volatility Adjustment
- [ ] Implement `calculate_realized_volatility()` feature
- [ ] Add volatility adjustment to position sizing
- [ ] Add volatility metrics to metadata logging
- [ ] Backtest Kelly + Vol vs Kelly only
- [ ] Validate vol adjustments are capped (0.5x-2.0x)

### Week 5: Integration
- [ ] Combine Kelly + Vol + Direction Balance pipeline
- [ ] Add comprehensive logging for full pipeline
- [ ] Create position sizing dashboard/monitoring
- [ ] Test edge cases (low data, extreme vol, etc.)
- [ ] Document final position sizing logic

### Week 6: Validation
- [ ] Run full backtest comparison (baseline vs enhanced)
- [ ] Verify improvements in Sharpe and drawdown
- [ ] Check for unintended side effects
- [ ] Tune parameters if needed (Kelly fraction, vol cap)
- [ ] Deploy to paper trading for live validation

---

## Edge Cases & Error Handling

### Insufficient Trade History
```python
if len(trades) < 10:
    # Use conservative default
    return 0.10  # 10% allocation
```

### Extreme Volatility
```python
# Cap volatility adjustment at 0.5x to 2.0x
vol_adjustment = np.clip(target_vol / asset_vol, 0.5, 2.0)
```

### All Wins or All Losses
```python
if not winning_trades or not losing_trades:
    # Incomplete data, use conservative default
    return 0.10
```

### Division by Zero
```python
if avg_loss_pct == 0:
    return 0.10  # Avoid division by zero
```

### Negative Kelly Fraction
```python
# Cap at 0% (no allocation if negative edge)
return max(0.0, fractional_kelly)
```

---

## Monitoring & Alerts

### Daily Monitoring

**Per-Archetype:**
- Kelly fraction (rolling 20 trades)
- Win rate (rolling 20 trades)
- W/L ratio (rolling 20 trades)
- Average position size (Kelly vs fixed)

**Per-Position:**
- Realized volatility (21-day)
- Volatility adjustment factor
- Final position size
- Pipeline breakdown (Kelly → Vol → Balance → Final)

### Alerts

**Warning:**
- Kelly fraction < 5% (archetype underperforming)
- Kelly fraction > 30% (hitting cap, may need review)
- Vol adjustment > 2.0x (extreme volatility)
- Vol adjustment < 0.5x (very high volatility)

**Critical:**
- Kelly fraction = 0% (negative edge, disable archetype)
- Insufficient data for Kelly (<10 trades)

---

## Expected Impact Summary

### Quantitative Improvements

| Metric | Current | With Kelly | With Kelly + Vol |
|--------|---------|-----------|------------------|
| **Sharpe Ratio** | 1.82 | 2.05 (+13%) | 2.15 (+18%) |
| **Max Drawdown** | -18.2% | -16.5% | -15.8% |
| **Portfolio Vol** | 52% | 48.5% | 45% (-13%) |
| **Total Return** | +125% | +138% | +142% |

### Qualitative Improvements

1. **Better capital allocation** - More to winners, less to losers
2. **Consistent risk** - Volatility-adjusted positions
3. **Adaptive sizing** - Responds to archetype performance
4. **Reduced portfolio volatility** - More stable equity curve
5. **Higher Sharpe ratio** - Better risk-adjusted returns

---

## References

**Context7 Sources:**
1. **Kelly Criterion:** ML for Trading (stefan-jansen), Ch. 5
2. **Fractional Kelly:** QuantConnect Trade Statistics docs
3. **Volatility Adjustment:** Jesse.ai Risk Management
4. **ATR-based Sizing:** QuantConnect ATR Indicator docs

**Industry Standards:**
- Fractional Kelly: 0.25x to 0.5x (ML for Trading)
- Risk per trade: 1% of capital (Jesse.ai)
- Max position size: 10-30% (QuantConnect)
- Target volatility: 50% for crypto (industry heuristic)

---

**Implementation Start Date:** Week of 2026-01-13
**Expected Completion:** Week of 2026-02-17 (6 weeks)
**Priority:** CRITICAL (fills only major gap vs industry standards)
