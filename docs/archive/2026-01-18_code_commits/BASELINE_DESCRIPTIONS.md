# BASELINE MODEL DESCRIPTIONS

**Purpose:** Detailed explanations of each baseline model, why it's included, and what it tests.

---

## PHILOSOPHY

Baselines represent fundamental market hypotheses that have stood the test of time across decades and asset classes. These are not strawmen—they are legitimate strategies that many traders use profitably.

**Your archetypes must beat these to justify their complexity.**

---

## B0: BUY AND HOLD

### Description
Stay long 100% of the time. Never sell.

### Implementation
```python
def generate_signals(self, df):
    signals = pd.DataFrame(index=df.index)
    signals['direction'] = 'long'
    signals['size'] = 1.0
    return signals
```

### Market Hypothesis
"Asset prices rise over time. The best strategy is to hold and ignore short-term volatility."

### What It Tests
- Whether active trading adds value over passive investment
- Market beta (raw directional exposure)
- Whether your trading costs exceed your edge

### Expected Performance

**Bull Markets (2020-2021):** Excellent (PF 2.0-3.0)
- Market goes up, buy-and-hold wins
- No transaction costs except initial entry

**Bear Markets (2022):** Poor (PF 0.3-0.7)
- Market goes down, loses money
- Still only pays transaction costs once

**Sideways Markets:** Mediocre (PF 0.8-1.2)
- Chops around breakeven
- Transaction costs minimal

### Why It's Important
If your complex archetype underperforms buy-and-hold in a bull market, you're actively destroying value. Your filters and exits are harming performance, not helping.

**Red flag:** If B0 has the highest profit factor on test period, you're in a strong bull market and trend-following should dominate.

### When to Deploy B0
- When you believe in long-term asset appreciation
- When transaction costs are high (B0 minimizes trading)
- When you want pure beta exposure
- When all active strategies fail (best fallback)

### Limitations
- No downside protection
- Full exposure to drawdowns
- Cannot profit in bear markets
- Ignores all price information except "eventually up"

---

## B1: SMA200 TREND

### Description
Long when price > 200-period simple moving average. Flat otherwise.

### Implementation
```python
def generate_signals(self, df):
    sma200 = df['close'].rolling(window=200).mean()

    signals = pd.DataFrame(index=df.index)
    signals['direction'] = 'flat'
    signals.loc[df['close'] > sma200, 'direction'] = 'long'
    signals['size'] = 1.0
    return signals
```

### Market Hypothesis
"Trends persist. When price is above its long-term average, the uptrend is likely to continue."

### What It Tests
- Simple trend-following effectiveness
- Whether regime filtering adds value over buy-and-hold
- Classic technical analysis validity

### Expected Performance

**Bull Markets:** Very Good (PF 2.5-4.0)
- Captures most of uptrend
- Avoids some corrections when price dips below SMA200

**Bear Markets:** Good (PF 1.5-2.5)
- Exits to flat when price crosses below SMA200
- Avoids much of downside
- Better than buy-and-hold in bear markets

**Sideways Markets:** Mediocre (PF 1.0-1.5)
- Whipsaws around SMA200
- Multiple false signals
- Transaction costs eat into edge

### Why It's Important
SMA200 is one of the most widely-followed indicators in trading. If you can't beat this simple filter, your complex features are likely noise.

**Benchmark:** Many professional trend-followers use variations of this strategy. Your archetype should beat it by at least +0.3 PF to justify complexity.

### When to Deploy B1
- When markets are trending (bull or bear)
- When you want simple, explainable strategy
- When you need downside protection vs buy-and-hold
- As a regime filter for other strategies

### Parameter Choice
**200 periods** is standard because:
- Approximately 200 trading days in a year (for daily charts)
- For hourly: 200 hours ≈ 8-9 days (recent trend)
- Widely followed, creates self-fulfilling prophecy
- Robust across many assets and timeframes

### Limitations
- Lags turning points (slow to enter/exit)
- Whipsaws in sideways markets
- Only long side (no shorts)
- No position sizing (full size or flat)

---

## B2: SMA CROSSOVER

### Description
Long when fast SMA (50) > slow SMA (200). Flat otherwise.

### Implementation
```python
def generate_signals(self, df):
    fast_sma = df['close'].rolling(window=50).mean()
    slow_sma = df['close'].rolling(window=200).mean()

    signals = pd.DataFrame(index=df.index)
    signals['direction'] = 'flat'
    signals.loc[fast_sma > slow_sma, 'direction'] = 'long'
    signals['size'] = 1.0
    return signals
```

### Market Hypothesis
"When short-term momentum exceeds long-term trend, the market is accelerating upward."

### What It Tests
- Dual-timeframe trend confirmation
- Momentum vs trend
- Whether faster signals improve over single SMA

### Expected Performance

**Bull Markets:** Good (PF 2.0-3.5)
- Enters earlier than SMA200 alone (fast crosses up first)
- Captures momentum moves
- May exit earlier on corrections

**Bear Markets:** Good (PF 1.5-2.5)
- Exits when fast crosses below slow
- Downside protection similar to B1

**Sideways Markets:** Poor (PF 0.8-1.3)
- More whipsaws than B1 (faster signals)
- More transaction costs
- Golden cross / death cross can be false

### Why It's Important
Crossover systems are Trend Following 101. Countless traders use variations (MACD, dual MA, golden cross). If your archetype can't beat this, you're not adding value.

**Comparison to B1:** Usually more responsive but also more whipsaw-prone.

### When to Deploy B2
- When you expect strong, sustained trends
- When early entry is valuable (vs late entry)
- When you can tolerate more trading frequency
- As a momentum confirmation layer

### Parameter Choices
- **Fast = 50:** Short-term trend (2-3 days on hourly, ~2 months on daily)
- **Slow = 200:** Long-term trend (same as B1)
- **Ratio = 4x:** Standard ratio (50/200 or 20/50 are common)

**Golden Cross:** When 50 SMA crosses above 200 SMA (bullish signal)
**Death Cross:** When 50 SMA crosses below 200 SMA (bearish signal)

### Limitations
- More false signals than B1
- Higher transaction costs (more trades)
- Still lags major turning points
- No shorts, no position sizing

---

## B3: RSI MEAN REVERSION

### Description
Long when RSI < 30 (oversold). Flat when RSI > 70 (overbought). Hold between.

### Implementation
```python
def generate_signals(self, df):
    # Calculate RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    signals = pd.DataFrame(index=df.index)
    signals['direction'] = 'flat'
    signals.loc[rsi < 30, 'direction'] = 'long'
    signals.loc[rsi > 70, 'direction'] = 'flat'

    # Forward fill to maintain position
    signals['direction'] = signals['direction'].replace('flat', np.nan).fillna(method='ffill').fillna('flat')
    signals['size'] = 1.0
    return signals
```

### Market Hypothesis
"Prices oscillate around a mean. When price moves too far in one direction (oversold), it will revert."

### What It Tests
- Mean reversion vs trend-following
- Whether "buy the dip" works
- Countertrend strategies
- Short-term price exhaustion signals

### Expected Performance

**Bull Markets:** Mediocre (PF 1.5-2.0)
- Buying dips works in uptrends
- But trend-following (B1, B2) typically better
- Exits too early in strong trends

**Bear Markets:** Poor (PF 0.7-1.2)
- "Catching falling knives"
- Oversold can stay oversold in downtrends
- Mean reversion fails in trending markets

**Sideways Markets:** Good (PF 1.8-2.5)
- Best baseline for range-bound markets
- Buys lows, sells highs
- Edges from oscillation

### Why It's Important
Tests the opposite hypothesis from trend-following baselines. If B3 outperforms B1/B2, you're in a mean-reverting market, not a trending market.

**Market regime indicator:**
- If B3 > B1 → sideways/choppy market
- If B1 > B3 → trending market

### When to Deploy B3
- In sideways, range-bound markets
- When volatility is high but no clear trend
- As a counterbalance to trend-following strategies
- When you believe "buy the dip" works

### Parameter Choices
- **RSI Period = 14:** Standard setting (2 weeks on daily, ~14 hours on hourly)
- **Oversold = 30:** Classic threshold (some use 20 for stronger signal)
- **Overbought = 70:** Classic threshold (some use 80)

### Limitations
- Fails in strong trends (premature entries)
- No stop loss (can hold losers too long)
- Only long side (no mean reversion shorts)
- Binary signal (no gradation of oversold)

---

## B4: VOLATILITY-TARGETED TREND

### Description
Follow SMA200 trend but size position inversely to volatility. Target constant risk exposure.

### Implementation
```python
def generate_signals(self, df):
    # Calculate realized volatility
    returns = df['close'].pct_change()
    realized_vol = returns.rolling(window=100).std() * np.sqrt(365 * 24)  # Annualized

    # Calculate position size (inverse vol targeting)
    target_vol = 0.15  # 15% annualized
    position_size = (target_vol / realized_vol).clip(0, 2.0)  # Cap at 2x leverage

    # Trend filter
    sma200 = df['close'].rolling(window=200).mean()

    signals = pd.DataFrame(index=df.index)
    signals['direction'] = 'flat'
    signals.loc[df['close'] > sma200, 'direction'] = 'long'
    signals['size'] = position_size
    signals['size'] = signals['size'].fillna(0)
    return signals
```

### Market Hypothesis
"Trends persist, but position size should adjust for risk. In high volatility, reduce exposure. In low volatility, increase exposure."

### What It Tests
- Risk-managed trend following
- Dynamic position sizing
- Volatility as a risk signal
- Whether sizing improves over fixed-size B1

### Expected Performance

**Bull Markets:** Excellent (PF 2.5-4.5, Sharpe 1.5-2.5)
- Captures trends like B1
- Better Sharpe due to volatility management
- Reduces exposure during volatile periods
- Increases exposure during calm uptrends

**Bear Markets:** Very Good (PF 2.0-3.0)
- Already flat due to SMA200 filter
- If long during volatility spike, reduces size automatically

**Sideways Markets:** Good (PF 1.3-2.0)
- Better than B1 due to position sizing
- Smaller positions during choppy periods

### Why It's Important
This is a professional-grade baseline. Many hedge funds use volatility targeting. If your archetype can't beat this, you're not competitive with institutional strategies.

**Key insight:** Position sizing is often more important than entry timing.

### When to Deploy B4
- When risk management is priority
- When you want consistent risk exposure
- When you have access to leverage
- As a Sharpe-optimized trend strategy

### Parameter Choices
- **Lookback = 100:** Recent volatility (4 days on hourly, ~100 days on daily)
- **Target Vol = 15%:** Moderate risk target (adjust based on risk tolerance)
- **Max Leverage = 2.0x:** Conservative cap (some funds go to 3-4x)
- **Base Strategy = SMA200:** Could use B2 instead

### How It Works
```
Example:
Current price: $50,000
SMA200: $48,000 → Trend is UP → direction = 'long'

Recent volatility: 40% annualized → HIGH
Position size: 15% / 40% = 0.375 → Use 37.5% of capital

Next week:
Volatility drops to 20% annualized → LOW
Position size: 15% / 20% = 0.75 → Use 75% of capital

Effect: Same risk (15% vol), different capital allocation
```

### Limitations
- Requires leverage to be effective (size > 1.0)
- Volatility can spike suddenly (gap risk)
- Uses historical vol (backward-looking)
- More complex to implement live

---

## COMPARATIVE ANALYSIS

### When Each Baseline Wins

| Market Regime | Expected Winner | Why |
|---------------|----------------|-----|
| Strong Bull | B0 or B4 | B0 captures all upside, B4 with leverage can beat it |
| Moderate Bull | B1 or B4 | Trend + risk management wins |
| Strong Bear | B1 or B2 | Exit to flat, avoid downside |
| Sideways Choppy | B3 | Mean reversion works in ranges |
| High Volatility | B4 | Vol targeting reduces risk automatically |
| Low Volatility | B0 or B4 | B4 increases size, B0 holds full position |

### Baseline Correlations

**Highly Correlated (>0.8):**
- B1 and B2 (both trend-following)
- B1 and B4 (same direction, different sizing)

**Uncorrelated (~0.0):**
- B3 and B1 (mean reversion vs trend)

**Negative Correlation (<-0.3):**
- B3 often enters when B1 exits (countertrend)

### Expected Baseline Rankings

**Typical test period results (crypto 2023):**
```
1. B4_VolTarget     PF: 2.8, Sharpe: 2.1
2. B1_SMA200        PF: 2.5, Sharpe: 1.7
3. B2_SMACross      PF: 2.2, Sharpe: 1.4
4. B0_BuyHold       PF: 1.8, Sharpe: 1.0
5. B3_RSI           PF: 1.3, Sharpe: 0.8
```

**If your results differ significantly:**
- Different market regime (check what actually happened)
- Data quality issues
- Transaction cost assumptions
- Implementation bugs

---

## USING BASELINES FOR DIAGNOSIS

### If B0 Wins
**Market is:** Strongly trending up with minimal corrections
**Implication:** Any active trading is harmful (costs > edge)
**Action:** Consider if trend is sustainable, or if B0 just got lucky

### If B1 or B2 Win
**Market is:** Trending with some corrections
**Implication:** Trend-following works, this is a healthy baseline environment
**Action:** Your archetypes should beat these; if not, investigate overfit

### If B3 Wins
**Market is:** Choppy, range-bound, mean-reverting
**Implication:** Trend-following failing, oscillation winning
**Action:** Check if your archetypes are designed for trending markets (may need regime adaptation)

### If B4 Wins
**Market is:** Volatile with trends
**Implication:** Risk management is valuable
**Action:** Consider adding vol-based position sizing to your archetypes

### If Nothing Works (All PF < 1.5)
**Market is:** Extremely difficult (high vol, no clear direction, or unusual regime)
**Implication:** Even simple strategies struggle
**Action:** Consider sitting out, or requiring higher bar for archetype deployment

---

## CUSTOMIZING BASELINES

### Adding Your Own Baseline

```python
class MyCustomBaseline(BaseModel):
    """B5: Your hypothesis here."""

    def generate_signals(self, df):
        # Your logic here
        signals = pd.DataFrame(index=df.index)
        signals['direction'] = 'long'  # or 'short' or 'flat'
        signals['size'] = 1.0
        return signals

    def get_param_hash(self):
        return "mycustom_v1"
```

**Good candidates:**
- Momentum (price > price[N] periods ago)
- Bollinger Band mean reversion
- Volume-weighted signals
- Multi-asset correlation
- Macro factor (VIX, DXY)

**Bad candidates:**
- Overly complex (defeats purpose of baseline)
- Requires extensive data beyond OHLCV (not comparable)
- Already tested by archetypes (redundant)

### When to Add a Baseline

**Add if:**
- Represents a distinct market hypothesis
- Can be implemented in <20 lines of code
- Uses only OHLCV data (or widely available features)
- You would seriously consider deploying it

**Don't add if:**
- Just a parameter variation of existing baseline (use different params)
- Requires extensive feature engineering
- Is a prototype archetype (test as archetype, not baseline)
- Nobody would actually trade it (strawman)

---

**Remember: Baselines are the control group. Your archetypes are the treatment. Design your experiment properly.**
