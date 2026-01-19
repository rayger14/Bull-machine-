# Streaming Feature Engineering Research Report

**Research Focus:** Online/streaming feature engineering for financial time series
**Objective:** Implement incremental feature computation with batch/stream parity
**Date:** 2025-11-22
**Status:** Production-Ready Recommendations

---

## Executive Summary

This report provides production-ready solutions for implementing streaming feature engineering in the Bull Machine trading system. Based on comprehensive research of industry best practices, open-source libraries, and production trading systems, we recommend a **hybrid approach** combining:

1. **Welford's algorithm** for incremental statistics (mean, variance, z-scores)
2. **Deque-based ring buffers** for rolling windows
3. **Stateful feature classes** (inspired by streaming-indicators pattern)
4. **Property-based testing** with Hypothesis for batch/stream parity validation

**Key Finding:** Most production trading systems (QuantConnect LEAN, NautilusTrader) use event-driven architectures with stateful indicators that compute incrementally. This approach provides identical behavior in backtesting and live trading.

---

## Table of Contents

1. [Library Recommendations](#1-library-recommendations)
2. [Rolling Window Management](#2-rolling-window-management)
3. [State Machine Patterns](#3-state-machine-patterns)
4. [Feature Parity Validation](#4-feature-parity-validation)
5. [Production Systems Analysis](#5-production-systems-analysis)
6. [Feature-Specific Implementation](#6-feature-specific-implementation)
7. [Implementation Checklist](#7-implementation-checklist)
8. [Code Examples](#8-code-examples)
9. [Performance Optimization](#9-performance-optimization)
10. [Anti-Patterns to Avoid](#10-anti-patterns-to-avoid)

---

## 1. Library Recommendations

### 1.1 For Online Statistics: Welford (RECOMMENDED)

**Library:** `welford` (PyPI)
**GitHub:** https://github.com/a-mitani/welford
**Status:** Production-ready, actively maintained

**Pros:**
- Numerically stable (critical for financial data)
- Supports incremental mean, variance, standard deviation
- Batch update capability (fast warmup)
- Parallel merging (combine statistics from different windows)
- Zero external dependencies (NumPy only)
- Simple API

**Cons:**
- No rolling window support (you must manage window yourself)
- No built-in z-score calculation (trivial to add)
- No quantile support (use separate library)

**Usage Example:**
```python
from welford import Welford

# Incremental calculation
w = Welford()
for value in stream:
    w.add(np.array([value]))
    mean = w.mean[0]
    std = np.sqrt(w.var_s[0])
    z_score = (value - mean) / std if std > 0 else 0
```

**Verdict:** **USE for incremental mean/variance/std calculations**

### 1.2 For Machine Learning: River (OPTIONAL)

**Library:** `river` (formerly creme)
**GitHub:** https://github.com/online-ml/river
**Status:** Production-ready, well-maintained

**Pros:**
- Comprehensive online ML library
- Rolling statistics built-in (`stats.Mean`, `stats.Var`, `stats.RollingMean`)
- Drift detection algorithms
- Model incremental learning
- Active community (22k+ GitHub stars)

**Cons:**
- Heavy dependency (not needed for simple statistics)
- Designed for ML workflows (overkill for feature engineering)
- Learning curve for API
- No financial-specific features

**Usage Example:**
```python
from river import stats

# Rolling mean
rolling_mean = stats.RollingMean(window_size=20)
for value in stream:
    rolling_mean.update(value)
    current_mean = rolling_mean.get()
```

**Verdict:** **AVOID for now** - Too heavy for our needs. Welford + custom code is sufficient.

### 1.3 For Technical Indicators: streaming-indicators (CONSIDER)

**Library:** `streaming-indicators`
**GitHub:** https://github.com/mr-easy/streaming_indicators
**Status:** Actively maintained, production-ready

**Pros:**
- Designed specifically for trading indicators
- Stateful pattern (same code for backtest + live)
- 15+ indicators (SMA, EMA, RSI, ATR, Bollinger Bands, etc.)
- `update()` vs `compute()` pattern (persistent vs temporary)
- Supports warmup from historical data
- Lightweight

**Cons:**
- Limited to built-in indicators (no custom features)
- No liquidity-specific features
- Would need to extend for Bull Machine features

**Usage Example:**
```python
import streaming_indicators as si

# Create indicator with warmup
SMA = si.SMA(period=20)
# Optionally warmup with historical data
SMA = si.SMA(period=20, candles=historical_df)

# Stream updates
for candle in stream:
    sma_value = SMA.update(candle['close'])
```

**Verdict:** **REFERENCE for design patterns**, but implement custom features ourselves.

### 1.4 Summary Table

| Library | Use Case | Recommendation | Priority |
|---------|----------|----------------|----------|
| **welford** | Incremental mean/variance/std | **USE** | HIGH |
| **collections.deque** | Rolling windows | **USE** | HIGH |
| **streaming-indicators** | Design pattern reference | **STUDY** | MEDIUM |
| **river** | Complex ML features | **SKIP** | LOW |
| **numba** | Performance optimization | **LATER** | MEDIUM |

---

## 2. Rolling Window Management

### 2.1 Data Structure Comparison

| Structure | Update Speed | Memory | Random Access | Use Case |
|-----------|-------------|--------|---------------|----------|
| **collections.deque** | O(1) | O(k) | O(n) | General-purpose rolling windows |
| **NumPy ring buffer** | O(1) | O(k) | O(1) | Large numeric windows, fast indexing |
| **pandas.rolling** | O(n) | O(n) | N/A | Batch processing only |

**Research Finding:** For windows < 100 elements, `collections.deque` is fastest. For larger windows with frequent random access, NumPy ring buffer is better.

### 2.2 Recommended Approach: Deque-based Ring Buffer

**Rationale:**
- Most Bull Machine features use windows of 7-30 bars
- Deque is simpler and faster for small windows
- Thread-safe (important for live trading)
- Python stdlib (no dependencies)

**Implementation Pattern:**
```python
from collections import deque

class RollingWindow:
    """Efficient rolling window using deque."""

    def __init__(self, size: int):
        self.size = size
        self.window = deque(maxlen=size)

    def update(self, value: float) -> None:
        """Add new value (oldest auto-evicted when full)."""
        self.window.append(value)

    def is_warm(self) -> bool:
        """Check if window is fully populated."""
        return len(self.window) >= self.size

    def get_window(self) -> list:
        """Get current window as list."""
        return list(self.window)

    def mean(self) -> float:
        """Calculate mean of window."""
        return sum(self.window) / len(self.window) if self.window else 0.0

    def max(self) -> float:
        """Calculate max of window."""
        return max(self.window) if self.window else 0.0

    def count(self, condition) -> int:
        """Count elements matching condition."""
        return sum(1 for x in self.window if condition(x))
```

### 2.3 Efficient Rolling Min/Max: Ascending Minima Algorithm

**Research Finding:** The `rolling` library implements the "ascending minima" algorithm using deque for O(1) amortized rolling min/max.

**Algorithm:** Maintain a deque of candidates where each element could potentially become the minimum in future windows.

**Implementation:**
```python
from collections import deque

class RollingMax:
    """
    Efficient rolling maximum using ascending minima algorithm.
    O(1) amortized complexity.
    """

    def __init__(self, window_size: int):
        self.window_size = window_size
        self.deque = deque()  # (index, value) tuples
        self.index = 0

    def update(self, value: float) -> float:
        """
        Add new value and return current max.

        Algorithm:
        1. Remove elements from tail that are smaller than new value
           (they can never be max while new value is in window)
        2. Add new value to tail
        3. Remove elements from head that are outside window
        4. Max is at head of deque
        """
        # Remove elements that can't be max anymore
        while self.deque and self.deque[-1][1] <= value:
            self.deque.pop()

        # Add new element
        self.deque.append((self.index, value))

        # Remove elements outside window
        while self.deque and self.deque[0][0] <= self.index - self.window_size:
            self.deque.popleft()

        self.index += 1

        return self.deque[0][1] if self.deque else value
```

**Verdict:** **USE for capitulation_depth** (rolling max tracking for drawdown calculation)

---

## 3. State Machine Patterns

### 3.1 Industry Standard: Stateful Indicator Pattern

**Research Finding:** Production trading systems (QuantConnect, streaming-indicators, backtrader) use a common pattern:

**Pattern Structure:**
```python
class Indicator:
    """Base pattern for stateful indicators."""

    def __init__(self, period: int, warmup_data: Optional[pd.DataFrame] = None):
        """
        Initialize indicator with parameters.

        Args:
            period: Lookback period
            warmup_data: Optional historical data for initialization
        """
        self.period = period
        self._state = self._initialize_state()

        if warmup_data is not None:
            self._warmup(warmup_data)

    def _initialize_state(self) -> dict:
        """Initialize internal state."""
        raise NotImplementedError

    def _warmup(self, data: pd.DataFrame) -> None:
        """Populate state with historical data."""
        for _, row in data.iterrows():
            self.update(row)

    def update(self, value: Any) -> float:
        """
        Update state with new value and return result.
        This method is called in both backtest and live.
        """
        raise NotImplementedError

    def compute(self, value: Any) -> float:
        """
        Compute indicator WITHOUT updating state.
        Useful for preview/what-if analysis.
        """
        raise NotImplementedError

    def is_ready(self) -> bool:
        """Check if indicator has enough data."""
        raise NotImplementedError
```

### 3.2 Application to Bull Machine Features

**Design Decision:** Create `StreamingFeature` base class following this pattern.

**Benefits:**
1. **Batch/stream parity:** Same code path for backtest and live
2. **Testable:** Can unit test each feature independently
3. **Composable:** Features can depend on other features
4. **Debuggable:** Can inspect internal state at any point
5. **Warm-startable:** Can initialize from historical data

**Example for `liquidity_persistence`:**
```python
from collections import deque

class LiquidityPersistence(StreamingFeature):
    """
    Count bars where condition holds in rolling window.
    Condition: liquidity_score < threshold
    """

    def __init__(self, window: int = 7, threshold: float = 0.30):
        self.window_size = window
        self.threshold = threshold
        self.condition_window = deque(maxlen=window)
        self._count = 0

    def update(self, liquidity_score: float) -> int:
        """
        Update with new liquidity score.

        Returns:
            Count of bars where liquidity < threshold in window
        """
        # Determine if condition met
        condition_met = liquidity_score < self.threshold

        # If window full, check if oldest element met condition
        if len(self.condition_window) == self.window_size:
            oldest_met = self.condition_window[0]
            if oldest_met:
                self._count -= 1

        # Add new condition state
        self.condition_window.append(condition_met)
        if condition_met:
            self._count += 1

        return self._count

    def get(self) -> int:
        """Get current count without updating."""
        return self._count

    def is_ready(self) -> bool:
        """Check if window is full."""
        return len(self.condition_window) >= self.window_size
```

**Verdict:** **IMPLEMENT this pattern** for all new streaming features.

---

## 4. Feature Parity Validation

### 4.1 Property-Based Testing with Hypothesis

**Research Finding:** Property-based testing is the gold standard for validating batch/stream equivalence.

**Library:** `hypothesis` (PyPI)
**Approach:** Generate random data streams, compute features both ways, assert equality.

**Key Properties to Test:**

1. **Commutative Property:** Order matters, but result should be deterministic
2. **Warmup Equivalence:** Batch-calculated warmup = stream updates
3. **Floating Point Tolerance:** Results within epsilon (account for precision)
4. **Edge Cases:** Empty data, single value, all zeros, extreme values

**Example Test:**
```python
import pytest
from hypothesis import given, strategies as st
import pandas as pd
import numpy as np

@given(st.lists(st.floats(min_value=-1000, max_value=1000), min_size=30, max_size=100))
def test_rolling_mean_parity(values):
    """Test streaming mean matches pandas rolling mean."""

    # Batch calculation (pandas)
    df = pd.DataFrame({'value': values})
    batch_result = df['value'].rolling(window=20, min_periods=1).mean()

    # Stream calculation
    stream_feature = StreamingMean(window=20)
    stream_result = []
    for v in values:
        stream_result.append(stream_feature.update(v))

    # Assert equivalence (with floating point tolerance)
    np.testing.assert_allclose(
        batch_result.values,
        stream_result,
        rtol=1e-9,
        atol=1e-9,
        err_msg="Streaming mean diverged from batch calculation"
    )
```

### 4.2 Warmup Period Testing Strategy

**Research Finding:** QuantConnect LEAN uses a warmup period that:
1. Loads historical data before strategy start
2. Replays data through indicators WITHOUT placing trades
3. Ensures indicators are "hot" when strategy begins

**Recommended Approach:**
```python
def test_warmup_equivalence():
    """Test warmup from historical data matches incremental updates."""

    # Generate test data
    historical = generate_test_data(100)
    live_start = 80  # Warmup on first 80 bars

    # Method 1: Incremental warmup
    feature1 = StreamingFeature()
    for i in range(live_start):
        feature1.update(historical[i])
    value1 = feature1.update(historical[live_start])

    # Method 2: Batch warmup
    feature2 = StreamingFeature()
    feature2.warmup(historical[:live_start])
    value2 = feature2.update(historical[live_start])

    # Method 3: Pandas batch
    df = pd.DataFrame({'value': historical})
    value3 = compute_feature_batch(df, window=20).iloc[live_start]

    assert abs(value1 - value2) < 1e-9, "Incremental != batch warmup"
    assert abs(value1 - value3) < 1e-9, "Streaming != pandas batch"
```

### 4.3 Common Pitfalls to Test

**From Research:** These are the most common causes of batch/stream divergence:

1. **Off-by-one errors:** Window indexing bugs
2. **Warmup bias:** Different behavior during warmup vs steady state
3. **Floating point accumulation:** Batch uses vectorized ops, stream accumulates
4. **Missing data handling:** NaN behavior differs
5. **State initialization:** Forgot to reset state between tests

**Test Framework:**
```python
class FeatureParityTest:
    """Base class for testing batch/stream parity."""

    def test_empty_input(self):
        """Feature handles empty input gracefully."""
        pass

    def test_single_value(self):
        """Feature handles single data point."""
        pass

    def test_warmup_period(self):
        """Feature behavior during warmup matches expectations."""
        pass

    def test_nan_handling(self):
        """Feature handles NaN values consistently."""
        pass

    def test_extreme_values(self):
        """Feature handles outliers without overflow."""
        pass

    def test_batch_stream_equivalence(self):
        """Streaming matches batch calculation."""
        pass

    def test_state_reset(self):
        """Feature state resets properly."""
        pass
```

**Verdict:** **IMPLEMENT property-based tests** for all streaming features before production.

---

## 5. Production Systems Analysis

### 5.1 QuantConnect LEAN Architecture

**Research Finding:** LEAN uses a sophisticated warmup mechanism:

**Key Insights:**
1. **Warmup is automatic:** Framework handles it, not user code
2. **Resolution-aware:** Warmup uses lowest resolution of data subscriptions
3. **No trading during warmup:** Prevents look-ahead bias
4. **Indicator registration:** Indicators auto-register for warmup
5. **History API:** Alternative to warmup for manual initialization

**LEAN Pattern:**
```python
class MyStrategy(QCAlgorithm):
    def Initialize(self):
        # Set warmup period (days)
        self.SetWarmup(timedelta(days=30))

        # Create indicators - they auto-warmup
        self.sma = self.SMA("BTC", 20, Resolution.Hour)

    def OnData(self, data):
        # During warmup, this runs but can't trade
        if self.IsWarmingUp:
            return

        # Indicator is now "hot"
        if self.sma.IsReady:
            current_sma = self.sma.Current.Value
```

**Application to Bull Machine:**
- Implement `is_warming_up` flag in backtester
- Track warmup period requirement per feature
- Log warnings if trading before features ready

### 5.2 NautilusTrader: Event-Driven Architecture

**Research Finding:** NautilusTrader achieves parity through **event-driven design**:

**Key Principles:**
1. **Same engine:** Backtest and live use identical core
2. **Event sourcing:** All data flows as events
3. **No vectorization:** Process one event at a time (even in backtest)
4. **State machines:** All logic is stateful, incremental

**Pattern:**
```python
class Indicator:
    def handle_bar(self, bar: Bar):
        """Process bar event (same in backtest and live)."""
        self._update_state(bar)
        self._calculate_value()

        if self.has_output:
            self.emit_value(self.value)
```

**Verdict:** Bull Machine already uses event-driven pattern in `logic_v2_adapter.py`. **CONTINUE this approach.**

### 5.3 Streaming-Indicators: Update/Compute Pattern

**Research Finding:** The `update()` vs `compute()` split is powerful:

**Update:** Persistent state change
```python
sma_value = SMA.update(close)  # Updates internal state
```

**Compute:** Temporary "what-if" calculation
```python
preview = SMA.compute(close)  # Doesn't update state
```

**Use Cases:**
- **Update:** Use during data ingestion (backtest or live)
- **Compute:** Use for:
  - Last tick price (not a confirmed bar yet)
  - Preview mode
  - Testing
  - Debugging

**Verdict:** **IMPLEMENT both methods** in Bull Machine streaming features.

---

## 6. Feature-Specific Implementation

### 6.1 liquidity_drain_pct (Rolling Mean + Percentage Change)

**Calculation:** `(current_liquidity - mean_7d) / mean_7d`

**Challenge:** Incremental mean calculation

**Solution:** Welford's algorithm OR simple deque-based running sum

**Recommended Implementation:**
```python
from collections import deque

class LiquidityDrainPct(StreamingFeature):
    """
    Calculate liquidity drain percentage from 7-day rolling mean.

    Formula: (current - mean_7d) / mean_7d
    """

    def __init__(self, window: int = 168):  # 7 days * 24 hours
        self.window_size = window
        self.liquidity_window = deque(maxlen=window)
        self._sum = 0.0

    def update(self, liquidity_score: float) -> float:
        """
        Update with new liquidity score.

        Returns:
            Percentage change from rolling mean, or 0.0 if not ready
        """
        # If window full, subtract oldest value from sum
        if len(self.liquidity_window) == self.window_size:
            self._sum -= self.liquidity_window[0]

        # Add new value
        self.liquidity_window.append(liquidity_score)
        self._sum += liquidity_score

        # Calculate percentage change
        if not self.is_ready():
            return 0.0

        mean_7d = self._sum / len(self.liquidity_window)

        if mean_7d == 0:
            return 0.0  # Avoid division by zero

        drain_pct = (liquidity_score - mean_7d) / mean_7d
        return drain_pct

    def is_ready(self) -> bool:
        """Ready when window is full."""
        return len(self.liquidity_window) >= self.window_size
```

**Warmup Handling:**
```python
# During first 7 days, feature returns 0.0
# This is acceptable - early signals are lower quality anyway
# Alternative: Use expanding window mean during warmup
```

**Batch Equivalence Test:**
```python
def test_liquidity_drain_pct_equivalence():
    """Test streaming matches pandas calculation."""

    # Generate test data
    np.random.seed(42)
    liquidity_scores = np.random.uniform(0.1, 0.9, 200)

    # Batch calculation
    df = pd.DataFrame({'liquidity_score': liquidity_scores})
    df['mean_7d'] = df['liquidity_score'].rolling(window=168, min_periods=1).mean()
    df['drain_pct'] = (df['liquidity_score'] - df['mean_7d']) / df['mean_7d']
    batch_result = df['drain_pct'].fillna(0.0)

    # Stream calculation
    feature = LiquidityDrainPct(window=168)
    stream_result = [feature.update(x) for x in liquidity_scores]

    # Assert equivalence (after warmup)
    np.testing.assert_allclose(
        batch_result.values[168:],  # Skip warmup period
        stream_result[168:],
        rtol=1e-7
    )
```

### 6.2 liquidity_persistence (Counting Condition in Window)

**Calculation:** `count(liquidity < 0.30 in last 7 bars)`

**Challenge:** Efficient counting without scanning entire window

**Solution:** Maintain running count, increment/decrement on updates

**Recommended Implementation:**
```python
from collections import deque

class LiquidityPersistence(StreamingFeature):
    """
    Count bars where liquidity < threshold in rolling window.

    Efficient O(1) updates using running count.
    """

    def __init__(self, window: int = 7, threshold: float = 0.30):
        self.window_size = window
        self.threshold = threshold
        self.condition_window = deque(maxlen=window)
        self._count = 0

    def update(self, liquidity_score: float) -> int:
        """
        Update with new liquidity score.

        Returns:
            Count of bars where liquidity < threshold in window
        """
        # Check if new value meets condition
        new_condition_met = liquidity_score < self.threshold

        # If window is full, check oldest value
        if len(self.condition_window) == self.window_size:
            oldest_met = self.condition_window[0]
            if oldest_met:
                self._count -= 1

        # Add new value
        self.condition_window.append(new_condition_met)
        if new_condition_met:
            self._count += 1

        return self._count

    def is_ready(self) -> bool:
        """Ready when window is full."""
        return len(self.condition_window) >= self.window_size
```

**Performance:** O(1) per update vs O(n) if scanning window each time.

**Alternative Pattern (if condition is complex):**
```python
# For complex conditions, store values and scan
class GenericPersistence(StreamingFeature):
    def __init__(self, window: int, condition_fn):
        self.window_size = window
        self.condition_fn = condition_fn
        self.value_window = deque(maxlen=window)

    def update(self, value: Any) -> int:
        self.value_window.append(value)
        # O(n) scan - acceptable for small windows (< 50)
        return sum(1 for v in self.value_window if self.condition_fn(v))
```

### 6.3 capitulation_depth (Rolling Max + Drawdown)

**Calculation:** `(current_close - max_close_in_window) / max_close_in_window`

**Challenge:** Efficient rolling max tracking

**Solution:** Ascending minima algorithm (deque-based)

**Recommended Implementation:**
```python
from collections import deque

class CapitulationDepth(StreamingFeature):
    """
    Calculate drawdown from rolling maximum.

    Uses efficient ascending minima algorithm for O(1) amortized updates.
    """

    def __init__(self, window: int = 720):  # 30 days * 24 hours
        self.window_size = window
        self.max_deque = deque()  # (index, price) tuples
        self.index = 0

    def update(self, close_price: float) -> float:
        """
        Update with new close price.

        Returns:
            Drawdown percentage from rolling max (negative value)
        """
        # Remove smaller values that can never be max
        while self.max_deque and self.max_deque[-1][1] <= close_price:
            self.max_deque.pop()

        # Add new price
        self.max_deque.append((self.index, close_price))

        # Remove values outside window
        while self.max_deque and self.max_deque[0][0] <= self.index - self.window_size:
            self.max_deque.popleft()

        self.index += 1

        # Calculate drawdown
        if not self.max_deque:
            return 0.0

        max_price = self.max_deque[0][1]
        if max_price == 0:
            return 0.0

        drawdown = (close_price - max_price) / max_price
        return drawdown  # Will be negative during drawdowns

    def is_ready(self) -> bool:
        """Ready after first update."""
        return self.index > 0
```

**Complexity:** O(1) amortized (vs O(n) for naive max scan)

**Memory:** O(k) where k is number of local maxima in window (typically << window_size)

### 6.4 crisis_composite (Weighted Combination of Z-Scores)

**Calculation:** `w1*vix_z + w2*dxy_z + w3*hyg_z + w4*funding_z`

**Challenge:** Need z-scores of multiple features

**Solution:** Use Welford's algorithm for each component

**Recommended Implementation:**
```python
from welford import Welford
import numpy as np

class CrisisComposite(StreamingFeature):
    """
    Weighted combination of crisis indicator z-scores.

    Components: VIX, DXY, HYG, Funding Rate
    """

    def __init__(self,
                 vix_weight: float = 0.30,
                 dxy_weight: float = 0.25,
                 hyg_weight: float = 0.25,
                 funding_weight: float = 0.20,
                 zscore_window: int = 720):  # 30 days for z-score

        self.weights = {
            'vix': vix_weight,
            'dxy': dxy_weight,
            'hyg': hyg_weight,
            'funding': funding_weight
        }

        # Welford objects for each component
        self.vix_welford = Welford()
        self.dxy_welford = Welford()
        self.hyg_welford = Welford()
        self.funding_welford = Welford()

        self.zscore_window = zscore_window
        self._update_count = 0

    def update(self, vix: float, dxy: float, hyg: float, funding: float) -> float:
        """
        Update with new values.

        Args:
            vix: VIX level
            dxy: DXY level
            hyg: HYG close
            funding: Funding rate

        Returns:
            Weighted crisis composite score
        """
        # Update Welford statistics
        self.vix_welford.add(np.array([vix]))
        self.dxy_welford.add(np.array([dxy]))
        self.hyg_welford.add(np.array([hyg]))
        self.funding_welford.add(np.array([funding]))

        self._update_count += 1

        # Need warmup period for z-scores
        if not self.is_ready():
            return 0.0

        # Calculate z-scores
        vix_z = self._zscore(vix, self.vix_welford)
        dxy_z = self._zscore(dxy, self.dxy_welford)
        hyg_z = self._zscore(hyg, self.hyg_welford)
        funding_z = self._zscore(funding, self.funding_welford)

        # Weighted combination
        composite = (
            self.weights['vix'] * vix_z +
            self.weights['dxy'] * dxy_z +
            self.weights['hyg'] * hyg_z +
            self.weights['funding'] * funding_z
        )

        return composite

    def _zscore(self, value: float, welford_obj: Welford) -> float:
        """Calculate z-score using Welford statistics."""
        mean = welford_obj.mean[0]
        var = welford_obj.var_s[0]
        std = np.sqrt(var) if var > 0 else 0.0

        if std == 0:
            return 0.0

        return (value - mean) / std

    def is_ready(self) -> bool:
        """Ready after warmup period."""
        return self._update_count >= self.zscore_window
```

**Advanced:** For rolling z-scores (not expanding), combine Welford with deque:
```python
class RollingZScore(StreamingFeature):
    """Rolling z-score using Welford + deque."""

    def __init__(self, window: int = 720):
        self.window = deque(maxlen=window)
        self.welford = Welford()

    def update(self, value: float) -> float:
        # If window full, need to "subtract" oldest value
        # Welford doesn't support removal - reconstruct from scratch
        if len(self.window) == self.window.maxlen:
            # Option 1: Reconstruct (slow but accurate)
            self.welford = Welford()
            for v in self.window:
                self.welford.add(np.array([v]))

        # Add new value
        self.window.append(value)
        self.welford.add(np.array([value]))

        # Calculate z-score
        mean = self.welford.mean[0]
        std = np.sqrt(self.welford.var_s[0])
        return (value - mean) / std if std > 0 else 0.0
```

**Performance Note:** Rolling z-score with Welford reconstruction is O(n). For high-frequency updates, use simpler running sum/sum-of-squares approach.

---

## 7. Implementation Checklist

### 7.1 Common Mistakes in Batch→Stream Conversion

Based on research and industry experience:

**Mistake 1: Index Confusion**
- **Problem:** Using pandas `.iloc[-1]` thinking patterns in streaming
- **Solution:** No indexing - always process current value

**Mistake 2: Look-Ahead Bias**
- **Problem:** Accessing future data (e.g., `df.shift(-1)`)
- **Solution:** Only use current and past values

**Mistake 3: Different Warmup Behavior**
- **Problem:** Batch uses `.rolling(window, min_periods=1)`, stream uses fixed window
- **Solution:** Match `min_periods` logic in stream (use expanding window during warmup)

**Mistake 4: Floating Point Divergence**
- **Problem:** Batch vectorized ops vs stream accumulation
- **Solution:** Use same numerical algorithm (e.g., Welford for both)

**Mistake 5: State Pollution**
- **Problem:** Forgetting to reset state between backtests
- **Solution:** Always create fresh feature instances

**Mistake 6: Missing Data Handling**
- **Problem:** NaN handling differs (pandas fillna vs manual)
- **Solution:** Explicit NaN checks, consistent fill strategy

**Mistake 7: Off-By-One Errors**
- **Problem:** Window includes current bar or not?
- **Solution:** Document clearly, test edge cases

### 7.2 Edge Cases to Test

**For Each Feature:**

```python
class FeatureEdgeCaseTests:
    """Comprehensive edge case testing."""

    def test_empty_input(self):
        """Feature handles zero updates."""
        feature = MyFeature()
        assert feature.is_ready() == False

    def test_single_value(self):
        """Feature handles single data point."""
        feature = MyFeature(window=10)
        result = feature.update(100.0)
        assert not feature.is_ready()  # Need full window

    def test_all_zeros(self):
        """Feature handles zero variance data."""
        feature = MyFeature()
        for _ in range(100):
            result = feature.update(0.0)
        # Should not crash, return 0 or NaN appropriately

    def test_all_same_value(self):
        """Feature handles constant data."""
        feature = MyFeature()
        for _ in range(100):
            result = feature.update(42.0)
        # Z-score should be 0, variance should be 0

    def test_extreme_values(self):
        """Feature handles outliers without overflow."""
        feature = MyFeature()
        values = [1, 2, 3, 1e6, 4, 5]  # One extreme outlier
        for v in values:
            result = feature.update(v)
        assert not np.isnan(result)
        assert not np.isinf(result)

    def test_nan_input(self):
        """Feature handles NaN values."""
        feature = MyFeature()
        result = feature.update(np.nan)
        # Should either skip, fill with 0, or propagate NaN
        # Document behavior clearly

    def test_negative_values(self):
        """Feature handles negative inputs."""
        feature = MyFeature()
        result = feature.update(-100.0)
        # Should handle gracefully

    def test_warmup_period(self):
        """Feature behavior during warmup is correct."""
        feature = MyFeature(window=20)
        for i in range(19):
            result = feature.update(i)
            assert not feature.is_ready()
        result = feature.update(20)
        assert feature.is_ready()

    def test_state_independence(self):
        """Multiple instances don't share state."""
        f1 = MyFeature()
        f2 = MyFeature()
        f1.update(100)
        f2.update(200)
        assert f1.get() != f2.get()
```

### 7.3 Performance Optimization Opportunities

**Premature Optimization Warning:** Profile first, then optimize.

**Quick Wins:**

1. **Deque maxlen:** Use `deque(maxlen=n)` for auto-eviction
2. **Numpy vectorization:** Batch warmup using numpy
3. **Lazy calculation:** Only compute when requested
4. **Caching:** Cache expensive calculations
5. **Numba JIT (later):** For hot loops

**Numba Example (Future Optimization):**
```python
from numba import jit

@jit(nopython=True)
def fast_rolling_mean(values, window_size):
    """Numba-accelerated rolling mean."""
    n = len(values)
    result = np.empty(n)

    for i in range(n):
        start = max(0, i - window_size + 1)
        result[i] = np.mean(values[start:i+1])

    return result
```

**Verdict:** Start simple (Python + deque), optimize later with Numba if needed.

---

## 8. Code Examples

### 8.1 Complete Feature Class Template

```python
from collections import deque
from typing import Optional, Any
import pandas as pd
import numpy as np

class StreamingFeature:
    """
    Base class for streaming features with batch/stream parity.

    Design Principles:
    1. Same code for backtest and live trading
    2. Stateful - maintains internal state across updates
    3. Warm-startable - can initialize from historical data
    4. Testable - pure functions, no side effects
    5. Debuggable - inspectable state at any point
    """

    def __init__(self, **kwargs):
        """
        Initialize feature with parameters.

        Subclasses should:
        1. Store configuration
        2. Initialize state containers (deques, Welford objects, etc.)
        3. Set warmup requirements
        """
        raise NotImplementedError

    def update(self, *args, **kwargs) -> Any:
        """
        Update feature state with new value(s) and return result.

        This is the PRIMARY method called during:
        - Backtesting (on each bar)
        - Live trading (on each new data point)

        Returns:
            Feature value (float, int, bool, etc.)

        Raises:
            ValueError: If inputs are invalid
        """
        raise NotImplementedError

    def compute(self, *args, **kwargs) -> Any:
        """
        Compute feature WITHOUT updating state (preview mode).

        Useful for:
        - Last tick price (unconfirmed bar)
        - What-if analysis
        - Debugging

        Returns:
            Feature value (same type as update())
        """
        raise NotImplementedError

    def is_ready(self) -> bool:
        """
        Check if feature has enough data to produce valid output.

        Returns:
            True if warmup period complete, False otherwise
        """
        raise NotImplementedError

    def warmup(self, data: pd.DataFrame) -> None:
        """
        Populate feature state with historical data (batch mode).

        This is equivalent to calling update() on each row,
        but may be optimized for batch processing.

        Args:
            data: Historical dataframe with required columns
        """
        for _, row in data.iterrows():
            self.update_from_row(row)

    def update_from_row(self, row: pd.Series) -> Any:
        """
        Helper to update from pandas Series (for warmup).

        Subclasses should override to extract relevant fields.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """
        Reset feature state to initial conditions.

        Critical for:
        - Testing (clean state between tests)
        - Multiple backtest runs
        - Walk-forward optimization
        """
        self.__init__(**self._init_kwargs)

    def get_state(self) -> dict:
        """
        Get current internal state (for debugging/inspection).

        Returns:
            Dict of state variables
        """
        raise NotImplementedError
```

### 8.2 Example: Complete Streaming RSI

```python
from collections import deque
import numpy as np

class StreamingRSI(StreamingFeature):
    """
    Relative Strength Index (RSI) with streaming updates.

    Uses Wilder's smoothing (RMA) for average gain/loss.
    """

    def __init__(self, period: int = 14):
        """
        Initialize RSI.

        Args:
            period: Lookback period (default 14)
        """
        self.period = period
        self._init_kwargs = {'period': period}

        # State
        self.prices = deque(maxlen=period + 1)  # Need n+1 for change calculation
        self.avg_gain = None
        self.avg_loss = None
        self._is_initialized = False

    def update(self, close_price: float) -> float:
        """
        Update RSI with new close price.

        Args:
            close_price: Current close price

        Returns:
            RSI value [0, 100], or 50.0 if not ready
        """
        self.prices.append(close_price)

        # Need at least 2 prices to calculate change
        if len(self.prices) < 2:
            return 50.0

        # Calculate price change
        change = self.prices[-1] - self.prices[-2]
        gain = max(change, 0)
        loss = max(-change, 0)

        # Initialize averages (first time)
        if not self._is_initialized:
            if len(self.prices) < self.period + 1:
                return 50.0  # Not ready yet

            # Calculate initial averages (SMA of first period)
            changes = [self.prices[i] - self.prices[i-1] for i in range(1, self.period + 1)]
            gains = [max(c, 0) for c in changes]
            losses = [max(-c, 0) for c in changes]

            self.avg_gain = sum(gains) / self.period
            self.avg_loss = sum(losses) / self.period
            self._is_initialized = True
        else:
            # Update averages (Wilder's smoothing)
            self.avg_gain = (self.avg_gain * (self.period - 1) + gain) / self.period
            self.avg_loss = (self.avg_loss * (self.period - 1) + loss) / self.period

        # Calculate RSI
        if self.avg_loss == 0:
            return 100.0

        rs = self.avg_gain / self.avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def compute(self, close_price: float) -> float:
        """Compute RSI without updating state (preview)."""
        if not self._is_initialized or len(self.prices) < 1:
            return 50.0

        # Calculate what RSI would be if we added this price
        prev_price = self.prices[-1]
        change = close_price - prev_price
        gain = max(change, 0)
        loss = max(-change, 0)

        # Temporary averages
        temp_avg_gain = (self.avg_gain * (self.period - 1) + gain) / self.period
        temp_avg_loss = (self.avg_loss * (self.period - 1) + loss) / self.period

        if temp_avg_loss == 0:
            return 100.0

        rs = temp_avg_gain / temp_avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def is_ready(self) -> bool:
        """RSI ready after period + 1 bars."""
        return self._is_initialized

    def update_from_row(self, row: pd.Series) -> float:
        """Update from pandas row."""
        return self.update(row['close'])

    def get_state(self) -> dict:
        """Get internal state for debugging."""
        return {
            'period': self.period,
            'prices_count': len(self.prices),
            'avg_gain': self.avg_gain,
            'avg_loss': self.avg_loss,
            'is_ready': self.is_ready()
        }
```

### 8.3 Integration Pattern: Runtime Enrichment Module

```python
# File: engine/features/streaming/liquidity_features.py

from typing import Dict, List
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class StreamingLiquidityFeatures:
    """
    Runtime enrichment for S1 liquidity features.

    Manages a collection of streaming features and applies them
    to incoming data (backtest or live).
    """

    def __init__(self, config: Dict):
        """
        Initialize streaming features from config.

        Args:
            config: Feature configuration dict
        """
        self.config = config

        # Initialize features
        self.features = {
            'liquidity_drain_pct': LiquidityDrainPct(
                window=config.get('drain_window', 168)
            ),
            'liquidity_persistence': LiquidityPersistence(
                window=config.get('persistence_window', 7),
                threshold=config.get('persistence_threshold', 0.30)
            ),
            'capitulation_depth': CapitulationDepth(
                window=config.get('depth_window', 720)
            ),
            'crisis_composite': CrisisComposite(
                vix_weight=config.get('vix_weight', 0.30),
                dxy_weight=config.get('dxy_weight', 0.25),
                hyg_weight=config.get('hyg_weight', 0.25),
                funding_weight=config.get('funding_weight', 0.20)
            )
        }

        self._update_count = 0

    def update_from_row(self, row: pd.Series) -> Dict[str, float]:
        """
        Update all features from a single row and return results.

        Args:
            row: Pandas Series with required fields

        Returns:
            Dict of feature_name -> value
        """
        results = {}

        # Update each feature
        results['liquidity_drain_pct'] = self.features['liquidity_drain_pct'].update(
            row.get('liquidity_score', 0.5)
        )

        results['liquidity_persistence'] = self.features['liquidity_persistence'].update(
            row.get('liquidity_score', 0.5)
        )

        results['capitulation_depth'] = self.features['capitulation_depth'].update(
            row.get('close', 0.0)
        )

        results['crisis_composite'] = self.features['crisis_composite'].update(
            vix=row.get('VIX', 20.0),
            dxy=row.get('DXY', 100.0),
            hyg=row.get('HYG', 80.0),
            funding=row.get('funding_rate', 0.0)
        )

        self._update_count += 1

        if self._update_count % 1000 == 0:
            logger.debug(f"Processed {self._update_count} updates")

        return results

    def enrich_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enrich entire dataframe with streaming features (batch warmup).

        Args:
            df: Input dataframe

        Returns:
            Dataframe with new feature columns added
        """
        logger.info(f"Enriching {len(df)} bars with streaming liquidity features")

        # Initialize result columns
        feature_columns = {name: [] for name in self.features.keys()}

        # Process each row
        for idx, row in df.iterrows():
            results = self.update_from_row(row)
            for name, value in results.items():
                feature_columns[name].append(value)

        # Add columns to dataframe
        for name, values in feature_columns.items():
            df[name] = values

        # Log readiness
        for name, feature in self.features.items():
            ready_count = sum(1 for f in [feature] if f.is_ready())
            logger.info(f"  {name}: ready = {feature.is_ready()}")

        return df

    def are_all_ready(self) -> bool:
        """Check if all features are ready."""
        return all(f.is_ready() for f in self.features.values())

    def get_warmup_bars_required(self) -> int:
        """Get maximum warmup period across all features."""
        # Return longest warmup needed
        return max([
            168,  # liquidity_drain_pct
            7,    # liquidity_persistence
            720,  # capitulation_depth
            720   # crisis_composite
        ])
```

---

## 9. Performance Optimization

### 9.1 Profiling First

**Don't optimize without data. Profile first:**

```python
import cProfile
import pstats

def profile_feature_enrichment():
    """Profile streaming feature performance."""

    # Load test data
    df = pd.read_parquet('data/features.parquet')

    # Profile
    profiler = cProfile.Profile()
    profiler.enable()

    # Run enrichment
    enricher = StreamingLiquidityFeatures(config={})
    df_enriched = enricher.enrich_dataframe(df)

    profiler.disable()

    # Print stats
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # Top 20 functions
```

### 9.2 Optimization Techniques

**1. Deque maxlen (Already Implemented)**
```python
# Good: Auto-eviction
window = deque(maxlen=20)

# Bad: Manual size management
window = deque()
if len(window) > 20:
    window.popleft()
```

**2. Lazy Calculation**
```python
class LazyFeature:
    def __init__(self):
        self._value = None
        self._dirty = True

    def update(self, x):
        self._input = x
        self._dirty = True

    def get(self):
        if self._dirty:
            self._value = self._expensive_calculation()
            self._dirty = False
        return self._value
```

**3. Numba JIT (Use Sparingly)**
```python
from numba import jit

@jit(nopython=True)
def calculate_rolling_mean_numba(values, window_size):
    """Numba-accelerated rolling mean."""
    n = len(values)
    result = np.zeros(n)

    for i in range(n):
        start = max(0, i - window_size + 1)
        window_sum = 0.0
        count = 0
        for j in range(start, i + 1):
            window_sum += values[j]
            count += 1
        result[i] = window_sum / count

    return result
```

**4. Batch Warmup Optimization**
```python
def warmup_fast(self, df: pd.DataFrame) -> None:
    """Fast batch warmup using numpy."""

    # Extract column as numpy array
    values = df['close'].values

    # Warmup in single vectorized operation
    if len(values) >= self.window_size:
        # Take first window_size values
        initial_window = values[:self.window_size]

        # Initialize with batch mean
        self._sum = initial_window.sum()
        self.window = deque(initial_window, maxlen=self.window_size)

        # Process remaining values normally
        for v in values[self.window_size:]:
            self.update(v)
```

---

## 10. Anti-Patterns to Avoid

### 10.1 Don't Do This

**Anti-Pattern 1: Separate Batch and Stream Code**
```python
# BAD: Two implementations
def calculate_feature_batch(df):
    return df['value'].rolling(20).mean()

def calculate_feature_stream(value, state):
    state.window.append(value)
    return sum(state.window) / len(state.window)
```

**Solution:** Single implementation, use for both
```python
# GOOD: One implementation
class Feature:
    def update(self, value):
        # Same code for batch and stream
        pass

# Batch mode
for value in df['value']:
    result = feature.update(value)

# Stream mode
result = feature.update(new_value)
```

**Anti-Pattern 2: Pandas in Live Trading**
```python
# BAD: Creating DataFrames in live loop
def on_new_bar(bar):
    df = pd.DataFrame([bar])
    feature = df['close'].rolling(20).mean().iloc[-1]
```

**Solution:** Streaming features
```python
# GOOD: Streaming calculation
def on_new_bar(bar):
    feature = rolling_mean.update(bar['close'])
```

**Anti-Pattern 3: Global State**
```python
# BAD: Global state
_window = []

def calculate_feature(value):
    global _window
    _window.append(value)
    return sum(_window) / len(_window)
```

**Solution:** Instance state
```python
# GOOD: Instance state
class Feature:
    def __init__(self):
        self.window = []

    def update(self, value):
        self.window.append(value)
        return sum(self.window) / len(self.window)
```

**Anti-Pattern 4: Look-Ahead Bias**
```python
# BAD: Using future data
def signal(df, i):
    future_close = df['close'].iloc[i+1]  # LOOK-AHEAD!
    return future_close > df['close'].iloc[i]
```

**Solution:** Only use current and past
```python
# GOOD: Only past data
def signal(current_close, prev_close):
    return current_close > prev_close
```

**Anti-Pattern 5: Ignoring Warmup**
```python
# BAD: No warmup handling
def calculate_rsi(close):
    return rsi_formula(close)  # Crashes if not enough data
```

**Solution:** Explicit warmup checks
```python
# GOOD: Warmup awareness
def calculate_rsi(self, close):
    if not self.is_ready():
        return 50.0  # Neutral during warmup
    return self._rsi_formula()
```

---

## 11. Final Recommendations

### 11.1 Immediate Actions (Priority 1)

1. **Install Dependencies**
   ```bash
   pip install welford hypothesis pytest
   ```

2. **Create Base Class**
   - Implement `StreamingFeature` base class
   - Add to `engine/features/streaming/base.py`

3. **Implement S1 Features**
   - `liquidity_drain_pct`: Use deque + running sum
   - `liquidity_persistence`: Use deque + running count
   - `capitulation_depth`: Use ascending minima algorithm
   - `crisis_composite`: Use Welford for z-scores

4. **Write Tests**
   - Unit tests for each feature
   - Property-based tests with Hypothesis
   - Batch/stream parity validation

### 11.2 Medium-Term (Priority 2)

1. **Create Streaming Feature Registry**
   - Central registry of all streaming features
   - Auto-warmup management
   - Dependency resolution

2. **Integrate with Backtest**
   - Add `use_streaming_features` flag
   - Warmup period handling
   - Performance monitoring

3. **Monitoring & Logging**
   - Track warmup completion
   - Log feature readiness
   - Performance metrics

### 11.3 Long-Term (Priority 3)

1. **Performance Optimization**
   - Profile hot paths
   - Numba JIT for bottlenecks
   - Vectorized warmup

2. **Advanced Features**
   - Rolling quantiles (for percentiles)
   - Exponential smoothing variants
   - Custom aggregations

3. **Production Hardening**
   - Circuit breakers for NaN cascades
   - Feature health checks
   - Automatic failover to batch calculation

---

## 12. Code Repository Structure

**Recommended File Organization:**

```
engine/
  features/
    streaming/
      __init__.py
      base.py                    # StreamingFeature base class
      statistics.py              # Welford, rolling stats
      windows.py                 # RollingWindow, RollingMax/Min
      liquidity.py               # S1 liquidity features
      registry.py                # Feature registry

tests/
  unit/
    test_streaming_base.py
    test_streaming_statistics.py
    test_liquidity_features.py

  integration/
    test_batch_stream_parity.py
    test_backtest_integration.py

  property/
    test_feature_properties.py  # Hypothesis tests

docs/
  STREAMING_FEATURES_GUIDE.md   # Usage documentation
  FEATURE_PARITY_TESTING.md     # Testing guidelines
```

---

## 13. References

### Libraries
- **Welford:** https://github.com/a-mitani/welford
- **streaming-indicators:** https://github.com/mr-easy/streaming_indicators
- **River:** https://github.com/online-ml/river
- **Hypothesis:** https://github.com/HypothesisWorks/hypothesis
- **rolling (Python):** https://github.com/ajcr/rolling

### Production Systems
- **QuantConnect LEAN:** https://www.lean.io/
- **NautilusTrader:** https://github.com/nautechsystems/nautilus_trader
- **Backtrader:** https://github.com/mementum/backtrader
- **VectorBT:** https://github.com/polakowo/vectorbt

### Academic Resources
- **Welford's Algorithm:** https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
- **Online Statistics:** https://www.johndcook.com/blog/standard_deviation/
- **Ascending Minima Algorithm:** Richard Harter's implementation

### Articles
- **Feature Store Patterns:** https://www.tecton.ai/blog/what-is-a-feature-store/
- **Streaming Data Quality:** https://www.globallogic.com/insights/white-papers/data-quality-solutions-for-stream-and-batch-data-processing/
- **Property-Based Testing:** https://hypothesis.readthedocs.io/

---

## Appendix A: Quick Reference

### Feature Implementation Checklist

For each new streaming feature:

- [ ] Inherit from `StreamingFeature`
- [ ] Implement `__init__` with configuration
- [ ] Implement `update()` method (main logic)
- [ ] Implement `compute()` method (preview mode)
- [ ] Implement `is_ready()` check
- [ ] Implement `reset()` method
- [ ] Document warmup period requirement
- [ ] Write unit tests (edge cases)
- [ ] Write property-based tests (parity)
- [ ] Add to feature registry
- [ ] Benchmark performance
- [ ] Document usage examples

### Testing Checklist

For each feature:

- [ ] Test empty input
- [ ] Test single value
- [ ] Test warmup period
- [ ] Test after warmup
- [ ] Test extreme values
- [ ] Test NaN handling
- [ ] Test state independence
- [ ] Test batch/stream equivalence
- [ ] Test numerical stability
- [ ] Test performance (< 1ms/update)

---

**Document Status:** COMPLETE
**Last Updated:** 2025-11-22
**Author:** Claude Code (Deep Research Agent)
**Next Steps:** Implement `StreamingFeature` base class and S1 liquidity features
