# Live-Ready Feature Engineering: Implementation Guide

**Quick Start Guide for Developers**

---

## Phase 1: Batch Mode S1 Features (Week 1)

### Step 1.1: Create Core Feature Logic

**File:** `engine/features/computer.py`

```bash
# Create file
touch engine/features/computer.py
```

**Implementation Checklist:**
- [ ] Define `FeatureComputer` abstract base class
- [ ] Implement `S1FeatureLogic` class with 7 pure functions
- [ ] Add docstrings with formula explanations
- [ ] No external dependencies (pandas, state, etc.) in logic class

**Validation:**
```python
# Quick test
from engine.features.computer import S1FeatureLogic
import numpy as np

logic = S1FeatureLogic()

# Test liquidity_drain_pct
liq_current = 0.20
liq_window = np.array([0.45, 0.42, 0.40])
result = logic.liquidity_drain_pct(liq_current, liq_window)

print(f"Expected: -0.53, Got: {result:.2f}")
# Should print: "Expected: -0.53, Got: -0.53"
```

---

### Step 1.2: Create Batch Engine

**File:** `engine/features/batch_engine.py`

```bash
touch engine/features/batch_engine.py
```

**Implementation Checklist:**
- [ ] `BatchFeatureEngine` class implementing `FeatureComputer`
- [ ] All 7 S1 features as vectorized pandas operations
- [ ] Input validation (check required columns)
- [ ] Delegates to `S1FeatureLogic` for formulas

**Testing:**
```bash
# Create test file
cat > test_batch_quick.py << 'EOF'
import pandas as pd
import numpy as np
from engine.features.batch_engine import BatchFeatureEngine

# Create sample data
np.random.seed(42)
df = pd.DataFrame({
    'timestamp': pd.date_range('2024-01-01', periods=1000, freq='1H'),
    'close': 40000 + np.cumsum(np.random.randn(1000) * 100),
    'liquidity_score': 0.5 + np.random.randn(1000) * 0.2,
    'volume_zscore': np.random.randn(1000),
    'wick_lower_ratio': np.random.rand(1000) * 0.3,
    'rv_z': np.random.randn(1000),
    'funding_Z': np.random.randn(1000),
    'return_z': np.random.randn(1000),
})
df = df.set_index('timestamp')

# Test batch engine
engine = BatchFeatureEngine()
df_enriched = engine.compute_features(df)

# Check features added
s1_features = [
    'liquidity_drain_pct',
    'liquidity_velocity',
    'liquidity_persistence',
    'capitulation_depth',
    'crisis_composite',
    'volume_climax_last_3b',
    'wick_exhaustion_last_3b',
]

for feat in s1_features:
    if feat in df_enriched.columns:
        print(f"✓ {feat}")
    else:
        print(f"✗ {feat} MISSING!")

print(f"\nShape: {df_enriched.shape}")
print(f"New columns: {len(df_enriched.columns) - len(df.columns)}")
EOF

python3 test_batch_quick.py
```

**Expected Output:**
```
✓ liquidity_drain_pct
✓ liquidity_velocity
✓ liquidity_persistence
✓ capitulation_depth
✓ crisis_composite
✓ volume_climax_last_3b
✓ wick_exhaustion_last_3b

Shape: (1000, 15)
New columns: 7
```

---

### Step 1.3: Integrate with FeatureStoreBuilder

**File:** `engine/features/builder.py` (modify existing)

**Changes:**
```python
# Find _build_tier2() method (around line 214)
# Add S1 enrichment AFTER existing MTF logic

def _build_tier2(self, df: pd.DataFrame, spec: BuildSpec) -> pd.DataFrame:
    """Build Tier 2: Multi-timeframe features + S1 runtime features."""

    # Existing MTF logic
    if mtf_path.exists():
        df_mtf = pd.read_parquet(mtf_path)
    else:
        df_mtf = self._build_mtf_simple(df, spec)

    # NEW: Add S1 runtime features using batch engine
    from engine.features.batch_engine import BatchFeatureEngine

    print("\n▶ Stage 2b: Adding S1 runtime features...")
    s1_engine = BatchFeatureEngine()
    df_enriched = s1_engine.compute_features(df_mtf)
    print(f"  ✓ S1 features: 7 columns added")

    return df_enriched
```

**Testing:**
```bash
# Test feature store build with S1 enrichment
python3 -c "
from engine.features.builder import FeatureStoreBuilder, BuildSpec

builder = FeatureStoreBuilder()
spec = BuildSpec(
    asset='BTC',
    start='2024-01-01',
    end='2024-01-31',
    tiers=[1, 2],
    validate=False
)

df, report = builder.build(spec)

# Check S1 features present
s1_features = [
    'liquidity_drain_pct',
    'liquidity_persistence',
    'capitulation_depth',
]

for feat in s1_features:
    if feat in df.columns:
        print(f'✓ {feat}')
    else:
        print(f'✗ {feat} MISSING')
"
```

---

### Step 1.4: Unit Tests for S1FeatureLogic

**File:** `tests/unit/features/test_s1_logic.py`

```bash
mkdir -p tests/unit/features
touch tests/unit/features/__init__.py
cat > tests/unit/features/test_s1_logic.py << 'EOF'
"""Unit tests for S1FeatureLogic pure functions."""

import pytest
import numpy as np
from engine.features.computer import S1FeatureLogic


class TestS1FeatureLogic:
    """Test S1 feature computation logic."""

    def setup_method(self):
        """Create S1FeatureLogic instance for tests."""
        self.logic = S1FeatureLogic()

    def test_liquidity_drain_pct_normal_drain(self):
        """Test normal liquidity drain scenario."""
        liq_current = 0.20
        liq_window = np.array([0.45, 0.42, 0.40, 0.38, 0.35])

        result = self.logic.liquidity_drain_pct(liq_current, liq_window)

        # Expected: (0.20 - 0.40) / 0.40 = -0.50
        expected = -0.50
        assert abs(result - expected) < 0.01

    def test_liquidity_drain_pct_recovery(self):
        """Test liquidity recovery (positive drain)."""
        liq_current = 0.60
        liq_window = np.array([0.40, 0.40, 0.40])

        result = self.logic.liquidity_drain_pct(liq_current, liq_window)

        # Expected: (0.60 - 0.40) / 0.40 = +0.50
        expected = 0.50
        assert abs(result - expected) < 0.01

    def test_liquidity_drain_pct_empty_window(self):
        """Test edge case: empty window."""
        result = self.logic.liquidity_drain_pct(0.20, np.array([]))
        assert result == 0.0

    def test_liquidity_drain_pct_zero_avg(self):
        """Test edge case: zero average (div by zero)."""
        liq_current = 0.20
        liq_window = np.array([0.0, 0.0, 0.0])

        result = self.logic.liquidity_drain_pct(liq_current, liq_window)
        assert result == 0.0  # Should handle gracefully

    def test_liquidity_velocity_draining(self):
        """Test liquidity velocity during drain."""
        result = self.logic.liquidity_velocity(
            liq_current=0.30,
            liq_prev=0.50,
            dt_hours=1.0
        )

        # Expected: (0.30 - 0.50) / 1.0 = -0.20
        expected = -0.20
        assert abs(result - expected) < 0.01

    def test_liquidity_velocity_recovering(self):
        """Test liquidity velocity during recovery."""
        result = self.logic.liquidity_velocity(
            liq_current=0.60,
            liq_prev=0.40,
            dt_hours=2.0
        )

        # Expected: (0.60 - 0.40) / 2.0 = +0.10
        expected = 0.10
        assert abs(result - expected) < 0.01

    def test_liquidity_persistence_all_draining(self):
        """Test persistence when all bars draining."""
        liq_drain_history = np.array([-0.4, -0.5, -0.35, -0.42, -0.38])

        result = self.logic.liquidity_persistence(
            liq_drain_history,
            drain_threshold=-0.3,
            lookback=10
        )

        # Expected: 5 bars (all below -0.3)
        assert result == 5

    def test_liquidity_persistence_mixed(self):
        """Test persistence with mixed draining/normal bars."""
        liq_drain_history = np.array([-0.1, -0.4, -0.5, -0.2, 0.1, -0.35])

        result = self.logic.liquidity_persistence(
            liq_drain_history,
            drain_threshold=-0.3
        )

        # Expected: 3 bars (-0.4, -0.5, -0.35)
        assert result == 3

    def test_capitulation_depth(self):
        """Test capitulation depth calculation."""
        price_current = 18000.0
        price_window = np.array([25000, 24000, 23000, 22000, 20000, 19000])

        result = self.logic.capitulation_depth(price_current, price_window, lookback=10)

        # Expected: (18000 - 25000) / 25000 = -0.28
        expected = -0.28
        assert abs(result - expected) < 0.01

    def test_capitulation_depth_at_high(self):
        """Test depth when price at high (no capitulation)."""
        price_current = 50000.0
        price_window = np.array([40000, 42000, 45000, 48000])

        result = self.logic.capitulation_depth(price_current, price_window)

        # Expected: (50000 - 50000) / 50000 = 0.0
        assert abs(result) < 0.01

    def test_crisis_composite_default_weights(self):
        """Test crisis composite with default weights."""
        result = self.logic.crisis_composite(
            rv_z=2.0,
            funding_z=1.5,
            return_z=-1.0
        )

        # Expected: 0.4*2.0 + 0.3*1.5 + 0.3*(-1.0) = 1.05
        expected = 1.05
        assert abs(result - expected) < 0.01

    def test_crisis_composite_custom_weights(self):
        """Test crisis composite with custom weights."""
        weights = {'rv': 0.5, 'funding': 0.3, 'return': 0.2}

        result = self.logic.crisis_composite(
            rv_z=2.0,
            funding_z=1.0,
            return_z=-1.0,
            weights=weights
        )

        # Expected: 0.5*2.0 + 0.3*1.0 + 0.2*(-1.0) = 1.1
        expected = 1.1
        assert abs(result - expected) < 0.01

    def test_volume_climax_last_3b(self):
        """Test volume climax detection."""
        volume_z_window = np.array([0.5, 2.5, 1.0, 0.8, 3.0, 1.5])

        result = self.logic.volume_climax_last_3b(volume_z_window)

        # Expected: max of last 3 bars = max(0.8, 3.0, 1.5) = 3.0
        expected = 3.0
        assert abs(result - expected) < 0.01

    def test_wick_exhaustion_last_3b(self):
        """Test wick exhaustion detection."""
        wick_lower_window = np.array([0.1, 0.2, 0.5, 0.3, 0.15])

        result = self.logic.wick_exhaustion_last_3b(wick_lower_window)

        # Expected: max of last 3 bars = max(0.5, 0.3, 0.15) = 0.5
        expected = 0.5
        assert abs(result - expected) < 0.01


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
EOF
```

**Run Tests:**
```bash
pytest tests/unit/features/test_s1_logic.py -v
```

**Expected Output:**
```
tests/unit/features/test_s1_logic.py::TestS1FeatureLogic::test_liquidity_drain_pct_normal_drain PASSED
tests/unit/features/test_s1_logic.py::TestS1FeatureLogic::test_liquidity_drain_pct_recovery PASSED
tests/unit/features/test_s1_logic.py::TestS1FeatureLogic::test_liquidity_drain_pct_empty_window PASSED
...
============================== 13 passed in 0.12s ===============================
```

---

### Step 1.5: Run Backtest with S1 Features

**Command:**
```bash
python3 bin/backtest_knowledge_v2.py \
  --config configs/mvp/mvp_bear_market_v1.json \
  --start 2022-01-01 \
  --end 2022-12-31
```

**Check Output:**
```
Feature Store Build:
  ✓ Tier 1: 8760 rows, 42 columns
  ✓ Tier 2: 8760 rows, 78 columns
  ✓ Stage 2b: Adding S1 runtime features...
  ✓ S1 features: 7 columns added

Backtest Results:
  Trades: 12
  PF: 2.34
  ...
```

**Validation Queries:**
```python
# After backtest, check feature distribution
import pandas as pd

df = pd.read_parquet('data/features_mtf/BTC_1H_2022-01-01_to_2022-12-31.parquet')

# Check S1 feature statistics
print("\nS1 Feature Statistics:")
print(df[['liquidity_drain_pct', 'liquidity_persistence', 'capitulation_depth']].describe())

# Check extreme values
print("\nExtreme Liquidity Drains (p99):")
threshold = df['liquidity_drain_pct'].quantile(0.01)  # 1st percentile (most negative)
extreme_drains = df[df['liquidity_drain_pct'] < threshold]
print(f"Found {len(extreme_drains)} extreme drain events")
print(extreme_drains[['liquidity_drain_pct', 'liquidity_persistence', 'capitulation_depth']].head())
```

---

## Phase 2: Streaming Feature Engine (Week 2)

### Step 2.1: Create RollingWindow Class

**File:** `engine/features/stream_engine.py`

```bash
touch engine/features/stream_engine.py
```

**Implementation Order:**
1. Start with `RollingWindow` class (circular buffer)
2. Test it in isolation before integrating

**Quick Test:**
```python
# Test RollingWindow
cat > test_rolling_window.py << 'EOF'
from engine.features.stream_engine import RollingWindow
import numpy as np

# Create window
window = RollingWindow(max_size=5)

# Add values
for val in [1, 2, 3, 4, 5]:
    window.append(val)

print(f"Full window: {list(window.buffer)}")
print(f"Mean: {window.mean()}")
print(f"Max: {window.max()}")

# Add more (should evict oldest)
window.append(6)
window.append(7)

print(f"After overflow: {list(window.buffer)}")  # Should be [3, 4, 5, 6, 7]
print(f"Mean: {window.mean()}")

# Test count_below
window_test = RollingWindow(max_size=10)
for val in [-0.5, -0.2, -0.4, 0.1, -0.3]:
    window_test.append(val)

count = window_test.count_below(-0.3)
print(f"Count below -0.3: {count}")  # Should be 2 (-0.5, -0.4)
EOF

python3 test_rolling_window.py
```

---

### Step 2.2: Create StreamFeatureEngine

**Implementation Checklist:**
- [ ] `StreamFeatureEngine` class implementing `FeatureComputer`
- [ ] Initialize all rolling windows in `__init__`
- [ ] `compute_features(bar: dict) -> dict` method
- [ ] `get_state()` and `load_state()` for crash recovery
- [ ] Delegates to `S1FeatureLogic` (same as batch!)

**Quick Test:**
```python
cat > test_stream_engine.py << 'EOF'
from engine.features.stream_engine import StreamFeatureEngine
import numpy as np

# Create engine
engine = StreamFeatureEngine()

# Process 200 bars to warm up windows
np.random.seed(42)
for i in range(200):
    bar = {
        'close': 40000 + i * 10,
        'liquidity_score': 0.5 + np.random.randn() * 0.1,
        'volume_zscore': np.random.randn(),
        'wick_lower_ratio': np.random.rand() * 0.3,
        'rv_z': np.random.randn(),
        'funding_Z': np.random.randn(),
        'return_z': np.random.randn(),
    }
    features = engine.compute_features(bar)

# Check last result
print("Features computed:")
for key, val in features.items():
    print(f"  {key}: {val:.4f}")

# Test state save/load
state = engine.get_state()
print(f"\nState saved: {len(state)} keys")
print(f"Bar count: {state['bar_count']}")

# Create new engine and restore
engine2 = StreamFeatureEngine()
engine2.load_state(state)

# Process same bar
test_bar = {
    'close': 42000,
    'liquidity_score': 0.25,
    'volume_zscore': 2.0,
    'wick_lower_ratio': 0.4,
    'rv_z': 1.5,
    'funding_Z': -0.5,
    'return_z': -1.0,
}

features1 = engine.compute_features(test_bar)
features2 = engine2.compute_features(test_bar)

print("\nState restore validation:")
for key in features1.keys():
    diff = abs(features1[key] - features2[key])
    status = "✓" if diff < 1e-6 else "✗"
    print(f"  {status} {key}: diff={diff:.9f}")
EOF

python3 test_stream_engine.py
```

---

### Step 2.3: Batch vs Stream Parity Test

**File:** `tests/integration/test_batch_stream_parity.py`

```bash
mkdir -p tests/integration
touch tests/integration/__init__.py
cat > tests/integration/test_batch_stream_parity.py << 'EOF'
"""Integration tests verifying batch and stream engines produce identical features."""

import pytest
import pandas as pd
import numpy as np
from engine.features.batch_engine import BatchFeatureEngine
from engine.features.stream_engine import StreamFeatureEngine


class TestBatchStreamParity:
    """Test batch and stream engines produce identical results."""

    def test_s1_features_parity(self):
        """Verify batch and stream produce identical S1 features."""
        # Create sample data
        np.random.seed(42)
        n_bars = 500

        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=n_bars, freq='1H'),
            'close': 40000 + np.cumsum(np.random.randn(n_bars) * 100),
            'liquidity_score': 0.5 + np.random.randn(n_bars) * 0.2,
            'volume_zscore': np.random.randn(n_bars),
            'wick_lower_ratio': np.random.rand(n_bars) * 0.3,
            'rv_z': np.random.randn(n_bars),
            'funding_Z': np.random.randn(n_bars),
            'return_z': np.random.randn(n_bars),
        })
        df = df.set_index('timestamp')

        # Batch mode
        batch_engine = BatchFeatureEngine()
        df_batch = batch_engine.compute_features(df.copy())

        # Stream mode (process bar by bar)
        stream_engine = StreamFeatureEngine()
        stream_results = []

        for idx, row in df.iterrows():
            bar = row.to_dict()
            bar['close'] = row['close']
            features = stream_engine.compute_features(bar)
            stream_results.append(features)

        df_stream = pd.DataFrame(stream_results, index=df.index)

        # Compare S1 features
        s1_features = [
            'liquidity_drain_pct',
            'liquidity_velocity',
            'liquidity_persistence',
            'capitulation_depth',
            'crisis_composite',
            'volume_climax_last_3b',
            'wick_exhaustion_last_3b',
        ]

        print("\nBatch vs Stream Parity Check:")
        for feature in s1_features:
            # Allow small numerical differences due to floating point
            batch_values = df_batch[feature].values
            stream_values = df_stream[feature].values

            # Check correlation (should be ~1.0)
            correlation = np.corrcoef(batch_values, stream_values)[0, 1]

            # Check max absolute difference
            max_diff = np.max(np.abs(batch_values - stream_values))

            print(f"  {feature}:")
            print(f"    Correlation: {correlation:.6f}")
            print(f"    Max diff: {max_diff:.9f}")

            assert correlation > 0.999, f"{feature} correlation too low: {correlation}"
            assert max_diff < 0.01, f"{feature} max diff too large: {max_diff}"

        print("\n✓ All features match!")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
EOF
```

**Run Test:**
```bash
pytest tests/integration/test_batch_stream_parity.py -v -s
```

**Expected Output:**
```
Batch vs Stream Parity Check:
  liquidity_drain_pct:
    Correlation: 0.999998
    Max diff: 0.000000123
  liquidity_velocity:
    Correlation: 1.000000
    Max diff: 0.000000000
  ...

✓ All features match!

============================== 1 passed in 2.34s ================================
```

---

## Phase 3: Stateful Archetypes (Week 3)

### Step 3.1: Create S1 State Machine

**File:** `engine/archetypes/state_machines/s1_state_machine.py`

```bash
mkdir -p engine/archetypes/state_machines
touch engine/archetypes/state_machines/__init__.py
cat > engine/archetypes/state_machines/s1_state_machine.py << 'EOF'
"""S1 Liquidity Vacuum State Machine."""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple


class S1State(Enum):
    """S1 Liquidity Vacuum states."""
    WATCHING = "watching"
    DRAINING = "draining"
    SIGNAL = "signal"


@dataclass
class S1StateMachine:
    """State machine for S1 Liquidity Vacuum archetype."""

    state: S1State = S1State.WATCHING
    bars_draining: int = 0
    drain_start_price: Optional[float] = None

    def update(
        self,
        liquidity_drain_pct: float,
        close: float,
        persistence_threshold: int = 8
    ) -> Tuple['S1StateMachine', bool]:
        """
        Update state machine with new bar data.

        Returns:
            (new_state_machine, emit_signal)
        """
        emit_signal = False
        new_state = self.state
        new_bars_draining = self.bars_draining
        new_drain_start_price = self.drain_start_price

        if self.state == S1State.WATCHING:
            # Transition to DRAINING if liquidity drops
            if liquidity_drain_pct < -0.3:
                new_state = S1State.DRAINING
                new_bars_draining = 1
                new_drain_start_price = close

        elif self.state == S1State.DRAINING:
            # Still draining?
            if liquidity_drain_pct < -0.3:
                new_bars_draining += 1

                # Check if threshold met
                if new_bars_draining >= persistence_threshold:
                    new_state = S1State.SIGNAL
            else:
                # Liquidity recovered, reset
                new_state = S1State.WATCHING
                new_bars_draining = 0
                new_drain_start_price = None

        elif self.state == S1State.SIGNAL:
            # Emit signal and reset
            emit_signal = True
            new_state = S1State.WATCHING
            new_bars_draining = 0
            new_drain_start_price = None

        # Create new immutable state machine
        new_sm = S1StateMachine(
            state=new_state,
            bars_draining=new_bars_draining,
            drain_start_price=new_drain_start_price
        )

        return new_sm, emit_signal

    def to_dict(self) -> dict:
        """Serialize for crash recovery."""
        return {
            'state': self.state.value,
            'bars_draining': self.bars_draining,
            'drain_start_price': self.drain_start_price,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'S1StateMachine':
        """Deserialize from crash recovery."""
        return cls(
            state=S1State(data['state']),
            bars_draining=data['bars_draining'],
            drain_start_price=data['drain_start_price'],
        )
EOF
```

**Test State Machine:**
```python
cat > test_s1_state_machine.py << 'EOF'
from engine.archetypes.state_machines.s1_state_machine import S1StateMachine, S1State

# Create state machine
sm = S1StateMachine()

print(f"Initial state: {sm.state.value}")

# Simulate liquidity drain
bars = [
    ('2024-01-01 10:00', -0.10, 42000),  # Normal
    ('2024-01-01 11:00', -0.35, 41800),  # Drain starts
    ('2024-01-01 12:00', -0.40, 41500),
    ('2024-01-01 13:00', -0.38, 41300),
    ('2024-01-01 14:00', -0.42, 41100),
    ('2024-01-01 15:00', -0.45, 40900),
    ('2024-01-01 16:00', -0.43, 40700),
    ('2024-01-01 17:00', -0.41, 40500),
    ('2024-01-01 18:00', -0.39, 40300),  # 8th draining bar
    ('2024-01-01 19:00', -0.25, 40800),  # Recovery
]

for ts, liq_drain, close in bars:
    sm, emit = sm.update(liq_drain, close, persistence_threshold=8)
    print(f"{ts}: liq_drain={liq_drain:+.2f}, state={sm.state.value}, "
          f"bars={sm.bars_draining}, emit={emit}")

# Test serialization
state_dict = sm.to_dict()
print(f"\nSerialized: {state_dict}")

sm_restored = S1StateMachine.from_dict(state_dict)
print(f"Restored: state={sm_restored.state.value}, bars={sm_restored.bars_draining}")
EOF

python3 test_s1_state_machine.py
```

---

## Phase 4: Live Deployment (Week 4)

### Step 4.1: Create Live Trader Script

**File:** `bin/live_trader.py`

```bash
cat > bin/live_trader.py << 'EOF'
#!/usr/bin/env python3
"""
Live trader using StreamFeatureEngine and stateful archetypes.

Usage:
    python3 bin/live_trader.py --config configs/live_production.json
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import time
import logging
from datetime import datetime

from engine.features.stream_engine import StreamFeatureEngine
from engine.features.state_persistence import StatePersistenceStore
from engine.archetypes.state_machines.s1_state_machine import S1StateMachine

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


class LiveTrader:
    """Live trading engine using streaming features."""

    def __init__(self, config_path: str):
        """Initialize live trader."""
        with open(config_path) as f:
            self.config = json.load(f)

        self.symbol = self.config.get('symbol', 'BTC')

        # Initialize streaming feature engine
        self.feature_engine = StreamFeatureEngine()

        # Initialize state machine
        self.s1_state = S1StateMachine()

        # State persistence
        self.state_store = StatePersistenceStore('data/live_state')

        # Try to restore from crash
        self._restore_state()

        logger.info(f"Live trader initialized for {self.symbol}")

    def _restore_state(self):
        """Restore state from previous session."""
        if not self.state_store.exists(self.symbol):
            logger.info("No previous state found, starting fresh")
            return

        state = self.state_store.load(self.symbol)

        # Restore feature engine
        self.feature_engine.load_state(state['feature_engine'])

        # Restore S1 state machine
        self.s1_state = S1StateMachine.from_dict(state['s1_state'])

        logger.info(
            f"State restored: bar_count={self.feature_engine.bar_count}, "
            f"s1_state={self.s1_state.state.value}"
        )

    def _save_state(self):
        """Save state for crash recovery."""
        state = {
            'feature_engine': self.feature_engine.get_state(),
            's1_state': self.s1_state.to_dict(),
            'last_update': datetime.now().isoformat(),
        }

        self.state_store.save(self.symbol, state)

    def process_bar(self, bar: dict):
        """Process new bar from API."""
        # Compute features
        start_time = time.perf_counter()
        features = self.feature_engine.compute_features(bar)
        latency_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            f"Bar {self.feature_engine.bar_count}: "
            f"close={bar['close']:.0f}, "
            f"liq_drain={features['liquidity_drain_pct']:+.3f}, "
            f"persist={features['liquidity_persistence']}, "
            f"latency={latency_ms:.2f}ms"
        )

        # Update S1 state machine
        self.s1_state, emit_signal = self.s1_state.update(
            features['liquidity_drain_pct'],
            bar['close'],
            persistence_threshold=8
        )

        if emit_signal:
            logger.info(
                f"🚨 S1 SIGNAL! Entry={bar['close']:.0f}, "
                f"persistence={features['liquidity_persistence']} bars"
            )
            # TODO: Execute trade

        # Save state every 100 bars
        if self.feature_engine.bar_count % 100 == 0:
            self._save_state()
            logger.info(f"State saved (bar {self.feature_engine.bar_count})")

    def run(self):
        """Main trading loop."""
        logger.info("Starting live trading loop...")

        # TODO: Connect to OKX WebSocket
        # For now, simulate with historical data
        import pandas as pd

        df = pd.read_parquet('data/features_mtf/BTC_1H_2024-01-01_to_2024-12-31.parquet')

        for idx, row in df.iterrows():
            bar = {
                'timestamp': idx,
                'close': row['close'],
                'liquidity_score': row.get('liquidity_score', 0.5),
                'volume_zscore': row.get('volume_zscore', 0.0),
                'wick_lower_ratio': row.get('wick_lower_ratio', 0.0),
                'rv_z': row.get('rv_z', 0.0),
                'funding_Z': row.get('funding_Z', 0.0),
                'return_z': row.get('return_z', 0.0),
            }

            self.process_bar(bar)

            # Simulate real-time (sleep 0.1s between bars)
            # In production, this would be event-driven
            time.sleep(0.1)

        # Final state save
        self._save_state()
        logger.info("Trading loop completed")


def main():
    parser = argparse.ArgumentParser(description='Live trader')
    parser.add_argument('--config', required=True, help='Config file path')
    args = parser.parse_args()

    trader = LiveTrader(args.config)
    trader.run()


if __name__ == '__main__':
    main()
EOF

chmod +x bin/live_trader.py
```

**Test Live Trader:**
```bash
# Create test config
cat > configs/live_test.json << 'EOF'
{
  "symbol": "BTC",
  "exchange": "okx",
  "s1_thresholds": {
    "liquidity_drain_threshold": -0.3,
    "persistence_threshold": 8
  }
}
EOF

# Run live trader (simulated)
python3 bin/live_trader.py --config configs/live_test.json
```

---

## Common Issues & Solutions

### Issue 1: Batch vs Stream Divergence

**Symptom:**
```
DIVERGENCE at bar 168!
  Feature: liquidity_drain_pct
  Batch: -0.3524
  Stream: -0.3500
```

**Diagnosis:**
Batch is including current bar in rolling window (lookahead).

**Fix:**
```python
# WRONG (includes current bar)
liq_7d_avg = liq.rolling(168).mean()

# CORRECT (excludes current bar)
liq_7d_avg = liq.rolling(168).mean().shift(1)
```

---

### Issue 2: State Corruption After Crash

**Symptom:**
```
ValueError: symbol mismatch: expected BTC, got ETH
```

**Diagnosis:**
Wrong state file loaded.

**Fix:**
```python
# Add validation to state_persistence.py
def load(self, symbol: str):
    state = json.load(f)

    # Validate symbol matches
    if state['_metadata']['symbol'] != symbol:
        raise ValueError(f"Symbol mismatch")

    return state
```

---

### Issue 3: Slow Stream Updates (>100ms)

**Symptom:**
```
p99 latency: 250.4 ms
```

**Diagnosis:**
RollingWindow.to_array() called too frequently.

**Fix:**
```python
# WRONG (converts to array every time)
def mean(self):
    return np.mean(self.to_array())

# CORRECT (cache if needed, or use deque directly)
def mean(self):
    if len(self.buffer) == 0:
        return 0.0
    return sum(self.buffer) / len(self.buffer)  # Faster than np.mean
```

---

## Verification Checklist

### Phase 1 Complete:
- [ ] S1FeatureLogic passes all unit tests
- [ ] BatchFeatureEngine adds 7 S1 features
- [ ] FeatureStoreBuilder integration works
- [ ] Backtest runs with S1 features
- [ ] No performance regression (< 5% slower)

### Phase 2 Complete:
- [ ] RollingWindow tests pass
- [ ] StreamFeatureEngine processes bars < 1ms
- [ ] Batch vs stream parity test passes
- [ ] State save/load works correctly
- [ ] No state corruption after 1000+ bars

### Phase 3 Complete:
- [ ] S1StateMachine transitions correctly
- [ ] State persistence across restarts
- [ ] Backtest results identical to stateless version
- [ ] State machine unit tests pass

### Phase 4 Complete:
- [ ] Live trader runs without crashes
- [ ] State recovers after manual kill
- [ ] Latency < 100ms p99
- [ ] Signals match backtest replay

---

## Next Steps

1. **Review this guide** with team
2. **Start Phase 1** (batch mode S1 features)
3. **Set up CI/CD** for parity tests
4. **Plan Phase 2** kickoff after Phase 1 validation
