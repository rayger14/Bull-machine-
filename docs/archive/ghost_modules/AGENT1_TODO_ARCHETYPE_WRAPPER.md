# Agent 1 TODO: ArchetypeModel Wrapper Implementation

**Status:** REQUIRED before full model comparison can run
**Priority:** HIGH
**Estimated Effort:** 2-4 hours

---

## Overview

Agent 2 has completed the baseline model comparison framework. The next step requires Agent 1 to implement a wrapper around the existing archetype logic to make it compatible with the new backtesting framework.

---

## What Agent 2 Built

✅ **Completed:**
- Model comparison framework (`engine/backtesting/comparison.py`)
- Baseline models (`engine/models/simple_classifier.py`)
- Comparison script (`examples/baseline_vs_archetype_comparison.py`)
- Baseline performance benchmarks

✅ **Results:**
- Baseline-Conservative: Test PF 3.17 (42.9% WR, 7 trades)
- Baseline-Aggressive: Test PF 2.10 (33.3% WR, 36 trades)

⏳ **Blocked:** Need ArchetypeModel wrapper to compare against archetype strategies

---

## Agent 1 Task: Implement ArchetypeModel Wrapper

### File to Create
**Path:** `/Users/raymondghandchi/Bull-machine-/Bull-machine-/engine/models/archetype_model.py`

### Requirements

The wrapper must implement the `BaseModel` interface to be compatible with the backtesting framework.

### Implementation Template

```python
"""
ArchetypeModel wrapper for logic_v2_adapter.

This adapter bridges the gap between the existing archetype logic
and the new backtesting framework's BaseModel interface.
"""

import pandas as pd
import json
from typing import Optional
from pathlib import Path

from engine.models.base import BaseModel, Signal, Position
# Import your existing archetype logic
# from engine.archetypes.logic_v2_adapter import LogicV2Adapter
# (adjust import based on actual structure)


class ArchetypeModel(BaseModel):
    """
    Wrapper around archetype logic for backtesting framework compatibility.

    This class makes archetype-based strategies compatible with the
    model comparison framework while preserving existing logic.

    Example:
        >>> model = ArchetypeModel(
        ...     config_path='configs/s1_v2_production.json',
        ...     name='S1-LiquidityVacuum'
        ... )
        >>> signal = model.predict(bar, position)
    """

    def __init__(self, config_path: str, name: str):
        """
        Initialize archetype model with config.

        Args:
            config_path: Path to archetype config JSON (e.g., 'configs/s1_v2_production.json')
            name: Model name for comparison reports (e.g., 'S1-LiquidityVacuum')
        """
        super().__init__(name=name)
        self.config_path = config_path

        # Load config
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # Initialize archetype logic adapter
        # TODO: Replace with actual logic_v2_adapter initialization
        # self.logic = LogicV2Adapter(config=self.config)

        self._is_fitted = True  # Archetypes use pre-optimized configs

    def fit(self, train_data: pd.DataFrame) -> None:
        """
        Optional: Fine-tune archetype parameters on training data.

        For now, archetypes use pre-optimized configs, so this is a no-op.
        Future: Could implement online learning or parameter adaptation.

        Args:
            train_data: Historical data for calibration
        """
        # No-op for now (configs are pre-optimized)
        self._is_fitted = True

    def predict(self, bar: pd.Series, position: Optional[Position] = None) -> Signal:
        """
        Generate signal using archetype logic.

        This method:
        1. Calls the existing archetype logic (logic_v2_adapter)
        2. Converts the result to a Signal object
        3. Returns signal compatible with backtesting framework

        Args:
            bar: Current bar data (pd.Series with OHLCV + features)
            position: Current open position (if any)

        Returns:
            Signal with direction, confidence, stop loss, metadata
        """
        # TODO: Replace with actual logic_v2_adapter call
        # Example pseudocode:

        # archetype_signal = self.logic.get_signal(bar)

        # if archetype_signal is None or archetype_signal.direction == 'hold':
        #     return Signal(
        #         direction='hold',
        #         confidence=0.0,
        #         entry_price=bar['close']
        #     )

        # Convert archetype signal to framework Signal
        # return Signal(
        #     direction=archetype_signal.direction,  # 'long' or 'short'
        #     confidence=archetype_signal.fusion_score,  # 0.0-1.0
        #     entry_price=bar['close'],
        #     stop_loss=archetype_signal.stop_loss,
        #     metadata={
        #         'archetype': archetype_signal.archetype_name,
        #         'fusion_score': archetype_signal.fusion_score,
        #         'archetype_confidence': archetype_signal.archetype_confidence
        #     }
        # )

        # Placeholder for now
        return Signal(
            direction='hold',
            confidence=0.0,
            entry_price=bar['close']
        )

    def get_params(self) -> dict:
        """
        Get model parameters.

        Returns:
            Dictionary of model configuration
        """
        return {
            'config_path': self.config_path,
            'config': self.config
        }
```

---

## Integration Steps

### 1. Create the wrapper
- Implement `engine/models/archetype_model.py` (template above)
- Import existing archetype logic (`logic_v2_adapter` or similar)
- Ensure it implements `BaseModel` interface

### 2. Update model exports
Edit `/Users/raymondghandchi/Bull-machine-/Bull-machine-/engine/models/__init__.py`:

```python
from .base import BaseModel, Signal, Position
from .simple_classifier import BuyHoldSellClassifier
from .archetype_model import ArchetypeModel  # Add this

__all__ = [
    'BaseModel',
    'Signal',
    'Position',
    'BuyHoldSellClassifier',
    'ArchetypeModel',  # Add this
]
```

### 3. Test the wrapper
```bash
# Quick test script
python3 -c "
from engine.models import ArchetypeModel
import pandas as pd

# Test S1 config
s1 = ArchetypeModel('configs/s1_v2_production.json', 'S1-Test')
print(f'Loaded: {s1.name}')

# Test signal generation with dummy bar
bar = pd.Series({'close': 50000, 'atr_14': 500})
signal = s1.predict(bar)
print(f'Signal: {signal.direction}')
"
```

### 4. Run full comparison
Uncomment archetype lines in `/Users/raymondghandchi/Bull-machine-/Bull-machine-/examples/baseline_vs_archetype_comparison.py`:

```python
# Lines 60-69: Uncomment these
archetype_s1 = ArchetypeModel(
    config_path='configs/s1_v2_production.json',
    name='S1-LiquidityVacuum'
)

archetype_s4 = ArchetypeModel(
    config_path='configs/s4_optimized_oos_2024.json',
    name='S4-FundingDivergence'
)

models = [
    baseline_conservative,
    baseline_aggressive,
    archetype_s1,  # Uncomment
    archetype_s4   # Uncomment
]
```

Then run:
```bash
python3 examples/baseline_vs_archetype_comparison.py
```

---

## Expected Output (After Implementation)

```
================================================================================
MODEL COMPARISON REPORT
================================================================================

Train Period: 2022-01-01 to 2022-12-31
Test Period:  2023-01-01 to 2023-12-31

Models Compared: 4

================================================================================
SUMMARY TABLE
================================================================================
                       Train_PF   Train_WR  Train_Trades  Test_PF   Test_WR  Test_Trades  Overfit
Model
Baseline-Conservative  1.28       31.1      61            3.17      42.9     7            -1.89
Baseline-Aggressive    1.10       34.0      106           2.10      33.3     36           -1.00
S1-LiquidityVacuum     X.XX       XX.X      XXX           X.XX      XX.X     XXX          X.XX
S4-FundingDivergence   X.XX       XX.X      XXX           X.XX      XX.X     XXX          X.XX

================================================================================
WINNER ANALYSIS
================================================================================
Best Test Profit Factor: [TBD]
Best Test Win Rate: [TBD]
Least Overfit: [TBD]
```

---

## Key Integration Points

### BaseModel Interface Requirements

Your `ArchetypeModel` must implement:

1. **`__init__(self, name: str)`**
   - Set model name
   - Load configuration

2. **`fit(self, train_data: pd.DataFrame) -> None`**
   - Optional training/calibration
   - Can be no-op if using pre-optimized configs

3. **`predict(self, bar: pd.Series, position: Optional[Position]) -> Signal`**
   - Core logic: generate trading signal
   - Must return `Signal` object with:
     - `direction`: 'long', 'short', or 'hold'
     - `confidence`: 0.0-1.0
     - `entry_price`: float
     - `stop_loss`: Optional[float]
     - `metadata`: dict (optional)

4. **`get_params(self) -> dict`**
   - Return model configuration

### Signal Object Structure

```python
@dataclass
class Signal:
    direction: str  # 'long', 'short', 'hold'
    confidence: float  # 0.0-1.0
    entry_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: dict = field(default_factory=dict)
```

### Position Object (for exit logic)

```python
@dataclass
class Position:
    entry_price: float
    entry_time: pd.Timestamp
    direction: str  # 'long' or 'short'
    size: float  # Position size in quote currency
    stop_loss: Optional[float] = None
```

---

## Existing Archetype Configs to Support

### S1: Liquidity Vacuum
**Config:** `configs/s1_v2_production.json`
**Strategy:** Buy liquidity voids during momentum continuation
**Expected:** High PF, low trade frequency

### S4: Funding Divergence
**Config:** `configs/s4_optimized_oos_2024.json`
**Strategy:** Buy when funding rate diverges from price action
**Expected:** Moderate PF, moderate trade frequency

---

## Testing Checklist

Before running full comparison, test:

- [ ] ArchetypeModel imports successfully
- [ ] Config loading works (no JSON errors)
- [ ] `predict()` returns valid Signal objects
- [ ] Signal direction is one of: 'long', 'short', 'hold'
- [ ] Confidence is in range [0.0, 1.0]
- [ ] Stop loss is calculated correctly
- [ ] Metadata contains archetype information
- [ ] No crashes on real bar data

---

## Common Issues and Solutions

### Issue 1: Import errors
**Problem:** Can't import existing archetype logic
**Solution:** Check actual path of logic_v2_adapter, adjust imports

### Issue 2: Signal format mismatch
**Problem:** Archetype logic returns different signal format
**Solution:** Add conversion layer in `predict()` method

### Issue 3: Missing features in bar data
**Problem:** Archetype logic expects features not in bar
**Solution:** Check `data/features_mtf/` for available features, add fallbacks

### Issue 4: Config not loading
**Problem:** JSON parsing errors
**Solution:** Validate config files, ensure correct paths

---

## Questions for Agent 1

1. Where is the existing archetype logic located?
   - `engine/archetypes/logic_v2_adapter.py`?
   - Different module?

2. What does the current signal format look like?
   - Need to know structure to convert to `Signal` object

3. Are there any dependencies or state that need to be maintained?
   - Cooldowns, position tracking, etc.

4. How are features accessed from bar data?
   - Do we need feature name mapping?

---

## Deliverables

Once Agent 1 completes:

1. **File created:**
   - `engine/models/archetype_model.py`

2. **File updated:**
   - `engine/models/__init__.py`

3. **Testing completed:**
   - Wrapper imports successfully
   - Signals generate correctly
   - No errors on sample data

4. **Ready for Agent 2:**
   - Uncomment archetype models in comparison script
   - Run full 4-model comparison
   - Analyze baseline vs archetype performance

---

## Timeline

**Estimated Effort:** 2-4 hours
- 1 hour: Understand existing archetype logic structure
- 1 hour: Implement wrapper class
- 0.5 hour: Testing and debugging
- 0.5 hour: Integration and validation

**Blocker:** Cannot proceed with full comparison until this is done

---

## Contact

**Handoff from:** Agent 2 (model comparison framework)
**Handoff to:** Agent 1 (archetype logic owner)
**Next step:** Agent 2 runs full comparison once wrapper is complete
