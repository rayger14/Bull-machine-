# Walk-Forward Validation - Executable Implementation Plan

**Date**: 2026-01-16
**Owner**: Performance Engineer
**Status**: Ready to Execute

---

## Quick Start (Copy-Paste Commands)

```bash
# Step 1: Create the production walk-forward script
python bin/walk_forward_production_engine.py --archetype S1 --test

# Step 2: Run full validation on all archetypes
python bin/walk_forward_production_engine.py --all

# Step 3: Generate comparison report
python bin/walk_forward_production_engine.py --report
```

---

## Implementation Checklist

### ✅ Prerequisites (Already Have)

- [x] Production backtest engine: `bin/backtest_full_engine_replay.py`
- [x] Real archetype implementations: `engine/strategies/archetypes/`
- [x] Archetype factory: `engine/archetypes/archetype_factory.py`
- [x] Optimized configs: `results/optimization_2026-01-16/*/best_config.json`
- [x] Feature data: `data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet`

### 🔧 Step 1: Enhance FullEngineBacktest (15 min)

**File**: `bin/backtest_full_engine_replay.py`

**Changes Needed**:

1. **Add parameter override support** (lines 117-130):

```python
class FullEngineBacktest:
    def __init__(self, config: Dict):
        # ... existing initialization ...

        # NEW: Support archetype-specific parameter overrides
        self.archetype_param_overrides = config.get('archetype_param_overrides', {})
```

2. **Modify archetype evaluation** (lines 523-560):

```python
def _evaluate_archetype(
    self,
    archetype_id: str,
    bar: pd.Series,
    context_data: pd.DataFrame
) -> Tuple[float, str]:
    """
    Evaluate specific archetype logic using PRODUCTION IMPLEMENTATIONS.
    Now supports parameter overrides for walk-forward validation.
    """
    regime_label = bar.get('regime_label', 'neutral')

    # NEW: Get config with walk-forward overrides
    archetype_config = self._get_archetype_config_with_overrides(archetype_id)

    # Call production archetype implementation via factory
    confidence, direction, metadata = self.archetype_factory.evaluate_archetype(
        archetype_id,
        bar,
        regime_label,
        config_override=archetype_config  # NEW: Pass overrides
    )

    return confidence, direction


def _get_archetype_config_with_overrides(self, archetype_id: str) -> Dict:
    """Get archetype config with walk-forward parameter overrides"""
    # Get base config from factory
    base_config = self.archetype_factory._get_archetype_config(
        {'id': archetype_id, 'slug': archetype_id}  # Minimal archetype dict
    )

    # Apply walk-forward overrides if present
    if archetype_id in self.archetype_param_overrides:
        overrides = self.archetype_param_overrides[archetype_id]
        base_config['thresholds'].update(overrides)
        logger.info(f"[Walk-Forward] Applied param overrides to {archetype_id}: {overrides}")

    return base_config
```

**Test**:
```bash
# Verify backtest still works without overrides
python bin/backtest_full_engine_replay.py
```

### 🔧 Step 2: Create Walk-Forward Script (45 min)

**File**: `bin/walk_forward_production_engine.py`

**Copy the implementation from WALK_FORWARD_PRODUCTION_ENGINE_DESIGN.md Section "Step 2"**

Key components:
- `WalkForwardWindow` dataclass
- `ProductionWalkForwardValidator` class
- `generate_windows()` method
- `run_window()` method (uses FullEngineBacktest)
- `validate_archetype()` method
- `_aggregate_results()` method
- `main()` execution

**Test**:
```bash
# Test window generation only (dry run)
python bin/walk_forward_production_engine.py --archetype S1 --dry-run

# Expected output: List of 8-10 windows with date ranges
```

### 🔧 Step 3: Integrate Archetype Factory Parameter Passing (30 min)

**File**: `engine/archetypes/archetype_factory.py`

**Changes**:

```python
def evaluate_archetype(
    self,
    archetype_slug: str,
    bar: pd.Series,
    regime_label: str = 'neutral',
    config_override: Optional[Dict] = None  # NEW parameter
) -> Tuple[float, str, Dict[str, Any]]:
    """
    Evaluate specific archetype on current bar.

    Args:
        archetype_slug: Archetype identifier
        bar: Current bar data
        regime_label: Current regime
        config_override: Optional config override for walk-forward validation

    Returns:
        (confidence, direction, metadata)
    """
    if archetype_slug not in self.instances:
        return 0.0, 'hold', {'reason': 'not_loaded'}

    archetype_data = self.instances[archetype_slug]
    instance = archetype_data['instance']

    # NEW: Apply config override if provided
    if config_override:
        original_config = instance.config
        instance.config = config_override
        logger.debug(f"[Factory] Temporarily overriding config for {archetype_slug}")

    try:
        result = instance.detect(bar, regime_label)

        if result is None or len(result) != 3:
            return 0.0, 'hold', {'reason': 'invalid_result'}

        archetype_name, confidence, metadata = result

        if archetype_name is None or confidence == 0.0:
            return 0.0, 'hold', metadata

        return confidence, archetype_data['direction'], metadata

    finally:
        # NEW: Restore original config
        if config_override:
            instance.config = original_config
```

**Test**:
```bash
# Test archetype detection with override
python -c "
from engine.archetypes.archetype_factory import ArchetypeFactory
import pandas as pd

factory = ArchetypeFactory({'enable_S1': True})
bar = pd.Series({'close': 45000, 'liquidity_score': 0.15, 'volume_z': 2.0})

# Test with override
override = {'thresholds': {'fusion_threshold': 0.25}}
conf, dir, meta = factory.evaluate_archetype('liquidity_vacuum', bar, config_override=override)
print(f'Confidence: {conf}, Direction: {dir}')
"
```

### 🔧 Step 4: Run Single Archetype Test (20 min)

**Execute**:
```bash
# Run walk-forward on S1 (Liquidity Vacuum)
python bin/walk_forward_production_engine.py \
    --archetype S1 \
    --config results/optimization_2026-01-16/S1/best_config.json \
    --data data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet \
    --output results/walk_forward_production/S1_validation.json
```

**Expected Output**:
```
================================================================================
WALK-FORWARD VALIDATION - PRODUCTION ENGINE
================================================================================
Loading data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet...
Loaded 26,280 bars from 2022-01-01 to 2024-12-31

============================================================
VALIDATING S1
============================================================
Generated 8 walk-forward windows

============================================================
Window 1/8
Train: 2022-01-01 to 2023-01-01
Embargo: 2023-01-01 to 2023-01-03
Test: 2023-01-03 to 2023-04-03
============================================================
ENTRY: S1 short @ $16,832.45, size=$1,200.00, SL=$17,253.00, TP=$15,990.00
EXIT: S1 stop_loss @ $17,253.00, PnL=$-421.00, held 18.5h
...
[Window 1 Results]
  Trades: 9
  Return: 3.2%
  Sharpe: 0.87
  Max DD: 8.3%
  Win Rate: 55.6%

[Continuing through windows 2-8...]

RESULTS: S1
  Windows: 8
  Total Trades: 73
  Total PnL: $1,247.32
  OOS Sharpe: 1.42
  OOS Degradation: 20.2%
  Robust: ❌

✅ Results saved: results/walk_forward_production/S1_validation.json
```

**Validation**:
- ✅ Uses FullEngineBacktest (check log for "FullEngineBacktest initialized")
- ✅ Calls real S1 implementation (check for "Archetype] S1: confidence=...")
- ✅ Applies regime penalties (check for "Regime Confidence")
- ✅ Tracks direction balance (check for "Direction Balance Monitor")
- ✅ Applies fees/slippage (check PnL calculations)

### 🔧 Step 5: Run All Archetypes (2 hours)

**Execute**:
```bash
# Run all 6 archetypes in parallel (if system supports)
python bin/walk_forward_production_engine.py --all

# OR run sequentially
for arch in S1 S4 S5 B H K; do
    python bin/walk_forward_production_engine.py --archetype $arch
done
```

**Monitor**:
```bash
# Watch progress
tail -f logs/walk_forward_production.log

# Check intermediate results
ls -lh results/walk_forward_production/*.json
```

### 🔧 Step 6: Generate Comparison Report (15 min)

**Execute**:
```bash
python bin/walk_forward_production_engine.py --report
```

**Expected Output**:
- `results/walk_forward_production/comparison_report.md`
- `results/walk_forward_production/production_ready_configs.json`
- `results/walk_forward_production/degradation_analysis.csv`

---

## Testing Protocol

### Unit Tests

```bash
# Test 1: Window generation
python -c "
from bin.walk_forward_production_engine import ProductionWalkForwardValidator
import pandas as pd

validator = ProductionWalkForwardValidator()
data = pd.DataFrame({'close': range(10000)}, index=pd.date_range('2022-01-01', periods=10000, freq='H'))
windows = validator.generate_windows(data)
print(f'Generated {len(windows)} windows')
assert len(windows) >= 8, 'Should generate at least 8 windows'
"

# Test 2: Config override application
python -c "
from bin.backtest_full_engine_replay import FullEngineBacktest

config = {
    'initial_capital': 10000,
    'archetype_param_overrides': {
        'S1': {'fusion_threshold': 0.99}
    }
}
backtest = FullEngineBacktest(config)
assert 'S1' in backtest.archetype_param_overrides
print('✅ Config override applied')
"

# Test 3: End-to-end sanity check
python bin/walk_forward_production_engine.py --archetype S1 --windows 2 --quick-test
```

### Integration Tests

```bash
# Test full pipeline on small dataset
python bin/walk_forward_production_engine.py \
    --archetype S1 \
    --data data/features_mtf/BTC_1H_2022-01-01_to_2022-06-30.parquet \
    --windows 2 \
    --output results/walk_forward_test.json

# Verify output structure
python -c "
import json
with open('results/walk_forward_test.json') as f:
    data = json.load(f)
    assert 'aggregate_metrics' in data
    assert 'oos_analysis' in data
    assert 'window_details' in data
    print('✅ Output structure valid')
"
```

---

## Troubleshooting

### Issue: "ArchetypeFactory not loading archetypes"

**Diagnosis**:
```bash
python -c "
from engine.archetypes.archetype_factory import ArchetypeFactory
factory = ArchetypeFactory({'enable_S1': True})
print(f'Loaded archetypes: {factory.get_active_archetypes()}')
"
```

**Fix**: Check `archetype_registry.yaml` and enable flags.

### Issue: "No regime_label column"

**Diagnosis**:
```bash
python -c "
import pandas as pd
df = pd.read_parquet('data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet')
print('Columns:', df.columns.tolist())
print('Has regime_label:', 'regime_label' in df.columns)
"
```

**Fix**: Run regime classification:
```bash
python bin/add_regime_labels_streaming.py \
    --input data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet \
    --output data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet
```

### Issue: "Walk-forward too slow"

**Optimization**:
- Reduce logging verbosity: `logging.basicConfig(level=logging.WARNING)`
- Parallelize windows: Use `multiprocessing.Pool`
- Use smaller test windows: `--test-days 60` instead of 90

---

## Performance Benchmarks

**Expected Runtime** (3-year dataset, 6 archetypes):
- Single archetype, single window: ~5 seconds
- Single archetype, all windows (8): ~45 seconds
- All archetypes (6), all windows: ~5 minutes

**Memory Usage**:
- Peak: ~2GB (feature data + backtest state)
- Average: ~1GB

---

## Deliverables Checklist

- [ ] `bin/walk_forward_production_engine.py` (new file)
- [ ] `bin/backtest_full_engine_replay.py` (enhanced with overrides)
- [ ] `engine/archetypes/archetype_factory.py` (enhanced with config_override)
- [ ] `results/walk_forward_production/S1_validation.json`
- [ ] `results/walk_forward_production/S4_validation.json`
- [ ] `results/walk_forward_production/S5_validation.json`
- [ ] `results/walk_forward_production/B_validation.json`
- [ ] `results/walk_forward_production/H_validation.json`
- [ ] `results/walk_forward_production/K_validation.json`
- [ ] `results/walk_forward_production/comparison_report.md`
- [ ] `results/walk_forward_production/production_ready_configs.json`

---

## Success Metrics

### Minimum Viable

- ✅ Script runs end-to-end without errors
- ✅ Uses FullEngineBacktest (verified in logs)
- ✅ Generates 8+ windows per archetype
- ✅ Produces valid JSON output

### Target

- ✅ 3+ archetypes production-ready (degradation <20%)
- ✅ Average OOS degradation <25%
- ✅ Zero runtime errors
- ✅ Complete documentation

### Stretch

- ✅ 5+ archetypes production-ready
- ✅ Average OOS degradation <20%
- ✅ Parallel execution support
- ✅ Automated reporting

---

## Next Actions

1. **Execute Step 1-3** (Infrastructure setup): ~90 min
2. **Execute Step 4** (Single archetype test): ~20 min
3. **Verify results** and debug if needed: ~30 min
4. **Execute Step 5** (All archetypes): ~2 hours
5. **Execute Step 6** (Reporting): ~15 min

**Total estimated time**: 4-5 hours

---

**Ready to begin?** Start with Step 1.
