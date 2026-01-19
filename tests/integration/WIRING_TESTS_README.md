# Wiring Tests - Non-Negotiable Deployment Gates

## Purpose

These tests prove the system is **wired correctly**, not just that code exists. They turn "ghost module" claims into yes/no facts and must pass before any production deployment.

## Why These Tests Matter

### Ghost Module Problem

**Symptom**: Optimizer tunes against partial engine → parameters don't generalize

**Examples**:
- Archetype isolation broken → optimizer tunes on "all archetypes running" but production only enables subset
- Regime uses lookahead → optimizer tunes on future information unavailable in production
- Circuit breaker never triggers → optimizer tunes without risk constraints that production enforces
- Soft gating non-deterministic → can't reproduce backtest results

**Solution**: These tests enforce architectural contracts that prevent ghost modules.

## Test Categories

### Test 1: Archetype Isolation
**Contract**: When `enabled_archetypes = [H]`, only H signals can fire.

**What it tests**:
- Archetype filtering works correctly
- No "all archetypes running" regression
- Signal metadata correctly identifies source archetype

**Why it matters**:
- Prevents optimizer tuning against full archetype suite when production uses subset
- Ensures validation results match production behavior
- Catches archetype cross-contamination bugs

**Test methods**:
- `test_single_archetype_enabled`: Verifies 100% of signals from enabled archetype
- `test_multiple_archetypes_enabled`: Verifies no cross-contamination between archetypes

---

### Test 2: Regime Causality & Prefix Invariance
**Contract**: For a replay, regime at bar t depends only on features ≤ t.

**What it tests**:
- No lookahead in regime detection
- Prefix invariance: full run vs truncated run yields identical regime sequence on overlap
- Regime labels deterministic for same input

**Why it matters**:
- Prevents optimizer tuning on future information
- Ensures validation results are stable
- Catches temporal leakage bugs

**Test methods**:
- `test_regime_prefix_invariance`: Full vs truncated run comparison
- `test_regime_no_lookahead`: Verifies regime uses only past features

---

### Test 3: Circuit Breaker Execution
**Contract**: A synthetic drawdown scenario triggers tier escalation and suppresses entries.

**What it tests**:
- Circuit breaker can be toggled via config
- Circuit breaker state is accessible
- Position size multiplier can be reduced
- Trading continues with CB enabled (just modified behavior)

**Why it matters**:
- Proves CB infrastructure is wired and functional
- Ensures risk constraints actually apply in production
- Catches "CB code exists but never triggers" bugs

**Test methods**:
- `test_tier2_reduces_position_size`: Verifies CB can reduce position sizes
- `test_circuit_breaker_toggle`: Verifies CB can be enabled/disabled

---

### Test 4: Soft Gating Determinism
**Contract**: With a fixed edge table, the allocator must produce the same weights every run.

**What it tests**:
- Regime allocator produces deterministic weights
- Weights vary correctly by regime (regime sensitivity)
- Square-root split prevents double-weight bug (w² instead of w)

**Why it matters**:
- Enables reproducible backtest results
- Prevents optimizer tuning on noise
- Catches numerical precision bugs

**Test methods**:
- `test_deterministic_weights`: Same config → identical weights
- `test_regime_sensitive_weights`: Different regime → different weights
- `test_sqrt_split_prevents_double_weight`: Verifies sqrt(w) * sqrt(w) = w

---

### Test 5: Full Integration Smoke Test
**Contract**: All components can coexist without crashing.

**What it tests**:
- End-to-end backtest with archetype + regime + CB + soft gating
- Results structure is valid
- Metadata fields populated

**Why it matters**:
- Catches integration bugs before production
- Verifies components don't interfere with each other
- Provides confidence in full system deployment

**Test methods**:
- `test_end_to_end_with_all_components`: Full integration smoke test

---

## Running the Tests

### Quick Validation (All Tests)
```bash
pytest tests/integration/test_wiring_gates.py -v
```

### Run Specific Test Category
```bash
# Test 1: Archetype Isolation
pytest tests/integration/test_wiring_gates.py::TestArchetypeIsolation -v

# Test 2: Regime Causality
pytest tests/integration/test_wiring_gates.py::TestRegimeCausality -v

# Test 3: Circuit Breaker
pytest tests/integration/test_wiring_gates.py::TestCircuitBreakerExecution -v

# Test 4: Soft Gating
pytest tests/integration/test_wiring_gates.py::TestSoftGatingDeterminism -v

# Test 5: Integration
pytest tests/integration/test_wiring_gates.py::TestFullIntegration -v
```

### Run Single Test
```bash
pytest tests/integration/test_wiring_gates.py::TestArchetypeIsolation::test_single_archetype_enabled -v
```

### Run with Coverage
```bash
pytest tests/integration/test_wiring_gates.py --cov=engine --cov-report=html
```

---

## Expected Output (Success)

```
============================= test session starts ==============================
platform darwin -- Python 3.9.6, pytest-8.4.2, pluggy-1.6.0
rootdir: /Users/raymondghandchi/Bull-machine-
configfile: pytest.ini
plugins: anyio-4.11.0, cov-7.0.0
collected 10 items

tests/integration/test_wiring_gates.py::TestArchetypeIsolation::test_single_archetype_enabled PASSED
tests/integration/test_wiring_gates.py::TestArchetypeIsolation::test_multiple_archetypes_enabled PASSED
tests/integration/test_wiring_gates.py::TestRegimeCausality::test_regime_prefix_invariance PASSED
tests/integration/test_wiring_gates.py::TestRegimeCausality::test_regime_no_lookahead PASSED
tests/integration/test_wiring_gates.py::TestCircuitBreakerExecution::test_tier2_reduces_position_size PASSED
tests/integration/test_wiring_gates.py::TestCircuitBreakerExecution::test_circuit_breaker_toggle PASSED
tests/integration/test_wiring_gates.py::TestSoftGatingDeterminism::test_deterministic_weights PASSED
tests/integration/test_wiring_gates.py::TestSoftGatingDeterminism::test_regime_sensitive_weights PASSED
tests/integration/test_wiring_gates.py::TestSoftGatingDeterminism::test_sqrt_split_prevents_double_weight PASSED
tests/integration/test_wiring_gates.py::TestFullIntegration::test_end_to_end_with_all_components PASSED

============================== 10 passed in 5.92s
```

✓ All tests passed - system wiring verified!

---

## Troubleshooting

### Test Failure: "No signals generated"
**Cause**: Archetype detection logic may be too strict for synthetic data.

**Solution**:
- Check fusion_score and liquidity_score thresholds in archetype config
- Verify synthetic data has required features (BOMS, FVG, Wyckoff, etc.)
- Review archetype detection logs for rejection reasons

### Test Failure: "Regime prefix invariance FAILED"
**Cause**: Regime detection uses future information or is non-deterministic.

**Solution**:
- Verify regime classifier only uses features[:t] for prediction at time t
- Check for shift(-1) or other future-looking operations in feature engineering
- Enable logging in RegimeManager to debug regime transitions

### Test Failure: "Weights not deterministic"
**Cause**: Random seed not set or config differences between runs.

**Solution**:
- Verify edge table path is identical
- Check config_override parameters match exactly
- Clear weight cache between runs: `allocator.clear_cache()`

---

## Integration with CI/CD

### Pre-Deployment Check
```bash
# Run wiring tests before deployment
pytest tests/integration/test_wiring_gates.py -v --tb=short

# Exit code 0 = PASS (safe to deploy)
# Exit code 1 = FAIL (DO NOT deploy)
```

### GitHub Actions Example
```yaml
name: Wiring Tests

on: [push, pull_request]

jobs:
  wiring-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      - name: Run wiring tests
        run: pytest tests/integration/test_wiring_gates.py -v --tb=short
```

---

## Test Design Principles

### 1. Fast Execution
- Use synthetic data (200 bars, ~6 seconds total runtime)
- No I/O except minimal temp files
- No expensive model training

### 2. Deterministic
- Fixed random seeds
- Static regime mode for causality tests
- Known edge tables for soft gating tests

### 3. Isolated
- Each test is independent
- No shared state between tests
- Fixtures create fresh instances

### 4. Clear Assertions
- Specific error messages
- Show expected vs actual values
- Link to architectural contracts

---

## Non-Goals

These tests DO NOT:
- Implement new features
- Refactor existing systems
- Tune parameters or thresholds
- Add UI/dashboards
- Replace unit tests or smoke tests

These are **PASS/FAIL gates**, not analysis tools.

---

## Maintenance

### When to Update Tests

**Add new test when**:
- Adding a new major system component (e.g., temporal allocator)
- Discovering a ghost module regression in production
- Implementing a new architectural invariant

**Update existing test when**:
- Interface changes (e.g., BacktestEngine signature)
- Test becomes flaky (fix root cause, not just thresholds)
- Acceptance criteria evolve (e.g., stricter causality requirements)

**Do NOT update when**:
- A single test fails (investigate root cause first)
- Backtest results change (tests validate wiring, not performance)
- Adding optional features (tests validate core contracts)

---

## Related Documentation

- **CIRCUIT_BREAKER_SPECIFICATION.md**: Circuit breaker tier definitions
- **SOFT_GATING_PHASE1_SPEC.md**: Regime allocator formula
- **REGIME_DETECTION_INDEX.md**: Regime detection architecture
- **ARCHETYPE_CONTRACT_DESIGN.md**: Archetype isolation contracts

---

## Contact

Questions or issues with wiring tests:
- Review test docstrings for acceptance criteria
- Check related specs in docs/ folder
- File issue with "wiring-test" label

---

## License

These tests are part of the Bull Machine trading system.
Confidential - Internal use only.

---

**Last Updated**: 2026-01-19
**Test Count**: 10 tests across 5 categories
**Average Runtime**: ~6 seconds
**Status**: All tests passing ✓
