# Wiring Tests - Quick Reference Card

## TL;DR: One-Page Cheat Sheet

### Purpose
**Turn "ghost module" claims into yes/no facts before production deployment.**

---

## Quick Run Commands

```bash
# All tests (6 seconds)
pytest tests/integration/test_wiring_gates.py -v

# Specific category
pytest tests/integration/test_wiring_gates.py::TestArchetypeIsolation -v

# Single test
pytest tests/integration/test_wiring_gates.py::TestArchetypeIsolation::test_single_archetype_enabled -v

# With coverage
pytest tests/integration/test_wiring_gates.py --cov=engine --cov-report=term
```

---

## The 4 Critical Contracts

| Test | Contract | What it Catches |
|------|----------|-----------------|
| **1. Archetype Isolation** | `enabled_archetypes=[H]` → only H fires | Optimizer tuning on wrong archetype mix |
| **2. Regime Causality** | `regime[t]` uses only `features[:t]` | Lookahead / temporal leakage |
| **3. Circuit Breaker** | Drawdown trigger → position size reduction | CB exists but never affects execution |
| **4. Soft Gating** | Same config → same weights every time | Non-deterministic allocation |

---

## Expected Output (Success)

```
============================== 10 passed in 5.92s ==============================
```

✓ **Safe to deploy**

---

## Expected Output (Failure)

```
FAILED tests/integration/test_wiring_gates.py::TestArchetypeIsolation::test_single_archetype_enabled
AssertionError: Found signals from other archetypes: {'S1', 'S4'}
Expected only 'H' or 'trap_within_trend'
```

❌ **DO NOT deploy - fix wiring bug first**

---

## Test Breakdown (10 tests total)

### Archetype Isolation (2 tests)
- `test_single_archetype_enabled` - Only enabled archetype fires
- `test_multiple_archetypes_enabled` - No cross-contamination

### Regime Causality (2 tests)
- `test_regime_prefix_invariance` - Full vs truncated run identical
- `test_regime_no_lookahead` - Only uses past features

### Circuit Breaker (2 tests)
- `test_tier2_reduces_position_size` - CB can reduce risk
- `test_circuit_breaker_toggle` - CB can be enabled/disabled

### Soft Gating (3 tests)
- `test_deterministic_weights` - Same config → same weights
- `test_regime_sensitive_weights` - Different regime → different weights
- `test_sqrt_split_prevents_double_weight` - sqrt(w) * sqrt(w) = w

### Integration (1 test)
- `test_end_to_end_with_all_components` - All systems coexist

---

## Common Failures & Fixes

### "No signals generated"
**Fix**: Check fusion_score thresholds in synthetic data
```python
test_data.loc[test_data.index[50], 'fusion_score'] = 0.8  # High enough to trigger
```

### "Regime prefix invariance FAILED"
**Fix**: Check for lookahead in regime features
```python
# Bad: Uses future data
df['regime'] = df['vix'].shift(-1)  # ❌

# Good: Uses only past data
df['regime'] = df['vix'].shift(1)  # ✓
```

### "Weights not deterministic"
**Fix**: Verify config identical between runs
```python
# Use config_override to ensure exact match
allocator = RegimeWeightAllocator(
    edge_table_path=edge_table,
    config_override={'k_shrinkage': 30, 'alpha': 4.0}  # Explicit config
)
```

---

## Integration with Deployment

### Pre-Deployment Checklist
```bash
# 1. Run wiring tests
pytest tests/integration/test_wiring_gates.py -v

# 2. If ALL PASS → safe to deploy
# 3. If ANY FAIL → DO NOT DEPLOY
```

### CI/CD Gate
```yaml
- name: Wiring Tests (Deployment Gate)
  run: pytest tests/integration/test_wiring_gates.py -v --tb=short
  # Exit code 0 = PASS (continue deployment)
  # Exit code 1 = FAIL (block deployment)
```

---

## When to Run These Tests

✓ **ALWAYS run before**:
- Production deployment
- Parameter optimization
- System architecture changes
- Adding new archetypes or strategies

✓ **Good practice to run after**:
- Refactoring core systems
- Upgrading dependencies
- Modifying regime detection
- Changing position sizing logic

❌ **Not needed for**:
- Documentation updates
- UI changes
- Log message tweaks
- Comment additions

---

## Test Design Philosophy

### These tests are:
- **Fast**: <10 seconds total runtime
- **Deterministic**: Same input → same output
- **Isolated**: No shared state
- **Clear**: Specific error messages with acceptance criteria

### These tests are NOT:
- Performance benchmarks
- Smoke tests (those are separate)
- Unit tests (those test individual functions)
- Analysis tools (they're PASS/FAIL gates)

---

## Files Modified by Wiring Tests

**Created**:
- `tests/integration/test_wiring_gates.py` (main test file)
- `tests/integration/WIRING_TESTS_README.md` (detailed docs)
- `tests/integration/WIRING_TESTS_QUICK_REF.md` (this file)

**Temporary files** (cleaned up automatically):
- `tmp_path/test_config.json` (archetype config)
- `tmp_path/edge_table.csv` (regime allocator edge data)
- `logs/test_cb/` (circuit breaker logs)

**NOT modified**:
- No production code changes
- No existing test modifications
- No config file updates

---

## Rollback Plan

If tests fail in CI/CD:

1. **Block deployment** (return exit code 1)
2. **Investigate root cause** (check test logs)
3. **Fix wiring bug** (not the test)
4. **Re-run tests** (verify fix)
5. **Deploy only after ALL PASS**

**Do NOT**:
- Skip failing tests
- Mark tests as xfail without investigation
- Deploy with failing wiring tests
- Tune test thresholds to make them pass

---

## Key Metrics

| Metric | Target | Actual |
|--------|--------|--------|
| Total tests | ≥8 | 10 ✓ |
| Runtime | <10s | ~6s ✓ |
| Coverage (wiring logic) | >80% | TBD |
| Flakiness rate | 0% | 0% ✓ |

---

## Questions?

**Test failing?**
1. Read the assertion error message
2. Check the docstring for acceptance criteria
3. Review the related spec (e.g., SOFT_GATING_PHASE1_SPEC.md)
4. Fix the wiring bug (not the test)

**Need to add new test?**
1. Check if it fits existing categories
2. Follow existing test patterns (fixtures, assertions)
3. Keep it fast (<2 seconds)
4. Make it deterministic

**Test becoming flaky?**
1. Investigate root cause (don't just increase retries)
2. Fix non-determinism in system under test
3. Add explicit random seed if needed
4. Document assumptions in test docstring

---

## Status Dashboard

```
Wiring Tests Status: ✓ ALL PASSING

Last Run: 2026-01-19
Total Tests: 10
Passed: 10
Failed: 0
Runtime: 5.92s

Safe to Deploy: YES ✓
```

---

**Remember**: These are NON-NEGOTIABLE deployment gates.
If any test fails, DO NOT deploy until root cause is fixed.

---

**Last Updated**: 2026-01-19
**Version**: 1.0
**Owner**: Backend Architect
