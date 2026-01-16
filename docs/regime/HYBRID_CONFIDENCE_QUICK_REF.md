# Hybrid Confidence - Quick Reference

**Status:** ✓ Validated for Production
**Date:** 2026-01-15

---

## Quick Facts

- **Integration:** ✓ Working correctly
- **Metrics Available:** `ensemble_agreement` (raw) + `regime_stability_forecast` (calibrated)
- **Calibration Effect:** Reduces confidence by ~7.4pp (conservative adjustment)
- **Recommended Metric:** Use `regime_stability_forecast` for decisions

---

## Usage

### Get Regime with Hybrid Confidence

```python
from engine.context.regime_service import RegimeService

# Initialize with calibration enabled (default)
service = RegimeService(
    mode='dynamic_ensemble',
    model_path='models/ensemble_regime_v1.pkl',
    enable_calibration=True,  # Hybrid approach
    calibrator_path='models/confidence_calibrator_v1.pkl'
)

# Get regime
result = service.get_regime(features, timestamp)

# Access hybrid metrics
raw_agreement = result['ensemble_agreement']          # Raw model agreement (0-1)
calibrated_forecast = result['regime_stability_forecast']  # Calibrated stability (0-1)
regime_confidence = result['regime_confidence']        # Backward compat (hysteresis)
```

### Interpretation

| Metric | Range | Meaning | Use For |
|--------|-------|---------|---------|
| `ensemble_agreement` | 0-1 | Raw model consensus | Diagnostics |
| `regime_stability_forecast` | 0-1 | Calibrated stability probability | Decision-making |
| `regime_confidence` | 0-1 | Hysteresis gap | Backward compat |

**Example:**
- `ensemble_agreement = 0.98` → Models agree 98%
- `regime_stability_forecast = 0.91` → Actual stability ~91% (calibrated down)
- **Use 0.91** for thresholds and decision-making

---

## Validation Results

### Integration: ✓ PASS
- Both metrics returned correctly
- No integration errors
- Tested on 17,521 bars (2023-2024)

### Calibration Effect
- **Direction:** Always reduces confidence (100%)
- **Magnitude:** Mean -7.4pp, range [-2.5pp to -8.5pp]
- **Interpretation:** Prevents overconfidence

### Key Insight
Calibrator learned that 98% ensemble agreement → 91% actual stability. This is a valuable reality check.

---

## Files

1. **Full Report:** `HYBRID_CONFIDENCE_BACKTEST_REPORT.md`
2. **Summary:** `HYBRID_CONFIDENCE_VALIDATION_SUMMARY.md`
3. **User Guide:** `HYBRID_CONFIDENCE_GUIDE.md`
4. **Validation Script:** `bin/backtest_hybrid_confidence_validation.py`

---

## Re-run Validation

```bash
# Run on different period
python3 bin/backtest_hybrid_confidence_validation.py

# Edit dates in script:
start_date = '2022-01-01'  # For crisis period
end_date = '2022-12-31'
```

---

## Next Steps

1. ✓ Integration validated
2. Test on 2022 crisis period (more regime diversity)
3. Update archetype documentation
4. Monitor in production

---

**Approved for Production:** YES ✓
**Last Validated:** 2026-01-15
