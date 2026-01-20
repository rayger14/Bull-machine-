# Hybrid Confidence Integration - Backtest Validation Report

**Generated:** 2026-01-15 19:55:14

---

## Executive Summary

This report validates the hybrid confidence integration in RegimeService, comparing raw ensemble agreement vs. calibrated stability forecast.

## Test Configuration

- **Period:** 2023-01-01 to 2024-12-31
- **Total Bars:** 17,521
- **Calibration:** ENABLED
- **Model:** ensemble_regime_v1.pkl
- **Calibrator:** confidence_calibrator_v1.pkl

## Regime Distribution

| Regime | Count | Percentage |
|--------|-------|------------|
| risk_on | 17,521 | 100.00% |

**Transitions:** 1
**Avg Bars/Regime:** 17521.0

## Confidence Metrics

### Raw Agreement (ensemble_agreement)

- Mean: 0.9880
- Median: 0.9880
- Std: 0.0028
- Range: [0.9609, 0.9947]

### Calibrated Forecast (regime_stability_forecast)

- Mean: 0.9137
- Median: 0.9137
- Std: 0.0000
- Range: [0.9137, 0.9137]
- Unique Values: 1

**Note:** Calibrated forecast returns constant value. This indicates all ensemble agreements in this period map to the same stability forecast.

### Calibration Effect

- Mean Delta: -0.0743
- Median Delta: -0.0743
- Increases: 0.0%
- Decreases: 100.0%

**Interpretation:** Calibration consistently reduces confidence, suggesting raw ensemble agreement is overconfident compared to actual stability.

## Confidence vs Stability Analysis

Tests hypothesis: Higher confidence → More stable regimes (fewer transitions)

### Raw Agreement

| Bucket | Bars/Regime | Transitions | Mean Confidence |
|--------|-------------|-------------|------------------|
| Q1 (Low) | 4388.00 | 1 | 0.985 |
| Q2 | 4373.00 | 1 | 0.987 |
| Q3 | 4389.00 | 1 | 0.989 |
| Q4 (High) | 4371.00 | 1 | 0.991 |

**Stability Improvement (Q4 vs Q1):** -0.4%
**Passes Monotonicity Test:** ✗ FAIL

### Calibrated Forecast

**Status:** Cannot analyze (constant value)

## Integration Validation

### Metrics Availability

- ensemble_agreement (raw): ✓ Present
- regime_stability_forecast (calibrated): ✓ Present
- regime_confidence (backward compat): ✓ Present

## Key Findings

1. **Integration Status:** ✓ Hybrid confidence metrics successfully integrated
   - Both raw and calibrated metrics are returned
   - No integration errors detected

2. **Calibration Effect:** Mean delta = -0.0743
   - Calibration reduces confidence in 100.0% of cases
   - This suggests raw agreement may be overconfident

3. **Stability Relationship:** ✗ No clear relationship observed
   - This may be due to limited regime diversity in test period
   - All samples in test period are 'risk_on' regime

## Recommendations

1. ⚠ **Calibrator returns constant value** - This is expected when all ensemble agreements are high (>0.9)
   - The calibrator has learned that even high agreement doesn't guarantee perfect stability
   - Consider testing on period with more regime diversity
2. **Test with crisis periods** - Validate calibration effect during regime transitions
3. **Monitor in production** - Track actual stability outcomes vs. forecasts

---

**Report End**
