================================================================================
STATE-AWARE CRISIS FEATURES VALIDATION REPORT
================================================================================

Generated: 2025-12-18 15:19:50

================================================================================
SYNTHETIC DATA VALIDATION
================================================================================

Overall Statistics:
  crash_stress_24h:
    Mean: 0.002, Std: 0.011
    P50: 0.000, P90: 0.000
    Activation Rate: 0.0%
  crash_stress_72h:
    Mean: 0.002, Std: 0.007
    P50: 0.000, P90: 0.006
    Activation Rate: 0.0%
  vol_persistence:
    Mean: 0.045, Std: 0.153
    P50: 0.000, P90: 0.069
    Activation Rate: 7.3%
  hours_since_crisis:
    Mean: 0.077, Std: 0.234
    P50: 0.000, P90: 0.179
    Activation Rate: 9.6%
  crash_frequency_7d:
    Mean: 0.393, Std: 0.882
    P50: 0.000, P90: 2.000
    Activation Rate: 18.5%
  funding_stress_ewma:
    Mean: 0.022, Std: 0.098
    P50: 0.000, P90: 0.007
    Activation Rate: 4.0%
  cascade_risk:
    Mean: 0.000, Std: 0.000
    P50: 0.000, P90: 0.000
    Activation Rate: 0.0%
  crisis_persistence:
    Mean: 0.202, Std: 0.546
    P50: 0.000, P90: 0.692
    Activation Rate: 16.2%
  vol_regime_shift:
    Mean: 0.000, Std: 0.000
    P50: 0.000, P90: 0.000
    Activation Rate: 0.0%
  drawdown_persistence:
    Mean: 0.000, Std: 0.000
    P50: 0.000, P90: 0.000
    Activation Rate: 0.0%
  aftershock_score:
    Mean: 0.030, Std: 0.109
    P50: 0.000, P90: 0.028
    Activation Rate: 5.8%

================================================================================
HISTORICAL DATA VALIDATION (BTC)
================================================================================

Overall Statistics:
  crash_stress_24h:
    Mean: 0.000, Std: 0.004
    P50: 0.000, P90: 0.000
    Activation Rate: 0.0%
  crash_stress_72h:
    Mean: 0.000, Std: 0.002
    P50: 0.000, P90: 0.000
    Activation Rate: 0.0%
  vol_persistence:
    Mean: 0.023, Std: 0.033
    P50: 0.008, P90: 0.065
    Activation Rate: 0.3%
  hours_since_crisis:
    Mean: 0.025, Std: 0.117
    P50: 0.000, P90: 0.009
    Activation Rate: 3.9%
  crash_frequency_7d:
    Mean: 0.077, Std: 0.276
    P50: 0.000, P90: 0.000
    Activation Rate: 7.4%
  funding_stress_ewma:
    Mean: 0.005, Std: 0.020
    P50: 0.000, P90: 0.005
    Activation Rate: 0.1%
  cascade_risk:
    Mean: 0.004, Std: 0.012
    P50: 0.000, P90: 0.009
    Activation Rate: 0.0%
  crisis_persistence:
    Mean: 0.115, Std: 0.148
    P50: 0.031, P90: 0.344
    Activation Rate: 27.4%
  vol_regime_shift:
    Mean: 0.034, Std: 0.156
    P50: 0.000, P90: 0.002
    Activation Rate: 4.6%
  drawdown_persistence:
    Mean: 0.379, Std: 0.443
    P50: 0.034, P90: 1.000
    Activation Rate: 43.7%
  aftershock_score:
    Mean: 0.066, Std: 0.083
    P50: 0.031, P90: 0.183
    Activation Rate: 6.7%

LUNA Crisis Window (May 9-12, 2022):
  crash_stress_24h: mean=0.009, activation=0.0%
  crash_stress_72h: mean=0.004, activation=0.0%
  vol_persistence: mean=0.076, activation=0.0%
  hours_since_crisis: mean=0.000, activation=0.0%
  crash_frequency_7d: mean=0.178, activation=17.8%
  funding_stress_ewma: mean=0.000, activation=0.0%
  cascade_risk: mean=0.000, activation=0.0%
  crisis_persistence: mean=0.068, activation=0.0%
  vol_regime_shift: mean=0.035, activation=8.2%
  drawdown_persistence: mean=1.000, activation=100.0%
  aftershock_score: mean=0.180, activation=30.1%

FTX Crisis Window (Nov 8-11, 2022):
  crash_stress_24h: mean=0.014, activation=0.0%
  crash_stress_72h: mean=0.011, activation=0.0%
  vol_persistence: mean=0.067, activation=0.0%
  hours_since_crisis: mean=0.000, activation=0.0%
  crash_frequency_7d: mean=0.767, activation=76.7%
  funding_stress_ewma: mean=0.000, activation=0.0%
  cascade_risk: mean=0.000, activation=0.0%
  crisis_persistence: mean=0.132, activation=24.7%
  vol_regime_shift: mean=0.362, activation=63.0%
  drawdown_persistence: mean=0.373, activation=64.4%
  aftershock_score: mean=0.355, activation=71.2%

================================================================================
SUCCESS CRITERIA ASSESSMENT
================================================================================

Criteria:
  Persistence: State features stay elevated 2-7 days after crisis events
  Activation: 10-30% overall activation rate (not always on)
  Crisis Response: >50% activation during LUNA/FTX windows

Assessment:
  Activation Rate: ⚠️  FAIL
  Crisis Response: ⚠️  FAIL

================================================================================
VALIDATION COMPLETE
================================================================================