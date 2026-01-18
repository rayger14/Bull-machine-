# Paper Trading Metrics Dashboard Specification

**Version**: 1.0
**Status**: Production Design
**Purpose**: Measure forward-looking reliability for backtest → paper → live capital transition

---

## Executive Summary

This dashboard measures **what actually matters** for live trading readiness:
- **Forward-looking reliability** (will this work tomorrow?)
- **Regime adaptation quality** (handles transitions smoothly?)
- **Metadata integrity** (boosts/vetoes firing correctly?)
- **Ensemble health** (diversity maintained, no redundancy?)

**Key Differentiator**: Traditional dashboards show historical PnL. This dashboard **predicts future failure modes** using statistical drift detection, regime alignment scoring, and metadata integrity monitoring.

---

## Table of Contents

1. [Metrics Taxonomy](#metrics-taxonomy)
2. [Real-Time Monitoring Layer](#real-time-monitoring-layer)
3. [Drift Detection System](#drift-detection-system)
4. [Archetype-Level Diagnostics](#archetype-level-diagnostics)
5. [Ensemble-Level Diagnostics](#ensemble-level-diagnostics)
6. [Kill-Switch System](#kill-switch-system)
7. [Comparison Framework](#comparison-framework)
8. [Dashboard Design](#dashboard-design)
9. [Technology Stack](#technology-stack)
10. [Implementation Roadmap](#implementation-roadmap)
11. [Database Schema](#database-schema)
12. [Alert Configuration](#alert-configuration)
13. [Risk Mitigation](#risk-mitigation)

---

## 1. Metrics Taxonomy

### 1.1 Metric Categories

```
TIER 1: CRITICAL (Kill-Switch Eligible)
├── System Health
│   ├── Daily PnL (% return)
│   ├── Drawdown from peak (%)
│   ├── Fill rate (% executed)
│   └── Regime detection lag (minutes)
├── Performance Degradation
│   ├── Rolling Sharpe ratio (1d, 7d, 30d)
│   ├── Win rate deviation (actual vs expected)
│   └── Consecutive loss count
└── Metadata Integrity
    ├── Boost/veto activation rate (%)
    ├── Archetype-regime misalignment count
    └── Confidence score distribution (mean, std)

TIER 2: WARNING (Manual Review Required)
├── Feature Drift
│   ├── PSI (Population Stability Index)
│   ├── KS statistic (Kolmogorov-Smirnov)
│   └── Feature distribution shifts
├── Performance Drift
│   ├── CUSUM (Cumulative Sum) for win rate
│   ├── Rolling correlation vs backtest
│   └── Trade duration drift
└── Regime Drift
    ├── HMM state transition frequency
    ├── Manual label agreement (%)
    └── Regime stability score

TIER 3: INFORMATIONAL (Trend Monitoring)
├── Archetype Activity
│   ├── Signal count (per hour/day)
│   ├── Domain boost distribution
│   └── Position hold time
├── Ensemble Metrics
│   ├── Signal overlap rate (%)
│   ├── Archetype correlation matrix
│   └── Regime coverage (%)
└── Execution Quality
    ├── Slippage (bps)
    ├── Latency (order to fill)
    └── Partial fill rate (%)
```

### 1.2 Metric Priorities

**Critical Path to Live Trading**:
1. **Week 1-2**: Metadata integrity (boosts/vetoes firing correctly?)
2. **Week 2-4**: Performance stability (Sharpe >1.5, DD <15%)
3. **Week 4-6**: Drift detection (features stable, no concept drift)
4. **Week 6-8**: Regime adaptation (handles transitions smoothly)
5. **Week 8+**: Ensemble optimization (diversity, overlap control)

---

## 2. Real-Time Monitoring Layer

### 2.1 System Health Metrics (Update Every 30s)

#### Core Performance Indicators

```python
# Real-time calculations
metrics = {
    # PnL Tracking
    "daily_pnl": current_equity - session_start_equity,
    "daily_pnl_pct": (current_equity / session_start_equity - 1) * 100,
    "unrealized_pnl": sum([pos.unrealized_pnl for pos in active_positions]),

    # Drawdown Monitoring
    "current_drawdown": (peak_equity - current_equity) / peak_equity * 100,
    "max_drawdown": max_historical_drawdown,
    "drawdown_duration": time_since_peak_hours,

    # Sharpe Ratio (Rolling Windows)
    "sharpe_1d": calc_sharpe(returns_24h, periods=24),
    "sharpe_7d": calc_sharpe(returns_7d, periods=168),
    "sharpe_30d": calc_sharpe(returns_30d, periods=720),

    # Fill Quality
    "fill_rate_24h": fills_executed / signals_generated,
    "avg_slippage_bps": mean([abs(fill_price - signal_price) / signal_price * 10000]),
    "partial_fill_rate": partial_fills / total_fills,
}
```

#### Alert Thresholds

| Metric | Warning | Critical | Action |
|--------|---------|----------|--------|
| Daily PnL | < -3% | < -5% | Hard stop |
| Current DD | > 15% | > 25% | Hard stop |
| Sharpe 7d | < 1.0 | < 0.5 | Review |
| Fill Rate | < 90% | < 80% | Hard stop |
| Slippage | > 20 bps | > 50 bps | Review |

### 2.2 Archetype Firing Rate (Update Every 1min)

```python
# Signal generation monitoring
archetype_metrics = {
    "signals_per_hour": {
        "A": 3.2,   # Spring (expected: 2-5/hr in bull)
        "C": 8.7,   # Liquidity Sweep (expected: 5-12/hr)
        "S1": 2.1,  # Liquidity Vacuum (expected: 1-3/hr in bear)
        # ... all 16 archetypes
    },
    "firing_rate_deviation": {
        "A": +12%,   # Within tolerance (<30%)
        "C": +45%,   # ⚠️ ALERT: Potential over-firing
        "S1": -5%,   # ✅ Normal
    },
    "dry_spell_duration": {
        "A": 0.8,   # Hours since last signal (normal)
        "S8": 6.2,  # ⚠️ ALERT: Expected signal every 4h
    }
}
```

**Alert Conditions**:
- Archetype silent for >2x expected interval
- Firing rate >50% above backtest average (potential overfitting to live conditions)
- Zero signals from all archetypes in regime for >1 hour (system failure)

### 2.3 Domain Boost Activation (Update Every 5min)

```python
# Metadata integrity check
domain_metrics = {
    "boost_activation_rate": {
        "wyckoff_events": 0.78,      # 78% of signals have Wyckoff boost
        "smc_4h_bos": 0.65,           # 65% have SMC boost
        "temporal_confluence": 0.42,  # 42% have temporal boost
        "hob_liquidity": 0.58,        # 58% have liquidity boost
        "macro_regime": 0.92,         # 92% have macro alignment
    },
    "veto_activation_rate": {
        "wyckoff_distribution": 0.15,  # 15% vetoed by Wyckoff
        "smc_choch_counter": 0.08,     # 8% vetoed by SMC
        "temporal_misalignment": 0.22, # 22% vetoed by temporal
    },
    "boost_multiplier_distribution": {
        "mean": 1.28,      # Expected: 1.2-1.4
        "std": 0.15,       # Expected: <0.2
        "min": 1.05,       # Expected: >1.0
        "max": 1.65,       # Expected: <2.0
    }
}
```

**Integrity Alerts**:
- Boost activation <50% for any engine (metadata failure?)
- Veto rate >40% (overly conservative, need recalibration)
- Boost multiplier mean <1.1 (boosts not working)
- Boost multiplier std >0.3 (inconsistent quality scoring)

### 2.4 Regime Detection Health (Update Every 1min)

```python
# HMM regime classifier monitoring
regime_metrics = {
    "current_regime": "risk_on",
    "regime_confidence": 0.87,         # HMM posterior probability
    "regime_stability": 0.92,          # 1 - transition_rate
    "detection_lag_minutes": 12,       # Time from market shift to HMM update

    "regime_distribution_24h": {
        "risk_on": 0.58,    # 58% of time in risk_on
        "risk_off": 0.25,   # 25% in risk_off
        "crisis": 0.08,     # 8% in crisis
        "neutral": 0.09,    # 9% in neutral
    },

    "transition_frequency": {
        "last_1h": 0,       # 0 transitions (stable)
        "last_4h": 1,       # 1 transition (normal)
        "last_24h": 4,      # 4 transitions (normal volatility)
    },

    "manual_label_agreement": 0.88,  # HMM agrees with expert labels 88%
}
```

**Regime Alerts**:
- Regime thrashing: >5 transitions in 1 hour (HMM unstable)
- Detection lag >30 minutes (missed regime shifts)
- Confidence <0.6 for >4 hours (unclear regime)
- Manual agreement <70% (HMM miscalibrated)

---

## 3. Drift Detection System

### 3.1 Feature Drift (Update Daily)

#### Population Stability Index (PSI)

```python
def calculate_psi(expected_dist, actual_dist, bins=10):
    """
    PSI measures distribution drift between backtest and live data.

    Thresholds:
    - PSI < 0.1: No drift
    - 0.1 < PSI < 0.25: Minor drift (monitor)
    - PSI > 0.25: Significant drift (ALERT)
    """
    expected_percents = np.histogram(expected_dist, bins=bins)[0] / len(expected_dist)
    actual_percents = np.histogram(actual_dist, bins=bins)[0] / len(actual_dist)

    psi = sum([
        (actual - expected) * np.log(actual / expected)
        for actual, expected in zip(actual_percents, expected_percents)
        if actual > 0 and expected > 0
    ])
    return psi

# Monitor top 20 features
feature_drift_report = {
    "wyckoff_spring_strength": {
        "psi": 0.08,  # ✅ No drift
        "status": "STABLE"
    },
    "smc_4h_bos_bullish": {
        "psi": 0.28,  # ⚠️ ALERT: Significant drift
        "status": "DRIFTING",
        "action": "Recalibrate archetype C thresholds"
    },
    "temporal_4h_alignment": {
        "psi": 0.15,  # ⚠️ Minor drift
        "status": "MONITOR"
    }
}
```

#### Kolmogorov-Smirnov Test

```python
from scipy.stats import ks_2samp

def detect_distribution_shift(backtest_data, live_data, alpha=0.05):
    """
    KS test for distribution changes.

    Returns:
    - statistic: KS distance (0-1)
    - p_value: Probability of null hypothesis (same distribution)
    - drift_detected: True if p < alpha
    """
    statistic, p_value = ks_2samp(backtest_data, live_data)
    return {
        "ks_statistic": statistic,
        "p_value": p_value,
        "drift_detected": p_value < alpha,
        "severity": "HIGH" if statistic > 0.2 else "MEDIUM" if statistic > 0.1 else "LOW"
    }

# Apply to all features daily
drift_tests = {
    feature_name: detect_distribution_shift(
        backtest_features[feature_name],
        live_features[feature_name]
    )
    for feature_name in critical_features
}
```

### 3.2 Performance Drift (Update Hourly)

#### CUSUM (Cumulative Sum) for Win Rate

```python
def cusum_win_rate(trades, expected_win_rate=0.58, drift_threshold=5.0):
    """
    Detect win rate degradation using CUSUM.

    Alerts when CUSUM exceeds drift_threshold (indicates sustained deviation).
    """
    cusum_pos = 0
    cusum_neg = 0
    alerts = []

    for i, trade in enumerate(trades):
        win = 1 if trade.pnl > 0 else 0
        deviation = win - expected_win_rate

        cusum_pos = max(0, cusum_pos + deviation - 0.01)  # Drift allowance
        cusum_neg = min(0, cusum_neg + deviation + 0.01)

        if cusum_pos > drift_threshold:
            alerts.append({"trade_idx": i, "type": "POSITIVE_DRIFT", "cusum": cusum_pos})
        if cusum_neg < -drift_threshold:
            alerts.append({"trade_idx": i, "type": "NEGATIVE_DRIFT", "cusum": cusum_neg})

    return {
        "cusum_positive": cusum_pos,
        "cusum_negative": cusum_neg,
        "alerts": alerts,
        "trend": "IMPROVING" if cusum_pos > 2 else "DEGRADING" if cusum_neg < -2 else "STABLE"
    }
```

#### Rolling Correlation vs Backtest

```python
def backtest_correlation(live_returns, backtest_returns, window=168):
    """
    Measure how closely live performance tracks backtest patterns.

    Low correlation (<0.5) suggests regime shift or model breakdown.
    """
    rolling_corr = []
    for i in range(len(live_returns) - window):
        live_window = live_returns[i:i+window]
        backtest_window = backtest_returns[i:i+window]
        corr = np.corrcoef(live_window, backtest_window)[0, 1]
        rolling_corr.append(corr)

    return {
        "current_correlation": rolling_corr[-1] if rolling_corr else None,
        "mean_correlation_7d": np.mean(rolling_corr[-168:]) if len(rolling_corr) > 168 else None,
        "correlation_trend": np.polyfit(range(len(rolling_corr[-168:])), rolling_corr[-168:], 1)[0],
        "alert": rolling_corr[-1] < 0.5 if rolling_corr else False
    }
```

### 3.3 Regime Drift (Update Every 15min)

```python
# Regime stability monitoring
regime_drift_metrics = {
    "transition_matrix_deviation": {
        "risk_on -> risk_off": {
            "backtest_prob": 0.12,
            "live_prob": 0.18,
            "deviation": +50%,  # ⚠️ ALERT: Transitions happening more frequently
        },
        "crisis -> risk_on": {
            "backtest_prob": 0.08,
            "live_prob": 0.05,
            "deviation": -37%,  # Slower recovery from crisis
        }
    },

    "regime_duration_stats": {
        "risk_on": {
            "backtest_avg_hours": 48,
            "live_avg_hours": 36,
            "deviation": -25%,  # Shorter bull runs
        },
        "crisis": {
            "backtest_avg_hours": 12,
            "live_avg_hours": 18,
            "deviation": +50%,  # ⚠️ Crisis periods lasting longer
        }
    },

    "hmm_confidence_trend": {
        "7d_avg": 0.82,
        "30d_avg": 0.87,
        "trend": "DECLINING",  # ⚠️ HMM becoming less confident
    }
}
```

---

## 4. Archetype-Level Diagnostics

### 4.1 Individual Archetype Dashboard

Each archetype gets a dedicated monitoring panel:

```
┌─────────────────────────────────────────────────────────────────────┐
│ ARCHETYPE A: SPRING (Wyckoff Reversal)                              │
│ Status: ✅ HEALTHY | Last Signal: 23min ago | Active Positions: 1  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│ PERFORMANCE SNAPSHOT                                                 │
│ ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│ │ Win Rate         │  │ Avg Confidence   │  │ Sharpe Ratio     │  │
│ │ 62% ✅           │  │ 0.87 ⚠️ -5%     │  │ 1.92 ✅          │  │
│ │ Expected: 58%    │  │ Expected: 0.92   │  │ Expected: 1.85   │  │
│ └──────────────────┘  └──────────────────┘  └──────────────────┘  │
│                                                                      │
│ SIGNAL ACTIVITY (Last 7 Days)                                       │
│ Signals Generated: 85                                                │
│ Signals Executed: 81 (95% fill rate ✅)                             │
│ Avg Slippage: 0.12% (12 bps ✅)                                     │
│ Position Hold Time: 6.8h (expected: 6.2h, +10% ✅)                  │
│                                                                      │
│ DOMAIN BOOST ACTIVATION                                             │
│ ┌────────────────────────────────────────────────────────────────┐ │
│ │ Wyckoff Events:   ████████████████████░░ 92% ✅               │ │
│ │ SMC 4H BOS:       ███████████████░░░░░░░ 78% ✅               │ │
│ │ Temporal 4H:      █████████░░░░░░░░░░░░░ 45% ⚠️ Low          │ │
│ │ HOB Liquidity:    ██████████████░░░░░░░░ 68% ✅               │ │
│ │ Macro Alignment:  ████████████████████░░ 96% ✅               │ │
│ └────────────────────────────────────────────────────────────────┘ │
│                                                                      │
│ REGIME DISTRIBUTION                                                  │
│ ┌────────────────────────────────────────────────────────────────┐ │
│ │ risk_on:  ██████████████░░░░░░ 70% ✅ (Expected regime)      │ │
│ │ neutral:  ██████░░░░░░░░░░░░░░ 30% ✅                         │ │
│ │ risk_off: ░░░░░░░░░░░░░░░░░░░░  0% ✅ (Correctly filtered)   │ │
│ │ crisis:   ░░░░░░░░░░░░░░░░░░░░  0% ✅ (Correctly filtered)   │ │
│ └────────────────────────────────────────────────────────────────┘ │
│                                                                      │
│ RECENT SIGNALS                                                       │
│ ┌──────────────────────────────────────────────────────────────┐   │
│ │ Time         │ Price    │ Dir  │ Conf  │ Boosts      │ PnL   │   │
│ ├──────────────────────────────────────────────────────────────┤   │
│ │ 23min ago    │ $43,250  │ LONG │ 0.91  │ W+S+M       │ +$120 │   │
│ │ 4h ago       │ $42,980  │ LONG │ 0.88  │ W+S         │ +$340 │   │
│ │ 8h ago       │ $42,750  │ LONG │ 0.84  │ W+H         │ -$80  │   │
│ └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│ 🚨 ALERTS                                                            │
│ ⚠️ Temporal boost activation at 45% (expected >60%)                │
│ ⚠️ Confidence scores 5% below backtest average                     │
│ ℹ️ Position hold time increasing (+10% trend)                      │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 Archetype Health Scorecard

```python
def calculate_archetype_health_score(archetype_id, live_metrics, backtest_baseline):
    """
    Composite health score (0-100) based on multiple factors.

    Components:
    - Performance alignment (40%): Win rate, Sharpe, DD within tolerance
    - Metadata integrity (30%): Boost/veto rates as expected
    - Regime alignment (20%): Firing in correct regimes
    - Execution quality (10%): Fill rate, slippage
    """

    # Performance alignment (0-40 points)
    win_rate_score = max(0, 40 - abs(live_metrics.win_rate - backtest_baseline.win_rate) * 200)
    sharpe_score = max(0, 40 - abs(live_metrics.sharpe - backtest_baseline.sharpe) / backtest_baseline.sharpe * 100)
    performance_score = (win_rate_score + sharpe_score) / 2

    # Metadata integrity (0-30 points)
    boost_rate_score = min(30, live_metrics.boost_activation_rate * 30)
    veto_alignment = 1 - abs(live_metrics.veto_rate - backtest_baseline.veto_rate)
    metadata_score = (boost_rate_score + veto_alignment * 30) / 2

    # Regime alignment (0-20 points)
    regime_misfire_penalty = live_metrics.wrong_regime_signals * 5
    regime_score = max(0, 20 - regime_misfire_penalty)

    # Execution quality (0-10 points)
    fill_rate_score = live_metrics.fill_rate * 10
    slippage_penalty = min(5, live_metrics.avg_slippage_bps / 10)
    execution_score = fill_rate_score - slippage_penalty

    total_score = performance_score + metadata_score + regime_score + execution_score

    return {
        "total_score": total_score,
        "grade": "A" if total_score >= 90 else "B" if total_score >= 75 else "C" if total_score >= 60 else "F",
        "breakdown": {
            "performance": performance_score,
            "metadata": metadata_score,
            "regime": regime_score,
            "execution": execution_score
        },
        "status": "HEALTHY" if total_score >= 75 else "DEGRADED" if total_score >= 60 else "CRITICAL"
    }
```

### 4.3 Alert Conditions per Archetype

| Condition | Threshold | Severity | Action |
|-----------|-----------|----------|--------|
| Win rate deviation | >15% | WARNING | Manual review |
| Win rate deviation | >30% | CRITICAL | Disable archetype |
| Confidence drift | >10% | WARNING | Recalibrate thresholds |
| Boost activation | <50% | CRITICAL | Check metadata pipeline |
| Wrong regime signals | >5 in 24h | WARNING | Review regime filter |
| Wrong regime signals | >15 in 24h | CRITICAL | Disable archetype |
| Fill rate | <90% | WARNING | Check execution |
| Fill rate | <80% | CRITICAL | Halt trading |
| Consecutive losses | >8 | WARNING | Reduce position size |
| Consecutive losses | >12 | CRITICAL | Disable archetype |

---

## 5. Ensemble-Level Diagnostics

### 5.1 Portfolio Health Dashboard

```
┌─────────────────────────────────────────────────────────────────────┐
│ ENSEMBLE HEALTH MONITOR                                              │
│ Active Archetypes: 14/16 | Portfolio Sharpe: 1.68 | Diversity: 82%  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│ SIGNAL DIVERSITY ANALYSIS                                            │
│ ┌──────────────────────────────────────────────────────────────┐   │
│ │ Overlap Rate: 42% ✅ (Target: 35-45%)                       │   │
│ │ Unique Signal Timestamps: 68% ✅                             │   │
│ │ Archetype Correlation (avg): 0.28 ✅ (Target: <0.4)         │   │
│ └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│ ARCHETYPE CORRELATION MATRIX                                         │
│     A    C    S1   S2   S4   S5   S8   B0   B1   B2                │
│ A  1.00 0.32 0.15 0.22 0.18 0.25 0.08 0.41 0.28 0.19               │
│ C  0.32 1.00 0.52⚠️0.28 0.31 0.35 0.12 0.38 0.42 0.25              │
│ S1 0.15 0.52⚠️1.00 0.44 0.38 0.29 0.18 0.22 0.31 0.28              │
│                                                                      │
│ ⚠️ High correlation detected: C ↔ S1 (0.52)                        │
│ → Recommendation: Review archetype logic overlap                    │
│                                                                      │
│ REGIME COVERAGE (Last 24h)                                           │
│ ┌────────────────────────────────────────────────────────────────┐ │
│ │ Regime     │ Duration │ Active Archetypes │ Signal Count      │ │
│ ├────────────────────────────────────────────────────────────────┤ │
│ │ risk_on    │ 14h      │ A,C,S8,B0,B1,B2  │ 142 ✅           │ │
│ │ neutral    │ 6h       │ C,S1,S4          │ 38 ✅            │ │
│ │ risk_off   │ 3h       │ S1,S2,S4,S5      │ 24 ✅            │ │
│ │ crisis     │ 1h       │ S1,S5            │ 8 ✅             │ │
│ └────────────────────────────────────────────────────────────────┘ │
│                                                                      │
│ CAPITAL ALLOCATION                                                   │
│ Total Capital: $100,000                                              │
│ Deployed: $45,200 (45.2% ✅ Target: 40-60%)                         │
│ Reserved: $54,800 (54.8%)                                            │
│                                                                      │
│ ┌────────────────────────────────────────────────────────────────┐ │
│ │ Archetype │ Positions │ Capital │ PnL 24h │ Contribution     │ │
│ ├────────────────────────────────────────────────────────────────┤ │
│ │ A         │ 2         │ $8,400  │ +$460   │ 32% ✅           │ │
│ │ C         │ 4         │ $12,600 │ +$320   │ 22% ⚠️ Overlap  │ │
│ │ S1        │ 3         │ $9,200  │ +$180   │ 12% ✅           │ │
│ │ S8        │ 1         │ $5,000  │ +$120   │ 8% ✅            │ │
│ │ Others    │ 5         │ $10,000 │ +$340   │ 26% ✅           │ │
│ └────────────────────────────────────────────────────────────────┘ │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.2 Signal Overlap Analysis

```python
def analyze_signal_overlap(signals_df, window_minutes=15):
    """
    Detect concurrent signals to measure ensemble diversity.

    High overlap (>50%) indicates redundancy.
    Low overlap (<30%) indicates insufficient confirmation.
    """
    overlap_events = []

    for timestamp in signals_df.timestamp.unique():
        window_start = timestamp - pd.Timedelta(minutes=window_minutes)
        window_end = timestamp + pd.Timedelta(minutes=window_minutes)

        concurrent_signals = signals_df[
            (signals_df.timestamp >= window_start) &
            (signals_df.timestamp <= window_end)
        ]

        if len(concurrent_signals) > 1:
            overlap_events.append({
                "timestamp": timestamp,
                "archetypes": concurrent_signals.archetype_id.tolist(),
                "count": len(concurrent_signals),
                "direction_agreement": len(concurrent_signals.direction.unique()) == 1,
                "avg_confidence": concurrent_signals.confidence.mean()
            })

    return {
        "total_signals": len(signals_df),
        "overlap_events": len(overlap_events),
        "overlap_rate": len(overlap_events) / len(signals_df),
        "avg_overlap_size": np.mean([e["count"] for e in overlap_events]),
        "direction_agreement_rate": sum([e["direction_agreement"] for e in overlap_events]) / len(overlap_events),
        "high_overlap_archetypes": identify_frequent_pairs(overlap_events)
    }
```

### 5.3 Ensemble Health Metrics

```python
ensemble_health = {
    "diversity_score": {
        "signal_overlap_rate": 0.42,          # 42% of signals overlap (target: 35-45%)
        "archetype_correlation_avg": 0.28,    # Low correlation = good diversity
        "unique_timestamp_rate": 0.68,        # 68% of signals at unique times
        "grade": "A"  # A: >0.6, B: >0.5, C: >0.4, F: <0.4
    },

    "regime_coverage": {
        "risk_on_uptime": 0.95,   # 95% of risk_on periods had active signals
        "risk_off_uptime": 0.88,  # 88% of risk_off periods had active signals
        "crisis_uptime": 0.72,    # 72% of crisis periods had active signals
        "neutral_uptime": 0.65,   # 65% of neutral periods had active signals
        "grade": "B+"
    },

    "capital_efficiency": {
        "utilization_rate": 0.45,           # 45% of capital deployed
        "avg_position_duration_hours": 6.8, # Turnover rate
        "idle_capital_opportunity_cost": 120, # $ lost from unused capital
        "grade": "A"
    },

    "meta_model_readiness": {
        "overlap_data_points": 1240,        # Number of overlap events for training
        "confluence_strength_variance": 0.15, # Diversity in 2-of-3, 3-of-4 patterns
        "boost_multiplier_range": [1.05, 1.65], # Quality score distribution
        "ready_for_meta_learner": True
    }
}
```

---

## 6. Kill-Switch System

### 6.1 Hard Stop Conditions (Immediate Halt)

```python
HARD_STOP_CONDITIONS = {
    "daily_loss_limit": {
        "threshold": -5.0,  # -5% daily loss
        "check_interval_seconds": 60,
        "action": "HALT_ALL_TRADING",
        "notification": ["SMS", "EMAIL", "SLACK_URGENT"]
    },

    "drawdown_limit": {
        "threshold": -25.0,  # -25% from peak
        "check_interval_seconds": 60,
        "action": "HALT_ALL_TRADING",
        "notification": ["SMS", "EMAIL", "SLACK_URGENT"]
    },

    "fill_rate_collapse": {
        "threshold": 0.80,  # <80% fill rate
        "lookback_hours": 4,
        "min_signals": 10,  # Need at least 10 signals to trigger
        "action": "HALT_NEW_POSITIONS",
        "notification": ["EMAIL", "SLACK_URGENT"]
    },

    "regime_thrashing": {
        "threshold": 5,  # >5 regime transitions in 1 hour
        "check_interval_seconds": 300,
        "action": "HALT_NEW_POSITIONS",
        "notification": ["EMAIL", "SLACK"]
    },

    "metadata_failure": {
        "condition": "boost_activation_rate < 0.1 AND veto_activation_rate < 0.05",
        "duration_minutes": 30,  # Persist for 30min to avoid false alarms
        "action": "HALT_ALL_TRADING",
        "notification": ["SMS", "EMAIL", "SLACK_URGENT"]
    },

    "execution_failure": {
        "threshold": 10,  # >10 consecutive execution errors
        "action": "HALT_ALL_TRADING",
        "notification": ["SMS", "EMAIL", "SLACK_URGENT"]
    }
}
```

### 6.2 Soft Stop Conditions (Manual Review Required)

```python
SOFT_STOP_CONDITIONS = {
    "win_rate_degradation": {
        "threshold": 0.40,  # Win rate <40%
        "expected": 0.55,
        "lookback_trades": 50,
        "action": "ALERT_OPERATOR",
        "notification": ["EMAIL", "SLACK"],
        "suggested_action": "Review recent trades, check for regime mismatch"
    },

    "sharpe_decline": {
        "threshold": 0.5,  # Sharpe <0.5 for 7 days
        "duration_days": 7,
        "action": "ALERT_OPERATOR",
        "notification": ["EMAIL"],
        "suggested_action": "Consider reducing position sizes or disabling underperformers"
    },

    "archetype_degradation": {
        "threshold": 3,  # >3 archetypes with health score <60
        "action": "ALERT_OPERATOR",
        "notification": ["EMAIL", "SLACK"],
        "suggested_action": "Recalibrate thresholds or disable degraded archetypes"
    },

    "feature_drift_critical": {
        "threshold": 5,  # >5 features with PSI >0.35
        "action": "ALERT_OPERATOR",
        "notification": ["EMAIL", "SLACK"],
        "suggested_action": "Retrain feature engineering pipeline or halt affected archetypes"
    },

    "slippage_increase": {
        "threshold": 50,  # >50 bps avg slippage
        "expected": 15,   # Expected <15 bps
        "lookback_hours": 24,
        "action": "ALERT_OPERATOR",
        "notification": ["EMAIL"],
        "suggested_action": "Review execution venue, consider limit orders"
    }
}
```

### 6.3 Kill-Switch Control Panel

```
┌─────────────────────────────────────────────────────────────────────┐
│ 🚨 KILL-SWITCH CONTROL PANEL                                        │
│ System Status: ACTIVE | Last Check: 5s ago                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│ HARD STOP CONDITIONS                                                 │
│ ┌──────────────────────────────────────────────────────────────┐   │
│ │ Condition           │ Current  │ Threshold │ Status         │   │
│ ├──────────────────────────────────────────────────────────────┤   │
│ │ Daily Loss          │ -2.8%    │ -5.0%     │ ✅ SAFE        │   │
│ │ Drawdown from Peak  │ -12.5%   │ -25.0%    │ ✅ SAFE        │   │
│ │ Fill Rate (4h)      │ 94%      │ <80%      │ ✅ SAFE        │   │
│ │ Regime Thrashing    │ 1/hr     │ >5/hr     │ ✅ SAFE        │   │
│ │ Metadata Integrity  │ Active   │ Failure   │ ✅ SAFE        │   │
│ │ Execution Errors    │ 2        │ >10       │ ✅ SAFE        │   │
│ └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│ SOFT STOP CONDITIONS                                                 │
│ ┌──────────────────────────────────────────────────────────────┐   │
│ │ Condition           │ Current  │ Threshold │ Status         │   │
│ ├──────────────────────────────────────────────────────────────┤   │
│ │ Win Rate (50 trades)│ 61%      │ <40%      │ ✅ SAFE        │   │
│ │ Sharpe 7d           │ 1.42     │ <0.5      │ ✅ SAFE        │   │
│ │ Degraded Archetypes │ 1        │ >3        │ ✅ SAFE        │   │
│ │ Feature Drift (PSI) │ 2        │ >5        │ ✅ SAFE        │   │
│ │ Slippage (24h)      │ 18 bps   │ >50 bps   │ ✅ SAFE        │   │
│ └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│ MANUAL CONTROLS                                                      │
│ ┌──────────────────────────────────────────────────────────────┐   │
│ │ [🛑 EMERGENCY HALT]  - Stop all trading immediately          │   │
│ │ [⏸️ PAUSE NEW POSITIONS] - Close existing, block new entries │   │
│ │ [🔄 RESTART SYSTEM]  - Reset all monitors and resume          │   │
│ │ [📊 EXPORT LOGS]     - Download diagnostic data              │   │
│ └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│ RECENT ALERTS (Last 24h)                                             │
│ 08:42 ⚠️ Archetype C overlap at 48% (target <45%)                  │
│ 06:15 ℹ️ Regime transition: risk_on → neutral                      │
│ 02:30 ⚠️ Feature drift: smc_4h_bos_bullish (PSI: 0.28)            │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 7. Comparison Framework

### 7.1 Backtest vs Paper Trading Comparison

```python
comparison_metrics = {
    "performance": {
        "sharpe_ratio": {
            "backtest": 1.85,
            "paper": 1.62,
            "deviation_pct": -12.4,
            "tolerance": 20,  # ±20%
            "status": "✅ WITHIN_TOLERANCE",
            "acceptable_range": [1.48, 2.22]
        },
        "win_rate": {
            "backtest": 0.58,
            "paper": 0.61,
            "deviation_pct": +5.2,
            "tolerance": 10,
            "status": "✅ WITHIN_TOLERANCE",
            "acceptable_range": [0.52, 0.64]
        },
        "max_drawdown": {
            "backtest": -18.2,
            "paper": -12.5,
            "deviation_pct": "BETTER",
            "tolerance": -25,  # Failure if worse than -25%
            "status": "✅ IMPROVED"
        },
        "calmar_ratio": {
            "backtest": 2.1,
            "paper": 2.4,
            "deviation_pct": +14.3,
            "status": "✅ IMPROVED"
        }
    },

    "execution": {
        "fill_rate": {
            "backtest": 1.00,  # Simulated 100%
            "paper": 0.94,
            "deviation_pct": -6.0,
            "tolerance": 10,
            "status": "✅ ACCEPTABLE",
            "note": "Expected degradation from simulation"
        },
        "avg_slippage_bps": {
            "backtest": 0,
            "paper": 12,
            "tolerance": 20,
            "status": "✅ ACCEPTABLE"
        },
        "avg_trade_duration_hours": {
            "backtest": 6.2,
            "paper": 7.1,
            "deviation_pct": +14.5,
            "status": "⚠️ MONITOR",
            "note": "Positions held longer than backtest"
        }
    },

    "signal_quality": {
        "signal_overlap_rate": {
            "backtest": 0.42,
            "paper": 0.48,
            "deviation_pct": +14.3,
            "target_range": [0.35, 0.50],
            "status": "⚠️ HIGH_END",
            "note": "Approaching upper limit"
        },
        "avg_confidence_score": {
            "backtest": 0.87,
            "paper": 0.84,
            "deviation_pct": -3.4,
            "status": "✅ STABLE"
        },
        "boost_activation_rate": {
            "backtest": 0.72,
            "paper": 0.68,
            "deviation_pct": -5.6,
            "status": "✅ STABLE"
        }
    }
}
```

### 7.2 Statistical Significance Tests

```python
from scipy import stats

def test_performance_significance(backtest_returns, paper_returns, alpha=0.05):
    """
    Determine if performance difference is statistically significant.
    """
    # Welch's t-test (unequal variances)
    t_stat, p_value = stats.ttest_ind(paper_returns, backtest_returns, equal_var=False)

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.std(backtest_returns)**2 + np.std(paper_returns)**2) / 2)
    cohens_d = (np.mean(paper_returns) - np.mean(backtest_returns)) / pooled_std

    return {
        "t_statistic": t_stat,
        "p_value": p_value,
        "significant": p_value < alpha,
        "cohens_d": cohens_d,
        "effect_size": "large" if abs(cohens_d) > 0.8 else "medium" if abs(cohens_d) > 0.5 else "small",
        "interpretation": interpret_significance(p_value, cohens_d)
    }

def interpret_significance(p_value, cohens_d):
    if p_value >= 0.05:
        return "No significant difference - performance consistent with backtest"
    elif cohens_d > 0:
        return "Significant IMPROVEMENT over backtest"
    else:
        return "Significant DEGRADATION from backtest - INVESTIGATE"
```

### 7.3 Go-Live Acceptance Criteria

```python
GO_LIVE_CRITERIA = {
    "mandatory_pass": {
        "sharpe_ratio": {
            "condition": "paper_sharpe >= backtest_sharpe * 0.8",
            "current": True,
            "note": "Must be within 20% of backtest"
        },
        "max_drawdown": {
            "condition": "paper_dd > -25%",
            "current": True,
            "note": "Absolute limit"
        },
        "win_rate": {
            "condition": "paper_win_rate >= 0.50",
            "current": True,
            "note": "Must be profitable"
        },
        "fill_rate": {
            "condition": "paper_fill_rate >= 0.90",
            "current": True,
            "note": "Execution quality threshold"
        },
        "metadata_integrity": {
            "condition": "boost_activation_rate >= 0.50",
            "current": True,
            "note": "Domain engines must be active"
        },
        "regime_detection": {
            "condition": "hmm_accuracy >= 0.70",
            "current": True,
            "note": "Regime classifier must be reliable"
        },
        "archetype_health": {
            "condition": "healthy_archetypes >= 12",  # 75% of 16
            "current": True,
            "note": "Majority must be functional"
        }
    },

    "nice_to_have": {
        "sharpe_improvement": {
            "condition": "paper_sharpe > backtest_sharpe",
            "current": False,
            "impact": "LOW"
        },
        "low_slippage": {
            "condition": "avg_slippage < 15 bps",
            "current": False,
            "impact": "MEDIUM"
        },
        "high_diversity": {
            "condition": "signal_overlap_rate < 0.40",
            "current": False,
            "impact": "MEDIUM"
        }
    },

    "overall_readiness": {
        "mandatory_passed": 7,
        "mandatory_total": 7,
        "nice_to_have_passed": 1,
        "nice_to_have_total": 3,
        "ready_for_live": True,
        "confidence": "HIGH"
    }
}
```

---

## 8. Dashboard Design

### 8.1 Main Dashboard (Overview)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ 🎯 BULL MACHINE PAPER TRADING DASHBOARD                                      │
│ Status: 🟢 ACTIVE | Uptime: 12d 4h 23m | Last Update: 3s ago                │
│ Paper Start: Dec 1, 2025 | Days to Live: 48 days remaining                  │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│ ┌────────── CRITICAL METRICS ──────────┐                                     │
│ │                                       │                                     │
│ │  PnL (Today)      Drawdown   Sharpe  │                                     │
│ │  +$1,240 (+1.2%)  -8.3%      1.62    │                                     │
│ │  ✅ On Track      ✅ Safe    ⚠️ -12% │                                     │
│ │                                       │                                     │
│ │  Fill Rate        Signals    Overlap │                                     │
│ │  94%              142/day    42%     │                                     │
│ │  ✅ Good          ✅ Active  ✅ OK   │                                     │
│ └───────────────────────────────────────┘                                     │
│                                                                               │
│ ┌──────────────────── PERFORMANCE CHART (7-Day Rolling) ──────────────────┐  │
│ │                                                                          │  │
│ │  PnL %                                                                   │  │
│ │   15% ┤                                                        ╭─────    │  │
│ │   10% ┤                                             ╭──────────╯         │  │
│ │    5% ┤                                   ╭─────────╯                    │  │
│ │    0% ┼───────────────────────────────────╯                             │  │
│ │   -5% ┤                                                                  │  │
│ │       └──────────────────────────────────────────────────────────────   │  │
│ │        Dec 10    Dec 12    Dec 14    Dec 16    Dec 18    Dec 20        │  │
│ │                                                                          │  │
│ │  [Sharpe: 1.62] [Win Rate: 61%] [Max DD: -12.5%]                        │  │
│ └──────────────────────────────────────────────────────────────────────────┘  │
│                                                                               │
│ ┌─────────── ARCHETYPE HEALTH ──────────┐ ┌──── REGIME STATUS ────┐         │
│ │                                        │ │                        │         │
│ │ Active: 14/16 | Health Score: 82/100  │ │ Current: risk_on       │         │
│ │                                        │ │ Confidence: 87%        │         │
│ │ ✅ A (Spring)          Score: 88      │ │ Duration: 3h 24m       │         │
│ │ ⚠️ C (Liquidity Sweep) Score: 72      │ │                        │         │
│ │ ✅ S1 (Vacuum)         Score: 91      │ │ Distribution (24h):    │         │
│ │ ✅ S8 (Funding Div)    Score: 85      │ │ ████ risk_on    45%   │         │
│ │ ✅ B0 (BOS/CHOCH)      Score: 94      │ │ ███ risk_off    25%   │         │
│ │                                        │ │ ██ neutral      20%   │         │
│ │ [View All 16 →]                        │ │ █ crisis        10%   │         │
│ │                                        │ │                        │         │
│ └────────────────────────────────────────┘ └────────────────────────┘         │
│                                                                               │
│ ┌────────────────── DOMAIN ENGINE STATUS ──────────────────┐                 │
│ │                                                           │                 │
│ │ Wyckoff Events:     ████████████████░░ 78% ✅            │                 │
│ │ SMC 4H BOS:         ██████████████░░░░ 68% ✅            │                 │
│ │ Temporal 4H:        ████████░░░░░░░░░░ 42% ⚠️ Low       │                 │
│ │ HOB Liquidity:      ███████████░░░░░░░ 58% ✅            │                 │
│ │ Macro Regime:       ████████████████████ 92% ✅          │                 │
│ │                                                           │                 │
│ └───────────────────────────────────────────────────────────┘                 │
│                                                                               │
│ ┌────────────────────────── 🚨 ALERTS ────────────────────────────┐          │
│ │                                                                  │          │
│ │ 🔴 CRITICAL (0)                                                  │          │
│ │ None - All systems nominal                                       │          │
│ │                                                                  │          │
│ │ ⚠️ WARNINGS (3)                                                  │          │
│ │ • Archetype C overlap at 48% (target <45%) [2h ago]             │          │
│ │ • Feature drift: smc_4h_bos_bullish (PSI: 0.28) [8h ago]        │          │
│ │ • Temporal boost activation at 42% (expected >60%) [12h ago]    │          │
│ │                                                                  │          │
│ │ ℹ️ INFO (5)                                                      │          │
│ │ • Regime transition: neutral → risk_on [45m ago]                │          │
│ │ • Archetype A generated 12 signals (last 4h)                    │          │
│ │ [View All Alerts →]                                              │          │
│ │                                                                  │          │
│ └──────────────────────────────────────────────────────────────────┘          │
│                                                                               │
│ ┌──────────── QUICK ACTIONS ────────────┐                                    │
│ │ [📊 View Detailed Report]              │                                    │
│ │ [🔍 Drill into Archetype C]            │                                    │
│ │ [📈 Compare vs Backtest]               │                                    │
│ │ [🚨 Kill-Switch Panel]                 │                                    │
│ │ [📥 Export Logs]                       │                                    │
│ └────────────────────────────────────────┘                                    │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 8.2 Drill-Down: Archetype Detail View

(See Section 4.1 for full archetype dashboard)

### 8.3 Drill-Down: Drift Detection View

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ 📊 DRIFT DETECTION DASHBOARD                                                 │
│ Feature Stability Analysis | Last Update: 1h ago                             │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│ FEATURE DRIFT SUMMARY                                                         │
│ Total Features Monitored: 47                                                 │
│ Stable: 38 (81%) | Minor Drift: 7 (15%) | Critical Drift: 2 (4%) ⚠️         │
│                                                                               │
│ ┌─────────────────────── CRITICAL DRIFT ALERTS ─────────────────────────┐   │
│ │                                                                        │   │
│ │ Feature: smc_4h_bos_bullish                                            │   │
│ │ PSI: 0.28 ⚠️ (Threshold: 0.25)                                        │   │
│ │ KS Statistic: 0.18 (p-value: 0.02) - SIGNIFICANT SHIFT                │   │
│ │                                                                        │   │
│ │ Distribution Comparison:                                               │   │
│ │ Backtest: Mean=0.42, Std=0.15 | Paper: Mean=0.51, Std=0.18            │   │
│ │                                                                        │   │
│ │ Impact: Affects Archetype C (Liquidity Sweep)                         │   │
│ │ Action: Recalibrate thresholds or monitor for 48h                     │   │
│ │                                                                        │   │
│ │ ────────────────────────────────────────────────────────────────────  │   │
│ │                                                                        │   │
│ │ Feature: temporal_4h_alignment                                         │   │
│ │ PSI: 0.31 🔴 (Threshold: 0.25)                                        │   │
│ │ KS Statistic: 0.22 (p-value: 0.008) - SIGNIFICANT SHIFT               │   │
│ │                                                                        │   │
│ │ Distribution Comparison:                                               │   │
│ │ Backtest: Mean=0.68, Std=0.12 | Paper: Mean=0.54, Std=0.16            │   │
│ │                                                                        │   │
│ │ Impact: Affects 8 archetypes (A, C, S1, S4, B0, B1, B2, S8)           │   │
│ │ Action: URGENT - Review temporal feature engineering                  │   │
│ │                                                                        │   │
│ └────────────────────────────────────────────────────────────────────────┘   │
│                                                                               │
│ ┌─────────────────────── MINOR DRIFT (Monitoring) ──────────────────────┐   │
│ │ Feature                        │ PSI   │ Status      │ Trend          │   │
│ ├────────────────────────────────────────────────────────────────────────┤   │
│ │ wyckoff_spring_strength        │ 0.15  │ ⚠️ Monitor  │ Stable         │   │
│ │ smc_order_block_quality        │ 0.18  │ ⚠️ Monitor  │ Increasing     │   │
│ │ hob_liquidity_score            │ 0.12  │ ⚠️ Monitor  │ Decreasing     │   │
│ │ macro_risk_score               │ 0.21  │ ⚠️ Monitor  │ Stable         │   │
│ └────────────────────────────────────────────────────────────────────────┘   │
│                                                                               │
│ PERFORMANCE DRIFT (CUSUM Analysis)                                            │
│ ┌────────────────────────────────────────────────────────────────────────┐   │
│ │ Win Rate CUSUM:                                                        │   │
│ │   5 ┤                                                                  │   │
│ │   3 ┤                                         ╭────────────            │   │
│ │   1 ┤                           ╭─────────────╯                        │   │
│ │   0 ┼───────────────────────────╯                                      │   │
│ │  -1 ┤                                                                  │   │
│ │  -5 ┤                                                                  │   │
│ │     └──────────────────────────────────────────────────────────────   │   │
│ │      Dec 1   Dec 5   Dec 10   Dec 15   Dec 20                         │   │
│ │                                                                        │   │
│ │ Interpretation: POSITIVE DRIFT - Win rate improving ✅                │   │
│ │ Current CUSUM: +2.8 (Alert threshold: ±5.0)                           │   │
│ └────────────────────────────────────────────────────────────────────────┘   │
│                                                                               │
│ REGIME DRIFT                                                                  │
│ ┌────────────────────────────────────────────────────────────────────────┐   │
│ │ HMM Transition Matrix Deviation:                                       │   │
│ │                                                                        │   │
│ │           │ risk_on │ risk_off │ crisis │ neutral │                   │   │
│ │ ──────────┼─────────┼──────────┼────────┼─────────┤                   │   │
│ │ risk_on   │  0.92   │   0.06   │  0.01  │  0.01   │ ✅ Stable        │   │
│ │ risk_off  │  0.08   │   0.85   │  0.05  │  0.02   │ ✅ Stable        │   │
│ │ crisis    │  0.05   │   0.15   │  0.75  │  0.05   │ ⚠️ +12% vs BT    │   │
│ │ neutral   │  0.15   │   0.10   │  0.02  │  0.73   │ ✅ Stable        │   │
│ │                                                                        │   │
│ │ Alert: Crisis regime lasting 50% longer than backtest                 │   │
│ └────────────────────────────────────────────────────────────────────────┘   │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 8.4 Drill-Down: Comparison View (Backtest vs Paper)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ 📈 BACKTEST vs PAPER TRADING COMPARISON                                      │
│ Paper Period: Dec 1 - Dec 20, 2025 (20 days) | Backtest: 2022-2023          │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│ ┌────────────────── PERFORMANCE METRICS ─────────────────┐                   │
│ │                                                         │                   │
│ │ Metric              │ Backtest │ Paper   │ Deviation  │ Status           │
│ ├─────────────────────┼──────────┼─────────┼────────────┼──────────────────┤
│ │ Sharpe Ratio        │ 1.85     │ 1.62    │ -12%       │ ⚠️ Within tol   │
│ │ Sortino Ratio       │ 2.42     │ 2.18    │ -10%       │ ✅ Good         │
│ │ Calmar Ratio        │ 2.10     │ 2.40    │ +14%       │ ✅ Better       │
│ │ Win Rate            │ 58%      │ 61%     │ +5%        │ ✅ Improved     │
│ │ Max Drawdown        │ -18.2%   │ -12.5%  │ Better     │ ✅ Improved     │
│ │ Avg Drawdown        │ -5.8%    │ -4.2%   │ Better     │ ✅ Improved     │
│ │ Recovery Time (avg) │ 8.2h     │ 6.5h    │ -21%       │ ✅ Faster       │
│ │ Profit Factor       │ 1.85     │ 1.92    │ +4%        │ ✅ Improved     │
│ │                                                         │                   │
│ └─────────────────────────────────────────────────────────┘                   │
│                                                                               │
│ ┌────────────────── EXECUTION QUALITY ────────────────────┐                  │
│ │                                                          │                  │
│ │ Metric              │ Backtest │ Paper   │ Deviation   │ Status          │
│ ├─────────────────────┼──────────┼─────────┼─────────────┼─────────────────┤
│ │ Fill Rate           │ 100%     │ 94%     │ -6%         │ ✅ Expected    │
│ │ Avg Slippage        │ 0 bps    │ 12 bps  │ N/A         │ ✅ Acceptable  │
│ │ Partial Fills       │ 0%       │ 3%      │ N/A         │ ✅ Low         │
│ │ Order Latency       │ 0ms      │ 45ms    │ N/A         │ ✅ Fast        │
│ │ Trade Duration      │ 6.2h     │ 7.1h    │ +15%        │ ⚠️ Longer      │
│ │                                                          │                  │
│ └──────────────────────────────────────────────────────────┘                  │
│                                                                               │
│ ┌────────────────── SIGNAL QUALITY ───────────────────────┐                  │
│ │                                                          │                  │
│ │ Metric              │ Backtest │ Paper   │ Deviation   │ Status          │
│ ├─────────────────────┼──────────┼─────────┼─────────────┼─────────────────┤
│ │ Signals/Day         │ 148      │ 142     │ -4%         │ ✅ Stable      │
│ │ Overlap Rate        │ 42%      │ 48%     │ +14%        │ ⚠️ High        │
│ │ Avg Confidence      │ 0.87     │ 0.84    │ -3%         │ ✅ Stable      │
│ │ Boost Activation    │ 72%      │ 68%     │ -6%         │ ✅ Stable      │
│ │ Veto Rate           │ 18%      │ 22%     │ +22%        │ ⚠️ Higher      │
│ │                                                          │                  │
│ └──────────────────────────────────────────────────────────┘                  │
│                                                                               │
│ ┌──────────────── STATISTICAL SIGNIFICANCE ───────────────┐                  │
│ │                                                          │                  │
│ │ Sharpe Ratio T-Test:                                     │                  │
│ │   t-statistic: -1.45                                     │                  │
│ │   p-value: 0.15                                          │                  │
│ │   Result: NOT SIGNIFICANT ✅                            │                  │
│ │   Interpretation: Performance difference likely due to   │                  │
│ │                   random variance, not systematic issue  │                  │
│ │                                                          │                  │
│ │ Win Rate Chi-Square Test:                                │                  │
│ │   χ²: 2.8                                                │                  │
│ │   p-value: 0.09                                          │                  │
│ │   Result: NOT SIGNIFICANT ✅                            │                  │
│ │                                                          │                  │
│ └──────────────────────────────────────────────────────────┘                  │
│                                                                               │
│ ┌────────────────── GO-LIVE READINESS ────────────────────┐                  │
│ │                                                          │                  │
│ │ Mandatory Criteria:     7/7 PASSED ✅                   │                  │
│ │ Nice-to-Have Criteria:  1/3 PASSED ⚠️                   │                  │
│ │                                                          │                  │
│ │ Overall Readiness: APPROVED FOR LIVE TRADING            │                  │
│ │ Confidence: HIGH                                         │                  │
│ │                                                          │                  │
│ │ Recommendation:                                          │                  │
│ │ ✅ Proceed to live trading with $100K initial capital   │                  │
│ │ ⚠️ Monitor archetype C overlap (reduce if >50%)         │                  │
│ │ ⚠️ Track temporal feature drift (recalibrate if PSI>0.3)│                  │
│ │                                                          │                  │
│ └──────────────────────────────────────────────────────────┘                  │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 8.5 Mobile View (Responsive Design)

```
┌─────────────────────────┐
│ 🎯 BULL MACHINE         │
│ Status: 🟢 ACTIVE       │
│ Last Update: 5s ago     │
├─────────────────────────┤
│                         │
│ PnL Today: +$1,240      │
│ (+1.2%) ✅              │
│                         │
│ Drawdown: -8.3% ✅      │
│ Sharpe: 1.62 ⚠️         │
│                         │
│ ───── Quick Stats ───── │
│                         │
│ Fill Rate: 94% ✅       │
│ Signals: 142/day ✅     │
│ Overlap: 42% ✅         │
│                         │
│ ─── Archetype Health ── │
│                         │
│ A: ✅ 88  C: ⚠️ 72      │
│ S1: ✅ 91 S8: ✅ 85     │
│                         │
│ [View All 16 →]         │
│                         │
│ ──── Alerts (3) ────    │
│                         │
│ ⚠️ C overlap at 48%    │
│ ⚠️ Feature drift: SMC  │
│ ℹ️ Regime → risk_on    │
│                         │
│ [View All →]            │
│                         │
│ ──── Quick Actions ──── │
│                         │
│ [📊 Full Dashboard]     │
│ [🚨 Kill-Switch]        │
│ [📥 Export]             │
│                         │
└─────────────────────────┘
```

---

## 9. Technology Stack

### 9.1 Recommended Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     PRESENTATION LAYER                       │
├─────────────────────────────────────────────────────────────┤
│ Streamlit Dashboard (Primary)                               │
│ - Real-time updates via WebSocket                           │
│ - Interactive Plotly charts                                 │
│ - Mobile-responsive layout                                  │
│                                                              │
│ Grafana (Production Monitoring)                             │
│ - Prometheus metrics integration                            │
│ - Custom alerting rules                                     │
│ - Historical trend analysis                                 │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                      APPLICATION LAYER                       │
├─────────────────────────────────────────────────────────────┤
│ FastAPI Backend                                             │
│ - RESTful API endpoints                                     │
│ - WebSocket server for real-time updates                   │
│ - Background tasks (Celery)                                 │
│                                                              │
│ Metrics Calculator Service                                  │
│ - Real-time metric computation                              │
│ - Statistical tests (PSI, KS, CUSUM)                        │
│ - Drift detection algorithms                                │
│                                                              │
│ Alert Manager                                               │
│ - Condition evaluation                                      │
│ - Notification routing                                      │
│ - Kill-switch automation                                    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                       DATA LAYER                             │
├─────────────────────────────────────────────────────────────┤
│ TimescaleDB (Time-Series Storage)                           │
│ - Trade history                                             │
│ - Performance metrics (minute-level)                        │
│ - Feature distributions                                     │
│ - Regime transitions                                        │
│                                                              │
│ Redis (Real-Time Cache)                                     │
│ - Current metrics (30s-5min TTL)                            │
│ - Active positions                                          │
│ - Recent signals queue                                      │
│                                                              │
│ PostgreSQL (Persistent Storage)                             │
│ - Archetype configurations                                  │
│ - Alert history                                             │
│ - Backtest baselines                                        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   INFRASTRUCTURE LAYER                       │
├─────────────────────────────────────────────────────────────┤
│ Prometheus (Metrics Collection)                             │
│ - System metrics (CPU, memory)                              │
│ - Application metrics (latency, errors)                     │
│ - Custom business metrics                                   │
│                                                              │
│ AlertManager (Notification Routing)                         │
│ - Email (SendGrid)                                          │
│ - SMS (Twilio)                                              │
│ - Slack/Telegram webhooks                                   │
└─────────────────────────────────────────────────────────────┘
```

### 9.2 Technology Choices

#### Dashboard Framework: **Streamlit** (Primary)

**Pros**:
- Python-native (easy integration with existing codebase)
- Rapid development (< 1 week for full dashboard)
- Built-in caching and session state
- Mobile-responsive out of the box
- Easy deployment (Streamlit Cloud, Docker)

**Cons**:
- Limited customization vs React/Vue
- Performance issues with >10K data points (mitigated by aggregation)

**Alternative**: Plotly Dash (more flexible, steeper learning curve)

#### Backend API: **FastAPI**

**Pros**:
- Async support (handle many concurrent connections)
- Automatic OpenAPI docs
- WebSocket support for real-time updates
- Type hints → automatic validation

**Example Endpoint**:
```python
from fastapi import FastAPI, WebSocket
from fastapi.responses import StreamingResponse
import asyncio

app = FastAPI()

@app.get("/api/metrics/real-time")
async def get_real_time_metrics():
    """Returns current system metrics (cached 30s)"""
    return {
        "daily_pnl": get_cached_metric("daily_pnl"),
        "sharpe_7d": get_cached_metric("sharpe_7d"),
        "fill_rate_24h": get_cached_metric("fill_rate_24h"),
        # ... all real-time metrics
    }

@app.websocket("/ws/metrics")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for live metric streaming"""
    await websocket.accept()
    while True:
        metrics = calculate_real_time_metrics()
        await websocket.send_json(metrics)
        await asyncio.sleep(5)  # Update every 5s
```

#### Time-Series Database: **TimescaleDB**

**Pros**:
- PostgreSQL extension (familiar SQL interface)
- Automatic partitioning by time
- Continuous aggregates (pre-computed rollups)
- Compression (10x space savings)

**Schema Example**:
```sql
-- Hypertable for trade history
CREATE TABLE trades (
    time TIMESTAMPTZ NOT NULL,
    archetype_id TEXT,
    direction TEXT,
    entry_price DOUBLE PRECISION,
    exit_price DOUBLE PRECISION,
    pnl DOUBLE PRECISION,
    confidence DOUBLE PRECISION,
    boost_multiplier DOUBLE PRECISION,
    regime TEXT,
    metadata JSONB
);

SELECT create_hypertable('trades', 'time');

-- Continuous aggregate for hourly metrics
CREATE MATERIALIZED VIEW hourly_metrics
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', time) AS hour,
    archetype_id,
    COUNT(*) AS signal_count,
    AVG(confidence) AS avg_confidence,
    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END)::FLOAT / COUNT(*) AS win_rate,
    SUM(pnl) AS total_pnl
FROM trades
GROUP BY hour, archetype_id;
```

#### Real-Time Cache: **Redis**

**Use Cases**:
- Cache expensive calculations (Sharpe ratio, PSI scores)
- Pub/Sub for alert notifications
- Rate limiting (prevent dashboard spam)

**Example**:
```python
import redis
from functools import wraps

redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)

def cache_metric(ttl_seconds=30):
    """Decorator to cache metrics with TTL"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = f"metric:{func.__name__}:{args}:{kwargs}"
            cached_value = redis_client.get(cache_key)

            if cached_value:
                return json.loads(cached_value)

            result = func(*args, **kwargs)
            redis_client.setex(cache_key, ttl_seconds, json.dumps(result))
            return result
        return wrapper
    return decorator

@cache_metric(ttl_seconds=60)
def calculate_sharpe_7d():
    """Expensive calculation - cached for 1 minute"""
    returns = fetch_returns_from_db(days=7)
    return np.sqrt(168) * returns.mean() / returns.std()
```

#### Alerting: **Prometheus + AlertManager**

**Prometheus Metrics**:
```python
from prometheus_client import Counter, Gauge, Histogram

# Define metrics
daily_pnl_gauge = Gauge('daily_pnl_pct', 'Daily PnL percentage')
fill_rate_gauge = Gauge('fill_rate_24h', 'Fill rate over 24 hours')
signal_counter = Counter('signals_generated', 'Total signals', ['archetype_id'])
trade_duration_histogram = Histogram('trade_duration_hours', 'Trade duration in hours')

# Update metrics
daily_pnl_gauge.set(calculate_daily_pnl())
fill_rate_gauge.set(calculate_fill_rate())
signal_counter.labels(archetype_id='A').inc()
trade_duration_histogram.observe(6.8)
```

**AlertManager Rules** (`alerts.yml`):
```yaml
groups:
  - name: bull_machine_alerts
    interval: 30s
    rules:
      - alert: DailyLossLimit
        expr: daily_pnl_pct < -5
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Daily loss limit exceeded"
          description: "PnL: {{ $value }}% (threshold: -5%)"

      - alert: FillRateCollapse
        expr: fill_rate_24h < 0.80
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Fill rate below 80%"
          description: "Current: {{ $value }} (threshold: 0.80)"
```

### 9.3 Deployment Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      PRODUCTION SETUP                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  [Streamlit Dashboard] ←──────┐                             │
│  Port 8501                     │                             │
│  (Public, password-protected)  │                             │
│                                │                             │
│  [Grafana]                     │                             │
│  Port 3000                     ├── Nginx Reverse Proxy      │
│  (Internal team access)        │    (SSL, rate limiting)    │
│                                │                             │
│  [FastAPI Backend] ────────────┘                             │
│  Port 8000                                                   │
│  (Internal only)                                             │
│                                ↓                             │
│  [TimescaleDB]          [Redis]         [Prometheus]        │
│  Port 5432              Port 6379       Port 9090           │
│  (Internal only)        (Internal)      (Internal)          │
│                                                              │
│  All services in Docker Compose or Kubernetes               │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Docker Compose** (`docker-compose.yml`):
```yaml
version: '3.8'

services:
  dashboard:
    build: ./dashboard
    ports:
      - "8501:8501"
    environment:
      - BACKEND_URL=http://backend:8000
      - REDIS_URL=redis://redis:6379
    depends_on:
      - backend
      - redis

  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://timescale:5432/bull_machine
      - REDIS_URL=redis://redis:6379
    depends_on:
      - timescale
      - redis

  timescale:
    image: timescale/timescaledb:latest-pg14
    ports:
      - "5432:5432"
    environment:
      - POSTGRES_DB=bull_machine
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - timescale_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
    volumes:
      - grafana_data:/var/lib/grafana

volumes:
  timescale_data:
  redis_data:
  prometheus_data:
  grafana_data:
```

---

## 10. Implementation Roadmap

### Phase 1: Foundation (Week 1-2)

**Goals**:
- Database schema setup
- Basic API endpoints
- Core metric calculations

**Tasks**:
```
□ Set up TimescaleDB with hypertables
  - trades table
  - signals table
  - metrics_cache table
  - regime_transitions table

□ Implement FastAPI backend
  - /api/metrics/real-time endpoint
  - /api/metrics/archetype/{id} endpoint
  - /api/metrics/drift endpoint
  - WebSocket /ws/metrics

□ Core metric calculators
  - Sharpe ratio (1d, 7d, 30d rolling)
  - Drawdown tracking
  - Fill rate monitoring
  - Domain boost activation rate

□ Redis caching layer
  - Cache real-time metrics (30s TTL)
  - Pub/Sub for alerts

□ Basic Streamlit dashboard
  - Main overview page
  - Critical metrics display
  - Simple charts (PnL, Sharpe)
```

**Deliverable**: Functional dashboard showing real-time PnL, Sharpe, and fill rate.

---

### Phase 2: Archetype & Drift Monitoring (Week 3-4)

**Goals**:
- Archetype-level diagnostics
- Drift detection implementation
- Alert system

**Tasks**:
```
□ Archetype health scorecard
  - Health score calculation
  - Domain boost tracking
  - Regime alignment monitoring

□ Drift detection algorithms
  - PSI calculation for top 20 features
  - KS test implementation
  - CUSUM for win rate

□ Alert system
  - Condition evaluation engine
  - Notification routing (email, Slack)
  - Alert history storage

□ Streamlit drill-down views
  - Individual archetype dashboards
  - Drift detection page
  - Alert history page
```

**Deliverable**: Full archetype monitoring + drift alerts.

---

### Phase 3: Ensemble & Kill-Switch (Week 5-6)

**Goals**:
- Ensemble health monitoring
- Kill-switch implementation
- Comparison framework

**Tasks**:
```
□ Ensemble metrics
  - Signal overlap analysis
  - Archetype correlation matrix
  - Regime coverage tracking

□ Kill-switch system
  - Hard stop conditions
  - Soft stop conditions
  - Manual halt controls

□ Comparison framework
  - Backtest vs paper metrics
  - Statistical significance tests
  - Go-live readiness scoring

□ Prometheus + Grafana setup
  - Custom dashboards
  - Alert rules
  - Production monitoring
```

**Deliverable**: Complete monitoring system with automated kill-switches.

---

### Phase 4: Polish & Production (Week 7-8)

**Goals**:
- Mobile responsiveness
- Performance optimization
- Documentation

**Tasks**:
```
□ Mobile-friendly UI
  - Responsive layouts
  - Touch-friendly controls
  - Simplified mobile view

□ Performance optimization
  - Query optimization
  - Caching strategy refinement
  - Database indexing

□ Documentation
  - Operator manual
  - API documentation
  - Troubleshooting guide

□ Testing
  - Load testing (1000 signals/day)
  - Failover testing
  - Alert notification testing
```

**Deliverable**: Production-ready dashboard with 99.9% uptime target.

---

## 11. Database Schema

### 11.1 Core Tables

```sql
-- Trade history (hypertable)
CREATE TABLE trades (
    time TIMESTAMPTZ NOT NULL,
    trade_id UUID PRIMARY KEY,
    archetype_id TEXT NOT NULL,
    direction TEXT NOT NULL,  -- 'LONG' or 'SHORT'
    entry_price DOUBLE PRECISION NOT NULL,
    entry_time TIMESTAMPTZ NOT NULL,
    exit_price DOUBLE PRECISION,
    exit_time TIMESTAMPTZ,
    pnl DOUBLE PRECISION,
    pnl_pct DOUBLE PRECISION,
    confidence DOUBLE PRECISION,
    boost_multiplier DOUBLE PRECISION,
    regime TEXT,
    domain_boosts JSONB,  -- {'wyckoff': true, 'smc': false, ...}
    metadata JSONB
);

SELECT create_hypertable('trades', 'time');
CREATE INDEX idx_trades_archetype ON trades(archetype_id, time DESC);
CREATE INDEX idx_trades_regime ON trades(regime, time DESC);

-- Signals generated (includes non-executed)
CREATE TABLE signals (
    time TIMESTAMPTZ NOT NULL,
    signal_id UUID PRIMARY KEY,
    archetype_id TEXT NOT NULL,
    direction TEXT NOT NULL,
    price DOUBLE PRECISION NOT NULL,
    confidence DOUBLE PRECISION NOT NULL,
    boost_multiplier DOUBLE PRECISION,
    regime TEXT,
    domain_boosts JSONB,
    executed BOOLEAN DEFAULT FALSE,
    execution_price DOUBLE PRECISION,
    execution_time TIMESTAMPTZ,
    slippage_bps DOUBLE PRECISION,
    metadata JSONB
);

SELECT create_hypertable('signals', 'time');
CREATE INDEX idx_signals_archetype ON signals(archetype_id, time DESC);
CREATE INDEX idx_signals_executed ON signals(executed, time DESC);

-- Regime transitions
CREATE TABLE regime_transitions (
    time TIMESTAMPTZ NOT NULL,
    from_regime TEXT,
    to_regime TEXT,
    confidence DOUBLE PRECISION,
    detection_lag_minutes INTEGER,
    metadata JSONB
);

SELECT create_hypertable('regime_transitions', 'time');

-- Feature distributions (daily snapshots)
CREATE TABLE feature_distributions (
    date DATE NOT NULL,
    feature_name TEXT NOT NULL,
    distribution JSONB,  -- Histogram bins
    mean DOUBLE PRECISION,
    std DOUBLE PRECISION,
    min DOUBLE PRECISION,
    max DOUBLE PRECISION,
    psi DOUBLE PRECISION,  -- vs backtest
    ks_statistic DOUBLE PRECISION,
    PRIMARY KEY (date, feature_name)
);

-- Alert history
CREATE TABLE alerts (
    time TIMESTAMPTZ NOT NULL,
    alert_id UUID PRIMARY KEY,
    severity TEXT NOT NULL,  -- 'INFO', 'WARNING', 'CRITICAL'
    condition_name TEXT NOT NULL,
    message TEXT,
    current_value DOUBLE PRECISION,
    threshold DOUBLE PRECISION,
    acknowledged BOOLEAN DEFAULT FALSE,
    acknowledged_at TIMESTAMPTZ,
    acknowledged_by TEXT,
    metadata JSONB
);

SELECT create_hypertable('alerts', 'time');
CREATE INDEX idx_alerts_severity ON alerts(severity, time DESC);
CREATE INDEX idx_alerts_acknowledged ON alerts(acknowledged, time DESC);

-- Archetype health scores (hourly snapshots)
CREATE TABLE archetype_health (
    time TIMESTAMPTZ NOT NULL,
    archetype_id TEXT NOT NULL,
    health_score DOUBLE PRECISION,
    performance_score DOUBLE PRECISION,
    metadata_score DOUBLE PRECISION,
    regime_score DOUBLE PRECISION,
    execution_score DOUBLE PRECISION,
    status TEXT,  -- 'HEALTHY', 'DEGRADED', 'CRITICAL'
    PRIMARY KEY (time, archetype_id)
);

SELECT create_hypertable('archetype_health', 'time');
```

### 11.2 Continuous Aggregates (Pre-computed Rollups)

```sql
-- Hourly archetype metrics
CREATE MATERIALIZED VIEW hourly_archetype_metrics
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', time) AS hour,
    archetype_id,
    COUNT(*) AS signal_count,
    SUM(CASE WHEN executed THEN 1 ELSE 0 END) AS executed_count,
    AVG(confidence) AS avg_confidence,
    AVG(boost_multiplier) AS avg_boost,
    AVG(slippage_bps) AS avg_slippage
FROM signals
GROUP BY hour, archetype_id;

-- Daily performance metrics
CREATE MATERIALIZED VIEW daily_performance
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', time) AS day,
    COUNT(*) AS total_trades,
    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END)::FLOAT / NULLIF(COUNT(*), 0) AS win_rate,
    SUM(pnl) AS total_pnl,
    AVG(pnl_pct) AS avg_return,
    STDDEV(pnl_pct) AS return_volatility
FROM trades
GROUP BY day;

-- Refresh policies (auto-update every 5 minutes)
SELECT add_continuous_aggregate_policy('hourly_archetype_metrics',
    start_offset => INTERVAL '1 day',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '5 minutes');

SELECT add_continuous_aggregate_policy('daily_performance',
    start_offset => INTERVAL '7 days',
    end_offset => INTERVAL '1 day',
    schedule_interval => INTERVAL '1 hour');
```

### 11.3 Backtest Baselines (Static Reference Data)

```sql
-- Store backtest results for comparison
CREATE TABLE backtest_baselines (
    archetype_id TEXT PRIMARY KEY,
    period_start DATE,
    period_end DATE,
    sharpe_ratio DOUBLE PRECISION,
    sortino_ratio DOUBLE PRECISION,
    calmar_ratio DOUBLE PRECISION,
    win_rate DOUBLE PRECISION,
    max_drawdown DOUBLE PRECISION,
    avg_trade_duration_hours DOUBLE PRECISION,
    total_signals INTEGER,
    boost_activation_rate DOUBLE PRECISION,
    veto_rate DOUBLE PRECISION,
    avg_confidence DOUBLE PRECISION,
    regime_distribution JSONB,  -- {'risk_on': 0.6, 'risk_off': 0.25, ...}
    metadata JSONB
);

-- Populate with backtest results
INSERT INTO backtest_baselines (archetype_id, sharpe_ratio, win_rate, ...)
VALUES
    ('A', 1.92, 0.58, ...),
    ('C', 1.78, 0.61, ...),
    ('S1', 2.15, 0.55, ...);
```

---

## 12. Alert Configuration

### 12.1 Alert Severity Levels

| Severity | Description | Response Time | Notification Channels |
|----------|-------------|---------------|----------------------|
| **INFO** | Informational events (regime transitions, archetype signals) | No action required | Dashboard only |
| **WARNING** | Potential issues requiring monitoring (drift, minor degradation) | Review within 4 hours | Email, Slack |
| **CRITICAL** | Serious issues requiring immediate attention (kill-switch triggers) | Immediate review | SMS, Email, Slack (urgent) |

### 12.2 Alert Definitions

```python
ALERT_CONDITIONS = {
    # CRITICAL ALERTS
    "daily_loss_limit": {
        "severity": "CRITICAL",
        "condition": "daily_pnl_pct < -5.0",
        "check_interval_seconds": 60,
        "notification": ["SMS", "EMAIL", "SLACK_URGENT"],
        "action": "HALT_ALL_TRADING",
        "message_template": "🚨 CRITICAL: Daily loss limit exceeded at {daily_pnl_pct:.2f}% (threshold: -5.0%)"
    },

    "drawdown_limit": {
        "severity": "CRITICAL",
        "condition": "current_drawdown > 25.0",
        "check_interval_seconds": 60,
        "notification": ["SMS", "EMAIL", "SLACK_URGENT"],
        "action": "HALT_ALL_TRADING",
        "message_template": "🚨 CRITICAL: Drawdown exceeded at {current_drawdown:.2f}% (threshold: 25%)"
    },

    "fill_rate_collapse": {
        "severity": "CRITICAL",
        "condition": "fill_rate_4h < 0.80 AND signal_count_4h >= 10",
        "check_interval_seconds": 300,
        "notification": ["EMAIL", "SLACK_URGENT"],
        "action": "HALT_NEW_POSITIONS",
        "message_template": "🚨 CRITICAL: Fill rate collapsed to {fill_rate_4h:.1f}% (threshold: 80%)"
    },

    "metadata_failure": {
        "severity": "CRITICAL",
        "condition": "boost_activation_rate < 0.1 AND veto_activation_rate < 0.05",
        "duration_minutes": 30,
        "notification": ["SMS", "EMAIL", "SLACK_URGENT"],
        "action": "HALT_ALL_TRADING",
        "message_template": "🚨 CRITICAL: Metadata pipeline failure - boosts: {boost_activation_rate:.1f}%, vetoes: {veto_activation_rate:.1f}%"
    },

    # WARNING ALERTS
    "win_rate_degradation": {
        "severity": "WARNING",
        "condition": "win_rate_50_trades < 0.40",
        "check_interval_seconds": 3600,
        "notification": ["EMAIL", "SLACK"],
        "action": "ALERT_OPERATOR",
        "message_template": "⚠️ WARNING: Win rate dropped to {win_rate_50_trades:.1f}% over 50 trades (expected: 55%)"
    },

    "archetype_overlap_high": {
        "severity": "WARNING",
        "condition": "signal_overlap_rate > 0.50",
        "check_interval_seconds": 3600,
        "notification": ["EMAIL", "SLACK"],
        "action": "ALERT_OPERATOR",
        "message_template": "⚠️ WARNING: Signal overlap at {signal_overlap_rate:.1f}% (target: <45%)"
    },

    "feature_drift_detected": {
        "severity": "WARNING",
        "condition": "features_with_psi_above_0_25 >= 3",
        "check_interval_seconds": 3600,
        "notification": ["EMAIL", "SLACK"],
        "action": "ALERT_OPERATOR",
        "message_template": "⚠️ WARNING: {features_with_psi_above_0_25} features showing drift (PSI > 0.25)"
    },

    "sharpe_decline": {
        "severity": "WARNING",
        "condition": "sharpe_7d < 0.5",
        "duration_hours": 168,  # 7 days
        "notification": ["EMAIL"],
        "action": "ALERT_OPERATOR",
        "message_template": "⚠️ WARNING: Sharpe ratio at {sharpe_7d:.2f} for 7 consecutive days (threshold: 0.5)"
    },

    # INFO ALERTS
    "regime_transition": {
        "severity": "INFO",
        "condition": "regime_changed",
        "notification": ["DASHBOARD"],
        "message_template": "ℹ️ Regime transition: {from_regime} → {to_regime} (confidence: {confidence:.2f})"
    },

    "archetype_high_activity": {
        "severity": "INFO",
        "condition": "archetype_signals_1h > expected_signals_1h * 2",
        "notification": ["DASHBOARD"],
        "message_template": "ℹ️ Archetype {archetype_id} showing high activity: {archetype_signals_1h} signals in 1h (expected: ~{expected_signals_1h})"
    }
}
```

### 12.3 Notification Templates

```python
# Email template (HTML)
EMAIL_TEMPLATE_CRITICAL = """
<html>
<body>
    <h2 style="color: #d32f2f;">🚨 BULL MACHINE CRITICAL ALERT</h2>
    <p><strong>Condition:</strong> {condition_name}</p>
    <p><strong>Message:</strong> {message}</p>
    <p><strong>Current Value:</strong> {current_value}</p>
    <p><strong>Threshold:</strong> {threshold}</p>
    <p><strong>Time:</strong> {timestamp}</p>

    <h3>Immediate Actions:</h3>
    <ul>
        <li>{action_description}</li>
        <li>Review dashboard: <a href="{dashboard_url}">Click here</a></li>
        <li>Check kill-switch panel: <a href="{killswitch_url}">Click here</a></li>
    </ul>

    <p style="color: #757575; font-size: 12px;">
        This is an automated alert from Bull Machine Paper Trading System.
        To acknowledge, visit the dashboard or reply to this email.
    </p>
</body>
</html>
"""

# Slack template (webhook payload)
SLACK_TEMPLATE_CRITICAL = {
    "blocks": [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": "🚨 BULL MACHINE CRITICAL ALERT"
            }
        },
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*Condition:*\n{condition_name}"},
                {"type": "mrkdwn", "text": f"*Severity:*\nCRITICAL"},
                {"type": "mrkdwn", "text": f"*Current Value:*\n{current_value}"},
                {"type": "mrkdwn", "text": f"*Threshold:*\n{threshold}"}
            ]
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Message:* {message}"
            }
        },
        {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "View Dashboard"},
                    "url": dashboard_url,
                    "style": "danger"
                },
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Acknowledge"},
                    "url": f"{api_url}/alerts/{alert_id}/acknowledge"
                }
            ]
        }
    ]
}

# SMS template (plain text, <160 chars)
SMS_TEMPLATE_CRITICAL = "🚨 BULL MACHINE: {condition_name} - {message}. Check dashboard immediately: {short_url}"
```

### 12.4 Alert Deduplication & Throttling

```python
class AlertManager:
    def __init__(self, redis_client):
        self.redis = redis_client
        self.throttle_window_seconds = 300  # 5 minutes

    def should_send_alert(self, condition_name, severity):
        """Prevent alert spam with deduplication"""
        cache_key = f"alert_sent:{condition_name}"

        # Check if alert was recently sent
        if self.redis.exists(cache_key):
            return False

        # Set throttle window (longer for lower severity)
        ttl = {
            "INFO": 300,      # 5 minutes
            "WARNING": 1800,  # 30 minutes
            "CRITICAL": 600   # 10 minutes (still throttle to avoid spam)
        }[severity]

        self.redis.setex(cache_key, ttl, "1")
        return True

    def acknowledge_alert(self, alert_id, acknowledged_by):
        """Mark alert as acknowledged (stops repeat notifications)"""
        db.execute("""
            UPDATE alerts
            SET acknowledged = TRUE,
                acknowledged_at = NOW(),
                acknowledged_by = %s
            WHERE alert_id = %s
        """, (acknowledged_by, alert_id))
```

---

## 13. Risk Mitigation

### 13.1 Dashboard Reliability

**Risk**: Dashboard downtime prevents monitoring.

**Mitigation**:
1. **Redundant Monitoring**: Run Grafana in parallel (separate infrastructure)
2. **Health Checks**: Prometheus monitors dashboard uptime
3. **Fallback Alerts**: Email/SMS alerts work even if dashboard is down
4. **Data Persistence**: TimescaleDB ensures no data loss during downtime

```python
# Health check endpoint
@app.get("/health")
async def health_check():
    checks = {
        "database": check_database_connection(),
        "redis": check_redis_connection(),
        "metrics_calculator": check_metrics_service(),
        "alert_manager": check_alert_service()
    }

    all_healthy = all(checks.values())
    status_code = 200 if all_healthy else 503

    return JSONResponse(
        status_code=status_code,
        content={"status": "healthy" if all_healthy else "degraded", "checks": checks}
    )
```

### 13.2 False Positive Alerts

**Risk**: Alert fatigue from too many false alarms.

**Mitigation**:
1. **Grace Periods**: Alerts require sustained condition (e.g., "Sharpe <0.5 for 7 days")
2. **Statistical Significance**: Use p-values to confirm drift (not just noise)
3. **Dynamic Thresholds**: Adjust thresholds based on regime (e.g., higher DD tolerance in crisis)
4. **Alert Review**: Weekly review of alert accuracy, tune thresholds

```python
def evaluate_condition_with_grace_period(condition, duration_minutes=30):
    """Only trigger alert if condition persists for duration"""
    cache_key = f"condition_start:{condition.name}"

    if condition.is_true():
        # Check how long condition has been true
        start_time = redis.get(cache_key)
        if start_time is None:
            redis.set(cache_key, time.time())
            return False  # Just started, wait for grace period

        elapsed = time.time() - float(start_time)
        if elapsed >= duration_minutes * 60:
            return True  # Condition persisted, trigger alert
    else:
        # Condition resolved, reset timer
        redis.delete(cache_key)

    return False
```

### 13.3 Data Quality Issues

**Risk**: Bad data (missing features, stale prices) causes incorrect metrics.

**Mitigation**:
1. **Data Validation**: Check for NaN, inf, stale timestamps
2. **Sanity Checks**: Alert if metrics are physically impossible (e.g., fill rate >100%)
3. **Audit Logs**: Track all data sources and transformations
4. **Fallback Values**: Use last known good value if current data is suspect

```python
def validate_metric(metric_name, value):
    """Sanity checks for metric values"""
    if metric_name == "fill_rate":
        assert 0 <= value <= 1, f"Invalid fill rate: {value}"
    elif metric_name == "sharpe_ratio":
        assert -5 <= value <= 10, f"Unrealistic Sharpe: {value}"
    elif metric_name == "drawdown":
        assert value <= 0, f"Drawdown must be negative: {value}"

    # Check for NaN/inf
    if not np.isfinite(value):
        raise ValueError(f"Non-finite metric value: {metric_name}={value}")

    return value
```

### 13.4 Kill-Switch Failure

**Risk**: Kill-switch doesn't trigger when it should.

**Mitigation**:
1. **Redundant Checks**: Kill-switch evaluated in multiple places (dashboard, Prometheus, cron job)
2. **Manual Override**: Always allow human operator to halt trading
3. **Dead Man's Switch**: If dashboard stops updating, assume failure and halt
4. **Testing**: Monthly kill-switch drills (simulate conditions, verify halt)

```python
# Redundant kill-switch check (runs independently of dashboard)
# Schedule: Every 60 seconds via cron

def independent_kill_switch_check():
    """Backup kill-switch that runs outside dashboard"""
    metrics = fetch_critical_metrics_from_db()

    for condition_name, config in HARD_STOP_CONDITIONS.items():
        if evaluate_condition(condition_name, metrics):
            logger.critical(f"KILL-SWITCH TRIGGERED: {condition_name}")
            halt_all_trading()
            send_emergency_notification(condition_name, metrics)
            break

if __name__ == "__main__":
    independent_kill_switch_check()
```

### 13.5 Performance Degradation

**Risk**: Dashboard becomes slow/unresponsive under load.

**Mitigation**:
1. **Caching Strategy**: Cache expensive calculations (Sharpe, PSI) for 30s-5min
2. **Database Optimization**: Use continuous aggregates, proper indexing
3. **Lazy Loading**: Load drill-down data only when requested
4. **Load Testing**: Test with 10x expected load (10K signals/day)

```python
# Performance monitoring decorator
import time
from functools import wraps

def monitor_performance(threshold_seconds=1.0):
    """Alert if function execution exceeds threshold"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start

            if duration > threshold_seconds:
                logger.warning(f"SLOW QUERY: {func.__name__} took {duration:.2f}s")
                prometheus_slow_query_counter.labels(function=func.__name__).inc()

            return result
        return wrapper
    return decorator

@monitor_performance(threshold_seconds=0.5)
def calculate_archetype_health(archetype_id):
    # ... expensive calculation
    pass
```

---

## Summary & Next Steps

### Key Deliverables Checklist

- [x] **Metrics Specification**: Complete taxonomy of 50+ metrics across 3 tiers
- [x] **Dashboard Wireframes**: Main view, archetype detail, drift detection, comparison
- [x] **Technology Stack**: Streamlit + FastAPI + TimescaleDB + Redis + Prometheus
- [x] **Database Schema**: Hypertables, continuous aggregates, alert storage
- [x] **Alert Configuration**: 15+ alert conditions with severity levels and notification routing
- [x] **Kill-Switch System**: Hard/soft stop conditions with manual override
- [x] **Comparison Framework**: Backtest vs paper with statistical significance tests
- [x] **Implementation Roadmap**: 4-phase rollout over 8 weeks
- [x] **Risk Mitigation**: Strategies for reliability, false positives, data quality

### Success Criteria (60-Day Paper Trading)

**Week 1-2** (Foundation):
- ✅ Dashboard operational with real-time metrics
- ✅ All 16 archetypes reporting health scores
- ✅ Basic alerts (PnL, DD, fill rate) active

**Week 3-4** (Stability):
- ✅ No critical alerts for 7 consecutive days
- ✅ Sharpe ratio within 20% of backtest
- ✅ Fill rate >90%

**Week 5-6** (Validation):
- ✅ Feature drift PSI <0.25 for 90% of features
- ✅ Win rate >50%
- ✅ Metadata integrity maintained (boost rate >50%)

**Week 7-8** (Go-Live Readiness):
- ✅ All mandatory criteria passed (7/7)
- ✅ Statistical significance tests confirm performance consistency
- ✅ Kill-switch tested and verified
- ✅ Documentation complete

### Go-Live Decision Framework

**APPROVED FOR LIVE TRADING** if:
1. Sharpe ratio ≥ 1.48 (80% of backtest)
2. Max drawdown < 25%
3. Win rate ≥ 50%
4. Fill rate ≥ 90%
5. Boost activation rate ≥ 50%
6. Regime detection accuracy ≥ 70%
7. ≥12 archetypes with health score >75

**REQUIRES ADDITIONAL PAPER TRADING** if:
- Any mandatory criterion fails
- >5 features show critical drift (PSI >0.35)
- >3 archetypes degraded (health <60)
- Kill-switch triggered >2 times

**DO NOT GO LIVE** if:
- Daily loss limit hit
- Metadata pipeline failure
- Fill rate <80% for >24 hours

---

## Appendix: Quick Reference

### Critical Metrics Summary

| Metric | Update Freq | Alert Threshold | Action |
|--------|-------------|-----------------|--------|
| Daily PnL | 30s | < -5% | HALT |
| Drawdown | 30s | > 25% | HALT |
| Sharpe 7d | 5min | < 0.5 for 7d | REVIEW |
| Fill Rate | 5min | < 80% | HALT |
| Win Rate | 1h | < 40% | REVIEW |
| Boost Rate | 5min | < 50% | REVIEW |
| Feature PSI | Daily | > 0.25 | MONITOR |
| Regime Thrashing | 5min | >5 transitions/h | HALT |

### Dashboard URLs (Production)

- **Main Dashboard**: https://dashboard.bullmachine.ai
- **Grafana**: https://metrics.bullmachine.ai
- **API Docs**: https://api.bullmachine.ai/docs
- **Kill-Switch Panel**: https://dashboard.bullmachine.ai/killswitch

### Emergency Contacts

- **SMS Alerts**: +1-XXX-XXX-XXXX
- **Email**: alerts@bullmachine.ai
- **Slack**: #bull-machine-alerts (urgent), #bull-machine-ops (general)

---

**END OF SPECIFICATION**

This dashboard is designed to **measure what actually matters**: forward-looking reliability, not just historical PnL. It detects drift, degradation, and failure modes BEFORE live capital is at risk, ensuring a safe and confident transition from backtest to live trading.
