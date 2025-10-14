# Bull Machine ML Stack

Comprehensive ML enhancements optimized for precision without rewriting core wisdom.

## Modules Implemented

### 1. Kelly-Lite Dynamic Risk Sizing
**File**: `kelly_lite_sizer.py`

**Purpose**: ML-based position sizer that optimizes risk % (0-2%) per trade using gradient boosting.

**Inputs**:
- Fusion score
- Regime classification
- Realized volatility (RV 20d/60d)
- VIX proxy
- Recent drawdown
- Expected R-multiple
- Consecutive losses

**Model**: Gradient Boosting Regressor

**Guardrails**:
- Hard clamp: 0-2%
- Decay after ≥2 consecutive losers (0.7^(n-1))
- Hard cap in risk_off (1%) and crisis (0.5%)
- Drawdown scaling for >10% DD

**Target**: +0.2-0.4 PF uplift with ≤10% max DD

**Usage**:
```python
from engine.ml import KellyLiteSizer

sizer = KellyLiteSizer.load("models/kelly_lite_sizer.pkl")

risk_pct = sizer.predict_risk_pct(
    fusion_score=0.75,
    regime='neutral',
    rv_20d=40.0,
    rv_60d=45.0,
    vix_proxy=20.0,
    recent_dd=-0.03,
    expected_r=1.5,
    consecutive_losses=0
)
# Returns: 0.0095 (0.95%)
```

---

### 2. ML Fusion Scorer (XGBoost)
**File**: `fusion_scorer_ml.py`

**Purpose**: Learns optimal domain weight combinations dynamically using XGBoost to replace static fusion aggregation.

**Inputs**:
- Domain scores (Wyckoff, SMC, HOB, Momentum, Temporal)
- Domain interactions (Wy×SMC, Wy×HOB, Mom×HOB)
- Macro context (RV, VIX, regime)
- Market structure (ADX, ATR, volume)
- Temporal features (hour, day-of-week)

**Model**: XGBoost Classifier

**Performance** (BTC 1-year):
- Train AUC: 0.985
- Test AUC: 0.784
- Optimal Threshold: 0.425
- Precision: 0.427 | Recall: 0.721 | F1: 0.537

**Top Features**:
1. day_of_week (0.143)
2. atr_normalized (0.106)
3. rv_60d (0.099)
4. wyckoff_hob_product (0.096)
5. momentum_score (0.095)

**Target**: PF ≥1.8, trades ≥50, WR +5-10%

**Usage**:
```python
from engine.ml import FusionScorerML

scorer = FusionScorerML.load("models/fusion_scorer_xgb.pkl")

should_enter, fusion_score = scorer.should_enter(
    domain_scores={'wyckoff': 0.8, 'smc': 0.7, 'hob': 0.75, 'momentum': 0.65},
    macro_features={'rv_20d': 38.0, 'rv_60d': 42.0, 'vix_proxy': 18.0, 'regime': 'neutral'},
    market_features={'adx': 28.0, 'atr_normalized': 0.025, 'volume_ratio': 1.2},
    timestamp=pd.Timestamp.now()
)
# Returns: (True, 0.68)
```

---

### 3. Enhanced Macro Signals
**File**: `macro_signals_enhanced.py`

**Purpose**: Comprehensive macro signal engine with trap detection and greenlights.

**Signals Implemented**:

#### Traps & Vetoes:
1. **Funding Rate Trap**
   - Trigger: funding >0.01 AND oi >0.015
   - Action: Suppress entries
   - Source: ZeroIKA post:58

2. **DXY + VIX Double Trap**
   - Trigger: DXY >105 AND VIX >30
   - Action: Suppress + risk_mult 0.5x
   - Source: Wyckoff Insider post:42

3. **Yield Curve Inversion**
   - Trigger: 2Y > 10Y (spread <0%)
   - Action: threshold +0.05, risk_mult 0.8x
   - Source: Wyckoff Insider post:42

4. **VIX + 2Y Yield Regime Shift**
   - Trigger: VIX >30 AND 2Y >4.5%
   - Action: Suppress
   - Source: ZeroIKA post:58

#### Greenlights & Boosts:
1. **TOTAL2 Divergence (Altseason)**
   - Trigger: TOTAL2/TOTAL >0.405 AND BTC.D <55%
   - Action: threshold -0.05, momentum +0.05
   - Source: Wyckoff Insider post:35

2. **BTC.D + Oil Signal**
   - Trigger: BTC.D declining
   - Action: threshold -0.03
   - Source: Moneytaur post:41

3. **Low VIX + Low DXY (Risk-On)**
   - Trigger: VIX <15 AND DXY <98
   - Action: threshold -0.10, risk_mult 1.15x

4. **Neutral Funding (Healthy Market)**
   - Trigger: |funding| <0.005
   - Action: threshold -0.05

**Usage**:
```python
from engine.ml import MacroSignalsEnhanced

engine = MacroSignalsEnhanced()

macro_snapshot = {
    'VIX': 18.0,
    'DXY': 102.0,
    'funding': 0.006,
    'oi': 0.012,
    'YIELD_2Y': 4.2,
    'YIELD_10Y': 4.5,
    'BTC.D': 54.0,
    'TOTAL': 100.0,
    'TOTAL2': 42.0
}

adjustments = engine.analyze_macro_conditions(macro_snapshot)
# Returns: {
#     'enter_threshold_delta': -0.08,
#     'risk_multiplier': 1.0,
#     'suppress': False,
#     'notes': ['ALTSEASON BOOST: ...', 'NEUTRAL FUNDING: ...']
# }
```

---

## Integration Pattern

All ML modules return bounded deltas compatible with existing regime policy:

```python
{
    "enter_threshold_delta": [-0.10, +0.10],
    "risk_multiplier": [0.0, 1.5],
    "weight_nudges": {"wyckoff": ±0.05, "momentum": ±0.05},
    "suppress": bool,
    "notes": ["signal descriptions"]
}
```

Apply after regime policy, renormalize weights, and enforce caps from config.

---

## Training Data Requirements

### Kelly-Lite Sizer
- Requires: Trade-level results with PnL, regime, macro features
- Minimum: 500 trades across multiple regimes
- Source: hybrid_runner trade logs

### Fusion Scorer ML
- Requires: Hourly bars with domain scores + forward returns
- Minimum: 6-12 months of data (4,000-8,000 bars)
- Source: Feature store (v18) + macro features

---

## Models Trained

| Model | Path | Performance | Date |
|-------|------|-------------|------|
| Fusion Scorer XGB | `models/fusion_scorer_xgb.pkl` | AUC 0.784 | 2025-10-14 |

---

## Roadmap (Coming Soon)

### 4. Smart Exit Optimizer (LSTM/GBM)
- Predict optimal R-multiple exits per regime/volatility
- Target: Avg R +0.5, PF ≥2.0

### 5. Cooldown Optimizer (Contextual Bandit)
- Learn optimal cooldown bars per regime
- Target: WR +2-5pp, trades ≥80% baseline

### 6. Time-of-Day Session Alpha
- Boost/suppress entries by hour/session
- Target: PF +0.1 with neutral trade count

### 7. Stop Placement Tuner
- Choose best stop offset contextually (ATR vs wick-based)
- Target: Avg R +0.2-0.4, smaller tail losses

---

## References

**Trading Profiles**:
- Wyckoff Insider (Post 42, Oct 12 2025): DXY+VIX synergy, yield curve
- Moneytaur (Post 41, Oct 2025): BTC.D + Oil altseason
- ZeroIKA (Post 58, Oct 2025): Funding traps, VIX+yield regime shifts

**Academic**:
- Kelly Criterion (fractional sizing for growth)
- XGBoost paper (Chen & Guestrin, 2016)

---

## Performance Targets

| Metric | Baseline | With ML Stack | Target |
|--------|----------|---------------|--------|
| Profit Factor | 1.6-2.0 | 1.8-2.4 | +0.2-0.4 |
| Sharpe Ratio | 1.5-2.5 | 1.7-2.7 | +0.2 |
| Win Rate | 60-70% | 65-75% | +5-10pp |
| Max Drawdown | 8-10% | 6-8% | -2-4pp |
| Avg R-Multiple | 0.5-1.0 | 0.7-1.4 | +0.2-0.4 |

---

## License

MIT License - Bull Machine Project
