# Adaptive Fusion Implementation - Status & Next Steps

**Status:** Core module implemented, ready for config + wiring

---

## ✅ What's Done

### 1. Infrastructure (Already Existed)
- `engine/context/regime_classifier.py` - GMM classifier returning probability distributions
- `models/regime_classifier_gmm.pkl` - Trained model (risk_on/neutral/risk_off/crisis)
- Regime labels map naturally: `risk_on=bull, neutral=chop, risk_off=bear, crisis=crisis`

### 2. Adaptive Fusion Module (NEW - Just Created)
**File:** `engine/fusion/adaptive.py`

**Functions:**
- `ema_smooth()` - Smooth regime probabilities (alpha=0.2 default)
- `adapt_weights()` - Blend fusion weights across regimes (with min_weight floor)
- `adapt_gates()` - Blend entry thresholds (min_liquidity, final_fusion_floor)
- `adapt_exit_params()` - Blend exit policy (trail_atr, max_bars)
- `regime_size_mult()` - Blend sizing multiplier with limits
- `adapt_ml_threshold()` - Blend ML filter threshold

**Class:**
- `AdaptiveFusion` - Stateful coordinator maintaining EMA state

---

## 📋 Next Steps (In Order)

### Step 1: Create Adaptive Config (Seed Values)
Create `configs/btc_v8_adaptive.json` with regime profiles:

```json
{
  "profile": "v8_adaptive_fusion",
  "description": "v8 bull winner + adaptive regime morphing",

  "fusion_regime_profiles": {
    "risk_on":  {"wyckoff": 0.46, "liquidity": 0.26, "momentum": 0.22, "temporal": 0.06},
    "neutral":  {"wyckoff": 0.42, "liquidity": 0.28, "momentum": 0.20, "temporal": 0.10},
    "risk_off": {"wyckoff": 0.38, "liquidity": 0.32, "momentum": 0.16, "temporal": 0.14},
    "crisis":   {"wyckoff": 0.34, "liquidity": 0.38, "momentum": 0.12, "temporal": 0.16}
  },

  "gates_regime_profiles": {
    "risk_on":  {"min_liquidity": 0.16, "final_fusion_floor": 0.30},
    "neutral":  {"min_liquidity": 0.18, "final_fusion_floor": 0.34},
    "risk_off": {"min_liquidity": 0.22, "final_fusion_floor": 0.38},
    "crisis":   {"min_liquidity": 0.26, "final_fusion_floor": 0.45}
  },

  "exit_regime_profiles": {
    "risk_on":  {"trail_atr": 1.25, "max_bars": 92},
    "neutral":  {"trail_atr": 1.10, "max_bars": 72},
    "risk_off": {"trail_atr": 0.95, "max_bars": 54},
    "crisis":   {"trail_atr": 0.85, "max_bars": 36}
  },

  "sizing_regime_curve": {
    "risk_on": 1.20,
    "neutral": 0.90,
    "risk_off": 0.75,
    "crisis": 0.50
  },

  "sizing_limits": {"min": 0.6, "max": 1.35},

  "ml_thresholds_by_regime": {
    "risk_on": 0.28,
    "neutral": 0.35,
    "risk_off": 0.42,
    "crisis": 0.55
  },

  "fusion_adapt": {
    "enable": true,
    "ema_alpha": 0.2,
    "min_weight": 0.05
  },

  "regime_classifier": {
    "model_path": "models/regime_classifier_gmm.pkl",
    "feature_order": ["VIX", "DXY", "MOVE", "YIELD_2Y", "YIELD_10Y",
                     "USDT.D", "BTC.D", "TOTAL", "TOTAL2",
                     "funding", "oi", "rv_20d", "rv_60d"]
  },

  ... (rest of v8 config - archetypes, ML filter, runtime liquidity, etc.)
}
```

**Seed values above are educated guesses based on:**
- v8 winner for `risk_on` (bull market)
- Gradual shifts toward liquidity+temporal in defensive regimes
- Tighter gates in bear/crisis (avoid whipsaws)
- Tighter stops in bear/crisis (faster exits)
- Size scaling 1.2x bull → 0.5x crisis

### Step 2: Wire into Backtest Engine
**File:** `bin/backtest_knowledge_v2.py` (main loop before entry logic)

```python
from engine.context.regime_classifier import RegimeClassifier
from engine.fusion/adaptive import AdaptiveFusion

# Load regime classifier (once at startup)
regime_classifier = RegimeClassifier.load(
    cfg["regime_classifier"]["model_path"],
    cfg["regime_classifier"]["feature_order"]
)

# Initialize adaptive fusion coordinator
adaptive_fusion = AdaptiveFusion(cfg)

# Main backtest loop
for idx, row in df.iterrows():
    ... (existing feature extraction) ...

    # Classify regime from macro features
    macro_row = {feat: row.get(feat, np.nan) for feat in cfg["regime_classifier"]["feature_order"]}
    regime_info = regime_classifier.classify(macro_row)

    # Get adapted parameters for current bar
    adapted = adaptive_fusion.update(regime_info)

    # Override fusion weights if adaptive enabled
    if adapted["fusion_weights"]:
        fusion_weights = adapted["fusion_weights"]
    else:
        fusion_weights = cfg["fusion"]["weights"]  # fallback to static

    # Compute fusion score with adapted weights
    fusion_score = (
        fusion_weights["wyckoff"] * row["wyckoff_score"] +
        fusion_weights["liquidity"] * row["liquidity_score"] +
        fusion_weights["momentum"] * row["momentum_score"] +
        fusion_weights.get("temporal", 0) * row.get("temporal_score", 0)
    )

    # Apply adapted entry gates
    if adapted["gates"]:
        min_liquidity_threshold = adapted["gates"]["min_liquidity"]
        final_fusion_floor = adapted["gates"]["final_fusion_floor"]
    else:
        min_liquidity_threshold = cfg["gates"]["min_liquidity"]
        final_fusion_floor = cfg["gates"]["final_fusion_floor"]

    # Entry logic with adapted thresholds
    if row["liquidity_score"] < min_liquidity_threshold:
        continue
    if fusion_score < final_fusion_floor:
        continue

    # Apply adapted ML threshold (if ML filter enabled)
    if cfg["ml_filter"]["enabled"] and adapted["ml_threshold"]:
        ml_threshold = adapted["ml_threshold"]
    else:
        ml_threshold = cfg["ml_filter"]["threshold"]

    if ml_predict_proba(row) < ml_threshold:
        continue

    # Compute position size with adapted multiplier
    base_size = compute_dynamic_size(row, cfg)  # existing logic
    if adapted["size_mult"]:
        position_size = base_size * adapted["size_mult"]
    else:
        position_size = base_size

    ... (entry execution) ...

    # For exits: use adapted trail_atr and max_bars
    if in_position:
        if adapted["exit_params"]:
            trail_atr_mult = adapted["exit_params"]["trail_atr"]
            max_bars_in_trade = adapted["exit_params"]["max_bars"]
        else:
            trail_atr_mult = cfg["exits"]["trail_atr_mult"]
            max_bars_in_trade = cfg["exits"]["max_bars"]

        ... (exit logic with adapted params) ...
```

### Step 3: Test on Fixed Regime (Validation)
Before full 2022-2024 span, test adaptive behavior on single-regime periods:

**Bull Period (2024 only):**
```bash
python3 bin/backtest_knowledge_v2.py \
  --asset BTC --start 2024-01-01 --end 2024-12-31 \
  --config configs/btc_v8_adaptive.json

# Expect: weights converge to risk_on profile, performance ≈ v8 winner (PF 20.96)
```

**Bear Period (2022 only):**
```bash
python3 bin/backtest_knowledge_v2.py \
  --asset BTC --start 2022-01-01 --end 2022-12-31 \
  --config configs/btc_v8_adaptive.json

# Expect: weights converge to risk_off profile, fewer trades, tighter stops
# Target: PF > 1.5, DD < 12%, trades > 20
```

**Chop Period (2023 only):**
```bash
python3 bin/backtest_knowledge_v2.py \
  --asset BTC --start 2023-01-01 --end 2023-12-31 \
  --config configs/btc_v8_adaptive.json

# Expect: weights hover between neutral/risk_off, selective entries
# Target: PF > 2.0, DD < 10%, trades > 25
```

### Step 4: Full-Span Validation (2022-2024)
```bash
python3 bin/backtest_knowledge_v2.py \
  --asset BTC --start 2022-01-01 --end 2024-12-31 \
  --config configs/btc_v8_adaptive.json \
  --export-regime-telemetry reports/ml/regime_telemetry.csv

# Target metrics (full span):
# - Overall PF > 3.0 (blended)
# - Max DD < 15% (vs 0% in 2024, 13.2% in 2022-2023 for v8)
# - Trades: 80-120 (balanced across regimes)
# - Smooth equity curve (no regime-switch whipsaws)
```

**Regime Telemetry Logging:**
Add to backtest export:
```csv
timestamp, regime, p_risk_on, p_neutral, p_risk_off, p_crisis, fusion_wt_wyckoff, fusion_wt_liquidity, min_liq_gate, final_fusion_gate, size_mult, ml_threshold
```

### Step 5: Ablation Test (Fixed vs Adaptive)
Compare against v8 static config:

| Config | 2024 PF | 2022-2023 PF | Combined PF | Max DD | Trades |
|--------|---------|--------------|-------------|---------|--------|
| v8 Static (bull-only) | 20.96 | 0.56 ❌ | N/A | 13.2% | 95 |
| v8 Adaptive | ~18-20 | ~1.8-2.5 ✅ | >3.0 ✅ | <15% | 100-120 |

**Key Insight:** Adaptive should sacrifice ~10-15% PF in bull markets to gain 3-4x PF in bear markets.

### Step 6: Calibration & Production Lock
If ablation passes:
1. **Freeze regime profiles** (no more hand-tuning)
2. **Weekly threshold recalibration:**
   ```bash
   python3 bin/train/calibrate_thresholds.py \
     --model models/btc_trade_quality_filter_v1.pkl \
     --recent-data reports/ml/trades_last_90d.csv \
     --output configs/ml_thresholds_runtime.json
   ```
3. **Lock as `configs/btc_v8_production.json`**
4. **Start 30-day paper trading**

---

## 🔬 Validation Checklist

- [ ] Create `configs/btc_v8_adaptive.json` with regime profiles
- [ ] Wire `AdaptiveFusion` into `bin/backtest_knowledge_v2.py` main loop
- [ ] Test 2024 only (expect PF ≈ 20.96, validates bull profile)
- [ ] Test 2022 only (expect PF > 1.5, validates bear profile)
- [ ] Test 2023 only (expect PF > 2.0, validates chop profile)
- [ ] Test 2022-2024 full span (target: PF > 3.0, DD < 15%)
- [ ] Export regime telemetry for inspection
- [ ] Ablation: compare fixed vs adaptive weights
- [ ] Leak check: ensure regime features use only past data
- [ ] Lock production config if validation passes

---

## 🎯 Expected Outcomes

**Regime-Specific Performance:**
- Bull (risk_on): PF 15-20, trades 35-45, DD 0-2%
- Chop (neutral): PF 2-3, trades 25-35, DD 5-8%
- Bear (risk_off): PF 1.5-2.5, trades 20-30, DD 8-12%
- Crisis: Abstain or minimal (PF >1, trades <5, tight stops)

**Blended (2022-2024):**
- Overall PF: 3.0-4.5 (vs 0.56 bear-only for static v8)
- Max DD: 12-15% (vs 13.2% bear-only)
- Sharpe: 1.5-2.0 (improved risk-adjusted returns)
- Smoother equity curve (no regime-switch shocks)

---

## 🚀 Path to Specter

Once adaptive approach is validated:
1. **Log all adapted parameters + outcomes** (regime_probs, weights, gates, PNL per trade)
2. **Build dataset:** `[macro_features, regime_probs] → [optimal_weights, gates, sizing]`
3. **Train Specter (RL or supervised):**
   - Input: Current macro state + recent performance
   - Output: Regime profiles (replaces hand-tuned profiles)
   - Objective: Maximize Sharpe across full regime spectrum
4. **Deploy:** Specter continuously learns optimal profiles, no more hand-tuning

**Adaptive Fusion is the bridge to Specter** - provides the framework and training data.

---

## 📝 Notes

- Regime labels: `risk_on=bull, neutral=chop, risk_off=bear, crisis=crisis`
- EMA alpha=0.2 → 5-bar smoothing window (~5 hours at 1H bars)
- Min weight floor=0.05 prevents any component from zeroing out
- Sizing limits [0.6, 1.35] prevent extreme leverage swings
- All blending is convex (weighted average), guarantees smooth transitions
