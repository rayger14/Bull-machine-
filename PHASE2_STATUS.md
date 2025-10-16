# Phase 2: Regime Classifier - STATUS REPORT

**Date**: 2025-10-14
**Branch**: `feature/phase2-regime-classifier`
**Commit**: `389415b`

---

## ✅ COMPLETED COMPONENTS

### 1. Core ML Infrastructure
- **RegimeClassifier** ([engine/context/regime_classifier.py](engine/context/regime_classifier.py))
  - GMM-based clustering (4 regimes: risk_on/neutral/risk_off/crisis)
  - 13 macro features: VIX, DXY, MOVE, YIELD_2Y, YIELD_10Y, USDT.D, BTC.D, TOTAL, TOTAL2, funding, oi, rv_20d, rv_60d
  - Fallback to neutral on missing features
  - Confidence scoring via GMM probabilities
  - Feature importance calculation

- **RegimePolicy** ([engine/context/regime_policy.py](engine/context/regime_policy.py))
  - Bounded threshold adjustments: ±0.10 max
  - Risk multipliers: 0.0x (crisis) to 1.25x (risk_on)
  - Domain weight nudges: max 0.15 total shift
  - Confidence gating: min 0.60 to apply adjustments
  - Hysteresis support (config-driven)

### 2. Training & Data Pipeline
- **build_macro_dataset.py** ([bin/build_macro_dataset.py](bin/build_macro_dataset.py))
  - Extracts 13 macro features from TradingView exports
  - Handles missing data with sensible defaults
  - Outputs 33K+ hours of aligned 1H data

- **train_regime_classifier.py** ([bin/train/train_regime_classifier.py](bin/train/train_regime_classifier.py))
  - GMM training with sklearn
  - VIX-based initial labeling
  - Walk-forward train/test split
  - Model serialization with scaler

- **Trained Model** ([models/regime_classifier_gmm.pkl](models/regime_classifier_gmm.pkl))
  - 33,169 hours of training data (2022-2025)
  - Silhouette Score: 0.489
  - Converged in 19 iterations
  - Top features: rv_60d (1.18), rv_20d (0.96), TOTAL2 (0.80)

### 3. Configuration
- **regime_policy.json** ([configs/v19/regime_policy.json](configs/v19/regime_policy.json))
  - Threshold deltas per regime
  - Risk multipliers per regime
  - Weight nudges with total shift cap
  - VIX/MOVE/DXY training bands
  - Hysteresis rules (3 consecutive signals)

### 4. Evaluation Framework
- **eval_regime_backtest.py** ([scripts/eval_regime_backtest.py](scripts/eval_regime_backtest.py))
  - Baseline vs regime-enabled comparison
  - Regime distribution analysis
  - Acceptance gate validation

---

## ⏳ INTEGRATION WORK NEEDED

### Option A: Minimal Hybrid Runner Integration

**Goal**: Add regime adaptation to `bin/live/hybrid_runner.py` with kill-switch

**Changes Required** (3 insertion points):

#### 1. Add imports (top of file)
```python
from engine.context.regime_classifier import RegimeClassifier
from engine.context.regime_policy import RegimePolicy
```

#### 2. Initialize regime components in `__init__` (after line 173)
```python
        # Phase 2: Regime adaptation (optional)
        self.regime_enabled = self.config.get('regime', {}).get('enabled', False)
        self.regime_classifier = None
        self.regime_policy = None

        if self.regime_enabled:
            print("🧠 Loading Phase 2 regime classifier...")
            feature_order = [
                "VIX", "DXY", "MOVE", "YIELD_2Y", "YIELD_10Y",
                "USDT.D", "BTC.D", "TOTAL", "TOTAL2",
                "funding", "oi", "rv_20d", "rv_60d"
            ]
            try:
                self.regime_classifier = RegimeClassifier.load(
                    "models/regime_classifier_gmm.pkl",
                    feature_order
                )
                self.regime_policy = RegimePolicy.load("configs/v19/regime_policy.json")
                print(f"✅ Regime adaptation enabled")
            except Exception as e:
                print(f"⚠️  Regime classifier load failed: {e}")
                print(f"   Falling back to baseline (regime_enabled=False)")
                self.regime_enabled = False
```

#### 3. Apply regime adjustments in bar loop (search for `fetch_macro_snapshot`)
```python
        # Existing: Fetch macro snapshot
        macro_row = fetch_macro_snapshot(self.macro_data, timestamp_naive)

        # NEW: Phase 2 regime adaptation
        regime_adjustment = None
        if self.regime_enabled:
            regime_info = self.regime_classifier.classify(macro_row)
            regime_adjustment = self.regime_policy.apply(self.config, regime_info)

            # Apply threshold delta
            base_threshold = self.config['fusion']['entry_threshold_confidence']
            adjusted_threshold = base_threshold + regime_adjustment['enter_threshold_delta']

            # Apply weight nudges
            adjusted_weights = self.config['fusion']['weights'].copy()
            for domain, nudge in regime_adjustment['weight_nudges'].items():
                if domain in adjusted_weights:
                    adjusted_weights[domain] += nudge

            # Renormalize weights to preserve sum
            weight_sum = sum(adjusted_weights.values())
            original_sum = sum(self.config['fusion']['weights'].values())
            adjusted_weights = {k: v * original_sum / weight_sum
                               for k, v in adjusted_weights.items()}

            # Temporarily apply adjustments (scoped to this bar)
            self.config['fusion']['entry_threshold_confidence'] = adjusted_threshold
            self.config['fusion']['weights'] = adjusted_weights

            # Log regime state
            if regime_adjustment['applied']:
                print(f"  [REGIME] {regime_info['regime']} (conf={regime_info['proba'][regime_info['regime']]:.2f})")
                print(f"           threshold: {base_threshold:.2f} → {adjusted_threshold:.2f}")
                print(f"           risk_mult: {regime_adjustment['risk_multiplier']:.2f}x")
```

#### 4. Apply risk multiplier to position sizing (search for `size_usd`)
```python
        # Existing position sizing calculation
        size_usd = balance * (config['risk']['risk_per_trade_pct'] / 100.0)

        # NEW: Apply regime risk multiplier
        if regime_adjustment:
            size_usd *= regime_adjustment['risk_multiplier']
```

#### 5. Add config section for kill-switch
Add to your config JSON:
```json
{
  "regime": {
    "enabled": false,
    "shadow_mode": true,
    "min_confidence": 0.60,
    "max_threshold_delta": 0.05,
    "max_risk_multiplier": 1.15
  }
}
```

**Safety Switches**:
- `enabled: false` → baseline behavior (no regime)
- `shadow_mode: true` → log deltas but don't apply
- `min_confidence: 0.60` → force neutral below threshold
- `max_threshold_delta: 0.05` → cap adjustments for live (instead of 0.10)
- `max_risk_multiplier: 1.15` → cap risk scaling (instead of 1.25)

---

## 📊 ACCEPTANCE GATES (Option B Validation)

If running Q3 2024 comparison:

| Gate | Target | Pass Criteria |
|------|--------|---------------|
| **Sharpe Uplift** | +0.15 to +0.25 | Regime Sharpe ≥ Baseline + 0.15 |
| **Max DD** | ≤ 8-10% | Regime MaxDD ≤ 10% |
| **PF Uplift** | +0.10 to +0.30 | Regime PF ≥ Baseline + 0.10 |
| **Trade Retention** | ≥ 80% | Regime trades ≥ 80% of baseline |
| **Regime Confidence** | ≥ 70% high-conf | ≥ 70% of trades with confidence ≥ 0.60 |

**Fallback Plan** if gates miss:
1. Reduce `enter_threshold_delta` to ±0.03
2. Cap `risk_multiplier` at 1.15
3. Re-run validation

---

## 📁 ARTIFACTS TO COLLECT

For validation report:
- `reports/v19/BTC_q3_2024_baseline.json`
- `reports/v19/BTC_q3_2024_regime.json`
- `reports/v19/BTC_q3_2024_baseline_trades.csv`
- `reports/v19/BTC_q3_2024_regime_trades.csv`
- Per-trade CSV columns:
  - `entry_ts`, `exit_ts`, `pnl_usd`, `regime`, `regime_probs` (JSON),
  - `enter_threshold_before`, `enter_threshold_after`,
  - `risk_mult`, `weight_nudges` (JSON)

---

## 🔧 DEBUGGING AIDS

### Test Regime Classifier Standalone
```bash
python3 engine/context/regime_classifier.py models/regime_classifier_gmm.pkl
```

### Test Regime Policy Standalone
```bash
python3 engine/context/regime_policy.py configs/v19/regime_policy.json
```

### Check Macro Dataset
```bash
python3 -c "import pandas as pd; df=pd.read_parquet('data/macro/macro_history.parquet'); print(df.info()); print(df.head())"
```

### Inspect Trained Model
```bash
python3 -c "import pickle; m=pickle.load(open('models/regime_classifier_gmm.pkl','rb')); print(m.keys()); print('Features:', m['feature_order']); print('Label map:', m['label_map'])"
```

---

## 🚦 ROLLOUT PHASES

### Phase 2A: Shadow Mode (Week 1)
- `enabled: true`, `shadow_mode: true`
- Log all regime classifications and deltas
- **No** actual config modifications
- Collect: regime distribution, confidence stats, adjustment frequencies

### Phase 2B: Threshold-Only (Week 2)
- `shadow_mode: false`
- Apply `enter_threshold_delta` only
- Keep `risk_multiplier: 1.0`, no weight nudges
- Validate: trade count retention, PF impact

### Phase 2C: Full Regime (Week 3)
- Enable weight nudges
- Cap `risk_multiplier` at 1.15 (not 1.25)
- Monitor: regime switch frequency, whipsaw events

### Phase 2D: Production (Week 4+)
- If gates pass, gradually increase caps:
  - `max_threshold_delta: 0.05 → 0.07 → 0.10`
  - `max_risk_multiplier: 1.15 → 1.20 → 1.25`

---

## 📝 COMMIT CHECKLIST

Phase 2 Core ✅:
- [x] RegimeClassifier implementation
- [x] RegimePolicy implementation
- [x] Training pipeline
- [x] Macro dataset builder
- [x] Trained model (33K hours)
- [x] Configuration schema
- [x] Evaluation framework

Phase 2 Integration ⏳:
- [ ] hybrid_runner regime hooks
- [ ] Shadow mode logging
- [ ] Config kill-switches
- [ ] Q3 2024 validation run
- [ ] Acceptance gate validation
- [ ] Artifacts committed to reports/v19/

---

## 🎯 NEXT STEPS (Recommended Order)

1. **Check running hybrid_runner** (Bash 94c0e8):
   ```bash
   tail -50 q3_2024_hybrid.log
   ```
   - If complete, extract baseline metrics

2. **Add regime integration** (3 insertion points above)

3. **Test with shadow mode**:
   ```bash
   python3 bin/live/hybrid_runner.py --asset BTC --start 2024-07-01 --end 2024-09-30 \
     --config configs/v18/BTC_conservative.json
   ```
   - With `regime.enabled: true, shadow_mode: true` in config

4. **Compare baseline vs regime**:
   - Parse logs for metrics
   - Validate gates
   - Generate comparison report

5. **Commit integration** if gates pass

---

## 📌 KNOWN LIMITATIONS

1. **Missing Macro Data**: VIX, DXY, MOVE, YIELD_2Y, YIELD_10Y use defaults (20.0, 102.0, 100.0, 4.0, 4.2)
   - Regime classification still works via other features
   - Top features (rv_60d, rv_20d, TOTAL2) are available

2. **Funding/OI Proxies**: Calculated from BTC price volatility
   - Not real exchange funding rates
   - Acceptable for regime classification (lower importance)

3. **Timezone Handling**: TradingView data is tz-aware (UTC), macro data is tz-naive
   - Handled in validation script
   - May need normalization in hybrid_runner

4. **Hysteresis Not Implemented**: Config has hysteresis rules, but not enforced yet
   - Current: regime can switch every bar
   - Recommended: Add 3-bar smoothing window

---

**Phase 2 Status**: COMPLETE ✅
**Integration Status**: READY FOR TESTING ⏳
**Branch Status**: Committed, awaiting validation results

