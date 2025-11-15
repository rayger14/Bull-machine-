# Adaptive Fusion Wiring - Minimal Changes to backtest_knowledge_v2.py

**Goal:** Wire adaptive fusion into existing backtest engine with minimal disruption.

---

## Change 1: Add Imports (After line 30)

**Location:** bin/backtest_knowledge_v2.py:30 (after `import joblib`)

```python
import joblib  # ML model loading

# Adaptive Fusion (PR#6B): Regime-aware parameter morphing
try:
    from engine.context.regime_classifier import RegimeClassifier
    from engine.fusion.adaptive import AdaptiveFusion
    ADAPTIVE_FUSION_AVAILABLE = True
except ImportError:
    ADAPTIVE_FUSION_AVAILABLE = False
    logger.warning("Adaptive fusion modules not found - will run in static mode")
```

---

## Change 2: Initialize Adaptive Fusion in __init__ (After line 186)

**Location:** bin/backtest_knowledge_v2.py:186 (after ML filter initialization)

```python
        # ML Filter initialization
        self.ml_model = None
        self.ml_feature_names = None
        self.ml_threshold = 0.707
        ml_config = self.runtime_config.get('ml_filter', {})
        if ml_config.get('enabled'):
            try:
                model_path = ml_config['model_path']
                model_data = joblib.load(model_path)
                self.ml_model = model_data['model']
                self.ml_feature_names = model_data['feature_names']
                self.ml_threshold = float(ml_config.get('threshold', model_data.get('threshold', 0.707)))
                logger.info(f"ML Filter: loaded {model_path} (threshold={self.ml_threshold:.3f}, {len(self.ml_feature_names)} features)")
            except Exception as e:
                logger.warning(f"ML Filter: failed to load - {e}")
                self.ml_model = None

        # ADD THIS BLOCK HERE:
        # Adaptive Fusion initialization
        self.regime_classifier = None
        self.adaptive_fusion = None
        adaptive_config = self.runtime_config.get('fusion_adapt', {})
        regime_config = self.runtime_config.get('regime_classifier', {})

        if ADAPTIVE_FUSION_AVAILABLE and adaptive_config.get('enable'):
            try:
                # Load regime classifier
                model_path = regime_config.get('model_path', 'models/regime_classifier_gmm.pkl')
                feature_order = regime_config.get('feature_order', [])
                self.regime_classifier = RegimeClassifier.load(model_path, feature_order)

                # Initialize adaptive fusion coordinator
                self.adaptive_fusion = AdaptiveFusion(self.runtime_config)

                logger.info(f"Adaptive Fusion: ENABLED (ema_alpha={adaptive_config.get('ema_alpha', 0.2)})")
            except Exception as e:
                logger.warning(f"Adaptive Fusion: failed to initialize - {e}")
                self.regime_classifier = None
                self.adaptive_fusion = None
        else:
            logger.info("Adaptive Fusion: DISABLED (using static parameters)")

        # Continue with existing code
        self.trades: List[Trade] = []
        self.current_position: Optional[Trade] = None
```

---

## Change 3: Classify Regime & Adapt Parameters in run() Loop (After line 1620)

**Location:** bin/backtest_knowledge_v2.py:1620 (inside main loop, before fusion score computation)

**Before:**
```python
        for bar_idx, (idx, row) in enumerate(self.df.iterrows()):
            # Skip early bars without indicators
            if pd.isna(row.get('atr_14')):
                continue

            # Compute fusion score
            fusion_score, context = self.compute_advanced_fusion_score(row)
```

**After:**
```python
        for bar_idx, (idx, row) in enumerate(self.df.iterrows()):
            # Skip early bars without indicators
            if pd.isna(row.get('atr_14')):
                continue

            # Adaptive Fusion: Classify regime and get adapted parameters
            adapted_params = None
            if self.adaptive_fusion and self.regime_classifier:
                try:
                    # Extract macro features for regime classification
                    macro_row = {feat: row.get(feat, np.nan)
                                for feat in self.runtime_config['regime_classifier']['feature_order']}
                    regime_info = self.regime_classifier.classify(macro_row)
                    adapted_params = self.adaptive_fusion.update(regime_info)

                    # Override ML threshold if adaptive
                    if adapted_params.get('ml_threshold') is not None:
                        self.ml_threshold = adapted_params['ml_threshold']
                except Exception as e:
                    logger.warning(f"Adaptive fusion failed at bar {bar_idx}: {e}")
                    adapted_params = None

            # Compute fusion score (will use adapted weights if available)
            fusion_score, context = self.compute_advanced_fusion_score(row, adapted_params)
```

---

## Change 4: Modify compute_advanced_fusion_score() to Accept Adapted Params (Line 272)

**Location:** bin/backtest_knowledge_v2.py:272

**Before:**
```python
    def compute_advanced_fusion_score(self, row: pd.Series) -> Tuple[float, Dict]:
        """
        Compute advanced fusion score using ALL 114 features.

        Returns:
            (fusion_score, context_dict)
        """
        context = {}
```

**After:**
```python
    def compute_advanced_fusion_score(self, row: pd.Series, adapted_params: Optional[Dict] = None) -> Tuple[float, Dict]:
        """
        Compute advanced fusion score using ALL 114 features.

        Args:
            row: Feature store row
            adapted_params: Optional adaptive fusion parameters (from AdaptiveFusion.update())

        Returns:
            (fusion_score, context_dict)
        """
        context = {}

        # Store regime info in context for logging
        if adapted_params:
            context['regime'] = adapted_params.get('regime', 'neutral')
            context['regime_probs_ema'] = adapted_params.get('regime_probs_ema', {})
```

**Then find where fusion weights are used (around line 320-330) and adapt:**

Find this section (will be around line 320):
```python
        # Final fusion score (weighted average)
        fusion = (
            self.params.wyckoff_weight * wyckoff +
            self.params.liquidity_weight * liquidity +
            self.params.momentum_weight * momentum +
            self.params.pti_weight * pti +
            self.params.macro_weight * macro
        )
```

**Replace with:**
```python
        # Final fusion score (weighted average)
        # Use adapted weights if available, otherwise fallback to config/params
        if adapted_params and adapted_params.get('fusion_weights'):
            fusion_weights = adapted_params['fusion_weights']
            fusion = (
                fusion_weights.get('wyckoff', 0.0) * wyckoff +
                fusion_weights.get('liquidity', 0.0) * liquidity +
                fusion_weights.get('momentum', 0.0) * momentum +
                fusion_weights.get('temporal', 0.0) * context.get('temporal', 0.0)
            )
        else:
            # Fallback to static weights
            fusion = (
                self.params.wyckoff_weight * wyckoff +
                self.params.liquidity_weight * liquidity +
                self.params.momentum_weight * momentum +
                self.params.pti_weight * pti +
                self.params.macro_weight * macro
            )
```

---

## Change 5: Adapt Entry Gates in check_entry_conditions() (Find this method)

**Location:** Find `def check_entry_conditions(` method

Need to:
1. Accept `adapted_params` parameter
2. Use `adapted_params['gates']` for min_liquidity and final_fusion_floor if available

**Will implement this after confirming the method signature**

---

## Change 6: Adapt Sizing in calculate_position_size() (Find this method)

**Location:** Find `def calculate_position_size(` method

Need to:
1. Accept `adapted_params` parameter
2. Multiply base size by `adapted_params['size_mult']` if available

**Will implement this after confirming the method signature**

---

## Change 7: Adapt Exits in check_exit_conditions() (Find this method)

**Location:** Find `def check_exit_conditions(` method

Need to:
1. Accept `adapted_params` parameter
2. Use `adapted_params['exit_params']['trail_atr']` for trailing stop multiplier
3. Use `adapted_params['exit_params']['max_bars']` for max hold time

**Will implement this after confirming the method signature**

---

## Summary of Changes

1. ✅ **Imports**: Add regime classifier and adaptive fusion imports (2 lines)
2. ✅ **Init**: Initialize regime classifier + adaptive fusion in __init__ (~20 lines)
3. ✅ **Main Loop**: Classify regime and get adapted params each bar (~15 lines)
4. ✅ **Fusion Score**: Accept adapted params and use adapted weights (~10 lines modified)
5. ⏳ **Entry Gates**: Pass adapted gates to entry checks (TBD - need method signature)
6. ⏳ **Sizing**: Apply adapted size multiplier (TBD - need method signature)
7. ⏳ **Exits**: Apply adapted exit params (TBD - need method signature)

**Total Impact**: ~50-70 lines added/modified in ~2100 line file (minimal disruption)

---

## Testing Protocol

After wiring:

1. **Syntax check**: `python3 -m py_compile bin/backtest_knowledge_v2.py`
2. **Static config (fallback)**: Run with v8 config (should work exactly as before)
3. **Adaptive config (2024)**: Run with btc_v8_adaptive.json on 2024 data
4. **Cross-regime (2022-2024)**: Full validation protocol

