# LogisticRegimeModel V3 - Hybrid Solution (ML + Rules)

**Problem**: V3 model has 0% crisis recall despite proper training
**Solution**: Use ML for normal regimes, rules for crisis detection
**Timeline**: 2-4 hours implementation
**Expected Result**: 60%+ crisis recall, maintains good neutral/risk_off/risk_on

---

## Architecture

### Two-Stage Classification

```
Stage 1: Rule-Based Crisis Override
├─ IF crash_frequency_7d >= 2        → crisis (high confidence)
├─ IF crisis_persistence >= 0.7      → crisis (sustained crisis)
├─ IF RV_7 > 100 AND drawdown > 30%  → crisis (extreme vol + drawdown)
└─ ELSE → proceed to Stage 2

Stage 2: ML-Based Regime Classification
├─ Use LogisticRegimeModel v3
├─ Classify as: neutral, risk_off, risk_on
└─ Return probabilities for all 4 classes
```

### Rationale
- **Crisis is too important to miss**: Better false positives than false negatives
- **Crisis is rare and extreme**: Rule-based detection is reliable
- **ML excels at normal regimes**: Use strengths of each approach
- **Graceful degradation**: If rules fail, ML still provides fallback

---

## Implementation

### 1. Create Hybrid Classifier

```python
# File: engine/context/hybrid_regime_model.py

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class HybridRegimeModel:
    """
    Hybrid regime classifier: Rule-based crisis + ML-based normal regimes.

    Crisis Detection Rules (OR logic):
    1. crash_frequency_7d >= 2 (multiple crashes in week)
    2. crisis_persistence >= 0.7 (sustained crisis score)
    3. RV_7 > 100 AND drawdown_persistence > 0.8 (extreme vol + deep drawdown)

    Normal Regime Classification:
    - Uses LogisticRegimeModel v3
    - Classes: risk_off, neutral, risk_on
    """

    def __init__(self, model_path: str = None):
        """
        Initialize hybrid model.

        Args:
            model_path: Path to trained LogisticRegimeModel v3 pkl file
        """
        # Load ML model
        if model_path is None:
            model_path = Path(__file__).parent.parent.parent / 'models' / 'logistic_regime_v3.pkl'

        import pickle
        with open(model_path, 'rb') as f:
            self.ml_artifact = pickle.load(f)

        self.calibrator = self.ml_artifact['calibrator']
        self.scaler = self.ml_artifact['scaler']
        self.feature_order = self.ml_artifact['feature_order']

        logger.info(f"Loaded hybrid regime model from {model_path}")
        logger.info(f"  ML classes: {list(self.calibrator.classes_)}")
        logger.info(f"  Features: {len(self.feature_order)}")

    def _check_crisis_rules(self, features: Dict[str, float]) -> tuple:
        """
        Check rule-based crisis detection.

        Returns:
            (is_crisis: bool, confidence: float, trigger: str)
        """
        # Rule 1: Multiple crashes in last 7 days
        if features.get('crash_frequency_7d', 0) >= 2:
            confidence = min(1.0, features['crash_frequency_7d'] / 3.0)
            return True, confidence, 'crash_frequency_7d >= 2'

        # Rule 2: Sustained crisis persistence
        if features.get('crisis_persistence', 0) >= 0.7:
            confidence = features['crisis_persistence']
            return True, confidence, 'crisis_persistence >= 0.7'

        # Rule 3: Extreme volatility + deep drawdown
        rv7 = features.get('RV_7', 0)
        dd_persist = features.get('drawdown_persistence', 0)
        if rv7 > 100 and dd_persist > 0.8:
            confidence = min(1.0, (rv7 / 150) * dd_persist)
            return True, confidence, f'RV_7={rv7:.0f} + drawdown={dd_persist:.2f}'

        return False, 0.0, None

    def classify(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Full hybrid classification.

        Args:
            features: Dict mapping feature name → value

        Returns:
            Dict with:
            - regime_label: str (crisis/risk_off/neutral/risk_on)
            - regime_probs: Dict[str, float] (posterior probabilities)
            - regime_confidence: float
            - regime_source: str ('rule_crisis' or 'ml')
            - crisis_trigger: str (if crisis detected by rules)
        """
        # Stage 1: Check crisis rules
        is_crisis, crisis_conf, trigger = self._check_crisis_rules(features)

        if is_crisis:
            # Crisis detected by rules
            return {
                'regime_label': 'crisis',
                'regime_probs': {
                    'crisis': crisis_conf,
                    'risk_off': 1 - crisis_conf,
                    'neutral': 0.0,
                    'risk_on': 0.0
                },
                'regime_confidence': crisis_conf,
                'regime_source': 'rule_crisis',
                'crisis_trigger': trigger
            }

        # Stage 2: ML classification for normal regimes
        # Extract features
        x = np.array([features.get(f, 0.0) for f in self.feature_order], dtype=float)
        x = np.nan_to_num(x, nan=0.0)

        # Scale and predict
        x_scaled = self.scaler.transform([x])
        ml_probs_raw = self.calibrator.predict_proba(x_scaled)[0]

        # Map to regime labels
        ml_classes = self.calibrator.classes_
        ml_probs = {str(ml_classes[i]): float(ml_probs_raw[i]) for i in range(len(ml_classes))}

        # Ensure all 4 regimes have probabilities
        full_probs = {
            'crisis': ml_probs.get('crisis', 0.0),
            'risk_off': ml_probs.get('risk_off', 0.0),
            'neutral': ml_probs.get('neutral', 0.0),
            'risk_on': ml_probs.get('risk_on', 0.0)
        }

        # Find top regime
        regime_label = max(full_probs.items(), key=lambda x: x[1])[0]

        # Compute confidence
        sorted_probs = sorted(full_probs.values(), reverse=True)
        confidence = sorted_probs[0] - sorted_probs[1]

        return {
            'regime_label': regime_label,
            'regime_probs': full_probs,
            'regime_confidence': confidence,
            'regime_source': 'ml',
            'crisis_trigger': None
        }

    def classify_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify entire DataFrame with hybrid logic.

        Args:
            df: DataFrame with feature columns

        Returns:
            DataFrame with regime columns added
        """
        logger.info(f"Classifying {len(df)} bars with Hybrid Regime Model...")

        results = []
        for idx, row in df.iterrows():
            features = row.to_dict()
            result = self.classify(features)
            results.append(result)

        # Convert to DataFrame
        result_df = df.copy()
        result_df['regime_label'] = [r['regime_label'] for r in results]
        result_df['regime_confidence'] = [r['regime_confidence'] for r in results]
        result_df['regime_source'] = [r['regime_source'] for r in results]

        # Add probabilities
        for regime in ['crisis', 'risk_off', 'neutral', 'risk_on']:
            result_df[f'regime_proba_{regime}'] = [r['regime_probs'][regime] for r in results]

        # Log distribution
        regime_dist = result_df['regime_label'].value_counts()
        logger.info(f"\nRegime distribution:")
        for regime, count in regime_dist.items():
            pct = count / len(result_df) * 100
            source_mask = result_df['regime_label'] == regime
            sources = result_df.loc[source_mask, 'regime_source'].value_counts()
            logger.info(f"  {regime:12s}: {count:6d} ({pct:5.1f}%) - {dict(sources)}")

        # Crisis triggers
        crisis_mask = result_df['regime_source'] == 'rule_crisis'
        n_crisis = crisis_mask.sum()
        if n_crisis > 0:
            logger.info(f"\nCrisis detections: {n_crisis} bars (rule-based)")

        return result_df


# Example usage
if __name__ == '__main__':
    import sys
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

    print("=" * 80)
    print("Hybrid Regime Model - ML + Rule-Based Crisis Detection")
    print("=" * 80)

    model = HybridRegimeModel()

    # Test crisis detection
    print("\nTest 1: Normal market (should use ML)")
    features = {f: 0.0 for f in model.feature_order}
    result = model.classify(features)
    print(f"  Regime: {result['regime_label']}")
    print(f"  Source: {result['regime_source']}")
    print(f"  Confidence: {result['regime_confidence']:.3f}")

    print("\nTest 2: Crisis (crash_frequency_7d=3)")
    features['crash_frequency_7d'] = 3.0
    result = model.classify(features)
    print(f"  Regime: {result['regime_label']}")
    print(f"  Source: {result['regime_source']}")
    print(f"  Trigger: {result.get('crisis_trigger')}")
    print(f"  Confidence: {result['regime_confidence']:.3f}")
```

### 2. Validate on LUNA Period

```python
# File: bin/validate_hybrid_regime_luna.py

#!/usr/bin/env python3
"""
Validate Hybrid Regime Model on LUNA crash period.

Expected: 60%+ crisis recall
"""

import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.context.hybrid_regime_model import HybridRegimeModel
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def main():
    # Load data
    data_path = Path(__file__).parent.parent / 'data' / 'features_mtf' / 'BTC_1H_2022-01-01_to_2024-12-31.parquet'
    df = pd.read_parquet(data_path)

    # LUNA period
    luna_df = df.loc['2022-05-07':'2022-05-15']
    logger.info(f"LUNA crash period: {len(luna_df)} bars")

    # Load hybrid model
    model = HybridRegimeModel()

    # Classify
    result_df = model.classify_batch(luna_df)

    # Crisis recall
    crisis_count = (result_df['regime_label'] == 'crisis').sum()
    crisis_recall = crisis_count / len(result_df) * 100

    logger.info("\n" + "=" * 80)
    logger.info("LUNA Crisis Detection Results")
    logger.info("=" * 80)
    logger.info(f"\nCrisis bars detected: {crisis_count}/{len(result_df)} ({crisis_recall:.1f}%)")
    logger.info(f"Target: >60%")

    if crisis_recall >= 60:
        logger.info(f"✅ SUCCESS: {crisis_recall:.1f}% >= 60%")
    else:
        logger.warning(f"⚠️ BELOW TARGET: {crisis_recall:.1f}% < 60%")

    # Distribution
    logger.info("\nPredicted regime distribution:")
    for regime, count in result_df['regime_label'].value_counts().items():
        pct = count / len(result_df) * 100
        logger.info(f"  {regime:12s}: {count:6d} ({pct:5.1f}%)")

    # Crisis triggers
    crisis_mask = result_df['regime_label'] == 'crisis'
    if crisis_mask.sum() > 0:
        logger.info("\nCrisis detection details:")
        logger.info(f"  crash_frequency_7d: {luna_df.loc[crisis_mask, 'crash_frequency_7d'].mean():.2f} avg")
        logger.info(f"  crisis_persistence: {luna_df.loc[crisis_mask, 'crisis_persistence'].mean():.3f} avg")
        logger.info(f"  RV_7: {luna_df.loc[crisis_mask, 'RV_7'].mean():.1f} avg")

if __name__ == '__main__':
    main()
```

---

## Validation Plan

### Step 1: Implement Hybrid Model (1 hour)
```bash
# Create hybrid model class
touch engine/context/hybrid_regime_model.py
# Copy implementation from above

# Create validation script
touch bin/validate_hybrid_regime_luna.py
# Copy validation script from above
```

### Step 2: Test on LUNA (15 min)
```bash
python3 bin/validate_hybrid_regime_luna.py
```

**Expected Output**:
- Crisis recall: 70-90% (rule-based should catch most)
- Crisis trigger: Mostly `crash_frequency_7d >= 2` or `crisis_persistence >= 0.7`
- False positives: Low (crisis features are reliable)

### Step 3: Test on Full 2022 (15 min)
```bash
# Validate on entire 2022
python3 -c "
from engine.context.hybrid_regime_model import HybridRegimeModel
import pandas as pd

df = pd.read_parquet('data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet')
df_2022 = df.loc['2022']

model = HybridRegimeModel()
result = model.classify_batch(df_2022)

# Check distribution
print(result['regime_label'].value_counts())
print(f'\nCrisis detections: {(result[\"regime_label\"] == \"crisis\").sum()}')
print(f'Rule-based: {(result[\"regime_source\"] == \"rule_crisis\").sum()}')
"
```

**Expected Output**:
- Crisis: 1-5% of 2022 (should catch LUNA + FTX)
- Risk_off: 40-50%
- Neutral: 20-30%
- Risk_on: 20-30%

### Step 4: Integrate into Production (2 hours)
1. Update `RegimeService` to use `HybridRegimeModel`
2. Update backtest scripts to support hybrid classification
3. Re-run Phase 1 backtest (2022 with hybrid regime)
4. Compare vs pure ML v3

---

## Expected Results

### Before (V3 ML Only)
```
LUNA Crisis Detection:
  - Recall: 0%
  - Avg crisis prob: 2.5%
  - All classified as risk_off

2022 Distribution:
  - Crisis: 0.0%
  - Risk_off: 67.5%
  - Neutral: 11.5%
  - Risk_on: 21.0%
```

### After (V3 Hybrid)
```
LUNA Crisis Detection:
  - Recall: 70-90% ← FIXED
  - Trigger: crash_frequency_7d >= 2
  - Rule-based override

2022 Distribution:
  - Crisis: 2-4% (LUNA + FTX)
  - Risk_off: 45-55%
  - Neutral: 20-30%
  - Risk_on: 20-25%
```

---

## Advantages

1. **Robust Crisis Detection**: Rules catch extreme events ML misses
2. **Best of Both Worlds**: ML for nuanced regimes, rules for obvious crises
3. **Interpretable**: Crisis triggers are explicit, actionable
4. **Production-Ready**: Can deploy today with high confidence
5. **Graceful Degradation**: If rules fail, ML still works

---

## Disadvantages

1. **Rule Maintenance**: Need to tune thresholds over time
2. **Not Pure ML**: Hybrid is less elegant than pure learning
3. **Potential Over-Detection**: Rules may fire on non-crisis volatility spikes
4. **Feature Dependency**: Relies on `crash_frequency_7d`, `crisis_persistence` being correct

---

## Next Steps

1. ✅ **Document hybrid approach** (this file)
2. **Implement `HybridRegimeModel` class**
3. **Validate on LUNA** (expect 70%+ recall)
4. **Validate on full 2022** (expect 2-4% crisis)
5. **Integrate into `RegimeService`**
6. **Re-run Phase 1 backtest**
7. **Compare PF: V2 (0.58) vs V3 Hybrid (target: 1.2+)**

---

## Decision Point

**Proceed with Hybrid Model?**

✅ **YES** - Recommended for immediate production
- Fast to implement (2-4 hours)
- High confidence in crisis detection
- Maintains ML benefits for normal regimes
- Can improve ML later while rules provide safety net

❌ **NO** - If you prefer pure ML
- Need to collect more crisis training data (2018-2021)
- Retrain with 20% crisis SMOTE (not 10%)
- Try different model architectures (Random Forest, Gradient Boosting)
- Timeline: 1-2 weeks

**Recommendation**: Proceed with Hybrid for Phase 1, improve pure ML in parallel for Phase 2.
