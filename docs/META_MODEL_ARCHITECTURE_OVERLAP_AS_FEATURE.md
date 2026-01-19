# Meta-Model Architecture: Treating Archetype Overlap as a Feature

**Version**: 1.0
**Author**: System Architect (Claude Code)
**Date**: 2025-12-17
**Status**: Architecture Specification

---

## Executive Summary

Traditional thinking treats archetype signal overlap as redundancy to be eliminated. This architecture **inverts the paradigm**: overlap becomes a feature that encodes confluence strength, disagreement signals, and optimal combination patterns.

**Core Insight**: Instead of forcing 16 archetypes to be independent (hard), build a meta-model that learns WHICH overlaps predict wins/losses (tractable).

**Key Results Expected**:
- Win rate improvement: 55% → 60-65% (5-10% absolute gain)
- Sharpe ratio improvement: +0.3 (e.g., 1.5 → 1.8)
- Trade frequency optimization: Filter out 30-40% of losing signals
- Interpretability: SHAP analysis reveals which overlap patterns work

---

## 1. The Overlap-as-Feature Paradigm Shift

### 1.1 Old Approach (Redundancy Elimination)

```python
# Traditional thinking: avoid overlap
if archetype_A.fires() and archetype_C.fires():
    # Redundant signal - pick one (arbitrary)
    signal = archetype_A.signal  # Or C, or merge somehow
```

**Problems**:
- Throws away valuable confluence information
- Forces artificial independence constraints
- Misses interaction effects (A+C together might be better than either alone)
- No mechanism to detect harmful overlaps (A+C+G might be worse than A+C)

### 1.2 New Approach (Overlap as Signal)

```python
# Meta-model sees overlap as structured feature
if archetype_A.fires() and archetype_C.fires():
    overlap_features = {
        'num_firing': 2,
        'A_conf': 0.87,
        'C_conf': 0.92,
        'avg_conf': 0.895,
        'all_agree_long': True,
        'regime': 'risk_on',
        'A_and_C': 1,  # Binary interaction flag
    }

    # Meta-model learned: A+C in risk_on → 68% win rate (take it!)
    meta_prob = meta_model.predict(overlap_features)  # 0.72

    if meta_prob > 0.65:
        return 'TAKE'
```

**Advantages**:
- Captures confluence (2 archetypes agreeing)
- Detects harmful overcrowding (5 archetypes = confusion)
- Learns regime-specific patterns (A+C works in bull, not bear)
- Quantifies uncertainty (mixed signals = skip)

---

## 2. Meta-Model Architecture

### 2.1 System Integration

```
┌─────────────────────────────────────────────────────────────┐
│                     BULL MACHINE PIPELINE                    │
└─────────────────────────────────────────────────────────────┘

1. Feature Store (970 features)
   └─> [liquidity_score, funding_Z, wyckoff_score, ...]

2. Domain Engines (6 engines)
   ├─> Wyckoff Engine
   ├─> SMC Engine
   ├─> Liquidity Engine
   ├─> Momentum Engine
   ├─> Macro Engine
   └─> Temporal Fusion Engine

3. Archetypes (16 pattern detectors)
   ├─> A (Spring/UTAD) → LONG (conf: 0.82)
   ├─> C (Wick Trap) → LONG (conf: 0.91)
   ├─> G (Liquidity Sweep) → LONG (conf: 0.78)
   ├─> S1 (Liquidity Vacuum) → LONG (conf: 0.85)
   └─> ... 12 others

4. Meta-Model Layer ← NEW
   ├─> Extract overlap features
   ├─> Classify with XGBoost/LightGBM
   ├─> Output: P(WIN) in [0, 1]
   └─> Decision: TAKE if P(WIN) > 0.65

5. Trade Execution
   └─> Execute position with confidence-based sizing
```

### 2.2 Meta-Model Input Layer

**Total Input Dimensionality**: ~70 features

#### 2.2.1 Per-Archetype Signals (16 × 5 = 80 features → sparse)

For each archetype (A, B, C, ..., S8):
- `{archetype}_fired`: Binary (0 or 1)
- `{archetype}_conf`: Confidence [0.0, 1.0]
- `{archetype}_direction`: Encoded (-1 SHORT, 0 FLAT, 1 LONG)
- `{archetype}_boost`: Domain boost multiplier [0.8, 1.2]
- `{archetype}_fusion_score`: Meta-fusion score [0.0, 1.0]

**Example**:
```python
features['A_fired'] = 1
features['A_conf'] = 0.82
features['A_direction'] = 1  # LONG
features['A_boost'] = 1.15
features['A_fusion_score'] = 0.78
```

**Dimensionality Reduction**: Use sparse encoding (only fired archetypes contribute non-zero values).

#### 2.2.2 Overlap Aggregates (15 features)

Derived from archetype signals:

```python
# Count-based features
features['num_archetypes_fired'] = 3  # A, C, G
features['num_long_signals'] = 3
features['num_short_signals'] = 0
features['num_flat_signals'] = 13

# Confidence statistics
features['avg_confidence'] = 0.857  # Mean of [0.82, 0.91, 0.78]
features['max_confidence'] = 0.91
features['min_confidence'] = 0.78
features['conf_std'] = 0.059  # Std dev (low = high agreement)

# Direction agreement
features['all_agree_long'] = 1  # Binary
features['all_agree_short'] = 0
features['mixed_signals'] = 0  # Binary (LONG and SHORT present)

# Boost aggregates
features['avg_boost'] = 1.02
features['max_boost'] = 1.15

# Fusion aggregates
features['avg_fusion_score'] = 0.81
features['max_fusion_score'] = 0.85
```

#### 2.2.3 Pairwise Interactions (Top 20 pairs)

Based on EDA (explore which pairs correlate with wins):

```python
# Binary flags for high-value pairs
features['A_and_C'] = 1  # Both A and C fired
features['A_and_G'] = 1  # Both A and G fired
features['C_and_G'] = 1  # Both C and G fired
features['S1_and_S4'] = 0  # Bear combinations
features['H_and_K'] = 0  # Bull momentum + trap

# ... top 20 pairs based on correlation with wins
```

**Selection Strategy**: Run mutual information analysis on historical data to identify top 20 interactions.

#### 2.2.4 Regime Context (7 features)

```python
# Current regime (4 binary one-hot encoded)
features['regime_risk_on'] = 1
features['regime_neutral'] = 0
features['regime_risk_off'] = 0
features['regime_crisis'] = 0

# Regime stability
features['regime_confidence'] = 0.82  # HMM confidence
features['regime_duration'] = 48  # Hours in current regime
features['regime_volatility'] = 0.15  # Regime transition rate (last 7d)
```

#### 2.2.5 Market Context (10-15 features, optional)

```python
# Volatility
features['atr_percentile'] = 0.65  # ATR rank in 90d window
features['rv_21d'] = 58.0  # Realized volatility (annualized %)

# Trend strength
features['adx_14'] = 32.5  # Trend strength
features['tf4h_trend'] = 1  # 4H external trend (LONG=1, SHORT=-1, FLAT=0)

# Liquidity
features['liquidity_score'] = 0.72
features['liquidity_drain_pct'] = -0.15  # Below 7d avg

# Volume
features['volume_zscore'] = 1.8  # Volume z-score

# Macro
features['VIX_Z'] = 0.5  # VIX z-score
features['DXY_Z'] = -0.3  # DXY z-score
features['funding_Z'] = 0.8  # Funding rate z-score
```

**Total Features**: ~70 (sparse archetype flags + 15 aggregates + 20 interactions + 7 regime + 15 market)

### 2.3 Meta-Model Output Layer

#### Option A: Binary Classification (Recommended for MVP)

```python
# Output: Single probability
output = meta_model.predict_proba(X)[0, 1]  # P(WIN)

# Decision logic
if output > 0.65:
    action = 'TAKE'  # High confidence win
elif output < 0.35:
    action = 'FADE'  # High confidence loss (optional: SHORT)
else:
    action = 'SKIP'  # Uncertain, avoid
```

**Labeling Strategy**:
```python
# For each historical signal:
actual_return_72h = compute_forward_return(signal_time, horizon=72)

if actual_return_72h > 2.0:  # 2% profit
    label = 1  # WIN
elif actual_return_72h < -1.0:  # -1% loss
    label = 0  # LOSS
else:
    label = None  # Neutral (exclude from training)
```

#### Option B: Regression (Expected Return)

```python
# Output: Predicted return in basis points
output = meta_model.predict(X)[0]  # e.g., +85 bps

# Decision logic
if output > 50:  # >0.5% expected return
    action = 'TAKE'
    position_size_mult = min(output / 100, 2.0)  # Scale with confidence
else:
    action = 'SKIP'
```

#### Option C: Multi-Class (Advanced)

```python
# Output: 4-class probabilities
probs = meta_model.predict_proba(X)[0]
# [P(STRONG_TAKE), P(WEAK_TAKE), P(SKIP), P(FADE)]

action = np.argmax(probs)
position_size = {
    'STRONG_TAKE': 2.0,  # 2x position
    'WEAK_TAKE': 0.5,    # 0.5x position
    'SKIP': 0.0,
    'FADE': -1.0         # SHORT
}[action]
```

**Recommendation**: Start with **Option A** (binary classification), migrate to **Option C** after validation.

---

## 3. Training Data Generation

### 3.1 Historical Backtest Labels

**Data Source**: Archetype signal logs from existing backtests (2022-2024)

```python
# Extract all historical signals
signals_df = load_historical_signals(
    period='2022-01-01 to 2024-12-31',
    archetypes=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'K', 'L', 'M',
                'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8']
)

# Compute forward returns
for idx, signal in signals_df.iterrows():
    entry_price = signal['entry_price']
    exit_price_72h = df_prices.loc[signal['timestamp'] + 72h, 'close']

    return_72h = (exit_price_72h - entry_price) / entry_price * 100

    # Label
    if return_72h > 2.0:
        signals_df.loc[idx, 'label'] = 1  # WIN
    elif return_72h < -1.0:
        signals_df.loc[idx, 'label'] = 0  # LOSS
    else:
        signals_df.loc[idx, 'label'] = None  # Neutral (skip)

# Remove neutrals
signals_df = signals_df.dropna(subset=['label'])

print(f"Training samples: {len(signals_df)}")
print(f"Win rate: {signals_df['label'].mean():.2%}")
```

**Expected Dataset Size**:
- 16 archetypes × 1,500 signals/regime × 3 regimes = 72,000 total signals
- After filtering neutrals: ~40,000 usable samples
- 80/20 split: 32,000 train, 8,000 validation

### 3.2 Stratified Sampling

**Prevent Imbalance**:

```python
from sklearn.model_selection import StratifiedShuffleSplit

# Stratify by regime AND label
signals_df['stratify_key'] = (
    signals_df['regime_label'] + '_' + signals_df['label'].astype(str)
)

splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_idx, val_idx in splitter.split(signals_df, signals_df['stratify_key']):
    train_df = signals_df.iloc[train_idx]
    val_df = signals_df.iloc[val_idx]

# Verify balance
print("Training set regime distribution:")
print(train_df['regime_label'].value_counts(normalize=True))

print("\nValidation set label balance:")
print(val_df['label'].value_counts(normalize=True))
```

**Objective**: Ensure each regime and each label is equally represented in train/val.

### 3.3 Walk-Forward Validation

**Prevent Lookahead Bias**:

```python
# Walk-forward protocol (6-month train, 1-month test)
train_windows = [
    ('2022-01', '2022-06', '2022-07'),  # Train Jan-Jun, test Jul
    ('2022-02', '2022-07', '2022-08'),  # Train Feb-Jul, test Aug
    ('2022-03', '2022-08', '2022-09'),  # Rolling...
    # ... continue through 2024
]

oos_results = []

for train_start, train_end, test_month in train_windows:
    # Train on historical window
    train_df = signals_df[
        (signals_df['timestamp'] >= train_start) &
        (signals_df['timestamp'] <= train_end)
    ]

    # Test on future month
    test_df = signals_df[
        signals_df['timestamp'].dt.to_period('M') == test_month
    ]

    # Train model
    model = train_xgboost(train_df)

    # Evaluate OOS
    oos_preds = model.predict_proba(test_df[features])[:, 1]
    oos_auc = roc_auc_score(test_df['label'], oos_preds)

    oos_results.append({
        'test_month': test_month,
        'oos_auc': oos_auc,
        'num_signals': len(test_df)
    })

# Average OOS performance
print(f"Mean OOS AUC: {np.mean([r['oos_auc'] for r in oos_results]):.3f}")
```

**Acceptance Criterion**: Mean OOS AUC > 0.65 (better than random guessing)

---

## 4. Model Selection

### 4.1 Candidate Models

#### 4.1.1 Logistic Regression (Baseline)

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Prepare features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train
model = LogisticRegression(
    penalty='l2',
    C=1.0,
    max_iter=1000,
    random_state=42
)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
auc = roc_auc_score(y_val, y_pred_proba)

print(f"Logistic Regression AUC: {auc:.3f}")
```

**Pros**:
- Interpretable (feature weights)
- Fast training and inference
- Works well with linear relationships

**Cons**:
- Cannot capture nonlinear interactions
- Requires manual interaction terms (A×C, A×G)

**Use Case**: Baseline for comparison, interpretability check

#### 4.1.2 Random Forest (Strong Baseline)

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=20,
    min_samples_leaf=10,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Feature importance
importances = pd.DataFrame({
    'feature': feature_names,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(importances.head(20))
```

**Pros**:
- Handles nonlinear interactions automatically
- Built-in feature importance
- Robust to outliers

**Cons**:
- Can overfit with deep trees
- Slower inference than LightGBM

**Use Case**: Strong baseline, feature importance analysis

#### 4.1.3 XGBoost/LightGBM (Production Choice)

```python
import lightgbm as lgb

# Prepare LightGBM dataset
train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

# Hyperparameters (tune with Optuna)
params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'max_depth': 8,
    'min_data_in_leaf': 20,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'verbosity': -1,
    'seed': 42
}

# Train
model = lgb.train(
    params,
    train_data,
    num_boost_round=500,
    valid_sets=[train_data, val_data],
    valid_names=['train', 'val'],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=50)
    ]
)

# Evaluate
y_pred_proba = model.predict(X_val)
auc = roc_auc_score(y_val, y_pred_proba)

print(f"LightGBM AUC: {auc:.3f}")
print(f"Best iteration: {model.best_iteration}")
```

**Pros**:
- Best predictive accuracy
- Fast training and inference
- Handles missing features gracefully
- Built-in regularization

**Cons**:
- Harder to interpret (use SHAP)
- Requires careful hyperparameter tuning

**Use Case**: **Production model** (best Sharpe ratio improvement)

#### 4.1.4 Neural Network (MLP) - Advanced

```python
import torch
import torch.nn as nn

class MetaModel(nn.Module):
    def __init__(self, input_dim=70):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)

# Train
model = MetaModel(input_dim=X_train.shape[1])
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ... training loop
```

**Pros**:
- Can learn complex nonlinear patterns
- End-to-end differentiable
- Extensible (add attention, embeddings)

**Cons**:
- Requires more data (40k might be marginal)
- Risk of overfitting
- Black box (requires SHAP for interpretability)

**Use Case**: Future upgrade after LightGBM validation

### 4.2 Recommended Approach

**Phase 1 (MVP)**:
1. Train Logistic Regression (baseline)
2. Train Random Forest (strong baseline)
3. Train LightGBM (production candidate)
4. Compare on OOS AUC, Sharpe ratio improvement

**Phase 2 (Production)**:
- Deploy LightGBM if OOS AUC > 0.65 AND Sharpe +0.3
- Use SHAP for interpretability
- Monitor feature drift weekly

**Phase 3 (Future)**:
- Experiment with Neural Network (MLP or Transformer)
- Add attention mechanism for archetype interactions
- Incremental learning (retrain monthly)

---

## 5. Feature Engineering for Overlap

### 5.1 Overlap Pattern Encoding

**Challenge**: How to represent "A+C+G fired together" without combinatorial explosion?

#### Approach 1: One-Hot Encoding (Sparse)

```python
# Only encode combinations that appear >10 times historically
from collections import Counter

# Count all overlap patterns
overlap_patterns = []
for idx, row in signals_df.iterrows():
    fired = [arch for arch in archetypes if row[f'{arch}_fired'] == 1]
    pattern = ','.join(sorted(fired))  # "A,C,G"
    overlap_patterns.append(pattern)

pattern_counts = Counter(overlap_patterns)

# Filter rare patterns
common_patterns = [p for p, count in pattern_counts.items() if count >= 10]

print(f"Total unique patterns: {len(pattern_counts)}")
print(f"Common patterns (n>=10): {len(common_patterns)}")

# Create one-hot features
for pattern in common_patterns:
    signals_df[f'pattern_{pattern}'] = (
        signals_df['overlap_pattern'] == pattern
    ).astype(int)
```

**Problem**: Still creates 100-500 binary features (sparse but large)

#### Approach 2: Hashing Trick

```python
# Hash overlap pattern to fixed-size feature space
def hash_pattern(pattern: str, n_buckets: int = 100) -> int:
    """Hash pattern to bucket [0, n_buckets-1]"""
    return hash(pattern) % n_buckets

# Create hash features
signals_df['pattern_hash'] = signals_df['overlap_pattern'].apply(
    lambda p: hash_pattern(p, n_buckets=100)
)

# One-hot encode hash buckets
hash_features = pd.get_dummies(
    signals_df['pattern_hash'],
    prefix='hash'
)
```

**Pros**: Fixed dimensionality (100 features)
**Cons**: Hash collisions (different patterns map to same bucket)

#### Approach 3: Count-Based Features (Recommended)

```python
# Aggregate statistics (no pattern enumeration)
def extract_overlap_features(row):
    """Extract overlap features from signal row"""
    features = {}

    # Count-based
    fired_archs = [a for a in archetypes if row[f'{a}_fired'] == 1]
    features['num_fired'] = len(fired_archs)
    features['num_long'] = sum(row[f'{a}_direction'] == 1 for a in fired_archs)
    features['num_short'] = sum(row[f'{a}_direction'] == -1 for a in fired_archs)

    # Confidence statistics
    confs = [row[f'{a}_conf'] for a in fired_archs]
    features['avg_conf'] = np.mean(confs) if confs else 0.0
    features['max_conf'] = np.max(confs) if confs else 0.0
    features['min_conf'] = np.min(confs) if confs else 0.0
    features['conf_std'] = np.std(confs) if len(confs) > 1 else 0.0

    # Direction agreement
    features['all_agree_long'] = int(features['num_long'] == features['num_fired'])
    features['all_agree_short'] = int(features['num_short'] == features['num_fired'])
    features['mixed_signals'] = int(
        features['num_long'] > 0 and features['num_short'] > 0
    )

    # Boost statistics
    boosts = [row[f'{a}_boost'] for a in fired_archs]
    features['avg_boost'] = np.mean(boosts) if boosts else 1.0
    features['max_boost'] = np.max(boosts) if boosts else 1.0

    return features

# Apply
overlap_features = signals_df.apply(extract_overlap_features, axis=1)
overlap_features_df = pd.DataFrame(list(overlap_features))
```

**Pros**: Fixed dimensionality (15 features), interpretable
**Cons**: Loses specific pattern identity (but pairwise interactions capture this)

#### Approach 4: Pairwise Interactions (For Logistic Regression)

```python
# Identify top 20 archetype pairs correlated with wins
from itertools import combinations
from sklearn.feature_selection import mutual_info_classif

# Generate all pairs
pairs = list(combinations(archetypes, 2))

# Compute mutual information for each pair
pair_scores = []
for a1, a2 in pairs:
    pair_feature = (
        signals_df[f'{a1}_fired'] & signals_df[f'{a2}_fired']
    ).astype(int)

    mi_score = mutual_info_classif(
        pair_feature.values.reshape(-1, 1),
        signals_df['label'],
        random_state=42
    )[0]

    pair_scores.append({
        'pair': f'{a1}_and_{a2}',
        'mi_score': mi_score,
        'frequency': pair_feature.sum()
    })

# Rank by MI score
pair_scores_df = pd.DataFrame(pair_scores).sort_values('mi_score', ascending=False)

# Select top 20 pairs
top_pairs = pair_scores_df.head(20)['pair'].tolist()

print("Top 20 archetype pairs by mutual information:")
print(pair_scores_df.head(20))

# Create binary features
for pair in top_pairs:
    a1, a2 = pair.split('_and_')
    signals_df[pair] = (
        signals_df[f'{a1}_fired'] & signals_df[f'{a2}_fired']
    ).astype(int)
```

**Pros**: Captures specific high-value interactions
**Cons**: Requires preprocessing, pairs may change over time

### 5.2 Recommended Feature Set

**Final Feature Vector** (~70 features):

```python
features = {
    # Individual archetype flags (16 × 1 = 16 binary)
    'A_fired': 0/1,
    'B_fired': 0/1,
    # ... S8_fired

    # Aggregate statistics (15 features)
    'num_fired': int,
    'num_long': int,
    'num_short': int,
    'avg_conf': float,
    'max_conf': float,
    'min_conf': float,
    'conf_std': float,
    'all_agree_long': 0/1,
    'all_agree_short': 0/1,
    'mixed_signals': 0/1,
    'avg_boost': float,
    'max_boost': float,
    'avg_fusion_score': float,
    'max_fusion_score': float,
    'min_fusion_score': float,

    # Top 20 pairwise interactions (20 binary)
    'A_and_C': 0/1,
    'A_and_G': 0/1,
    # ... top 20 from MI analysis

    # Regime context (7 features)
    'regime_risk_on': 0/1,
    'regime_neutral': 0/1,
    'regime_risk_off': 0/1,
    'regime_crisis': 0/1,
    'regime_confidence': float,
    'regime_duration': int,
    'regime_volatility': float,

    # Market context (10 features)
    'atr_percentile': float,
    'adx_14': float,
    'liquidity_score': float,
    'liquidity_drain_pct': float,
    'volume_zscore': float,
    'VIX_Z': float,
    'DXY_Z': float,
    'funding_Z': float,
    'rv_21d': float,
    'tf4h_trend': -1/0/1,
}
```

**Total**: 16 + 15 + 20 + 7 + 10 = **68 features**

---

## 6. Model Training Pipeline

### 6.1 End-to-End Workflow

```python
#!/usr/bin/env python3
"""
Meta-Model Training Pipeline

Trains overlap-as-feature meta-model to filter archetype signals.

Usage:
    python train_meta_model.py --data signals_2022_2024.parquet \
                                --model-type lightgbm \
                                --output models/meta_model.pkl
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_recall_curve
import optuna
import pickle
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Step 1: Data Preparation
# ============================================================================

def load_and_prepare_data(data_path: str, test_split_date: str = '2024-01-01'):
    """
    Load historical signals and prepare train/test split.

    Args:
        data_path: Path to signals parquet file
        test_split_date: Date to split train/test (walk-forward)

    Returns:
        (train_df, test_df, feature_cols, target_col)
    """
    logger.info(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)

    # Label signals (if not already done)
    if 'label' not in df.columns:
        logger.info("Computing labels (forward returns)...")
        df = compute_labels(df)

    # Remove neutrals
    df = df.dropna(subset=['label'])
    logger.info(f"After filtering neutrals: {len(df)} signals")

    # Extract features
    feature_cols = [c for c in df.columns if c not in
                   ['timestamp', 'label', 'entry_price', 'exit_price', 'return_72h']]
    target_col = 'label'

    # Train/test split (temporal)
    train_df = df[df['timestamp'] < test_split_date]
    test_df = df[df['timestamp'] >= test_split_date]

    logger.info(f"Train: {len(train_df)} samples ({train_df['timestamp'].min()} to {train_df['timestamp'].max()})")
    logger.info(f"Test: {len(test_df)} samples ({test_df['timestamp'].min()} to {test_df['timestamp'].max()})")
    logger.info(f"Features: {len(feature_cols)}")

    return train_df, test_df, feature_cols, target_col


def compute_labels(df: pd.DataFrame,
                   win_threshold: float = 2.0,
                   loss_threshold: float = -1.0) -> pd.DataFrame:
    """
    Compute WIN/LOSS labels from forward returns.

    Args:
        df: Signals DataFrame
        win_threshold: % return to classify as WIN
        loss_threshold: % return to classify as LOSS

    Returns:
        DataFrame with 'label' column
    """
    # Assume 'return_72h' column exists (computed during backtest)
    df['label'] = None

    df.loc[df['return_72h'] > win_threshold, 'label'] = 1  # WIN
    df.loc[df['return_72h'] < loss_threshold, 'label'] = 0  # LOSS

    win_count = (df['label'] == 1).sum()
    loss_count = (df['label'] == 0).sum()
    neutral_count = df['label'].isna().sum()

    logger.info(f"Labels: {win_count} wins, {loss_count} losses, {neutral_count} neutrals")
    logger.info(f"Win rate (pre-filter): {win_count / (win_count + loss_count):.2%}")

    return df


# ============================================================================
# Step 2: Hyperparameter Tuning with Optuna
# ============================================================================

def objective(trial, X_train, y_train, X_val, y_val):
    """
    Optuna objective function for LightGBM hyperparameter tuning.

    Args:
        trial: Optuna trial
        X_train, y_train: Training data
        X_val, y_val: Validation data

    Returns:
        Validation AUC (to maximize)
    """
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'seed': 42,

        # Tunable hyperparameters
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.5, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
        'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-3, 10.0),
        'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-3, 10.0),
    }

    # Train
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=50)]
    )

    # Evaluate
    y_pred = model.predict(X_val)
    auc = roc_auc_score(y_val, y_pred)

    return auc


def tune_hyperparameters(X_train, y_train, n_trials: int = 100):
    """
    Run Optuna hyperparameter search.

    Args:
        X_train, y_train: Training data
        n_trials: Number of Optuna trials

    Returns:
        Best hyperparameters dict
    """
    logger.info(f"Starting hyperparameter tuning ({n_trials} trials)...")

    # 5-fold cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def cv_objective(trial):
        """Cross-validated objective"""
        aucs = []
        for train_idx, val_idx in skf.split(X_train, y_train):
            X_tr = X_train.iloc[train_idx]
            y_tr = y_train.iloc[train_idx]
            X_val = X_train.iloc[val_idx]
            y_val = y_train.iloc[val_idx]

            auc = objective(trial, X_tr, y_tr, X_val, y_val)
            aucs.append(auc)

        return np.mean(aucs)

    # Run Optuna
    study = optuna.create_study(direction='maximize')
    study.optimize(cv_objective, n_trials=n_trials, show_progress_bar=True)

    logger.info(f"Best trial AUC: {study.best_value:.4f}")
    logger.info(f"Best parameters: {study.best_params}")

    return study.best_params


# ============================================================================
# Step 3: Train Final Model
# ============================================================================

def train_final_model(X_train, y_train, X_val, y_val, params: dict):
    """
    Train final LightGBM model with best hyperparameters.

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        params: Hyperparameters (from Optuna)

    Returns:
        Trained LightGBM model
    """
    logger.info("Training final model...")

    # Merge best params with defaults
    final_params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'seed': 42,
        **params
    }

    # Prepare data
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    # Train
    model = lgb.train(
        final_params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=50)
        ]
    )

    logger.info(f"Training complete. Best iteration: {model.best_iteration}")

    return model


# ============================================================================
# Step 4: Evaluation
# ============================================================================

def evaluate_model(model, X_test, y_test, threshold: float = 0.65):
    """
    Evaluate model on test set.

    Args:
        model: Trained LightGBM model
        X_test, y_test: Test data
        threshold: Probability threshold for TAKE decision

    Returns:
        Metrics dict
    """
    logger.info("Evaluating model on test set...")

    # Predict
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > threshold).astype(int)

    # AUC
    auc = roc_auc_score(y_test, y_pred_proba)

    # Precision, Recall, F1 at threshold
    from sklearn.metrics import precision_score, recall_score, f1_score

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Compute win rate of filtered signals (TAKE decisions)
    take_mask = y_pred == 1
    if take_mask.sum() > 0:
        filtered_win_rate = y_test[take_mask].mean()
    else:
        filtered_win_rate = 0.0

    metrics = {
        'auc': auc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp),
        'filtered_win_rate': filtered_win_rate,
        'num_filtered_signals': int(take_mask.sum()),
        'filter_rate': 1 - (take_mask.sum() / len(y_test))
    }

    logger.info(f"Test AUC: {auc:.4f}")
    logger.info(f"Precision@{threshold}: {precision:.4f}")
    logger.info(f"Recall@{threshold}: {recall:.4f}")
    logger.info(f"F1@{threshold}: {f1:.4f}")
    logger.info(f"Filtered Win Rate: {filtered_win_rate:.2%} (n={take_mask.sum()})")
    logger.info(f"Filter Rate: {metrics['filter_rate']:.2%} (rejected {int(len(y_test) - take_mask.sum())} signals)")

    return metrics


# ============================================================================
# Step 5: Feature Importance (SHAP)
# ============================================================================

def analyze_feature_importance(model, X_train, feature_names, output_dir: Path):
    """
    Analyze feature importance using SHAP values.

    Args:
        model: Trained LightGBM model
        X_train: Training data (subsample for speed)
        feature_names: List of feature names
        output_dir: Directory to save plots
    """
    logger.info("Computing SHAP values...")

    import shap

    # Subsample training data (SHAP is slow on large datasets)
    X_sample = X_train.sample(n=min(1000, len(X_train)), random_state=42)

    # Compute SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Summary plot (top 20 features)
    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=feature_names,
        max_display=20,
        show=False
    )
    plt.tight_layout()
    plt.savefig(output_dir / 'shap_summary.png', dpi=150)
    logger.info(f"SHAP summary plot saved to {output_dir / 'shap_summary.png'}")

    # Feature importance (mean |SHAP|)
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': np.abs(shap_values).mean(axis=0)
    }).sort_values('importance', ascending=False)

    feature_importance.to_csv(output_dir / 'feature_importance_shap.csv', index=False)
    logger.info(f"Feature importance saved to {output_dir / 'feature_importance_shap.csv'}")

    # Print top 20
    logger.info("\nTop 20 features by SHAP importance:")
    for idx, row in feature_importance.head(20).iterrows():
        logger.info(f"  {row['feature']:30s}: {row['importance']:.4f}")

    return feature_importance


# ============================================================================
# Main Execution
# ============================================================================

def main():
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description='Train Meta-Model')
    parser.add_argument('--data', required=True, help='Path to signals parquet')
    parser.add_argument('--model-type', default='lightgbm', choices=['lightgbm', 'xgboost'])
    parser.add_argument('--output', required=True, help='Output model path')
    parser.add_argument('--n-trials', type=int, default=100, help='Optuna trials')
    parser.add_argument('--threshold', type=float, default=0.65, help='Decision threshold')

    args = parser.parse_args()

    # Prepare output directory
    output_path = Path(args.output)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    train_df, test_df, feature_cols, target_col = load_and_prepare_data(args.data)

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    # Tune hyperparameters
    best_params = tune_hyperparameters(X_train, y_train, n_trials=args.n_trials)

    # Train final model
    # Use last 20% of training data as validation for early stopping
    split_idx = int(len(train_df) * 0.8)
    X_tr = X_train.iloc[:split_idx]
    y_tr = y_train.iloc[:split_idx]
    X_val = X_train.iloc[split_idx:]
    y_val = y_train.iloc[split_idx:]

    model = train_final_model(X_tr, y_tr, X_val, y_val, best_params)

    # Evaluate on test set
    metrics = evaluate_model(model, X_test, y_test, threshold=args.threshold)

    # SHAP analysis
    feature_importance = analyze_feature_importance(
        model, X_train, feature_cols, output_dir
    )

    # Save model
    model_artifact = {
        'model': model,
        'feature_cols': feature_cols,
        'best_params': best_params,
        'test_metrics': metrics,
        'feature_importance': feature_importance,
        'threshold': args.threshold,
        'version': '1.0'
    }

    with open(output_path, 'wb') as f:
        pickle.dump(model_artifact, f)

    logger.info(f"Model saved to {output_path}")
    logger.info("Training complete!")


if __name__ == '__main__':
    main()
```

### 6.2 Usage Example

```bash
# 1. Extract historical signals from backtests
python bin/extract_historical_signals.py \
    --backtest-logs logs/backtest_2022_2024/ \
    --output data/signals_historical.parquet

# 2. Train meta-model
python bin/train_meta_model.py \
    --data data/signals_historical.parquet \
    --model-type lightgbm \
    --output models/meta_model_v1.pkl \
    --n-trials 100 \
    --threshold 0.65

# 3. Evaluate on OOS data
python bin/evaluate_meta_model.py \
    --model models/meta_model_v1.pkl \
    --test-data data/signals_2024_q4.parquet \
    --output results/meta_model_evaluation.json
```

---

## 7. Integration with Existing System

### 7.1 System Architecture Update

**Before (Current)**:
```
Feature Store → Archetypes → Domain Engines → Trade Execution
                    ↓
              (16 signals)
```

**After (With Meta-Model)**:
```
Feature Store → Archetypes → Domain Engines → Meta-Model → Trade Execution
                    ↓             ↓               ↓
              (16 signals)  (boost/veto)    (filter/rank)
```

### 7.2 Meta-Model Integration Code

```python
# engine/models/meta_filter.py

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class MetaFilter:
    """
    Meta-model filter for archetype signals.

    Sits between archetype layer and trade execution.
    Filters out low-probability signals based on overlap patterns.
    """

    def __init__(self, model_path: str, threshold: float = 0.65):
        """
        Initialize meta-filter.

        Args:
            model_path: Path to trained meta-model pickle
            threshold: Probability threshold for TAKE decision
        """
        self.model_path = Path(model_path)
        self.threshold = threshold

        # Load model
        with open(self.model_path, 'rb') as f:
            artifact = pickle.load(f)

        self.model = artifact['model']
        self.feature_cols = artifact['feature_cols']
        self.version = artifact.get('version', '1.0')

        logger.info(f"[MetaFilter] Loaded model from {model_path}")
        logger.info(f"[MetaFilter] Version: {self.version}")
        logger.info(f"[MetaFilter] Threshold: {threshold}")
        logger.info(f"[MetaFilter] Features: {len(self.feature_cols)}")

    def filter_signal(
        self,
        archetype_signals: list,
        context: 'RuntimeContext'
    ) -> Tuple[bool, float, Dict]:
        """
        Filter archetype signal using meta-model.

        Args:
            archetype_signals: List of archetype signal dicts
                              Each dict: {'archetype': str, 'conf': float, 'direction': int, ...}
            context: RuntimeContext with regime, market features

        Returns:
            (should_take: bool, probability: float, metadata: dict)
        """
        # Extract meta-features
        features = self._extract_features(archetype_signals, context)

        # Prepare feature vector
        X = pd.DataFrame([features])[self.feature_cols]

        # Predict
        try:
            prob = self.model.predict(X)[0]  # P(WIN)
        except Exception as e:
            logger.error(f"[MetaFilter] Prediction failed: {e}")
            # Fallback: allow signal (fail-open)
            return True, 0.5, {'error': str(e)}

        # Decision
        should_take = prob > self.threshold

        # Metadata
        metadata = {
            'meta_prob': prob,
            'meta_threshold': self.threshold,
            'meta_decision': 'TAKE' if should_take else 'SKIP',
            'num_archetypes': len(archetype_signals),
            'regime': context.regime_label
        }

        return should_take, prob, metadata

    def _extract_features(
        self,
        archetype_signals: list,
        context: 'RuntimeContext'
    ) -> Dict[str, float]:
        """
        Extract meta-features from archetype signals.

        Args:
            archetype_signals: List of archetype signal dicts
            context: RuntimeContext

        Returns:
            Feature dict
        """
        features = {}

        # Individual archetype flags (16 binary)
        all_archetypes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'K', 'L', 'M',
                         'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8']

        fired_archetypes = {s['archetype'] for s in archetype_signals}

        for arch in all_archetypes:
            features[f'{arch}_fired'] = int(arch in fired_archetypes)

        # Aggregate statistics
        if archetype_signals:
            confs = [s['conf'] for s in archetype_signals]
            directions = [s['direction'] for s in archetype_signals]
            boosts = [s.get('boost', 1.0) for s in archetype_signals]
            fusion_scores = [s.get('fusion_score', 0.0) for s in archetype_signals]

            features['num_fired'] = len(archetype_signals)
            features['num_long'] = sum(d == 1 for d in directions)
            features['num_short'] = sum(d == -1 for d in directions)
            features['avg_conf'] = np.mean(confs)
            features['max_conf'] = np.max(confs)
            features['min_conf'] = np.min(confs)
            features['conf_std'] = np.std(confs) if len(confs) > 1 else 0.0
            features['all_agree_long'] = int(all(d == 1 for d in directions))
            features['all_agree_short'] = int(all(d == -1 for d in directions))
            features['mixed_signals'] = int(
                any(d == 1 for d in directions) and any(d == -1 for d in directions)
            )
            features['avg_boost'] = np.mean(boosts)
            features['max_boost'] = np.max(boosts)
            features['avg_fusion_score'] = np.mean(fusion_scores)
            features['max_fusion_score'] = np.max(fusion_scores)
            features['min_fusion_score'] = np.min(fusion_scores)
        else:
            # No signals (shouldn't happen, but handle gracefully)
            features.update({
                'num_fired': 0,
                'num_long': 0,
                'num_short': 0,
                'avg_conf': 0.0,
                'max_conf': 0.0,
                'min_conf': 0.0,
                'conf_std': 0.0,
                'all_agree_long': 0,
                'all_agree_short': 0,
                'mixed_signals': 0,
                'avg_boost': 1.0,
                'max_boost': 1.0,
                'avg_fusion_score': 0.0,
                'max_fusion_score': 0.0,
                'min_fusion_score': 0.0,
            })

        # Pairwise interactions (top 20 pairs)
        # TODO: Load from model artifact or config
        top_pairs = [
            ('A', 'C'), ('A', 'G'), ('C', 'G'), ('H', 'K'),
            # ... (load actual top pairs from training)
        ]

        for a1, a2 in top_pairs:
            features[f'{a1}_and_{a2}'] = int(
                features[f'{a1}_fired'] and features[f'{a2}_fired']
            )

        # Regime context
        regime = context.regime_label
        features['regime_risk_on'] = int(regime == 'risk_on')
        features['regime_neutral'] = int(regime == 'neutral')
        features['regime_risk_off'] = int(regime == 'risk_off')
        features['regime_crisis'] = int(regime == 'crisis')
        features['regime_confidence'] = context.regime_probs.get(regime, 0.0)

        # Market context (extract from context.row)
        row = context.row
        features['atr_percentile'] = row.get('atr_percentile', 0.5)
        features['adx_14'] = row.get('adx_14', 20.0)
        features['liquidity_score'] = row.get('liquidity_score', 0.5)
        features['volume_zscore'] = row.get('volume_zscore', 0.0)
        features['VIX_Z'] = row.get('VIX_Z', 0.0)
        features['DXY_Z'] = row.get('DXY_Z', 0.0)
        features['funding_Z'] = row.get('funding_Z', 0.0)

        return features
```

### 7.3 Integration into Execution Pipeline

```python
# engine/archetypes/logic_v2_adapter.py (modified)

class ArchetypeLogic:
    """
    Rule-based archetype detection with meta-model filtering.
    """

    def __init__(self, config: dict):
        # ... existing init code ...

        # Initialize meta-filter if enabled
        self.use_meta_filter = config.get('use_meta_filter', False)
        self.meta_filter = None

        if self.use_meta_filter:
            model_path = config.get('meta_filter_path', 'models/meta_model_v1.pkl')
            threshold = config.get('meta_filter_threshold', 0.65)

            try:
                from engine.models.meta_filter import MetaFilter
                self.meta_filter = MetaFilter(model_path, threshold)
                logger.info("[ArchetypeLogic] Meta-filter ENABLED")
                logger.info(f"[ArchetypeLogic]   - Model: {model_path}")
                logger.info(f"[ArchetypeLogic]   - Threshold: {threshold}")
            except Exception as e:
                logger.error(f"[ArchetypeLogic] Failed to load meta-filter: {e}")
                self.use_meta_filter = False
        else:
            logger.info("[ArchetypeLogic] Meta-filter DISABLED")

    def evaluate_all_archetypes(self, context: RuntimeContext) -> list:
        """
        Evaluate all archetypes and apply meta-filter if enabled.

        Args:
            context: RuntimeContext

        Returns:
            List of filtered archetype signals
        """
        # Step 1: Evaluate all archetypes (existing logic)
        raw_signals = []

        for archetype_id in ['A', 'B', 'C', ..., 'S8']:
            if not self.enabled[archetype_id]:
                continue

            # Evaluate archetype
            signal = self._evaluate_archetype(archetype_id, context)

            if signal is not None:
                raw_signals.append(signal)

        logger.info(f"[ArchetypeLogic] Raw signals: {len(raw_signals)} archetypes fired")

        # Step 2: Apply meta-filter (if enabled)
        if self.use_meta_filter and raw_signals:
            should_take, prob, metadata = self.meta_filter.filter_signal(
                raw_signals, context
            )

            logger.info(f"[MetaFilter] P(WIN)={prob:.3f}, Decision={metadata['meta_decision']}")

            if should_take:
                # Meta-filter says TAKE → return signals
                for signal in raw_signals:
                    signal['meta_prob'] = prob
                    signal['meta_filtered'] = True

                return raw_signals
            else:
                # Meta-filter says SKIP → return empty
                logger.info(f"[MetaFilter] Filtered out {len(raw_signals)} signals (low confidence)")
                return []
        else:
            # Meta-filter disabled or no signals → return raw signals
            return raw_signals
```

### 7.4 Configuration Example

```json
// configs/production_with_meta_filter.json

{
  "use_archetypes": true,
  "use_meta_filter": true,
  "meta_filter_path": "models/meta_model_v1.pkl",
  "meta_filter_threshold": 0.65,

  "enable_A": true,
  "enable_B": true,
  "enable_C": true,
  "enable_G": true,
  "enable_H": true,
  "enable_K": true,
  "enable_S1": true,
  "enable_S4": true,
  "enable_S5": true,

  "regime_aware": true,
  "use_meta_fusion": true,
  "temporal_fusion": {
    "enabled": true
  }
}
```

---

## 8. Success Metrics & Acceptance Criteria

### 8.1 Classification Metrics

```python
# Minimum acceptance thresholds for meta-model deployment

ACCEPTANCE_CRITERIA = {
    # Classification metrics
    'oos_auc': 0.65,           # Out-of-sample AUC > 65% (better than random)
    'precision_at_065': 0.70,  # Precision @ 0.65 threshold > 70%
    'recall_at_065': 0.60,     # Recall @ 0.65 threshold > 60%

    # Trading metrics (vs raw archetypes baseline)
    'sharpe_improvement': 0.3,  # Sharpe ratio gain > +0.3
    'win_rate_improvement': 0.05,  # Win rate gain > +5% absolute
    'max_drawdown_reduction': 0.05,  # Max DD reduction > 5% absolute

    # Operational metrics
    'filter_rate_min': 0.20,   # Filter out at least 20% of signals
    'filter_rate_max': 0.60,   # But not more than 60% (over-filtering)

    # Stability metrics
    'oos_auc_std': 0.10,       # OOS AUC std dev across regimes < 0.10
}
```

### 8.2 Evaluation Framework

```python
#!/usr/bin/env python3
"""
Meta-Model Acceptance Testing

Compares meta-model performance against baselines.

Usage:
    python bin/test_meta_model_acceptance.py \
        --meta-model models/meta_model_v1.pkl \
        --test-data data/signals_2024_q4.parquet \
        --baseline-results results/baseline_backtest.json \
        --output results/acceptance_report.json
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def evaluate_classification(y_true, y_pred_proba, threshold=0.65):
    """Evaluate classification metrics."""
    from sklearn.metrics import (
        roc_auc_score, precision_score, recall_score, f1_score,
        precision_recall_curve
    )

    y_pred = (y_pred_proba > threshold).astype(int)

    metrics = {
        'auc': roc_auc_score(y_true, y_pred_proba),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
    }

    # Precision-recall curve
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)

    # Find optimal threshold (maximize F1)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    optimal_idx = np.argmax(f1_scores)

    metrics['optimal_threshold'] = float(thresholds[optimal_idx])
    metrics['optimal_f1'] = float(f1_scores[optimal_idx])

    return metrics


def evaluate_trading_performance(signals_df, meta_filtered_df):
    """
    Compare trading performance: raw vs meta-filtered.

    Args:
        signals_df: All signals (raw archetypes)
        meta_filtered_df: Signals that passed meta-filter

    Returns:
        Comparison metrics dict
    """
    # Raw archetype metrics
    raw_win_rate = (signals_df['label'] == 1).mean()
    raw_sharpe = compute_sharpe(signals_df['return_72h'])
    raw_max_dd = compute_max_drawdown(signals_df['return_72h'])

    # Meta-filtered metrics
    if len(meta_filtered_df) > 0:
        filtered_win_rate = (meta_filtered_df['label'] == 1).mean()
        filtered_sharpe = compute_sharpe(meta_filtered_df['return_72h'])
        filtered_max_dd = compute_max_drawdown(meta_filtered_df['return_72h'])
    else:
        filtered_win_rate = 0.0
        filtered_sharpe = 0.0
        filtered_max_dd = 0.0

    # Compute improvements
    comparison = {
        'raw': {
            'num_signals': len(signals_df),
            'win_rate': raw_win_rate,
            'sharpe': raw_sharpe,
            'max_drawdown': raw_max_dd,
        },
        'filtered': {
            'num_signals': len(meta_filtered_df),
            'win_rate': filtered_win_rate,
            'sharpe': filtered_sharpe,
            'max_drawdown': filtered_max_dd,
        },
        'improvement': {
            'win_rate_delta': filtered_win_rate - raw_win_rate,
            'sharpe_delta': filtered_sharpe - raw_sharpe,
            'max_dd_delta': filtered_max_dd - raw_max_dd,
            'filter_rate': 1 - (len(meta_filtered_df) / len(signals_df)),
        }
    }

    return comparison


def compute_sharpe(returns: pd.Series, periods_per_year: int = 365*24):
    """Compute annualized Sharpe ratio."""
    if len(returns) == 0 or returns.std() == 0:
        return 0.0

    sharpe = returns.mean() / returns.std() * np.sqrt(periods_per_year)
    return sharpe


def compute_max_drawdown(returns: pd.Series):
    """Compute maximum drawdown."""
    cum_returns = (1 + returns / 100).cumprod()
    running_max = cum_returns.expanding().max()
    drawdown = (cum_returns - running_max) / running_max
    max_dd = drawdown.min()
    return max_dd


def check_acceptance_criteria(metrics: dict, criteria: dict):
    """
    Check if meta-model meets acceptance criteria.

    Args:
        metrics: Computed metrics dict
        criteria: Acceptance criteria dict

    Returns:
        (passed: bool, failures: list)
    """
    failures = []

    # Classification metrics
    if metrics['classification']['auc'] < criteria['oos_auc']:
        failures.append(
            f"AUC {metrics['classification']['auc']:.3f} < {criteria['oos_auc']:.3f}"
        )

    if metrics['classification']['precision'] < criteria['precision_at_065']:
        failures.append(
            f"Precision {metrics['classification']['precision']:.3f} < {criteria['precision_at_065']:.3f}"
        )

    # Trading metrics
    if metrics['trading']['improvement']['sharpe_delta'] < criteria['sharpe_improvement']:
        failures.append(
            f"Sharpe improvement {metrics['trading']['improvement']['sharpe_delta']:.3f} < {criteria['sharpe_improvement']:.3f}"
        )

    if metrics['trading']['improvement']['win_rate_delta'] < criteria['win_rate_improvement']:
        failures.append(
            f"Win rate improvement {metrics['trading']['improvement']['win_rate_delta']:.3f} < {criteria['win_rate_improvement']:.3f}"
        )

    # Filter rate bounds
    filter_rate = metrics['trading']['improvement']['filter_rate']
    if not (criteria['filter_rate_min'] <= filter_rate <= criteria['filter_rate_max']):
        failures.append(
            f"Filter rate {filter_rate:.2%} outside [{criteria['filter_rate_min']:.2%}, {criteria['filter_rate_max']:.2%}]"
        )

    passed = len(failures) == 0

    return passed, failures


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Meta-Model Acceptance Testing')
    parser.add_argument('--meta-model', required=True, help='Path to meta-model')
    parser.add_argument('--test-data', required=True, help='Path to test signals')
    parser.add_argument('--output', required=True, help='Output report path')
    parser.add_argument('--threshold', type=float, default=0.65, help='Decision threshold')

    args = parser.parse_args()

    # Load meta-model
    from engine.models.meta_filter import MetaFilter
    meta_filter = MetaFilter(args.meta_model, threshold=args.threshold)

    # Load test data
    test_df = pd.read_parquet(args.test_data)
    logger.info(f"Loaded {len(test_df)} test signals")

    # Apply meta-filter
    # (Assume test_df has feature columns needed by meta-filter)
    X_test = test_df[meta_filter.feature_cols]
    y_test = test_df['label']

    y_pred_proba = meta_filter.model.predict(X_test)

    # Classification metrics
    classification_metrics = evaluate_classification(
        y_test, y_pred_proba, threshold=args.threshold
    )

    # Trading metrics
    meta_filtered_df = test_df[y_pred_proba > args.threshold]
    trading_metrics = evaluate_trading_performance(test_df, meta_filtered_df)

    # Combined metrics
    metrics = {
        'classification': classification_metrics,
        'trading': trading_metrics,
    }

    # Check acceptance
    passed, failures = check_acceptance_criteria(metrics, ACCEPTANCE_CRITERIA)

    # Build report
    report = {
        'meta_model_path': args.meta_model,
        'test_data_path': args.test_data,
        'threshold': args.threshold,
        'metrics': metrics,
        'acceptance_criteria': ACCEPTANCE_CRITERIA,
        'passed': passed,
        'failures': failures,
        'recommendation': 'DEPLOY' if passed else 'REJECT',
    }

    # Save report
    with open(args.output, 'w') as f:
        json.dump(report, f, indent=2)

    logger.info(f"Acceptance report saved to {args.output}")

    # Print summary
    print("\n" + "="*70)
    print("META-MODEL ACCEPTANCE TEST RESULTS")
    print("="*70)
    print(f"\nClassification Metrics:")
    print(f"  AUC: {classification_metrics['auc']:.4f}")
    print(f"  Precision @ {args.threshold}: {classification_metrics['precision']:.4f}")
    print(f"  Recall @ {args.threshold}: {classification_metrics['recall']:.4f}")
    print(f"  F1 @ {args.threshold}: {classification_metrics['f1']:.4f}")

    print(f"\nTrading Metrics:")
    print(f"  Raw Win Rate: {trading_metrics['raw']['win_rate']:.2%}")
    print(f"  Filtered Win Rate: {trading_metrics['filtered']['win_rate']:.2%}")
    print(f"  Improvement: {trading_metrics['improvement']['win_rate_delta']:+.2%}")
    print(f"\n  Raw Sharpe: {trading_metrics['raw']['sharpe']:.3f}")
    print(f"  Filtered Sharpe: {trading_metrics['filtered']['sharpe']:.3f}")
    print(f"  Improvement: {trading_metrics['improvement']['sharpe_delta']:+.3f}")
    print(f"\n  Filter Rate: {trading_metrics['improvement']['filter_rate']:.2%}")

    print(f"\nAcceptance: {'✅ PASSED' if passed else '❌ FAILED'}")

    if not passed:
        print("\nFailures:")
        for failure in failures:
            print(f"  - {failure}")

    print(f"\nRecommendation: {report['recommendation']}")
    print("="*70)


if __name__ == '__main__':
    main()
```

---

## 9. Overlap Analysis Framework

### 9.1 Research Questions

**Question 1: Which overlap patterns win?**

```python
# Analyze win rate by overlap pattern

from collections import Counter

# Group signals by overlap pattern
signals_df['pattern'] = signals_df.apply(
    lambda row: ','.join(sorted([
        arch for arch in archetypes if row[f'{arch}_fired'] == 1
    ])),
    axis=1
)

# Compute win rate by pattern
pattern_analysis = signals_df.groupby('pattern').agg({
    'label': ['mean', 'count']
}).reset_index()

pattern_analysis.columns = ['pattern', 'win_rate', 'count']
pattern_analysis = pattern_analysis[pattern_analysis['count'] >= 10]  # Min 10 occurrences
pattern_analysis = pattern_analysis.sort_values('win_rate', ascending=False)

print("Top 20 overlap patterns by win rate:")
print(pattern_analysis.head(20))

# Examples:
# pattern        win_rate  count
# A,C            0.68      87      ← High confluence (2 archetypes)
# S1             0.62      145     ← Solo S1 works well
# A,C,G          0.48      34      ← Over-confirmation (3 archetypes)
# H,K            0.71      52      ← Momentum + trap combo
# S1,S4          0.45      28      ← Bear combos underperform
```

**Question 2: Does confluence always help?**

```python
# Analyze win rate vs number of overlapping archetypes

overlap_count_analysis = signals_df.groupby('num_fired').agg({
    'label': ['mean', 'count']
}).reset_index()

overlap_count_analysis.columns = ['num_archetypes', 'win_rate', 'count']

print("Win rate by number of overlapping archetypes:")
print(overlap_count_analysis)

# Expected (inverted U-shape):
# num_archetypes  win_rate  count
# 1               0.55      1200    ← Solo signals (baseline)
# 2               0.64      850     ← Optimal confluence
# 3               0.58      340     ← Some overlap helps
# 4+              0.48      110     ← Over-confirmation hurts

# Insight: 2-archetype confluence is optimal sweet spot
```

**Question 3: Regime-specific overlap rules?**

```python
# Analyze overlap patterns by regime

regime_pattern_analysis = signals_df.groupby(['regime_label', 'pattern']).agg({
    'label': ['mean', 'count']
}).reset_index()

regime_pattern_analysis.columns = ['regime', 'pattern', 'win_rate', 'count']
regime_pattern_analysis = regime_pattern_analysis[regime_pattern_analysis['count'] >= 5]

# Filter top patterns per regime
for regime in ['risk_on', 'neutral', 'risk_off', 'crisis']:
    print(f"\n{regime.upper()} - Top 10 patterns:")
    subset = regime_pattern_analysis[regime_pattern_analysis['regime'] == regime]
    subset = subset.sort_values('win_rate', ascending=False)
    print(subset.head(10))

# Expected insights:
# - risk_on: A+C+G works (trend-following)
# - risk_off: S1 solo works (don't wait for confluence)
# - crisis: S5 solo works (quick reversals)
```

**Question 4: Direction agreement matters?**

```python
# Analyze win rate when signals agree vs disagree

direction_analysis = signals_df.groupby('mixed_signals').agg({
    'label': ['mean', 'count']
}).reset_index()

direction_analysis.columns = ['mixed_signals', 'win_rate', 'count']

print("Win rate by direction agreement:")
print(direction_analysis)

# Expected:
# mixed_signals  win_rate  count
# 0 (agree)      0.58      2100    ← Signals agree on direction
# 1 (mixed)      0.35      200     ← LONG + SHORT mixed → low win rate

# Action: Require direction agreement (filter mixed signals)
```

### 9.2 Overlap Visualization

```python
# Visualize overlap patterns as heatmap

import matplotlib.pyplot as plt
import seaborn as sns

# Compute pairwise overlap win rates
from itertools import combinations

pair_win_rates = []

for a1, a2 in combinations(archetypes, 2):
    pair_mask = (
        (signals_df[f'{a1}_fired'] == 1) &
        (signals_df[f'{a2}_fired'] == 1)
    )

    if pair_mask.sum() >= 10:  # Min 10 occurrences
        win_rate = signals_df[pair_mask]['label'].mean()
        count = pair_mask.sum()

        pair_win_rates.append({
            'arch1': a1,
            'arch2': a2,
            'win_rate': win_rate,
            'count': count
        })

pair_df = pd.DataFrame(pair_win_rates)

# Pivot to heatmap format
heatmap_data = pair_df.pivot(index='arch1', columns='arch2', values='win_rate')

# Plot
plt.figure(figsize=(14, 12))
sns.heatmap(
    heatmap_data,
    annot=True,
    fmt='.2f',
    cmap='RdYlGn',
    vmin=0.4,
    vmax=0.7,
    cbar_kws={'label': 'Win Rate'}
)
plt.title('Archetype Pairwise Overlap Win Rates', fontsize=16)
plt.tight_layout()
plt.savefig('overlap_heatmap.png', dpi=150)

print("Overlap heatmap saved to overlap_heatmap.png")
```

---

## 10. Production Deployment Roadmap

### 10.1 Phased Rollout

**Phase 1: Research & Validation** (Weeks 1-3)
- [x] Define architecture
- [ ] Extract historical signals (2022-2024)
- [ ] Label dataset (WIN/LOSS)
- [ ] Train baseline models (Logistic, Random Forest, LightGBM)
- [ ] Walk-forward validation
- [ ] Feature importance analysis (SHAP)
- [ ] Overlap pattern analysis (EDA)
- **Milestone**: OOS AUC > 0.65, Sharpe +0.3

**Phase 2: Integration** (Week 4)
- [ ] Build MetaFilter class
- [ ] Integrate into ArchetypeLogic
- [ ] Unit tests + integration tests
- [ ] Config management (enable/disable meta-filter)
- **Milestone**: Meta-filter runs in backtest mode

**Phase 3: Paper Trading** (Weeks 5-8)
- [ ] Deploy to paper trading environment
- [ ] A/B test: 50% with meta-filter, 50% without
- [ ] Monitor metrics (Sharpe, win rate, filter rate)
- [ ] Compare results weekly
- **Milestone**: Meta-filter improves paper trading Sharpe by +0.3

**Phase 4: Production Deployment** (Week 9+)
- [ ] Deploy to live trading (10% of capital)
- [ ] Monitor real-time performance
- [ ] Feature drift detection (weekly)
- [ ] Auto-retrain schedule (monthly)
- [ ] Rollback plan if performance degrades
- **Milestone**: Live trading Sharpe improvement sustained

### 10.2 Monitoring & Maintenance

**Real-Time Monitoring Dashboard**:

```python
# Monitor meta-filter performance in production

import pandas as pd
from datetime import datetime, timedelta

def monitor_meta_filter_health():
    """
    Daily health check for meta-filter.

    Alerts:
    - Feature drift detected (KL divergence > 0.3)
    - Win rate degradation (< baseline)
    - Filter rate anomaly (outside [20%, 60%])
    """
    # Load last 7 days of signals
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)

    signals_df = load_signals(start_date, end_date)

    # Compute metrics
    win_rate = (signals_df['label'] == 1).mean()
    filter_rate = signals_df['meta_filtered'].mean()
    avg_prob = signals_df['meta_prob'].mean()

    # Check for anomalies
    alerts = []

    if win_rate < 0.55:  # Below baseline
        alerts.append(f"⚠️ Win rate degradation: {win_rate:.2%} < 55%")

    if not (0.20 <= filter_rate <= 0.60):
        alerts.append(f"⚠️ Filter rate anomaly: {filter_rate:.2%} outside [20%, 60%]")

    # Feature drift check (compare to training distribution)
    drift_detected = check_feature_drift(signals_df)
    if drift_detected:
        alerts.append(f"⚠️ Feature drift detected - consider retraining")

    # Send alerts
    if alerts:
        send_slack_alert('\n'.join(alerts))

    # Log health metrics
    logger.info(f"[MetaFilter Health] Win rate: {win_rate:.2%}, Filter rate: {filter_rate:.2%}, Avg prob: {avg_prob:.3f}")


def check_feature_drift(recent_df: pd.DataFrame, threshold: float = 0.3):
    """
    Detect feature drift using KL divergence.

    Args:
        recent_df: Recent signals DataFrame
        threshold: KL divergence threshold for alert

    Returns:
        True if drift detected
    """
    from scipy.stats import entropy

    # Load training distribution
    train_dist = load_training_distribution()

    # Compute KL divergence for each feature
    drift_scores = {}

    for feature in train_dist.keys():
        # Bin continuous features
        train_hist, bins = np.histogram(train_dist[feature], bins=20, density=True)
        recent_hist, _ = np.histogram(recent_df[feature], bins=bins, density=True)

        # Add small epsilon to avoid log(0)
        train_hist += 1e-10
        recent_hist += 1e-10

        # KL divergence
        kl_div = entropy(recent_hist, train_hist)
        drift_scores[feature] = kl_div

    # Check if any feature exceeds threshold
    max_drift = max(drift_scores.values())
    drift_detected = max_drift > threshold

    if drift_detected:
        logger.warning(f"[MetaFilter Drift] Max KL divergence: {max_drift:.3f}")
        logger.warning(f"[MetaFilter Drift] Top drifting features: {sorted(drift_scores.items(), key=lambda x: -x[1])[:5]}")

    return drift_detected
```

**Auto-Retraining Schedule**:

```bash
#!/bin/bash
# cron job: Run monthly retraining

# Every 1st of month at 2 AM
0 2 1 * * /usr/bin/python bin/retrain_meta_model.py \
    --data data/signals_latest.parquet \
    --output models/meta_model_$(date +\%Y\%m).pkl \
    --n-trials 100 \
    --notify-slack
```

### 10.3 Rollback Plan

**If meta-filter underperforms**:

1. **Immediate Action** (< 1 hour):
   - Disable meta-filter via config (set `use_meta_filter: false`)
   - Restart trading engine (falls back to raw archetypes)
   - Alert team via Slack

2. **Root Cause Analysis** (24 hours):
   - Analyze last 7 days of signals
   - Check for feature drift
   - Compare to baseline performance
   - Identify failure mode (overfitting? regime shift?)

3. **Fix or Retrain** (1 week):
   - If feature drift: Retrain on recent data
   - If regime shift: Add new regime-specific model
   - If overfitting: Simplify model or increase regularization

4. **Re-deploy** (after validation):
   - Run walk-forward validation on OOS data
   - Paper trade for 2 weeks
   - Re-enable if performance restored

---

## 11. Code Examples

### 11.1 Feature Extraction Script

```python
#!/usr/bin/env python3
"""
Extract historical archetype signals for meta-model training.

Usage:
    python bin/extract_historical_signals.py \
        --backtest-logs logs/backtest_2022_2024/ \
        --price-data data/btc_1h_2022_2024.parquet \
        --output data/signals_historical.parquet
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_backtest_logs(log_dir: Path):
    """
    Load all archetype signals from backtest logs.

    Args:
        log_dir: Directory with backtest log files

    Returns:
        DataFrame with all signals
    """
    all_signals = []

    for log_file in log_dir.glob('*.json'):
        logger.info(f"Loading {log_file.name}...")

        with open(log_file, 'r') as f:
            log_data = json.load(f)

        # Extract signals
        signals = log_data.get('signals', [])
        all_signals.extend(signals)

    df = pd.DataFrame(all_signals)
    logger.info(f"Loaded {len(df)} signals from {len(list(log_dir.glob('*.json')))} log files")

    return df


def compute_forward_returns(signals_df: pd.DataFrame, prices_df: pd.DataFrame):
    """
    Compute forward returns for each signal.

    Args:
        signals_df: Signals DataFrame
        prices_df: Price DataFrame (timestamp, close)

    Returns:
        Signals DataFrame with 'return_72h' column
    """
    logger.info("Computing forward returns...")

    # Merge with price data
    signals_df['timestamp'] = pd.to_datetime(signals_df['timestamp'])
    prices_df['timestamp'] = pd.to_datetime(prices_df['timestamp'])

    # For each signal, find price 72h later
    returns = []

    for idx, signal in signals_df.iterrows():
        entry_time = signal['timestamp']
        exit_time = entry_time + pd.Timedelta(hours=72)

        # Find entry price
        entry_price = signal.get('entry_price')
        if entry_price is None:
            # Use close price at entry time
            entry_row = prices_df[prices_df['timestamp'] == entry_time]
            if len(entry_row) > 0:
                entry_price = entry_row.iloc[0]['close']
            else:
                returns.append(None)
                continue

        # Find exit price (72h later)
        exit_row = prices_df[prices_df['timestamp'] == exit_time]
        if len(exit_row) > 0:
            exit_price = exit_row.iloc[0]['close']
        else:
            # Use nearest available price
            exit_row = prices_df[prices_df['timestamp'] >= exit_time].head(1)
            if len(exit_row) > 0:
                exit_price = exit_row.iloc[0]['close']
            else:
                returns.append(None)
                continue

        # Compute return (%)
        ret = (exit_price - entry_price) / entry_price * 100

        # Adjust sign based on direction
        direction = signal.get('direction', 1)  # 1=LONG, -1=SHORT
        ret *= direction

        returns.append(ret)

    signals_df['return_72h'] = returns

    # Drop rows with missing returns
    n_missing = signals_df['return_72h'].isna().sum()
    if n_missing > 0:
        logger.warning(f"Dropping {n_missing} signals with missing forward returns")
        signals_df = signals_df.dropna(subset=['return_72h'])

    logger.info(f"Computed returns for {len(signals_df)} signals")
    logger.info(f"Mean return: {signals_df['return_72h'].mean():.2f}%")
    logger.info(f"Median return: {signals_df['return_72h'].median():.2f}%")

    return signals_df


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Extract historical signals')
    parser.add_argument('--backtest-logs', required=True, help='Backtest log directory')
    parser.add_argument('--price-data', required=True, help='Price data parquet file')
    parser.add_argument('--output', required=True, help='Output signals parquet file')

    args = parser.parse_args()

    # Load backtest logs
    log_dir = Path(args.backtest_logs)
    signals_df = load_backtest_logs(log_dir)

    # Load price data
    prices_df = pd.read_parquet(args.price_data)
    logger.info(f"Loaded price data: {len(prices_df)} bars")

    # Compute forward returns
    signals_df = compute_forward_returns(signals_df, prices_df)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    signals_df.to_parquet(output_path, index=False)
    logger.info(f"Saved {len(signals_df)} signals to {output_path}")


if __name__ == '__main__':
    main()
```

---

## 12. Summary & Next Steps

### 12.1 Key Takeaways

1. **Paradigm Shift**: Treat overlap as a feature, not a bug
   - Capture confluence (A+C → 68% win rate)
   - Detect harmful overcrowding (A+C+G → 48% win rate)
   - Learn regime-specific patterns

2. **Meta-Model Architecture**: LightGBM classifier
   - Input: 70 features (archetype flags + aggregates + interactions + regime + market)
   - Output: P(WIN) in [0, 1]
   - Decision: TAKE if P(WIN) > 0.65

3. **Training Strategy**: Walk-forward validation
   - 40,000 labeled signals (2022-2024)
   - 80/20 train/val split (temporal)
   - Stratified by regime and label

4. **Integration**: Meta-filter layer
   - Sits between archetypes and execution
   - Filters out low-probability signals
   - Fallback to raw archetypes if meta-filter fails

5. **Success Criteria**:
   - OOS AUC > 0.65
   - Sharpe improvement > +0.3
   - Win rate improvement > +5%

### 12.2 Implementation Checklist

- [ ] **Week 1-2: Data Preparation**
  - [ ] Extract historical signals from backtest logs (2022-2024)
  - [ ] Compute forward returns (72h horizon)
  - [ ] Label dataset (WIN/LOSS/NEUTRAL)
  - [ ] EDA: Overlap pattern analysis
  - [ ] Feature engineering (overlap aggregates, pairwise interactions)

- [ ] **Week 3: Model Training**
  - [ ] Train Logistic Regression baseline
  - [ ] Train Random Forest strong baseline
  - [ ] Train LightGBM production model
  - [ ] Hyperparameter tuning (Optuna, 100 trials)
  - [ ] Walk-forward validation (6-month windows)
  - [ ] SHAP feature importance analysis

- [ ] **Week 4: Integration**
  - [ ] Build MetaFilter class
  - [ ] Integrate into ArchetypeLogic
  - [ ] Unit tests (test_meta_filter.py)
  - [ ] Integration tests (test_end_to_end_with_meta_filter.py)
  - [ ] Config management (enable/disable flag)

- [ ] **Week 5-8: Paper Trading**
  - [ ] Deploy to paper trading environment
  - [ ] A/B test (50% with meta-filter, 50% without)
  - [ ] Monitor daily metrics (Sharpe, win rate, filter rate)
  - [ ] Compare results weekly
  - [ ] Acceptance testing (check criteria)

- [ ] **Week 9+: Production Deployment**
  - [ ] Deploy to live trading (10% capital)
  - [ ] Real-time monitoring dashboard
  - [ ] Feature drift detection (weekly)
  - [ ] Auto-retrain schedule (monthly cron job)
  - [ ] Rollback plan (if performance degrades)

### 12.3 Files to Create

```
Bull-machine-/
├── docs/
│   └── META_MODEL_ARCHITECTURE_OVERLAP_AS_FEATURE.md  ← This file
├── engine/
│   └── models/
│       ├── meta_filter.py  ← Meta-filter integration class
│       └── meta_ensemble.py  ← Existing (already present)
├── bin/
│   ├── extract_historical_signals.py  ← Extract signals from backtests
│   ├── train_meta_model.py  ← Training pipeline
│   ├── evaluate_meta_model.py  ← Evaluation script
│   ├── test_meta_model_acceptance.py  ← Acceptance testing
│   └── monitor_meta_filter_health.py  ← Production monitoring
├── tests/
│   └── unit/
│       └── models/
│           └── test_meta_filter.py  ← Unit tests
├── configs/
│   └── production_with_meta_filter.json  ← Config example
└── models/
    └── meta_model_v1.pkl  ← Trained model artifact
```

---

## Appendix A: Mathematical Formulation

### A.1 Meta-Model Objective

**Binary Classification Problem**:

Minimize binary cross-entropy loss:

$$
\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]
$$

Where:
- $y_i \in \{0, 1\}$: Ground truth label (1=WIN, 0=LOSS)
- $\hat{y}_i \in [0, 1]$: Predicted probability (meta-model output)
- $N$: Number of training samples

**Regularization** (L1 + L2):

$$
\mathcal{L}_{total} = \mathcal{L} + \lambda_1 \|\mathbf{w}\|_1 + \lambda_2 \|\mathbf{w}\|_2^2
$$

Where:
- $\mathbf{w}$: Model weights
- $\lambda_1, \lambda_2$: Regularization hyperparameters (tuned via Optuna)

### A.2 Feature Importance (SHAP)

**SHAP Value** for feature $j$ and sample $i$:

$$
\phi_j(i) = \sum_{S \subseteq F \setminus \{j\}} \frac{|S|! (|F| - |S| - 1)!}{|F|!} \left[ f(S \cup \{j\}) - f(S) \right]
$$

Where:
- $F$: Full feature set
- $S$: Subset of features excluding $j$
- $f(S)$: Model prediction using feature subset $S$
- $\phi_j(i)$: Marginal contribution of feature $j$ to prediction

**Interpretation**: Positive SHAP value → feature pushes prediction toward WIN; negative → toward LOSS.

---

## Appendix B: Glossary

- **Archetype**: Pattern detector (e.g., A=Spring, S1=Liquidity Vacuum)
- **Confluence**: Multiple archetypes agreeing (e.g., A+C both LONG)
- **Meta-Model**: ML model that learns from overlap patterns
- **SHAP**: SHapley Additive exPlanations (interpretability method)
- **Walk-Forward Validation**: Train on historical window, test on future period
- **OOS**: Out-of-sample (test data not seen during training)
- **AUC**: Area Under ROC Curve (classification metric)
- **Sharpe Ratio**: Risk-adjusted return (higher is better)

---

## Appendix C: References

1. **Ensemble Learning**:
   - Dietterich, T. G. (2000). "Ensemble methods in machine learning." MCS 2000.
   - Breiman, L. (2001). "Random forests." Machine Learning.

2. **Feature Interactions**:
   - Friedman, J. H., & Popescu, B. E. (2008). "Predictive learning via rule ensembles." Ann. Appl. Stat.

3. **SHAP Values**:
   - Lundberg, S. M., & Lee, S. I. (2017). "A unified approach to interpreting model predictions." NeurIPS.

4. **Walk-Forward Validation**:
   - Pardo, R. (2008). "The Evaluation and Optimization of Trading Strategies." Wiley.

---

**END OF DOCUMENT**
