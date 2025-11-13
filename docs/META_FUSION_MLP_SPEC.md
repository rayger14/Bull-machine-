# Meta-Fusion MLP Architecture Specification
## PyTorch Integration for Router v10

**Version**: 1.0
**Date**: 2025-11-05
**Status**: Design Phase

---

## Philosophy: Augment, Don't Replace

The meta-fusion learner is **NOT** a replacement for rule-based fusion logic. Instead, it acts as a **quality multiplier** that scales the fusion score based on learned patterns.

### Key Principles:
1. **Rule-based logic produces candidates** → MLP scores quality → Final fusion = rule_fusion × quality_mult
2. **Vetoes are sacred** → MLP never overrides hard vetoes (event suppression, regime confidence floors, etc.)
3. **Confidence gating** → If MLP is uncertain (high entropy/variance), fall back to pure rules
4. **Interpretability** → Log feature importances, attention weights, and quality scores for every decision

---

## Architecture Overview

### Input Layer (Feature Vector)

```python
class FusionFeatureExtractor:
    """
    Extracts 40-50 features from current bar + context for MLP input.
    """

    def extract_features(self, bar, module_scores, context, archetype_id):
        """
        Returns feature vector of shape (n_features,) where n_features ~= 45-50
        """
        features = []

        # 1. Module Scores (8 features)
        features.extend([
            module_scores['wyckoff'],        # Wyckoff phase alignment
            module_scores['liquidity'],      # Liquidity grab quality
            module_scores['bojan'],          # Wick/trap quality
            module_scores['smc'],            # SMC structure score
            module_scores['macro'],          # Macro regime score
            module_scores['temporal'],       # Session/time alignment
            module_scores['psych_pti'],      # PTI psychology score
            module_scores['momentum']        # Momentum/trend strength
        ])

        # 2. MTF Agreement (6 features)
        features.extend([
            context['mtf_1h_agree'],         # 1H timeframe agrees (0/1)
            context['mtf_4h_agree'],         # 4H agrees
            context['mtf_1d_agree'],         # 1D agrees
            context['mtf_alignment_score'],  # Weighted MTF score (0-1)
            context['mtf_dispersion'],       # How spread out are MTF signals (lower = tighter)
            context['mtf_consensus_strength'] # Strength of MTF consensus
        ])

        # 3. Structural Quality (8 features)
        features.extend([
            context['ob_quality'],           # Order block formation quality (if present)
            context['fvg_quality'],          # Fair value gap quality
            context['sweep_quality'],        # Liquidity sweep quality
            context['displacement_z'],       # Z-score of price displacement
            context['distance_to_eq'],       # Distance to equilibrium (normalized)
            context['distance_to_poc'],      # Distance to point of control
            context['vacuum_score'],         # Liquidity vacuum score
            context['structural_confluence'] # Overall structural confluence
        ])

        # 4. Regime & Macro (6 features)
        features.extend([
            context['regime_confidence'],    # GMM confidence
            *self._one_hot_regime(context['regime_label']),  # 5 regime one-hot
        ])

        # 5. Volatility & Risk (5 features)
        features.extend([
            context['atr_percentile'],       # Where is current ATR vs historical?
            context['realized_vol_z'],       # Realized volatility z-score
            context['vol_regime'],           # Volatility regime (low/med/high)
            context['regime_stability'],     # How stable is regime (lower = more transitions)
            context['event_proximity']       # Distance to next major event (0-1)
        ])

        # 6. Archetype Context (6 features)
        features.extend([
            *self._one_hot_archetype(archetype_id),  # 5 active archetypes one-hot
            context['archetype_quality']     # Archetype-specific quality score
        ])

        # 7. Historical Context (6 features)
        features.extend([
            context['bars_since_last_ob'],   # Recency of last OB
            context['bars_since_last_sweep'],# Recency of last sweep
            context['bars_since_last_mitigation'], # Recency of mitigation
            context['recent_win_streak'],    # Recent win streak (signed, -5 to +5)
            context['recent_pnl_z'],         # Z-score of recent PnL
            context['drawdown_pct']          # Current drawdown from peak equity
        ])

        # 8. Session & Temporal (4 features)
        features.extend([
            context['session_hour'],         # Hour of day (0-23, normalized)
            context['day_of_week'],          # Day of week (0-6, normalized)
            context['is_london'],            # London session (0/1)
            context['is_ny']                 # NY session (0/1)
        ])

        return np.array(features, dtype=np.float32)
```

**Total Input Features**: ~49 features

---

### Model Architecture

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MetaFusionMLP(nn.Module):
    """
    Lightweight MLP for fusion quality scoring.

    Architecture: 49 → 128 → 64 → 32 → 1
    Activation: GELU (smoother than ReLU, better gradients)
    Regularization: Dropout (0.2), Batch Norm
    Output: Sigmoid (quality score 0-1)
    """

    def __init__(self, input_dim=49, hidden_dims=[128, 64, 32], dropout=0.2):
        super().__init__()

        self.input_bn = nn.BatchNorm1d(input_dim)

        # Layer 1: 49 → 128
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.bn1 = nn.BatchNorm1d(hidden_dims[0])
        self.dropout1 = nn.Dropout(dropout)

        # Layer 2: 128 → 64
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.bn2 = nn.BatchNorm1d(hidden_dims[1])
        self.dropout2 = nn.Dropout(dropout)

        # Layer 3: 64 → 32
        self.fc3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.bn3 = nn.BatchNorm1d(hidden_dims[2])
        self.dropout3 = nn.Dropout(dropout)

        # Output layer: 32 → 1
        self.fc_out = nn.Linear(hidden_dims[2], 1)

    def forward(self, x):
        """
        Args:
            x: (batch_size, 49) feature tensor

        Returns:
            quality_score: (batch_size, 1) quality prediction [0, 1]
        """
        # Input normalization
        x = self.input_bn(x)

        # Layer 1
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.gelu(x)
        x = self.dropout1(x)

        # Layer 2
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.gelu(x)
        x = self.dropout2(x)

        # Layer 3
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.gelu(x)
        x = self.dropout3(x)

        # Output (sigmoid for 0-1 quality score)
        x = self.fc_out(x)
        quality_score = torch.sigmoid(x)

        return quality_score
```

---

## Training Pipeline

### Data Preparation

```python
class FusionTrainingDataset:
    """
    Build training dataset from historical trades + near-misses.
    """

    def build_dataset(self, trade_logs, feature_store, bars_df):
        """
        Create (X, y) pairs for training.

        Positive samples: All historical trades
        Negative samples: Near-miss candidates (archetype fired but below threshold)

        Target: Future R (risk-adjusted return) or binary win/loss
        """

        X_features = []
        y_targets = []

        # Positive samples from trade logs
        for trade in trade_logs:
            entry_bar = bars_df.loc[trade['entry_time']]
            features = self.extract_features(entry_bar, trade['context'])

            # Target: R-multiple (PnL / stop_distance)
            target_r = trade['net_pnl'] / abs(trade['stop_distance'])
            # Normalize to 0-1 (sigmoid-friendly)
            target_quality = self._r_to_quality(target_r)  # R > 2 → 1.0, R < -1 → 0.0

            X_features.append(features)
            y_targets.append(target_quality)

        # Negative samples: Near-misses (archetype detected but rejected)
        near_misses = self.extract_near_misses(bars_df, threshold=0.9)
        for miss in near_misses:
            features = self.extract_features(miss['bar'], miss['context'])
            # Target: 0.0 (setup failed to meet threshold, likely poor quality)
            y_targets.append(0.0)
            X_features.append(features)

        # Hard negative mining: Include losing trades with higher weight
        losing_trades = [t for t in trade_logs if t['net_pnl'] < 0]
        for trade in losing_trades:
            # Add 2x to emphasize learning from losses
            features = self.extract_features(bars_df.loc[trade['entry_time']], trade['context'])
            target_quality = self._r_to_quality(trade['net_pnl'] / abs(trade['stop_distance']))
            X_features.append(features)
            y_targets.append(target_quality)

        return np.array(X_features), np.array(y_targets)

    def _r_to_quality(self, r):
        """
        Map R-multiple to quality score [0, 1]:
        - R > 2 → 1.0 (excellent)
        - R = 0 → 0.5 (breakeven)
        - R < -1 → 0.0 (poor)
        """
        return 1 / (1 + np.exp(-1.5 * r))  # Sigmoid-like mapping
```

### Training Loop

```python
def train_meta_fusion(model, train_loader, val_loader, epochs=100):
    """
    Train meta-fusion MLP with early stopping and calibration.
    """

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

    # Loss: BCE (binary cross-entropy) for quality prediction
    criterion = nn.BCELoss()

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()

            y_pred = model(X_batch).squeeze()
            loss = criterion(y_pred, y_batch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                y_pred = model(X_batch).squeeze()
                loss = criterion(y_pred, y_batch)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'models/meta_fusion_best.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 10:
                print(f"Early stopping at epoch {epoch}")
                break

        scheduler.step(val_loss)

        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

    # Load best model
    model.load_state_dict(torch.load('models/meta_fusion_best.pth'))

    # Calibrate with Platt scaling (important for confidence!)
    calibrated_model = calibrate_model(model, val_loader)

    return calibrated_model

def calibrate_model(model, val_loader):
    """
    Platt scaling to calibrate probabilities (important for uncertainty estimation).
    """
    from sklearn.calibration import CalibratedClassifierCV

    # Extract predictions and targets
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            y_pred = model(X_batch).squeeze()
            all_preds.extend(y_pred.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())

    # Fit Platt scaler
    from scipy.special import logit
    platt_scaler = LogisticRegression()
    platt_scaler.fit(logit(np.clip(all_preds, 1e-6, 1-1e-6)).reshape(-1, 1), all_targets)

    # Wrap model with calibration
    model.platt_scaler = platt_scaler
    return model
```

---

## Integration into Backtest Engine

### Inference Wrapper

```python
class MetaFusionInference:
    """
    Production inference wrapper with fallback and confidence gating.
    """

    def __init__(self, model_path, feature_extractor, min_confidence=0.15):
        self.model = MetaFusionMLP()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        self.feature_extractor = feature_extractor
        self.min_confidence = min_confidence  # Entropy threshold for fallback

    def predict_quality(self, bar, module_scores, context, archetype_id):
        """
        Predict quality score with confidence gating.

        Returns:
            quality_score: float [0, 1]
            confidence: float [0, 1] - higher = more certain
            use_prediction: bool - False if fallback to rules
        """
        # Extract features
        features = self.feature_extractor.extract_features(
            bar, module_scores, context, archetype_id
        )

        # Inference
        with torch.no_grad():
            X = torch.FloatTensor(features).unsqueeze(0)
            quality_score = self.model(X).item()

        # Estimate confidence via dropout sampling (MC Dropout)
        confidences = []
        self.model.train()  # Enable dropout
        with torch.no_grad():
            for _ in range(10):  # 10 forward passes
                pred = self.model(X).item()
                confidences.append(pred)
        self.model.eval()

        # Confidence = inverse of std deviation
        confidence = 1 - np.std(confidences)

        # Fallback to rules if low confidence
        use_prediction = confidence >= self.min_confidence

        return quality_score, confidence, use_prediction
```

### Modified Fusion Calculation

```python
# Inside RouterAwareBacktest.compute_advanced_fusion_score()

def compute_advanced_fusion_score_with_meta(self, row, adapted_params):
    """
    Enhanced fusion with meta-learner quality multiplier.
    """

    # Step 1: Compute rule-based fusion (existing logic)
    fusion_rule, context = self.compute_advanced_fusion_score_original(row, adapted_params)

    # Step 2: If meta-learner enabled, get quality multiplier
    if self.meta_fusion_enabled:
        quality_score, confidence, use_pred = self.meta_fusion.predict_quality(
            row,
            module_scores=context['module_scores'],
            context=context,
            archetype_id=context['active_archetype']
        )

        if use_pred:
            # Apply quality multiplier (capped range)
            alpha = self.config['meta_fusion']['alpha']  # Sensitivity (e.g., 0.5)
            baseline = self.config['meta_fusion']['baseline']  # Neutral point (e.g., 0.6)
            min_mult = self.config['meta_fusion']['min_mult']  # e.g., 0.75
            max_mult = self.config['meta_fusion']['max_mult']  # e.g., 1.25

            multiplier = np.clip(
                1 + alpha * (quality_score - baseline),
                min_mult,
                max_mult
            )

            fusion_final = fusion_rule * multiplier

            # Log for analysis
            self.log_meta_decision(row, fusion_rule, quality_score, multiplier, fusion_final, confidence)
        else:
            # Low confidence → fallback to pure rules
            fusion_final = fusion_rule
            self.log_meta_fallback(row, confidence)
    else:
        fusion_final = fusion_rule

    return fusion_final, context
```

---

## Evaluation Metrics

### Training Metrics
- **BCE Loss**: Binary cross-entropy on validation set
- **AUC-ROC**: Discrimination between good/bad setups
- **AUC-PR**: Precision-recall (better for imbalanced data)
- **ECE (Expected Calibration Error)**: Measures confidence calibration (target: < 0.05)

### Trading Metrics (OOS)
- **Profit Factor**: Must improve vs baseline
- **Sharpe Ratio**: Risk-adjusted returns
- **Max Drawdown**: Must not increase significantly
- **Win Rate**: Should improve (especially for trap-within-trend)
- **Average R**: Closer to positive (reduce large losses)

### Acceptance Criteria
- Validation AUC-ROC ≥ 0.70
- ECE ≤ 0.05 (well-calibrated)
- OOS Profit Factor improvement ≥ 5% vs baseline
- OOS Sharpe improvement ≥ 0.1 vs baseline
- No increase in max drawdown

---

## Safety Mechanisms

### 1. Confidence Gating
If model uncertainty (std of MC dropout predictions) is high, fall back to pure rule-based fusion.

### 2. Multiplier Caps
Quality multiplier bounded to [0.75, 1.25] to prevent extreme scaling.

### 3. Veto Preservation
MLP **never** overrides vetoes:
- Event suppression windows
- Regime confidence floors
- Hard structural vetoes (wrong side of equilibrium, etc.)

### 4. Feature Drift Detection
Monitor input feature distributions in production. If drift detected (e.g., KL divergence > threshold), freeze model and alert.

### 5. Per-Archetype Kill Switch
If an archetype's performance degrades below PF 1.0 for N consecutive trades, disable its meta-fusion multiplier.

---

## File Structure

```
models/
  meta_fusion_v1.pth             # Trained model weights
  meta_fusion_v1_calibrated.pkl  # Platt scaler for calibration

engine/ml/
  meta_fusion.py                 # MetaFusionMLP class
  feature_extractor.py           # FusionFeatureExtractor
  inference.py                   # MetaFusionInference wrapper
  training.py                    # Training loop
  dataset.py                     # FusionTrainingDataset

bin/
  train_meta_fusion.py           # Training script
  evaluate_meta_fusion.py        # Evaluation script

configs/
  meta_fusion_config.json        # Hyperparameters (alpha, baseline, caps, etc.)
```

---

## Next Steps

1. **Week 1-2**: Build training dataset from 2022-2023 trade logs + near-misses
2. **Week 3**: Train MetaFusionMLP on 2022-2023, validate on Q1 2024
3. **Week 4**: Integrate into backtest engine, run full OOS test on Q2-Q4 2024
4. **Week 5**: Ablation tests (with/without meta-fusion) and acceptance gate

---

## Expected Impact

**Conservative Estimate**:
- Improve trap-within-trend WR from 46% → 52% (better quality filtering)
- Reduce OB retest false positives (maintain 90% WR, increase detection by 20%)
- Overall system PF improvement: +0.2 to +0.3
- Annual PnL gain: +$400-600

**Optimistic Estimate**:
- Trap-within-trend WR → 55%
- OB retest detection +30%
- System PF improvement: +0.4 to +0.6
- Annual PnL gain: +$800-1,000
