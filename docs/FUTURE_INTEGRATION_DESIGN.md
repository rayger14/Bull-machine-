# Future Integration Design: B0 + Archetypes

**Version:** 1.0.0
**Date:** 2025-12-03
**Status:** DESIGN OPTIONS
**Purpose:** Evaluate long-term integration strategies for dual-system architecture

---

## Executive Summary

Currently, B0 and Archetypes run as **independent systems** with a capital router. This document explores three future integration options:

1. **Option A: Keep Separate** (Capital Router) - Simplest
2. **Option B: Fix Wrapper** (Unified Framework) - Most pragmatic
3. **Option C: Build Meta-System** (Ensemble ML) - Most sophisticated

**Recommendation:** Defer decision until 3-6 months of live data collected. Start with Option A (current state), migrate to Option B if archetypes validated.

---

## Current State: Independent Systems

### Architecture

```
┌────────────────────────────────┐
│     System B0 (v2 Framework)   │
│     - BuyHoldSellClassifier    │
│     - Clean architecture       │
│     - 500 LOC                  │
└────────────────┬───────────────┘
                 │
                 ▼
         ┌───────────────┐
         │ Capital Router│
         │ (Allocation)  │
         └───────┬───────┘
                 │
                 ▼
         ┌───────────────┐
         │   Market      │
         └───────────────┘
                 ▲
                 │
         ┌───────┴───────┐
         │ Capital Router│
         └───────┬───────┘
                 │
                 ▼
┌────────────────┴───────────────┐
│  System ARCHETYPES (v1 Legacy) │
│  - ArchetypeModel wrapper      │
│  - Old backtester (39k LOC)    │
│  - S4/S5/S1                    │
└────────────────────────────────┘
```

### Pros
- Systems operate independently (failure isolation)
- B0 ready now, archetypes can catch up
- Easy to disable one system without affecting the other
- Clear separation of concerns

### Cons
- Duplicate infrastructure (two backtesters, two monitoring systems)
- Manual capital allocation (no automatic optimization)
- Archetypes stuck on legacy framework (tech debt)
- Higher maintenance cost (maintain both systems)

---

## Option A: Keep Separate (Capital Router)

### Overview

**Philosophy:** Don't fix what isn't broken. Keep systems independent, improve capital router intelligence.

**Timeline:** 0 months (already implemented)

**Effort:** LOW (minor enhancements only)

**Risk:** LOW (no major changes)

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    INTELLIGENT CAPITAL ROUTER                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input:                                                         │
│  - B0 recent performance (PF, WR, DD)                           │
│  - S4 recent performance                                        │
│  - S5 recent performance                                        │
│  - S1 recent performance (if enabled)                           │
│  - Current market regime (risk_on/neutral/risk_off/crisis)      │
│  - Historical regime performance matrix                         │
│                                                                 │
│  Algorithm:                                                     │
│  1. Calculate rolling 30-day PF for each system                 │
│  2. Adjust allocation based on relative performance             │
│     - If System X PF > avg * 1.3: +10% allocation               │
│     - If System X PF < avg * 0.7: -10% allocation               │
│  3. Apply regime weighting                                      │
│     - Risk_off: Increase S4/S1, decrease S5                     │
│     - Risk_on: Increase S5/B0, decrease S4                      │
│     - Crisis: Increase S1/B0, decrease others                   │
│  4. Respect limits:                                             │
│     - Min allocation per system: 10%                            │
│     - Max allocation per system: 70%                            │
│     - Total: 100%                                               │
│                                                                 │
│  Output:                                                        │
│  - B0 allocation: X%                                            │
│  - S4 allocation: Y%                                            │
│  - S5 allocation: Z%                                            │
│  - S1 allocation: W%                                            │
│  - Rebalance frequency: Monthly                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

       ↓               ↓               ↓               ↓
   ┌───────┐      ┌───────┐      ┌───────┐      ┌───────┐
   │  B0   │      │  S4   │      │  S5   │      │  S1   │
   │ (v2)  │      │ (v1)  │      │ (v1)  │      │ (v1)  │
   └───┬───┘      └───┬───┘      └───┬───┘      └───┬───┘
       │              │              │              │
       └──────────────┴──────────────┴──────────────┘
                         ↓
                   ┌──────────┐
                   │  Market  │
                   └──────────┘
```

### Implementation

**Month 1-2: Enhance Capital Router**

**File:** `engine/capital_router.py` (NEW)

```python
class CapitalRouter:
    """
    Intelligent capital allocation across multiple independent systems.

    Features:
    - Performance-based allocation (rolling 30-day PF)
    - Regime-aware weighting
    - Risk limits enforcement
    - Monthly rebalancing
    """

    def __init__(self, total_capital: float):
        self.total_capital = total_capital
        self.systems = {
            'B0': {'min': 0.10, 'max': 0.70},
            'S4': {'min': 0.10, 'max': 0.50},
            'S5': {'min': 0.10, 'max': 0.50},
            'S1': {'min': 0.05, 'max': 0.30}
        }
        self.current_allocation = {
            'B0': 0.50,
            'S4': 0.25,
            'S5': 0.25,
            'S1': 0.00  # Disabled until validated
        }

    def calculate_allocation(
        self,
        performance_30d: Dict[str, float],  # System -> PF
        current_regime: str,  # risk_on/neutral/risk_off/crisis
        regime_performance: Dict[str, Dict[str, float]]  # System -> Regime -> PF
    ) -> Dict[str, float]:
        """
        Calculate optimal allocation based on recent performance and regime.

        Returns:
            Dict[str, float]: System -> Allocation percentage
        """
        # 1. Performance-based adjustment
        avg_pf = np.mean(list(performance_30d.values()))
        perf_scores = {}
        for system, pf in performance_30d.items():
            if pf > avg_pf * 1.3:
                perf_scores[system] = 1.1  # +10%
            elif pf < avg_pf * 0.7:
                perf_scores[system] = 0.9  # -10%
            else:
                perf_scores[system] = 1.0  # No change

        # 2. Regime-based weighting
        regime_weights = self._get_regime_weights(current_regime)

        # 3. Combine performance and regime
        combined_scores = {}
        for system in self.systems.keys():
            perf_score = perf_scores.get(system, 1.0)
            regime_weight = regime_weights.get(system, 1.0)
            combined_scores[system] = perf_score * regime_weight

        # 4. Normalize to allocation percentages
        total_score = sum(combined_scores.values())
        new_allocation = {
            system: score / total_score
            for system, score in combined_scores.items()
        }

        # 5. Apply limits (min/max per system)
        new_allocation = self._apply_limits(new_allocation)

        # 6. Smooth transition (don't change more than 15% per month)
        new_allocation = self._smooth_transition(new_allocation, max_change=0.15)

        return new_allocation

    def _get_regime_weights(self, regime: str) -> Dict[str, float]:
        """Regime-specific system weights."""
        if regime == 'risk_on':
            return {'B0': 1.2, 'S4': 0.6, 'S5': 1.5, 'S1': 0.5}
        elif regime == 'neutral':
            return {'B0': 1.0, 'S4': 1.0, 'S5': 1.0, 'S1': 0.8}
        elif regime == 'risk_off':
            return {'B0': 1.1, 'S4': 1.5, 'S5': 0.6, 'S1': 1.2}
        elif regime == 'crisis':
            return {'B0': 1.3, 'S4': 1.0, 'S5': 1.2, 'S1': 2.0}
        else:
            return {'B0': 1.0, 'S4': 1.0, 'S5': 1.0, 'S1': 1.0}

    def _apply_limits(self, allocation: Dict[str, float]) -> Dict[str, float]:
        """Enforce min/max limits per system."""
        adjusted = {}
        for system, alloc in allocation.items():
            limits = self.systems[system]
            adjusted[system] = np.clip(alloc, limits['min'], limits['max'])

        # Renormalize to 100%
        total = sum(adjusted.values())
        adjusted = {k: v / total for k, v in adjusted.items()}

        return adjusted

    def _smooth_transition(
        self,
        new_allocation: Dict[str, float],
        max_change: float = 0.15
    ) -> Dict[str, float]:
        """Limit allocation changes to avoid drastic rebalancing."""
        smoothed = {}
        for system, new_alloc in new_allocation.items():
            current_alloc = self.current_allocation.get(system, 0.0)
            change = new_alloc - current_alloc
            if abs(change) > max_change:
                change = np.sign(change) * max_change
            smoothed[system] = current_alloc + change

        # Renormalize
        total = sum(smoothed.values())
        smoothed = {k: v / total for k, v in smoothed.items()}

        return smoothed
```

**Testing:**
- Unit tests (allocation logic)
- Backtest on historical data (2022-2024)
- Paper trading (1 month)

### Pros

**Simplicity:**
- No changes to B0 or archetypes
- Easy to understand and maintain
- Clear separation of concerns

**Flexibility:**
- Can add new systems easily (just plug into router)
- Can disable systems independently
- Can adjust allocation algorithm without touching trading logic

**Risk Management:**
- Systems isolated (one fails, others unaffected)
- Easy to roll back (just change allocation)
- Clear monitoring (per-system performance)

**Low Cost:**
- Minimal development (1-2 weeks)
- No risky refactoring
- Uses existing infrastructure

### Cons

**Inefficiency:**
- Duplicate infrastructure (two backtesters)
- Manual integration (router is separate layer)
- Higher maintenance (two codebases)

**Missed Opportunities:**
- Systems don't learn from each other
- No signal combination (just parallel operation)
- Can't leverage archetype features in B0

**Tech Debt:**
- Archetypes still on legacy framework
- Wrapper bugs not fully resolved
- Old backtester (39k LOC) still in use

### When to Choose

**Choose Option A if:**
- Want to deploy quickly (already implemented)
- Risk-averse (minimal changes)
- Archetypes performance uncertain (keep options open)
- Team small (can't afford large refactoring)

**Time Horizon:** 3-12 months (short-term to medium-term)

---

## Option B: Fix Wrapper (Unified Framework)

### Overview

**Philosophy:** Migrate archetypes to new framework (v2), deprecate old backtester, unified codebase.

**Timeline:** 2-3 months

**Effort:** MEDIUM (fix wrapper, migrate archetypes, deprecate v1)

**Risk:** MEDIUM (risk of breaking archetypes during migration)

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                  UNIFIED FRAMEWORK (v2)                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  BaseModel Interface (Abstract)                                 │
│    ├─ fit(train_data)                                           │
│    ├─ predict(bar, position) -> Signal                          │
│    └─ requires_enrichment() -> bool                             │
│                                                                 │
│  Models:                                                        │
│    ├─ BuyHoldSellClassifier (B0) ✅                             │
│    ├─ ArchetypeModel (S4/S5/S1) ⚠️ (needs fixes)                │
│    └─ Future models...                                          │
│                                                                 │
│  BacktestEngine (Shared)                                        │
│    ├─ Load data                                                 │
│    ├─ Apply enrichment (if model.requires_enrichment())         │
│    ├─ Run backtest                                              │
│    └─ Generate metrics                                          │
│                                                                 │
│  Capital Allocation (Shared)                                    │
│    ├─ Performance tracking                                      │
│    ├─ Dynamic rebalancing                                       │
│    └─ Risk management                                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

            ↓                       ↓
    ┌──────────────┐       ┌──────────────┐
    │  B0 (v2)     │       │  S4/S5/S1    │
    │  - Simple    │       │  - Complex   │
    │  - Native    │       │  - Wrapped   │
    └──────┬───────┘       └──────┬───────┘
           │                      │
           └──────────┬───────────┘
                      ↓
               ┌──────────────┐
               │   Market     │
               └──────────────┘
```

### Implementation

**Month 1: Fix ArchetypeModel Wrapper**

**Tasks:**
1. Fix regime gating bug (currently blocks S1)
2. Add RegimeClassifier instantiation
3. Add enrichment hooks
4. Test on historical data (2022-2024)

**File:** `engine/models/archetype_model.py`

```python
class ArchetypeModel(BaseModel):
    """
    Wrapper around logic_v2_adapter for archetype-based trading.

    Bridges old archetype logic to new v2 framework.
    """

    def __init__(
        self,
        config_path: str,
        archetype_name: str,  # 'liquidity_vacuum', 'funding_divergence', 'long_squeeze'
        name: str
    ):
        super().__init__(name=name)
        self.config_path = config_path
        self.archetype_name = archetype_name
        self.config = self._load_config(config_path)

        # FIX 1: Instantiate RegimeClassifier
        self.regime_classifier = RegimeClassifier(
            model_path='models/regime_classifier_gmm.pkl'
        )

        # Load archetype logic
        self.logic = LogicV2Adapter(self.config)

    def requires_enrichment(self) -> bool:
        """Check if this archetype needs runtime enrichment."""
        return self.archetype_name in ['liquidity_vacuum', 'funding_divergence', 'long_squeeze']

    def enrich(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply runtime enrichment (archetype-specific features).

        This is called by BacktestEngine before running backtest.
        """
        if self.archetype_name == 'liquidity_vacuum':
            from engine.strategies.archetypes.bear.liquidity_vacuum_runtime import apply_liquidity_vacuum_enrichment
            data = apply_liquidity_vacuum_enrichment(data)
        elif self.archetype_name == 'funding_divergence':
            from engine.strategies.archetypes.bear.funding_divergence_runtime import apply_funding_divergence_enrichment
            data = apply_funding_divergence_enrichment(data)
        elif self.archetype_name == 'long_squeeze':
            from engine.strategies.archetypes.bear.long_squeeze_runtime import apply_long_squeeze_enrichment
            data = apply_long_squeeze_enrichment(data)

        return data

    def predict(self, bar: pd.Series, position: Optional[Position] = None) -> Signal:
        """
        Generate signal using archetype logic.

        FIX 2: Classify regime properly (not hardcoded 'neutral')
        FIX 3: Build complete RuntimeContext
        """
        # Classify regime (don't rely on macro_regime column)
        regime_label = self.regime_classifier.predict_single(bar)
        regime_probs = self.regime_classifier.predict_proba_single(bar)

        # Build RuntimeContext
        context = RuntimeContext(
            bar=bar,
            regime_label=regime_label,
            regime_probs=regime_probs,
            timestamp=bar.name
        )

        # Call archetype detector
        archetype_code, confidence, stop_loss_price = self.logic.detect(context)

        # Convert to Signal
        if archetype_code is None:
            return Signal(direction='hold', confidence=0.0)
        else:
            return Signal(
                direction='long',  # All current archetypes are long-only
                confidence=confidence,
                entry_price=bar['close'],
                stop_loss=stop_loss_price
            )
```

**File:** `engine/backtesting/engine.py` (UPDATE)

```python
class BacktestEngine:
    """
    Model-agnostic backtesting engine.

    FIX: Add enrichment hook for archetypes.
    """

    def __init__(self, model: BaseModel, data: pd.DataFrame):
        self.model = model
        self.data = data

    def run(self, start_date, end_date):
        """
        Run backtest.

        FIX: Apply enrichment if model requires it.
        """
        # Enrichment hook (NEW)
        if self.model.requires_enrichment():
            logger.info(f"Applying runtime enrichment for {self.model.name}...")
            self.data = self.model.enrich(self.data)
            logger.info(f"Enrichment complete. Data shape: {self.data.shape}")

        # Filter date range
        mask = (self.data.index >= start_date) & (self.data.index <= end_date)
        test_data = self.data[mask]

        # Run backtest (unchanged)
        results = self._backtest_loop(test_data)

        return results

    def _backtest_loop(self, data):
        # ... (unchanged)
```

**Testing:**
1. Unit tests (wrapper fixes)
2. Integration tests (full backtest with enrichment)
3. Validation: Compare v1 vs v2 results (should match)
4. Paper trading (1 month)

**Month 2: Deprecate Old Backtester**

**Tasks:**
1. Migrate all archetype configs to v2 format (if needed)
2. Update documentation (point to v2 framework)
3. Archive `bin/backtest_knowledge_v2.py` (mark as legacy)
4. Remove v1 dependencies (cleanup)

**Testing:**
1. Full regression test (all archetypes on v2)
2. Performance comparison (v1 vs v2 should match)
3. Paper trading (1 month on v2 only)

**Month 3: Optimization and Cleanup**

**Tasks:**
1. Optimize performance (v2 should be faster than v1)
2. Add missing features (if any)
3. Full documentation update
4. Deploy to production

### Pros

**Unified Codebase:**
- Single framework for all models (B0 + archetypes)
- Single backtester (shared infrastructure)
- Easier maintenance (one codebase, not two)

**Clean Architecture:**
- BaseModel interface (easy to add new models)
- Enrichment hooks (standardized pattern)
- Deprecate legacy code (reduce tech debt)

**Better Integration:**
- Models use same data pipeline
- Models use same backtesting engine
- Models use same monitoring infrastructure

**Long-Term Scalability:**
- Easy to add new models (just implement BaseModel)
- Easy to add new features (extend BaseModel)
- Easy to test (unified testing framework)

### Cons

**Migration Risk:**
- Archetypes may break during migration
- Results may not match v1 exactly (floating point, rounding)
- Regression testing required (time-consuming)

**Development Cost:**
- 2-3 months of engineering time
- Requires careful testing (can't break production)
- Requires documentation updates

**Temporary Instability:**
- Archetypes may be offline during migration
- Need to run v1 and v2 in parallel (redundancy)
- Rollback plan required

### When to Choose

**Choose Option B if:**
- Archetypes validated (proven value in live trading)
- Want to reduce tech debt (clean up legacy code)
- Team has capacity (2-3 months of work)
- Long-term investment (1+ year time horizon)

**Time Horizon:** 6-24 months (medium-term to long-term)

---

## Option C: Build Meta-System (Ensemble ML)

### Overview

**Philosophy:** Treat B0 and archetypes as **features** for a meta-learner. Train ML model to optimally combine signals.

**Timeline:** 3-6 months

**Effort:** HIGH (ML engineering, data collection, training, validation)

**Risk:** HIGH (complex, may not beat simple approaches)

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       META-SYSTEM (ML)                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input Features:                                                │
│    ├─ B0 signal (long/hold), confidence                         │
│    ├─ S4 signal (long/hold), confidence, fusion score           │
│    ├─ S5 signal (long/hold), confidence, fusion score           │
│    ├─ S1 signal (long/hold), confidence, fusion score           │
│    ├─ Market features (price, volume, ATR, RSI, etc.)           │
│    ├─ Regime features (risk_on/neutral/risk_off/crisis)         │
│    └─ Context features (time of day, day of week, etc.)         │
│                                                                 │
│  ML Model:                                                      │
│    ├─ XGBoost or Random Forest or Neural Network               │
│    ├─ Trained on historical data (2022-2024)                    │
│    ├─ Target: Profitable long entry (1 = yes, 0 = no)           │
│    └─ Output: Probability of profitable entry (0.0-1.0)         │
│                                                                 │
│  Decision Logic:                                                │
│    IF meta_probability > threshold (e.g., 0.6):                 │
│        entry_signal = True                                      │
│        confidence = meta_probability                            │
│        stop_loss = weighted_avg([B0_stop, S4_stop, S5_stop])    │
│    ELSE:                                                        │
│        entry_signal = False                                     │
│                                                                 │
│  Output:                                                        │
│    - Signal (long/hold)                                         │
│    - Confidence (0.0-1.0)                                       │
│    - Stop loss (weighted ensemble)                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

        ↓
┌─────────────────┐
│  Market (BTC)   │
└─────────────────┘
```

### Implementation

**Month 1-2: Data Collection and Feature Engineering**

**Goal:** Collect historical signals from B0, S4, S5, S1 on 2022-2024 data.

**Features to Generate:**

```python
# For each 1H bar in 2022-2024:
features = {
    # System signals
    'b0_signal': 1 if B0 would enter else 0,
    'b0_confidence': B0 confidence (always 1.0 for B0, or use drawdown magnitude),
    's4_signal': 1 if S4 would enter else 0,
    's4_confidence': S4 fusion score,
    's4_fusion_funding': S4 funding component,
    's4_fusion_liquidity': S4 liquidity component,
    's5_signal': 1 if S5 would enter else 0,
    's5_confidence': S5 fusion score,
    's5_fusion_funding': S5 funding component (positive),
    's5_fusion_rsi': S5 RSI component,
    's1_signal': 1 if S1 would enter else 0,
    's1_confidence': S1 fusion score,
    's1_fusion_liquidity': S1 liquidity component,
    's1_fusion_crisis': S1 crisis component,

    # Market features
    'close': Current close price,
    'volume': Current volume,
    'atr_14': ATR,
    'rsi_14': RSI,
    'drawdown_30d': 30-day drawdown,

    # Regime features
    'regime_risk_on': 1 if risk_on else 0,
    'regime_neutral': 1 if neutral else 0,
    'regime_risk_off': 1 if risk_off else 0,
    'regime_crisis': 1 if crisis else 0,

    # Context features
    'hour_of_day': 0-23,
    'day_of_week': 0-6,
}

# Target (label):
target = 1 if (entry at this bar would be profitable after 7 days) else 0
```

**File:** `bin/generate_meta_features.py` (NEW)

```python
def generate_meta_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Generate meta-features for ML training.

    For each bar, collect signals from all systems.
    """
    # Initialize models
    b0 = BuyHoldSellClassifier(...)
    s4 = ArchetypeModel(config='s4_optimized.json', archetype='funding_divergence')
    s5 = ArchetypeModel(config='s5_optimized.json', archetype='long_squeeze')
    s1 = ArchetypeModel(config='s1_v2_production.json', archetype='liquidity_vacuum')

    features = []
    for idx, bar in data.iterrows():
        # Get signals from each system
        b0_signal = b0.predict(bar)
        s4_signal = s4.predict(bar)
        s5_signal = s5.predict(bar)
        s1_signal = s1.predict(bar)

        # Compute target (profitable entry?)
        future_price = data.loc[idx + pd.Timedelta(days=7), 'close']
        target = 1 if (future_price > bar['close'] * 1.03) else 0  # >3% profit in 7 days

        # Construct feature vector
        features.append({
            'timestamp': idx,
            'b0_signal': 1 if b0_signal.direction == 'long' else 0,
            's4_signal': 1 if s4_signal.direction == 'long' else 0,
            's5_signal': 1 if s5_signal.direction == 'long' else 0,
            's1_signal': 1 if s1_signal.direction == 'long' else 0,
            # ... (add all features)
            'target': target
        })

    return pd.DataFrame(features)
```

**Month 3-4: Model Training and Validation**

**Goal:** Train ML model to predict profitable entries.

**File:** `bin/train_meta_learner.py` (NEW)

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, roc_auc_score

def train_meta_learner(features: pd.DataFrame):
    """
    Train meta-learner on historical signals.
    """
    # Split features and target
    X = features.drop(columns=['timestamp', 'target'])
    y = features['target']

    # Time series cross-validation (5 folds)
    tscv = TimeSeriesSplit(n_splits=5)

    # Train Random Forest
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=50,
        class_weight='balanced',  # Handle class imbalance
        random_state=42
    )

    # Cross-validation
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)
        y_proba = rf.predict_proba(X_val)[:, 1]

        print(f"Fold {fold+1}:")
        print(classification_report(y_val, y_pred))
        print(f"ROC AUC: {roc_auc_score(y_val, y_proba):.3f}\n")

    # Train on full dataset
    rf.fit(X, y)

    # Feature importance
    importances = rf.feature_importances_
    feature_names = X.columns
    print("Top 10 Most Important Features:")
    for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1])[:10]:
        print(f"  {name}: {imp:.4f}")

    # Save model
    joblib.dump(rf, 'models/meta_learner.pkl')

    return rf
```

**Month 5: Backtesting Meta-System**

**Goal:** Backtest meta-system on 2022-2024 data, compare to B0 and archetypes.

**File:** `engine/models/meta_system.py` (NEW)

```python
class MetaSystem(BaseModel):
    """
    Meta-system that combines B0 + S4 + S5 + S1 signals using ML.
    """

    def __init__(self, model_path: str):
        super().__init__(name='MetaSystem')
        self.model = joblib.load(model_path)  # Trained RF/XGBoost

        # Initialize subsystems
        self.b0 = BuyHoldSellClassifier(...)
        self.s4 = ArchetypeModel(...)
        self.s5 = ArchetypeModel(...)
        self.s1 = ArchetypeModel(...)

    def predict(self, bar: pd.Series, position: Optional[Position] = None) -> Signal:
        """
        Generate signal using meta-learner.
        """
        # Get signals from subsystems
        b0_signal = self.b0.predict(bar)
        s4_signal = self.s4.predict(bar)
        s5_signal = self.s5.predict(bar)
        s1_signal = self.s1.predict(bar)

        # Construct feature vector
        features = {
            'b0_signal': 1 if b0_signal.direction == 'long' else 0,
            's4_signal': 1 if s4_signal.direction == 'long' else 0,
            's5_signal': 1 if s5_signal.direction == 'long' else 0,
            's1_signal': 1 if s1_signal.direction == 'long' else 0,
            # ... (add all features)
        }

        X = pd.DataFrame([features])

        # Predict with meta-learner
        proba = self.model.predict_proba(X)[0, 1]  # Probability of profitable entry

        # Decision
        if proba > 0.6:  # Threshold (tune on validation set)
            # Weighted ensemble stop loss
            stops = [
                b0_signal.stop_loss if b0_signal.direction == 'long' else None,
                s4_signal.stop_loss if s4_signal.direction == 'long' else None,
                s5_signal.stop_loss if s5_signal.direction == 'long' else None,
                s1_signal.stop_loss if s1_signal.direction == 'long' else None
            ]
            stops = [s for s in stops if s is not None]
            weighted_stop = np.mean(stops) if stops else bar['close'] * 0.95

            return Signal(
                direction='long',
                confidence=proba,
                entry_price=bar['close'],
                stop_loss=weighted_stop
            )
        else:
            return Signal(direction='hold', confidence=0.0)
```

**Month 6: Optimization and Deployment**

**Goal:** Tune meta-system, deploy to paper trading.

**Tasks:**
1. Tune probability threshold (0.5, 0.6, 0.7?)
2. Feature selection (remove low-importance features)
3. Test different ML models (XGBoost, LightGBM, Neural Network)
4. Walk-forward validation (3 folds)
5. Paper trading (1 month)
6. If successful: Deploy to live

### Pros

**Optimal Signal Combination:**
- ML learns optimal weights for each system
- Adapts to changing market conditions
- Can discover non-obvious patterns (e.g., "B0 + S4 at same time = stronger signal")

**Highest Potential:**
- May achieve PF > 3.5 (beat individual systems)
- May reduce drawdown (smarter exits)
- May improve win rate (filter false signals)

**Automatic Optimization:**
- No manual capital allocation needed
- Model learns from data (not hardcoded rules)
- Can retrain quarterly with new data

**Advanced Features:**
- Can incorporate additional features (sentiment, on-chain, etc.)
- Can add more systems easily (just add features)
- Can use deep learning (if needed)

### Cons

**Complexity:**
- Most complex approach (ML engineering required)
- Hard to explain (black box)
- Requires large dataset (1+ years of data)

**Overfitting Risk:**
- ML models can overfit to training data
- May not generalize to new regimes
- Requires careful cross-validation

**Maintenance Cost:**
- Need to retrain model regularly (quarterly?)
- Need to monitor feature drift
- Need ML expertise on team

**Delayed Deployment:**
- 3-6 months before ready
- Requires extensive validation
- Higher risk of failure (may not beat simple approaches)

**Dependency:**
- Requires ALL subsystems working (B0, S4, S5, S1)
- If one subsystem breaks, meta-system affected
- More moving parts = more things to break

### When to Choose

**Choose Option C if:**
- All systems validated (B0, S4, S5, S1 working well)
- Want absolute best performance (willing to invest)
- Team has ML expertise
- Long-term investment (1-2 year time horizon)
- Have large dataset (2+ years of live data)

**Time Horizon:** 12-24 months (long-term)

---

## Decision Criteria

### Choose Option A (Keep Separate) If:

```
✓ Want to deploy quickly (0 months)
✓ Risk-averse (minimal changes)
✓ Archetypes performance uncertain
✓ Team small (1-2 engineers)
✓ Time horizon: 3-12 months
✓ Budget: Low ($10k infrastructure)
```

**Recommendation:** **START HERE** (current state)

### Choose Option B (Fix Wrapper) If:

```
✓ Archetypes validated (PF > 1.5 in live trading)
✓ Want to reduce tech debt
✓ Team medium (2-3 engineers)
✓ Time horizon: 6-24 months
✓ Budget: Medium ($50k engineering time)
```

**Recommendation:** **MIGRATE AFTER 3-6 MONTHS** (if archetypes validated)

### Choose Option C (Meta-System) If:

```
✓ All systems validated (B0, S4, S5, S1)
✓ Want maximum performance
✓ Team large (3+ engineers, ML expertise)
✓ Time horizon: 12-24 months
✓ Budget: High ($100k+ engineering time)
✓ Have 2+ years of live data
```

**Recommendation:** **CONSIDER AFTER 12 MONTHS** (if individual systems proven)

---

## Hybrid Approach (Recommended)

### Phase 1 (Months 0-3): Option A

- Deploy B0 immediately
- Deploy S4/S5 with capital router
- Collect live performance data
- Validate archetypes in production

### Phase 2 (Months 3-6): Evaluate

- If archetypes validated (PF > 1.5): Proceed to Phase 3
- If archetypes fail (PF < 1.5): Stay on Option A (B0 only)

### Phase 3 (Months 6-12): Option B

- Fix wrapper (1-2 months)
- Migrate archetypes to v2 framework (1 month)
- Deprecate old backtester (1 month)
- Deploy unified system (1 month)

### Phase 4 (Months 12-24): Option C (Optional)

- Collect 12 months of live data (all systems)
- Train meta-learner (2-3 months)
- Validate meta-system (1 month)
- Deploy if outperforms (PF > 3.5)

**Total Timeline:** 24 months from start to Meta-System

**Risk Gates:**
- Gate 1 (Month 3): Archetypes validated? Yes → Continue | No → Stay on Option A
- Gate 2 (Month 12): Unified framework stable? Yes → Continue | No → Rollback
- Gate 3 (Month 24): Meta-system outperforms? Yes → Deploy | No → Keep Option B

---

## Summary Table

| Dimension | Option A (Router) | Option B (Wrapper Fix) | Option C (Meta-System) |
|-----------|-------------------|------------------------|------------------------|
| **Timeline** | 0 months | 2-3 months | 3-6 months |
| **Effort** | LOW | MEDIUM | HIGH |
| **Risk** | LOW | MEDIUM | HIGH |
| **Cost** | $10k | $50k | $100k+ |
| **Complexity** | Simple | Medium | Complex |
| **Performance Potential** | Good (2.0-2.5 PF) | Good (2.5-3.0 PF) | Excellent (3.0-3.5+ PF) |
| **Maintenance** | HIGH (two systems) | MEDIUM (one system) | HIGH (ML retraining) |
| **Tech Debt** | HIGH (legacy code remains) | LOW (clean architecture) | MEDIUM (ML complexity) |
| **Scalability** | Medium | HIGH | MEDIUM |
| **When to Choose** | Now | After 3-6 months | After 12 months |

---

## Recommendation

**Start with Option A (current state), migrate to Option B after validation, consider Option C as long-term optimization.**

**Rationale:**
1. Option A is already implemented (zero delay)
2. Allows validation of archetypes with minimal risk
3. Option B is natural next step (if archetypes work)
4. Option C is aspirational (requires proof of concept first)

**Timeline:**
- Months 0-3: Option A (evaluate)
- Months 3-12: Option B (if validated)
- Months 12-24: Option C (if justified)

**Decision Points:**
- Month 3: Do archetypes beat PF 1.5? Yes → Proceed to Option B | No → Stay on Option A
- Month 12: Is unified framework stable? Yes → Consider Option C | No → Stick with Option B
- Month 24: Does meta-system beat individual systems by >20%? Yes → Deploy | No → Keep Option B

---

**Document Owner:** Architecture Team
**Last Updated:** 2025-12-03
**Next Review:** After 3 months of live data (re-evaluate integration path)
