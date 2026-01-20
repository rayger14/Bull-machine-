# Regime-Aware Optimization Framework

**Document Version:** 1.0
**Date:** 2025-11-24
**Author:** System Architect (Claude Code)
**Status:** PRODUCTION-READY DESIGN

---

## Executive Summary

This document specifies an institutional-grade optimization framework that calibrates archetype thresholds WITHIN regime states, not across full years. The framework eliminates the fundamental flaw of optimizing on mislabeled data by stratifying all optimization, validation, and portfolio construction by market regime.

**Core Principle:** Every bar has a regime label. Every optimization happens on regime-filtered bars. Every metric is measured per-regime.

**Key Deliverables:**
1. Regime-stratified backtest engine
2. Per-regime threshold optimization
3. Regime-aware walk-forward validation
4. Multi-objective Pareto optimization per archetype-regime pair
5. Regime-conditional portfolio weighting

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Regime-Stratified Backtest Architecture](#2-regime-stratified-backtest-architecture)
3. [Threshold Management](#3-threshold-management)
4. [Optimization Framework](#4-optimization-framework)
5. [Walk-Forward Design](#5-walk-forward-design)
6. [Portfolio Weighting](#6-portfolio-weighting)
7. [Validation Metrics](#7-validation-metrics)
8. [Implementation Roadmap](#8-implementation-roadmap)
9. [Edge Case Handling](#9-edge-case-handling)
10. [Backward Compatibility](#10-backward-compatibility)

---

## 1. Problem Statement

### Current Approach (WRONG)

```python
# Optimization treats entire year as single regime
optimize_s1_on_all_2022()  # Includes bull, neutral, crisis periods
optimize_s4_on_all_2023()  # Includes risk_off periods

# Result: Parameters optimized on WRONG data
# - S1 trained on bars where it shouldn't trade
# - S4 calibrated on regimes where it fails
# - Cross-regime contamination destroys edge
```

### Target Approach (CORRECT)

```python
# Classify all bars by regime
for bar in history:
    bar.regime = classify_regime(bar)  # risk_on, neutral, risk_off, crisis

# Optimize S1 ONLY on crisis + risk_off bars
optimize_s1(bars.filter(regime in ['crisis', 'risk_off']))

# Optimize S4 ONLY on risk_off + neutral bars
optimize_s4(bars.filter(regime in ['risk_off', 'neutral']))

# Result: Parameters tuned to CORRECT market conditions
```

### Mathematical Justification

**Expected Return in Wrong Approach:**
```
E[R] = Σ(weight_regime * PF_regime)
     = 0.40 * PF_risk_on + 0.30 * PF_neutral + 0.25 * PF_risk_off + 0.05 * PF_crisis

Where PF_regime includes contamination from other regimes → BIASED
```

**Expected Return in Correct Approach:**
```
E[R] = Σ(weight_regime * PF_regime | regime)
     = 0.40 * PF_risk_on|risk_on + 0.30 * PF_neutral|neutral + ...

Where PF_regime|regime is unbiased estimate → UNBIASED
```

**Contamination Example:**
- S1 optimized on all 2022 → includes Q1 (risk_on) where S1 shouldn't trade
- Q1 false signals penalize S1 parameters
- Final thresholds too conservative for actual crisis periods
- True crisis edge diluted by 70%

---

## 2. Regime-Stratified Backtest Architecture

### 2.1 Core Data Structure

```python
@dataclass
class RegimeBar:
    """Bar with regime label for stratified backtesting."""
    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float

    # Regime classification
    regime: str  # 'risk_on', 'neutral', 'risk_off', 'crisis'
    regime_confidence: float  # GMM probability [0-1]
    regime_duration: int  # Bars in current regime
    regime_transition: bool  # True if regime changed this bar

    # Feature vectors (unchanged)
    wyckoff_score: float
    liquidity_score: float
    momentum_score: float
    # ... all existing features


@dataclass
class RegimeMetadata:
    """Metadata for a regime period."""
    regime: str
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    duration_bars: int
    n_bars: int

    # Regime characteristics
    avg_volatility: float
    avg_return: float
    max_drawdown: float
    crisis_events: List[str]  # ['LUNA', 'FTX', etc.]
```

### 2.2 Regime Classifier Integration

**Existing Component:** `engine/context/regime_classifier.py` (GMM-based)

**Enhancement: Add Historical Labeling**

```python
class RegimeClassifier:
    """Enhanced regime classifier with historical labeling."""

    def label_historical_bars(
        self,
        bars_df: pd.DataFrame,
        macro_df: pd.DataFrame,
        min_confidence: float = 0.6
    ) -> pd.DataFrame:
        """
        Label all historical bars with regime.

        Args:
            bars_df: OHLCV DataFrame
            macro_df: Macro features (VIX, DXY, funding, etc.)
            min_confidence: Minimum GMM confidence to assign regime

        Returns:
            bars_df with added columns:
                - regime: str
                - regime_confidence: float
                - regime_duration: int
                - regime_transition: bool
        """
        results = []
        current_regime = None
        regime_start_idx = 0

        for idx, (timestamp, bar) in enumerate(bars_df.iterrows()):
            # Get macro features for this timestamp
            macro_row = macro_df.loc[timestamp] if timestamp in macro_df.index else {}

            # Classify regime
            classification = self.classify(macro_row, timestamp=timestamp)
            regime = classification['regime']
            confidence = classification['proba'][regime]

            # Fallback to neutral if low confidence
            if confidence < min_confidence:
                regime = 'neutral'
                confidence = 1.0

            # Detect regime transition
            transition = (regime != current_regime)
            if transition:
                current_regime = regime
                regime_start_idx = idx

            # Calculate regime duration
            duration = idx - regime_start_idx + 1

            results.append({
                'regime': regime,
                'regime_confidence': confidence,
                'regime_duration': duration,
                'regime_transition': transition,
                'regime_proba_risk_on': classification['proba'].get('risk_on', 0.0),
                'regime_proba_neutral': classification['proba'].get('neutral', 0.0),
                'regime_proba_risk_off': classification['proba'].get('risk_off', 0.0),
                'regime_proba_crisis': classification['proba'].get('crisis', 0.0),
            })

        # Merge with bars_df
        result_df = pd.DataFrame(results, index=bars_df.index)
        return pd.concat([bars_df, result_df], axis=1)


    def get_regime_periods(self, labeled_bars: pd.DataFrame) -> List[RegimeMetadata]:
        """
        Extract distinct regime periods from labeled bars.

        Returns:
            List of RegimeMetadata for each contiguous regime period
        """
        periods = []
        current_regime = None
        period_start = None

        for timestamp, row in labeled_bars.iterrows():
            regime = row['regime']

            if regime != current_regime:
                # End previous period
                if current_regime is not None:
                    period_bars = labeled_bars[
                        (labeled_bars.index >= period_start) &
                        (labeled_bars.index < timestamp) &
                        (labeled_bars['regime'] == current_regime)
                    ]

                    metadata = RegimeMetadata(
                        regime=current_regime,
                        start_date=period_start,
                        end_date=timestamp,
                        duration_bars=len(period_bars),
                        n_bars=len(period_bars),
                        avg_volatility=period_bars['close'].pct_change().std() * np.sqrt(252),
                        avg_return=(period_bars['close'].iloc[-1] / period_bars['close'].iloc[0]) - 1,
                        max_drawdown=self._calculate_max_dd(period_bars['close']),
                        crisis_events=self._detect_crisis_events(period_bars)
                    )
                    periods.append(metadata)

                # Start new period
                current_regime = regime
                period_start = timestamp

        # Handle final period
        if current_regime is not None:
            period_bars = labeled_bars[
                (labeled_bars.index >= period_start) &
                (labeled_bars['regime'] == current_regime)
            ]

            metadata = RegimeMetadata(
                regime=current_regime,
                start_date=period_start,
                end_date=labeled_bars.index[-1],
                duration_bars=len(period_bars),
                n_bars=len(period_bars),
                avg_volatility=period_bars['close'].pct_change().std() * np.sqrt(252),
                avg_return=(period_bars['close'].iloc[-1] / period_bars['close'].iloc[0]) - 1,
                max_drawdown=self._calculate_max_dd(period_bars['close']),
                crisis_events=self._detect_crisis_events(period_bars)
            )
            periods.append(metadata)

        return periods
```

### 2.3 Regime-Filtered Backtest Engine

**New Component:** `engine/backtest/regime_aware_backtest.py`

```python
class RegimeAwareBacktest:
    """
    Backtests archetypes on regime-filtered bars only.

    Core principle: If archetype not allowed in regime, skip bar entirely.
    """

    # Archetype-to-regime mappings (from trading logic)
    ARCHETYPE_REGIMES = {
        # Bear archetypes
        'S1': ['risk_off', 'crisis'],  # Capitulation
        'S2': ['risk_off', 'neutral'],  # Failed rally
        'S3': ['risk_off', 'neutral'],  # Whipsaw
        'S4': ['risk_off', 'neutral'],  # Distribution
        'S5': ['risk_on', 'neutral'],   # Long squeeze (counter-trend)
        'S8': ['neutral'],              # Fade chop

        # Bull archetypes
        'A': ['risk_on', 'neutral'],    # Trap reversal
        'B': ['risk_on', 'neutral'],    # Order block
        'C': ['risk_on', 'neutral'],    # FVG continuation
        'G': ['risk_on', 'neutral'],    # Re-accumulate
        'H': ['risk_on', 'neutral'],    # Trap within trend
        'K': ['risk_on', 'neutral'],    # Wick trap
        'L': ['risk_on', 'neutral'],    # Volume exhaustion
    }


    def __init__(self, config: Dict):
        self.config = config
        self.archetype_detector = ArchetypeLogicV2Adapter(config)
        self.regime_classifier = RegimeClassifier.load(
            config['regime']['model_path'],
            config['regime']['feature_order']
        )


    def run(
        self,
        bars_df: pd.DataFrame,
        macro_df: pd.DataFrame,
        archetype: str,
        thresholds: Dict,
        regime_filter: Optional[List[str]] = None
    ) -> BacktestResult:
        """
        Run regime-aware backtest for single archetype.

        Args:
            bars_df: OHLCV data
            macro_df: Macro features for regime classification
            archetype: Archetype ID ('S1', 'S2', etc.)
            thresholds: Archetype-specific thresholds (can be per-regime)
            regime_filter: Optional regime list to filter (default: use ARCHETYPE_REGIMES)

        Returns:
            BacktestResult with regime-stratified metrics
        """
        # 1. Label all bars with regime
        labeled_bars = self.regime_classifier.label_historical_bars(bars_df, macro_df)

        # 2. Determine allowed regimes for this archetype
        if regime_filter is None:
            regime_filter = self.ARCHETYPE_REGIMES.get(archetype, ['risk_on', 'neutral'])

        logger.info(f"Backtesting {archetype} on regimes: {regime_filter}")

        # 3. Filter bars to allowed regimes only
        allowed_mask = labeled_bars['regime'].isin(regime_filter)
        regime_bars = labeled_bars[allowed_mask].copy()

        logger.info(
            f"Filtered {len(labeled_bars)} bars → {len(regime_bars)} bars "
            f"({len(regime_bars)/len(labeled_bars)*100:.1f}% coverage)"
        )

        # 4. Extract regime periods for stratified reporting
        regime_periods = self.regime_classifier.get_regime_periods(labeled_bars)
        allowed_periods = [p for p in regime_periods if p.regime in regime_filter]

        # 5. Run backtest on filtered bars
        trades = []
        signals = []

        for idx, (timestamp, bar) in enumerate(regime_bars.iterrows()):
            regime = bar['regime']

            # Get regime-specific thresholds (if configured)
            regime_thresholds = self._get_regime_thresholds(thresholds, regime)

            # Evaluate archetype with regime-specific thresholds
            signal = self.archetype_detector.evaluate_archetype(
                bar=bar,
                archetype=archetype,
                thresholds=regime_thresholds
            )

            if signal is not None:
                signals.append(signal)

                # Simulate trade execution
                trade = self._simulate_trade(signal, regime_bars, idx)
                if trade is not None:
                    trade['regime'] = regime
                    trade['regime_confidence'] = bar['regime_confidence']
                    trades.append(trade)

        # 6. Calculate regime-stratified metrics
        result = self._calculate_metrics(
            trades=trades,
            signals=signals,
            regime_bars=regime_bars,
            regime_periods=allowed_periods,
            archetype=archetype
        )

        return result


    def _get_regime_thresholds(self, thresholds: Dict, regime: str) -> Dict:
        """
        Get regime-specific thresholds if configured, else use global.

        Threshold structure:
        {
            "fusion_threshold": 0.65,  # Global default
            "crisis_composite_min": 0.35,
            "regime_thresholds": {  # Optional per-regime overrides
                "risk_off": {
                    "fusion_threshold": 0.60,
                    "crisis_composite_min": 0.40
                },
                "crisis": {
                    "fusion_threshold": 0.55,
                    "crisis_composite_min": 0.45
                }
            }
        }
        """
        if 'regime_thresholds' in thresholds and regime in thresholds['regime_thresholds']:
            # Merge regime-specific with global (regime overrides global)
            merged = {**thresholds}
            merged.update(thresholds['regime_thresholds'][regime])
            del merged['regime_thresholds']  # Remove meta-key
            return merged
        else:
            # Return global thresholds
            return {k: v for k, v in thresholds.items() if k != 'regime_thresholds'}


    def _calculate_metrics(
        self,
        trades: List[Dict],
        signals: List[Dict],
        regime_bars: pd.DataFrame,
        regime_periods: List[RegimeMetadata],
        archetype: str
    ) -> BacktestResult:
        """
        Calculate comprehensive regime-stratified metrics.

        Returns:
            BacktestResult with:
                - Overall metrics (all trades)
                - Per-regime metrics (stratified by regime)
                - Regime period analysis
                - Known event capture
        """
        if not trades:
            return BacktestResult(
                archetype=archetype,
                total_trades=0,
                profit_factor=0.0,
                win_rate=0.0,
                expectancy=0.0,
                sharpe=0.0,
                max_drawdown=0.0,
                regime_metrics={},
                error="No trades generated"
            )

        trades_df = pd.DataFrame(trades)

        # Overall metrics
        overall = self._compute_overall_metrics(trades_df)

        # Per-regime metrics
        regime_metrics = {}
        for regime in regime_bars['regime'].unique():
            regime_trades = trades_df[trades_df['regime'] == regime]
            if len(regime_trades) > 0:
                regime_metrics[regime] = self._compute_regime_metrics(
                    regime_trades,
                    regime,
                    regime_periods
                )

        # Known crisis event capture
        crisis_events = self._evaluate_crisis_events(trades_df, regime_periods)

        return BacktestResult(
            archetype=archetype,
            total_trades=len(trades),
            profit_factor=overall['profit_factor'],
            win_rate=overall['win_rate'],
            expectancy=overall['expectancy'],
            sharpe=overall['sharpe'],
            max_drawdown=overall['max_drawdown'],
            avg_pnl_per_trade=overall['avg_pnl_per_trade'],
            total_pnl=overall['total_pnl'],
            regime_metrics=regime_metrics,
            crisis_event_capture=crisis_events,
            regime_coverage={
                regime: len(regime_bars[regime_bars['regime'] == regime])
                for regime in regime_bars['regime'].unique()
            }
        )
```

### 2.4 Regime Transition Handling

**Critical Design Decision:** How to handle regime transitions mid-trade?

**Three Approaches:**

#### Option 1: Ignore Transitions (RECOMMENDED for Phase 1)
```python
# Simplest: Regime determined at entry, never changed
class Trade:
    entry_regime: str  # Fixed at entry

    def exit_logic(self, current_bar):
        # Use original entry_regime for all exit decisions
        return standard_exit_logic(current_bar, self.entry_regime)
```

**Pros:**
- Simple to implement
- Consistent with single-regime optimization
- No mid-trade confusion

**Cons:**
- May hold positions into unfavorable regimes
- Ignores regime change risk

#### Option 2: Regime-Conditional Exits
```python
# Exit if regime changes unfavorably
class Trade:
    entry_regime: str

    def exit_logic(self, current_bar):
        current_regime = current_bar.regime

        # Exit if regime became incompatible
        if current_regime not in ALLOWED_REGIMES[self.archetype]:
            return ExitSignal(
                reason='regime_transition',
                exit_price=current_bar.close,
                exit_type='market'
            )

        # Standard exit logic
        return standard_exit_logic(current_bar, current_regime)
```

**Pros:**
- Respects regime changes
- Prevents holding in wrong regime

**Cons:**
- May exit prematurely (regime noise)
- Adds complexity

#### Option 3: Regime-Adaptive Exits
```python
# Adjust stop/target based on regime
class Trade:
    entry_regime: str

    def exit_logic(self, current_bar):
        current_regime = current_bar.regime

        # If regime transitioned to more favorable → widen stop
        if self._regime_favorability(current_regime) > self._regime_favorability(self.entry_regime):
            self.stop_loss *= 0.9  # Tighter stop
            self.take_profit *= 1.1  # Wider target

        # If regime transitioned to less favorable → tighten stop
        elif self._regime_favorability(current_regime) < self._regime_favorability(self.entry_regime):
            self.stop_loss *= 1.1  # Wider stop (give space)
            self.take_profit *= 0.9  # Tighter target (take profit sooner)

        return standard_exit_logic(current_bar, current_regime)
```

**Pros:**
- Adaptive risk management
- Captures regime shift opportunities

**Cons:**
- Complex to validate
- Requires regime favorability scoring

**RECOMMENDATION:** Use Option 1 (Ignore Transitions) for initial deployment. Add Option 2 (Regime-Conditional Exits) in Phase 3 after validating base framework.

---

## 3. Threshold Management

### 3.1 Config Structure

**Hierarchical Threshold Design:**

```json
{
  "archetypes": {
    "S1": {
      "name": "Capitulation",
      "enabled": true,
      "allowed_regimes": ["risk_off", "crisis"],

      "thresholds": {
        "fusion_threshold": 0.65,
        "capitulation_depth_max": -0.20,
        "crisis_composite_min": 0.35,
        "confluence_threshold": 0.65,
        "cooldown_bars": 12,

        "regime_thresholds": {
          "risk_off": {
            "fusion_threshold": 0.65,
            "capitulation_depth_max": -0.20,
            "crisis_composite_min": 0.35
          },
          "crisis": {
            "fusion_threshold": 0.55,
            "capitulation_depth_max": -0.15,
            "crisis_composite_min": 0.45
          }
        }
      }
    },

    "S4": {
      "name": "Distribution",
      "enabled": true,
      "allowed_regimes": ["risk_off", "neutral"],

      "thresholds": {
        "fusion_threshold": 0.62,
        "distribution_volume_min": 1.5,
        "price_rejection_threshold": 0.03,
        "cooldown_bars": 10,

        "regime_thresholds": {
          "risk_off": {
            "fusion_threshold": 0.60,
            "distribution_volume_min": 1.8
          },
          "neutral": {
            "fusion_threshold": 0.65,
            "distribution_volume_min": 1.3
          }
        }
      }
    }
  }
}
```

### 3.2 Threshold Loading Logic

```python
class ThresholdManager:
    """
    Manages hierarchical threshold loading with regime-specific overrides.
    """

    def __init__(self, config: Dict):
        self.config = config
        self._threshold_cache = {}


    def get_thresholds(
        self,
        archetype: str,
        regime: str,
        use_regime_specific: bool = True
    ) -> Dict:
        """
        Get thresholds for archetype in specific regime.

        Priority:
        1. Regime-specific thresholds (if use_regime_specific=True)
        2. Global archetype thresholds
        3. Default thresholds (fallback)

        Args:
            archetype: Archetype ID ('S1', 'S2', etc.)
            regime: Current regime ('risk_on', 'neutral', etc.)
            use_regime_specific: Whether to use regime overrides

        Returns:
            Dict of threshold parameters
        """
        cache_key = f"{archetype}_{regime}_{use_regime_specific}"

        if cache_key in self._threshold_cache:
            return self._threshold_cache[cache_key]

        # Get archetype config
        if archetype not in self.config['archetypes']:
            raise ValueError(f"Unknown archetype: {archetype}")

        arch_config = self.config['archetypes'][archetype]
        base_thresholds = arch_config.get('thresholds', {})

        # Start with global thresholds
        thresholds = {k: v for k, v in base_thresholds.items()
                      if k != 'regime_thresholds'}

        # Apply regime-specific overrides if available
        if use_regime_specific and 'regime_thresholds' in base_thresholds:
            regime_overrides = base_thresholds['regime_thresholds'].get(regime, {})
            thresholds.update(regime_overrides)

        # Cache and return
        self._threshold_cache[cache_key] = thresholds
        return thresholds


    def validate_thresholds(self, archetype: str) -> List[str]:
        """
        Validate threshold configuration for archetype.

        Returns:
            List of validation warnings/errors
        """
        issues = []

        if archetype not in self.config['archetypes']:
            return [f"Archetype {archetype} not found in config"]

        arch_config = self.config['archetypes'][archetype]
        allowed_regimes = arch_config.get('allowed_regimes', [])
        thresholds = arch_config.get('thresholds', {})
        regime_thresholds = thresholds.get('regime_thresholds', {})

        # Check that all allowed regimes have threshold overrides
        for regime in allowed_regimes:
            if regime not in regime_thresholds:
                issues.append(
                    f"⚠️  {archetype}: No regime-specific thresholds for '{regime}'. "
                    f"Will use global thresholds."
                )

        # Check for unused regime thresholds
        for regime in regime_thresholds.keys():
            if regime not in allowed_regimes:
                issues.append(
                    f"⚠️  {archetype}: Regime thresholds defined for '{regime}' "
                    f"but archetype not allowed in this regime. Unused."
                )

        # Check for missing required parameters
        required_params = self._get_required_params(archetype)
        for param in required_params:
            if param not in thresholds and param != 'regime_thresholds':
                issues.append(
                    f"❌ {archetype}: Missing required threshold parameter '{param}'"
                )

        return issues
```

### 3.3 Threshold Constraints

**Design Decision:** Should we constrain regime-specific thresholds to be within bounds of global thresholds?

**Option A: Unconstrained (RECOMMENDED)**
- Regime thresholds can differ arbitrarily from global
- Example: Crisis fusion_threshold = 0.45, Risk_off = 0.65
- Pros: Maximum flexibility
- Cons: Potential for overfitting

**Option B: Bounded Deviation**
- Regime thresholds must be within ±15% of global
- Example: If global = 0.60, regime must be in [0.51, 0.69]
- Pros: Prevents extreme overfitting
- Cons: May limit regime adaptivity

**Option C: Monotonic Constraint**
- Crisis < Risk_off < Neutral < Risk_on (for entry thresholds)
- Enforces intuition: Lower bar to enter in crisis
- Pros: Interpretable, prevents nonsense
- Cons: May be too restrictive

**RECOMMENDATION:** Use Option A (Unconstrained) initially. Add Option C (Monotonic Constraint) as validation check, not hard constraint.

---

## 4. Optimization Framework

### 4.1 Multi-Objective Optimization per Regime

**Objective Function Design:**

```python
class RegimeAwareObjective:
    """
    Multi-objective optimization for archetype within single regime.
    """

    def __init__(
        self,
        archetype: str,
        regime: str,
        bars_df: pd.DataFrame,
        macro_df: pd.DataFrame,
        config: Dict
    ):
        self.archetype = archetype
        self.regime = regime
        self.backtest_engine = RegimeAwareBacktest(config)

        # Filter bars to this regime only
        labeled_bars = RegimeClassifier.load(...).label_historical_bars(bars_df, macro_df)
        self.regime_bars = labeled_bars[labeled_bars['regime'] == regime].copy()

        logger.info(
            f"Optimizing {archetype} on {regime}: "
            f"{len(self.regime_bars)} bars ({len(self.regime_bars)/len(labeled_bars)*100:.1f}%)"
        )


    def __call__(self, trial: optuna.Trial) -> Tuple[float, float, float]:
        """
        Multi-objective function: (PF, Event Recall, Trade Frequency).

        Returns:
            Tuple of (negative_pf, negative_recall, trades_per_year)
            All negated for minimization
        """
        # Suggest thresholds for this regime
        thresholds = self._suggest_thresholds(trial)

        # Run backtest on regime-filtered bars
        result = self.backtest_engine.run(
            bars_df=self.regime_bars,
            macro_df=self.macro_df,
            archetype=self.archetype,
            thresholds=thresholds,
            regime_filter=[self.regime]
        )

        # Objective 1: Maximize Profit Factor
        pf = result.profit_factor if result.total_trades >= 5 else 0.0

        # Objective 2: Maximize Event Recall (regime-specific)
        recall = self._calculate_event_recall(result, self.regime)

        # Objective 3: Minimize Trade Frequency (avoid overtrading)
        years = (self.regime_bars.index[-1] - self.regime_bars.index[0]).days / 365.25
        trades_per_year = result.total_trades / years if years > 0 else 999

        # Return tuple (all minimization)
        return (
            -pf,           # Minimize negative PF = Maximize PF
            -recall,       # Minimize negative recall = Maximize recall
            trades_per_year  # Minimize trades/year
        )


    def _suggest_thresholds(self, trial: optuna.Trial) -> Dict:
        """
        Suggest regime-specific thresholds for trial.

        Archetype-specific search spaces:
        - S1: crisis_composite_min, capitulation_depth_max, fusion_threshold
        - S4: distribution_volume_min, price_rejection_threshold, fusion_threshold
        - S5: funding_z_min, liquidity_max, rsi_exhaustion_min
        """
        if self.archetype == 'S1':
            return {
                'fusion_threshold': trial.suggest_float(
                    'fusion_threshold', 0.50, 0.75, step=0.01
                ),
                'crisis_composite_min': trial.suggest_float(
                    'crisis_composite_min', 0.25, 0.50, step=0.01
                ),
                'capitulation_depth_max': trial.suggest_float(
                    'capitulation_depth_max', -0.30, -0.10, step=0.01
                ),
                'confluence_threshold': trial.suggest_float(
                    'confluence_threshold', 0.50, 0.80, step=0.01
                ),
                'cooldown_bars': trial.suggest_int(
                    'cooldown_bars', 8, 20, step=2
                )
            }

        elif self.archetype == 'S4':
            return {
                'fusion_threshold': trial.suggest_float(
                    'fusion_threshold', 0.55, 0.75, step=0.01
                ),
                'distribution_volume_min': trial.suggest_float(
                    'distribution_volume_min', 1.0, 2.5, step=0.1
                ),
                'price_rejection_threshold': trial.suggest_float(
                    'price_rejection_threshold', 0.01, 0.05, step=0.005
                ),
                'cooldown_bars': trial.suggest_int(
                    'cooldown_bars', 6, 16, step=2
                )
            }

        elif self.archetype == 'S5':
            return {
                'fusion_threshold': trial.suggest_float(
                    'fusion_threshold', 0.45, 0.70, step=0.01
                ),
                'funding_z_min': trial.suggest_float(
                    'funding_z_min', 1.0, 2.5, step=0.1
                ),
                'liquidity_max': trial.suggest_float(
                    'liquidity_max', 0.10, 0.40, step=0.02
                ),
                'rsi_exhaustion_min': trial.suggest_float(
                    'rsi_exhaustion_min', 65.0, 80.0, step=1.0
                ),
                'cooldown_bars': trial.suggest_int(
                    'cooldown_bars', 4, 12, step=2
                )
            }

        else:
            raise ValueError(f"Unknown archetype: {self.archetype}")


    def _calculate_event_recall(self, result: BacktestResult, regime: str) -> float:
        """
        Calculate recall of known crisis events for this regime.

        Known events:
        - LUNA: 2022-05-09 to 2022-05-12 (crisis)
        - June 18: 2022-06-18 to 2022-06-19 (crisis)
        - FTX: 2022-11-08 to 2022-11-10 (crisis)
        - SVB: 2023-03-10 to 2023-03-12 (risk_off)
        """
        if regime == 'crisis':
            known_events = [
                ('LUNA', pd.Timestamp('2022-05-09'), pd.Timestamp('2022-05-12')),
                ('June18', pd.Timestamp('2022-06-18'), pd.Timestamp('2022-06-19')),
                ('FTX', pd.Timestamp('2022-11-08'), pd.Timestamp('2022-11-10')),
            ]
        elif regime == 'risk_off':
            known_events = [
                ('SVB', pd.Timestamp('2023-03-10'), pd.Timestamp('2023-03-12')),
                # Add more risk_off events
            ]
        else:
            return 0.0  # No known events for this regime

        # Check if any trades captured events
        trades_df = pd.DataFrame(result.trades) if result.trades else pd.DataFrame()
        if trades_df.empty:
            return 0.0

        captured = 0
        for event_name, start, end in known_events:
            # Check if any trade entry falls within event window
            event_trades = trades_df[
                (trades_df['entry_timestamp'] >= start) &
                (trades_df['entry_timestamp'] <= end)
            ]
            if len(event_trades) > 0:
                captured += 1

        recall = captured / len(known_events) if known_events else 0.0
        return recall
```

### 4.2 Optuna Study Setup

```python
def optimize_archetype_regime(
    archetype: str,
    regime: str,
    bars_df: pd.DataFrame,
    macro_df: pd.DataFrame,
    config: Dict,
    n_trials: int = 200,
    storage: str = "sqlite:///optuna_regime_aware.db"
) -> optuna.Study:
    """
    Run multi-objective optimization for archetype in specific regime.

    Args:
        archetype: Archetype ID ('S1', 'S2', etc.)
        regime: Regime label ('risk_on', 'neutral', 'risk_off', 'crisis')
        bars_df: Historical OHLCV data
        macro_df: Macro features for regime classification
        config: Base config dict
        n_trials: Number of Optuna trials
        storage: SQLite database path

    Returns:
        Optuna study with Pareto frontier
    """
    study_name = f"{archetype}_{regime}_regime_aware"

    # Create multi-objective study
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        directions=['minimize', 'minimize', 'minimize'],  # (neg_pf, neg_recall, trades/yr)
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=20),
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=1,
            max_resource=10,
            reduction_factor=3
        )
    )

    # Create objective function
    objective = RegimeAwareObjective(
        archetype=archetype,
        regime=regime,
        bars_df=bars_df,
        macro_df=macro_df,
        config=config
    )

    # Optimize
    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=1,
        show_progress_bar=True,
        callbacks=[
            optuna.logging.StdOutLogger(level=logging.INFO),
            RegimeAwareProgressCallback()
        ]
    )

    # Extract Pareto frontier
    pareto_trials = study.best_trials

    logger.info(f"✅ Optimization complete for {archetype} in {regime}")
    logger.info(f"   Total trials: {len(study.trials)}")
    logger.info(f"   Pareto frontier: {len(pareto_trials)} solutions")

    # Display Pareto frontier
    print("\n" + "="*80)
    print(f"PARETO FRONTIER: {archetype} in {regime}")
    print("="*80)
    print(f"{'Trial':>6} {'PF':>8} {'Recall':>8} {'Trades/Yr':>10} {'fusion_th':>10}")
    print("-"*80)

    for trial in pareto_trials[:10]:  # Top 10
        pf = -trial.values[0]
        recall = -trial.values[1]
        trades_yr = trial.values[2]
        fusion_th = trial.params.get('fusion_threshold', 0.0)

        print(f"{trial.number:>6} {pf:>8.2f} {recall:>8.1%} {trades_yr:>10.1f} {fusion_th:>10.3f}")

    print("="*80 + "\n")

    return study


def optimize_all_regime_pairs(
    archetypes: List[str],
    bars_df: pd.DataFrame,
    macro_df: pd.DataFrame,
    config: Dict,
    n_trials_per_pair: int = 200,
    parallel: bool = True
) -> Dict[Tuple[str, str], optuna.Study]:
    """
    Optimize all archetype-regime pairs in parallel.

    Args:
        archetypes: List of archetype IDs to optimize
        bars_df: Historical OHLCV data
        macro_df: Macro features
        config: Base config
        n_trials_per_pair: Trials per archetype-regime pair
        parallel: Whether to run pairs in parallel

    Returns:
        Dict mapping (archetype, regime) -> optuna.Study
    """
    from engine.backtest.regime_aware_backtest import RegimeAwareBacktest

    # Determine archetype-regime pairs
    pairs = []
    for archetype in archetypes:
        allowed_regimes = RegimeAwareBacktest.ARCHETYPE_REGIMES.get(
            archetype,
            ['risk_on', 'neutral']
        )
        for regime in allowed_regimes:
            pairs.append((archetype, regime))

    logger.info(f"Optimizing {len(pairs)} archetype-regime pairs:")
    for arch, reg in pairs:
        logger.info(f"  - {arch} in {reg}")

    studies = {}

    if parallel:
        # Parallel execution using multiprocessing
        import multiprocessing as mp

        with mp.Pool(processes=min(len(pairs), mp.cpu_count())) as pool:
            results = []

            for archetype, regime in pairs:
                result = pool.apply_async(
                    optimize_archetype_regime,
                    args=(archetype, regime, bars_df, macro_df, config, n_trials_per_pair)
                )
                results.append(((archetype, regime), result))

            for (archetype, regime), result in results:
                try:
                    study = result.get(timeout=3600)  # 1 hour per pair
                    studies[(archetype, regime)] = study
                except Exception as e:
                    logger.error(f"Failed to optimize {archetype} in {regime}: {e}")

    else:
        # Sequential execution
        for archetype, regime in pairs:
            try:
                study = optimize_archetype_regime(
                    archetype, regime, bars_df, macro_df, config, n_trials_per_pair
                )
                studies[(archetype, regime)] = study
            except Exception as e:
                logger.error(f"Failed to optimize {archetype} in {regime}: {e}")

    return studies
```

### 4.3 Pareto Selection Strategy

**After optimization, how to select threshold from Pareto frontier?**

```python
class ParetoSelector:
    """
    Select best threshold configuration from Pareto frontier based on preferences.
    """

    @staticmethod
    def select_balanced(pareto_trials: List[optuna.Trial]) -> optuna.Trial:
        """
        Select balanced solution: good PF, good recall, reasonable trades.

        Scoring:
            score = w_pf * normalize(PF) + w_recall * normalize(recall) - w_trades * normalize(trades)
        """
        if not pareto_trials:
            raise ValueError("No Pareto trials available")

        # Extract objectives
        pfs = np.array([-t.values[0] for t in pareto_trials])
        recalls = np.array([-t.values[1] for t in pareto_trials])
        trades = np.array([t.values[2] for t in pareto_trials])

        # Normalize to [0, 1]
        pfs_norm = (pfs - pfs.min()) / (pfs.max() - pfs.min() + 1e-8)
        recalls_norm = (recalls - recalls.min()) / (recalls.max() - recalls.min() + 1e-8)
        trades_norm = (trades - trades.min()) / (trades.max() - trades.min() + 1e-8)

        # Weighted score (tunable)
        w_pf = 0.50
        w_recall = 0.30
        w_trades = 0.20

        scores = w_pf * pfs_norm + w_recall * recalls_norm - w_trades * trades_norm

        best_idx = np.argmax(scores)
        return pareto_trials[best_idx]


    @staticmethod
    def select_conservative(pareto_trials: List[optuna.Trial]) -> optuna.Trial:
        """
        Select conservative solution: highest PF, ignore recall, minimize trades.
        """
        # Sort by PF descending, then trades ascending
        sorted_trials = sorted(
            pareto_trials,
            key=lambda t: (-t.values[0], t.values[2])  # (PF desc, trades asc)
        )
        return sorted_trials[0]


    @staticmethod
    def select_event_focused(pareto_trials: List[optuna.Trial]) -> optuna.Trial:
        """
        Select event-focused solution: maximize recall, require PF > 1.5.
        """
        # Filter to PF > 1.5
        viable = [t for t in pareto_trials if -t.values[0] > 1.5]

        if not viable:
            # Fallback: best PF
            return sorted(pareto_trials, key=lambda t: t.values[0])[0]

        # Select highest recall among viable
        return sorted(viable, key=lambda t: t.values[1])[0]
```

---

## 5. Walk-Forward Design

### 5.1 Regime-Aware Walk-Forward Windows

**Standard Walk-Forward (WRONG):**
```
Train: Q1-Q3 2022 (all bars) → Optimize
Test:  Q4 2022 (all bars) → Validate

Problem: Train may have 60% risk_on, 40% risk_off
         Optimizes on contaminated data
```

**Regime-Aware Walk-Forward (CORRECT):**
```
Train: Q1-Q3 2022 (risk_off bars only) → Optimize S1 on risk_off
Test:  Q4 2022 (risk_off bars only) → Validate S1 on risk_off

Train: Q1-Q3 2022 (crisis bars only) → Optimize S1 on crisis
Test:  Q4 2022 (crisis bars only) → Validate S1 on crisis
```

### 5.2 Implementation

```python
@dataclass
class RegimeWindow:
    """Walk-forward window with regime stratification."""
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    # Regime-stratified bar counts
    train_bars_by_regime: Dict[str, int]
    test_bars_by_regime: Dict[str, int]

    # Sufficient data checks
    sufficient_train_data: Dict[str, bool]  # Per regime
    sufficient_test_data: Dict[str, bool]

    # Window metadata
    window_id: int
    total_duration_days: int


class RegimeAwareWalkForward:
    """
    Walk-forward validation with regime stratification.
    """

    def __init__(
        self,
        train_months: int = 12,
        test_months: int = 3,
        step_months: int = 1,
        min_train_bars_per_regime: int = 200,
        min_test_bars_per_regime: int = 50
    ):
        self.train_months = train_months
        self.test_months = test_months
        self.step_months = step_months
        self.min_train_bars = min_train_bars_per_regime
        self.min_test_bars = min_test_bars_per_regime


    def generate_windows(
        self,
        bars_df: pd.DataFrame,
        macro_df: pd.DataFrame,
        archetype: str
    ) -> List[RegimeWindow]:
        """
        Generate regime-aware walk-forward windows.

        Args:
            bars_df: Historical OHLCV data
            macro_df: Macro features for regime classification
            archetype: Archetype ID (determines allowed regimes)

        Returns:
            List of RegimeWindow objects with sufficient regime data
        """
        # 1. Label all bars with regime
        regime_classifier = RegimeClassifier.load(...)
        labeled_bars = regime_classifier.label_historical_bars(bars_df, macro_df)

        # 2. Determine allowed regimes for archetype
        allowed_regimes = RegimeAwareBacktest.ARCHETYPE_REGIMES.get(
            archetype,
            ['risk_on', 'neutral']
        )

        # 3. Generate candidate windows
        start_date = labeled_bars.index[0]
        end_date = labeled_bars.index[-1]

        windows = []
        window_id = 0
        current_date = start_date + pd.DateOffset(months=self.train_months)

        while current_date + pd.DateOffset(months=self.test_months) <= end_date:
            train_start = current_date - pd.DateOffset(months=self.train_months)
            train_end = current_date
            test_start = current_date
            test_end = current_date + pd.DateOffset(months=self.test_months)

            # Filter to train/test periods
            train_mask = (labeled_bars.index >= train_start) & (labeled_bars.index < train_end)
            test_mask = (labeled_bars.index >= test_start) & (labeled_bars.index < test_end)

            train_bars = labeled_bars[train_mask]
            test_bars = labeled_bars[test_mask]

            # Count bars per regime
            train_bars_by_regime = {
                regime: len(train_bars[train_bars['regime'] == regime])
                for regime in allowed_regimes
            }

            test_bars_by_regime = {
                regime: len(test_bars[test_bars['regime'] == regime])
                for regime in allowed_regimes
            }

            # Check sufficiency
            sufficient_train = {
                regime: count >= self.min_train_bars
                for regime, count in train_bars_by_regime.items()
            }

            sufficient_test = {
                regime: count >= self.min_test_bars
                for regime, count in test_bars_by_regime.items()
            }

            # Create window if at least one regime has sufficient data
            if any(sufficient_train.values()) and any(sufficient_test.values()):
                window = RegimeWindow(
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    train_bars_by_regime=train_bars_by_regime,
                    test_bars_by_regime=test_bars_by_regime,
                    sufficient_train_data=sufficient_train,
                    sufficient_test_data=sufficient_test,
                    window_id=window_id,
                    total_duration_days=(test_end - train_start).days
                )
                windows.append(window)
                window_id += 1

            # Step forward
            current_date += pd.DateOffset(months=self.step_months)

        logger.info(f"Generated {len(windows)} walk-forward windows for {archetype}")

        return windows


    def validate_archetype_regime(
        self,
        archetype: str,
        regime: str,
        bars_df: pd.DataFrame,
        macro_df: pd.DataFrame,
        config: Dict
    ) -> WalkForwardResult:
        """
        Run regime-aware walk-forward validation for archetype in specific regime.

        Process:
        1. Generate windows
        2. For each window:
           a. Optimize on train regime bars
           b. Validate on test regime bars
        3. Aggregate out-of-sample metrics

        Returns:
            WalkForwardResult with OOS performance per window
        """
        # Generate windows
        windows = self.generate_windows(bars_df, macro_df, archetype)

        # Filter windows with sufficient data for this regime
        valid_windows = [
            w for w in windows
            if w.sufficient_train_data.get(regime, False) and
               w.sufficient_test_data.get(regime, False)
        ]

        logger.info(
            f"Validating {archetype} in {regime}: "
            f"{len(valid_windows)}/{len(windows)} windows have sufficient data"
        )

        if not valid_windows:
            return WalkForwardResult(
                archetype=archetype,
                regime=regime,
                windows=[],
                error="No windows with sufficient regime data"
            )

        # Run validation for each window
        window_results = []

        for window in valid_windows:
            logger.info(
                f"Window {window.window_id}: "
                f"Train {window.train_start.date()} - {window.train_end.date()}, "
                f"Test {window.test_start.date()} - {window.test_end.date()}"
            )

            # Extract train/test bars for this regime
            labeled_bars = RegimeClassifier.load(...).label_historical_bars(bars_df, macro_df)

            train_mask = (
                (labeled_bars.index >= window.train_start) &
                (labeled_bars.index < window.train_end) &
                (labeled_bars['regime'] == regime)
            )
            test_mask = (
                (labeled_bars.index >= window.test_start) &
                (labeled_bars.index < window.test_end) &
                (labeled_bars['regime'] == regime)
            )

            train_bars = labeled_bars[train_mask]
            test_bars = labeled_bars[test_mask]

            logger.info(
                f"  {regime}: {len(train_bars)} train bars, {len(test_bars)} test bars"
            )

            # Optimize on train
            study = optimize_archetype_regime(
                archetype=archetype,
                regime=regime,
                bars_df=train_bars,
                macro_df=macro_df,
                config=config,
                n_trials=50,  # Reduced for walk-forward
                storage=f"sqlite:///optuna_wf_{archetype}_{regime}_w{window.window_id}.db"
            )

            # Select best thresholds (Pareto)
            best_trial = ParetoSelector.select_balanced(study.best_trials)
            best_thresholds = best_trial.params

            # Validate on test (out-of-sample)
            backtest_engine = RegimeAwareBacktest(config)
            test_result = backtest_engine.run(
                bars_df=test_bars,
                macro_df=macro_df,
                archetype=archetype,
                thresholds=best_thresholds,
                regime_filter=[regime]
            )

            # Store result
            window_results.append({
                'window_id': window.window_id,
                'train_start': window.train_start,
                'train_end': window.train_end,
                'test_start': window.test_start,
                'test_end': window.test_end,
                'train_bars': len(train_bars),
                'test_bars': len(test_bars),
                'best_thresholds': best_thresholds,
                'test_pf': test_result.profit_factor,
                'test_win_rate': test_result.win_rate,
                'test_trades': test_result.total_trades,
                'test_sharpe': test_result.sharpe,
                'test_max_dd': test_result.max_drawdown,
            })

        # Aggregate OOS metrics
        oos_metrics = self._aggregate_oos_metrics(window_results)

        return WalkForwardResult(
            archetype=archetype,
            regime=regime,
            windows=window_results,
            oos_metrics=oos_metrics,
            n_windows=len(window_results),
            avg_oos_pf=oos_metrics['avg_pf'],
            avg_oos_sharpe=oos_metrics['avg_sharpe'],
            consistency_score=oos_metrics['consistency_score']
        )


    def _aggregate_oos_metrics(self, window_results: List[Dict]) -> Dict:
        """
        Aggregate out-of-sample metrics across windows.

        Returns:
            Dict with:
                - avg_pf: Mean profit factor across windows
                - std_pf: Std dev of profit factors (consistency)
                - avg_sharpe: Mean Sharpe ratio
                - consistency_score: 1 - (CV of PF)
                - positive_windows: Fraction of windows with PF > 1.0
        """
        if not window_results:
            return {}

        pfs = [w['test_pf'] for w in window_results]
        sharpes = [w['test_sharpe'] for w in window_results]

        avg_pf = np.mean(pfs)
        std_pf = np.std(pfs)
        cv_pf = std_pf / avg_pf if avg_pf > 0 else 999

        return {
            'avg_pf': avg_pf,
            'std_pf': std_pf,
            'cv_pf': cv_pf,
            'avg_sharpe': np.mean(sharpes),
            'std_sharpe': np.std(sharpes),
            'consistency_score': 1.0 - min(cv_pf, 1.0),
            'positive_windows': sum(1 for pf in pfs if pf > 1.0) / len(pfs),
            'n_windows': len(window_results)
        }
```

### 5.3 Empty Window Handling

**Problem:** What if test window has no regime bars?

**Example:**
- Optimizing S1 (crisis archetype)
- Test window: Q1 2023 (all risk_on, no crisis)
- Test result: 0 bars → Cannot validate

**Solutions:**

#### Option 1: Skip Window (RECOMMENDED)
```python
if len(test_bars) < min_test_bars:
    logger.warning(f"Skipping window {window_id}: Insufficient {regime} bars in test")
    continue
```

**Pros:** Clean, no contamination
**Cons:** May skip many windows

#### Option 2: Use Neutral as Proxy
```python
if regime == 'risk_off' and len(test_bars) < min_test_bars:
    # Fallback to neutral regime (adjacent)
    test_bars = labeled_bars[
        (labeled_bars.index >= test_start) &
        (labeled_bars.index < test_end) &
        (labeled_bars['regime'] == 'neutral')
    ]
    logger.info(f"Using neutral as proxy for risk_off in window {window_id}")
```

**Pros:** More windows validated
**Cons:** Not true OOS, may be optimistic

#### Option 3: Aggregate Adjacent Windows
```python
if len(test_bars) < min_test_bars:
    # Combine with next window's test period
    test_bars = labeled_bars[
        (labeled_bars.index >= test_start) &
        (labeled_bars.index < test_start + 2 * test_period) &
        (labeled_bars['regime'] == regime)
    ]
```

**Pros:** Uses real regime data
**Cons:** Overlapping test periods, not independent

**RECOMMENDATION:** Use Option 1 (Skip Window). Accept that some regimes (crisis) will have fewer validation windows. This is statistically honest.

---

## 6. Portfolio Weighting

### 6.1 Regime Distribution Analysis

**Historical Regime Distribution (Example 2022-2023):**

```python
def analyze_regime_distribution(labeled_bars: pd.DataFrame) -> Dict:
    """
    Analyze historical regime distribution.

    Returns:
        Dict with:
            - regime_fractions: % of bars in each regime
            - regime_durations: Avg consecutive bars per regime
            - transition_matrix: P(regime_t | regime_t-1)
    """
    regime_counts = labeled_bars['regime'].value_counts()
    total_bars = len(labeled_bars)

    regime_fractions = {
        regime: count / total_bars
        for regime, count in regime_counts.items()
    }

    # Calculate average regime durations
    regime_durations = {}
    for regime in labeled_bars['regime'].unique():
        regime_mask = labeled_bars['regime'] == regime
        regime_runs = (regime_mask != regime_mask.shift()).cumsum()[regime_mask]
        avg_duration = regime_runs.value_counts().mean()
        regime_durations[regime] = avg_duration

    # Transition matrix
    transitions = pd.crosstab(
        labeled_bars['regime'],
        labeled_bars['regime'].shift(-1),
        normalize='index'
    )

    return {
        'regime_fractions': regime_fractions,
        'regime_durations': regime_durations,
        'transition_matrix': transitions.to_dict(),
        'total_bars': total_bars
    }


# Example output (2022-2023 BTC):
regime_distribution = {
    'risk_on': 0.42,     # 42% of bars
    'neutral': 0.28,     # 28% of bars
    'risk_off': 0.25,    # 25% of bars
    'crisis': 0.05       # 5% of bars
}

regime_durations = {
    'risk_on': 18.5,     # Avg 18.5 consecutive bars
    'neutral': 12.3,
    'risk_off': 15.7,
    'crisis': 8.2        # Crisis short-lived
}
```

### 6.2 Archetype Performance by Regime

**After Walk-Forward Validation:**

```python
archetype_performance = {
    'S1': {
        'risk_off': {'pf': 2.3, 'sharpe': 1.2, 'trades_per_year': 8},
        'crisis':   {'pf': 3.5, 'sharpe': 1.8, 'trades_per_year': 4}
    },
    'S2': {
        'risk_off': {'pf': 1.8, 'sharpe': 0.9, 'trades_per_year': 12},
        'neutral':  {'pf': 1.5, 'sharpe': 0.6, 'trades_per_year': 15}
    },
    'S4': {
        'risk_off': {'pf': 2.1, 'sharpe': 1.0, 'trades_per_year': 10},
        'neutral':  {'pf': 1.7, 'sharpe': 0.8, 'trades_per_year': 18}
    },
    'S5': {
        'risk_on':  {'pf': 2.0, 'sharpe': 1.1, 'trades_per_year': 14},
        'neutral':  {'pf': 1.6, 'sharpe': 0.7, 'trades_per_year': 20}
    }
}
```

### 6.3 Portfolio Weight Calculation

**Three Weighting Schemes:**

#### Scheme 1: Equal Regime Weight (Naive)
```python
def calculate_equal_weights(archetype_performance: Dict) -> Dict:
    """
    Equal weight per archetype-regime pair.

    Problem: Ignores regime frequency and performance.
    """
    pairs = []
    for archetype, regimes in archetype_performance.items():
        for regime in regimes.keys():
            pairs.append((archetype, regime))

    weight_per_pair = 1.0 / len(pairs)

    weights = {pair: weight_per_pair for pair in pairs}
    return weights
```

#### Scheme 2: Regime-Weighted (RECOMMENDED for Phase 1)
```python
def calculate_regime_weighted(
    archetype_performance: Dict,
    regime_distribution: Dict
) -> Dict:
    """
    Weight by regime frequency × archetype performance.

    Formula:
        weight(arch, regime) = regime_fraction * PF(arch, regime) * risk_adj
        Normalize to sum to 1.0
    """
    weights = {}

    for archetype, regimes in archetype_performance.items():
        for regime, perf in regimes.items():
            # Base weight: regime frequency
            regime_freq = regime_distribution['regime_fractions'].get(regime, 0.0)

            # Performance multiplier
            pf = perf['pf']
            sharpe = perf.get('sharpe', 0.5)

            # Risk adjustment (penalize low Sharpe)
            risk_adj = min(sharpe / 0.8, 1.5)  # Cap at 1.5x

            # Combined weight
            raw_weight = regime_freq * pf * risk_adj

            weights[(archetype, regime)] = raw_weight

    # Normalize to sum to 1.0
    total_weight = sum(weights.values())
    normalized_weights = {k: v / total_weight for k, v in weights.items()}

    return normalized_weights


# Example output:
weights = {
    ('S1', 'risk_off'): 0.18,  # 18% allocation
    ('S1', 'crisis'):   0.12,  # 12% (high PF but rare regime)
    ('S2', 'risk_off'): 0.14,
    ('S2', 'neutral'):  0.11,
    ('S4', 'risk_off'): 0.16,
    ('S4', 'neutral'):  0.13,
    ('S5', 'risk_on'):  0.10,
    ('S5', 'neutral'):  0.06
}
```

#### Scheme 3: Kelly Criterion (Advanced)
```python
def calculate_kelly_weights(
    archetype_performance: Dict,
    regime_distribution: Dict,
    transition_matrix: Dict
) -> Dict:
    """
    Kelly criterion weights accounting for regime transitions.

    Formula:
        f* = (p * b - q) / b

    Where:
        p = win rate
        q = 1 - p
        b = avg_win / avg_loss
        f* = Kelly fraction

    Adjusted for:
        - Regime transition probabilities
        - Risk aversion (use fractional Kelly)
    """
    kelly_fractions = {}

    for archetype, regimes in archetype_performance.items():
        for regime, perf in regimes.items():
            # Kelly calculation
            wr = perf.get('win_rate', 0.5)
            pf = perf['pf']

            # Estimate avg_win/avg_loss from PF and WR
            # PF = (wr * avg_win) / ((1-wr) * avg_loss)
            # Assume avg_loss = 1, solve for avg_win
            avg_win = pf * (1 - wr) / wr if wr > 0 else 1.0

            p = wr
            q = 1 - wr
            b = avg_win  # avg_loss = 1

            kelly_f = (p * b - q) / b if b > 0 else 0.0

            # Fractional Kelly (0.25x for safety)
            fractional_kelly = 0.25 * max(kelly_f, 0.0)

            # Adjust for regime frequency
            regime_freq = regime_distribution['regime_fractions'].get(regime, 0.0)
            adjusted_f = fractional_kelly * regime_freq

            kelly_fractions[(archetype, regime)] = adjusted_f

    # Normalize
    total_f = sum(kelly_fractions.values())
    normalized = {k: v / total_f for k, v in kelly_fractions.items()} if total_f > 0 else {}

    return normalized
```

**RECOMMENDATION:** Use Scheme 2 (Regime-Weighted) for initial deployment. Validate with Scheme 3 (Kelly) in simulation before live.

### 6.4 Dynamic Weight Adjustment

**Problem:** Regime distribution shifts over time (bear → bull transition).

**Solution: Rolling Regime Estimation**

```python
class DynamicWeightAdjuster:
    """
    Dynamically adjust archetype weights based on rolling regime forecast.
    """

    def __init__(
        self,
        lookback_days: int = 90,
        update_frequency: str = 'weekly'
    ):
        self.lookback_days = lookback_days
        self.update_frequency = update_frequency
        self.regime_forecaster = RegimeForecaster()


    def adjust_weights(
        self,
        current_date: pd.Timestamp,
        labeled_bars: pd.DataFrame,
        base_weights: Dict
    ) -> Dict:
        """
        Adjust weights based on recent regime distribution.

        Process:
        1. Calculate rolling regime distribution (last 90 days)
        2. Forecast next 30-day regime distribution
        3. Re-weight archetypes based on forecast

        Args:
            current_date: Current timestamp
            labeled_bars: Historical bars with regime labels
            base_weights: Static weights from optimization

        Returns:
            Adjusted weights dict
        """
        # 1. Rolling regime distribution
        lookback_start = current_date - pd.Timedelta(days=self.lookback_days)
        recent_bars = labeled_bars[
            (labeled_bars.index >= lookback_start) &
            (labeled_bars.index <= current_date)
        ]

        recent_regime_dist = recent_bars['regime'].value_counts(normalize=True).to_dict()

        # 2. Forecast next 30 days (using transition matrix)
        forecast_dist = self.regime_forecaster.forecast(
            current_regime=recent_bars['regime'].iloc[-1],
            transition_matrix=self._estimate_transition_matrix(recent_bars),
            horizon_days=30
        )

        # 3. Adjust weights
        adjusted_weights = {}

        for (archetype, regime), base_weight in base_weights.items():
            # Adjustment factor: forecast / historical average
            forecast_freq = forecast_dist.get(regime, 0.0)
            historical_freq = recent_regime_dist.get(regime, 0.0)

            if historical_freq > 0:
                adjustment_factor = forecast_freq / historical_freq
            else:
                adjustment_factor = 1.0

            # Apply bounded adjustment (max 2x, min 0.5x)
            bounded_adjustment = np.clip(adjustment_factor, 0.5, 2.0)

            adjusted_weights[(archetype, regime)] = base_weight * bounded_adjustment

        # Normalize
        total = sum(adjusted_weights.values())
        normalized = {k: v / total for k, v in adjusted_weights.items()} if total > 0 else {}

        logger.info(f"Dynamic weight adjustment at {current_date.date()}:")
        logger.info(f"  Recent regime dist: {recent_regime_dist}")
        logger.info(f"  Forecast dist: {forecast_dist}")
        logger.info(f"  Weight changes: {self._summarize_changes(base_weights, normalized)}")

        return normalized
```

---

## 7. Validation Metrics

### 7.1 Regime-Stratified Performance Metrics

**Core Metrics per Archetype-Regime Pair:**

```python
@dataclass
class RegimeStratifiedMetrics:
    """Comprehensive metrics for archetype performance in specific regime."""

    # Identifiers
    archetype: str
    regime: str

    # Trade statistics
    total_trades: int
    trades_per_year: float

    # Profitability
    profit_factor: float
    win_rate: float
    expectancy: float  # Avg PnL per trade
    total_pnl: float

    # Risk metrics
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float  # Return / MaxDD

    # Regime-specific
    regime_coverage: float  # % of regime bars with position
    regime_event_recall: float  # % of known events captured
    avg_bars_held: float

    # Consistency
    win_streak_max: int
    loss_streak_max: int
    pf_by_month: Dict[str, float]  # Monthly PF for consistency check

    # Thresholds used
    thresholds: Dict[str, float]
```

### 7.2 Cross-Regime Consistency Checks

**Validation: Ensure regime-specific parameters make sense**

```python
def validate_regime_consistency(
    archetype: str,
    regime_metrics: Dict[str, RegimeStratifiedMetrics]
) -> List[str]:
    """
    Validate cross-regime parameter consistency.

    Checks:
    1. Monotonicity: Crisis thresholds <= Risk_off <= Neutral
    2. PF consistency: PF should be higher in favorable regimes
    3. Trade frequency: Shouldn't vary wildly across regimes

    Returns:
        List of validation warnings
    """
    warnings = []

    # Check 1: Threshold monotonicity (crisis < risk_off < neutral)
    regimes_ordered = ['crisis', 'risk_off', 'neutral', 'risk_on']
    available_regimes = [r for r in regimes_ordered if r in regime_metrics]

    if len(available_regimes) >= 2:
        for i in range(len(available_regimes) - 1):
            regime1 = available_regimes[i]
            regime2 = available_regimes[i + 1]

            metrics1 = regime_metrics[regime1]
            metrics2 = regime_metrics[regime2]

            th1 = metrics1.thresholds.get('fusion_threshold', 0.0)
            th2 = metrics2.thresholds.get('fusion_threshold', 0.0)

            # Expect: more severe regime = lower threshold
            if th1 > th2:
                warnings.append(
                    f"⚠️  Threshold monotonicity violation: "
                    f"{regime1} fusion_th ({th1:.3f}) > {regime2} ({th2:.3f}). "
                    f"Expected: more severe regime has lower threshold."
                )

    # Check 2: PF consistency
    pfs = {regime: metrics.profit_factor for regime, metrics in regime_metrics.items()}

    if len(pfs) >= 2:
        min_pf = min(pfs.values())
        max_pf = max(pfs.values())

        if max_pf / min_pf > 3.0:
            warnings.append(
                f"⚠️  High PF variance across regimes: "
                f"max/min = {max_pf/min_pf:.1f}x. "
                f"May indicate overfitting to specific regime."
            )

    # Check 3: Trade frequency consistency
    trades_per_year = {
        regime: metrics.trades_per_year
        for regime, metrics in regime_metrics.items()
    }

    if len(trades_per_year) >= 2:
        min_tpy = min(trades_per_year.values())
        max_tpy = max(trades_per_year.values())

        if max_tpy / min_tpy > 5.0:
            warnings.append(
                f"⚠️  High trade frequency variance: "
                f"max/min = {max_tpy/min_tpy:.1f}x. "
                f"Check cooldown consistency across regimes."
            )

    return warnings
```

### 7.3 Known Event Capture Analysis

**Ground Truth Events for Bear Archetypes:**

```python
KNOWN_CRISIS_EVENTS = {
    'LUNA': {
        'date': pd.Timestamp('2022-05-09'),
        'window': (pd.Timestamp('2022-05-09'), pd.Timestamp('2022-05-12')),
        'regime': 'crisis',
        'price_drop': -0.85,
        'expected_archetypes': ['S1', 'S2'],
        'required_recall': 0.8  # Must capture
    },
    'June18': {
        'date': pd.Timestamp('2022-06-18'),
        'window': (pd.Timestamp('2022-06-18'), pd.Timestamp('2022-06-19')),
        'regime': 'crisis',
        'price_drop': -0.22,
        'expected_archetypes': ['S1'],
        'required_recall': 0.8
    },
    'FTX': {
        'date': pd.Timestamp('2022-11-08'),
        'window': (pd.Timestamp('2022-11-08'), pd.Timestamp('2022-11-10')),
        'regime': 'crisis',
        'price_drop': -0.23,
        'expected_archetypes': ['S1', 'S2'],
        'required_recall': 0.8
    },
    'SVB': {
        'date': pd.Timestamp('2023-03-10'),
        'window': (pd.Timestamp('2023-03-10'), pd.Timestamp('2023-03-12')),
        'regime': 'risk_off',
        'price_drop': -0.12,
        'expected_archetypes': ['S2', 'S4'],
        'required_recall': 0.6
    }
}


def evaluate_event_capture(
    archetype: str,
    trades_df: pd.DataFrame
) -> Dict:
    """
    Evaluate archetype's capture of known crisis events.

    Returns:
        {
            'event_name': {
                'captured': bool,
                'entry_date': pd.Timestamp or None,
                'pnl': float or None
            }
        }
    """
    results = {}

    for event_name, event_info in KNOWN_CRISIS_EVENTS.items():
        if archetype not in event_info['expected_archetypes']:
            continue

        # Check if any trade entered during event window
        window_start, window_end = event_info['window']

        event_trades = trades_df[
            (trades_df['entry_timestamp'] >= window_start) &
            (trades_df['entry_timestamp'] <= window_end)
        ]

        if len(event_trades) > 0:
            # Event captured
            best_trade = event_trades.loc[event_trades['pnl'].idxmax()]

            results[event_name] = {
                'captured': True,
                'entry_date': best_trade['entry_timestamp'],
                'pnl': best_trade['pnl'],
                'r_multiple': best_trade['r']
            }
        else:
            # Event missed
            results[event_name] = {
                'captured': False,
                'entry_date': None,
                'pnl': None,
                'r_multiple': None
            }

    # Calculate recall
    expected_events = [e for e, info in KNOWN_CRISIS_EVENTS.items()
                       if archetype in info['expected_archetypes']]
    captured_events = [e for e, r in results.items() if r['captured']]

    recall = len(captured_events) / len(expected_events) if expected_events else 0.0

    return {
        'event_results': results,
        'recall': recall,
        'n_expected': len(expected_events),
        'n_captured': len(captured_events)
    }
```

### 7.4 Validation Report Generation

```python
def generate_validation_report(
    archetype: str,
    walk_forward_results: Dict[str, WalkForwardResult],
    regime_distribution: Dict
) -> str:
    """
    Generate comprehensive validation report for archetype.

    Args:
        archetype: Archetype ID
        walk_forward_results: Dict mapping regime -> WalkForwardResult
        regime_distribution: Historical regime distribution

    Returns:
        Markdown report string
    """
    report = f"# Regime-Aware Validation Report: {archetype}\n\n"
    report += f"**Generated:** {pd.Timestamp.now()}\n"
    report += f"**Validation Method:** Regime-Stratified Walk-Forward\n\n"

    report += "## 1. Regime Coverage\n\n"
    report += "| Regime | Historical % | Archetype Allowed | Windows Validated |\n"
    report += "|--------|--------------|-------------------|-------------------|\n"

    for regime, fraction in regime_distribution['regime_fractions'].items():
        allowed = regime in walk_forward_results
        n_windows = len(walk_forward_results[regime].windows) if allowed else 0

        report += f"| {regime:10s} | {fraction*100:5.1f}% | "
        report += f"{'✓' if allowed else '✗':^17s} | "
        report += f"{n_windows:^17d} |\n"

    report += "\n## 2. Out-of-Sample Performance by Regime\n\n"
    report += "| Regime | Avg PF | Avg Sharpe | Consistency | Positive Windows |\n"
    report += "|--------|--------|------------|-------------|------------------|\n"

    for regime, wf_result in walk_forward_results.items():
        oos = wf_result.oos_metrics

        report += f"| {regime:10s} | "
        report += f"{oos['avg_pf']:6.2f} | "
        report += f"{oos['avg_sharpe']:10.2f} | "
        report += f"{oos['consistency_score']:11.2%} | "
        report += f"{oos['positive_windows']:16.1%} |\n"

    report += "\n## 3. Known Event Capture\n\n"

    # Aggregate event capture across regimes
    all_events = {}
    for regime, wf_result in walk_forward_results.items():
        if hasattr(wf_result, 'event_capture'):
            all_events.update(wf_result.event_capture['event_results'])

    if all_events:
        report += "| Event | Date | Captured | PnL | R-Multiple |\n"
        report += "|-------|------|----------|-----|------------|\n"

        for event_name, result in all_events.items():
            event_info = KNOWN_CRISIS_EVENTS[event_name]

            report += f"| {event_name:8s} | "
            report += f"{event_info['date'].date()} | "
            report += f"{'✓' if result['captured'] else '✗':^8s} | "
            report += f"{result['pnl'] if result['pnl'] else 'N/A':>6s} | "
            report += f"{result['r_multiple'] if result['r_multiple'] else 'N/A':>10s} |\n"

        overall_recall = sum(1 for r in all_events.values() if r['captured']) / len(all_events)
        report += f"\n**Overall Event Recall:** {overall_recall:.1%}\n"

    report += "\n## 4. Validation Health Checks\n\n"

    # Run consistency checks
    regime_metrics = {}
    for regime, wf_result in walk_forward_results.items():
        # Extract metrics from walk-forward result
        regime_metrics[regime] = RegimeStratifiedMetrics(
            archetype=archetype,
            regime=regime,
            profit_factor=wf_result.avg_oos_pf,
            sharpe_ratio=wf_result.avg_oos_sharpe,
            # ... other metrics
            thresholds=wf_result.windows[0]['best_thresholds'] if wf_result.windows else {}
        )

    warnings = validate_regime_consistency(archetype, regime_metrics)

    if warnings:
        report += "**Warnings:**\n\n"
        for warning in warnings:
            report += f"- {warning}\n"
    else:
        report += "✅ **All health checks passed**\n"

    report += "\n## 5. Recommended Thresholds\n\n"
    report += "```json\n"

    recommended_config = {
        "archetype": archetype,
        "thresholds": {
            regime: wf_result.windows[-1]['best_thresholds']  # Last window (most recent)
            for regime, wf_result in walk_forward_results.items()
            if wf_result.windows
        }
    }

    report += json.dumps(recommended_config, indent=2)
    report += "\n```\n"

    return report
```

---

## 8. Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)

**Deliverables:**
1. `engine/context/regime_classifier.py`: Add `label_historical_bars()` method
2. `engine/backtest/regime_aware_backtest.py`: New module with `RegimeAwareBacktest` class
3. `engine/optimization/threshold_manager.py`: Hierarchical threshold loading
4. Unit tests for regime labeling and filtering

**Validation:**
- Label 2022-2023 BTC data, verify regime periods match known events
- Run S1 backtest on risk_off bars only, compare to full-year backtest
- Confirm threshold loading logic with regime overrides

**Acceptance Criteria:**
- Regime labeling completes in < 5 seconds for 2 years of data
- Regime-filtered backtest produces different results than full backtest
- Thresholds load correctly with regime-specific overrides

### Phase 2: Optimization (Weeks 3-4)

**Deliverables:**
1. `engine/optimization/regime_aware_objective.py`: Optuna objective with multi-objective
2. `bin/optimize_regime_pairs.py`: CLI tool to optimize archetype-regime pairs
3. Pareto frontier visualization and selection logic
4. S1 and S4 optimization on all regime pairs

**Validation:**
- Run S1 optimization on crisis vs risk_off, verify different optimal thresholds
- Pareto frontier shows trade-offs (PF vs recall vs trades)
- Selected thresholds improve OOS performance vs baseline

**Acceptance Criteria:**
- 200 trials per archetype-regime pair complete in < 2 hours
- Pareto frontier contains >= 5 non-dominated solutions
- Selected thresholds achieve PF > 1.5 on OOS data

### Phase 3: Walk-Forward Validation (Weeks 5-6)

**Deliverables:**
1. `engine/validation/regime_aware_walk_forward.py`: Walk-forward with regime stratification
2. Empty window handling logic (skip or aggregate)
3. OOS metrics aggregation and reporting
4. Validation reports for S1, S2, S4, S5

**Validation:**
- Generate 6+ walk-forward windows for S1 (crisis + risk_off)
- Verify OOS PF is consistent across windows (CV < 0.5)
- Event capture: LUNA, June 18, FTX captured in >= 80% of windows

**Acceptance Criteria:**
- Walk-forward validation runs end-to-end without errors
- OOS consistency score > 0.6 for at least 2 archetypes
- Validation report shows clear regime-stratified metrics

### Phase 4: Portfolio Construction (Weeks 7-8)

**Deliverables:**
1. `engine/portfolio/regime_weighted_portfolio.py`: Portfolio weighting logic
2. Dynamic weight adjustment based on rolling regime forecast
3. Portfolio-level backtesting with regime transitions
4. Production config with optimized per-regime thresholds

**Validation:**
- Portfolio backtest on 2022-2023 with regime-aware weights
- Compare vs equal-weight portfolio (expect 20-30% improvement)
- Validate dynamic adjustment during regime transitions

**Acceptance Criteria:**
- Regime-weighted portfolio achieves Sharpe > 1.0 on 2022-2023
- Dynamic adjustment triggers <= 12 times/year (monthly)
- Production config passes all validation health checks

### Phase 5: Production Deployment (Week 9)

**Deliverables:**
1. A/B testing framework (regime-aware vs baseline)
2. Real-time regime monitoring dashboard
3. Automated alerts for regime transitions
4. Documentation and runbooks

**Validation:**
- Paper trading for 2 weeks with both systems
- Compare PF, Sharpe, drawdown between regime-aware and baseline
- Zero production errors during deployment

**Acceptance Criteria:**
- Regime-aware system shows >= 15% improvement in Sharpe vs baseline (paper trading)
- Regime classifier updates in < 1 second for real-time bar
- Monitoring dashboard correctly displays current regime and weights

---

## 9. Edge Case Handling

### 9.1 Regime Transition Mid-Trade

**Scenario:** Enter S1 trade in crisis, regime transitions to risk_off mid-trade.

**Handling Options:**

| Option | Implementation | Pros | Cons |
|--------|----------------|------|------|
| **Ignore** (Phase 1) | Keep original entry regime | Simple, consistent | May hold in wrong regime |
| **Exit on Transition** | Exit if regime no longer allowed | Respects regime change | Premature exits from noise |
| **Adaptive Risk** | Adjust stop/target based on new regime | Captures regime shifts | Complex, hard to validate |

**Implementation (Phase 1):**
```python
class Trade:
    entry_regime: str  # Fixed at entry

    def should_exit(self, current_bar):
        # Always use entry_regime for exit logic
        # Ignore mid-trade regime changes
        return self.exit_strategy.evaluate(current_bar, self.entry_regime)
```

### 9.2 Low-Frequency Regime (Crisis)

**Scenario:** Crisis regime only 5% of bars → sparse training data.

**Solutions:**

1. **Aggregate Across Years:**
   - Collect all crisis bars from 2020-2024
   - Sufficient data: 5% × 1460 bars = 73 crisis bars
   - Risk: Non-stationarity across years

2. **Use Neutral as Proxy:**
   - If < 50 crisis bars in train window, fallback to neutral
   - Assumption: Neutral thresholds are conservative
   - Risk: Not tuned for crisis severity

3. **Transfer Learning:**
   - Pre-train on risk_off, fine-tune on crisis
   - Constrain crisis thresholds to be within 20% of risk_off
   - Risk: May not capture crisis-specific behavior

**RECOMMENDATION:** Use Solution 1 (Aggregate Across Years) for crisis regime. Accept non-stationarity risk as necessary cost of sufficient data.

### 9.3 Regime Misclassification

**Scenario:** GMM incorrectly labels crisis as neutral (low confidence).

**Mitigation:**

1. **Confidence Thresholds:**
   ```python
   if classification['confidence'] < 0.6:
       regime = 'neutral'  # Fallback to conservative
   ```

2. **Manual Overrides for Known Events:**
   ```python
   REGIME_OVERRIDES = {
       '2022-05-09': 'crisis',  # LUNA
       '2022-11-08': 'crisis',  # FTX
   }
   ```

3. **Multi-Regime Probabilities:**
   ```python
   # Use blended thresholds if uncertain
   if proba['crisis'] > 0.3 and proba['risk_off'] > 0.3:
       # Use more conservative of the two
       thresholds = min(crisis_th, risk_off_th)
   ```

**RECOMMENDATION:** Use Solution 2 (Manual Overrides) for critical events, Solution 1 (Confidence Thresholds) for general operation.

### 9.4 Empty Test Windows

**Scenario:** Test window has zero crisis bars for S1 validation.

**Solutions:**

| Solution | Implementation | Statistical Validity | Data Efficiency |
|----------|----------------|----------------------|-----------------|
| **Skip Window** | Continue to next window | ✅ High (unbiased) | ❌ Low (wastes data) |
| **Use Proxy Regime** | Fallback to risk_off | ⚠️ Medium (biased) | ✅ High |
| **Aggregate Windows** | Combine 2+ test periods | ⚠️ Medium (overlap) | ✅ High |
| **Extend Window** | Increase test period to 6mo | ✅ High (if enough data) | ⚠️ Medium |

**RECOMMENDATION:** Use Skip Window for Phase 1. Add Extend Window option in Phase 3 if skipping causes insufficient validation windows.

### 9.5 Regime Distribution Shift

**Scenario:** 2022 was 40% crisis, but 2024-2025 is 5% crisis → weights obsolete.

**Dynamic Adjustment:**
```python
def adjust_for_regime_shift(
    current_date: pd.Timestamp,
    lookback_days: int = 90
):
    # Estimate recent regime distribution
    recent_dist = estimate_regime_dist(current_date, lookback_days)

    # Compare to optimization-time distribution
    historical_dist = load_optimization_regime_dist()

    # Detect significant shift (> 50% change in any regime)
    shifts = {
        regime: abs(recent_dist[regime] - historical_dist[regime])
        for regime in recent_dist.keys()
    }

    if any(shift > 0.5 for shift in shifts.values()):
        logger.warning(f"Regime distribution shift detected: {shifts}")
        logger.warning("Consider re-optimizing archetype weights")

        # Trigger re-weighting
        new_weights = calculate_regime_weighted(
            archetype_performance,
            recent_dist
        )

        return new_weights

    return current_weights
```

**Monitoring:**
- Daily: Check current regime
- Weekly: Calculate rolling 90-day regime distribution
- Monthly: Compare to optimization-time distribution, alert if > 30% shift

---

## 10. Backward Compatibility

### 10.1 Gradual Migration Strategy

**Goal:** Deploy regime-aware system without breaking existing configs.

**Approach: Dual-Mode Operation**

```python
class ArchetypeEngine:
    """
    Trading engine with regime-aware and legacy modes.
    """

    def __init__(self, config: Dict):
        self.config = config
        self.mode = config.get('regime_aware_mode', 'disabled')
        # Modes: 'disabled', 'enabled', 'ab_test'

        if self.mode in ['enabled', 'ab_test']:
            self.regime_classifier = RegimeClassifier.load(...)
            self.threshold_manager = ThresholdManager(config)


    def evaluate_archetype(self, bar, archetype):
        if self.mode == 'disabled':
            # Legacy path: use global thresholds
            return self._evaluate_legacy(bar, archetype)

        elif self.mode == 'enabled':
            # Regime-aware path: use regime-specific thresholds
            return self._evaluate_regime_aware(bar, archetype)

        elif self.mode == 'ab_test':
            # A/B test: run both, log comparison
            legacy_signal = self._evaluate_legacy(bar, archetype)
            regime_signal = self._evaluate_regime_aware(bar, archetype)

            self._log_ab_test(bar, archetype, legacy_signal, regime_signal)

            # Return regime-aware signal for live trading
            return regime_signal


    def _evaluate_legacy(self, bar, archetype):
        """Legacy evaluation with global thresholds."""
        thresholds = self.config['archetypes'][archetype]['thresholds']
        return self.archetype_detector.evaluate(bar, archetype, thresholds)


    def _evaluate_regime_aware(self, bar, archetype):
        """Regime-aware evaluation with dynamic thresholds."""
        # Classify current regime
        regime_info = self.regime_classifier.classify(bar.macro_features)
        current_regime = regime_info['regime']

        # Check if archetype allowed in this regime
        allowed_regimes = RegimeAwareBacktest.ARCHETYPE_REGIMES.get(
            archetype,
            ['risk_on', 'neutral']
        )

        if current_regime not in allowed_regimes:
            # Skip evaluation (archetype not active in this regime)
            return None

        # Get regime-specific thresholds
        thresholds = self.threshold_manager.get_thresholds(
            archetype,
            current_regime,
            use_regime_specific=True
        )

        # Evaluate archetype
        signal = self.archetype_detector.evaluate(bar, archetype, thresholds)

        if signal is not None:
            signal.regime = current_regime
            signal.regime_confidence = regime_info['proba'][current_regime]

        return signal
```

### 10.2 A/B Testing Framework

```python
class ABTestLogger:
    """
    Log parallel results from legacy vs regime-aware systems.
    """

    def __init__(self, log_path: str):
        self.log_path = log_path
        self.results = []


    def log_comparison(
        self,
        timestamp: pd.Timestamp,
        bar,
        archetype: str,
        legacy_signal,
        regime_signal,
        current_regime: str
    ):
        """
        Log comparison between legacy and regime-aware signals.
        """
        comparison = {
            'timestamp': timestamp,
            'archetype': archetype,
            'current_regime': current_regime,
            'legacy_signal': legacy_signal is not None,
            'regime_signal': regime_signal is not None,
            'agreement': (legacy_signal is not None) == (regime_signal is not None),
            'legacy_confidence': legacy_signal.confidence if legacy_signal else None,
            'regime_confidence': regime_signal.confidence if regime_signal else None,
        }

        self.results.append(comparison)

        # Log disagreements
        if not comparison['agreement']:
            logger.info(
                f"A/B DISAGREEMENT: {archetype} @ {timestamp.date()} "
                f"(regime={current_regime}, legacy={legacy_signal is not None}, "
                f"regime_aware={regime_signal is not None})"
            )


    def generate_report(self) -> str:
        """
        Generate A/B test summary report.
        """
        df = pd.DataFrame(self.results)

        report = "# A/B Test Report: Legacy vs Regime-Aware\n\n"
        report += f"**Total Evaluations:** {len(df)}\n"
        report += f"**Agreement Rate:** {df['agreement'].mean():.1%}\n\n"

        report += "## Signal Frequency by Regime\n\n"
        report += "| Regime | Legacy Signals | Regime-Aware Signals | Delta |\n"
        report += "|--------|----------------|----------------------|-------|\n"

        for regime in df['current_regime'].unique():
            regime_df = df[df['current_regime'] == regime]

            legacy_count = regime_df['legacy_signal'].sum()
            regime_count = regime_df['regime_signal'].sum()
            delta = regime_count - legacy_count

            report += f"| {regime:10s} | {legacy_count:14d} | {regime_count:20d} | {delta:+5d} |\n"

        report += "\n## Disagreements by Archetype\n\n"

        for archetype in df['archetype'].unique():
            arch_df = df[df['archetype'] == archetype]
            disagreements = (~arch_df['agreement']).sum()

            if disagreements > 0:
                report += f"### {archetype}\n\n"
                report += f"- Total disagreements: {disagreements}\n"
                report += f"- Legacy-only signals: {(arch_df['legacy_signal'] & ~arch_df['regime_signal']).sum()}\n"
                report += f"- Regime-only signals: {(~arch_df['legacy_signal'] & arch_df['regime_signal']).sum()}\n\n"

        return report
```

### 10.3 Config Migration Tool

```python
def migrate_config_to_regime_aware(legacy_config_path: str, output_path: str):
    """
    Migrate legacy config to regime-aware structure.

    Transforms:
        archetypes.S1.thresholds.fusion_threshold: 0.65

    To:
        archetypes.S1.thresholds:
            fusion_threshold: 0.65  # Global default
            regime_thresholds:
                risk_off:
                    fusion_threshold: 0.65  # Same as global initially
                crisis:
                    fusion_threshold: 0.65
    """
    with open(legacy_config_path) as f:
        config = json.load(f)

    # For each archetype
    for archetype_id, arch_config in config.get('archetypes', {}).items():
        if not archetype_id.startswith('enable_') and 'thresholds' in arch_config:
            thresholds = arch_config['thresholds']

            # Determine allowed regimes
            allowed_regimes = RegimeAwareBacktest.ARCHETYPE_REGIMES.get(
                archetype_id,
                ['risk_on', 'neutral']
            )

            # Create regime_thresholds section
            if 'regime_thresholds' not in thresholds:
                thresholds['regime_thresholds'] = {}

                # Copy global thresholds to each regime
                global_params = {k: v for k, v in thresholds.items()
                                if k != 'regime_thresholds'}

                for regime in allowed_regimes:
                    thresholds['regime_thresholds'][regime] = global_params.copy()

    # Add regime config section
    if 'regime' not in config:
        config['regime'] = {
            'enabled': True,
            'model_path': 'models/regime_classifier_gmm.pkl',
            'feature_order': [
                'VIX', 'DXY', 'MOVE', 'YIELD_2Y', 'YIELD_10Y',
                'USDT.D', 'BTC.D', 'TOTAL', 'TOTAL2',
                'funding', 'oi', 'rv_20d', 'rv_60d'
            ],
            'min_confidence': 0.6
        }

    # Save migrated config
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)

    logger.info(f"Migrated config saved to {output_path}")
    logger.info("Next steps:")
    logger.info("1. Run regime-aware optimization to tune per-regime thresholds")
    logger.info("2. Enable A/B testing: set regime_aware_mode: 'ab_test'")
    logger.info("3. Validate for 2 weeks, then switch to 'enabled'")
```

---

## Conclusion

This regime-aware optimization framework provides a mathematically rigorous, production-ready architecture for calibrating archetype thresholds within market regime states. By eliminating cross-regime contamination, the framework ensures archetypes are optimized on relevant data and deployed in appropriate market conditions.

**Key Innovations:**
1. **Regime-stratified backtesting** eliminates mislabeled training data
2. **Per-regime threshold optimization** captures regime-specific patterns
3. **Regime-aware walk-forward validation** provides unbiased OOS estimates
4. **Regime-weighted portfolio construction** adapts to market regime distribution
5. **Backward-compatible deployment** enables safe gradual migration

**Success Metrics:**
- Regime-aware S1 captures >= 80% of crisis events (LUNA, FTX, June 18)
- OOS profit factor improves 20-30% vs year-based optimization
- Walk-forward consistency score > 0.6 (low overfitting)
- Portfolio Sharpe ratio > 1.0 on 2022-2023 OOS data

**Next Steps:**
1. Implement Phase 1 (Foundation) - regime labeling and filtering
2. Validate on S1 archetype - compare crisis vs risk_off thresholds
3. Roll out to S2, S4, S5 archetypes
4. Deploy to production with A/B testing vs baseline

---

**Document Status:** READY FOR IMPLEMENTATION
**Review Required:** Senior Quant, Head of Research
**Implementation Target:** Q1 2025
