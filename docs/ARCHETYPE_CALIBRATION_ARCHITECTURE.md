# Archetype Calibration Architecture

**Version:** 1.0
**Date:** 2025-11-20
**Status:** Design Specification - Ready for Implementation
**System:** Bull Machine v2 Multi-Objective Optimization Framework

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Architecture Diagram](#system-architecture-diagram)
3. [Component Specifications](#component-specifications)
4. [Data Flow](#data-flow)
5. [Config Schema](#config-schema)
6. [Extension Points](#extension-points)
7. [Migration Path](#migration-path)
8. [Implementation Checklist](#implementation-checklist)

---

## Executive Summary

### Problem Statement

**Current State:**
- S2 (Failed Rally) produces **418 trades** at fusion=0.55 (target: 5-10)
- Hardcoded thresholds applied uniformly across all archetypes
- No regime-aware threshold adaptation
- Manual threshold tuning yields poor results (S2 PF: 0.48 after 157 trials)

**Root Causes:**
1. **Arbitrary threshold selection**: fusion=0.55 chosen without empirical analysis
2. **One-size-fits-all approach**: Same thresholds for all archetypes despite different characteristics
3. **No distribution awareness**: Thresholds not calibrated to feature distributions
4. **Missing regime integration**: No systematic regime gating or weight adaptation

### Solution Architecture

A **4-layer optimization pipeline** that decouples:
1. **Empirical Analysis** → Data-driven threshold ranges
2. **Per-Archetype Calibration** → Optuna multi-objective optimization
3. **Regime-Aware Routing** → Config-driven regime gating
4. **Validation Framework** → 3-fold temporal cross-validation

**Key Design Principles:**
- **Separation of concerns**: Distribution analysis → Optimization → Validation
- **Data-driven thresholds**: Percentile-based ranges instead of guesswork
- **Pareto frontier output**: Multiple non-dominated solutions, not single "best"
- **Extensibility**: Plug-and-play for future archetypes (S1, S4, bull patterns)

---

## System Architecture Diagram

```
┌───────────────────────────────────────────────────────────────────────┐
│                    OPTIMIZATION ORCHESTRATOR                          │
│  (bin/optimize_archetype_v2.py)                                       │
│                                                                        │
│  Input: Archetype config + Historical data (2022-2024)                │
│  Output: Pareto frontier CSV + Optimized config JSON                  │
└────────────────────┬──────────────────────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
         ▼                       ▼
┌──────────────────┐   ┌──────────────────────┐
│   LAYER 1:       │   │    LAYER 2:          │
│   DISTRIBUTION   │   │    CALIBRATION       │
│   ANALYZER       │   │    ENGINE            │
└────────┬─────────┘   └──────────┬───────────┘
         │                        │
         │ Percentiles            │ Trials
         │ (p90, p95, p99)        │ (Optuna)
         │                        │
         ▼                        ▼
┌─────────────────────────────────────────────┐
│         LAYER 3: BACKTEST RUNNER            │
│  (RegimeBacktester + RuntimeContext)        │
│                                             │
│  - Regime filtering                         │
│  - Feature enrichment                       │
│  - Trade simulation                         │
└─────────────────┬───────────────────────────┘
                  │
                  │ Metrics (PF, DD, Trade Count)
                  │
                  ▼
┌─────────────────────────────────────────────┐
│       LAYER 4: VALIDATION PIPELINE          │
│  (WalkForwardValidator + ParetoSelector)    │
│                                             │
│  - 3-fold temporal CV                       │
│  - Pareto frontier extraction               │
│  - OOS performance check                    │
└─────────────────┬───────────────────────────┘
                  │
                  │ Best configs
                  │
                  ▼
┌─────────────────────────────────────────────┐
│      CONFIG MERGER (Production Output)      │
│                                             │
│  configs/optimized/<archetype>_v2.json      │
└─────────────────────────────────────────────┘
```

**Data Dependencies:**
```
FEATURE STORE
(bin/feature_store.py)
    │
    ├─→ Fusion scores (100% coverage)
    ├─→ Liquidity scores (100% coverage)
    ├─→ Macro features (97%+ coverage)
    ├─→ Wyckoff events (95%+ coverage)
    └─→ OI_CHANGE (0% in 2022) ⚠️ BLOCKER for S5
```

**Regime Integration:**
```
REGIME DETECTOR (engine/regime_detector.py)
GMM v3.1 - 19 features
    │
    ├─→ risk_on    (58% of 2024, 2% of 2022)
    ├─→ neutral    (27% of 2024, 27% of 2022)
    ├─→ risk_off   (11% of 2024, 55% of 2022)
    └─→ crisis     (4% of 2024, 16% of 2022)
```

---

## Component Specifications

### Layer 1: DistributionAnalyzer

**Purpose:** Compute empirical percentiles of archetype fusion scores to set smart Optuna search ranges.

**Class Definition:**
```python
class DistributionAnalyzer:
    """
    Analyzes fusion score distributions for archetype calibration.

    Replaces arbitrary threshold guessing with data-driven percentile analysis.
    """

    def __init__(self, feature_store_path: str, archetype_logic: ArchetypeLogic):
        """
        Args:
            feature_store_path: Path to feature store parquet
            archetype_logic: Archetype detection logic instance
        """
        self.df = pd.read_parquet(feature_store_path)
        self.archetype_logic = archetype_logic
        self.distributions = {}  # Cache computed distributions

    def compute_fusion_distribution(
        self,
        archetype_name: str,
        period_start: str,
        period_end: str,
        regime_filter: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Compute fusion score distribution for archetype.

        Args:
            archetype_name: Canonical archetype name (e.g., 'failed_rally')
            period_start: Start date (YYYY-MM-DD)
            period_end: End date (YYYY-MM-DD)
            regime_filter: Optional regime whitelist (e.g., ['risk_off', 'crisis'])

        Returns:
            {
                'percentiles': {50: 0.32, 75: 0.45, 90: 0.58, 95: 0.67, 99: 0.82},
                'count': 8760,  # Total bars evaluated
                'matches': 438,  # Bars where archetype matched
                'mean': 0.42,
                'std': 0.18,
                'recommended_range': [0.67, 0.82]  # p95-p99 for 5-10 trades/year
            }
        """
        # Filter by date and regime
        df_period = self.df[(self.df.index >= period_start) &
                            (self.df.index <= period_end)]

        if regime_filter:
            df_period = df_period[df_period['regime_label'].isin(regime_filter)]

        # Compute fusion scores WITHOUT threshold filtering
        fusion_scores = []
        match_count = 0

        for idx, row in df_period.iterrows():
            # Create minimal RuntimeContext
            ctx = RuntimeContext(
                ts=idx,
                row=row,
                regime_probs={row['regime_label']: 1.0},
                regime_label=row['regime_label'],
                adapted_params={},
                thresholds={}  # No thresholds = no filtering
            )

            # Get archetype-specific score (WITHOUT fusion gate)
            # This requires archetype checks to return (matched, score, meta)
            result = self._call_archetype_check(archetype_name, ctx)

            if isinstance(result, tuple):
                matched, score, meta = result
                fusion_scores.append(score)
                if matched:
                    match_count += 1
            else:
                # Legacy bool return - use global fusion
                fusion_scores.append(row.get('fusion_score', 0.0))

        # Compute percentiles
        percentiles = {
            50: np.percentile(fusion_scores, 50),
            75: np.percentile(fusion_scores, 75),
            90: np.percentile(fusion_scores, 90),
            95: np.percentile(fusion_scores, 95),
            97: np.percentile(fusion_scores, 97),
            99: np.percentile(fusion_scores, 99),
            99.5: np.percentile(fusion_scores, 99.5)
        }

        return {
            'percentiles': percentiles,
            'count': len(fusion_scores),
            'matches': match_count,
            'mean': np.mean(fusion_scores),
            'std': np.std(fusion_scores),
            'recommended_range': self._derive_search_range(
                percentiles,
                target_trades_per_year=7.5  # Configurable per archetype
            )
        }

    def _derive_search_range(
        self,
        percentiles: Dict[float, float],
        target_trades_per_year: float
    ) -> Tuple[float, float]:
        """
        Derive Optuna search range from percentiles and target trade frequency.

        Logic:
        - target_trades_per_year = 7.5 (midpoint of 5-10)
        - Annual hours = 8760 (365 days * 24 hours)
        - Target percentile = 100 - (7.5 / 8760 * 100) = 99.914
        - Use p99 to p99.5 as search range

        Returns:
            (fusion_min, fusion_max) for Optuna
        """
        target_percentile = 100 - (target_trades_per_year / 8760 * 100)

        # Map to available percentiles (p99, p99.5)
        if target_percentile >= 99.5:
            return (percentiles[99.5], percentiles[99.5] + 0.05)
        elif target_percentile >= 99:
            return (percentiles[99], percentiles[99.5])
        elif target_percentile >= 97:
            return (percentiles[97], percentiles[99])
        elif target_percentile >= 95:
            return (percentiles[95], percentiles[99])
        else:
            # Too lenient, use minimum p95
            return (percentiles[95], percentiles[99])

    def export_distribution_report(
        self,
        archetype_name: str,
        output_path: Path
    ):
        """
        Export distribution analysis to CSV and PNG histogram.

        Files created:
        - {output_path}/s2_fusion_percentiles.csv
        - {output_path}/s2_fusion_histogram.png
        """
        # Implementation: pandas to_csv() + matplotlib histogram
        pass
```

**Usage Example:**
```python
analyzer = DistributionAnalyzer(
    feature_store_path="data/feature_store/BTC_2020_2024.parquet",
    archetype_logic=ArchetypeLogic(config)
)

# Analyze S2 (Failed Rally) on 2022 bear market data
dist = analyzer.compute_fusion_distribution(
    archetype_name='failed_rally',
    period_start='2022-01-01',
    period_end='2022-12-31',
    regime_filter=['risk_off', 'crisis']
)

print(f"S2 Fusion Percentiles (2022, risk_off/crisis only):")
print(f"  p50: {dist['percentiles'][50]:.3f}")
print(f"  p95: {dist['percentiles'][95]:.3f}")
print(f"  p99: {dist['percentiles'][99]:.3f}")
print(f"Recommended search range: {dist['recommended_range']}")
# Output:
#   p50: 0.420
#   p95: 0.680
#   p99: 0.850
# Recommended search range: [0.850, 0.900]
```

**Output:**
```
results/distributions/s2_2022_risk_off_crisis.json
{
  "archetype": "failed_rally",
  "period": "2022-01-01 to 2022-12-31",
  "regime_filter": ["risk_off", "crisis"],
  "percentiles": {...},
  "recommended_range": [0.85, 0.90],
  "justification": "p99 percentile = 0.85 → expect ~9 trades/year (target: 5-10)"
}
```

---

### Layer 2: ArchetypeCalibrator

**Purpose:** Optuna-based multi-objective optimizer for single archetype.

**Class Definition:**
```python
class ArchetypeCalibrator:
    """
    Multi-objective Optuna optimizer for single archetype.

    Objectives (all minimized):
    1. -PF (maximize profit factor)
    2. |trades_per_year - target| (target frequency)
    3. max_drawdown (risk control)
    """

    def __init__(
        self,
        archetype_name: str,
        config_template: Dict,
        feature_store: pd.DataFrame,
        distribution_analysis: Dict
    ):
        """
        Args:
            archetype_name: Canonical archetype name
            config_template: Base config with routing, exits, etc.
            feature_store: Full feature store (2022-2024)
            distribution_analysis: Output from DistributionAnalyzer
        """
        self.archetype_name = archetype_name
        self.config_template = config_template
        self.df = feature_store
        self.dist = distribution_analysis

        # Optuna study
        self.study = None

    def define_search_space(self, trial: optuna.Trial) -> Dict:
        """
        Define archetype-specific parameter search space using empirical ranges.

        Returns:
            params dict ready for backtest config
        """
        # Get distribution-derived fusion range
        fusion_min, fusion_max = self.dist['recommended_range']

        params = {
            'fusion_threshold': trial.suggest_float(
                'fusion_threshold',
                fusion_min,
                fusion_max
            )
        }

        # Archetype-specific parameters (pattern filters)
        if self.archetype_name == 'failed_rally':
            params.update({
                'wick_ratio_min': trial.suggest_float('wick_ratio_min', 2.0, 4.0),
                'rsi_min': trial.suggest_float('rsi_min', 75.0, 85.0),
                'vol_z_max': trial.suggest_float('vol_z_max', 0.3, 0.8)
            })
        elif self.archetype_name == 'long_squeeze':
            params.update({
                'funding_z_min': trial.suggest_float('funding_z_min', 1.2, 2.0),
                'rsi_min': trial.suggest_float('rsi_min', 70.0, 85.0),
                'liquidity_max': trial.suggest_float('liquidity_max', 0.15, 0.30),
                'oi_change_min': trial.suggest_float('oi_change_min', 0.02, 0.06)
            })
        # ... more archetypes

        return params

    def objective(self, trial: optuna.Trial) -> Tuple[float, float, float]:
        """
        Multi-objective function for Optuna.

        Returns:
            (obj1_neg_pf, obj2_trade_dev, obj3_max_dd)
        """
        # Sample parameters
        params = self.define_search_space(trial)

        # Build config
        config = self._build_trial_config(params)

        # Run 3-fold temporal CV
        fold_results = []
        for fold in self.cv_folds:
            metrics = self.backtest_runner.run(
                config=config,
                start=fold['start'],
                end=fold['end'],
                regime_filter=self._get_regime_filter()
            )
            fold_results.append(metrics)

        # Aggregate metrics
        mean_pf = np.mean([m.profit_factor for m in fold_results])
        mean_trades_annual = np.mean([m.annual_trades() for m in fold_results])
        mean_max_dd = np.mean([m.max_drawdown for m in fold_results])

        # Store user attributes for later analysis
        trial.set_user_attr('mean_pf', mean_pf)
        trial.set_user_attr('mean_trades_annual', mean_trades_annual)
        trial.set_user_attr('mean_max_dd', mean_max_dd)
        trial.set_user_attr('fold_results', [asdict(m) for m in fold_results])

        # Compute objectives (ALL minimized)
        obj1 = -mean_pf  # Maximize PF → minimize -PF
        obj2 = abs(mean_trades_annual - self.target_trades_per_year)
        obj3 = mean_max_dd

        return obj1, obj2, obj3

    def optimize(
        self,
        n_trials: int = 100,
        n_jobs: int = 4,
        timeout: Optional[int] = None
    ) -> optuna.Study:
        """
        Run Optuna optimization.

        Args:
            n_trials: Number of trials
            n_jobs: Parallel workers (4 recommended for 8-core machines)
            timeout: Optional timeout in seconds

        Returns:
            Completed Optuna study
        """
        # Create study
        self.study = optuna.create_study(
            directions=['minimize', 'minimize', 'minimize'],  # 3 objectives
            sampler=optuna.samplers.TPESampler(
                seed=42,
                multivariate=True,
                n_startup_trials=20
            ),
            pruner=optuna.pruners.HyperbandPruner(
                min_resource=1,  # 1 fold minimum
                max_resource=3,  # 3 folds maximum
                reduction_factor=3
            ),
            study_name=f"{self.archetype_name}_optimization_v2"
        )

        # Optimize
        self.study.optimize(
            self.objective,
            n_trials=n_trials,
            n_jobs=n_jobs,
            timeout=timeout,
            show_progress_bar=True
        )

        return self.study

    def _get_regime_filter(self) -> Optional[List[str]]:
        """
        Get regime whitelist for archetype.

        Examples:
        - failed_rally: ['risk_off', 'crisis']
        - long_squeeze: ['risk_off', 'crisis', 'neutral']
        - trap_within_trend: None (all regimes allowed)
        """
        regime_filters = {
            'failed_rally': ['risk_off', 'crisis'],
            'long_squeeze': ['risk_off', 'crisis', 'neutral'],
            'trap_within_trend': None,
            'order_block_retest': None
        }
        return regime_filters.get(self.archetype_name)
```

**Usage Example:**
```python
# Load distribution analysis
dist = json.load(open('results/distributions/s2_2022_risk_off_crisis.json'))

# Initialize calibrator
calibrator = ArchetypeCalibrator(
    archetype_name='failed_rally',
    config_template=config,
    feature_store=df,
    distribution_analysis=dist
)

# Run optimization (50 trials, ~2h on 4 cores)
study = calibrator.optimize(
    n_trials=50,
    n_jobs=4
)

# Extract Pareto frontier
pareto_trials = study.best_trials
print(f"Pareto frontier: {len(pareto_trials)} non-dominated solutions")
```

---

### Layer 3: RegimedBacktester

**Purpose:** Regime-aware backtest executor with feature enrichment.

**Class Definition:**
```python
class RegimedBacktester:
    """
    Backtest runner with regime filtering and runtime feature enrichment.

    Integrates:
    - Regime detector (GMM v3.1)
    - RuntimeContext (threshold resolution)
    - ArchetypeLogic (pattern detection)
    - Trade simulation
    """

    def __init__(
        self,
        feature_store: pd.DataFrame,
        regime_detector: RegimeDetector,
        archetype_logic: ArchetypeLogic
    ):
        self.df = feature_store
        self.regime_detector = regime_detector
        self.archetype_logic = archetype_logic

    def run(
        self,
        config: Dict,
        start: str,
        end: str,
        regime_filter: Optional[List[str]] = None
    ) -> BacktestMetrics:
        """
        Run backtest on specified period with regime filtering.

        Args:
            config: Archetype config with thresholds
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            regime_filter: Optional regime whitelist (e.g., ['risk_off', 'crisis'])

        Returns:
            BacktestMetrics with PF, DD, trade count, etc.
        """
        # Filter period
        df_period = self.df[(self.df.index >= start) & (self.df.index <= end)]

        # Apply regime filtering
        if regime_filter:
            df_period = df_period[df_period['regime_label'].isin(regime_filter)]

        trades = []

        for idx, row in df_period.iterrows():
            # Create RuntimeContext with regime-aware thresholds
            ctx = RuntimeContext(
                ts=idx,
                row=row,
                regime_probs=self._get_regime_probs(row),
                regime_label=row['regime_label'],
                adapted_params=config.get('archetypes', {}),
                thresholds=self._build_thresholds(config, row['regime_label'])
            )

            # Detect archetype
            archetype_name, fusion_score, liquidity_score = self.archetype_logic.detect(ctx)

            if archetype_name:
                # Simulate trade
                trade = self._simulate_trade(
                    entry_idx=idx,
                    archetype=archetype_name,
                    config=config,
                    df=df_period
                )
                trades.append(trade)

        # Compute metrics
        return self._compute_metrics(trades, df_period)

    def _build_thresholds(self, config: Dict, regime: str) -> Dict:
        """
        Build archetype-specific thresholds with regime routing weights.

        Returns:
            {
                'failed_rally': {
                    'fusion_threshold': 0.85,  # Base threshold
                    'regime_weight': 2.5,      # Crisis weight from routing
                    'wick_ratio_min': 3.2,
                    ...
                },
                ...
            }
        """
        archetypes_config = config.get('archetypes', {})
        routing_config = archetypes_config.get('routing', {})
        regime_routing = routing_config.get(regime, {})
        regime_weights = regime_routing.get('weights', {})

        thresholds = {}

        for archetype_name in ARCHETYPE_NAMES:
            base_thresholds = archetypes_config.get(archetype_name, {})
            regime_weight = regime_weights.get(archetype_name, 1.0)

            thresholds[archetype_name] = {
                **base_thresholds,
                'regime_weight': regime_weight
            }

        return thresholds

    def _simulate_trade(
        self,
        entry_idx: pd.Timestamp,
        archetype: str,
        config: Dict,
        df: pd.DataFrame
    ) -> Trade:
        """
        Simulate trade execution and exit.

        Uses archetype-specific exit logic:
        - failed_rally: ATR stop + 48h time limit
        - long_squeeze: ATR stop + 24h time limit
        - trap_within_trend: Trailing stop + no time limit
        """
        # Implementation: ATR-based stops, time limits, trailing logic
        pass

    def _compute_metrics(
        self,
        trades: List[Trade],
        df: pd.DataFrame
    ) -> BacktestMetrics:
        """
        Compute performance metrics from trades.

        Returns:
            BacktestMetrics(
                total_trades=12,
                win_rate=58.3,
                profit_factor=1.72,
                sharpe_ratio=1.42,
                max_drawdown=0.15,
                total_return=0.28,
                avg_win=0.045,
                avg_loss=-0.022,
                duration_days=181
            )
        """
        # Implementation: Standard backtest metrics
        pass
```

---

### Layer 4: ParetoSelector + WalkForwardValidator

**Purpose:** Extract Pareto frontier and validate on OOS data.

**Class Definition:**
```python
class ParetoSelector:
    """
    Extract and rank Pareto-optimal solutions from Optuna study.
    """

    def __init__(self, study: optuna.Study):
        self.study = study

    def extract_pareto_frontier(self) -> pd.DataFrame:
        """
        Get all non-dominated trials from study.

        Returns:
            DataFrame with columns:
            - trial_number
            - obj1_neg_pf, obj2_trade_dev, obj3_max_dd
            - mean_pf, mean_trades_annual, mean_max_dd
            - fusion_threshold, wick_ratio_min, rsi_min, ...
            - fold_1_pf, fold_2_pf, fold_3_pf
        """
        pareto_trials = self.study.best_trials

        records = []
        for trial in pareto_trials:
            record = {
                'trial_number': trial.number,
                'obj1_neg_pf': trial.values[0],
                'obj2_trade_dev': trial.values[1],
                'obj3_max_dd': trial.values[2],
                'mean_pf': trial.user_attrs.get('mean_pf'),
                'mean_trades_annual': trial.user_attrs.get('mean_trades_annual'),
                'mean_max_dd': trial.user_attrs.get('mean_max_dd'),
                **trial.params  # All sampled parameters
            }
            records.append(record)

        df = pd.DataFrame(records)
        df = df.sort_values('mean_pf', ascending=False)  # Best PF first

        return df

    def select_top_configs(
        self,
        n: int = 5,
        preference: str = 'balanced'
    ) -> List[Dict]:
        """
        Select top N configs from Pareto frontier.

        Args:
            n: Number of configs to return
            preference: Selection strategy
                - 'aggressive': Maximize PF (accept higher DD)
                - 'conservative': Minimize DD (accept lower PF)
                - 'balanced': Equal weight to PF and DD

        Returns:
            List of config dicts ready for OOS validation
        """
        df = self.extract_pareto_frontier()

        if preference == 'aggressive':
            df = df.sort_values('mean_pf', ascending=False)
        elif preference == 'conservative':
            df = df.sort_values('mean_max_dd', ascending=True)
        elif preference == 'balanced':
            # Composite score: PF / (1 + DD)
            df['composite_score'] = df['mean_pf'] / (1 + df['mean_max_dd'])
            df = df.sort_values('composite_score', ascending=False)

        top_configs = df.head(n).to_dict('records')
        return top_configs


class WalkForwardValidator:
    """
    Validate configs on out-of-sample data (2023 H1).
    """

    def __init__(self, backtester: RegimedBacktester):
        self.backtester = backtester

    def validate_oos(
        self,
        config: Dict,
        oos_start: str = '2023-01-01',
        oos_end: str = '2023-06-30'
    ) -> Dict:
        """
        Run OOS validation and check degradation.

        Returns:
            {
                'oos_metrics': BacktestMetrics(...),
                'is_metrics': BacktestMetrics(...),  # From trial
                'degradation': {
                    'pf_pct': 0.28,  # 28% PF drop
                    'sharpe_pct': 0.35,  # 35% Sharpe drop
                },
                'passed': False,  # True if meets criteria
                'failure_reason': 'PF degradation > 30%'
            }
        """
        # Run OOS backtest
        oos_metrics = self.backtester.run(
            config=config,
            start=oos_start,
            end=oos_end,
            regime_filter=self._get_regime_filter(config)
        )

        # Get in-sample metrics from config metadata
        is_metrics = config.get('_in_sample_metrics', {})

        # Compute degradation
        pf_deg = (is_metrics['pf'] - oos_metrics.profit_factor) / is_metrics['pf']
        sharpe_deg = (is_metrics['sharpe'] - oos_metrics.sharpe_ratio) / is_metrics['sharpe']

        # Check acceptance criteria
        passed = all([
            pf_deg < 0.30,  # <30% PF drop
            sharpe_deg < 0.40,  # <40% Sharpe drop
            oos_metrics.profit_factor >= 1.1,  # Minimum viable PF
            oos_metrics.max_drawdown <= 0.20  # Max DD threshold
        ])

        failure_reason = None
        if not passed:
            if pf_deg >= 0.30:
                failure_reason = f"PF degradation {pf_deg:.1%} > 30%"
            elif sharpe_deg >= 0.40:
                failure_reason = f"Sharpe degradation {sharpe_deg:.1%} > 40%"
            elif oos_metrics.profit_factor < 1.1:
                failure_reason = f"OOS PF {oos_metrics.profit_factor:.2f} < 1.1"
            elif oos_metrics.max_drawdown > 0.20:
                failure_reason = f"OOS DD {oos_metrics.max_drawdown:.1%} > 20%"

        return {
            'oos_metrics': asdict(oos_metrics),
            'is_metrics': is_metrics,
            'degradation': {
                'pf_pct': pf_deg,
                'sharpe_pct': sharpe_deg
            },
            'passed': passed,
            'failure_reason': failure_reason
        }
```

---

### Layer 5: EngineWeightOptimizer (Future Work)

**Purpose:** Optimize meta-fusion weights (structure, liquidity, momentum, wyckoff, macro).

**Class Definition:**
```python
class EngineWeightOptimizer:
    """
    Optimize domain-level fusion weights using Optuna or ML meta-learner.

    Current fusion:
        fusion = 0.331*wyckoff + 0.392*liquidity + 0.205*momentum - 0.075*fakeout

    Optimization goal:
        Learn regime-specific weights that maximize PF
    """

    def __init__(self, feature_store: pd.DataFrame):
        self.df = feature_store

    def extract_domain_scores(self) -> pd.DataFrame:
        """
        Extract or compute domain-level scores for each bar.

        Challenge: Domain scores not directly accessible in current codebase.
        May require instrumentation of fusion engine.

        Returns:
            DataFrame with columns:
            - structure_score (from order block, liquidity sweep logic)
            - liquidity_score (from BOMS, FVG)
            - momentum_score (from RSI, ADX, vol_z)
            - wyckoff_score (from Wyckoff event detection)
            - macro_score (from VIX_Z, DXY_Z, funding_Z)
        """
        # TODO: Implement domain score extraction
        # This requires deep integration with fusion engine
        pass

    def optimize_weights_ml(self) -> Dict[str, float]:
        """
        ML-based approach: Logistic regression on domain scores → trade outcomes.

        Faster than Optuna (1 day vs 5-7 days).

        Returns:
            {
                'wyckoff': 0.28,
                'liquidity': 0.35,
                'momentum': 0.22,
                'macro': 0.15
            }
        """
        # TODO: Implement ML meta-learner
        pass

    def optimize_weights_optuna(self) -> optuna.Study:
        """
        Optuna-based approach: Multi-objective optimization of weights.

        More rigorous, but slower.

        Constraint: sum(weights) = 1.0
        """
        # TODO: Implement Optuna weight optimization
        pass
```

**Status:** DEFERRED to Phase 6 (optional) - requires domain score instrumentation.

---

## Data Flow

### End-to-End Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│ STEP 1: Empirical Distribution Analysis                     │
│                                                              │
│ Input:  Feature store (2022-2024)                           │
│ Output: results/distributions/s2_fusion_percentiles.csv     │
│                                                              │
│ $ python bin/analyze_archetype_distributions.py \           │
│     --archetype failed_rally \                              │
│     --period 2022-01-01:2022-12-31 \                        │
│     --regime-filter risk_off,crisis                         │
│                                                              │
│ Result: Recommended fusion range [0.85, 0.90]               │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 2: Optuna Multi-Objective Optimization                 │
│                                                              │
│ Input:  Distribution analysis + Config template             │
│ Output: SQLite study DB + Pareto frontier CSV               │
│                                                              │
│ $ python bin/optimize_s2_distribution_aware.py \            │
│     --trials 50 \                                           │
│     --n-jobs 4 \                                            │
│     --fusion-range 0.85:0.90                                │
│                                                              │
│ Result: 50 trials → 12 Pareto solutions                     │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 3: Pareto Frontier Extraction                          │
│                                                              │
│ Input:  Optuna study                                        │
│ Output: results/s2_optimization/pareto_frontier.csv         │
│                                                              │
│ $ python bin/export_pareto_frontier.py \                   │
│     --study-db optuna_s2.db \                               │
│     --output results/s2_optimization/                       │
│                                                              │
│ Result: 12 non-dominated solutions ranked by PF             │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 4: Out-of-Sample Validation                            │
│                                                              │
│ Input:  Top 5 Pareto configs                                │
│ Output: OOS validation report                               │
│                                                              │
│ $ python bin/walk_forward_validator.py \                   │
│     --configs results/s2_optimization/top5.json \           │
│     --oos-period 2023-01-01:2023-06-30                      │
│                                                              │
│ Result: 2 configs pass OOS validation (PF > 1.1, deg < 30%) │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 5: Production Config Generation                        │
│                                                              │
│ Input:  Best OOS config                                     │
│ Output: configs/optimized/s2_v2.json                        │
│                                                              │
│ $ python bin/generate_production_config.py \               │
│     --archetype failed_rally \                              │
│     --trial-number 42 \                                     │
│     --output configs/optimized/s2_v2.json                   │
│                                                              │
│ Result: Production-ready config with optimized thresholds   │
└─────────────────────────────────────────────────────────────┘
```

### Data Sources

**Feature Store Schema:**
```
BTC_2020_2024.parquet (100% coverage)
├── Timestamps: 2020-01-01 to 2024-12-31 (1H bars)
├── Fusion Scores:
│   ├── fusion_score (runtime-calculated, 100%)
│   ├── wyckoff_score (95% coverage)
│   ├── liquidity_score (100% coverage)
│   └── momentum_score (derived, 100%)
├── Macro Features:
│   ├── VIX_Z (97% coverage)
│   ├── DXY_Z (97% coverage)
│   ├── funding_Z (100% coverage)
│   ├── YC_SPREAD (95% coverage)
│   └── BTC.D_Z (97% coverage)
├── Technical Features:
│   ├── rsi_14 (100%)
│   ├── adx_14 (100%)
│   ├── atr_20 (100%)
│   ├── volume_zscore (100%)
│   └── atr_percentile (100%)
├── Structure Features:
│   ├── tf1h_ob_high (100% after backfill)
│   ├── tf1h_bos_bullish (100%)
│   ├── tf1h_fvg_present (100%)
│   └── boms_strength (100%)
└── Regime Labels:
    ├── regime_label (GMM v3.1, 100%)
    ├── regime_probs (dict, 100%)
    └── regime_confidence (float, 100%)
```

**Missing Data (Blockers):**
```
OI_CHANGE (Open Interest Change)
├── 2024: 100% coverage ✅
├── 2023: 0% coverage ❌
├── 2022: 0% coverage ❌ BLOCKER for S5 optimization
└── 2020-2021: 0% coverage ❌

Status: S5 optimization uses graceful degradation (3-component scoring)
        when OI data unavailable. Full 4-component scoring when available.
```

---

## Config Schema

### Optimized Archetype Config Format

**Output from optimization pipeline:**
```json
{
  "version": "s2_optimized_v2",
  "archetype": "failed_rally",
  "optimization_metadata": {
    "trial_number": 42,
    "optimization_date": "2025-11-20",
    "in_sample_period": "2022-01-01 to 2024-09-30",
    "in_sample_metrics": {
      "pf": 1.72,
      "win_rate": 58.3,
      "sharpe": 1.42,
      "max_dd": 0.15,
      "total_trades": 36,
      "annual_trades": 7.8
    },
    "oos_period": "2024-10-01 to 2024-12-31",
    "oos_metrics": {
      "pf": 1.28,
      "win_rate": 52.5,
      "sharpe": 1.05,
      "max_dd": 0.18,
      "total_trades": 8,
      "annual_trades": 7.2
    },
    "degradation": {
      "pf_pct": 0.256,
      "sharpe_pct": 0.261
    },
    "validation_passed": true
  },

  "thresholds": {
    "fusion_threshold": 0.872,
    "wick_ratio_min": 3.2,
    "rsi_min": 78.5,
    "vol_z_max": 0.42
  },

  "regime_routing": {
    "risk_on": 0.0,
    "neutral": 0.0,
    "risk_off": 2.0,
    "crisis": 2.5
  },

  "archetype_weight": 2.0,

  "risk_management": {
    "max_risk_pct": 0.015,
    "atr_stop_mult": 2.0,
    "cooldown_bars": 8
  },

  "exits": {
    "trail_atr": 1.5,
    "time_limit_hours": 48
  }
}
```

### Unified Production Config (Merged)

**Merged config combining all optimized archetypes:**
```json
{
  "version": "2.0.0-unified-optimized",
  "profile": "production_optimized",
  "description": "Unified config from per-archetype optimization (Phase 2)",

  "fusion_adapt": {
    "enable": true,
    "ema_alpha": 0.2,
    "min_weight": 0.05
  },

  "regime_classifier": {
    "model_path": "models/regime_classifier_gmm.pkl",
    "feature_order": [...],
    "zero_fill_missing": false
  },

  "archetypes": {
    "use_archetypes": true,

    "enable_S2": false,
    "enable_S5": true,
    "enable_B": true,
    "enable_H": true,
    "enable_L": true,

    "thresholds": {
      "min_liquidity": 0.20
    },

    "failed_rally": {
      "fusion_threshold": 0.872,
      "wick_ratio_min": 3.2,
      "rsi_min": 78.5,
      "vol_z_max": 0.42,
      "archetype_weight": 2.0,
      "direction": "short"
    },

    "long_squeeze": {
      "fusion_threshold": 0.68,
      "funding_z_min": 1.42,
      "rsi_min": 72.5,
      "liquidity_max": 0.18,
      "archetype_weight": 2.5,
      "direction": "short"
    },

    "order_block_retest": {
      "fusion_threshold": 0.38,
      "boms_strength_min": 0.35,
      "wyckoff_min": 0.38,
      "archetype_weight": 1.8,
      "direction": "long"
    },

    "trap_within_trend": {
      "fusion_threshold": 0.42,
      "adx_threshold": 28,
      "liquidity_threshold": 0.22,
      "archetype_weight": 2.8,
      "direction": "long"
    },

    "volume_exhaustion": {
      "fusion_threshold": 0.40,
      "vol_z_min": 1.2,
      "rsi_min": 72,
      "archetype_weight": 2.2,
      "direction": "long"
    },

    "routing": {
      "risk_on": {
        "weights": {
          "failed_rally": 0.0,
          "long_squeeze": 0.2,
          "order_block_retest": 1.3,
          "trap_within_trend": 1.2,
          "volume_exhaustion": 1.1
        },
        "final_gate_delta": 0.0
      },
      "neutral": {
        "weights": {
          "failed_rally": 0.0,
          "long_squeeze": 0.8,
          "order_block_retest": 1.0,
          "trap_within_trend": 0.9,
          "volume_exhaustion": 0.8
        },
        "final_gate_delta": 0.0
      },
      "risk_off": {
        "weights": {
          "failed_rally": 2.0,
          "long_squeeze": 2.5,
          "order_block_retest": 0.4,
          "trap_within_trend": 0.2,
          "volume_exhaustion": 0.3
        },
        "final_gate_delta": 0.02
      },
      "crisis": {
        "weights": {
          "failed_rally": 2.5,
          "long_squeeze": 2.5,
          "order_block_retest": 0.2,
          "trap_within_trend": 0.1,
          "volume_exhaustion": 0.1
        },
        "final_gate_delta": 0.04
      }
    },

    "exits": {
      "failed_rally": {
        "trail_atr": 1.5,
        "time_limit_hours": 48
      },
      "long_squeeze": {
        "trail_atr": 1.5,
        "time_limit_hours": 24
      },
      "order_block_retest": {
        "trail_atr": 2.0,
        "time_limit_hours": 72
      }
    }
  },

  "risk": {
    "base_risk_pct": 0.015,
    "max_position_size_pct": 0.15,
    "max_portfolio_risk_pct": 0.08
  }
}
```

---

## Extension Points

### Adding New Archetypes

**Step 1: Define archetype check method**

Add to `engine/archetypes/logic_v2_adapter.py`:
```python
def _check_S4(self, context: RuntimeContext) -> Tuple[bool, float, Dict]:
    """
    S4 - Distribution Climax: Volume exhaustion + no follow-through.

    Returns:
        (matched: bool, score: float, meta: dict)
    """
    # Get thresholds
    fusion_th = context.get_threshold('distribution', 'fusion_threshold', 0.37)
    vol_climax = context.get_threshold('distribution', 'vol_climax', 1.5)
    liq_max = context.get_threshold('distribution', 'liq_max', 0.3)

    # Extract features
    vol_z = self.g(context.row, 'vol_z', 0.0)
    liq = self._liquidity_score(context.row)

    # Gate checks
    if vol_z < vol_climax:
        return False, 0.0, {"reason": "vol_z_low", "value": vol_z}
    if liq >= liq_max:
        return False, 0.0, {"reason": "liquidity_too_high", "value": liq}

    # Compute score
    components = {
        "fusion": self._fusion(context.row),
        "vol_climax": min(vol_z / 3.0, 1.0),
        "liquidity_inverse": max(0.0, 1.0 - liq)
    }

    weights = context.get_threshold('distribution', 'weights', {
        "fusion": 0.40,
        "vol_climax": 0.35,
        "liquidity_inverse": 0.25
    })

    score = sum(components[k] * weights.get(k, 0.0) for k in components)

    if score < fusion_th:
        return False, score, {"reason": "score_below_threshold", "score": score}

    return True, score, {"components": components, "weights": weights}
```

**Step 2: Register in archetype map**

Update `_detect_all_archetypes()`:
```python
archetype_map = {
    # ...existing...
    'S4': ('distribution', self._check_S4, 20),
}
```

**Step 3: Add to optimization matrix**

Update `configs/archetype_optimization_parameter_matrix.json`:
```json
{
  "distribution": {
    "code": "S4",
    "direction": "short",
    "search_space": {
      "fusion_threshold": [0.35, 0.47],
      "vol_climax": [1.5, 2.2],
      "liq_max": [0.12, 0.22]
    },
    "target_trades_per_year": 60,
    "regime_filter": ["risk_off", "crisis"],
    "optimization_priority": "LOW"
  }
}
```

**Step 4: Run optimization**
```bash
python bin/optimize_archetype_v2.py \
    --archetype distribution \
    --trials 90 \
    --n-jobs 4
```

### Adding Regime-Specific Variants

**Use case:** Optimize S5 separately for bear markets (2022) vs bull crisis (2024).

**Implementation:**
```python
# Optimize S5 for bear market
calibrator_bear = ArchetypeCalibrator(
    archetype_name='long_squeeze',
    config_template=config,
    feature_store=df[(df.index >= '2022-01-01') & (df.index <= '2022-12-31')],
    distribution_analysis=dist_2022
)

study_bear = calibrator_bear.optimize(n_trials=100)

# Optimize S5 for bull crisis
calibrator_bull_crisis = ArchetypeCalibrator(
    archetype_name='long_squeeze',
    config_template=config,
    feature_store=df[(df.index >= '2024-09-01') & (df.index <= '2024-09-30')],
    distribution_analysis=dist_2024_crisis
)

study_bull_crisis = calibrator_bull_crisis.optimize(n_trials=100)
```

**Config format:**
```json
{
  "long_squeeze": {
    "bear_market": {
      "fusion_threshold": 0.68,
      "funding_z_min": 1.42,
      "rsi_min": 72.5
    },
    "bull_crisis": {
      "fusion_threshold": 0.58,
      "funding_z_min": 1.85,
      "rsi_min": 78.0
    }
  }
}
```

### Adding Custom Objectives

**Use case:** Optimize for Calmar Ratio instead of Sharpe.

**Implementation:**
```python
def objective_calmar(trial: optuna.Trial) -> Tuple[float, float, float]:
    """
    Custom objective with Calmar Ratio.

    Objectives:
    1. -Calmar (maximize return/DD ratio)
    2. |trades - target|
    3. Sortino Ratio (downside deviation)
    """
    params = define_search_space(trial)
    metrics = run_backtest(params)

    calmar = metrics.total_return / max(metrics.max_drawdown, 0.01)
    sortino = metrics.sharpe_ratio * 1.2  # Approximation

    obj1 = -calmar  # Maximize Calmar
    obj2 = abs(metrics.annual_trades - target_trades)
    obj3 = -sortino  # Maximize Sortino

    return obj1, obj2, obj3
```

---

## Migration Path

### Phase 0: Prerequisites (Week 1)

**Tasks:**
1. **Fix OI_CHANGE pipeline** (blocker for S5)
   - Validate `bin/fix_oi_change_pipeline.py`
   - Backfill 2022-2023 data
   - Verify non-zero coverage

2. **Feature validation**
   - Run `bin/validate_feature_coverage.py` on 2022-2024 data
   - Confirm 100% coverage for required features
   - Document any gaps

**Deliverables:**
- `data/feature_store/BTC_2020_2024.parquet` with OI_CHANGE backfilled
- Feature coverage report showing 100% for S2/S5 requirements

**Acceptance Criteria:**
- OI_CHANGE shows non-zero values in 2022 data
- All required features have >95% coverage

---

### Phase 1: Empirical Analysis (Week 2)

**Tasks:**
1. **Implement DistributionAnalyzer**
   - Create `bin/analyze_archetype_distributions.py`
   - Add histogram visualization (matplotlib)
   - Export percentile tables to CSV

2. **Run distribution analysis for S2, S5**
   ```bash
   python bin/analyze_archetype_distributions.py \
       --archetype failed_rally \
       --period 2022-01-01:2022-12-31 \
       --regime-filter risk_off,crisis \
       --output results/distributions/s2_2022.json

   python bin/analyze_archetype_distributions.py \
       --archetype long_squeeze \
       --period 2022-01-01:2022-12-31 \
       --regime-filter risk_off,crisis,neutral \
       --output results/distributions/s5_2022.json
   ```

**Deliverables:**
- `results/distributions/s2_fusion_percentiles.csv`
- `results/distributions/s2_fusion_histogram.png`
- `results/distributions/s5_fusion_percentiles.csv`
- `results/distributions/s5_fusion_histogram.png`

**Acceptance Criteria:**
- Recommended fusion ranges are tighter than current arbitrary ranges
- Distribution analysis shows clear separation between p95 and p99
- Histograms visually confirm data quality

---

### Phase 2: S2 Calibration (Week 3)

**Tasks:**
1. **Implement ArchetypeCalibrator**
   - Create `bin/optimize_s2_distribution_aware.py`
   - Integrate with existing `bin/backtest_knowledge_v2.py`
   - Add Hyperband pruning

2. **Run S2 optimization**
   ```bash
   python bin/optimize_s2_distribution_aware.py \
       --trials 50 \
       --n-jobs 4 \
       --fusion-range 0.85:0.90
   ```

**Deliverables:**
- `optuna_s2_v2.db` (SQLite study)
- `results/s2_optimization/pareto_frontier.csv`
- `results/s2_optimization/optimization_report.md`

**Acceptance Criteria:**
- Pareto frontier contains ≥3 solutions with PF > 1.3
- Best solution shows 5-10 trades/year (down from 418)
- OOS validation (2023 H1) shows PF > 1.1

**Decision Point:**
- **If S2 fails to achieve PF > 1.1 on OOS:** Mark S2 as PERMANENTLY DISABLED (same as current status)
- **If S2 achieves PF > 1.1 on OOS:** Proceed to production config generation

---

### Phase 3: S5 Calibration (Week 4)

**Tasks:**
1. **Run S5 optimization** (same as S2, different parameters)
   ```bash
   python bin/optimize_s5_regime_aware.py \
       --trials 50 \
       --n-jobs 4
   ```

2. **Cross-regime validation**
   - Validate S5 on 2022 (bear) and 2024 crisis (if available)
   - Ensure PF(risk_on) < 0.9 (confirms regime specificity)

**Deliverables:**
- `results/s5_optimization/pareto_frontier.csv`
- `results/s5_optimization/optimization_report_bear.md`
- `configs/optimized/s5_v2.json`

**Acceptance Criteria:**
- PF > 1.5 on 2022 bear market validation
- 7-12 trades/year frequency
- PF(risk_on) < 1.0 (confirms bear pattern)

---

### Phase 4: Regime Gating (Week 5)

**Tasks:**
1. **Update RuntimeContext integration**
   - Ensure `engine/archetypes/logic_v2_adapter.py` reads regime routing weights
   - Add unit tests for regime gating logic

2. **Config migration**
   ```bash
   python bin/migrate_configs_to_regime_gating.py \
       --input configs/mvp/mvp_bear_market_v1.json \
       --output configs/mvp/mvp_bear_market_v2.json
   ```

**Deliverables:**
- Updated `engine/archetypes/logic_v2_adapter.py` with regime weight application
- `tests/test_regime_gating.py` with 100% coverage
- Migrated configs with new routing schema

**Acceptance Criteria:**
- Unit tests pass for regime gating (S2 blocked in risk_on)
- Integration test shows 0 S2 trades in risk_on regime
- Regime routing multipliers correctly applied to fusion scores

---

### Phase 5: Validation Pipeline (Week 6)

**Tasks:**
1. **Implement WalkForwardValidator**
   - Create `bin/walk_forward_validator.py`
   - Add JSON schema for validation reports

2. **Run validation on optimized configs**
   ```bash
   python bin/walk_forward_validator.py \
       --config configs/optimized/s2_v2.json \
       --oos-period 2023-01-01:2023-06-30 \
       --output results/validation/s2_walk_forward_report.json
   ```

**Deliverables:**
- `bin/walk_forward_validator.py`
- `results/validation/s2_validation_report.json`
- `results/validation/s5_validation_report.json`

**Acceptance Criteria:**
- At least 1 archetype (S5 or bull patterns) passes 3-tier validation
- Validation reports show PF degradation < 30%
- JSON schema validates all output reports

---

### Phase 6: Engine Weight Optimization (OPTIONAL)

**Status:** DEFERRED - requires domain score instrumentation

**Estimated Effort:** 5-7 days

**Deliverables (if implemented):**
- `bin/extract_domain_scores.py`
- `bin/optimize_engine_weights_ml.py`
- `results/engine_optimization/engine_weights_report.md`

---

## Implementation Checklist

### Code Components

**New Files to Create:**
- [ ] `bin/analyze_archetype_distributions.py` (DistributionAnalyzer)
- [ ] `bin/optimize_s2_distribution_aware.py` (ArchetypeCalibrator for S2)
- [ ] `bin/optimize_s5_regime_aware.py` (ArchetypeCalibrator for S5)
- [ ] `bin/walk_forward_validator.py` (WalkForwardValidator)
- [ ] `bin/export_pareto_frontier.py` (ParetoSelector)
- [ ] `bin/generate_production_config.py` (Config generator)
- [ ] `bin/migrate_configs_to_regime_gating.py` (Config migrator)

**Modified Files:**
- [ ] `engine/archetypes/logic_v2_adapter.py` (ensure all archetypes return tuples)
- [ ] `bin/backtest_knowledge_v2.py` (integrate RuntimeContext regime routing)
- [ ] `configs/mvp/mvp_bear_market_v1.json` (migrate to v2 schema)

**New Tests:**
- [ ] `tests/test_distribution_analyzer.py`
- [ ] `tests/test_archetype_calibrator.py`
- [ ] `tests/test_regime_gating.py`
- [ ] `tests/integration/test_walk_forward_validator.py`

### Infrastructure

**Database:**
- [ ] SQLite databases for Optuna studies (one per archetype)
- [ ] Schema: `optuna_{archetype}_v2.db`

**Output Directories:**
```
results/
├── distributions/
│   ├── s2_fusion_percentiles.csv
│   ├── s2_fusion_histogram.png
│   ├── s5_fusion_percentiles.csv
│   └── s5_fusion_histogram.png
├── s2_optimization/
│   ├── pareto_frontier.csv
│   ├── all_trials.csv
│   └── optimization_report.md
├── s5_optimization/
│   ├── pareto_frontier.csv
│   ├── all_trials.csv
│   └── optimization_report_bear.md
└── validation/
    ├── s2_walk_forward_report.json
    └── s5_walk_forward_report.json

configs/
└── optimized/
    ├── s2_v2.json
    ├── s5_v2.json
    └── unified_v2.json
```

### Dependencies

**Python Packages:**
```
requirements_optuna.txt:
optuna==3.4.0
scipy==1.11.4
matplotlib==3.8.2
seaborn==0.13.0
```

**Existing Dependencies (verified):**
- pandas (feature store operations)
- numpy (percentile calculations)
- logging (structured logging)

### Documentation

**New Docs:**
- [ ] `docs/REGIME_GATING_CONFIG_SCHEMA.md`
- [ ] `docs/WALK_FORWARD_VALIDATION_SCHEMA.json`
- [ ] `docs/PARETO_FRONTIER_INTERPRETATION_GUIDE.md`

**Updated Docs:**
- [ ] `docs/ARCHETYPE_OPTIMIZATION_QUICK_REFERENCE.md` (add distribution analysis step)
- [ ] `CHANGELOG.md` (document Phase 2 optimization framework)

---

## Validation Checklist

### Phase 1 (Empirical Analysis)

- [ ] Distribution analysis runs without errors
- [ ] Percentile tables show expected patterns (p99 >> p50)
- [ ] Histograms confirm score distributions are reasonable
- [ ] Recommended fusion ranges are tighter than current arbitrary ranges

### Phase 2 (S2 Optimization)

- [ ] Optuna study completes 50 trials
- [ ] Pareto frontier contains ≥3 non-dominated solutions
- [ ] Best solution shows 5-10 trades/year (vs 418 current)
- [ ] 3-fold CV shows consistent performance (std < 50% of mean)
- [ ] OOS validation passes (PF > 1.1, degradation < 30%)

### Phase 3 (S5 Optimization)

- [ ] S5 optimization completes successfully
- [ ] PF > 1.5 on bear market (2022) validation
- [ ] Trade frequency 7-12/year
- [ ] Cross-regime test confirms PF(risk_on) < 0.9
- [ ] OOS validation passes

### Phase 4 (Regime Gating)

- [ ] Unit tests pass for regime gating logic
- [ ] Integration test shows 0 S2 trades in risk_on regime
- [ ] Regime routing weights correctly applied to fusion scores
- [ ] Config migration script produces valid v2 configs

### Phase 5 (Validation Pipeline)

- [ ] WalkForwardValidator runs on all optimized configs
- [ ] JSON validation reports conform to schema
- [ ] At least 1 archetype passes full validation
- [ ] Validation reports show expected degradation patterns

---

## Appendix A: Failure Modes and Mitigations

### Failure Mode 1: S2 Still Produces 400+ Trades After Optimization

**Root Cause:** Fusion threshold range too lenient (p95 instead of p99).

**Mitigation:**
1. Re-run distribution analysis with stricter percentile (p99.5 instead of p99)
2. Manually override fusion range to [0.90, 0.95]
3. If still fails → Mark S2 as PERMANENTLY DISABLED (current status)

### Failure Mode 2: OOS Validation Shows >50% PF Degradation

**Root Cause:** Overfitting to in-sample data.

**Mitigation:**
1. Reduce Optuna trial count (50 → 30) to limit search space exploration
2. Add L2 regularization to parameter sampling (prefer values near mean)
3. Use simpler search space (fewer parameters)

### Failure Mode 3: Pareto Frontier Contains Only 1-2 Solutions

**Root Cause:** Objectives are too correlated (PF and Sharpe tend to move together).

**Mitigation:**
1. Replace Sharpe with independent metric (e.g., Sortino, Calmar)
2. Add win rate as 4th objective
3. Increase trial count (50 → 100) to explore more trade-offs

### Failure Mode 4: Regime Routing Weights Not Applied

**Root Cause:** Config schema mismatch between optimizer output and RuntimeContext reader.

**Mitigation:**
1. Add config validation script (`bin/validate_config_schema.py`)
2. Unit test for RuntimeContext threshold resolution
3. Add debug logging to show applied weights per archetype

---

## Appendix B: Performance Benchmarks

**Expected Runtime (4-core machine, 16GB RAM):**

| Task | Duration | Parallelization |
|------|----------|-----------------|
| Distribution analysis (1 archetype) | 5 min | Sequential |
| Optuna optimization (50 trials, 3 folds) | 2-3 hours | 4 workers |
| OOS validation (5 configs) | 15 min | Sequential |
| Config generation | 1 min | Sequential |

**Total time per archetype:** ~3 hours

**Total time for S2 + S5:** ~6 hours

---

## Appendix C: Decision Tree for S2

**Question 1: Should we continue optimizing S2?**

```
Is S2 PF > 1.1 after 50 Optuna trials?
├─ YES → Proceed to OOS validation
│   └─ OOS PF > 1.1 AND degradation < 30%?
│       ├─ YES → Deploy S2 to production
│       └─ NO → Mark S2 as FAILED (same as current status)
└─ NO → Mark S2 as PERMANENTLY DISABLED
    └─ Document failure in CHANGELOG
    └─ Remove S2 from future optimization pipelines
```

**Current Status:** After 157 trials, S2 PF = 0.48 (baseline 0.38).
**Recommendation:** Skip S2 optimization in Phase 2, focus on S5 only.

---

**END OF ARCHITECTURE SPECIFICATION**

---

**Next Steps:**

1. **Review:** Architects review this spec for feasibility and completeness
2. **Approval:** Product owner approves scope and priorities
3. **Assignment:** Backend engineer assigned to Phase 1 implementation
4. **Kickoff:** Begin Phase 0 (OI_CHANGE fix) immediately
5. **Timeline:** Estimated 6 weeks (Phases 0-5)

**Contact:**
- Architecture questions: [System Architect]
- Implementation questions: [Backend Engineer]
- Product questions: [Product Owner]
