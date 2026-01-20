# Trading Archetype Testing Best Practices
## Comprehensive Research Report for Bull Machine

**Generated:** 2025-12-15
**Purpose:** Validate 19 trading archetypes produce diverse, realistic trades
**Research Sources:** Context7 (backtrader, vectorbt, freqtrade, scipy), Web Research, Industry Best Practices

---

## Executive Summary

This report provides a comprehensive testing framework for validating trading system archetypes. Based on research from leading backtesting libraries (backtrader, vectorbt, zipline, freqtrade) and industry best practices, it addresses three critical questions:

1. **How to validate archetypes produce DIFFERENT trades** (diversity metrics)
2. **How to validate trades are REALISTIC** (sanity checks)
3. **How to implement efficient SMOKE TESTS** (quick validation)

---

## 1. Testing Framework Recommendation

### Recommended Framework: **VectorBT** (with Pandas/SciPy)

**Why VectorBT?**

1. **Array-based vectorization** - Test 19 archetypes simultaneously
2. **Built-in strategy comparison** - Native support for multi-strategy portfolios
3. **Performance metrics** - Comprehensive statistics (Sharpe, Sortino, Calmar, etc.)
4. **Correlation analysis** - Easy pairwise strategy comparison
5. **Speed** - 10-100x faster than event-driven frameworks for parameter sweeps

**Alternative Frameworks:**

| Framework | Pros | Cons | Best For |
|-----------|------|------|----------|
| **VectorBT** | Fast, vectorized, great for multiple strategies | Limited event-driven features | Parameter optimization, strategy comparison |
| **Backtrader** | Mature, event-driven, good documentation | Slower for large parameter spaces | Production deployment, live trading |
| **Zipline** | Industry-standard, PyData integration | Complex setup, no longer actively maintained | Academic research, Quantopian-style analysis |
| **Freqtrade** | Built for crypto, active community | Opinionated architecture | Crypto-specific strategies |

**Recommendation for Bull Machine:**
- **Primary:** VectorBT for smoke tests and diversity validation
- **Secondary:** Your existing custom framework for production backtesting
- **Validation:** Use both to cross-validate results

---

## 2. Diversity Validation Methodology

### 2.1 Core Diversity Metrics

#### A. Trade Overlap Analysis

**Metric:** Pairwise trade overlap percentage

```python
def calculate_trade_overlap(trades_a, trades_b):
    """
    Calculate percentage of trades that occur on same bars.

    Args:
        trades_a: Boolean series of entry signals for archetype A
        trades_b: Boolean series of entry signals for archetype B

    Returns:
        overlap_pct: Percentage of overlapping trades (0-100)
    """
    total_trades_a = trades_a.sum()
    total_trades_b = trades_b.sum()

    if total_trades_a == 0 or total_trades_b == 0:
        return 0.0

    # Count bars where both fired
    overlap = (trades_a & trades_b).sum()

    # Calculate as percentage of smaller strategy's trades
    min_trades = min(total_trades_a, total_trades_b)
    overlap_pct = (overlap / min_trades) * 100

    return overlap_pct
```

**Healthy Thresholds:**
- **< 20% overlap**: Excellent diversity (ideal for unrelated archetypes)
- **20-40% overlap**: Good diversity (acceptable for similar archetypes)
- **40-60% overlap**: Moderate overlap (acceptable only for variations of same pattern)
- **> 60% overlap**: Poor diversity (indicates redundant strategies)

**Special Cases:**
- Related archetypes (e.g., "trap_reversal" vs "failed_continuation"): 30-50% acceptable
- Completely different archetypes (e.g., "order_block_retest" vs "liquidity_vacuum"): < 15% target

#### B. Return Correlation Analysis

**Metric:** Pearson and Spearman correlation of returns

```python
import pandas as pd
from scipy.stats import pearsonr, spearmanr

def calculate_return_correlation(pf_dict):
    """
    Calculate correlation matrix for strategy returns.

    Args:
        pf_dict: Dict mapping archetype names to Portfolio objects

    Returns:
        corr_matrix: DataFrame with correlation coefficients
        p_values: DataFrame with statistical significance
    """
    # Extract returns for each strategy
    returns = pd.DataFrame({
        name: pf.returns
        for name, pf in pf_dict.items()
    })

    # Pearson correlation (linear relationships)
    pearson_corr = returns.corr(method='pearson')

    # Spearman correlation (monotonic relationships)
    spearman_corr = returns.corr(method='spearman')

    # Calculate p-values for significance
    n_strategies = len(pf_dict)
    p_values = pd.DataFrame(
        index=pearson_corr.index,
        columns=pearson_corr.columns,
        dtype=float
    )

    for i in pearson_corr.index:
        for j in pearson_corr.columns:
            if i != j:
                _, p_val = pearsonr(returns[i], returns[j])
                p_values.loc[i, j] = p_val
            else:
                p_values.loc[i, j] = 0.0

    return pearson_corr, spearman_corr, p_values
```

**Healthy Thresholds:**
- **Correlation < 0.3**: Excellent diversity
- **Correlation 0.3-0.5**: Good diversity
- **Correlation 0.5-0.7**: Moderate (strategies somewhat similar)
- **Correlation > 0.7**: Poor diversity (strategies too similar)

**Red Flags:**
- Correlation > 0.85: Strategies are nearly identical
- p-value < 0.05 with high correlation: Statistically significant redundancy

#### C. Entry Distribution Analysis

**Metric:** Chi-square test for temporal distribution

```python
from scipy.stats import chisquare
import numpy as np

def test_entry_distribution_diversity(archetype_entries_dict, n_bins=20):
    """
    Test if archetypes fire in different temporal regions.

    Args:
        archetype_entries_dict: Dict of archetype -> boolean entry signals
        n_bins: Number of time bins to divide data into

    Returns:
        diversity_score: 0-100 (higher = more diverse)
        chi_square_results: Statistical test results
    """
    n_bars = len(next(iter(archetype_entries_dict.values())))
    bin_size = n_bars // n_bins

    # Count entries per bin for each archetype
    bin_counts = {}
    for arch_name, entries in archetype_entries_dict.items():
        counts = []
        for i in range(n_bins):
            start = i * bin_size
            end = start + bin_size
            counts.append(entries.iloc[start:end].sum())
        bin_counts[arch_name] = np.array(counts)

    # Calculate pairwise chi-square tests
    chi_square_results = {}
    for arch_a in bin_counts:
        for arch_b in bin_counts:
            if arch_a < arch_b:  # Avoid duplicates
                chi2, p_value = chisquare(
                    bin_counts[arch_a] + 1,  # +1 to avoid zeros
                    bin_counts[arch_b] + 1
                )
                chi_square_results[f"{arch_a}_vs_{arch_b}"] = {
                    'chi2': chi2,
                    'p_value': p_value,
                    'different': p_value < 0.05
                }

    # Diversity score: % of pairs with different distributions
    n_different = sum(1 for r in chi_square_results.values() if r['different'])
    n_pairs = len(chi_square_results)
    diversity_score = (n_different / n_pairs) * 100 if n_pairs > 0 else 0

    return diversity_score, chi_square_results
```

**Healthy Thresholds:**
- **> 70% different**: Excellent temporal diversity
- **50-70% different**: Good diversity
- **< 50% different**: Poor diversity (archetypes clustering on same events)

#### D. Feature Dependency Analysis

**Metric:** Mutual information and feature importance overlap

```python
from sklearn.feature_selection import mutual_info_classif

def analyze_feature_dependencies(df, archetype_entries_dict):
    """
    Identify which features drive each archetype.

    Args:
        df: DataFrame with all features
        archetype_entries_dict: Dict of archetype -> entry signals

    Returns:
        feature_importance_matrix: DataFrame showing which features matter per archetype
        dependency_overlap: Pairwise overlap in feature dependencies
    """
    feature_cols = [c for c in df.columns if c not in ['timestamp', 'close', 'open', 'high', 'low', 'volume']]

    feature_importance = {}
    for arch_name, entries in archetype_entries_dict.items():
        # Calculate mutual information between features and entries
        mi_scores = mutual_info_classif(
            df[feature_cols].fillna(0),
            entries.astype(int),
            random_state=42
        )

        # Normalize to 0-1
        mi_scores = mi_scores / mi_scores.max() if mi_scores.max() > 0 else mi_scores
        feature_importance[arch_name] = pd.Series(mi_scores, index=feature_cols)

    importance_df = pd.DataFrame(feature_importance)

    # Calculate pairwise feature dependency overlap
    dependency_overlap = pd.DataFrame(
        index=archetype_entries_dict.keys(),
        columns=archetype_entries_dict.keys(),
        dtype=float
    )

    for arch_a in importance_df.columns:
        for arch_b in importance_df.columns:
            # Get top 5 features for each
            top_a = set(importance_df[arch_a].nlargest(5).index)
            top_b = set(importance_df[arch_b].nlargest(5).index)

            # Calculate Jaccard similarity
            overlap = len(top_a & top_b) / len(top_a | top_b) if len(top_a | top_b) > 0 else 0
            dependency_overlap.loc[arch_a, arch_b] = overlap

    return importance_df, dependency_overlap
```

**Healthy Thresholds:**
- **< 0.4 overlap**: Excellent feature diversity
- **0.4-0.6 overlap**: Acceptable
- **> 0.6 overlap**: Strategies using same features (low diversity)

### 2.2 Comprehensive Diversity Test Suite

```python
def comprehensive_diversity_test(
    df,
    archetype_entries_dict,
    portfolios_dict,
    overlap_threshold=40,
    correlation_threshold=0.5
):
    """
    Run complete diversity validation suite.

    Args:
        df: OHLCV dataframe with features
        archetype_entries_dict: Dict of archetype -> boolean entries
        portfolios_dict: Dict of archetype -> Portfolio objects
        overlap_threshold: Max acceptable trade overlap %
        correlation_threshold: Max acceptable correlation

    Returns:
        results: Dict with all test results
        passed: Boolean indicating if all tests passed
    """
    results = {
        'trade_overlap': {},
        'return_correlation': {},
        'temporal_distribution': {},
        'feature_dependency': {},
        'summary': {}
    }

    archetypes = list(archetype_entries_dict.keys())

    # 1. Trade Overlap Analysis
    print("Testing trade overlap...")
    overlap_matrix = pd.DataFrame(
        index=archetypes,
        columns=archetypes,
        dtype=float
    )

    for i, arch_a in enumerate(archetypes):
        for j, arch_b in enumerate(archetypes):
            if i <= j:
                overlap = calculate_trade_overlap(
                    archetype_entries_dict[arch_a],
                    archetype_entries_dict[arch_b]
                )
                overlap_matrix.loc[arch_a, arch_b] = overlap
                overlap_matrix.loc[arch_b, arch_a] = overlap

    results['trade_overlap']['matrix'] = overlap_matrix

    # Count violations (excluding diagonal)
    violations = 0
    for i, arch_a in enumerate(archetypes):
        for j, arch_b in enumerate(archetypes):
            if i < j:
                if overlap_matrix.loc[arch_a, arch_b] > overlap_threshold:
                    violations += 1
                    print(f"  WARNING: {arch_a} vs {arch_b}: {overlap_matrix.loc[arch_a, arch_b]:.1f}% overlap")

    results['trade_overlap']['violations'] = violations
    results['trade_overlap']['passed'] = violations == 0

    # 2. Return Correlation Analysis
    print("\nTesting return correlations...")
    pearson_corr, spearman_corr, p_values = calculate_return_correlation(portfolios_dict)

    results['return_correlation']['pearson'] = pearson_corr
    results['return_correlation']['spearman'] = spearman_corr
    results['return_correlation']['p_values'] = p_values

    corr_violations = 0
    for i, arch_a in enumerate(archetypes):
        for j, arch_b in enumerate(archetypes):
            if i < j:
                corr = pearson_corr.loc[arch_a, arch_b]
                if abs(corr) > correlation_threshold:
                    corr_violations += 1
                    print(f"  WARNING: {arch_a} vs {arch_b}: {corr:.3f} correlation")

    results['return_correlation']['violations'] = corr_violations
    results['return_correlation']['passed'] = corr_violations == 0

    # 3. Temporal Distribution Analysis
    print("\nTesting temporal distribution diversity...")
    diversity_score, chi_square_results = test_entry_distribution_diversity(
        archetype_entries_dict
    )

    results['temporal_distribution']['diversity_score'] = diversity_score
    results['temporal_distribution']['chi_square_results'] = chi_square_results
    results['temporal_distribution']['passed'] = diversity_score >= 50

    print(f"  Diversity score: {diversity_score:.1f}%")

    # 4. Feature Dependency Analysis
    print("\nTesting feature dependency overlap...")
    importance_df, dependency_overlap = analyze_feature_dependencies(
        df, archetype_entries_dict
    )

    results['feature_dependency']['importance'] = importance_df
    results['feature_dependency']['overlap'] = dependency_overlap

    # 5. Summary
    results['summary'] = {
        'total_archetypes': len(archetypes),
        'trade_overlap_violations': violations,
        'correlation_violations': corr_violations,
        'temporal_diversity_score': diversity_score,
        'all_tests_passed': (
            results['trade_overlap']['passed'] and
            results['return_correlation']['passed'] and
            results['temporal_distribution']['passed']
        )
    }

    print("\n" + "="*60)
    print("DIVERSITY TEST SUMMARY")
    print("="*60)
    print(f"Total Archetypes: {len(archetypes)}")
    print(f"Trade Overlap Violations: {violations}")
    print(f"Correlation Violations: {corr_violations}")
    print(f"Temporal Diversity Score: {diversity_score:.1f}%")
    print(f"Overall Status: {'PASSED' if results['summary']['all_tests_passed'] else 'FAILED'}")
    print("="*60)

    return results, results['summary']['all_tests_passed']
```

---

## 3. Realism Validation Checks

### 3.1 Signal Quality Checks

```python
def validate_signal_realism(df, entries, exits, archetype_name):
    """
    Comprehensive realism checks for trading signals.

    Args:
        df: OHLCV dataframe
        entries: Boolean series of entry signals
        exits: Boolean series of exit signals
        archetype_name: Name of archetype being tested

    Returns:
        checks: Dict with all check results
        passed: Boolean indicating all checks passed
    """
    checks = {}

    # Check 1: Minimum signal count
    n_entries = entries.sum()
    checks['signal_count'] = {
        'value': n_entries,
        'passed': 10 <= n_entries <= len(df) * 0.2,  # 10 min, max 20% of bars
        'message': f"Signal count: {n_entries} (expect 10-{int(len(df)*0.2)})"
    }

    # Check 2: No consecutive entries without exit
    entry_indices = df[entries].index
    has_no_overlap = True
    for i in range(len(entry_indices) - 1):
        # Check if exit occurred between consecutive entries
        start = df.index.get_loc(entry_indices[i])
        end = df.index.get_loc(entry_indices[i+1])
        if not exits.iloc[start:end].any():
            has_no_overlap = False
            break

    checks['no_overlapping_positions'] = {
        'value': has_no_overlap,
        'passed': has_no_overlap,
        'message': "Entries don't overlap without exits" if has_no_overlap else "FAILED: Overlapping entries detected"
    }

    # Check 3: Entry/exit ratio sanity
    n_exits = exits.sum()
    ratio = n_exits / n_entries if n_entries > 0 else 0
    checks['entry_exit_ratio'] = {
        'value': ratio,
        'passed': 0.5 <= ratio <= 1.5,  # Allow some difference for open positions
        'message': f"Entry/Exit ratio: {ratio:.2f} (expect 0.5-1.5)"
    }

    # Check 4: Average holding period
    hold_periods = []
    open_entry_idx = None
    for idx, row in df.iterrows():
        if entries.loc[idx] and open_entry_idx is None:
            open_entry_idx = idx
        elif exits.loc[idx] and open_entry_idx is not None:
            bars_held = df.index.get_loc(idx) - df.index.get_loc(open_entry_idx)
            hold_periods.append(bars_held)
            open_entry_idx = None

    avg_hold = np.mean(hold_periods) if hold_periods else 0
    checks['holding_period'] = {
        'value': avg_hold,
        'passed': 1 <= avg_hold <= 500,  # 1 to 500 bars
        'message': f"Avg holding period: {avg_hold:.1f} bars (expect 1-500)"
    }

    # Check 5: Signal clustering (shouldn't all be in one region)
    entry_bars = df[entries].index
    if len(entry_bars) > 0:
        first_quarter_count = sum(1 for idx in entry_bars if df.index.get_loc(idx) < len(df) * 0.25)
        clustering_ratio = first_quarter_count / len(entry_bars)
        checks['temporal_clustering'] = {
            'value': clustering_ratio,
            'passed': clustering_ratio < 0.6,  # No more than 60% in first quarter
            'message': f"First quarter clustering: {clustering_ratio:.1%} (expect < 60%)"
        }
    else:
        checks['temporal_clustering'] = {
            'value': 0,
            'passed': False,
            'message': "No signals to analyze"
        }

    # Check 6: Not all entries on same day of week (if using hourly+ data)
    if 'timestamp' in df.columns:
        entry_timestamps = df.loc[entries, 'timestamp']
        if len(entry_timestamps) > 0:
            weekdays = entry_timestamps.dt.dayofweek
            weekday_counts = weekdays.value_counts()
            max_weekday_pct = weekday_counts.max() / len(weekdays)
            checks['weekday_distribution'] = {
                'value': max_weekday_pct,
                'passed': max_weekday_pct < 0.5,  # No more than 50% on one weekday
                'message': f"Max weekday concentration: {max_weekday_pct:.1%} (expect < 50%)"
            }
        else:
            checks['weekday_distribution'] = {'value': 0, 'passed': True, 'message': 'N/A'}

    # Overall pass/fail
    passed = all(check['passed'] for check in checks.values())

    # Print results
    print(f"\nRealism Checks for {archetype_name}:")
    print("-" * 60)
    for check_name, check_data in checks.items():
        status = "✓ PASS" if check_data['passed'] else "✗ FAIL"
        print(f"{status} {check_data['message']}")
    print("-" * 60)
    print(f"Overall: {'PASSED' if passed else 'FAILED'}\n")

    return checks, passed
```

### 3.2 Confidence Score Validation

```python
def validate_confidence_scores(df, entries, confidence_scores):
    """
    Validate that confidence scores are meaningful predictors.

    Args:
        df: OHLCV dataframe
        entries: Boolean entry signals
        confidence_scores: Float series of confidence values (0-1)

    Returns:
        validation_results: Dict with validation metrics
    """
    # Filter to only entries
    entry_confidence = confidence_scores[entries]

    if len(entry_confidence) == 0:
        return {'passed': False, 'reason': 'No entries to analyze'}

    results = {}

    # Check 1: Confidence range
    results['confidence_range'] = {
        'min': entry_confidence.min(),
        'max': entry_confidence.max(),
        'mean': entry_confidence.mean(),
        'std': entry_confidence.std(),
        'passed': (
            0.0 <= entry_confidence.min() <= 1.0 and
            0.0 <= entry_confidence.max() <= 1.0 and
            entry_confidence.std() > 0.05  # Some variation
        )
    }

    # Check 2: Not all same value (shows it's actually being calculated)
    unique_values = entry_confidence.nunique()
    results['score_diversity'] = {
        'unique_values': unique_values,
        'passed': unique_values > len(entry_confidence) * 0.3  # At least 30% unique
    }

    # Check 3: Correlation with outcome (if we have trade results)
    # This would require forward-looking trade PnL, skipped for smoke test

    results['overall_passed'] = all(
        r.get('passed', False) for r in results.values()
    )

    return results
```

### 3.3 Red Flags for Overfitting

```python
def detect_overfitting_red_flags(backtest_results, train_period, test_period):
    """
    Identify common overfitting patterns.

    Args:
        backtest_results: Dict with 'train' and 'test' results
        train_period: (start, end) dates for training
        test_period: (start, end) dates for testing

    Returns:
        red_flags: List of detected issues
        severity: 'low', 'medium', 'high'
    """
    red_flags = []

    train_metrics = backtest_results['train']
    test_metrics = backtest_results['test']

    # Red Flag 1: Huge performance drop
    sharpe_drop = train_metrics['sharpe_ratio'] - test_metrics['sharpe_ratio']
    if sharpe_drop > 1.0:
        red_flags.append({
            'flag': 'Severe Sharpe degradation',
            'severity': 'high',
            'details': f"Train Sharpe: {train_metrics['sharpe_ratio']:.2f}, Test Sharpe: {test_metrics['sharpe_ratio']:.2f}"
        })

    # Red Flag 2: Win rate collapse
    win_rate_drop = train_metrics['win_rate'] - test_metrics['win_rate']
    if win_rate_drop > 0.2:  # 20% drop
        red_flags.append({
            'flag': 'Win rate collapse',
            'severity': 'high',
            'details': f"Train WR: {train_metrics['win_rate']:.1%}, Test WR: {test_metrics['win_rate']:.1%}"
        })

    # Red Flag 3: Too few trades in test
    if test_metrics['n_trades'] < 10:
        red_flags.append({
            'flag': 'Insufficient test trades',
            'severity': 'medium',
            'details': f"Only {test_metrics['n_trades']} trades in test period"
        })

    # Red Flag 4: Perfect training performance
    if train_metrics['win_rate'] > 0.8:
        red_flags.append({
            'flag': 'Unrealistic training performance',
            'severity': 'medium',
            'details': f"Win rate too high: {train_metrics['win_rate']:.1%}"
        })

    # Red Flag 5: Negative test returns
    if test_metrics['total_return'] < 0:
        red_flags.append({
            'flag': 'Negative out-of-sample returns',
            'severity': 'high',
            'details': f"Test return: {test_metrics['total_return']:.2%}"
        })

    # Determine overall severity
    if any(f['severity'] == 'high' for f in red_flags):
        severity = 'high'
    elif any(f['severity'] == 'medium' for f in red_flags):
        severity = 'medium'
    else:
        severity = 'low'

    return red_flags, severity
```

---

## 4. Smoke Test Strategy

### 4.1 Recommended Smoke Test Configuration

**Test Period Selection:**

```python
# Smoke test should cover multiple market regimes
SMOKE_TEST_CONFIG = {
    'periods': [
        {
            'name': 'bull_2024',
            'start': '2024-01-01',
            'end': '2024-06-30',
            'expected_regime': 'bull',
            'min_signals_per_archetype': 3
        },
        {
            'name': 'bear_2022',
            'start': '2022-05-01',
            'end': '2022-12-31',
            'expected_regime': 'bear',
            'min_signals_per_archetype': 3
        },
        {
            'name': 'neutral_2023',
            'start': '2023-06-01',
            'end': '2023-12-31',
            'expected_regime': 'neutral',
            'min_signals_per_archetype': 2
        }
    ],
    'assets': ['BTC-USD'],  # Single asset for speed
    'timeframe': '1h',  # Fast enough to run quickly
    'min_total_signals_per_archetype': 10,
    'max_runtime_seconds': 60  # Should complete in 1 minute
}
```

**Why These Periods?**
- **Bear 2022:** Tests archetypes in harsh downtrend
- **Bull 2024:** Tests in strong uptrend
- **Neutral 2023:** Tests in range-bound conditions
- **Total ~18 months:** Enough data, fast to run

### 4.2 Smoke Test Script

```python
#!/usr/bin/env python3
"""
Smoke test for 19 trading archetypes.
Quick validation before full backtesting.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import time


class ArchetypeSmokeTest:
    """Fast validation suite for trading archetypes."""

    def __init__(self, config):
        self.config = config
        self.results = {}
        self.start_time = None

    def run(self, archetypes_dict, data_loader):
        """
        Run smoke test on all archetypes.

        Args:
            archetypes_dict: Dict mapping archetype_name -> ArchetypeClass
            data_loader: Function to load OHLCV data

        Returns:
            results: Dict with test results
            passed: Boolean indicating all tests passed
        """
        self.start_time = time.time()
        print("="*70)
        print("ARCHETYPE SMOKE TEST")
        print("="*70)

        all_passed = True

        for period_config in self.config['periods']:
            print(f"\nTesting period: {period_config['name']}")
            print(f"  {period_config['start']} to {period_config['end']}")
            print(f"  Expected regime: {period_config['expected_regime']}")

            # Load data
            df = data_loader(
                period_config['start'],
                period_config['end'],
                self.config['timeframe']
            )

            period_results = {}

            for arch_name, archetype in archetypes_dict.items():
                # Run archetype
                entries, exits, confidence = archetype.generate_signals(df)

                # Validate signals
                signal_checks, signal_passed = validate_signal_realism(
                    df, entries, exits, arch_name
                )

                # Validate confidence
                confidence_checks = validate_confidence_scores(
                    df, entries, confidence
                )

                n_signals = entries.sum()

                period_results[arch_name] = {
                    'n_signals': n_signals,
                    'signal_checks_passed': signal_passed,
                    'confidence_checks_passed': confidence_checks['overall_passed'],
                    'meets_minimum': n_signals >= period_config['min_signals_per_archetype']
                }

                # Print quick summary
                status = "✓" if period_results[arch_name]['meets_minimum'] else "✗"
                print(f"  {status} {arch_name}: {n_signals} signals")

                if not period_results[arch_name]['meets_minimum']:
                    all_passed = False

            self.results[period_config['name']] = period_results

        # Cross-period validation
        print("\n" + "="*70)
        print("CROSS-PERIOD VALIDATION")
        print("="*70)

        total_signals_per_archetype = {}
        for arch_name in archetypes_dict.keys():
            total = sum(
                self.results[period]['arch_name']['n_signals']
                for period in self.results
            )
            total_signals_per_archetype[arch_name] = total

            meets_total = total >= self.config['min_total_signals_per_archetype']
            status = "✓" if meets_total else "✗"
            print(f"{status} {arch_name}: {total} total signals (need {self.config['min_total_signals_per_archetype']})")

            if not meets_total:
                all_passed = False

        # Runtime check
        elapsed = time.time() - self.start_time
        print(f"\nRuntime: {elapsed:.1f}s (limit: {self.config['max_runtime_seconds']}s)")

        if elapsed > self.config['max_runtime_seconds']:
            print("✗ FAILED: Smoke test too slow")
            all_passed = False
        else:
            print("✓ PASSED: Runtime acceptable")

        # Final summary
        print("\n" + "="*70)
        print(f"SMOKE TEST: {'PASSED' if all_passed else 'FAILED'}")
        print("="*70)

        return self.results, all_passed


# Usage example
if __name__ == '__main__':
    # Import your archetypes
    from engine.archetypes import (
        TrapReversal,
        OrderBlockRetest,
        FVGContinuation,
        # ... import all 19 archetypes
    )

    # Create archetype instances
    archetypes = {
        'trap_reversal': TrapReversal(config),
        'order_block_retest': OrderBlockRetest(config),
        'fvg_continuation': FVGContinuation(config),
        # ... add all 19 archetypes
    }

    # Define data loader
    def load_data(start, end, timeframe):
        # Your data loading logic
        pass

    # Run smoke test
    smoke_test = ArchetypeSmokeTest(SMOKE_TEST_CONFIG)
    results, passed = smoke_test.run(archetypes, load_data)

    # Exit with appropriate code
    import sys
    sys.exit(0 if passed else 1)
```

### 4.3 Metrics to Track in Smoke Test

```python
SMOKE_TEST_METRICS = {
    'per_archetype': [
        'n_signals',
        'signal_frequency',  # signals per 100 bars
        'avg_confidence',
        'confidence_std',
        'passed_realism_checks'
    ],
    'pairwise': [
        'trade_overlap_pct',
        'return_correlation',
        'feature_overlap'
    ],
    'overall': [
        'total_archetypes_tested',
        'archetypes_passed',
        'archetypes_failed',
        'avg_diversity_score',
        'runtime_seconds'
    ]
}
```

---

## 5. Success Criteria

### 5.1 Individual Archetype Pass Criteria

An archetype passes smoke test if:

1. **Signal Count:** 10-20% of total bars have signals
2. **Signal Distribution:** Not clustered in < 25% of time period
3. **Entry/Exit Balance:** Ratio between 0.5-1.5
4. **Confidence Scores:** Range 0-1 with stddev > 0.05
5. **Holding Period:** 1-500 bars average
6. **No Overlap:** Entries don't occur without prior exit

### 5.2 Portfolio-Level Pass Criteria

The full archetype suite passes if:

1. **Diversity Metrics:**
   - Average pairwise trade overlap < 30%
   - Average return correlation < 0.4
   - Temporal diversity score > 60%
   - Feature dependency overlap < 0.5

2. **Coverage:**
   - At least 15/19 archetypes generate signals
   - Each market regime (bull/bear/neutral) triggers different archetypes
   - Total unique trading days covered > 50% of test period

3. **Performance:**
   - At least 10/19 archetypes have positive Sharpe ratio
   - Portfolio Sharpe > 1.0
   - Max drawdown < 30%

4. **Realism:**
   - No archetype has > 70% win rate
   - All archetypes pass no-lookahead tests
   - Confidence scores correlate with outcomes (r > 0.2)

### 5.3 Walk-Forward Validation Criteria

For production readiness:

```python
WALK_FORWARD_CRITERIA = {
    'efficiency_ratio': 0.5,  # WFE > 50%
    'min_oos_periods': 5,  # At least 5 out-of-sample windows
    'max_sharpe_degradation': 0.5,  # IS Sharpe - OOS Sharpe < 0.5
    'min_oos_trades_per_period': 10,
    'correlation_stability': 0.7  # Correlation of IS vs OOS metrics
}

def validate_walk_forward(wf_results):
    """
    Validate walk-forward optimization results.

    Args:
        wf_results: Dict with in-sample and out-of-sample results

    Returns:
        passed: Boolean
        metrics: Dict with validation metrics
    """
    metrics = {}

    # Calculate Walk Forward Efficiency
    is_returns = [r['return'] for r in wf_results['in_sample']]
    oos_returns = [r['return'] for r in wf_results['out_of_sample']]

    avg_is_return = np.mean(is_returns)
    avg_oos_return = np.mean(oos_returns)

    wfe = avg_oos_return / avg_is_return if avg_is_return != 0 else 0
    metrics['wfe'] = wfe

    # Sharpe degradation
    is_sharpe = np.mean([r['sharpe'] for r in wf_results['in_sample']])
    oos_sharpe = np.mean([r['sharpe'] for r in wf_results['out_of_sample']])
    metrics['sharpe_degradation'] = is_sharpe - oos_sharpe

    # Correlation of metrics
    metrics['return_correlation'] = np.corrcoef(is_returns, oos_returns)[0, 1]

    # Pass/fail
    passed = (
        wfe >= WALK_FORWARD_CRITERIA['efficiency_ratio'] and
        len(wf_results['out_of_sample']) >= WALK_FORWARD_CRITERIA['min_oos_periods'] and
        metrics['sharpe_degradation'] <= WALK_FORWARD_CRITERIA['max_sharpe_degradation']
    )

    return passed, metrics
```

---

## 6. Implementation Roadmap

### Phase 1: Setup (Week 1)
- [ ] Install VectorBT and dependencies
- [ ] Create data pipeline for smoke test periods
- [ ] Implement basic diversity metrics (overlap, correlation)

### Phase 2: Smoke Test (Week 2)
- [ ] Build smoke test harness
- [ ] Test all 19 archetypes individually
- [ ] Validate signal count and distribution
- [ ] Check for lookahead bias

### Phase 3: Diversity Validation (Week 3)
- [ ] Implement pairwise overlap analysis
- [ ] Calculate return correlations
- [ ] Test temporal distribution diversity
- [ ] Analyze feature dependencies

### Phase 4: Realism Checks (Week 4)
- [ ] Implement realism validation suite
- [ ] Test confidence score meaningfulness
- [ ] Check for overfitting red flags
- [ ] Validate entry/exit logic

### Phase 5: Integration (Week 5)
- [ ] Integrate smoke test into CI/CD
- [ ] Create automated reporting
- [ ] Set up alerts for failed tests
- [ ] Document results and thresholds

---

## 7. Tools and Libraries

```python
# requirements.txt
vectorbt>=0.24.0
pandas>=1.5.0
numpy>=1.23.0
scipy>=1.9.0
scikit-learn>=1.2.0
matplotlib>=3.6.0
seaborn>=0.12.0
pytest>=7.0.0
```

### Recommended File Structure

```
tests/
├── smoke/
│   ├── test_archetype_smoke.py
│   ├── test_diversity_metrics.py
│   └── test_realism_checks.py
├── integration/
│   ├── test_full_backtest.py
│   └── test_walk_forward.py
├── unit/
│   ├── test_individual_archetypes.py
│   └── test_feature_engineering.py
└── fixtures/
    ├── market_data.py
    └── archetype_configs.py

engine/
├── archetypes/
│   ├── __init__.py
│   ├── base.py
│   ├── trap_reversal.py
│   ├── order_block_retest.py
│   └── ... (all 19 archetypes)
└── testing/
    ├── diversity_metrics.py
    ├── realism_checks.py
    └── smoke_test.py
```

---

## 8. Key Takeaways

### For Your 19 Archetypes:

1. **Diversity is Critical**
   - Aim for < 30% trade overlap between unrelated archetypes
   - Keep return correlation < 0.4
   - Ensure archetypes use different feature sets

2. **Realism Over Perfection**
   - 60-70% win rate is realistic; 80%+ is suspicious
   - Allow for drawdowns and losing streaks
   - Validate on multiple market regimes

3. **Smoke Tests Save Time**
   - Run on 6-12 months of data
   - Should complete in < 60 seconds
   - Catches 80% of issues before full backtest

4. **Walk-Forward is Gold Standard**
   - WFE > 50-60% indicates robustness
   - Test on at least 5 OOS periods
   - Monitor metric stability across periods

5. **Continuous Validation**
   - Integrate smoke tests into CI/CD
   - Re-run diversity metrics monthly
   - Track correlation drift over time

### Common Pitfalls to Avoid:

- **Over-optimization:** Too many parameters = overfitting
- **Data snooping:** Testing on same period repeatedly
- **Ignoring correlation:** Similar strategies reduce portfolio benefit
- **Unrealistic expectations:** Perfect backtests don't work live
- **Single regime testing:** Must test bull, bear, and neutral markets

---

## 9. References

### Research Sources:

1. **VectorBT Documentation** - Strategy comparison and portfolio analysis
2. **Backtrader Documentation** - Backtesting best practices
3. **Freqtrade Documentation** - Lookahead analysis and validation
4. **SciPy Documentation** - Statistical correlation tests
5. **Industry Research:**
   - Walk-forward optimization methodology
   - Strategy diversification metrics
   - Robustness testing frameworks
   - Statistical significance in trading systems

### Key Papers & Articles:

- "Backtesting" by Campbell R. Harvey (CME Group)
- Walk-Forward Optimization (Wikipedia, QuantInsti)
- Strategy Diversification Metrics (QuantifiedStrategies)
- Robustness Testing Guide (Build Alpha)

---

## Appendix A: Complete Example

See `/Users/raymondghandchi/Bull-machine-/Bull-machine-/tests/smoke/test_archetype_diversity.py` for full implementation.

## Appendix B: Visualization Tools

```python
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_diversity_matrix(overlap_matrix, title="Trade Overlap Matrix"):
    """Create heatmap of trade overlap between archetypes."""
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        overlap_matrix,
        annot=True,
        fmt='.1f',
        cmap='RdYlGn_r',
        vmin=0,
        vmax=100,
        cbar_kws={'label': 'Overlap %'}
    )
    plt.title(title)
    plt.xlabel('Archetype')
    plt.ylabel('Archetype')
    plt.tight_layout()
    plt.savefig(f'{title.replace(" ", "_")}.png', dpi=150)
    plt.close()

def visualize_correlation_matrix(corr_matrix, title="Return Correlation Matrix"):
    """Create heatmap of return correlations."""
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        vmin=-1,
        vmax=1,
        center=0,
        cbar_kws={'label': 'Correlation'}
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f'{title.replace(" ", "_")}.png', dpi=150)
    plt.close()
```

---

**End of Report**

*For questions or clarifications, refer to the inline code comments or consult the referenced documentation.*
