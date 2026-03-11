#!/usr/bin/env python3
"""
Post-Optimization Validation Suite

Runs a full backtest with merged Optuna params, then validates with:
1. Full-period backtest (2020-2024) vs baseline
2. OOS backtest (2023-2024) vs baseline
3. Deflated Sharpe Ratio (DSR) — accounts for multiple testing
4. Monte Carlo trade bootstrapping (10K resamples)
5. Parameter boundary check

Usage:
    python3 bin/validate_optimization.py
    python3 bin/validate_optimization.py --mc-sims 50000
    python3 bin/validate_optimization.py --html  # generate quantstats report (if installed)

Author: Claude Code
Date: 2026-03-10
"""

import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "bull_machine_isolated_v11_fixed.json"
FEATURE_STORE = PROJECT_ROOT / "data" / "features_mtf" / "BTC_1H_FEATURES_V12_ENHANCED.parquet"

# Baseline from last full backtest (pre-optimization)
BASELINE = {
    'full': {
        'profit_factor': 1.547,
        'total_trades': 972,
        'win_rate': 77.78,
        'max_drawdown': -13.55,
        'sharpe_ratio': 1.14,
        'total_pnl': 148133,
    },
}

# ─────────────────────────────────────────────────────────────────────
# Backtest runner
# ─────────────────────────────────────────────────────────────────────

def run_backtest(config, features_df, start_date, end_date,
                 initial_cash=100_000.0, commission_rate=0.0002, slippage_bps=3.0):
    """Run standalone backtest and return stats + trade log."""
    from bin.backtest_v11_standalone import StandaloneBacktestEngine

    engine = StandaloneBacktestEngine(
        config=config, initial_cash=initial_cash,
        commission_rate=commission_rate, slippage_bps=slippage_bps,
        features_df=features_df,
    )
    engine.run(start_date=start_date, end_date=end_date)
    stats = engine.get_performance_stats()

    # Get trade log
    trade_log = []
    if hasattr(engine, 'pnl_tracker') and hasattr(engine.pnl_tracker, 'completed_trades'):
        for t in engine.pnl_tracker.completed_trades:
            trade_log.append({
                'archetype': getattr(t, 'archetype', 'unknown'),
                'pnl': getattr(t, 'pnl', 0),
                'entry_time': getattr(t, 'entry_time', None),
                'exit_time': getattr(t, 'exit_time', None),
                'direction': getattr(t, 'direction', 'long'),
                'hold_hours': getattr(t, 'hold_hours', 0),
            })

    # Get equity curve
    equity_curve = []
    if hasattr(engine, 'pnl_tracker') and hasattr(engine.pnl_tracker, 'equity_curve'):
        equity_curve = engine.pnl_tracker.equity_curve

    return stats, trade_log, equity_curve


# ─────────────────────────────────────────────────────────────────────
# Deflated Sharpe Ratio (DSR)
# ─────────────────────────────────────────────────────────────────────

def compute_dsr(observed_sharpe, all_trial_sharpes, n_obs, skew=0.0, kurtosis=3.0):
    """
    Compute the Deflated Sharpe Ratio.

    Based on Bailey & Lopez de Prado (2014):
    "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting,
     and Non-Normality"

    Args:
        observed_sharpe: Best trial's Sharpe ratio
        all_trial_sharpes: List of Sharpe ratios from all trials
        n_obs: Number of observations (bars or returns)
        skew: Skewness of returns
        kurtosis: Kurtosis of returns (excess kurtosis, normal=3)

    Returns:
        DSR probability (0-1), higher is better
    """
    from scipy import stats as scipy_stats

    n_trials = len(all_trial_sharpes)
    if n_trials < 2 or n_obs < 10:
        return 0.0

    # Expected maximum Sharpe under null (all strategies are random)
    # E[max(SR)] ≈ (1 - γ) * Φ^(-1)(1 - 1/N) + γ * Φ^(-1)(1 - 1/(N*e))
    # Simplified: E[max] ≈ sqrt(2 * ln(N)) - (ln(pi) + ln(ln(N))) / (2 * sqrt(2 * ln(N)))
    if n_trials > 1:
        ln_n = np.log(n_trials)
        expected_max_sr = np.sqrt(2 * ln_n) - (np.log(np.pi) + np.log(ln_n)) / (2 * np.sqrt(2 * ln_n))
    else:
        expected_max_sr = 0.0

    # Standard error of Sharpe ratio (accounting for non-normality)
    se_sr = np.sqrt(
        (1 + 0.5 * observed_sharpe**2 - skew * observed_sharpe +
         ((kurtosis - 3) / 4.0) * observed_sharpe**2) / (n_obs - 1)
    )

    if se_sr < 1e-10:
        return 0.0

    # DSR = P(SR > E[max(SR)]) using the PSR formula
    z_score = (observed_sharpe - expected_max_sr) / se_sr
    dsr = scipy_stats.norm.cdf(z_score)

    return dsr


# ─────────────────────────────────────────────────────────────────────
# Monte Carlo Trade Bootstrapping
# ─────────────────────────────────────────────────────────────────────

def monte_carlo_bootstrap(trade_pnls, n_simulations=10000, initial_cash=100_000.0,
                          skip_pct=0.10, seed=42):
    """
    Monte Carlo validation by resampling trades.

    Methods:
    1. Reshuffle: Randomize order of trades
    2. Skip: Randomly skip 10% of trades each simulation

    Returns dict with percentile distributions of key metrics.
    """
    rng = np.random.RandomState(seed)
    pnls = np.array(trade_pnls, dtype=float)
    n_trades = len(pnls)

    if n_trades < 10:
        return {'error': 'Too few trades for Monte Carlo'}

    results = {
        'final_equity': [],
        'max_drawdown_pct': [],
        'profit_factor': [],
        'sharpe': [],
        'total_return_pct': [],
    }

    for _ in range(n_simulations):
        # Randomly skip trades
        mask = rng.random(n_trades) > skip_pct
        sim_pnls = pnls[mask].copy()

        # Shuffle order
        rng.shuffle(sim_pnls)

        if len(sim_pnls) < 5:
            continue

        # Compute equity curve
        equity = initial_cash + np.cumsum(sim_pnls)
        equity_with_start = np.concatenate([[initial_cash], equity])

        # Max drawdown
        running_max = np.maximum.accumulate(equity_with_start)
        drawdowns = (equity_with_start - running_max) / running_max * 100
        max_dd = drawdowns.min()

        # Profit factor
        wins = sim_pnls[sim_pnls > 0].sum()
        losses = abs(sim_pnls[sim_pnls < 0].sum())
        pf = wins / losses if losses > 0 else float('inf')

        # Sharpe (annualized, assuming ~8760 hours/year, avg hold ~36h)
        if sim_pnls.std() > 0:
            trades_per_year = len(sim_pnls) / 5.0  # ~5 years of data
            sharpe = (sim_pnls.mean() / sim_pnls.std()) * np.sqrt(trades_per_year)
        else:
            sharpe = 0.0

        final_eq = equity[-1]
        total_ret = (final_eq - initial_cash) / initial_cash * 100

        results['final_equity'].append(final_eq)
        results['max_drawdown_pct'].append(max_dd)
        results['profit_factor'].append(min(pf, 10.0))  # Cap for display
        results['sharpe'].append(sharpe)
        results['total_return_pct'].append(total_ret)

    # Compute percentiles
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    summary = {}
    for metric, values in results.items():
        if not values:
            continue
        arr = np.array(values)
        summary[metric] = {
            'mean': float(arr.mean()),
            'std': float(arr.std()),
            'percentiles': {p: float(np.percentile(arr, p)) for p in percentiles},
        }

    return summary


# ─────────────────────────────────────────────────────────────────────
# Parameter Boundary Check
# ─────────────────────────────────────────────────────────────────────

def check_param_boundaries():
    """Check if any optimized params are at their search space boundaries."""
    # Group definitions from optuna_parallel_group.py
    groups = {
        1: {
            'bt_wick_trap': (0.03, 0.18),
            'bt_retest_cluster': (0.03, 0.15),
            'bt_liquidity_sweep': (0.03, 0.18),
            'bt_trap_within_trend': (0.03, 0.15),
        },
        2: {
            'bt_spring': (0.03, 0.18),
            'bt_failed_continuation': (0.03, 0.18),
            'bt_order_block_retest': (0.03, 0.18),
            'bt_liquidity_vacuum': (0.03, 0.15),
        },
        3: {
            'bt_funding_divergence': (0.03, 0.20),
            'bt_long_squeeze': (0.03, 0.20),
            'bt_fvg_continuation': (0.03, 0.20),
            'bt_exhaustion_reversal': (0.03, 0.20),
        },
        5: {
            'wick_pct_K': (0.20, 0.50),
            'wick_pct_G': (0.20, 0.50),
            'vol_z_L': (0.5, 2.0),
            'rsi_upper_L': (65.0, 80.0),
            'rsi_lower_L': (20.0, 35.0),
            'rsi_upper_F': (72.0, 85.0),
            'rsi_lower_F': (15.0, 28.0),
            'bos_atr_B': (0.8, 3.0),
            'funding_z_S4': (-2.0, -0.5),
            'funding_z_S5': (0.5, 2.0),
        },
        6: {
            'atr_stop_wick_trap': (1.5, 5.0),
            'atr_tp_wick_trap': (2.0, 8.0),
            'atr_stop_retest_cluster': (1.5, 5.0),
            'atr_tp_retest_cluster': (2.0, 8.0),
            'atr_stop_liquidity_sweep': (1.5, 5.0),
            'atr_tp_liquidity_sweep': (2.0, 8.0),
            'atr_stop_trap_within_trend': (1.5, 5.0),
            'atr_tp_trap_within_trend': (2.0, 8.0),
            'atr_stop_spring': (0.8, 4.0),
            'atr_tp_spring': (2.0, 8.0),
            'atr_stop_failed_continuation': (0.8, 4.0),
            'atr_tp_failed_continuation': (2.0, 8.0),
        },
    }

    # Load results
    warnings = []
    for gnum, params in groups.items():
        name = {1: 'top_earners', 2: 'mid_tier', 3: 'new_archetypes',
                5: 'structural_gates', 6: 'atr_multipliers'}[gnum]
        results_path = PROJECT_ROOT / f'results/optuna_group_{gnum}_{name}/results.json'
        if not results_path.exists():
            continue

        with open(results_path) as f:
            results = json.load(f)

        best_params = results.get('best_params', {})
        for pkey, (lo, hi) in params.items():
            val = best_params.get(pkey)
            if val is None:
                continue
            margin = (hi - lo) * 0.05  # 5% margin
            at_low = val <= lo + margin
            at_high = val >= hi - margin
            if at_low or at_high:
                boundary = 'LOW' if at_low else 'HIGH'
                warnings.append(f"  {pkey} = {val:.4f} at {boundary} boundary [{lo}, {hi}]")

    return warnings


# ─────────────────────────────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────────────────────────────

def print_sep(char='=', width=90):
    print(char * width)

def print_header(text):
    print_sep()
    print(f"  {text}")
    print_sep()

def print_comparison(label, old_val, new_val, fmt='.3f', better='higher'):
    """Print old vs new with improvement indicator."""
    if isinstance(old_val, str) or isinstance(new_val, str):
        print(f"  {label:<25s} {str(old_val):>12s} {str(new_val):>12s}")
        return

    delta = new_val - old_val
    if better == 'higher':
        improved = delta > 0
    elif better == 'lower':
        improved = delta < 0
    else:
        improved = abs(delta) < abs(old_val) * 0.05  # within 5%

    pct = (delta / abs(old_val) * 100) if old_val != 0 else 0
    indicator = '+' if improved else '-' if not improved else '~'
    arrow = f"({indicator}{abs(pct):.1f}%)"

    print(f"  {label:<25s} {old_val:>12{fmt}} {new_val:>12{fmt}}  {arrow}")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Post-optimization validation suite")
    parser.add_argument('--mc-sims', type=int, default=10000, help='Monte Carlo simulations')
    parser.add_argument('--html', action='store_true', help='Generate quantstats HTML report')
    parser.add_argument('--skip-full', action='store_true', help='Skip full-period backtest')
    args = parser.parse_args()

    print_header("POST-OPTIMIZATION VALIDATION SUITE")
    print(f"  Config: {CONFIG_PATH.name}")
    print(f"  Feature store: {FEATURE_STORE.name}")
    print(f"  Monte Carlo sims: {args.mc_sims:,}")
    print()

    # Load config and features
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    features_df = pd.read_parquet(FEATURE_STORE)
    print(f"  Loaded {len(features_df):,} bars")
    print()

    # ── 1. Full-period backtest (2020-2024) ──────────────────────────
    if not args.skip_full:
        print_header("1. FULL-PERIOD BACKTEST (2020-01-01 to 2024-12-31)")
        t0 = time.time()
        full_stats, full_trades, full_equity = run_backtest(
            config, features_df, '2020-01-01', '2024-12-31'
        )
        elapsed = time.time() - t0
        print(f"  Completed in {elapsed:.0f}s")
        print()

        bl = BASELINE['full']
        print(f"  {'Metric':<25s} {'Baseline':>12s} {'Optimized':>12s}  {'Change'}")
        print(f"  {'-'*25} {'-'*12} {'-'*12}  {'-'*12}")
        print_comparison('Profit Factor', bl['profit_factor'], full_stats.get('profit_factor', 0))
        print_comparison('Total PnL ($)', bl['total_pnl'], full_stats.get('total_pnl', 0), fmt=',.0f')
        print_comparison('Total Trades', bl['total_trades'], full_stats.get('total_trades', 0), fmt='.0f', better='neutral')
        print_comparison('Win Rate (%)', bl['win_rate'], full_stats.get('win_rate', 0))
        print_comparison('Max Drawdown (%)', bl['max_drawdown'], full_stats.get('max_drawdown', 0), better='lower')
        print_comparison('Sharpe Ratio', bl['sharpe_ratio'], full_stats.get('sharpe_ratio', 0))
        print()
    else:
        full_stats, full_trades, full_equity = None, None, None

    # ── 2. OOS backtest (2023-2024) ──────────────────────────────────
    print_header("2. OUT-OF-SAMPLE BACKTEST (2023-01-01 to 2024-12-31)")
    t0 = time.time()
    oos_stats, oos_trades, oos_equity = run_backtest(
        config, features_df, '2023-01-01', '2024-12-31'
    )
    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.0f}s")
    print()

    print(f"  Profit Factor:  {oos_stats.get('profit_factor', 0):.3f}")
    print(f"  Total Trades:   {oos_stats.get('total_trades', 0)}")
    print(f"  Win Rate:       {oos_stats.get('win_rate', 0):.1f}%")
    print(f"  Max Drawdown:   {oos_stats.get('max_drawdown', 0):.1f}%")
    print(f"  Sharpe Ratio:   {oos_stats.get('sharpe_ratio', 0):.3f}")
    print(f"  Total PnL:      ${oos_stats.get('total_pnl', 0):,.0f}")
    print()

    # ── 3. Deflated Sharpe Ratio ─────────────────────────────────────
    print_header("3. DEFLATED SHARPE RATIO (DSR)")

    # Collect all trial Sharpe ratios from the 5 groups
    all_sharpes = []
    for gnum in [1, 2, 3, 5, 6]:
        name = {1: 'top_earners', 2: 'mid_tier', 3: 'new_archetypes',
                5: 'structural_gates', 6: 'atr_multipliers'}[gnum]
        results_path = PROJECT_ROOT / f'results/optuna_group_{gnum}_{name}/results.json'
        if results_path.exists():
            with open(results_path) as f:
                r = json.load(f)
            # Use train Sharpe as proxy (we don't have all trial Sharpes, just the best)
            train_sharpe = r.get('train_metrics', {}).get('sharpe_ratio', 0)
            oos_sharpe = r.get('oos_metrics', {}).get('sharpe_ratio', 0)
            all_sharpes.extend([train_sharpe, oos_sharpe])

    # Total trials across all groups
    total_trials = 50 + 50 + 50 + 15 + 15  # Groups 1-3 (50 each) + 5-6 (15 each)

    # Use the full-period or OOS Sharpe as the observed
    if full_stats:
        observed_sharpe = full_stats.get('sharpe_ratio', 0)
        n_obs = full_stats.get('total_trades', 100)
    else:
        observed_sharpe = oos_stats.get('sharpe_ratio', 0)
        n_obs = oos_stats.get('total_trades', 100)

    # Compute return statistics for DSR
    use_trades = full_trades if full_trades else oos_trades
    if use_trades:
        pnls = [t['pnl'] for t in use_trades]
        pnl_arr = np.array(pnls)
        ret_skew = float(pd.Series(pnls).skew()) if len(pnls) > 3 else 0.0
        ret_kurt = float(pd.Series(pnls).kurtosis() + 3) if len(pnls) > 3 else 3.0  # excess -> regular
    else:
        ret_skew, ret_kurt = 0.0, 3.0

    # Generate synthetic trial Sharpes (assume distribution around observed)
    # Since we don't have individual trial Sharpes, approximate with N trials
    rng = np.random.RandomState(42)
    synthetic_sharpes = rng.normal(observed_sharpe * 0.7, observed_sharpe * 0.3, total_trials)
    synthetic_sharpes = np.clip(synthetic_sharpes, -0.5, observed_sharpe * 1.5)

    try:
        dsr = compute_dsr(observed_sharpe, synthetic_sharpes, n_obs, ret_skew, ret_kurt)
        print(f"  Observed Sharpe:     {observed_sharpe:.3f}")
        print(f"  Total trials tested: {total_trials}")
        print(f"  Return skewness:     {ret_skew:.3f}")
        print(f"  Return kurtosis:     {ret_kurt:.3f}")
        print(f"  DSR probability:     {dsr:.4f}")
        print()
        if dsr > 0.95:
            print(f"  VERDICT: STRONG — Sharpe is statistically significant after {total_trials} trials")
        elif dsr > 0.80:
            print(f"  VERDICT: GOOD — Sharpe likely real, minor selection bias concern")
        elif dsr > 0.50:
            print(f"  VERDICT: MARGINAL — Cannot reject null hypothesis of random Sharpe")
        else:
            print(f"  VERDICT: WEAK — Sharpe likely due to selection bias from {total_trials} trials")
    except Exception as e:
        print(f"  DSR computation failed: {e}")
        dsr = 0.0
    print()

    # ── 4. Monte Carlo Trade Bootstrapping ───────────────────────────
    print_header("4. MONTE CARLO TRADE BOOTSTRAPPING")

    if use_trades:
        pnls = [t['pnl'] for t in use_trades]
        print(f"  Trades: {len(pnls)}")
        print(f"  Simulations: {args.mc_sims:,}")
        print(f"  Skip rate: 10%")
        print()

        mc = monte_carlo_bootstrap(pnls, n_simulations=args.mc_sims)

        if 'error' not in mc:
            for metric in ['profit_factor', 'max_drawdown_pct', 'sharpe', 'total_return_pct']:
                if metric not in mc:
                    continue
                data = mc[metric]
                pcts = data['percentiles']
                label = {
                    'profit_factor': 'Profit Factor',
                    'max_drawdown_pct': 'Max Drawdown (%)',
                    'sharpe': 'Sharpe Ratio',
                    'total_return_pct': 'Total Return (%)',
                }[metric]

                print(f"  {label}:")
                print(f"    Mean:  {data['mean']:>10.2f}  |  Std:   {data['std']:>10.2f}")
                print(f"    1st%:  {pcts[1]:>10.2f}  |  5th%:  {pcts[5]:>10.2f}  |  "
                      f"10th%: {pcts[10]:>10.2f}")
                print(f"    25th%: {pcts[25]:>10.2f}  |  50th%: {pcts[50]:>10.2f}  |  "
                      f"75th%: {pcts[75]:>10.2f}")
                print(f"    90th%: {pcts[90]:>10.2f}  |  95th%: {pcts[95]:>10.2f}  |  "
                      f"99th%: {pcts[99]:>10.2f}")
                print()

            # Risk assessment
            worst_dd = mc['max_drawdown_pct']['percentiles'][1]
            worst_pf = mc['profit_factor']['percentiles'][5]
            print(f"  RISK ASSESSMENT:")
            print(f"    Worst-case DD (1st percentile): {worst_dd:.1f}%")
            print(f"    Worst-case PF (5th percentile): {worst_pf:.2f}")
            if worst_pf > 1.0:
                print(f"    95% of scenarios remain profitable (PF > 1.0)")
            else:
                pct_profitable = sum(1 for pf in mc['profit_factor']['percentiles'].values()
                                    if pf > 1.0) / len(mc['profit_factor']['percentiles']) * 100
                print(f"    ~{pct_profitable:.0f}% of percentile scenarios profitable")
        else:
            print(f"  {mc['error']}")
    else:
        print("  No trades available for Monte Carlo")
    print()

    # ── 5. Parameter Boundary Check ──────────────────────────────────
    print_header("5. PARAMETER BOUNDARY CHECK")
    warnings = check_param_boundaries()
    if warnings:
        print(f"  WARNING: {len(warnings)} params at search boundaries:")
        for w in warnings:
            print(w)
        print()
        print("  Consider expanding search ranges for these parameters.")
    else:
        print("  All optimized params are within search boundaries.")
    print()

    # ── 6. Per-Archetype Breakdown ───────────────────────────────────
    if use_trades:
        print_header("6. PER-ARCHETYPE BREAKDOWN")
        trade_df = pd.DataFrame(use_trades)
        if 'archetype' in trade_df.columns and 'pnl' in trade_df.columns:
            arch_stats = trade_df.groupby('archetype').agg(
                trades=('pnl', 'count'),
                total_pnl=('pnl', 'sum'),
                avg_pnl=('pnl', 'mean'),
                win_rate=('pnl', lambda x: (x > 0).mean() * 100),
                pf=('pnl', lambda x: x[x > 0].sum() / abs(x[x < 0].sum()) if (x < 0).any() else float('inf')),
            ).sort_values('total_pnl', ascending=False)

            print(f"  {'Archetype':<25s} {'Trades':>7s} {'PnL':>10s} {'Avg':>8s} {'WR%':>6s} {'PF':>6s}")
            print(f"  {'-'*25} {'-'*7} {'-'*10} {'-'*8} {'-'*6} {'-'*6}")
            for arch, row in arch_stats.iterrows():
                pf_str = f"{row['pf']:.2f}" if row['pf'] < 100 else "inf"
                print(f"  {arch:<25s} {row['trades']:>7.0f} ${row['total_pnl']:>9,.0f} "
                      f"${row['avg_pnl']:>7,.0f} {row['win_rate']:>5.1f}% {pf_str:>6s}")
            print()

    # ── 7. HTML Report (optional) ────────────────────────────────────
    if args.html and full_equity:
        print_header("7. QUANTSTATS HTML REPORT")
        try:
            import quantstats as qs
            # Build returns series from equity curve
            eq_series = pd.Series([e[1] for e in full_equity],
                                  index=pd.to_datetime([e[0] for e in full_equity]))
            returns = eq_series.pct_change().dropna()
            output_path = PROJECT_ROOT / 'results' / 'optimization_validation_report.html'
            qs.reports.html(returns, output=str(output_path),
                           title='Bull Machine Post-Optimization Validation')
            print(f"  Report saved to: {output_path}")
        except ImportError:
            print("  quantstats not installed. Run: pip install quantstats")
        except Exception as e:
            print(f"  Report generation failed: {e}")
        print()

    # ── Final Verdict ────────────────────────────────────────────────
    print_header("FINAL VERDICT")

    checks = []

    # Check 1: Full-period PF improvement
    if full_stats:
        pf_improved = full_stats.get('profit_factor', 0) > BASELINE['full']['profit_factor']
        checks.append(('PF improved vs baseline', pf_improved))

    # Check 2: OOS PF > 1.2
    oos_pf = oos_stats.get('profit_factor', 0)
    checks.append(('OOS PF > 1.2', oos_pf > 1.2))

    # Check 3: DSR > 0.80
    checks.append(('DSR > 0.80', dsr > 0.80))

    # Check 4: Monte Carlo 5th percentile PF > 1.0
    if use_trades and 'error' not in mc:
        mc_5th_pf = mc['profit_factor']['percentiles'][5]
        checks.append(('MC 5th% PF > 1.0', mc_5th_pf > 1.0))

    # Check 5: No params at boundaries
    checks.append(('No params at boundaries', len(warnings) == 0))

    passed = sum(1 for _, v in checks if v)
    total = len(checks)

    for label, ok in checks:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {label}")

    print()
    if passed == total:
        print(f"  RESULT: ALL {total} CHECKS PASSED — Ready for deployment")
    elif passed >= total - 1:
        print(f"  RESULT: {passed}/{total} CHECKS PASSED — Minor concerns, likely safe")
    else:
        print(f"  RESULT: {passed}/{total} CHECKS PASSED — Review failed checks before deploying")

    print()
    print_sep()


if __name__ == '__main__':
    main()
