#!/usr/bin/env python3
"""
SPY Adaptive Max-Hold Analysis

Tests the best SPY config (rank #9) with adaptive max-hold logic:
- Base max_hold: 24 hours (current setting)
- Extended to 48-72 hours when:
  1. M1/M2 expansion phase (liquidity environment favorable)
  2. VIX < 20 (low fear)
  3. Trade is profitable (already in the money)
  4. Macro alignment strong

Compares:
- Original fixed max_hold
- Adaptive max_hold (market-aware)
- No max_hold (let all winners run)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import json
from dataclasses import dataclass, asdict
from typing import List, Dict
from bin.backtest_knowledge_v2 import KnowledgeParams, KnowledgeAwareBacktest


@dataclass
class AdaptiveParams:
    """Parameters for adaptive max-hold logic"""
    base_max_hold: int = 24  # Base max hold in hours
    extended_max_hold_bullish: int = 48  # Extended hold in strong bull
    extended_max_hold_very_bullish: int = 72  # Extended hold in very strong bull

    # Thresholds for extension
    m1m2_threshold: float = 0.7  # M1/M2 expansion score
    vix_threshold: float = 20.0  # VIX level
    macro_score_threshold: float = 0.6  # Macro alignment
    min_profit_pct: float = 0.5  # Minimum profit % to extend


def get_adaptive_max_hold(row: pd.Series, current_pnl_pct: float, adaptive_params: AdaptiveParams) -> int:
    """
    Calculate adaptive max-hold based on market conditions.

    Args:
        row: Current bar with MTF features
        current_pnl_pct: Current trade profit %
        adaptive_params: Adaptive parameters

    Returns:
        Max hold bars for this trade
    """
    # Start with base
    max_hold = adaptive_params.base_max_hold

    # Must be profitable to extend
    if current_pnl_pct < adaptive_params.min_profit_pct:
        return max_hold

    # Check market conditions
    m1m2_score = row.get('tf1d_m1m2_liquidity_score', 0.5)
    macro_score = row.get('mtf_macro_score', 0.5)

    # VIX proxy: Use ATR percentile (SPY doesn't have VIX in features)
    # Low ATR = low volatility = favorable for holding
    atr_percentile = row.get('atr_percentile', 0.5)  # Lower = less volatile
    low_volatility = atr_percentile < 0.3  # Bottom 30% of volatility

    # Count favorable conditions
    favorable_count = 0

    if m1m2_score >= adaptive_params.m1m2_threshold:
        favorable_count += 1

    if macro_score >= adaptive_params.macro_score_threshold:
        favorable_count += 1

    if low_volatility:
        favorable_count += 1

    # Extend based on favorable conditions
    if favorable_count >= 3:
        # Very bullish: extend to 72h
        max_hold = adaptive_params.extended_max_hold_very_bullish
    elif favorable_count >= 2:
        # Bullish: extend to 48h
        max_hold = adaptive_params.extended_max_hold_bullish

    return max_hold


class AdaptiveMaxHoldBacktest:
    """Backtest with adaptive max-hold logic"""

    def __init__(self, df: pd.DataFrame, params: KnowledgeParams, adaptive_params: AdaptiveParams, starting_capital: float = 10000.0):
        self.df = df.copy()
        self.params = params
        self.adaptive_params = adaptive_params
        self.starting_capital = starting_capital

    def run(self) -> Dict:
        """Run backtest with adaptive max-hold"""
        from bin.backtest_knowledge_v2 import Trade

        equity = self.starting_capital
        position = 0
        entry_price = 0.0
        entry_time = None
        entry_idx = 0
        position_size = 0.0
        trades = []
        peak = equity
        max_dd = 0.0

        # Compute fusion scores (copy from KnowledgeAwareBacktest)
        fusion_scores = []
        for idx, row in self.df.iterrows():
            # Simplified fusion score calculation
            wyckoff = row.get('tf1d_wyckoff_score', 0.5)
            liquidity = row.get('tf1d_boms_strength', 0.5)
            momentum = row.get('rsi_14', 50.0) / 100.0
            macro = row.get('mtf_macro_score', 0.5)

            score = (
                self.params.wyckoff_weight * wyckoff +
                self.params.liquidity_weight * liquidity +
                self.params.momentum_weight * momentum +
                self.params.macro_weight * macro
            )
            fusion_scores.append(min(1.0, max(0.0, score)))

        self.df['fusion_score'] = fusion_scores

        for i, (idx, row) in enumerate(self.df.iterrows()):
            current_price = row['close']
            fusion = row['fusion_score']

            # Entry logic
            if position == 0:
                # Check tier thresholds
                if fusion >= self.params.tier3_threshold:
                    # Enter long
                    position = 1
                    entry_price = current_price
                    entry_time = idx
                    entry_idx = i
                    position_size = equity * 0.95

            # Exit logic
            elif position != 0:
                bars_held = i - entry_idx
                current_pnl_pct = ((current_price - entry_price) / entry_price) * 100

                # Get adaptive max-hold
                adaptive_max_hold_bars = get_adaptive_max_hold(row, current_pnl_pct, self.adaptive_params)

                # Exit conditions
                exit_reason = None

                # 1. Stop loss
                atr = row.get('atr_14', entry_price * 0.01)
                stop_price = entry_price - (atr * self.params.atr_stop_mult)
                if current_price <= stop_price:
                    exit_reason = 'stop_loss'

                # 2. Signal neutralized
                elif fusion < self.params.tier3_threshold:
                    exit_reason = 'signal_neutralized'

                # 3. Adaptive max hold
                elif bars_held >= adaptive_max_hold_bars:
                    exit_reason = f'max_hold_{adaptive_max_hold_bars}h'

                # 4. End of data
                elif i == len(self.df) - 1:
                    exit_reason = 'end_of_period'

                if exit_reason:
                    # Close trade
                    pnl_pct = (current_price - entry_price) / entry_price * position
                    pnl = position_size * pnl_pct
                    equity += pnl

                    trade = Trade(
                        entry_time=entry_time,
                        entry_price=entry_price,
                        position_size=position_size,
                        direction=position,
                        exit_time=idx,
                        exit_price=current_price,
                        exit_reason=exit_reason
                    )
                    trades.append(trade)

                    # Update drawdown
                    if equity > peak:
                        peak = equity
                    dd = (peak - equity) / peak
                    if dd > max_dd:
                        max_dd = dd

                    # Reset
                    position = 0

        # Calculate metrics
        if not trades:
            return {
                'total_pnl': 0.0,
                'total_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'final_equity': equity,
                'trades': []
            }

        total_pnl = sum((t.exit_price - t.entry_price) / t.entry_price * t.direction * t.position_size for t in trades)
        winners = [t for t in trades if (t.exit_price - t.entry_price) * t.direction > 0]
        losers = [t for t in trades if (t.exit_price - t.entry_price) * t.direction <= 0]

        gross_profit = sum((t.exit_price - t.entry_price) / t.entry_price * t.direction * t.position_size for t in winners)
        gross_loss = abs(sum((t.exit_price - t.entry_price) / t.entry_price * t.direction * t.position_size for t in losers))

        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
        win_rate = len(winners) / len(trades) if trades else 0.0

        return {
            'total_pnl': total_pnl,
            'total_trades': len(trades),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'sharpe_ratio': 0.0,  # Simplified
            'max_drawdown': max_dd,
            'final_equity': equity,
            'trades': trades
        }


def main():
    print("="*80)
    print("SPY ADAPTIVE MAX-HOLD ANALYSIS")
    print("="*80)

    # Load SPY 2024 data
    df = pd.read_parquet('data/features_mtf/SPY_1H_2024-01-01_to_2025-10-17.parquet')
    df = df[(df.index >= '2024-01-01') & (df.index <= '2024-12-31')].copy()

    print(f"Loaded {len(df)} bars of SPY 2024 data\n")

    # Load best config (rank #9)
    with open('reports/optuna_results/SPY_knowledge_v3_strict_best_configs.json') as f:
        results = json.load(f)

    best_config = results['top_10_configs'][8]  # Rank #9 (0-indexed)
    params_dict = best_config['params']

    # Build params
    params = KnowledgeParams(
        wyckoff_weight=params_dict['wyckoff_weight'],
        liquidity_weight=params_dict['liquidity_weight'],
        momentum_weight=params_dict['momentum_weight'],
        macro_weight=params_dict['macro_weight'],
        pti_weight=params_dict['pti_weight'],
        tier1_threshold=params_dict['tier1_threshold'],
        tier2_threshold=params_dict['tier2_threshold'],
        tier3_threshold=params_dict['tier3_threshold'],
        require_m1m2_confirmation=params_dict['require_m1m2_confirmation'],
        require_macro_alignment=params_dict['require_macro_alignment'],
        atr_stop_mult=params_dict['atr_stop_mult'],
        trailing_atr_mult=params_dict['trailing_atr_mult'],
        max_hold_bars=params_dict['max_hold_bars'],
        max_risk_pct=params_dict['max_risk_pct'],
        volatility_scaling=params_dict['volatility_scaling'],
        use_smart_exits=True,
        breakeven_after_tp1=True
    )

    print(f"Original Config (Rank #9):")
    print(f"  Fixed max_hold: {params.max_hold_bars} hours")
    print(f"  Expected PNL: ${best_config['metrics']['total_pnl']:,.2f}")
    print(f"  Expected Trades: {best_config['metrics']['total_trades']}")
    print()

    # Test scenarios
    scenarios = [
        {
            'name': 'Baseline (Fixed 24h)',
            'adaptive': AdaptiveParams(base_max_hold=24, extended_max_hold_bullish=24, extended_max_hold_very_bullish=24)
        },
        {
            'name': 'Adaptive (24h → 48h)',
            'adaptive': AdaptiveParams(base_max_hold=24, extended_max_hold_bullish=48, extended_max_hold_very_bullish=48)
        },
        {
            'name': 'Adaptive (24h → 72h)',
            'adaptive': AdaptiveParams(base_max_hold=24, extended_max_hold_bullish=48, extended_max_hold_very_bullish=72)
        },
        {
            'name': 'Aggressive (24h → 96h)',
            'adaptive': AdaptiveParams(base_max_hold=24, extended_max_hold_bullish=72, extended_max_hold_very_bullish=96)
        }
    ]

    results_summary = []

    for scenario in scenarios:
        print(f"Testing: {scenario['name']}...")

        backtest = AdaptiveMaxHoldBacktest(df, params, scenario['adaptive'])
        result = backtest.run()

        results_summary.append({
            'scenario': scenario['name'],
            'total_pnl': result['total_pnl'],
            'total_trades': result['total_trades'],
            'win_rate': result['win_rate'],
            'profit_factor': result['profit_factor'],
            'max_drawdown': result['max_drawdown'],
            'final_equity': result['final_equity']
        })

        print(f"  PNL: ${result['total_pnl']:,.2f}, Trades: {result['total_trades']}, "
              f"WinRate: {result['win_rate']*100:.1f}%, PF: {result['profit_factor']:.2f}, "
              f"MaxDD: {result['max_drawdown']*100:.2f}%")
        print()

    # Display comparison table
    print("="*80)
    print("ADAPTIVE MAX-HOLD COMPARISON")
    print("="*80)

    header = f"{'Scenario':<25} {'PNL':>12} {'Trades':>8} {'WinRate':>8} {'PF':>8} {'MaxDD':>8} {'Equity':>12}"
    print(header)
    print("-"*80)

    baseline_pnl = results_summary[0]['total_pnl']

    for r in results_summary:
        delta_pnl = r['total_pnl'] - baseline_pnl
        delta_symbol = "+" if delta_pnl > 0 else ""

        line = f"{r['scenario']:<25} ${r['total_pnl']:>10,.2f} {r['total_trades']:>8} "
        line += f"{r['win_rate']*100:>7.1f}% {r['profit_factor']:>8.2f} "
        line += f"{r['max_drawdown']*100:>7.2f}% ${r['final_equity']:>10,.2f}"

        if delta_pnl != 0:
            line += f"  ({delta_symbol}${delta_pnl:,.2f})"

        print(line)

    print("="*80)

    # Find best scenario
    best_scenario = max(results_summary, key=lambda x: x['total_pnl'])
    improvement = best_scenario['total_pnl'] - baseline_pnl
    improvement_pct = (improvement / baseline_pnl * 100) if baseline_pnl > 0 else 0

    print(f"\nBEST SCENARIO: {best_scenario['scenario']}")
    print(f"  Total PNL: ${best_scenario['total_pnl']:,.2f}")
    print(f"  Improvement over baseline: +${improvement:,.2f} (+{improvement_pct:.1f}%)")
    print(f"  Win Rate: {best_scenario['win_rate']*100:.1f}%")
    print(f"  Profit Factor: {best_scenario['profit_factor']:.2f}")
    print(f"  Max Drawdown: {best_scenario['max_drawdown']*100:.2f}%")

    print("\n" + "="*80)
    print("CONCLUSION:")
    print("="*80)

    if improvement > 0:
        print(f"Adaptive max-hold logic IMPROVES performance by ${improvement:,.2f} (+{improvement_pct:.1f}%)")
        print(f"\nRecommendation: Use '{best_scenario['scenario']}' for SPY trading")
    else:
        print("Fixed max-hold performs better than adaptive logic for SPY.")
        print("\nRecommendation: Keep fixed 24h max-hold for SPY (mean-reverting asset)")

    print("="*80)


if __name__ == '__main__':
    main()
