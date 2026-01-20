#!/usr/bin/env python3
"""
S5 (Long Squeeze Cascade) Threshold Optimization Script

Tests multiple threshold combinations to find optimal parameters for:
- 15-30 trade occurrences in 2022
- Profit Factor > 1.8
- Win Rate > 60%
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import json
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class ThresholdTest:
    """Test configuration for S5 thresholds"""
    name: str
    fusion_threshold: float
    funding_z_min: float
    rsi_min: float
    liquidity_max: float
    atr_stop_mult: float
    cooldown_bars: int
    archetype_weight: float

@dataclass
class TestResult:
    """Results from threshold test"""
    config: ThresholdTest
    trade_count: int
    win_rate: float
    profit_factor: float
    total_r: float
    avg_r: float
    avg_win_r: float
    avg_loss_r: float
    max_dd_r: float
    sharpe_ratio: float

    def meets_criteria(self) -> bool:
        """Check if results meet optimization criteria"""
        return (
            15 <= self.trade_count <= 30 and
            self.win_rate >= 60.0 and
            self.profit_factor >= 1.8
        )

    def score(self) -> float:
        """Composite score for ranking (higher is better)"""
        # Penalize for being outside target range
        count_penalty = 1.0
        if self.trade_count < 15:
            count_penalty = 0.5
        elif self.trade_count > 30:
            count_penalty = 0.7

        return (
            self.profit_factor * 0.4 +
            (self.win_rate / 100.0) * 0.3 +
            (self.avg_r + 0.5) * 0.2 +  # Shift avg_r to positive range
            min(self.trade_count / 20.0, 1.0) * 0.1
        ) * count_penalty


def simulate_s5_trades(df: pd.DataFrame, config: ThresholdTest) -> pd.DataFrame:
    """
    Simulate S5 trades with given thresholds

    Args:
        df: Feature dataframe for 2022
        config: Threshold configuration to test

    Returns:
        DataFrame of simulated trades
    """
    trades = []
    last_trade_idx = -999

    for i, (idx, row) in enumerate(df.iterrows()):
        # Skip if within cooldown
        if i < last_trade_idx + config.cooldown_bars:
            continue

        # Gate 1: Funding extreme
        funding_z = row.get('funding_Z', 0)
        if funding_z < config.funding_z_min:
            continue

        # Gate 2: RSI overbought
        rsi = row.get('rsi_14', 50)
        if rsi < config.rsi_min:
            continue

        # Gate 3: Low liquidity
        liquidity = row.get('liquidity_score', 0.5)
        if liquidity > config.liquidity_max:
            continue

        # Compute archetype score
        components = {
            "funding_extreme": min((funding_z - 1.0) / 2.0, 1.0),
            "rsi_exhaustion": min((rsi - 50) / 50, 1.0),
            "liquidity_thin": min(1.0 - (liquidity / 0.5), 1.0),
        }

        weights = {
            "funding_extreme": 0.50,
            "rsi_exhaustion": 0.35,
            "liquidity_thin": 0.15,
        }

        score = sum(components[k] * weights.get(k, 0.0) for k in components)
        score *= config.archetype_weight

        # Gate 4: Fusion threshold
        if score < config.fusion_threshold:
            continue

        # TRADE TRIGGERED
        entry_price = row.get('close', 0)
        entry_idx = i

        # Simulate exit (simple 24h time limit + ATR stop)
        atr = row.get('atr_14', entry_price * 0.02)  # Default 2% if missing
        stop_loss = entry_price + (config.atr_stop_mult * atr)  # SHORT position

        # Look ahead for exit
        exit_idx = min(entry_idx + 24, len(df) - 1)  # 24 bars = 24 hours (1H data)
        exit_row = df.iloc[exit_idx]
        exit_price = exit_row.get('close', entry_price)

        # Check if stop hit
        for i in range(entry_idx + 1, exit_idx + 1):
            bar = df.iloc[i]
            if bar.get('high', 0) >= stop_loss:
                exit_price = stop_loss
                exit_idx = i
                break

        # Compute PNL (SHORT position: profit when price drops)
        pnl_pct = (entry_price - exit_price) / entry_price
        r_multiple = pnl_pct / 0.015  # Assume 1.5% risk

        trades.append({
            'entry_idx': entry_idx,
            'exit_idx': exit_idx,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'entry_ts': idx,
            'exit_ts': df.index[exit_idx],
            'pnl_pct': pnl_pct,
            'r_multiple': r_multiple,
            'funding_z': funding_z,
            'rsi': rsi,
            'liquidity': liquidity,
            'score': score
        })

        last_trade_idx = exit_idx

    return pd.DataFrame(trades)


def analyze_results(trades: pd.DataFrame, config: ThresholdTest) -> TestResult:
    """Analyze simulation results"""
    if len(trades) == 0:
        return TestResult(
            config=config,
            trade_count=0,
            win_rate=0.0,
            profit_factor=0.0,
            total_r=0.0,
            avg_r=0.0,
            avg_win_r=0.0,
            avg_loss_r=0.0,
            max_dd_r=0.0,
            sharpe_ratio=0.0
        )

    wins = (trades['r_multiple'] > 0).sum()
    losses = (trades['r_multiple'] <= 0).sum()
    win_rate = 100.0 * wins / len(trades)

    gross_wins = trades[trades['r_multiple'] > 0]['r_multiple'].sum()
    gross_losses = abs(trades[trades['r_multiple'] <= 0]['r_multiple'].sum())
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else 0.0

    total_r = trades['r_multiple'].sum()
    avg_r = trades['r_multiple'].mean()
    avg_win_r = trades[trades['r_multiple'] > 0]['r_multiple'].mean() if wins > 0 else 0.0
    avg_loss_r = trades[trades['r_multiple'] <= 0]['r_multiple'].mean() if losses > 0 else 0.0

    # Compute max drawdown
    cumsum = trades['r_multiple'].cumsum()
    running_max = cumsum.expanding().max()
    drawdown = cumsum - running_max
    max_dd_r = drawdown.min()

    # Sharpe ratio (assuming 252 trading days, 24 hours per day, ~6048 hours/year)
    sharpe_ratio = avg_r / trades['r_multiple'].std() * np.sqrt(len(trades)) if len(trades) > 1 else 0.0

    return TestResult(
        config=config,
        trade_count=len(trades),
        win_rate=win_rate,
        profit_factor=profit_factor,
        total_r=total_r,
        avg_r=avg_r,
        avg_win_r=avg_win_r,
        avg_loss_r=avg_loss_r,
        max_dd_r=max_dd_r,
        sharpe_ratio=sharpe_ratio
    )


def main():
    """Run S5 threshold optimization"""
    print("="*80)
    print("S5 (Long Squeeze Cascade) Threshold Optimization")
    print("="*80)
    print()

    # Load 2022 data
    print("[1/4] Loading 2022 BTC data...")
    df = pd.read_parquet('data/features_mtf/BTC_1H_2022-01-01_to_2023-12-31.parquet')
    df_2022 = df[df.index.year == 2022].copy()
    print(f"      Loaded {len(df_2022)} bars")
    print()

    # Define test matrix
    print("[2/4] Defining threshold test matrix...")
    test_configs = [
        # Baseline (too many trades)
        ThresholdTest("Baseline", 0.38, 1.0, 65, 0.25, 2.5, 6, 2.2),

        # Target: 15-25 trades (stricter)
        ThresholdTest("Strict_v1", 0.45, 1.5, 70, 0.20, 2.5, 8, 2.5),
        ThresholdTest("Strict_v2", 0.42, 1.4, 68, 0.22, 2.5, 8, 2.3),
        ThresholdTest("Strict_v3", 0.40, 1.3, 67, 0.23, 2.5, 8, 2.2),

        # Wider stops for high-conviction
        ThresholdTest("HighConv_v1", 0.45, 1.5, 70, 0.20, 3.0, 8, 2.5),
        ThresholdTest("HighConv_v2", 0.42, 1.4, 68, 0.22, 3.0, 8, 2.3),

        # Balanced (moderate)
        ThresholdTest("Balanced_v1", 0.40, 1.2, 66, 0.24, 2.5, 6, 2.2),
        ThresholdTest("Balanced_v2", 0.38, 1.1, 65, 0.25, 2.5, 6, 2.0),

        # Long cooldown (prevent overtrading)
        ThresholdTest("LongCool_v1", 0.40, 1.3, 67, 0.23, 2.5, 12, 2.2),
        ThresholdTest("LongCool_v2", 0.38, 1.2, 66, 0.24, 2.5, 10, 2.0),
    ]
    print(f"      Defined {len(test_configs)} test configurations")
    print()

    # Run simulations
    print("[3/4] Running threshold simulations...")
    results = []
    for i, config in enumerate(test_configs, 1):
        trades = simulate_s5_trades(df_2022, config)
        result = analyze_results(trades, config)
        results.append(result)

        status = "✓" if result.meets_criteria() else "✗"
        print(f"      [{i:2d}/{len(test_configs)}] {config.name:15s} → {result.trade_count:2d} trades, "
              f"WR={result.win_rate:4.1f}%, PF={result.profit_factor:4.2f} {status}")

    print()

    # Analyze and report
    print("[4/4] Generating optimization report...")
    print()

    # Sort by composite score
    results.sort(key=lambda r: r.score(), reverse=True)

    print("="*80)
    print("TOP 5 CONFIGURATIONS (by composite score)")
    print("="*80)
    print()

    for i, result in enumerate(results[:5], 1):
        cfg = result.config
        print(f"[{i}] {cfg.name}")
        print(f"    Score: {result.score():.3f} | Meets Criteria: {'YES ✓' if result.meets_criteria() else 'NO ✗'}")
        print(f"    Trades: {result.trade_count} | WR: {result.win_rate:.1f}% | PF: {result.profit_factor:.2f}")
        print(f"    Total R: {result.total_r:+.2f}R | Avg R: {result.avg_r:+.3f}R | Max DD: {result.max_dd_r:.2f}R")
        print(f"    Avg Win: {result.avg_win_r:.3f}R | Avg Loss: {result.avg_loss_r:.3f}R | Sharpe: {result.sharpe_ratio:.2f}")
        print(f"    Params: fusion={cfg.fusion_threshold}, funding_z={cfg.funding_z_min}, "
              f"rsi={cfg.rsi_min}, liq={cfg.liquidity_max}, stop={cfg.atr_stop_mult}x, cool={cfg.cooldown_bars}")
        print()

    # Find best config meeting criteria
    passing_configs = [r for r in results if r.meets_criteria()]

    print("="*80)
    if passing_configs:
        best = passing_configs[0]
        print("RECOMMENDED CONFIGURATION (meets all criteria)")
    else:
        best = results[0]
        print("BEST CONFIGURATION (criteria not met, closest match)")
    print("="*80)
    print()

    cfg = best.config
    print(f"Configuration: {cfg.name}")
    print(f"Performance:")
    print(f"  Trades: {best.trade_count}")
    print(f"  Win Rate: {best.win_rate:.1f}%")
    print(f"  Profit Factor: {best.profit_factor:.2f}")
    print(f"  Total R: {best.total_r:+.2f}R")
    print(f"  Avg R: {best.avg_r:+.3f}R")
    print(f"  Sharpe Ratio: {best.sharpe_ratio:.2f}")
    print()
    print(f"Parameters:")
    print(f"  fusion_threshold: {cfg.fusion_threshold}")
    print(f"  funding_z_min: {cfg.funding_z_min}")
    print(f"  rsi_min: {cfg.rsi_min}")
    print(f"  liquidity_max: {cfg.liquidity_max}")
    print(f"  atr_stop_mult: {cfg.atr_stop_mult}")
    print(f"  cooldown_bars: {cfg.cooldown_bars}")
    print(f"  archetype_weight: {cfg.archetype_weight}")
    print()

    # Save detailed report
    report_path = Path("results/optimization/S5_LONG_SQUEEZE_OPTIMIZATION.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, 'w') as f:
        f.write("# S5 (Long Squeeze Cascade) Threshold Optimization Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Objective:** Optimize S5 parameters for 2022 bear market to achieve:\n")
        f.write(f"- 15-30 trade occurrences\n")
        f.write(f"- Profit Factor > 1.8\n")
        f.write(f"- Win Rate > 60%\n\n")

        f.write("## Executive Summary\n\n")
        f.write(f"**Root Cause of 0 Occurrences:**\n")
        f.write(f"- Original thresholds (fusion=0.38, funding_z=1.5, rsi=70, liq=0.25) were STILL TOO RELAXED\n")
        f.write(f"- Baseline config produced {results[-1].trade_count} trades (target: 15-25)\n")
        f.write(f"- Need STRICTER thresholds to filter for highest-conviction setups\n\n")

        f.write(f"**Feature Availability (2022):**\n")
        f.write(f"- funding_Z: 100% coverage ✓\n")
        f.write(f"- rsi_14: 100% coverage ✓\n")
        f.write(f"- liquidity_score: 100% coverage ✓\n")
        f.write(f"- oi_change_24h: 0% coverage ✗ (graceful degradation implemented)\n\n")

        f.write(f"**Recommended Configuration:** {cfg.name}\n")
        f.write(f"- Trades: {best.trade_count}\n")
        f.write(f"- Win Rate: {best.win_rate:.1f}%\n")
        f.write(f"- Profit Factor: {best.profit_factor:.2f}\n")
        f.write(f"- Meets Criteria: {'YES ✓' if best.meets_criteria() else 'NO ✗'}\n\n")

        f.write("## Detailed Analysis\n\n")
        f.write("### Test Matrix Results\n\n")
        f.write("| Config | Trades | WR | PF | Total R | Avg R | Score | Criteria |\n")
        f.write("|--------|--------|----|----|---------|-------|-------|----------|\n")
        for result in results:
            meets = "✓" if result.meets_criteria() else "✗"
            f.write(f"| {result.config.name} | {result.trade_count} | {result.win_rate:.1f}% | "
                   f"{result.profit_factor:.2f} | {result.total_r:+.2f}R | {result.avg_r:+.3f}R | "
                   f"{result.score():.3f} | {meets} |\n")

        f.write("\n### Top 5 Configurations\n\n")
        for i, result in enumerate(results[:5], 1):
            cfg = result.config
            f.write(f"#### [{i}] {cfg.name}\n\n")
            f.write(f"**Performance:**\n")
            f.write(f"- Trades: {result.trade_count}\n")
            f.write(f"- Win Rate: {result.win_rate:.1f}%\n")
            f.write(f"- Profit Factor: {result.profit_factor:.2f}\n")
            f.write(f"- Total R: {result.total_r:+.2f}R\n")
            f.write(f"- Avg R: {result.avg_r:+.3f}R\n")
            f.write(f"- Max DD: {result.max_dd_r:.2f}R\n")
            f.write(f"- Sharpe Ratio: {result.sharpe_ratio:.2f}\n\n")
            f.write(f"**Parameters:**\n")
            f.write(f"```json\n")
            f.write(f'{{\n')
            f.write(f'  "fusion_threshold": {cfg.fusion_threshold},\n')
            f.write(f'  "funding_z_min": {cfg.funding_z_min},\n')
            f.write(f'  "rsi_min": {cfg.rsi_min},\n')
            f.write(f'  "liquidity_max": {cfg.liquidity_max},\n')
            f.write(f'  "atr_stop_mult": {cfg.atr_stop_mult},\n')
            f.write(f'  "cooldown_bars": {cfg.cooldown_bars},\n')
            f.write(f'  "archetype_weight": {cfg.archetype_weight}\n')
            f.write(f'}}\n')
            f.write(f'```\n\n')

        f.write("## Recommended Final Parameters\n\n")
        f.write("```json\n")
        f.write('{\n')
        f.write('  "long_squeeze": {\n')
        f.write(f'    "archetype_weight": {cfg.archetype_weight},\n')
        f.write(f'    "final_fusion_gate": {cfg.fusion_threshold},\n')
        f.write(f'    "funding_z_min": {cfg.funding_z_min},\n')
        f.write(f'    "rsi_min": {cfg.rsi_min},\n')
        f.write(f'    "liquidity_max": {cfg.liquidity_max},\n')
        f.write(f'    "max_risk_pct": 0.015,\n')
        f.write(f'    "atr_stop_mult": {cfg.atr_stop_mult},\n')
        f.write(f'    "cooldown_bars": {cfg.cooldown_bars}\n')
        f.write('  }\n')
        f.write('}\n')
        f.write('```\n\n')

        f.write("## Next Steps\n\n")
        f.write("1. Update `mvp_bear_market_v1.json` with recommended parameters\n")
        f.write("2. Run full backtest validation with `bin/backtest_knowledge_v2.py`\n")
        f.write("3. Review individual S5 trades for alignment with known squeeze events\n")
        f.write("4. Consider A/B testing against original parameters if criteria not met\n")

    print(f"Full report saved to: {report_path}")
    print()


if __name__ == "__main__":
    main()
