#!/usr/bin/env python3
"""
Analyze Full Year 2024 Trades for v3.0 Knowledge Engine

Shows detailed trade-by-trade breakdown with entry/exit reasons,
position sizing calculations, and risk management logic.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import json
from bin.backtest_knowledge_v2 import KnowledgeParams, KnowledgeAwareBacktest

def load_full_year_data(asset: str) -> pd.DataFrame:
    """Load full year 2024 feature store data."""
    feature_dir = Path('data/features_mtf')
    pattern = f"{asset}_1H_*.parquet"
    files = list(feature_dir.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No feature store found for {asset}")

    feature_path = sorted(files)[-1]
    print(f"Loading: {feature_path}")

    df = pd.read_parquet(feature_path)

    # Filter to FULL YEAR 2024
    start_ts = pd.Timestamp('2024-01-01', tz='UTC')
    end_ts = pd.Timestamp('2024-12-31', tz='UTC')
    df = df[(df.index >= start_ts) & (df.index <= end_ts)].copy()

    print(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")
    return df


def main():
    # Load BTC v3.0 best config
    config_path = Path('reports/optuna_results/BTC_knowledge_v3_best_configs.json')

    with open(config_path, 'r') as f:
        configs = json.load(f)

    best_config = configs['top_10_configs'][0]

    print("=" * 80)
    print("BTC v3.0 Full Year 2024 Trade Analysis")
    print("=" * 80)
    print(f"\nOptimizer Score: {best_config['score']:.3f}")
    print(f"Period: 2024-01-01 to 2024-12-31")
    print(f"\nBest Config Parameters:")
    for k, v in best_config['params'].items():
        print(f"  {k}: {v}")

    # Load full year data
    df = load_full_year_data('BTC')

    # Build params from config
    params = KnowledgeParams(
        wyckoff_weight=best_config['params']['wyckoff_weight'],
        liquidity_weight=best_config['params']['liquidity_weight'],
        momentum_weight=best_config['params']['momentum_weight'],
        macro_weight=best_config['params']['macro_weight'],
        pti_weight=best_config['params']['pti_weight'],
        tier1_threshold=best_config['params']['tier1_threshold'],
        tier2_threshold=best_config['params']['tier2_threshold'],
        tier3_threshold=best_config['params']['tier3_threshold'],
        require_m1m2_confirmation=best_config['params']['require_m1m2_confirmation'],
        require_macro_alignment=best_config['params']['require_macro_alignment'],
        atr_stop_mult=best_config['params']['atr_stop_mult'],
        trailing_atr_mult=best_config['params']['trailing_atr_mult'],
        max_hold_bars=best_config['params']['max_hold_bars'],
        max_risk_pct=best_config['params']['max_risk_pct'],
        volatility_scaling=best_config['params']['volatility_scaling'],
        use_smart_exits=True,
        breakeven_after_tp1=True
    )

    # Run backtest
    print("\n" + "=" * 80)
    print("Running Full Year Backtest...")
    print("=" * 80)

    backtest = KnowledgeAwareBacktest(df, params, starting_capital=10000.0)
    results = backtest.run()

    print("\n" + "=" * 80)
    print("Full Year 2024 Results Summary")
    print("=" * 80)
    print(f"Starting Capital: ${backtest.starting_capital:,.2f}")
    print(f"Ending Equity: ${results['final_equity']:,.2f}")
    print(f"Total PNL: ${results['total_pnl']:,.2f}")
    print(f"Total Trades: {results['total_trades']}")
    print(f"Win Rate: {results['win_rate']:.1%}")
    print(f"Profit Factor: {results['profit_factor']:.2f}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown']:.2%}")
    print(f"Gross Profit: ${results['gross_profit']:,.2f}")
    print(f"Gross Loss: ${results['gross_loss']:,.2f}")
    print(f"Avg Win: ${results['avg_win']:,.2f}")
    print(f"Avg Loss: ${results['avg_loss']:,.2f}")

    # Print detailed trade breakdown
    print("\n" + "=" * 80)
    print("DETAILED TRADE-BY-TRADE BREAKDOWN")
    print("=" * 80)

    trades = results['trades']

    for i, trade in enumerate(trades, 1):
        print(f"\n{'─' * 80}")
        print(f"TRADE #{i}")
        print(f"{'─' * 80}")

        print(f"\n📍 ENTRY:")
        print(f"  Timestamp: {trade['entry_time']}")
        print(f"  Price: ${trade['entry_price']:,.2f}")
        print(f"  Direction: {trade['direction']}")
        print(f"  Entry Tier: {trade.get('entry_tier', 'N/A')}")
        print(f"  Fusion Score: {trade.get('entry_fusion_score', 0.0):.3f}")

        print(f"\n💰 POSITION SIZING:")
        print(f"  Equity Before: ${trade.get('equity_before', 0.0):,.2f}")
        print(f"  Position Size: ${trade['position_size']:,.2f}")
        print(f"  Risk %: {trade.get('risk_pct', 0.0):.2%}")
        print(f"  ATR at Entry: ${trade.get('atr_at_entry', 0.0):.2f}")
        print(f"  Stop Distance: ${trade.get('stop_distance', 0.0):.2f}")

        print(f"\n🛑 EXIT:")
        print(f"  Timestamp: {trade['exit_time']}")
        print(f"  Price: ${trade['exit_price']:,.2f}")
        print(f"  Exit Reason: {trade['exit_reason']}")
        print(f"  Bars Held: {trade.get('bars_held', 0)}")

        print(f"\n📊 RESULT:")
        print(f"  PNL: ${trade['pnl']:,.2f}")
        print(f"  ROI: {(trade['pnl'] / trade['position_size'] * 100):.2f}%")
        print(f"  Equity After: ${trade.get('equity_after', 0.0):,.2f}")

        # Feature context at entry
        print(f"\n🔍 FEATURE CONTEXT AT ENTRY:")
        context = trade.get('entry_context', {})
        if context:
            print(f"  Wyckoff Score: {context.get('wyckoff', 0.0):.3f}")
            print(f"  Liquidity Score: {context.get('liquidity', 0.0):.3f}")
            print(f"  Momentum Score: {context.get('momentum', 0.0):.3f}")
            print(f"  Macro Score: {context.get('macro', 0.0):.3f}")
            print(f"  PTI Score: {context.get('pti', 0.0):.3f}")
            print(f"  FRVP Score: {context.get('frvp', 0.0):.3f}")
            print(f"  Macro Regime: {context.get('macro_regime', 'N/A')}")
            print(f"  VIX Level: {context.get('vix_level', 'N/A')}")

    # Export to CSV for Excel analysis
    output_path = Path('reports/optuna_results/BTC_v3_full_year_trades.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    trade_df = pd.DataFrame(trades)
    trade_df.to_csv(output_path, index=False)

    print("\n" + "=" * 80)
    print(f"✅ Trade details exported to: {output_path}")
    print("=" * 80)


if __name__ == '__main__':
    main()
