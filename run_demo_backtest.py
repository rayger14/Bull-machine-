#!/usr/bin/env python3
"""
Demo Backtest: BTC/ETH Performance Analysis
Shows what the full v1.3 backtest results would look like
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from bull_machine.backtest.datafeed import DataFeed
from bull_machine.backtest.broker import PaperBroker
from bull_machine.backtest.portfolio import Portfolio
from bull_machine.backtest.engine import BacktestEngine
from bull_machine.backtest.strategy_adapter_simple import generate_simple_signal as strategy_fn


def create_demo_config():
    return {
        "run_id": "demo_btc_eth_results",
        "data": {
            "sources": {
                "BTCUSD_1H": "/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_BTCUSD, 60_50ad4.csv",
                "ETHUSD_1H": "/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_ETHUSD, 60_2f4ab.csv",
            },
            "timeframes": ["1D", "4H", "1H"],
        },
        "broker": {"fee_bps": 10, "slippage_bps": 5, "spread_bps": 2},
        "portfolio": {"starting_cash": 100000, "exposure_cap_pct": 0.5},
        "engine": {"lookback_bars": 50, "seed": 42},  # Smaller for demo
    }


def run_demo_backtest():
    print("🚀 BULL MACHINE v1.4 - DEMO BACKTEST RESULTS")
    print("=" * 60)
    print("Demonstrating BTC/ETH Multi-Timeframe Strategy Performance\n")

    config = create_demo_config()

    # Setup components
    try:
        feed = DataFeed(config["data"]["sources"])
        broker = PaperBroker(**config["broker"])
        portfolio = Portfolio(
            config["portfolio"]["starting_cash"], config["portfolio"]["exposure_cap_pct"]
        )
        engine = BacktestEngine(config, feed, broker, portfolio)

        print("✅ Data loaded successfully:")
        for symbol in feed.frames:
            df = feed.frames[symbol]
            print(f"   📊 {symbol}: {len(df)} bars ({df.index[0]} to {df.index[-1]})")

        # Run backtest with demo strategy
        def demo_strategy(symbol: str, tf: str, window_df):
            return strategy_fn(symbol, tf, window_df, balance=config["portfolio"]["starting_cash"])

        symbols = list(config["data"]["sources"].keys())
        timeframes = config["data"]["timeframes"]

        print(f"\n🔄 Running backtest...")
        print(f"   Symbols: {symbols}")
        print(f"   Timeframes: {timeframes}")
        print(f"   Lookback: {config['engine']['lookback_bars']} bars")

        results = engine.run(demo_strategy, symbols, timeframes, out_dir="demo_results")

        # Analyze results
        print(f"\n📈 BACKTEST RESULTS")
        print("=" * 40)

        metrics = results["metrics"]
        print(f"💰 Total Trades: {metrics.get('trades', 0)}")
        print(f"📊 Win Rate: {metrics.get('win_rate', 0.0) * 100:.1f}%")
        print(f"💵 Average Win: ${metrics.get('avg_win', 0.0):,.2f}")
        print(f"💸 Average Loss: ${metrics.get('avg_loss', 0.0):,.2f}")
        print(f"🎯 Expectancy: ${metrics.get('expectancy', 0.0):,.2f}")
        print(f"📉 Max Drawdown: {metrics.get('max_dd', 0.0) * 100:.1f}%")
        print(f"📈 CAGR: {metrics.get('cagr', 0.0) * 100:.1f}%")
        print(f"⚡ Sharpe Ratio: {metrics.get('sharpe', 0.0):.2f}")

        # Portfolio performance
        final_equity = portfolio.equity()
        total_return = (
            (final_equity - config["portfolio"]["starting_cash"])
            / config["portfolio"]["starting_cash"]
            * 100
        )

        print(f"\n💼 PORTFOLIO PERFORMANCE")
        print("=" * 40)
        print(f"🏦 Starting Capital: ${config['portfolio']['starting_cash']:,}")
        print(f"💰 Final Equity: ${final_equity:,.2f}")
        print(f"📈 Total Return: {total_return:+.2f}%")
        print(f"💸 Realized PnL: ${portfolio.realized:,.2f}")
        print(f"📊 Unrealized PnL: ${portfolio.unrealized:,.2f}")

        # Risk analysis
        print(f"\n⚠️  RISK ANALYSIS")
        print("=" * 40)
        print(f"🛡️  Exposure Cap: {config['portfolio']['exposure_cap_pct'] * 100:.0f}%")
        print(f"📊 Active Positions: {len(portfolio.positions)}")
        print(f"💹 Max Drawdown: ${portfolio.drawdown():,.2f}")
        print(f"🏔️  High Water Mark: ${portfolio.high_water:,.2f}")

        # Strategy insights
        print(f"\n🧠 STRATEGY INSIGHTS")
        print("=" * 40)
        print(f"📡 Multi-Timeframe Analysis: 1D bias → 4H structure → 1H execution")
        print(f"🎯 Risk Management: 1% per trade, 3R TP ladder")
        print(f"🛡️  Stop Losses: ATR-based with breakeven @ TP1")
        print(f"⚖️  Position Sizing: Dynamic based on volatility")

        # Artifacts
        print(f"\n📁 GENERATED REPORTS")
        print("=" * 40)
        for name, path in results["artifacts"].items():
            print(f"   📄 {name.capitalize()}: {path}")

        return results

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = run_demo_backtest()

    if results:
        print(f"\n🎉 Demo completed successfully!")
        print(f"📊 This demonstrates the type of analysis you'll get from the full v1.3 pipeline")
        print(f"⚡ Full backtests include advanced Wyckoff, liquidity, and confluence analysis")
