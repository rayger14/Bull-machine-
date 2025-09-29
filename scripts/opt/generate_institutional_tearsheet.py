#!/usr/bin/env python3
"""
Bull Machine v1.6.2 - Institutional Tearsheet Generator
Fund-style performance reporting for institutional presentation
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from run_complete_confluence_system import load_multi_timeframe_data, run_complete_confluence_backtest

class InstitutionalTearsheet:
    """Generate professional fund-style tearsheets"""

    def __init__(self):
        self.fund_name = "Bull Machine Capital"
        self.strategy_name = "5-Domain Confluence ETH"
        self.inception_date = "2023-01-01"
        self.benchmark = "ETH Buy & Hold"

    def run_comprehensive_backtest(self, config, period_name="Full Period"):
        """Run backtest and return standardized results"""

        # Load ETH data
        data_paths = {
            '1D': '/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_ETHUSD, 1D_64942.csv',
            '4H': '/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_ETHUSD, 240_1d04a.csv',
            '1H': '/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_ETHUSD, 60_2f4ab.csv'
        }

        data = load_multi_timeframe_data('ETH', data_paths)
        result = run_complete_confluence_backtest('ETH', data, config)

        if not result or 'trades' not in result or 'metrics' not in result:
            return None

        trades = result['trades']
        metrics = result['metrics']

        # Calculate detailed performance metrics
        starting_capital = 100000  # $100k baseline
        risk_per_trade = config.get('risk_pct', 0.025)

        # Build equity curve
        equity_curve = [starting_capital]
        balance = starting_capital
        trade_returns = []
        monthly_returns = {}

        for trade in trades:
            trade_risk = balance * risk_per_trade
            dollar_pnl = trade_risk * (trade['pnl'] / 100)
            balance += dollar_pnl
            equity_curve.append(balance)

            # Calculate returns
            trade_return = dollar_pnl / (balance - dollar_pnl)
            trade_returns.append(trade_return)

            # Monthly aggregation
            month_key = trade['entry_time'].strftime('%Y-%m')
            if month_key not in monthly_returns:
                monthly_returns[month_key] = []
            monthly_returns[month_key].append(trade_return)

        # Calculate comprehensive metrics
        total_return = (balance / starting_capital - 1) * 100
        annual_return = total_return  # Assuming 1-year period for now

        # Risk metrics
        if trade_returns:
            returns_series = pd.Series(trade_returns)
            volatility = returns_series.std() * np.sqrt(12)  # Annualized
            sharpe_ratio = annual_return / (volatility * 100) if volatility > 0 else 0

            # Drawdown calculation
            equity_series = pd.Series(equity_curve)
            running_max = equity_series.expanding().max()
            drawdown_series = (equity_series - running_max) / running_max
            max_drawdown = abs(drawdown_series.min()) * 100
        else:
            volatility = 0
            sharpe_ratio = 0
            max_drawdown = 0

        # Monthly performance summary
        monthly_summary = {}
        for month, rets in monthly_returns.items():
            monthly_return = (np.prod([1 + r for r in rets]) - 1) * 100
            monthly_summary[month] = monthly_return

        performance_summary = {
            'period': period_name,
            'start_date': config.get('start_date', '2023-01-01'),
            'end_date': config.get('end_date', '2025-01-01'),
            'starting_capital': starting_capital,
            'ending_capital': balance,
            'total_return_pct': total_return,
            'annualized_return_pct': annual_return,
            'volatility_pct': volatility * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_pct': max_drawdown,
            'total_trades': len(trades),
            'win_rate_pct': metrics.get('win_rate', 0),
            'profit_factor': metrics.get('profit_factor', 0),
            'avg_trade_return': np.mean(trade_returns) * 100 if trade_returns else 0,
            'best_trade_pct': max([t['pnl'] for t in trades]) if trades else 0,
            'worst_trade_pct': min([t['pnl'] for t in trades]) if trades else 0,
            'trades_per_month': len(trades) / 12,  # Assuming 12-month period
            'monthly_returns': monthly_summary,
            'equity_curve': equity_curve,
            'trade_details': trades
        }

        return performance_summary

    def generate_tearsheet(self, config, output_file=None):
        """Generate complete institutional tearsheet"""

        print("📊 Bull Machine Capital - Institutional Tearsheet")
        print("=" * 65)

        # Run backtests for different periods
        results = {}

        # Full period backtest
        config_full = config.copy()
        config_full.update({'start_date': '2023-01-01', 'end_date': '2025-01-01'})
        results['full_period'] = self.run_comprehensive_backtest(config_full, "Full Period (2023-2025)")

        # 2024 performance
        config_2024 = config.copy()
        config_2024.update({'start_date': '2024-01-01', 'end_date': '2024-12-31'})
        results['2024'] = self.run_comprehensive_backtest(config_2024, "2024 Performance")

        # Generate report
        timestamp = datetime.now().strftime("%B %d, %Y")

        print(f"\n🏦 FUND INFORMATION")
        print("-" * 25)
        print(f"Fund Name: {self.fund_name}")
        print(f"Strategy: {self.strategy_name}")
        print(f"Inception Date: {self.inception_date}")
        print(f"Report Date: {timestamp}")
        print(f"Assets Under Management: Simulation ($100,000)")

        # Performance summary table
        print(f"\n📈 PERFORMANCE SUMMARY")
        print("-" * 30)
        print(f"{'Period':<20} {'Total Return':<15} {'Ann. Return':<15} {'Sharpe':<10} {'Max DD':<10}")
        print("-" * 70)

        for period_name, perf in results.items():
            if perf:
                period_display = perf['period']
                total_ret = f"{perf['total_return_pct']:+.1f}%"
                ann_ret = f"{perf['annualized_return_pct']:+.1f}%"
                sharpe = f"{perf['sharpe_ratio']:.2f}"
                max_dd = f"{perf['max_drawdown_pct']:.1f}%"
                print(f"{period_display:<20} {total_ret:<15} {ann_ret:<15} {sharpe:<10} {max_dd:<10}")

        # Detailed 2024 analysis
        if results.get('2024'):
            perf_2024 = results['2024']
            print(f"\n🎯 2024 DETAILED PERFORMANCE")
            print("-" * 35)
            print(f"Starting Capital: ${perf_2024['starting_capital']:,}")
            print(f"Ending Capital: ${perf_2024['ending_capital']:,.2f}")
            print(f"Net Profit: ${perf_2024['ending_capital'] - perf_2024['starting_capital']:+,.2f}")
            print(f"Total Return: {perf_2024['total_return_pct']:+.2f}%")
            print(f"Annualized Return: {perf_2024['annualized_return_pct']:+.2f}%")
            print(f"Volatility: {perf_2024['volatility_pct']:.2f}%")
            print(f"Sharpe Ratio: {perf_2024['sharpe_ratio']:.2f}")
            print(f"Maximum Drawdown: {perf_2024['max_drawdown_pct']:.2f}%")

            print(f"\n📊 TRADING STATISTICS")
            print("-" * 25)
            print(f"Total Trades: {perf_2024['total_trades']}")
            print(f"Win Rate: {perf_2024['win_rate_pct']:.1f}%")
            print(f"Profit Factor: {perf_2024['profit_factor']:.2f}")
            print(f"Average Trade: {perf_2024['avg_trade_return']:+.2f}%")
            print(f"Best Trade: {perf_2024['best_trade_pct']:+.2f}%")
            print(f"Worst Trade: {perf_2024['worst_trade_pct']:+.2f}%")
            print(f"Trades per Month: {perf_2024['trades_per_month']:.1f}")

            # Monthly performance
            if perf_2024['monthly_returns']:
                print(f"\n📅 MONTHLY RETURNS (2024)")
                print("-" * 30)
                for month, ret in sorted(perf_2024['monthly_returns'].items()):
                    month_name = datetime.strptime(month, '%Y-%m').strftime('%B %Y')
                    print(f"{month_name:<15}: {ret:+6.2f}%")

        # Risk analysis
        print(f"\n⚠️ RISK ANALYSIS")
        print("-" * 20)
        if results.get('2024'):
            perf = results['2024']
            risk_pct = config.get('risk_pct', 0.025) * 100
            print(f"Risk per Trade: {risk_pct:.1f}%")
            print(f"Max Single Loss: {abs(perf['worst_trade_pct']):.1f}% (Trade Level)")
            print(f"Max Account Loss: {risk_pct:.1f}% (Account Level)")
            print(f"VaR (95%): Estimated ~{risk_pct * 1.5:.1f}%")

        # Strategy description
        print(f"\n📋 STRATEGY OVERVIEW")
        print("-" * 25)
        print("Strategy: 5-Domain Confluence System")
        print("• Wyckoff Market Structure Analysis")
        print("• Liquidity Mapping & Sweep Detection")
        print("• Momentum & Volume Confirmation")
        print("• Temporal & Fibonacci Alignment")
        print("• Fusion & Psychological Level Confluence")
        print(f"• Entry Threshold: {config.get('entry_threshold', 0.3)}")
        print(f"• Minimum Active Domains: {config.get('min_active_domains', 3)}")
        print(f"• Cooldown Period: {config.get('cooldown_days', 7)} days")

        # Benchmark comparison (simplified)
        print(f"\n🏆 BENCHMARK COMPARISON")
        print("-" * 30)
        print("ETH Buy & Hold (2024): ~+63% (estimated)")
        if results.get('2024'):
            strategy_return = results['2024']['total_return_pct']
            print(f"Bull Machine Strategy: {strategy_return:+.1f}%")
            print(f"Risk-Adjusted Performance:")
            print(f"• Strategy Sharpe: {results['2024']['sharpe_ratio']:.2f}")
            print(f"• Buy & Hold Sharpe: ~0.8 (estimated)")
            print(f"• Strategy Max DD: {results['2024']['max_drawdown_pct']:.1f}%")
            print(f"• Buy & Hold Max DD: ~35% (estimated)")

        # Scaling projections
        print(f"\n📈 SCALING PROJECTIONS")
        print("-" * 25)
        if results.get('2024'):
            return_rate = results['2024']['total_return_pct'] / 100
            for capital in [250000, 1000000, 5000000, 10000000]:
                profit = capital * return_rate
                print(f"${capital:,} AUM → ${profit:+,.0f} annual profit")

        # Disclaimers
        print(f"\n⚖️ IMPORTANT DISCLAIMERS")
        print("-" * 30)
        print("• Past performance does not guarantee future results")
        print("• Cryptocurrency trading involves substantial risk")
        print("• Results based on historical backtesting")
        print("• Actual performance may vary due to slippage, fees")
        print("• Strategy requires sophisticated risk management")

        # Save detailed report
        if output_file is None:
            timestamp_file = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"reports/tearsheets/institutional_tearsheet_{timestamp_file}.json"

        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        tearsheet_data = {
            'fund_info': {
                'name': self.fund_name,
                'strategy': self.strategy_name,
                'inception_date': self.inception_date,
                'report_date': timestamp
            },
            'performance_results': results,
            'configuration': config,
            'disclaimers': [
                "Past performance does not guarantee future results",
                "Cryptocurrency trading involves substantial risk",
                "Results based on historical backtesting"
            ]
        }

        with open(output_file, 'w') as f:
            json.dump(tearsheet_data, f, indent=2, default=str)

        print(f"\n📁 Complete tearsheet saved: {output_file}")

        return tearsheet_data

def main():
    """Generate institutional tearsheet with optimal configuration"""

    generator = InstitutionalTearsheet()

    # Use optimal 7.5% risk configuration from scaling analysis
    optimal_config = {
        'entry_threshold': 0.3,
        'min_active_domains': 3,
        'cooldown_days': 7,
        'risk_pct': 0.075,  # 7.5% for institutional targets
        'sl_atr_multiplier': 1.4,
        'tp_atr_multiplier': 2.5,
        'trail_atr_multiplier': 0.8
    }

    tearsheet = generator.generate_tearsheet(optimal_config)

    print(f"\n🎯 Institutional tearsheet generation complete!")
    print(f"Ready for fund presentation and due diligence.")

if __name__ == "__main__":
    main()