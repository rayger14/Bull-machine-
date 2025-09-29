#!/usr/bin/env python3
"""
Bull Machine v1.6.2 - Risk Parameter Scaling Analysis
Test different risk levels to achieve institutional target returns (8-15% annual)
"""

from run_complete_confluence_system import load_multi_timeframe_data, run_complete_confluence_backtest
import json

def test_risk_scaling():
    print('🎯 Bull Machine v1.6.2 - Risk Scaling Analysis')
    print('=' * 55)
    print('Target: 8-15% annual returns for institutional deployment')
    print()

    # Load ETH data
    data_paths = {
        '1D': '/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_ETHUSD, 1D_64942.csv',
        '4H': '/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_ETHUSD, 240_1d04a.csv',
        '1H': '/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_ETHUSD, 60_2f4ab.csv'
    }

    data = load_multi_timeframe_data('ETH', data_paths)

    # Test different risk levels on 2024 data
    risk_levels = [
        {'risk_pct': 0.025, 'name': 'Conservative (2.5%)', 'target': 'Current baseline'},
        {'risk_pct': 0.05, 'name': 'Moderate (5.0%)', 'target': '~4-6% annual'},
        {'risk_pct': 0.075, 'name': 'Aggressive (7.5%)', 'target': '~6-9% annual'},
        {'risk_pct': 0.10, 'name': 'High Risk (10%)', 'target': '~8-12% annual'},
        {'risk_pct': 0.125, 'name': 'Maximum (12.5%)', 'target': '~10-15% annual'}
    ]

    results = []

    for risk_config in risk_levels:
        print(f"📊 Testing {risk_config['name']} - {risk_config['target']}")
        print("-" * 50)

        # Configuration for 2024 test
        config = {
            'entry_threshold': 0.3,
            'min_active_domains': 3,
            'cooldown_days': 7,
            'risk_pct': risk_config['risk_pct'],
            'sl_atr_multiplier': 1.4,
            'tp_atr_multiplier': 2.5,
            'trail_atr_multiplier': 0.8,
            'start_date': '2024-01-01',
            'end_date': '2024-12-31'
        }

        result = run_complete_confluence_backtest('ETH', data, config)

        if result and 'trades' in result and 'metrics' in result:
            trades = result['trades']
            metrics = result['metrics']

            # Calculate performance with $10k starting balance
            starting_balance = 10000
            balance = starting_balance

            trade_pnls = []
            for trade in trades:
                trade_risk = balance * risk_config['risk_pct']
                dollar_pnl = trade_risk * (trade['pnl'] / 100)
                balance += dollar_pnl
                trade_pnls.append(dollar_pnl)

            total_return = (balance / starting_balance - 1) * 100
            max_single_loss = min(trade_pnls) if trade_pnls else 0
            max_single_gain = max(trade_pnls) if trade_pnls else 0

            result_summary = {
                'risk_level': risk_config['name'],
                'risk_pct': risk_config['risk_pct'],
                'trades': len(trades),
                'total_return_pct': total_return,
                'final_balance': balance,
                'profit_loss_dollar': balance - starting_balance,
                'max_single_loss': max_single_loss,
                'max_single_gain': max_single_gain,
                'win_rate': metrics.get('win_rate', 0),
                'profit_factor': metrics.get('profit_factor', 0),
                'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                'max_dd_pct': metrics.get('max_drawdown_pct', 0)
            }

            results.append(result_summary)

            print(f"  Trades: {len(trades)}")
            print(f"  Total Return: {total_return:+.2f}%")
            print(f"  Final Balance: ${balance:,.2f}")
            print(f"  Profit/Loss: ${balance - starting_balance:+,.2f}")
            print(f"  Largest Loss: ${max_single_loss:+,.2f}")
            print(f"  Largest Gain: ${max_single_gain:+,.2f}")
            print(f"  Win Rate: {metrics.get('win_rate', 0):.1f}%")
            print(f"  Profit Factor: {metrics.get('profit_factor', 0):.2f}")

            # Risk assessment
            max_account_loss = abs(max_single_loss / starting_balance * 100)
            print(f"  Max Account Loss: {max_account_loss:.2f}%")

            # Target assessment
            if 8 <= total_return <= 15:
                status = "✅ INSTITUTIONAL TARGET"
            elif total_return >= 15:
                status = "⚠️ HIGH RETURN (Risk Check)"
            else:
                status = "❌ BELOW TARGET"

            print(f"  Assessment: {status}")
            print()

        else:
            print("  ❌ No valid backtest results")
            print()

    # Summary comparison
    print("🎯 RISK SCALING SUMMARY")
    print("=" * 60)
    print(f"{'Risk Level':<20} {'Return':<10} {'Max Loss':<12} {'Assessment':<20}")
    print("-" * 60)

    for result in results:
        risk_name = result['risk_level'].split('(')[0].strip()
        return_pct = f"{result['total_return_pct']:+.1f}%"
        max_loss = f"{abs(result['max_single_loss']/10000*100):.1f}%"

        if 8 <= result['total_return_pct'] <= 15:
            assessment = "✅ TARGET ACHIEVED"
        elif result['total_return_pct'] >= 15:
            assessment = "⚠️ HIGH RETURN"
        else:
            assessment = "❌ BELOW TARGET"

        print(f"{risk_name:<20} {return_pct:<10} {max_loss:<12} {assessment:<20}")

    print()
    print("💡 INSTITUTIONAL RECOMMENDATIONS")
    print("-" * 35)

    # Find optimal risk level
    target_results = [r for r in results if 8 <= r['total_return_pct'] <= 15]

    if target_results:
        # Prefer the most conservative option that hits target
        optimal = min(target_results, key=lambda x: x['risk_pct'])
        print(f"🎯 Optimal Risk Level: {optimal['risk_level']}")
        print(f"   Expected Return: {optimal['total_return_pct']:+.1f}% annual")
        print(f"   Max Single Loss: {abs(optimal['max_single_loss']/10000*100):.1f}% of account")
        print(f"   Profit Factor: {optimal['profit_factor']:.2f}")

        # Scaling projections
        print(f"\n📈 SCALING PROJECTIONS:")
        for capital in [100000, 1000000, 10000000]:  # $100k, $1M, $10M
            scaled_profit = capital * (optimal['total_return_pct'] / 100)
            scaled_max_loss = capital * (abs(optimal['max_single_loss']) / 10000)
            print(f"   ${capital:,} → ${scaled_profit:+,.0f} profit (Max loss: ${scaled_max_loss:,.0f})")

    else:
        print("⚠️ No risk level achieved 8-15% target range")
        print("   Consider strategy optimization or longer test periods")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"reports/opt/risk_scaling_analysis_{timestamp}.json"

    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'analysis_period': '2024-01-01 to 2024-12-31',
            'results': results,
            'recommendations': target_results
        }, f, indent=2, default=str)

    print(f"\n📁 Detailed analysis saved: {output_file}")

if __name__ == "__main__":
    from datetime import datetime
    test_risk_scaling()