#!/usr/bin/env python3
"""
Bull Machine v1.4.2 - ETH Comprehensive Backtest
Direct integration with production v1.3 engine
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import sys
import os

# Add Bull Machine to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bull_machine.version import __version__, get_version_banner

# ETH data files
ETH_DATA = {
    "1D": [
        ("/Users/raymondghandchi/Desktop/Chart Logs/COINBASE_ETHUSD, 1D_fa116.csv", "Desktop_1D"),
        ("/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_ETHUSD, 1D_64942.csv", "Downloads_1D")
    ],
    "4H": [
        ("/Users/raymondghandchi/Desktop/Chart Logs/COINBASE_ETHUSD, 240_ab8a9.csv", "Desktop_4H"),
        ("/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_ETHUSD, 240_1d04a.csv", "Downloads_4H")
    ],
    "6H": [
        ("/Users/raymondghandchi/Desktop/Chart Logs/COINBASE_ETHUSD, 360_3c856.csv", "Desktop_6H")
    ],
    "12H": [
        ("/Users/raymondghandchi/Desktop/Chart Logs/COINBASE_ETHUSD, 720_ffc2d.csv", "Desktop_12H")
    ],
    "1H": [
        ("/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_ETHUSD, 60_2f4ab.csv", "Downloads_1H")
    ]
}

def prepare_eth_csv(input_path: str, output_path: str) -> bool:
    """Prepare ETH CSV for Bull Machine format."""
    try:
        df = pd.read_csv(input_path)

        # Required columns
        required = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required):
            print(f"❌ Missing columns in {input_path}")
            return False

        # Create clean dataframe
        clean_df = pd.DataFrame()

        # Add timestamp
        if 'time' in df.columns:
            clean_df['timestamp'] = pd.to_datetime(df['time'], unit='s')
        else:
            clean_df['timestamp'] = pd.date_range(start='2020-06-01', periods=len(df), freq='D')

        # Add OHLC
        clean_df['open'] = df['open']
        clean_df['high'] = df['high']
        clean_df['low'] = df['low']
        clean_df['close'] = df['close']

        # Add volume
        if 'volume' in df.columns:
            clean_df['volume'] = df['volume']
        elif 'BUY+SELL V' in df.columns:
            clean_df['volume'] = df['BUY+SELL V']
        else:
            clean_df['volume'] = 100000

        # Save prepared CSV
        clean_df.to_csv(output_path, index=False)
        return True

    except Exception as e:
        print(f"❌ Error preparing {input_path}: {e}")
        return False

def run_single_backtest(csv_path: str, timeframe: str, label: str) -> dict:
    """Run backtest using v1.3 engine."""
    import subprocess

    result = {
        'timeframe': timeframe,
        'label': label,
        'file': csv_path,
        'success': False
    }

    try:
        # Run v1.3 with MTF enabled
        cmd = [
            'python3', '-m', 'bull_machine.app.main_v13',
            '--csv', csv_path,
            '--balance', '10000',
            '--mtf-enabled'
        ]

        print(f"\n🔄 Running: {' '.join(cmd)}")
        output = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        if output.returncode == 0:
            # Parse output for results
            lines = output.stdout.split('\n')
            for line in lines:
                if 'Total PnL:' in line:
                    result['total_pnl'] = float(line.split(':')[1].strip().replace('%', ''))
                elif 'Final Balance:' in line:
                    result['final_balance'] = float(line.split('$')[1].replace(',', ''))
                elif 'Total Trades:' in line:
                    result['total_trades'] = int(line.split(':')[1].strip())
                elif 'Win Rate:' in line:
                    result['win_rate'] = float(line.split(':')[1].strip().replace('%', ''))

            result['success'] = True
            result['output'] = output.stdout

        else:
            result['error'] = output.stderr
            print(f"❌ Error: {output.stderr[:200]}")

    except subprocess.TimeoutExpired:
        result['error'] = "Timeout"
        print(f"⏱️ Timeout running backtest")
    except Exception as e:
        result['error'] = str(e)
        print(f"❌ Exception: {e}")

    return result

def analyze_trades(output_text: str) -> list:
    """Extract trade details from output."""
    trades = []
    lines = output_text.split('\n')

    for i, line in enumerate(lines):
        if 'TRADE PLAN GENERATED' in line:
            trade = {}
            # Look at next few lines for details
            for j in range(i+1, min(i+15, len(lines))):
                detail = lines[j]
                if 'Direction:' in detail:
                    trade['direction'] = detail.split(':')[1].strip().split()[0]
                elif 'Entry:' in detail and 'Exit:' not in detail:
                    trade['entry'] = float(detail.split(':')[1].strip())
                elif 'Stop:' in detail:
                    trade['stop'] = float(detail.split(':')[1].strip())
                elif 'Expected R:' in detail:
                    trade['expected_r'] = float(detail.split(':')[1].strip())
                elif 'Size:' in detail:
                    trade['size'] = float(detail.split(':')[1].strip())

            if trade:
                trades.append(trade)

    return trades

def main():
    """Run comprehensive ETH backtests."""
    print("=" * 80)
    print(f"🚀 {get_version_banner()} - ETH Comprehensive Backtest")
    print("=" * 80)
    print(f"📅 Run Time: {datetime.now()}")
    print("=" * 80)

    all_results = {}
    summary_stats = {
        'total_files': 0,
        'successful': 0,
        'failed': 0,
        'best_return': None,
        'worst_return': None,
        'all_trades': []
    }

    # Create temp directory for prepared CSVs
    temp_dir = Path("temp_eth_data")
    temp_dir.mkdir(exist_ok=True)

    # Process each timeframe
    for timeframe, file_list in ETH_DATA.items():
        print(f"\n{'='*60}")
        print(f"📊 Processing {timeframe} Timeframe")
        print(f"{'='*60}")

        for file_path, label in file_list:
            if not Path(file_path).exists():
                print(f"⚠️  File not found: {file_path}")
                summary_stats['failed'] += 1
                continue

            summary_stats['total_files'] += 1

            # Prepare CSV
            temp_csv = temp_dir / f"eth_{timeframe}_{label}.csv"
            print(f"\n📂 Preparing: {Path(file_path).name}")

            if prepare_eth_csv(file_path, str(temp_csv)):
                print(f"✅ Prepared data saved to: {temp_csv.name}")

                # Check data
                df = pd.read_csv(temp_csv)
                print(f"📊 Data: {len(df)} bars")
                print(f"📅 Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
                print(f"💰 Price Range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")

                # Run backtest
                print(f"\n🔧 Running Bull Machine v1.4.2 Backtest...")
                result = run_single_backtest(str(temp_csv), timeframe, label)

                if result['success']:
                    summary_stats['successful'] += 1

                    # Display results
                    print(f"\n✅ Backtest Complete!")
                    print(f"📈 Performance Summary:")

                    if 'total_pnl' in result:
                        print(f"  Total PnL: {result['total_pnl']:+.2f}%")
                    if 'final_balance' in result:
                        print(f"  Final Balance: ${result['final_balance']:,.2f}")
                        return_pct = ((result['final_balance'] - 10000) / 10000) * 100
                        print(f"  Return: {return_pct:+.2f}%")

                        # Track best/worst
                        if summary_stats['best_return'] is None or return_pct > summary_stats['best_return'][1]:
                            summary_stats['best_return'] = (f"{timeframe}_{label}", return_pct)
                        if summary_stats['worst_return'] is None or return_pct < summary_stats['worst_return'][1]:
                            summary_stats['worst_return'] = (f"{timeframe}_{label}", return_pct)

                    if 'total_trades' in result:
                        print(f"  Total Trades: {result['total_trades']}")
                    if 'win_rate' in result:
                        print(f"  Win Rate: {result['win_rate']:.1f}%")

                    # Extract trades
                    if 'output' in result:
                        trades = analyze_trades(result['output'])
                        if trades:
                            result['trades'] = trades
                            summary_stats['all_trades'].extend(trades)
                            print(f"  Trades Found: {len(trades)}")

                            # Show sample trades
                            print(f"\n  Sample Trades:")
                            for trade in trades[:3]:
                                print(f"    {trade.get('direction', 'N/A').upper()} | "
                                      f"Entry: {trade.get('entry', 0):.2f} | "
                                      f"Stop: {trade.get('stop', 0):.2f}")

                else:
                    summary_stats['failed'] += 1
                    print(f"❌ Backtest failed: {result.get('error', 'Unknown error')}")

                # Store result
                key = f"{timeframe}_{label}"
                all_results[key] = result

            else:
                summary_stats['failed'] += 1

    # Overall Summary
    print(f"\n{'='*80}")
    print(f"📊 COMPREHENSIVE ETH BACKTEST SUMMARY")
    print(f"{'='*80}")

    print(f"\n🎯 Overall Statistics:")
    print(f"  Files Processed: {summary_stats['total_files']}")
    print(f"  Successful: {summary_stats['successful']}")
    print(f"  Failed: {summary_stats['failed']}")
    print(f"  Total Trades Generated: {len(summary_stats['all_trades'])}")

    if summary_stats['best_return']:
        print(f"\n🏆 Best Performance: {summary_stats['best_return'][0]}")
        print(f"  Return: {summary_stats['best_return'][1]:+.2f}%")

    if summary_stats['worst_return']:
        print(f"\n📉 Worst Performance: {summary_stats['worst_return'][0]}")
        print(f"  Return: {summary_stats['worst_return'][1]:+.2f}%")

    # Timeframe comparison
    print(f"\n📊 Results by Timeframe:")
    for key, result in all_results.items():
        if result['success'] and 'final_balance' in result:
            return_pct = ((result['final_balance'] - 10000) / 10000) * 100
            trades = result.get('total_trades', 0)
            win_rate = result.get('win_rate', 0)
            print(f"  {key}: {return_pct:+.2f}% | Trades: {trades} | Win Rate: {win_rate:.1f}%")

    # Save detailed results
    output_file = f"eth_backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump({
            'summary': summary_stats,
            'results': all_results,
            'version': __version__,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2, default=str)

    print(f"\n💾 Detailed results saved to: {output_file}")

    # Cleanup temp files
    import shutil
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
        print(f"🧹 Cleaned up temporary files")

    print(f"\n{'='*80}")
    print(f"✅ ETH Comprehensive Backtest Complete!")
    print(f"   Bull Machine {__version__}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()