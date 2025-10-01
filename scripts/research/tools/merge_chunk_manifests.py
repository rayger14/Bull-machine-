#!/usr/bin/env python3
"""
Merge Bull Machine chunk manifests into comprehensive reports
Aggregates multiple chunk results into single JSON/CSV for analysis
"""

import argparse
import json
import os
import pandas as pd
from typing import Dict, List, Any
from datetime import datetime

def load_chunk_manifests(checkpoint_dir: str) -> List[Dict[str, Any]]:
    """Load all chunk manifest files from checkpoint directory."""
    manifests = []

    if not os.path.exists(checkpoint_dir):
        print(f"❌ Checkpoint directory not found: {checkpoint_dir}")
        return manifests

    json_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.json')]

    for filename in sorted(json_files):
        filepath = os.path.join(checkpoint_dir, filename)
        try:
            with open(filepath, 'r') as f:
                manifest = json.load(f)
                manifests.append(manifest)
        except Exception as e:
            print(f"⚠️  Error loading {filename}: {e}")

    return manifests

def aggregate_performance(manifests: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate performance metrics across all chunks."""
    if not manifests:
        return {}

    # Collect all trades
    all_trades = []
    total_balance = 10000.0
    chunk_returns = []

    for manifest in manifests:
        trades = manifest.get('trades', [])
        all_trades.extend(trades)

        chunk_return = manifest.get('performance', {}).get('total_return', 0)
        chunk_returns.append(chunk_return)

    # Calculate final balance from all trades
    running_balance = 10000.0
    drawdowns = []
    peak = 10000.0

    for trade in all_trades:
        trade_pnl = trade.get('pnl', 0)
        running_balance += (trade_pnl / 100 * running_balance)

        peak = max(peak, running_balance)
        dd = (peak - running_balance) / peak * 100
        drawdowns.append(dd)

    total_return = (running_balance - 10000) / 10000 * 100

    # Win rate and profit factor
    winning_trades = [t for t in all_trades if t.get('pnl', 0) > 0]
    losing_trades = [t for t in all_trades if t.get('pnl', 0) <= 0]

    win_rate = len(winning_trades) / len(all_trades) * 100 if all_trades else 0

    if winning_trades and losing_trades:
        avg_win = sum(t['pnl'] for t in winning_trades) / len(winning_trades)
        avg_loss = sum(t['pnl'] for t in losing_trades) / len(losing_trades)
        profit_factor = abs(avg_win * len(winning_trades) / (avg_loss * len(losing_trades)))
    else:
        avg_win = sum(t['pnl'] for t in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(t['pnl'] for t in losing_trades) / len(losing_trades) if losing_trades else 0
        profit_factor = float('inf') if winning_trades and not losing_trades else 0

    # Calculate Sharpe ratio (simplified)
    if chunk_returns:
        returns_std = pd.Series(chunk_returns).std()
        sharpe_ratio = (sum(chunk_returns) / len(chunk_returns)) / returns_std if returns_std > 0 else 0
    else:
        sharpe_ratio = 0

    # R-multiple analysis
    r_multiples = [t.get('r_multiple', 0) for t in all_trades if 'r_multiple' in t]
    avg_r = sum(r_multiples) / len(r_multiples) if r_multiples else 0
    r_gt_1 = len([r for r in r_multiples if r > 1])
    r_gt_2 = len([r for r in r_multiples if r > 2])

    return {
        'total_trades': len(all_trades),
        'final_balance': running_balance,
        'total_return': total_return,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'max_drawdown': max(drawdowns) if drawdowns else 0,
        'sharpe_ratio': sharpe_ratio,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'avg_r_multiple': avg_r,
        'r_gt_1_count': r_gt_1,
        'r_gt_2_count': r_gt_2,
        'winning_trades': len(winning_trades),
        'losing_trades': len(losing_trades)
    }

def aggregate_engine_utilization(manifests: List[Dict[str, Any]]) -> Dict[str, int]:
    """Aggregate engine utilization across all chunks."""
    total_engines = {'smc': 0, 'wyckoff': 0, 'momentum': 0, 'hob': 0, 'macro_veto': 0, 'ethbtc_veto': 0}

    for manifest in manifests:
        engine_util = manifest.get('engine_utilization', {})
        for engine, count in engine_util.items():
            if engine in total_engines:
                total_engines[engine] += count

    return total_engines

def create_trades_dataframe(manifests: List[Dict[str, Any]]) -> pd.DataFrame:
    """Create comprehensive trades DataFrame."""
    all_trades = []

    for manifest in manifests:
        chunk_info = manifest.get('chunk_info', {})
        chunk_start = chunk_info.get('start', '')

        trades = manifest.get('trades', [])
        for trade in trades:
            trade_record = trade.copy()
            trade_record['chunk_start'] = chunk_start
            all_trades.append(trade_record)

    if not all_trades:
        return pd.DataFrame()

    df = pd.DataFrame(all_trades)

    # Add derived columns
    if 'entry_time' in df.columns:
        df['entry_time'] = pd.to_datetime(df['entry_time'])
        df['entry_date'] = df['entry_time'].dt.date
        df['entry_hour'] = df['entry_time'].dt.hour

    if 'pnl' in df.columns:
        df['is_winner'] = df['pnl'] > 0

    return df

def generate_summary_report(manifests: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate comprehensive summary report."""
    if not manifests:
        return {'error': 'No manifests found'}

    # Basic info
    first_chunk = manifests[0].get('chunk_info', {})
    last_chunk = manifests[-1].get('chunk_info', {})

    performance = aggregate_performance(manifests)
    engine_util = aggregate_engine_utilization(manifests)

    # Time period analysis
    period_start = first_chunk.get('start', 'Unknown')
    period_end = last_chunk.get('end', 'Unknown')

    # Chunk statistics
    total_chunks = len(manifests)
    successful_chunks = len([m for m in manifests if 'error' not in m])
    failed_chunks = total_chunks - successful_chunks

    # Exit type breakdown
    all_trades = []
    for manifest in manifests:
        all_trades.extend(manifest.get('trades', []))

    exit_types = {}
    for trade in all_trades:
        exit_type = trade.get('exit_type', 'unknown')
        exit_types[exit_type] = exit_types.get(exit_type, 0) + 1

    # Engine combination analysis
    engine_combinations = {}
    for trade in all_trades:
        engines = trade.get('engines', [])
        combo = '+'.join(sorted(engines)) if engines else 'none'
        engine_combinations[combo] = engine_combinations.get(combo, 0) + 1

    return {
        'summary': {
            'asset': first_chunk.get('asset', 'Unknown'),
            'period_start': period_start,
            'period_end': period_end,
            'total_chunks': total_chunks,
            'successful_chunks': successful_chunks,
            'failed_chunks': failed_chunks,
            'config_version': first_chunk.get('config_version', 'Unknown')
        },
        'performance': performance,
        'engine_utilization': engine_util,
        'exit_type_breakdown': exit_types,
        'engine_combinations': engine_combinations,
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'bull_machine_version': '1.7.1-enhanced',
            'total_manifests_processed': len(manifests)
        }
    }

def main():
    parser = argparse.ArgumentParser(description='Merge Bull Machine chunk manifests')
    parser.add_argument('--in_dir', required=True, help='Input checkpoint directory')
    parser.add_argument('--out_json', help='Output JSON summary file')
    parser.add_argument('--out_csv', help='Output CSV trades file')
    parser.add_argument('--out_dir', help='Output directory (will create JSON and CSV files)')

    args = parser.parse_args()

    # Set default output paths
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        if not args.out_json:
            args.out_json = os.path.join(args.out_dir, 'year_summary.json')
        if not args.out_csv:
            args.out_csv = os.path.join(args.out_dir, 'year_trades.csv')

    print("🔧 BULL MACHINE CHUNK MANIFEST MERGER")
    print("=" * 50)
    print(f"Input directory: {args.in_dir}")

    # Load manifests
    print("📂 Loading chunk manifests...")
    manifests = load_chunk_manifests(args.in_dir)

    if not manifests:
        print("❌ No valid manifests found!")
        return

    print(f"✅ Loaded {len(manifests)} manifests")

    # Generate summary
    print("📊 Aggregating performance data...")
    summary = generate_summary_report(manifests)

    # Create trades DataFrame
    print("📋 Creating trades DataFrame...")
    trades_df = create_trades_dataframe(manifests)

    # Save outputs
    if args.out_json:
        print(f"💾 Saving summary to: {args.out_json}")
        with open(args.out_json, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

    if args.out_csv and not trades_df.empty:
        print(f"💾 Saving trades to: {args.out_csv}")
        trades_df.to_csv(args.out_csv, index=False)

    # Print quick summary
    perf = summary.get('performance', {})
    print("\n📈 MERGED RESULTS SUMMARY:")
    print("-" * 50)
    print(f"Period: {summary['summary']['period_start']} → {summary['summary']['period_end']}")
    print(f"Total trades: {perf.get('total_trades', 0)}")
    print(f"Final balance: ${perf.get('final_balance', 0):,.2f}")
    print(f"Total return: {perf.get('total_return', 0):.2f}%")
    print(f"Win rate: {perf.get('win_rate', 0):.1f}%")
    print(f"Profit factor: {perf.get('profit_factor', 0):.2f}")
    print(f"Max drawdown: {perf.get('max_drawdown', 0):.2f}%")
    print(f"Sharpe ratio: {perf.get('sharpe_ratio', 0):.2f}")

    engine_util = summary.get('engine_utilization', {})
    total_signals = sum(engine_util.values())
    if total_signals > 0:
        print("\n🔧 ENGINE UTILIZATION:")
        for engine, count in engine_util.items():
            pct = count / total_signals * 100
            print(f"  {engine.upper()}: {count} ({pct:.1f}%)")

    print(f"\n✅ Merge complete! Summary saved to {args.out_json}")

if __name__ == "__main__":
    main()