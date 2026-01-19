#!/usr/bin/env python3
"""
Replay Runner: 60x Accelerated Validation

Replays historical data using cached feature stores at 60x speed to validate
backtest parity before live deployment.

Key differences from hybrid_runner.py:
- Uses pre-built feature stores (no live data fetching)
- Iterates bar-by-bar in timestamp order
- Uses KnowledgeAwareBacktest logic (not fast signals)
- Validates fusion → hooks → entries/exits exactly like backtest
- Writes same trade/telemetry logs for parity comparison

Usage:
    python3 bin/live/replay_runner.py \
      --asset BTC \
      --start 2024-01-01 --end 2024-12-31 \
      --features data/features_mtf/BTC_1H_2024-01-01_to_2024-12-31.parquet \
      --config configs/v3_replay_2024/BTC_2024_best.json \
      --speed 60 \
      --output reports/replay/BTC_2024_60x.json
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
import time
from dataclasses import asdict

from bin.backtest_knowledge_v2 import KnowledgeParams, KnowledgeAwareBacktest


class ReplayRunner:
    """Accelerated replay runner for validation testing."""

    def __init__(
        self,
        asset: str,
        features_path: str,
        config_path: str,
        start_date: str = None,
        end_date: str = None,
        speed: int = 60,
        output_path: str = None
    ):
        self.asset = asset.upper()
        self.features_path = features_path
        self.config_path = config_path
        self.start_date = start_date
        self.end_date = end_date
        self.speed = speed
        self.output_path = output_path or f"reports/replay/{asset}_replay_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        print(f"🎬 Bull Machine Replay Runner - {self.asset}")
        print(f"📊 Features: {features_path}")
        print(f"⚙️  Config: {config_path}")
        print(f"⏩ Speed: {speed}x")
        print(f"📅 Period: {start_date} → {end_date}")

        # Load frozen config
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # Load feature store
        print("\n📂 Loading feature store...")
        self.df = pd.read_parquet(features_path)

        # Filter to date range
        if start_date:
            self.df = self.df[self.df.index >= start_date]
        if end_date:
            self.df = self.df[self.df.index <= end_date]

        print(f"✅ Loaded {len(self.df)} bars of cached features")
        print(f"   Date range: {self.df.index.min()} to {self.df.index.max()}")
        print(f"   Features: {len(self.df.columns)} columns")

        # Convert config to KnowledgeParams
        self.params = self._config_to_params(self.config)

        # Initialize backtest engine
        self.backtest = KnowledgeAwareBacktest(
            self.df,
            self.params,
            starting_capital=10000.0
        )

    def _config_to_params(self, config: Dict) -> KnowledgeParams:
        """Convert frozen config JSON to KnowledgeParams."""
        # Extract params from config structure
        # Configs use different structures, handle both flat and nested

        # Try nested structure first (v3 configs)
        if 'params' in config:
            p = config['params']
        else:
            # Flat structure
            p = config

        return KnowledgeParams(
            wyckoff_weight=p.get('wyckoff_weight', 0.3),
            liquidity_weight=p.get('liquidity_weight', 0.25),
            momentum_weight=p.get('momentum_weight', 0.2),
            macro_weight=p.get('macro_weight', 0.15),
            pti_weight=p.get('pti_weight', 0.1),
            tier1_threshold=p.get('tier1_threshold', 0.75),
            tier2_threshold=p.get('tier2_threshold', 0.60),
            tier3_threshold=p.get('tier3_threshold', 0.45),
            require_m1m2_confirmation=p.get('require_m1m2_confirmation', False),
            require_macro_alignment=p.get('require_macro_alignment', False),
            atr_stop_mult=p.get('atr_stop_mult', 1.5),
            trailing_atr_mult=p.get('trailing_atr_mult', 2.0),
            max_hold_bars=p.get('max_hold_bars', 24),
            max_risk_pct=p.get('max_risk_pct', 0.02),
            volatility_scaling=p.get('volatility_scaling', True),
            use_smart_exits=p.get('use_smart_exits', True),
            breakeven_after_tp1=p.get('breakeven_after_tp1', True),
            adaptive_max_hold=p.get('adaptive_max_hold', False)
        )

    def run(self, verbose: bool = False) -> Dict:
        """
        Run replay at accelerated speed.

        Args:
            verbose: If True, print bar-by-bar debug info

        Returns:
            Dict with replay results and metrics
        """
        print("\n🎬 Starting replay...")
        print(f"{'='*80}")

        # Track timing
        replay_start = time.time()

        # Run backtest (already has all features loaded)
        results = self.backtest.run()

        replay_elapsed = time.time() - replay_start

        print(f"\n{'='*80}")
        print(f"✅ Replay complete!")
        print(f"⏱️  Wall time: {replay_elapsed:.2f}s")
        print(f"⏩ Effective speed: ~{len(self.df) / max(replay_elapsed, 1):.0f} bars/sec")

        # Print results
        self._print_results(results)

        # Write detailed output
        output = self._prepare_output(results, replay_elapsed)
        self._write_output(output)

        return output

    def _print_results(self, results: Dict):
        """Print replay results summary."""
        print(f"\n{'='*80}")
        print(f"REPLAY RESULTS - {self.asset}")
        print(f"{'='*80}")
        print(f"Total Trades:    {results['total_trades']}")
        print(f"Total PNL:       ${results['total_pnl']:,.2f}")
        print(f"Win Rate:        {results['win_rate']*100:.1f}%")
        print(f"Profit Factor:   {results['profit_factor']:.2f}")
        print(f"Max Drawdown:    {results['max_drawdown']*100:.2f}%")
        print(f"Final Equity:    ${results['final_equity']:,.2f}")
        print(f"Total Return:    {(results['final_equity']/10000 - 1)*100:.2f}%")

        # Exit reason breakdown
        print(f"\nExit Reasons:")
        exit_reasons = {}
        for trade in results['trades']:
            reason = str(trade.exit_reason)
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

        for reason, count in sorted(exit_reasons.items(), key=lambda x: -x[1]):
            pct = count / len(results['trades']) * 100 if results['trades'] else 0
            print(f"  {reason:20s}: {count:3d} ({pct:5.1f}%)")

        # Adaptive max-hold stats (if enabled)
        if self.params.adaptive_max_hold and hasattr(self.backtest, '_adaptive_extension_count'):
            print(f"\nAdaptive Max-Hold:")
            print(f"  Extensions: {self.backtest._adaptive_extension_count}")
            if self.backtest._adaptive_events:
                print(f"  Events logged: {len(self.backtest._adaptive_events)}")

        print(f"{'='*80}\n")

    def _prepare_output(self, results: Dict, elapsed_time: float) -> Dict:
        """Prepare comprehensive output for JSON export."""
        output = {
            'metadata': {
                'asset': self.asset,
                'features_path': self.features_path,
                'config_path': self.config_path,
                'start_date': str(self.df.index.min()),
                'end_date': str(self.df.index.max()),
                'total_bars': len(self.df),
                'replay_speed': self.speed,
                'wall_time_seconds': elapsed_time,
                'timestamp': datetime.now().isoformat()
            },
            'config': self.config,
            'metrics': {
                'total_trades': results['total_trades'],
                'total_pnl': results['total_pnl'],
                'win_rate': results['win_rate'],
                'profit_factor': results['profit_factor'],
                'max_drawdown': results['max_drawdown'],
                'final_equity': results['final_equity'],
                'total_return_pct': (results['final_equity'] / 10000 - 1) * 100
            },
            'trades': []
        }

        # Export trades
        for trade in results['trades']:
            # Calculate hold time in bars
            hold_bars = int((trade.exit_time - trade.entry_time).total_seconds() / 3600) if trade.exit_time else 0

            # Calculate PNL percentage
            pnl_pct = (trade.net_pnl / trade.position_size * 100) if trade.position_size > 0 else 0.0

            output['trades'].append({
                'entry_time': str(trade.entry_time),
                'exit_time': str(trade.exit_time),
                'direction': 'long' if trade.direction > 0 else 'short',
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'net_pnl': trade.net_pnl,
                'gross_pnl': trade.gross_pnl,
                'pnl_pct': pnl_pct,
                'exit_reason': str(trade.exit_reason),
                'hold_bars': hold_bars,
                'entry_reason': trade.entry_reason,
                'fusion_score': trade.entry_fusion_score,
                'wyckoff_phase': trade.wyckoff_phase
            })

        # Add adaptive events if available
        if hasattr(self.backtest, '_adaptive_events'):
            output['adaptive_events'] = [
                asdict(event) for event in self.backtest._adaptive_events
            ]

        return output

    def _write_output(self, output: Dict):
        """Write output to JSON file."""
        output_path = Path(self.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"📝 Results written to: {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Bull Machine Replay Runner (60x validation)')
    parser.add_argument('--asset', required=True, help='Asset symbol (BTC, ETH, SPY)')
    parser.add_argument('--features', required=True, help='Path to feature store parquet')
    parser.add_argument('--config', required=True, help='Path to frozen config JSON')
    parser.add_argument('--start', help='Start date (YYYY-MM-DD), optional')
    parser.add_argument('--end', help='End date (YYYY-MM-DD), optional')
    parser.add_argument('--speed', type=int, default=60, help='Replay speed multiplier (default: 60)')
    parser.add_argument('--output', help='Output path for results JSON')
    parser.add_argument('--verbose', action='store_true', help='Print bar-by-bar debug info')

    args = parser.parse_args()

    runner = ReplayRunner(
        asset=args.asset,
        features_path=args.features,
        config_path=args.config,
        start_date=args.start,
        end_date=args.end,
        speed=args.speed,
        output_path=args.output
    )

    results = runner.run(verbose=args.verbose)

    # Exit with success code
    sys.exit(0)


if __name__ == '__main__':
    main()
