#!/usr/bin/env python3
"""
BTC Multi-Timeframe Analysis with Bull Machine v1.7.2
Full MTF confluence system with proper temporal alignment
"""

from bull_machine_config import get_config_path, get_data_path
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from engine_factory import EngineFactory
from engine.timeframes.mtf_alignment import MTFAlignmentEngine

class BTCMTFAnalyzer:
    def __init__(self, asset="BTC"):
        """Initialize with MTF capabilities"""
        self.asset = asset

        try:
            # Load configuration
            config_path = get_config_path(asset)
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            print(f"✅ Loaded config: {config_path.name}")

            # Initialize engines using factory
            self.engines = EngineFactory.build_all_engines(self.config)

            # Initialize MTF alignment engine
            self.mtf_engine = MTFAlignmentEngine(self.config.get('timeframes', {}))
            print("✅ MTF alignment engine initialized")

        except Exception as e:
            print(f"❌ Initialization error: {e}")
            raise

        # Get calibrated thresholds
        self.conf_threshold = self.config['fusion']['calibration_thresholds']['confidence']
        self.strength_threshold = self.config['fusion']['calibration_thresholds']['strength']

    def load_mtf_data(self):
        """Load multi-timeframe BTC data"""
        data = {}
        timeframes = ['1h', '4h', '1d']

        for tf in timeframes:
            try:
                data_path = get_data_path(self.asset, tf)
                df = pd.read_csv(data_path)

                # Standardize data format
                df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
                df = df.set_index('time').sort_index()
                df.columns = df.columns.str.lower()

                # Ensure required columns
                required_cols = ['open', 'high', 'low', 'close', 'total buy volume']
                if all(col in df.columns for col in required_cols):
                    data[tf.upper()] = df
                    print(f"✅ {self.asset} {tf.upper()}: {len(df)} bars ({df.index[0]} to {df.index[-1]})")
                else:
                    print(f"⚠️  {tf.upper()} data missing required columns")

            except Exception as e:
                print(f"❌ Failed to load {tf} data: {e}")
                data[tf.upper()] = None

        return data

    def run_mtf_backtest(self, lookback_bars=500):
        """Run MTF confluence backtest"""

        print(f"\n🚀 RUNNING {self.asset} MTF CONFLUENCE BACKTEST")
        print("=" * 60)

        try:
            # Load MTF data
            data = self.load_mtf_data()

            if not all(data.get(tf) is not None for tf in ['1H', '4H', '1D']):
                print("❌ Missing required timeframe data")
                return None

            # Use 4H as primary timeframe
            df_4h = data['4H']
            df_1h = data['1H']
            df_1d = data['1D']

            # Validate data sufficiency
            if len(df_4h) < lookback_bars + 50:
                print(f"⚠️  Insufficient 4H data: {len(df_4h)} bars")
                lookback_bars = max(100, len(df_4h) - 50)
                print(f"📊 Adjusted to {lookback_bars} bars")

            return self._execute_mtf_backtest(df_1h, df_4h, df_1d, lookback_bars)

        except Exception as e:
            print(f"❌ MTF backtest failed: {e}")
            return None

    def _execute_mtf_backtest(self, df_1h, df_4h, df_1d, lookback_bars):
        """Execute MTF confluence backtest logic"""

        start_idx = len(df_4h) - lookback_bars
        end_idx = len(df_4h) - 1

        period_start = df_4h.index[start_idx]
        period_end = df_4h.index[end_idx]

        print(f"📅 MTF Backtest period: {period_start} to {period_end}")
        print(f"📊 Primary timeframe: 4H ({lookback_bars} bars)")
        print(f"🔄 Confluence timeframes: 1H, 4H, 1D")

        # Initialize backtest state
        starting_balance = 100000
        current_balance = starting_balance
        trades = []
        in_position = False

        # Execute MTF backtest loop
        for i in range(start_idx + 50, end_idx):
            current_bar = df_4h.iloc[i]
            current_timestamp = df_4h.index[i]
            current_price = current_bar['close']

            try:
                # Generate MTF confluence signals
                mtf_signals = self._generate_mtf_signals(df_1h, df_4h, df_1d, current_timestamp, i)

                if mtf_signals and not in_position:
                    # Check confluence requirements
                    if mtf_signals['confluence_score'] >= self.conf_threshold:
                        direction = mtf_signals['direction']
                        confidence = mtf_signals['confluence_score']

                        # Calculate position size (1% risk)
                        risk_amount = current_balance * 0.01
                        position_size = risk_amount / current_price

                        trade = {
                            'entry_time': current_timestamp,
                            'entry_price': current_price,
                            'direction': direction,
                            'size': position_size,
                            'confidence': confidence,
                            'mtf_signals': mtf_signals['timeframe_signals']
                        }

                        in_position = True
                        entry_trade = trade.copy()

                elif in_position:
                    # Check exit conditions
                    bars_in_trade = i - df_4h.index.get_loc(entry_trade['entry_time'])

                    should_exit = (
                        bars_in_trade >= 20 or  # Max 20 bars
                        (entry_trade['direction'] == 'long' and current_price >= entry_trade['entry_price'] * 1.025) or  # 2.5% profit
                        (entry_trade['direction'] == 'short' and current_price <= entry_trade['entry_price'] * 0.975) or  # 2.5% profit
                        (entry_trade['direction'] == 'long' and current_price <= entry_trade['entry_price'] * 0.985) or  # 1.5% stop
                        (entry_trade['direction'] == 'short' and current_price >= entry_trade['entry_price'] * 1.015)  # 1.5% stop
                    )

                    if should_exit:
                        # Exit position
                        if entry_trade['direction'] == 'long':
                            pnl = (current_price - entry_trade['entry_price']) * entry_trade['size']
                        else:
                            pnl = (entry_trade['entry_price'] - current_price) * entry_trade['size']

                        current_balance += pnl

                        trade_result = {
                            **entry_trade,
                            'exit_time': current_timestamp,
                            'exit_price': current_price,
                            'pnl': pnl,
                            'pnl_pct': (pnl / current_balance) * 100,
                            'duration_bars': bars_in_trade
                        }

                        trades.append(trade_result)
                        in_position = False

            except Exception as e:
                print(f"⚠️  MTF signal error at {current_timestamp}: {e}")
                continue

        # Calculate results
        total_return = ((current_balance - starting_balance) / starting_balance) * 100

        if trades:
            winning_trades = [t for t in trades if t['pnl'] > 0]
            win_rate = len(winning_trades) / len(trades) * 100

            if any(t['pnl'] <= 0 for t in trades):
                total_wins = sum(t['pnl'] for t in winning_trades)
                total_losses = abs(sum(t['pnl'] for t in trades if t['pnl'] <= 0))
                profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
            else:
                profit_factor = float('inf')
        else:
            win_rate = profit_factor = 0

        results = {
            'total_return': total_return,
            'total_pnl': current_balance - starting_balance,
            'final_balance': current_balance,
            'total_trades': len(trades),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'trades': trades,
            'period': f"{period_start} to {period_end}",
            'mtf_enabled': True,
            'confluence_threshold': self.conf_threshold
        }

        return results

    def _generate_mtf_signals(self, df_1h, df_4h, df_1d, current_timestamp, current_4h_idx):
        """Generate MTF confluence signals"""

        try:
            # Get aligned timeframe data for current timestamp
            timeframe_signals = {}

            # 1H signals (short-term momentum)
            try:
                h1_idx = df_1h.index.get_loc(current_timestamp, method='nearest')
                if h1_idx >= 20:
                    h1_data = df_1h.iloc[h1_idx-20:h1_idx+1]
                    h1_momentum = (h1_data['close'].iloc[-1] / h1_data['close'].iloc[-10] - 1) * 100

                    if h1_momentum > 1.0:
                        timeframe_signals['1H'] = {'direction': 'long', 'strength': min(abs(h1_momentum) / 3, 1.0)}
                    elif h1_momentum < -1.0:
                        timeframe_signals['1H'] = {'direction': 'short', 'strength': min(abs(h1_momentum) / 3, 1.0)}
            except (KeyError, IndexError):
                pass

            # 4H signals (primary timeframe)
            if current_4h_idx >= 20:
                h4_data = df_4h.iloc[current_4h_idx-20:current_4h_idx+1]
                h4_momentum = (h4_data['close'].iloc[-1] / h4_data['close'].iloc[-15] - 1) * 100

                if h4_momentum > 1.5:
                    timeframe_signals['4H'] = {'direction': 'long', 'strength': min(abs(h4_momentum) / 4, 1.0)}
                elif h4_momentum < -1.5:
                    timeframe_signals['4H'] = {'direction': 'short', 'strength': min(abs(h4_momentum) / 4, 1.0)}

            # 1D signals (trend filter)
            try:
                d1_idx = df_1d.index.get_loc(current_timestamp, method='nearest')
                if d1_idx >= 10:
                    d1_data = df_1d.iloc[d1_idx-10:d1_idx+1]
                    d1_trend = (d1_data['close'].iloc[-1] / d1_data['close'].iloc[-7] - 1) * 100

                    if d1_trend > 2.0:
                        timeframe_signals['1D'] = {'direction': 'long', 'strength': min(abs(d1_trend) / 7, 1.0)}
                    elif d1_trend < -2.0:
                        timeframe_signals['1D'] = {'direction': 'short', 'strength': min(abs(d1_trend) / 7, 1.0)}
            except (KeyError, IndexError):
                pass

            # Calculate confluence (require at least 2 timeframes)
            if len(timeframe_signals) >= 2:
                # Count directional votes
                directions = [signal['direction'] for signal in timeframe_signals.values()]
                direction_counts = {}
                for direction in directions:
                    direction_counts[direction] = direction_counts.get(direction, 0) + 1

                # Check if majority agrees (at least 2 out of 3)
                max_votes = max(direction_counts.values())
                if max_votes >= 2:
                    winning_direction = [d for d, count in direction_counts.items() if count == max_votes][0]

                    # Calculate confluence score based on aligned timeframes
                    aligned_signals = {tf: sig for tf, sig in timeframe_signals.items()
                                     if sig['direction'] == winning_direction}

                    total_strength = sum(signal['strength'] for signal in aligned_signals.values())
                    confluence_score = total_strength / len(aligned_signals)

                    return {
                        'direction': winning_direction,
                        'confluence_score': confluence_score,
                        'timeframe_signals': timeframe_signals,
                        'timeframes_aligned': len(aligned_signals)
                    }

            return None

        except Exception as e:
            return None

def main():
    """Main execution function"""

    print("🚀 BULL MACHINE v1.7.2 MTF BTC ANALYSIS")
    print("📊 Multi-timeframe confluence system")
    print()

    try:
        # Initialize MTF analyzer
        analyzer = BTCMTFAnalyzer("BTC")

        # Run MTF backtest
        results = analyzer.run_mtf_backtest(lookback_bars=800)

        if results:
            print(f"\n📈 {analyzer.asset} MTF CONFLUENCE RESULTS")
            print("=" * 50)
            print(f"Period: {results['period']}")
            print(f"MTF Enabled: {results['mtf_enabled']}")
            print(f"Confluence Threshold: {results['confluence_threshold']:.2f}")
            print(f"Total Return: {results['total_return']:+.2f}%")
            print(f"Total P&L: ${results['total_pnl']:+,.2f}")
            print(f"Win Rate: {results['win_rate']:.1f}%")
            print(f"Total Trades: {results['total_trades']}")
            print(f"Profit Factor: {results['profit_factor']:.2f}")
            print(f"Final Balance: ${results['final_balance']:,.2f}")

            # Show confluence breakdown
            if results['trades']:
                confluence_trades = [t for t in results['trades'] if 'mtf_signals' in t]
                if confluence_trades:
                    avg_tfs = np.mean([len(t['mtf_signals']) for t in confluence_trades])
                    print(f"\n🔄 MTF Performance:")
                    print(f"Average timeframes per trade: {avg_tfs:.1f}")
                    print(f"Confluence trades: {len(confluence_trades)}")

            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"results/{analyzer.asset}_mtf_backtest_{timestamp}.json"

            try:
                with open(results_file, 'w') as f:
                    serializable_results = json.loads(json.dumps(results, default=str))
                    json.dump(serializable_results, f, indent=2)
                print(f"\n💾 MTF results saved to: {results_file}")

            except Exception as e:
                print(f"⚠️  Could not save results: {e}")

        else:
            print("❌ MTF backtest failed - no results generated")

    except Exception as e:
        print(f"❌ Critical error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()