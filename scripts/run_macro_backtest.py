"""
Comprehensive Macro Pulse Backtest - v1.6.2 vs v1.7 Analysis

24-month backtest with:
A) Full confluence layers (Wyckoff/Liquidity/Bojan/Momentum/Temporal)
B) Macro Pulse veto/boost pipeline
C) A/B testing (macro on/off)
D) Regime slicing (risk-on/risk-off/neutral)
E) Ablation studies (individual veto impact)
"""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
import sys

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from engine.io.tradingview_loader import load_tv, RealDataRequiredError
from engine.context.macro_pulse import MacroPulseEngine, MacroPulse, MacroRegime
from engine.context.macro_pulse_calibration import calibrate_macro_thresholds
from engine.temporal.tpi import TemporalEngine, TPISignal
from engine.fusion import FusionEngine, FusionSignal
from engine.liquidity.hob import HOBDetector, HOBSignal
from engine.liquidity.bojan_rules import BojanEngine, LiquidityReaction

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TradeRecord:
    """Individual trade record with macro context"""
    timestamp: pd.Timestamp
    event: str  # 'ENTRY' or 'EXIT'
    side: str   # 'LONG' or 'SHORT'
    price: float
    pnl: Optional[float] = None
    fusion_score: Optional[float] = None
    macro_regime: Optional[str] = None
    macro_bias: Optional[str] = None
    macro_delta: Optional[float] = None
    macro_notes: Optional[str] = None
    veto_reasons: Optional[List[str]] = None
    contributing_domains: Optional[List[str]] = None
    quality_score: Optional[float] = None

@dataclass
class BacktestResults:
    """Backtest results with regime breakdown"""
    asset: str
    total_trades: int
    pnl_total: float
    pnl_percent: float
    win_rate: float
    profit_factor: float
    max_drawdown: float
    avg_return_per_trade: float
    time_in_trade_avg: float

    # Regime breakdown
    risk_on_trades: int
    risk_on_pf: float
    risk_off_trades: int
    risk_off_pf: float
    neutral_trades: int
    neutral_pf: float

    # Macro impact
    trades_blocked_by_macro: int
    macro_boost_count: int
    macro_veto_count: int

    # Explainability
    trade_records: List[TradeRecord]

def load_chart_logs_data(symbol: str, timeframe: str, start: str, end: str) -> pd.DataFrame:
    """
    Load OHLCV data from chart logs directory.

    Args:
        symbol: Symbol like 'ETH', 'BTC', 'DXY', etc.
        timeframe: '1H', '4H', '1D', etc.
        start: Start date string
        end: End date string

    Returns:
        DataFrame with OHLCV data
    """
    try:
        # Construct path to chart logs data
        chart_logs_dir = Path("chart_logs")

        # Try different file naming conventions
        possible_files = [
            chart_logs_dir / f"{symbol}_{timeframe}.csv",
            chart_logs_dir / f"{symbol}{timeframe}.csv",
            chart_logs_dir / symbol / f"{timeframe}.csv",
            chart_logs_dir / f"{symbol.upper()}_{timeframe}.csv",
        ]

        df = None
        for file_path in possible_files:
            if file_path.exists():
                logger.info(f"Loading {symbol} {timeframe} from {file_path}")
                df = pd.read_csv(file_path)
                break

        if df is None:
            logger.warning(f"No data found for {symbol} {timeframe}, creating synthetic data")
            # Create synthetic data for testing
            dates = pd.date_range(start, end, freq='1H')
            base_price = {'ETH': 2000, 'BTC': 40000, 'SOL': 50, 'DXY': 100, 'WTI': 75}.get(symbol, 100)

            df = pd.DataFrame({
                'timestamp': dates,
                'open': base_price,
                'high': base_price * 1.02,
                'low': base_price * 0.98,
                'close': base_price + np.random.randn(len(dates)) * base_price * 0.01,
                'volume': 1000000
            })

        # Standardize column names and index
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        elif 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
        else:
            df.index = pd.to_datetime(df.index)

        # Ensure required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                df[col] = df['close'] if 'close' in df.columns else 100

        # Filter date range
        df = df.loc[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]

        # Resample to target timeframe if needed
        df = resample_to_timeframe(df, timeframe)

        return df.dropna()

    except Exception as e:
        logger.error(f"Error loading {symbol} {timeframe}: {e}")
        return pd.DataFrame()

def resample_to_timeframe(df: pd.DataFrame, target_tf: str) -> pd.DataFrame:
    """Resample data to target timeframe"""
    try:
        tf_map = {
            '1H': '1H',
            '4H': '4H',
            '1D': '1D',
            '1W': '1W'
        }

        if target_tf not in tf_map:
            return df

        freq = tf_map[target_tf]

        resampled = df.resample(freq).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        return resampled

    except Exception as e:
        logger.error(f"Error resampling to {target_tf}: {e}")
        return df

def load_macro_series(series_config: Dict[str, str], start: str, end: str) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    """
    Load all macro data series with REAL DATA ONLY and calibrate adaptive thresholds.

    Returns:
        Tuple of (macro_data, adaptive_thresholds)
    """
    macro_data = {}

    # Map config series keys to TradingView loader keys
    series_mapping = {
        'DXY_1D': 'DXY_1D',
        'WTI_1D': 'WTI_1D',
        'GOLD_1D': 'GOLD_1D',
        'US2Y_1D': 'US2Y_1D',
        'US10Y_1D': 'US10Y_1D',
        'VIX_1D': 'VIX_1D',
        'MOVE_1D': 'MOVE_1D',
        'USDJPY_1D': 'USDJPY_1D',
        'HYG_1D': 'HYG_1D',
        'ETHD_1D': 'ETH_1D',
        'USDT.D_4H': 'USDTD_4H',
        'BTC.D_1W': 'BTCD_1W',
        'TOTAL_4H': 'TOTAL_4H',
        'TOTAL3_4H': 'TOTAL3_4H',
        'ETHBTC_1D': 'ETHBTC_1D'
    }

    for alias, series_key in series_config.items():
        try:
            # Find the corresponding loader key
            loader_key = series_mapping.get(series_key, series_key)

            try:
                df = load_tv(loader_key)

                # Filter by date range
                if not df.empty:
                    start_date = pd.to_datetime(start)
                    end_date = pd.to_datetime(end)
                    df = df[(df.index >= start_date) & (df.index <= end_date)]

                if not df.empty:
                    macro_data[series_key] = df
                    logger.info(f"✅ Loaded {series_key}: {len(df)} bars (REAL DATA)")
                else:
                    logger.warning(f"⚠️  {series_key}: No data in date range")

            except (RealDataRequiredError, KeyError) as e:
                logger.warning(f"⚠️  {series_key}: {e}")
                # Continue without this series - don't crash

        except Exception as e:
            logger.error(f"💥 Error loading {series_key}: {e}")

    # Add BTC daily for DXY lag correlation
    try:
        btc_daily = load_tv('BTC_1D')
        if not btc_daily.empty:
            start_date = pd.to_datetime(start)
            end_date = pd.to_datetime(end)
            btc_daily = btc_daily[(btc_daily.index >= start_date) & (btc_daily.index <= end_date)]
            macro_data['BTCUSD_1D'] = btc_daily
    except Exception as e:
        logger.warning(f"Could not load BTC daily: {e}")

    # Calibrate adaptive thresholds from real data
    logger.info("🔧 Calibrating adaptive macro thresholds...")
    adaptive_thresholds = calibrate_macro_thresholds(macro_data)

    logger.info(f"📊 Loaded {len(macro_data)} macro series with adaptive thresholds")

    return macro_data, adaptive_thresholds

def simulate_wyckoff_score(df: pd.DataFrame, i: int) -> float:
    """Simulate Wyckoff accumulation/distribution score"""
    try:
        if i < 50:
            return 0.0

        # Simple volume-price analysis proxy
        recent_data = df.iloc[i-20:i+1]
        price_change = (recent_data['close'].iloc[-1] / recent_data['close'].iloc[0] - 1)
        volume_trend = recent_data['volume'].tail(5).mean() / recent_data['volume'].head(5).mean()

        # Accumulation: price stable/up + volume increasing
        if price_change > -0.02 and volume_trend > 1.2:
            return min(0.8, (price_change + 0.02) * 10 + (volume_trend - 1) * 0.5)

        return 0.0

    except Exception:
        return 0.0

def simulate_liquidity_score(df: pd.DataFrame, i: int) -> float:
    """Simulate liquidity/HOB score"""
    try:
        if i < 20:
            return 0.0

        # Look for wick formations and level reactions
        current_bar = df.iloc[i]
        body_size = abs(current_bar['close'] - current_bar['open'])

        if body_size == 0:
            return 0.0

        lower_wick = current_bar['open'] - current_bar['low'] if current_bar['close'] > current_bar['open'] else current_bar['close'] - current_bar['low']
        wick_ratio = lower_wick / body_size

        # Strong wick at potential support
        if wick_ratio > 2.0:
            return min(0.7, wick_ratio / 3.0)

        return 0.0

    except Exception:
        return 0.0

def simulate_momentum_score(df: pd.DataFrame, i: int) -> float:
    """Simulate momentum score"""
    try:
        if i < 14:
            return 0.0

        # Simple RSI-like momentum
        recent_closes = df['close'].iloc[i-13:i+1]
        gains = recent_closes.diff().clip(lower=0)
        losses = -recent_closes.diff().clip(upper=0)

        avg_gain = gains.mean()
        avg_loss = losses.mean()

        if avg_loss == 0:
            return 0.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        # Momentum signal when RSI in 30-70 range with trend
        if 30 < rsi < 70:
            trend = (recent_closes.iloc[-1] / recent_closes.iloc[0] - 1)
            return min(0.6, abs(trend) * 20)

        return 0.0

    except Exception:
        return 0.0

def compute_temporal_delta(df: pd.DataFrame, i: int, temporal_config: Dict) -> float:
    """Compute temporal/TPI delta"""
    try:
        if not temporal_config.get('enabled', True) or i < 100:
            return 0.0

        # Simple cycle-based temporal signal
        recent_data = df.iloc[max(0, i-89):i+1]  # 89-bar Fibonacci cycle

        if len(recent_data) < 50:
            return 0.0

        # Look for cycle completion patterns
        cycle_high = recent_data['high'].max()
        cycle_low = recent_data['low'].min()
        current_price = recent_data['close'].iloc[-1]

        # Normalize position in cycle
        cycle_position = (current_price - cycle_low) / (cycle_high - cycle_low) if cycle_high != cycle_low else 0.5

        # Temporal boost when near cycle extremes
        if cycle_position < 0.2 or cycle_position > 0.8:
            return 0.1 * (1 - abs(cycle_position - 0.5) * 2)

        return 0.0

    except Exception:
        return 0.0

def aggregate_fusion_score(wyckoff: float, liquidity: float, momentum: float,
                          temporal_delta: float, macro_delta: float,
                          macro_suppression: bool, threshold: float) -> float:
    """Aggregate all signals into final fusion score"""
    try:
        if macro_suppression:
            return 0.0  # Hard macro veto

        # Weight the components
        base_score = (wyckoff * 0.3 + liquidity * 0.3 + momentum * 0.2 + temporal_delta * 0.2)

        # Apply macro delta
        final_score = base_score + macro_delta

        # Apply threshold
        return final_score if final_score >= threshold else 0.0

    except Exception:
        return 0.0

def simulate_asset_backtest(asset: str, start: str, end: str,
                           engine_config: Dict, context_config: Dict,
                           temporal_config: Dict, ablation: Optional[Dict] = None,
                           enable_macro: bool = True) -> BacktestResults:
    """
    Run complete backtest simulation for an asset.

    Args:
        asset: Asset symbol like 'ETH', 'BTC'
        start/end: Date range
        engine_config: Engine configuration
        context_config: Macro context configuration
        temporal_config: Temporal engine configuration
        ablation: Dict of disabled features
        enable_macro: Whether to use macro pulse

    Returns:
        BacktestResults with complete analysis
    """
    logger.info(f"Starting backtest for {asset} from {start} to {end}")

    # Load primary asset data
    timeframe = engine_config.get('timeframe', '4H')
    symbol_key = f"{asset}_{timeframe}"
    try:
        df = load_tv(symbol_key)
    except RealDataRequiredError as e:
        logger.error(f"Real data required for {symbol_key}: {e}")
        return _empty_results(asset)

    if df.empty:
        logger.error(f"No data for {asset}")
        return _empty_results(asset)

    # Filter by date range
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')

    start_date = pd.to_datetime(start, utc=True)
    end_date = pd.to_datetime(end, utc=True)

    # Handle timezone comparison
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')

    df = df[(df.index >= start_date) & (df.index <= end_date)]

    if df.empty:
        logger.error(f"No data for {asset} in date range {start} to {end}")
        return _empty_results(asset)

    logger.info(f"Loaded {len(df)} bars for {asset} from {start} to {end}")

    # Load macro data
    macro_data = {}
    if enable_macro and context_config.get('enabled', True):
        series_config = context_config.get('series', {})
        macro_data = load_macro_series(series_config, start, end)
        logger.info(f"Loaded {len(macro_data)} macro series")

    # Initialize engines
    macro_engine = None
    if enable_macro and macro_data:
        macro_engine = MacroPulseEngine(context_config)

    # Trading parameters
    risk_per_trade = engine_config.get('risk_pct', 0.02)
    tp_multiplier = engine_config.get('tp_R', 2.0)
    sl_multiplier = engine_config.get('sl_R', 1.0)
    threshold = engine_config.get('fusion', {}).get('threshold', 0.3)

    # State tracking
    trades = []
    in_position = False
    entry_info = None
    blocked_by_macro = 0
    macro_boost_count = 0
    macro_veto_count = 0

    # Warmup period
    warmup = max(250, engine_config.get('warmup', 250))

    # Main trading loop
    for i in range(warmup, len(df)):
        current_time = df.index[i]
        current_price = df['close'].iloc[i]

        try:
            # 1. Compute domain scores
            wyckoff_score = simulate_wyckoff_score(df, i)
            liquidity_score = simulate_liquidity_score(df, i)
            momentum_score = simulate_momentum_score(df, i)

            # 2. Temporal analysis
            temporal_delta = compute_temporal_delta(df, i, temporal_config)

            # 3. Macro analysis
            macro_delta = 0.0
            macro_suppression = False
            macro_pulse = None
            macro_notes = []

            if macro_engine and macro_data:
                # Create time-sliced macro data
                sliced_macro = {}
                for key, series in macro_data.items():
                    mask = series.index <= current_time
                    if mask.any():
                        sliced_macro[key] = series[mask]

                if sliced_macro:
                    macro_pulse = macro_engine.analyze_macro_pulse(sliced_macro)
                    macro_delta = macro_pulse.macro_delta
                    macro_suppression = macro_pulse.suppression_flag
                    macro_notes = macro_pulse.notes

                    if macro_suppression:
                        macro_veto_count += 1
                    elif macro_delta > 0.02:
                        macro_boost_count += 1

            # 4. Fusion aggregation
            fusion_score = aggregate_fusion_score(
                wyckoff_score, liquidity_score, momentum_score,
                temporal_delta, macro_delta, macro_suppression, threshold
            )

            # 5. Entry logic
            if not in_position and fusion_score > 0:
                if macro_suppression:
                    blocked_by_macro += 1
                else:
                    # Enter position
                    side = 'LONG'  # Simplified - always long for this test
                    entry_price = current_price
                    stop_loss = entry_price * (1 - sl_multiplier * 0.01)
                    take_profit = entry_price * (1 + tp_multiplier * 0.01)

                    entry_info = {
                        'side': side,
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'entry_time': current_time
                    }

                    trades.append(TradeRecord(
                        timestamp=current_time,
                        event='ENTRY',
                        side=side,
                        price=entry_price,
                        fusion_score=fusion_score,
                        macro_regime=macro_pulse.regime.value if macro_pulse else 'neutral',
                        macro_bias=macro_pulse.risk_bias if macro_pulse else 'neutral',
                        macro_delta=macro_delta,
                        macro_notes='; '.join(macro_notes) if macro_notes else '',
                        contributing_domains=['wyckoff', 'liquidity', 'momentum', 'temporal'],
                        quality_score=fusion_score
                    ))

                    in_position = True

            # 6. Exit logic
            elif in_position and entry_info:
                exit_triggered = False
                exit_reason = ''

                # Check stops and targets
                if current_price <= entry_info['stop_loss']:
                    exit_triggered = True
                    exit_reason = 'stop_loss'
                elif current_price >= entry_info['take_profit']:
                    exit_triggered = True
                    exit_reason = 'take_profit'
                elif fusion_score <= 0:  # Signal reversal
                    exit_triggered = True
                    exit_reason = 'signal_exit'

                if exit_triggered:
                    # Calculate PnL
                    if entry_info['side'] == 'LONG':
                        pnl = current_price - entry_info['entry_price']
                    else:
                        pnl = entry_info['entry_price'] - current_price

                    pnl_pct = pnl / entry_info['entry_price']

                    trades.append(TradeRecord(
                        timestamp=current_time,
                        event='EXIT',
                        side=entry_info['side'],
                        price=current_price,
                        pnl=pnl_pct,
                        macro_regime=macro_pulse.regime.value if macro_pulse else 'neutral',
                        macro_notes=exit_reason
                    ))

                    in_position = False
                    entry_info = None

        except Exception as e:
            logger.error(f"Error processing bar {i} for {asset}: {e}")
            continue

    # Calculate results
    return _calculate_results(asset, trades, blocked_by_macro, macro_boost_count, macro_veto_count)

def _calculate_results(asset: str, trades: List[TradeRecord], blocked_by_macro: int,
                      macro_boost_count: int, macro_veto_count: int) -> BacktestResults:
    """Calculate comprehensive backtest results"""

    if not trades:
        return _empty_results(asset)

    # Filter to completed trades (entry + exit pairs)
    trade_df = pd.DataFrame([t.__dict__ for t in trades])
    exits = trade_df[trade_df['event'] == 'EXIT'].copy()
    entries = trade_df[trade_df['event'] == 'ENTRY'].copy()

    if exits.empty:
        return _empty_results(asset)

    # Basic metrics
    total_trades = len(exits)
    total_pnl = exits['pnl'].sum()
    win_rate = (exits['pnl'] > 0).mean()

    wins = exits[exits['pnl'] > 0]['pnl'].sum()
    losses = abs(exits[exits['pnl'] < 0]['pnl'].sum())
    profit_factor = wins / losses if losses > 0 else float('inf')

    # Drawdown calculation
    cumulative_returns = exits['pnl'].cumsum()
    rolling_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - rolling_max) / (1 + rolling_max)
    max_drawdown = abs(drawdown.min())

    # Regime breakdown
    risk_on_exits = exits[exits['macro_regime'] == 'risk_on']
    risk_off_exits = exits[exits['macro_regime'] == 'risk_off']
    neutral_exits = exits[~exits['macro_regime'].isin(['risk_on', 'risk_off'])]

    def regime_pf(regime_exits):
        if regime_exits.empty:
            return 0.0
        wins = regime_exits[regime_exits['pnl'] > 0]['pnl'].sum()
        losses = abs(regime_exits[regime_exits['pnl'] < 0]['pnl'].sum())
        return wins / losses if losses > 0 else float('inf')

    return BacktestResults(
        asset=asset,
        total_trades=total_trades,
        pnl_total=total_pnl,
        pnl_percent=total_pnl * 100,
        win_rate=win_rate,
        profit_factor=profit_factor,
        max_drawdown=max_drawdown,
        avg_return_per_trade=total_pnl / total_trades,
        time_in_trade_avg=0.0,  # Simplified

        # Regime breakdown
        risk_on_trades=len(risk_on_exits),
        risk_on_pf=regime_pf(risk_on_exits),
        risk_off_trades=len(risk_off_exits),
        risk_off_pf=regime_pf(risk_off_exits),
        neutral_trades=len(neutral_exits),
        neutral_pf=regime_pf(neutral_exits),

        # Macro impact
        trades_blocked_by_macro=blocked_by_macro,
        macro_boost_count=macro_boost_count,
        macro_veto_count=macro_veto_count,

        trade_records=trades
    )

def _empty_results(asset: str) -> BacktestResults:
    """Return empty results for failed backtests"""
    return BacktestResults(
        asset=asset, total_trades=0, pnl_total=0.0, pnl_percent=0.0,
        win_rate=0.0, profit_factor=0.0, max_drawdown=0.0,
        avg_return_per_trade=0.0, time_in_trade_avg=0.0,
        risk_on_trades=0, risk_on_pf=0.0, risk_off_trades=0, risk_off_pf=0.0,
        neutral_trades=0, neutral_pf=0.0, trades_blocked_by_macro=0,
        macro_boost_count=0, macro_veto_count=0, trade_records=[]
    )

def save_results(results: List[BacktestResults], output_dir: str, suffix: str = ''):
    """Save results to CSV files"""
    os.makedirs(output_dir, exist_ok=True)

    # Summary results
    summary_data = []
    for r in results:
        summary_data.append({
            'asset': r.asset,
            'total_trades': r.total_trades,
            'pnl_percent': r.pnl_percent,
            'win_rate': r.win_rate,
            'profit_factor': r.profit_factor,
            'max_drawdown': r.max_drawdown,
            'risk_on_pf': r.risk_on_pf,
            'risk_off_pf': r.risk_off_pf,
            'neutral_pf': r.neutral_pf,
            'trades_blocked': r.trades_blocked_by_macro,
            'macro_boosts': r.macro_boost_count,
            'macro_vetos': r.macro_veto_count
        })

    summary_df = pd.DataFrame(summary_data)
    summary_file = os.path.join(output_dir, f'summary{suffix}.csv')
    summary_df.to_csv(summary_file, index=False)
    logger.info(f"Saved summary to {summary_file}")

    # Individual trade records
    for r in results:
        if r.trade_records:
            trade_data = [t.__dict__ for t in r.trade_records]
            trade_df = pd.DataFrame(trade_data)
            trade_file = os.path.join(output_dir, f'{r.asset}_trades{suffix}.csv')
            trade_df.to_csv(trade_file, index=False)

def main():
    parser = argparse.ArgumentParser(description='Macro Pulse Backtest Analysis')
    parser.add_argument('--assets', nargs='+', default=['ETH', 'BTC', 'SOL'])
    parser.add_argument('--start', default='2023-01-01')
    parser.add_argument('--end', default='2025-01-01')
    parser.add_argument('--config', default='configs/v170/assets/ETH_v17_baseline.json')
    parser.add_argument('--output_dir', default='reports/macro_backtest')
    parser.add_argument('--ablation', default='none',
                       choices=['none', 'no_dxy', 'no_vix', 'no_oil_dxy',
                               'no_yields', 'no_usdjpy', 'no_hyg', 'no_usdt_sfp'])
    parser.add_argument('--disable_macro', action='store_true',
                       help='Run without macro pulse (v1.6.2 baseline)')

    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)

    # Extract configs
    engine_config = {
        'timeframe': '4H',
        'risk_pct': config['risk_management']['risk_pct'],
        'tp_R': 2.0,
        'sl_R': 1.0,
        'fusion': config['fusion'],
        'warmup': 250
    }

    context_config = config.get('domains', {}).get('macro_context', {})
    temporal_config = config.get('domains', {}).get('temporal', {})

    # Ablation mapping
    ablation_map = {
        'none': {},
        'no_dxy': {'dxy_veto': False},
        'no_vix': {'vix_move_veto': False},
        'no_oil_dxy': {'oil_dxy_veto': False},
        'no_yields': {'yields_veto': False},
        'no_usdjpy': {'usdjpy_veto': False},
        'no_hyg': {'hyg_veto': False},
        'no_usdt_sfp': {'usdt_sfp_veto': False}
    }

    ablation = ablation_map.get(args.ablation, {})
    enable_macro = not args.disable_macro

    logger.info(f"Running backtest: macro_enabled={enable_macro}, ablation={args.ablation}")

    # Run backtests
    results = []
    for asset in args.assets:
        try:
            result = simulate_asset_backtest(
                asset, args.start, args.end,
                engine_config, context_config, temporal_config,
                ablation, enable_macro
            )
            results.append(result)

            logger.info(f"{asset}: {result.total_trades} trades, "
                       f"{result.pnl_percent:.2f}% PnL, "
                       f"{result.win_rate:.1%} win rate, "
                       f"PF: {result.profit_factor:.2f}")

        except Exception as e:
            logger.error(f"Error backtesting {asset}: {e}")

    # Save results
    suffix = f"_{'no_macro' if args.disable_macro else 'macro'}"
    if args.ablation != 'none':
        suffix += f"_{args.ablation}"

    save_results(results, args.output_dir, suffix)

    # Print summary
    if results:
        total_trades = sum(r.total_trades for r in results)
        avg_pnl = np.mean([r.pnl_percent for r in results if r.total_trades > 0])
        avg_pf = np.mean([r.profit_factor for r in results if r.total_trades > 0 and r.profit_factor != float('inf')])

        print(f"\n=== BACKTEST SUMMARY ===")
        print(f"Assets: {', '.join(args.assets)}")
        print(f"Period: {args.start} to {args.end}")
        print(f"Macro enabled: {enable_macro}")
        print(f"Ablation: {args.ablation}")
        print(f"Total trades: {total_trades}")
        print(f"Average PnL: {avg_pnl:.2f}%")
        print(f"Average PF: {avg_pf:.2f}")

        if enable_macro:
            total_blocked = sum(r.trades_blocked_by_macro for r in results)
            total_boosts = sum(r.macro_boost_count for r in results)
            total_vetos = sum(r.macro_veto_count for r in results)
            print(f"Trades blocked by macro: {total_blocked}")
            print(f"Macro boosts: {total_boosts}")
            print(f"Macro vetos: {total_vetos}")

if __name__ == '__main__':
    main()