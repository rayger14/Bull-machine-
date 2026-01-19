#!/usr/bin/env python3
"""
Regime-Stratified Backtest Engine

Core backtest engine that filters historical bars by regime before backtesting.
This ensures archetypes are only evaluated on allowed regime bars, eliminating
cross-regime contamination that destroys edge.

Architecture:
- Filters data to allowed regimes BEFORE backtest execution
- Computes all metrics ONLY on regime-filtered bars
- Provides regime distribution statistics
- Integrates with existing backtest infrastructure

Example Usage:
    # Test S1 only on crisis + risk_off bars
    results = backtest_regime_stratified(
        archetype='liquidity_vacuum',
        data=historical_df,
        config=s1_config,
        allowed_regimes=['crisis', 'risk_off']
    )

Author: Claude Code (Backend Architect)
Date: 2025-11-25
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class RegimeStratifiedResult:
    """Results from regime-stratified backtest"""
    archetype: str
    allowed_regimes: List[str]

    # Regime distribution
    total_bars: int
    regime_bars: int
    regime_pct: float

    # Trade metrics
    total_trades: int
    trades_per_year: float
    win_rate: float
    profit_factor: float
    total_r: float
    avg_r: float

    # Risk metrics
    max_dd_r: float
    sharpe_ratio: float
    calmar_ratio: float

    # Event recall (for crisis archetypes)
    events_detected: int
    events_total: int
    event_recall: float

    # OOS validation
    train_pf: Optional[float] = None
    test_pf: Optional[float] = None
    oos_consistency: Optional[float] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'archetype': self.archetype,
            'allowed_regimes': self.allowed_regimes,
            'total_bars': self.total_bars,
            'regime_bars': self.regime_bars,
            'regime_pct': round(self.regime_pct, 3),
            'total_trades': self.total_trades,
            'trades_per_year': round(self.trades_per_year, 2),
            'win_rate': round(self.win_rate, 2),
            'profit_factor': round(self.profit_factor, 3),
            'total_r': round(self.total_r, 2),
            'avg_r': round(self.avg_r, 3),
            'max_dd_r': round(self.max_dd_r, 3),
            'sharpe_ratio': round(self.sharpe_ratio, 3),
            'calmar_ratio': round(self.calmar_ratio, 3),
            'events_detected': self.events_detected,
            'events_total': self.events_total,
            'event_recall': round(self.event_recall, 2),
            'train_pf': round(self.train_pf, 3) if self.train_pf else None,
            'test_pf': round(self.test_pf, 3) if self.test_pf else None,
            'oos_consistency': round(self.oos_consistency, 3) if self.oos_consistency else None
        }

    def meets_criteria(self, min_pf: float = 2.0, min_wr: float = 50.0) -> bool:
        """Check if results meet optimization criteria"""
        return self.profit_factor >= min_pf and self.win_rate >= min_wr


def backtest_regime_stratified(
    archetype: str,
    data: pd.DataFrame,
    config: Dict,
    allowed_regimes: List[str],
    min_bars: int = 500,
    ground_truth_events: Optional[List[str]] = None
) -> RegimeStratifiedResult:
    """
    Backtest archetype only on allowed regime bars.

    Args:
        archetype: Archetype name (liquidity_vacuum, funding_divergence, etc.)
        data: Historical dataframe with regime_label column
        config: Backtest configuration
        allowed_regimes: List of allowed regime labels (e.g., ['risk_off', 'crisis'])
        min_bars: Minimum number of regime bars required
        ground_truth_events: Optional list of ground truth event dates for recall

    Returns:
        RegimeStratifiedResult with metrics computed ONLY on regime-filtered bars

    Raises:
        ValueError: If insufficient regime bars for backtest
    """
    # Validate input
    if 'regime_label' not in data.columns:
        raise ValueError("Data must contain 'regime_label' column. Run regime classifier first.")

    # Filter to allowed regimes
    regime_mask = data['regime_label'].isin(allowed_regimes)
    regime_data = data[regime_mask].copy()

    total_bars = len(data)
    regime_bars = len(regime_data)
    regime_pct = regime_bars / total_bars if total_bars > 0 else 0.0

    logger.info(f"Backtesting {archetype} on regimes: {allowed_regimes}")
    logger.info(f"  Total bars: {total_bars:,}")
    logger.info(f"  Regime bars: {regime_bars:,} ({regime_pct*100:.1f}%)")

    # Check minimum bars
    if regime_bars < min_bars:
        raise ValueError(
            f"Insufficient regime bars for {archetype}: {regime_bars} < {min_bars}. "
            f"Adjust allowed_regimes or date range."
        )

    # Execute backtest on regime-filtered data
    trades = _execute_backtest(archetype, regime_data, config)

    # Compute metrics
    metrics = _compute_metrics(trades, regime_data)

    # Compute event recall if ground truth provided
    event_recall_metrics = _compute_event_recall(
        trades,
        ground_truth_events or []
    )

    # Build result
    result = RegimeStratifiedResult(
        archetype=archetype,
        allowed_regimes=allowed_regimes,
        total_bars=total_bars,
        regime_bars=regime_bars,
        regime_pct=regime_pct,
        **metrics,
        **event_recall_metrics
    )

    logger.info(f"Backtest complete: PF={result.profit_factor:.2f}, WR={result.win_rate:.1f}%, "
                f"Trades={result.total_trades}, Event Recall={result.event_recall:.1f}%")

    return result


def _execute_backtest(
    archetype: str,
    data: pd.DataFrame,
    config: Dict
) -> pd.DataFrame:
    """
    Execute backtest logic on regime-filtered data.

    Args:
        archetype: Archetype name
        data: Regime-filtered historical data
        config: Backtest configuration

    Returns:
        DataFrame of trades with columns: [timestamp, entry_price, exit_price, r_multiple, win]
    """
    # Import backtest logic based on archetype
    if archetype == 'liquidity_vacuum':
        from engine.strategies.archetypes.bear.liquidity_vacuum_runtime import (
            apply_liquidity_vacuum_enrichment,
            evaluate_liquidity_vacuum_signal
        )

        # Apply runtime enrichment
        data = apply_liquidity_vacuum_enrichment(data, lookback=24, volume_lookback=24)

        trades = []
        last_trade_idx = -999
        cooldown = config.get('cooldown_bars', 12)
        atr_stop_mult = config.get('atr_stop_mult', 2.5)

        for i, (idx, row) in enumerate(data.iterrows()):
            # Cooldown check
            if i < last_trade_idx + cooldown:
                continue

            # Evaluate signal
            signal_strength = evaluate_liquidity_vacuum_signal(row, config)

            if signal_strength > 0:
                # Simulate trade execution
                entry_price = row.get('close', 0)
                atr = row.get('atr_20', row.get('atr_14', 0))
                stop_loss = entry_price - (atr * atr_stop_mult)
                target = entry_price + (atr * atr_stop_mult * 2.0)  # 2:1 RR

                # Simple forward-looking exit (look ahead 24 bars max)
                exit_idx = min(i + 24, len(data) - 1)
                exit_bar = data.iloc[exit_idx]

                # Check if stop hit or target hit
                low_after_entry = data.iloc[i:exit_idx+1]['low'].min()
                high_after_entry = data.iloc[i:exit_idx+1]['high'].max()

                if low_after_entry <= stop_loss:
                    # Stop hit
                    exit_price = stop_loss
                    r_multiple = -1.0
                    win = False
                elif high_after_entry >= target:
                    # Target hit
                    exit_price = target
                    r_multiple = 2.0
                    win = True
                else:
                    # Exit at end of window
                    exit_price = exit_bar.get('close', entry_price)
                    r_multiple = (exit_price - entry_price) / (entry_price - stop_loss)
                    win = r_multiple > 0

                trades.append({
                    'timestamp': idx,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'r_multiple': r_multiple,
                    'win': win,
                    'signal_strength': signal_strength
                })

                last_trade_idx = i

    elif archetype == 'funding_divergence':
        from engine.strategies.archetypes.bear.funding_divergence_runtime import (
            apply_s4_enrichment,
            evaluate_s4_signal
        )

        # Apply runtime enrichment
        data = apply_s4_enrichment(data, funding_lookback=24, price_lookback=12)

        trades = []
        last_trade_idx = -999
        cooldown = config.get('cooldown_bars', 12)
        atr_stop_mult = config.get('atr_stop_mult', 2.5)

        for i, (idx, row) in enumerate(data.iterrows()):
            # Cooldown check
            if i < last_trade_idx + cooldown:
                continue

            # Evaluate signal
            signal_strength = evaluate_s4_signal(row, config)

            if signal_strength > 0:
                # Similar trade simulation as S1
                entry_price = row.get('close', 0)
                atr = row.get('atr_20', row.get('atr_14', 0))
                stop_loss = entry_price - (atr * atr_stop_mult)
                target = entry_price + (atr * atr_stop_mult * 2.0)

                exit_idx = min(i + 24, len(data) - 1)
                exit_bar = data.iloc[exit_idx]

                low_after_entry = data.iloc[i:exit_idx+1]['low'].min()
                high_after_entry = data.iloc[i:exit_idx+1]['high'].max()

                if low_after_entry <= stop_loss:
                    exit_price = stop_loss
                    r_multiple = -1.0
                    win = False
                elif high_after_entry >= target:
                    exit_price = target
                    r_multiple = 2.0
                    win = True
                else:
                    exit_price = exit_bar.get('close', entry_price)
                    r_multiple = (exit_price - entry_price) / (entry_price - stop_loss)
                    win = r_multiple > 0

                trades.append({
                    'timestamp': idx,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'r_multiple': r_multiple,
                    'win': win,
                    'signal_strength': signal_strength
                })

                last_trade_idx = i

    else:
        # Generic archetype evaluation
        # This would integrate with logic_v2_adapter.py for other archetypes
        logger.warning(f"Generic backtest for {archetype} - implement specific logic for better accuracy")
        trades = []

    return pd.DataFrame(trades)


def _compute_metrics(trades: pd.DataFrame, data: pd.DataFrame) -> Dict:
    """
    Compute backtest metrics from trades.

    Args:
        trades: DataFrame of trades
        data: Historical data for time range calculation

    Returns:
        Dict with computed metrics
    """
    if len(trades) == 0:
        return {
            'total_trades': 0,
            'trades_per_year': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'total_r': 0.0,
            'avg_r': 0.0,
            'max_dd_r': 0.0,
            'sharpe_ratio': 0.0,
            'calmar_ratio': 0.0
        }

    # Basic metrics
    total_trades = len(trades)
    wins = trades['win'].sum()
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0

    # R-multiple metrics
    winning_r = trades[trades['win']]['r_multiple'].sum()
    losing_r = abs(trades[~trades['win']]['r_multiple'].sum())
    profit_factor = (winning_r / losing_r) if losing_r > 0 else 0.0
    total_r = trades['r_multiple'].sum()
    avg_r = trades['r_multiple'].mean()

    # Annualized metrics
    years = (data.index[-1] - data.index[0]).days / 365.25
    trades_per_year = total_trades / years if years > 0 else 0.0

    # Drawdown
    cumulative_r = trades['r_multiple'].cumsum()
    running_max = cumulative_r.expanding().max()
    drawdown = cumulative_r - running_max
    max_dd_r = drawdown.min() if len(drawdown) > 0 else 0.0

    # Sharpe ratio (assumes trades are independent)
    r_std = trades['r_multiple'].std()
    sharpe_ratio = (avg_r / r_std * np.sqrt(trades_per_year)) if r_std > 0 else 0.0

    # Calmar ratio
    calmar_ratio = (total_r / abs(max_dd_r)) if max_dd_r != 0 else 0.0

    return {
        'total_trades': total_trades,
        'trades_per_year': trades_per_year,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'total_r': total_r,
        'avg_r': avg_r,
        'max_dd_r': max_dd_r,
        'sharpe_ratio': sharpe_ratio,
        'calmar_ratio': calmar_ratio
    }


def _compute_event_recall(
    trades: pd.DataFrame,
    ground_truth_events: List[str]
) -> Dict:
    """
    Compute event recall for crisis archetypes.

    Args:
        trades: DataFrame of trades
        ground_truth_events: List of ground truth event dates (ISO format)

    Returns:
        Dict with event recall metrics
    """
    if not ground_truth_events or len(trades) == 0:
        return {
            'events_detected': 0,
            'events_total': len(ground_truth_events),
            'event_recall': 0.0
        }

    # Convert ground truth to datetime
    gt_dates = [pd.to_datetime(date) for date in ground_truth_events]

    # Check which events were detected (trade within ±48h of event)
    events_detected = 0
    for event_date in gt_dates:
        # Check if any trade occurred within ±48h
        for trade_ts in trades['timestamp']:
            time_diff = abs((trade_ts - event_date).total_seconds() / 3600)
            if time_diff <= 48:
                events_detected += 1
                break

    event_recall = (events_detected / len(ground_truth_events) * 100) if ground_truth_events else 0.0

    return {
        'events_detected': events_detected,
        'events_total': len(ground_truth_events),
        'event_recall': event_recall
    }


def get_regime_distribution(data: pd.DataFrame) -> Dict[str, float]:
    """
    Get regime distribution from historical data.

    Args:
        data: Historical dataframe with regime_label column

    Returns:
        Dict mapping regime -> percentage
    """
    if 'regime_label' not in data.columns:
        return {}

    regime_counts = data['regime_label'].value_counts()
    total = len(data)

    distribution = {
        regime: count / total
        for regime, count in regime_counts.items()
    }

    return distribution


def validate_regime_coverage(
    data: pd.DataFrame,
    required_regimes: List[str],
    min_bars_per_regime: int = 500
) -> Tuple[bool, str]:
    """
    Validate that data has sufficient coverage of required regimes.

    Args:
        data: Historical dataframe with regime_label column
        required_regimes: List of required regimes
        min_bars_per_regime: Minimum bars needed per regime

    Returns:
        (is_valid, message)
    """
    if 'regime_label' not in data.columns:
        return False, "Data missing regime_label column"

    regime_counts = data['regime_label'].value_counts()

    insufficient = []
    for regime in required_regimes:
        count = regime_counts.get(regime, 0)
        if count < min_bars_per_regime:
            insufficient.append(f"{regime}: {count} bars (need {min_bars_per_regime})")

    if insufficient:
        return False, f"Insufficient regime coverage: {', '.join(insufficient)}"

    return True, "Regime coverage validated"


if __name__ == "__main__":
    # Example usage
    import json

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Load feature data
    feature_file = Path('data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet')
    if not feature_file.exists():
        logger.error(f"Feature file not found: {feature_file}")
        sys.exit(1)

    logger.info("Loading feature data...")
    df = pd.read_parquet(feature_file)
    df_2022 = df[(df.index >= '2022-01-01') & (df.index < '2023-01-01')].copy()

    # Example: Test S1 on crisis + risk_off bars
    s1_config = {
        'fusion_threshold': 0.45,
        'liquidity_max': 0.15,
        'volume_z_min': 2.0,
        'wick_lower_min': 0.30,
        'cooldown_bars': 12,
        'atr_stop_mult': 2.5
    }

    ground_truth_events = [
        '2022-05-12',  # LUNA
        '2022-06-18',  # Capitulation
        '2022-11-09'   # FTX
    ]

    try:
        results = backtest_regime_stratified(
            archetype='liquidity_vacuum',
            data=df_2022,
            config=s1_config,
            allowed_regimes=['crisis', 'risk_off'],
            ground_truth_events=ground_truth_events
        )

        print("\nRegime-Stratified Backtest Results:")
        print(json.dumps(results.to_dict(), indent=2))

    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        sys.exit(1)
