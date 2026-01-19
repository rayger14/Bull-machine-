"""
Baseline models for benchmarking trading strategies.

These are simple, transparent strategies that any "fancy" system must beat.
All baselines use only basic features (price, volume) and have sensible defaults.

Philosophy:
- If you can't beat buy-and-hold, why trade at all?
- If you can't beat a simple SMA crossover, your complexity isn't justified.
- Baselines provide an honesty check against over-optimization.

Available Baselines:
- Baseline0_BuyAndHold: Always long (tests if market has edge)
- Baseline1_SMA200Trend: Follow 200-period SMA (trend-following)
- Baseline2_SMACrossover: Golden cross/death cross (50/200 SMA)
- Baseline3_RSIMeanReversion: Buy dips, sell rips (RSI 30/70)
- Baseline4_VolTargetTrend: SMA trend with volatility-adjusted sizing
- Baseline5_Cash: Always hold cash (sanity check)
"""

from .buy_and_hold import Baseline0_BuyAndHold
from .sma_trend import Baseline1_SMA200Trend
from .sma_crossover import Baseline2_SMACrossover
from .rsi_mean_reversion import Baseline3_RSIMeanReversion
from .vol_target_trend import Baseline4_VolTargetTrend
from .cash import Baseline5_Cash

__all__ = [
    'Baseline0_BuyAndHold',
    'Baseline1_SMA200Trend',
    'Baseline2_SMACrossover',
    'Baseline3_RSIMeanReversion',
    'Baseline4_VolTargetTrend',
    'Baseline5_Cash',
]


def get_all_baselines():
    """
    Get list of all baseline model classes.

    Returns:
        List of baseline model classes

    Usage:
        baselines = get_all_baselines()
        for baseline_cls in baselines:
            model = baseline_cls()
            model.fit(train_data)
            # ... run backtest
    """
    return [
        Baseline0_BuyAndHold,
        Baseline1_SMA200Trend,
        Baseline2_SMACrossover,
        Baseline3_RSIMeanReversion,
        Baseline4_VolTargetTrend,
        Baseline5_Cash,
    ]
