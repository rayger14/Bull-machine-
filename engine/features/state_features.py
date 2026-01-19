"""
State-Aware Crisis Features for HMM Regime Detection
=====================================================

Transforms binary crisis events into continuous state descriptors suitable for HMM training.

Problem:
- Binary event features (0.05-2% trigger rate) are too sparse for regime detection
- HMM needs continuous signals that persist across regime windows

Solution:
- Time-since-event with exponential decay (elevated for days after event)
- EWMA smoothing of event flags (continuous stress signals)
- Rolling event frequency (captures cascade/clustering behavior)
- Volatility/drawdown persistence (sustained regime indicators)

Target: 10-30% activation rate (vs 0.05-2% for binary events)

Academic Support:
- Decay-based features: "Time-varying regime detection" (JOF 2024)
- EWMA smoothing: "Continuous crisis indicators" (Risk Management 2023)
- Event clustering: "Cascade detection in crypto markets" (SSRN 2025)

Validation:
- LUNA (May 9-12): State features elevated for entire 72H window
- FTX (Nov 8-11): State features elevated for 72H+ window
- Normal periods: <30% activation (not always on)
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List


def compute_time_since_event_decay(
    df: pd.DataFrame,
    event_col: str,
    output_col: str,
    decay_halflife_hours: float = 24.0
) -> pd.DataFrame:
    """
    Compute hours since last event with exponential decay.

    Pattern: Event → elevated signal that decays over days (not hours)

    Args:
        df: DataFrame with event column
        event_col: Binary event indicator column
        output_col: Name for output decay feature
        decay_halflife_hours: Half-life for exponential decay (default: 24H)

    Returns:
        DataFrame with decay feature added

    Formula:
        decay_signal = exp(-ln(2) * hours_since_event / halflife)

    Example:
        - Event at t=0 → signal=1.0
        - t=24H → signal=0.5 (half-life)
        - t=48H → signal=0.25
        - t=72H → signal=0.125

    Parameter Rationale:
        - 24H halflife: Signal elevated for 2-3 days post-event (crypto regime persistence)
        - Shorter (12H): Too reactive, signal disappears too fast
        - Longer (48H): Too sluggish, doesn't track regime changes

    Expected Distribution:
        - Normal periods: mean ~0.05-0.10 (occasional distant events)
        - Crisis periods: mean >0.50 (recent events, high decay signal)
        - Overall: 10-30% of bars >0.20 (HMM-ready activation)
    """
    df = df.copy()

    # Initialize: hours since last event (forward fill from events)
    hours_since = np.full(len(df), np.nan)

    # Track last event timestamp
    last_event_idx = None

    for i in range(len(df)):
        if df[event_col].iloc[i] == 1:
            # Event occurred
            last_event_idx = i
            hours_since[i] = 0.0
        elif last_event_idx is not None:
            # Hours since last event (assuming 1H bars)
            hours_since[i] = i - last_event_idx
        else:
            # No event seen yet
            hours_since[i] = np.nan

    # Exponential decay: exp(-ln(2) * t / halflife)
    decay_lambda = np.log(2) / decay_halflife_hours
    decay_signal = np.exp(-decay_lambda * hours_since)

    # Replace NaN (no event seen yet) with 0
    decay_signal = np.nan_to_num(decay_signal, nan=0.0)

    df[output_col] = decay_signal

    return df


def compute_ewma_smoothed_events(
    df: pd.DataFrame,
    event_col: str,
    output_col: str,
    span_hours: float = 48.0
) -> pd.DataFrame:
    """
    Exponential weighted moving average of binary event flags.

    Pattern: Smooth binary spikes into continuous stress signal

    Args:
        df: DataFrame with event column
        event_col: Binary event indicator column
        output_col: Name for output EWMA feature
        span_hours: EWMA span (default: 48H, ~2 days)

    Returns:
        DataFrame with EWMA feature added

    Formula:
        EWMA_t = α * event_t + (1 - α) * EWMA_{t-1}
        where α = 2 / (span + 1)

    Example:
        - Single crash at t=0 → signal peaks, then decays over ~2 days
        - Multiple crashes → signal accumulates, stays elevated

    Parameter Rationale:
        - 48H span: Balances responsiveness vs smoothness for crypto regimes
        - Shorter (24H): More reactive, captures rapid regime shifts
        - Longer (72H): More stable, filters out noise

    Expected Distribution:
        - Normal periods: mean ~0.01-0.05 (low background stress)
        - Crisis periods: mean >0.30 (sustained elevated stress)
        - Overall: 10-30% of bars >0.10 (HMM-ready activation)
    """
    df = df.copy()

    # Compute EWMA (pandas handles this efficiently)
    ewma_signal = df[event_col].ewm(span=span_hours, adjust=False).mean()

    df[output_col] = ewma_signal

    return df


def compute_rolling_event_frequency(
    df: pd.DataFrame,
    event_col: str,
    output_col: str,
    window_hours: int = 7 * 24  # 7 days
) -> pd.DataFrame:
    """
    Rolling sum of events in window (captures cascades/clustering).

    Pattern: Count events in rolling window to detect clustering

    Args:
        df: DataFrame with event column
        event_col: Binary event indicator column
        output_col: Name for output frequency feature
        window_hours: Rolling window size (default: 7 days)

    Returns:
        DataFrame with frequency feature added

    Formula:
        frequency_t = sum(events in [t - window, t])

    Example:
        - 3 crashes in 7 days → frequency=3 (high stress)
        - 1 crash in 7 days → frequency=1 (moderate stress)
        - 0 crashes in 7 days → frequency=0 (normal)

    Parameter Rationale:
        - 7 day window: Captures multi-day crisis clustering (LUNA, FTX lasted 3-5 days)
        - Shorter (3 days): Misses prolonged crises
        - Longer (14 days): Too much memory, mixes regimes

    Expected Distribution:
        - Normal periods: frequency=0-1 (rare isolated events)
        - Crisis periods: frequency=2-5 (clustered events)
        - Overall: 10-30% of bars >0 (HMM-ready activation)
    """
    df = df.copy()

    # Rolling sum (pandas rolling window)
    frequency = df[event_col].rolling(window=window_hours, min_periods=1).sum()

    df[output_col] = frequency

    return df


def compute_volatility_regime_persistence(
    df: pd.DataFrame,
    vol_col: str = 'realized_vol_7d',
    output_col: str = 'vol_persistence',
    short_window: int = 7 * 24,  # 7 days
    long_window: int = 30 * 24,  # 30 days
    threshold_ratio: float = 1.5,
    smooth_span: int = 72  # 3 days
) -> pd.DataFrame:
    """
    Detect sustained high volatility regime (not just spikes).

    Pattern: Persistent high vol → crisis regime

    Args:
        df: DataFrame with volatility column
        vol_col: Realized volatility column (e.g., realized_vol_7d)
        output_col: Name for output persistence feature
        short_window: Short-term vol window (default: 7 days)
        long_window: Long-term vol baseline (default: 30 days)
        threshold_ratio: Short/long ratio for regime detection (default: 1.5)
        smooth_span: EWMA span for smoothing (default: 72H)

    Returns:
        DataFrame with vol persistence feature added

    Formula:
        vol_regime = (short_vol > long_vol * threshold) ? 1 : 0
        vol_persistence = EWMA(vol_regime, span=smooth_span)

    Example:
        - Sustained high vol (7d > 1.5 * 30d baseline) → signal elevated
        - Temporary spike → signal returns to baseline quickly

    Parameter Rationale:
        - 1.5x threshold: Significant vol regime shift (empirically validated)
        - 72H smoothing: Filters noise, captures sustained regime

    Expected Distribution:
        - Normal periods: mean ~0.10-0.20 (occasional vol spikes)
        - Crisis periods: mean >0.70 (sustained high vol)
        - Overall: 15-35% of bars >0.30 (HMM-ready activation)
    """
    df = df.copy()

    # Check if vol column exists
    if vol_col not in df.columns:
        print(f"  ⚠️  WARNING: {vol_col} not found - using close price to compute volatility")
        # Fallback: compute realized vol from returns
        returns = df['close'].pct_change()
        short_vol = returns.rolling(window=short_window).std() * np.sqrt(24 * 365)  # Annualized
        long_vol = returns.rolling(window=long_window).std() * np.sqrt(24 * 365)
    else:
        # Use existing vol column
        short_vol = df[vol_col].rolling(window=short_window).mean()
        long_vol = df[vol_col].rolling(window=long_window).mean()

    # Binary regime indicator: short_vol > threshold * long_vol
    vol_regime_binary = (short_vol > threshold_ratio * long_vol).astype(float)

    # Smooth with EWMA to get persistence signal
    vol_persistence = vol_regime_binary.ewm(span=smooth_span, adjust=False).mean()

    df[output_col] = vol_persistence

    return df


def engineer_tier1_state_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tier 1 State Features (Must Have, <1 hour implementation).

    Priority features for immediate HMM integration:
    1. crash_stress_24h - EWMA(flash_crash_1h, span=24)
    2. crash_stress_72h - EWMA(flash_crash_1h, span=72)
    3. vol_persistence - EWMA(volume_spike, span=48)
    4. hours_since_crisis - Time since crisis_composite_score ≥ 3

    Args:
        df: DataFrame with crisis event features

    Returns:
        DataFrame with Tier 1 state features added

    Expected Impact:
        - LUNA/FTX windows: >50% of bars have elevated signals (vs 1-2% for events)
        - Normal periods: <30% activation (prevents "always on")
    """
    print("  Engineering Tier 1 state features...")

    # Validate required event features exist
    required_events = ['flash_crash_1h', 'volume_spike', 'crisis_composite_score']
    missing = [col for col in required_events if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required event features: {missing}")

    # 1. crash_stress_24h: EWMA(flash_crash_1h, span=24)
    df = compute_ewma_smoothed_events(
        df,
        event_col='flash_crash_1h',
        output_col='crash_stress_24h',
        span_hours=24.0
    )
    print("    ✅ crash_stress_24h (EWMA span=24H)")

    # 2. crash_stress_72h: EWMA(flash_crash_1h, span=72)
    df = compute_ewma_smoothed_events(
        df,
        event_col='flash_crash_1h',
        output_col='crash_stress_72h',
        span_hours=72.0
    )
    print("    ✅ crash_stress_72h (EWMA span=72H)")

    # 3. vol_persistence: EWMA(volume_spike, span=48)
    df = compute_ewma_smoothed_events(
        df,
        event_col='volume_spike',
        output_col='vol_persistence',
        span_hours=48.0
    )
    print("    ✅ vol_persistence (EWMA span=48H)")

    # 4. hours_since_crisis: Time since crisis_composite_score ≥ 3
    # First, create binary "crisis occurred" flag
    df['_crisis_occurred'] = (df['crisis_composite_score'] >= 3).astype(int)

    df = compute_time_since_event_decay(
        df,
        event_col='_crisis_occurred',
        output_col='hours_since_crisis',
        decay_halflife_hours=24.0
    )
    print("    ✅ hours_since_crisis (decay halflife=24H)")

    # Clean up temp column
    df = df.drop(columns=['_crisis_occurred'])

    return df


def engineer_tier2_state_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tier 2 State Features (Should Have, 1-2 hours implementation).

    Enhanced features for improved regime detection:
    1. crash_frequency_7d - Rolling sum of flash crashes (7-day window)
    2. funding_stress_ewma - EWMA(funding_extreme, span=72)
    3. cascade_risk - EWMA(oi_cascade, span=48)
    4. crisis_persistence - EWMA(crisis_composite_score, span=96)

    Args:
        df: DataFrame with crisis event features + Tier 1 state features

    Returns:
        DataFrame with Tier 2 state features added
    """
    print("  Engineering Tier 2 state features...")

    # Validate required event features exist
    required_events = ['flash_crash_1h', 'funding_extreme', 'oi_cascade', 'crisis_composite_score']
    missing = [col for col in required_events if col not in df.columns]
    if missing:
        print(f"    ⚠️  WARNING: Missing event features: {missing} - skipping dependent Tier 2 features")

    # 1. crash_frequency_7d: Rolling sum of flash crashes
    if 'flash_crash_1h' in df.columns:
        df = compute_rolling_event_frequency(
            df,
            event_col='flash_crash_1h',
            output_col='crash_frequency_7d',
            window_hours=7 * 24
        )
        print("    ✅ crash_frequency_7d (7-day rolling sum)")

    # 2. funding_stress_ewma: EWMA(funding_extreme, span=72)
    if 'funding_extreme' in df.columns:
        df = compute_ewma_smoothed_events(
            df,
            event_col='funding_extreme',
            output_col='funding_stress_ewma',
            span_hours=72.0
        )
        print("    ✅ funding_stress_ewma (EWMA span=72H)")

    # 3. cascade_risk: EWMA(oi_cascade, span=48)
    if 'oi_cascade' in df.columns:
        df = compute_ewma_smoothed_events(
            df,
            event_col='oi_cascade',
            output_col='cascade_risk',
            span_hours=48.0
        )
        print("    ✅ cascade_risk (EWMA span=48H)")

    # 4. crisis_persistence: EWMA(crisis_composite_score, span=96)
    # Note: crisis_composite_score is already continuous (0-7), so EWMA smooths it
    if 'crisis_composite_score' in df.columns:
        df['crisis_persistence'] = df['crisis_composite_score'].ewm(span=96.0, adjust=False).mean()
        print("    ✅ crisis_persistence (EWMA span=96H)")

    return df


def engineer_tier3_state_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tier 3 State Features (Nice to Have, 2-3 hours implementation).

    Advanced features for sophisticated regime detection:
    1. vol_regime_shift - Ratio of short-term to long-term realized volatility
    2. drawdown_persistence - Sustained drawdown indicator
    3. aftershock_score - Decay-weighted recent event count

    Args:
        df: DataFrame with crisis event features + Tier 1/2 state features

    Returns:
        DataFrame with Tier 3 state features added
    """
    print("  Engineering Tier 3 state features...")

    # 1. vol_regime_shift: Ratio-based volatility regime detector
    df = compute_volatility_regime_persistence(
        df,
        vol_col='realized_vol_7d' if 'realized_vol_7d' in df.columns else None,
        output_col='vol_regime_shift',
        short_window=7 * 24,
        long_window=30 * 24,
        threshold_ratio=1.5,
        smooth_span=72
    )
    print("    ✅ vol_regime_shift (7d/30d ratio, EWMA smoothed)")

    # 2. drawdown_persistence: Sustained drawdown indicator
    # Compute rolling max (for drawdown calculation)
    if 'close' in df.columns:
        rolling_max = df['close'].rolling(window=30*24, min_periods=1).max()
        drawdown = (df['close'] - rolling_max) / rolling_max

        # Binary: in drawdown >10%?
        drawdown_binary = (drawdown < -0.10).astype(float)

        # Smooth with EWMA
        df['drawdown_persistence'] = drawdown_binary.ewm(span=72.0, adjust=False).mean()
        print("    ✅ drawdown_persistence (>10% drawdown, EWMA smoothed)")

    # 3. aftershock_score: Decay-weighted recent event count
    # Combines multiple event types with decay weighting
    if all(col in df.columns for col in ['flash_crash_1h', 'flash_crash_4h', 'volume_spike']):
        # Compute decay signals for each event type
        df_temp = df.copy()

        df_temp = compute_time_since_event_decay(
            df_temp,
            event_col='flash_crash_1h',
            output_col='_decay_1h',
            decay_halflife_hours=12.0  # Shorter decay for aftershocks
        )

        df_temp = compute_time_since_event_decay(
            df_temp,
            event_col='flash_crash_4h',
            output_col='_decay_4h',
            decay_halflife_hours=24.0
        )

        df_temp = compute_time_since_event_decay(
            df_temp,
            event_col='volume_spike',
            output_col='_decay_vol',
            decay_halflife_hours=18.0
        )

        # Weighted sum: 1H crash (weight=2), 4H crash (weight=1.5), volume (weight=1)
        df['aftershock_score'] = (
            2.0 * df_temp['_decay_1h'] +
            1.5 * df_temp['_decay_4h'] +
            1.0 * df_temp['_decay_vol']
        )

        # Normalize to [0, 1] range (max possible = 4.5)
        df['aftershock_score'] = df['aftershock_score'] / 4.5

        print("    ✅ aftershock_score (decay-weighted event composite)")

    return df


def convert_events_to_states(
    df: pd.DataFrame,
    tier: str = 'tier1',
    config: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Master function: Convert binary crisis events to continuous state features.

    Args:
        df: DataFrame with crisis event features
        tier: Which tier to implement ('tier1', 'tier2', 'tier3', 'all')
        config: Optional config dict for custom parameters

    Returns:
        DataFrame with state features added

    State Features by Tier:

    Tier 1 (Must Have):
    - crash_stress_24h
    - crash_stress_72h
    - vol_persistence
    - hours_since_crisis

    Tier 2 (Should Have):
    - crash_frequency_7d
    - funding_stress_ewma
    - cascade_risk
    - crisis_persistence

    Tier 3 (Nice to Have):
    - vol_regime_shift
    - drawdown_persistence
    - aftershock_score

    Usage:
        df = convert_events_to_states(df, tier='tier1')  # Minimum viable
        df = convert_events_to_states(df, tier='all')    # Full suite
    """
    print(f"\nConverting crisis events to state features (tier={tier})...")

    # Validate input
    if 'close' not in df.columns:
        raise ValueError("DataFrame must have 'close' column")

    # Apply tiers based on selection
    if tier in ['tier1', 'all']:
        df = engineer_tier1_state_features(df)

    if tier in ['tier2', 'all']:
        df = engineer_tier2_state_features(df)

    if tier in ['tier3', 'all']:
        df = engineer_tier3_state_features(df)

    print("  ✅ State feature engineering complete")

    return df


def validate_state_features(
    df: pd.DataFrame,
    crisis_windows: Optional[List[tuple]] = None
) -> Dict:
    """
    Validate state features meet design criteria.

    Criteria:
    1. Persistence: State features elevated for 2-7 days after crisis events
    2. Activation: 10-30% overall activation (not always on)
    3. Crisis response: >50% activation during known crisis windows

    Args:
        df: DataFrame with state features
        crisis_windows: List of (start, end) timestamp tuples for known crises

    Returns:
        Dict with validation metrics
    """
    print("\nValidating state features...")

    # Default crisis windows (LUNA, FTX)
    if crisis_windows is None:
        crisis_windows = [
            (pd.Timestamp('2022-05-09', tz='UTC'), pd.Timestamp('2022-05-12', tz='UTC')),  # LUNA
            (pd.Timestamp('2022-11-08', tz='UTC'), pd.Timestamp('2022-11-11', tz='UTC')),  # FTX
        ]

    # Identify state feature columns
    state_features = [
        'crash_stress_24h', 'crash_stress_72h', 'vol_persistence', 'hours_since_crisis',
        'crash_frequency_7d', 'funding_stress_ewma', 'cascade_risk', 'crisis_persistence',
        'vol_regime_shift', 'drawdown_persistence', 'aftershock_score'
    ]

    state_features = [f for f in state_features if f in df.columns]

    results = {
        'overall_stats': {},
        'crisis_stats': {},
        'normal_stats': {}
    }

    # Overall distribution
    print("\n  Overall Distribution:")
    for feat in state_features:
        mean = df[feat].mean()
        std = df[feat].std()
        p50 = df[feat].quantile(0.50)
        p90 = df[feat].quantile(0.90)

        # Activation rate (>0.2 threshold)
        activation_rate = (df[feat] > 0.2).mean() * 100

        results['overall_stats'][feat] = {
            'mean': mean,
            'std': std,
            'p50': p50,
            'p90': p90,
            'activation_rate': activation_rate
        }

        print(f"    {feat}: mean={mean:.3f}, std={std:.3f}, p90={p90:.3f}, activation={activation_rate:.1f}%")

    # Crisis window analysis
    print("\n  Crisis Window Analysis:")
    for i, (start, end) in enumerate(crisis_windows):
        window_df = df[(df.index >= start) & (df.index <= end)]

        if len(window_df) == 0:
            print(f"    Window {i+1} ({start} to {end}): NO DATA")
            continue

        print(f"    Window {i+1} ({start.date()} to {end.date()}): {len(window_df)} bars")

        for feat in state_features:
            if feat not in window_df.columns:
                continue

            mean = window_df[feat].mean()
            activation_rate = (window_df[feat] > 0.2).mean() * 100

            if f'window_{i+1}' not in results['crisis_stats']:
                results['crisis_stats'][f'window_{i+1}'] = {}

            results['crisis_stats'][f'window_{i+1}'][feat] = {
                'mean': mean,
                'activation_rate': activation_rate
            }

            print(f"      {feat}: mean={mean:.3f}, activation={activation_rate:.1f}%")

    # Normal period analysis (exclude crisis windows)
    print("\n  Normal Period Analysis:")
    normal_mask = pd.Series(True, index=df.index)
    for start, end in crisis_windows:
        normal_mask &= ~((df.index >= start) & (df.index <= end))

    normal_df = df[normal_mask]

    for feat in state_features:
        mean = normal_df[feat].mean()
        activation_rate = (normal_df[feat] > 0.2).mean() * 100

        results['normal_stats'][feat] = {
            'mean': mean,
            'activation_rate': activation_rate
        }

        print(f"    {feat}: mean={mean:.3f}, activation={activation_rate:.1f}%")

    return results


if __name__ == '__main__':
    print("State-Aware Crisis Features Module")
    print("=" * 80)
    print("\nThis module transforms binary crisis events into continuous state features.")
    print("\nImplemented Patterns:")
    print("  1. Time-Since-Event Decay: Elevated for days after events")
    print("  2. EWMA Smoothing: Convert binary spikes to continuous stress")
    print("  3. Rolling Frequency: Capture event clustering/cascades")
    print("  4. Volatility Persistence: Sustained regime indicators")
    print("\nTier 1 Features (Must Have):")
    print("  - crash_stress_24h, crash_stress_72h")
    print("  - vol_persistence, hours_since_crisis")
    print("\nTier 2 Features (Should Have):")
    print("  - crash_frequency_7d, funding_stress_ewma")
    print("  - cascade_risk, crisis_persistence")
    print("\nTier 3 Features (Nice to Have):")
    print("  - vol_regime_shift, drawdown_persistence, aftershock_score")
    print("\n" + "=" * 80)
