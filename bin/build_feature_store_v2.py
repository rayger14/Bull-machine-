#!/usr/bin/env python3
"""
Feature Store v2.0 Builder - Knowledge Architecture Integration

Extends v1 feature store with Week 1-4 knowledge features:
- Week 1: Structure (Internal/External, BOMS, Squiggle, Range Outcomes)
- Week 2: Psychology & Volume (PTI, FRVP, Fakeout Intensity)
- Week 4: Macro Echo (DXY/Oil/Yields/VIX correlations)

All features computed causally (past-only data, no future leak).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime
import argparse

from engine.io.tradingview_loader import load_tv
from engine.context.loader import load_macro_data, fetch_macro_snapshot
from engine.context.macro_engine import analyze_macro, create_default_macro_config
from engine.fusion.domain_fusion import analyze_fusion

# Week 1: Structure
from engine.structure.internal_external import detect_structure_state
from engine.structure.boms_detector import detect_boms
from engine.structure.squiggle_pattern import detect_squiggle_123
from engine.structure.range_classifier import classify_range_outcome

# Week 2: Psychology & Volume
from engine.psychology.pti import (
    detect_rsi_divergence, detect_volume_exhaustion,
    detect_wick_trap, detect_failed_breakout
)
from engine.volume.frvp import calculate_frvp
from engine.psychology.fakeout_intensity import detect_fakeout_intensity

# Week 4: Macro Echo
from engine.exits.macro_echo import analyze_macro_echo

# Contract validation
from engine.fusion.knowledge_hooks import assert_feature_contract


def compute_week1_features(window_1h: pd.DataFrame, window_4h: pd.DataFrame,
                           window_1d: pd.DataFrame, config: dict) -> dict:
    """
    Compute Week 1 structure features.

    Returns dict with all structure feature columns.
    """
    features = {}

    try:
        # Internal vs External
        structure_state = detect_structure_state(
            window_1h, window_4h, window_1d, config
        )
        features.update(structure_state.to_dict())

        # BOMS (detect on both 1H and 4H)
        boms_4h = detect_boms(window_4h, timeframe='4H', config=config)
        features.update(boms_4h.to_dict())

        # Squiggle
        squiggle = detect_squiggle_123(window_4h, timeframe='4H', config=config)
        features.update(squiggle.to_dict())

        # Range Outcomes
        range_outcome = classify_range_outcome(window_4h, timeframe='4H', config=config)
        features.update(range_outcome.to_dict())

    except Exception as e:
        # Fill with neutral/default values on error
        features = get_default_structure_features()

    return features


def compute_week2_features(window_1h: pd.DataFrame, config: dict) -> dict:
    """
    Compute Week 2 psychology & volume features.

    Returns dict with all psychology/volume feature columns.
    """
    features = {}

    try:
        # PTI components (calculate individually)
        rsi_div = detect_rsi_divergence(window_1h, lookback=20)
        vol_exh = detect_volume_exhaustion(window_1h, lookback=10)
        wick_trap = detect_wick_trap(window_1h, lookback=5)
        failed_bo = detect_failed_breakout(window_1h, lookback=20)

        # Combine into PTI
        pti_score = (
            rsi_div.get('divergence_strength', 0.0) * 0.30 +
            vol_exh.get('exhaustion_score', 0.0) * 0.25 +
            wick_trap.get('trap_strength', 0.0) * 0.25 +
            failed_bo.get('failure_score', 0.0) * 0.20
        )

        features['pti_score'] = pti_score
        features['pti_trap_type'] = 'bullish_trap' if pti_score > 0.6 else 'none'
        features['pti_confidence'] = pti_score
        features['pti_reversal_likely'] = pti_score > 0.7
        features['pti_rsi_divergence'] = rsi_div.get('divergence_strength', 0.0)
        features['pti_volume_exhaustion'] = vol_exh.get('exhaustion_score', 0.0)
        features['pti_wick_trap'] = wick_trap.get('trap_strength', 0.0)
        features['pti_failed_breakout'] = failed_bo.get('failure_score', 0.0)

        # FRVP
        frvp_result = calculate_frvp(window_1h, lookback=100, config=config)
        features.update(frvp_result.to_dict())

        # Fakeout Intensity
        fakeout = detect_fakeout_intensity(window_1h, lookback=30, config=config)
        features.update(fakeout.to_dict())

    except Exception as e:
        # Fill with neutral/default values on error
        features = get_default_psychology_features()

    return features


def compute_week4_features(macro_data: dict, timestamp: pd.Timestamp, config: dict) -> dict:
    """
    Compute Week 4 macro echo features.

    Returns dict with all macro echo feature columns.
    """
    features = {}

    try:
        # Fetch macro snapshot at timestamp
        ts_naive = timestamp.replace(tzinfo=None) if hasattr(timestamp, 'tzinfo') else timestamp
        snapshot = fetch_macro_snapshot(macro_data, ts_naive)

        # Analyze macro echo
        macro_echo = analyze_macro_echo({
            'DXY': snapshot.get('dxy_series', pd.Series([100.0])),
            'YIELDS_10Y': snapshot.get('yields_series', pd.Series([4.0])),
            'OIL': snapshot.get('oil_series', pd.Series([75.0])),
            'VIX': snapshot.get('vix_series', pd.Series([18.0]))
        }, lookback=7, config=config)

        features.update(macro_echo.to_dict())

    except Exception as e:
        # Fill with neutral/default values on error
        features = get_default_macro_features()

    return features


def get_default_structure_features() -> dict:
    """Default neutral values for structure features."""
    return {
        # Internal vs External
        'internal_phase': 'transition',
        'external_trend': 'neutral',
        'structure_alignment': False,
        'conflict_score': 0.0,
        'internal_strength': 0.5,
        'external_strength': 0.5,
        # BOMS
        'boms_detected': False,
        'boms_direction': 'none',
        'boms_volume_surge': 1.0,
        'boms_fvg_present': False,
        'boms_confirmation': 0,
        'boms_break_level': 0.0,
        'boms_displacement': 0.0,
        # Squiggle
        'squiggle_stage': 0,
        'squiggle_pattern_id': '',
        'squiggle_direction': 'none',
        'squiggle_entry_window': False,
        'squiggle_confidence': 0.0,
        'squiggle_bos_level': 0.0,
        'squiggle_retest_quality': 0.0,
        'squiggle_bars_since_bos': 999,
        # Range Outcomes
        'range_outcome': 'none',
        'range_outcome_direction': 'neutral',
        'range_outcome_confidence': 0.0,
        'range_high': 0.0,
        'range_low': 0.0,
        'breakout_strength': 0.0,
        'volume_confirmation': False,
        'bars_in_range': 0,
    }


def get_default_psychology_features() -> dict:
    """Default neutral values for psychology/volume features."""
    return {
        # PTI
        'pti_score': 0.0,
        'pti_trap_type': 'none',
        'pti_confidence': 0.0,
        'pti_reversal_likely': False,
        'pti_rsi_divergence': 0.0,
        'pti_volume_exhaustion': 0.0,
        'pti_wick_trap': 0.0,
        'pti_failed_breakout': 0.0,
        # FRVP
        'frvp_poc': 0.0,
        'frvp_va_high': 0.0,
        'frvp_va_low': 0.0,
        'frvp_hvn_count': 0,
        'frvp_lvn_count': 0,
        'frvp_current_position': 'in_va',
        'frvp_distance_to_poc': 0.0,
        'frvp_distance_to_va': 0.0,
        # Fakeout
        'fakeout_detected': False,
        'fakeout_intensity': 0.0,
        'fakeout_direction': 'none',
        'fakeout_breakout_level': 0.0,
        'fakeout_return_speed': 999,
        'fakeout_volume_weakness': 0.0,
        'fakeout_wick_rejection': 0.0,
        'fakeout_no_followthrough': 0.0,
    }


def get_default_macro_features() -> dict:
    """Default neutral values for macro echo features."""
    return {
        'macro_regime': 'neutral',
        'macro_dxy_trend': 'flat',
        'macro_yields_trend': 'flat',
        'macro_oil_trend': 'flat',
        'macro_vix_level': 'medium',
        'macro_correlation_score': 0.0,
        'macro_exit_recommended': False,
    }


def build_feature_store_v2(asset: str, start_date: str, end_date: str,
                          sample_every: int = 4, include_week1_4: bool = True):
    """
    Build feature store v2.0 with Week 1-4 knowledge features.

    Args:
        asset: Asset to build features for (BTC, ETH, SOL)
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        sample_every: Sample domain scores every N bars (default 4)
        include_week1_4: Include Week 1-4 features (default True)
    """
    print(f"🏗️  Building Feature Store v2.0: {asset}")
    print(f"📅 Period: {start_date} → {end_date}")
    print(f"⚡ Sampling: Every {sample_every} bars for domain scores")
    print(f"🧠 Week 1-4 Features: {'ENABLED' if include_week1_4 else 'DISABLED'}")
    print("=" * 70)

    # Load raw data
    print("\n📊 Loading OHLCV data...")
    df_1h = load_tv(f"{asset}_1H")
    df_4h = load_tv(f"{asset}_4H")
    df_1d = load_tv(f"{asset}_1D")

    # Filter to date range
    start_ts = pd.Timestamp(start_date, tz='UTC')
    end_ts = pd.Timestamp(end_date, tz='UTC')
    df_1h = df_1h[(df_1h.index >= start_ts) & (df_1h.index <= end_ts)].copy()
    df_4h = df_4h[(df_4h.index >= start_ts) & (df_4h.index <= end_ts)].copy()
    df_1d = df_1d[(df_1d.index >= start_ts) & (df_1d.index <= end_ts)].copy()

    # Standardize columns
    for df in [df_1h, df_4h, df_1d]:
        df.columns = [c.lower() for c in df.columns]

    print(f"   {len(df_1h)} 1H bars, {len(df_4h)} 4H bars, {len(df_1d)} 1D bars")

    # Load macro data
    print("\n📈 Loading macro data...")
    macro_data = load_macro_data()
    macro_config = create_default_macro_config()
    macro_config['macro_veto_threshold'] = 0.90  # Relaxed
    print(f"   Macro veto threshold: {macro_config['macro_veto_threshold']} (relaxed)")

    # Initialize feature dataframe
    print("\n🔧 Initializing feature store...")
    features = pd.DataFrame(index=df_1h.index)

    # OHLCV
    features['open'] = df_1h['open']
    features['high'] = df_1h['high']
    features['low'] = df_1h['low']
    features['close'] = df_1h['close']
    features['volume'] = df_1h['volume']

    # Technical indicators (from original builder)
    from bin.build_feature_store import calculate_atr, calculate_adx, calculate_rsi
    features['atr_20'] = calculate_atr(df_1h, 20)
    features['atr_14'] = calculate_atr(df_1h, 14)
    features['adx_14'] = calculate_adx(df_1h, 14)
    features['rsi_14'] = calculate_rsi(df_1h, 14)

    for period in [20, 50, 100]:
        features[f'sma_{period}'] = df_1h['close'].rolling(period).mean()

    print("   ✅ Technical indicators computed")

    # Macro veto flags
    print("\n🌍 Computing macro veto flags...")
    macro_veto_flags = []
    macro_exit_flags = []

    for idx, timestamp in enumerate(df_1h.index):
        if idx % 100 == 0:
            print(f"   Processing macro bar {idx}/{len(df_1h)}...")

        ts_naive = timestamp.replace(tzinfo=None) if hasattr(timestamp, 'tzinfo') else timestamp
        snapshot = fetch_macro_snapshot(macro_data, ts_naive)
        macro_result = analyze_macro(snapshot, macro_config)

        veto = macro_result['veto_strength'] >= macro_config['macro_veto_threshold']
        crisis_exit = macro_result['veto_strength'] >= 0.90

        macro_veto_flags.append(veto)
        macro_exit_flags.append(crisis_exit)

    features['macro_veto'] = macro_veto_flags
    features['macro_exit_flag'] = macro_exit_flags
    print("   ✅ Macro flags computed")

    # Domain scores (sampled)
    print(f"\n🔮 Computing domain scores (sampling every {sample_every} bars)...")

    wyckoff_scores = np.full(len(df_1h), 0.5)
    smc_scores = np.full(len(df_1h), 0.5)
    hob_scores = np.full(len(df_1h), 0.5)
    momentum_scores = np.full(len(df_1h), 0.5)
    temporal_scores = np.full(len(df_1h), 0.5)

    sampled_count = 0
    for i in range(len(df_1h)):
        if i % 100 == 0:
            print(f"   Processing domain bar {i}/{len(df_1h)}...")

        # Only compute every Nth bar
        if i % sample_every != 0 and i != len(df_1h) - 1:
            if i > 0:
                wyckoff_scores[i] = wyckoff_scores[i-1]
                smc_scores[i] = smc_scores[i-1]
                hob_scores[i] = hob_scores[i-1]
                momentum_scores[i] = momentum_scores[i-1]
                temporal_scores[i] = temporal_scores[i-1]
            continue

        current_time_1h = df_1h.index[i]

        # Get windows (causal - only past data)
        window_1h = df_1h.iloc[:i+1].tail(200)
        window_4h = df_4h[df_4h.index <= current_time_1h].tail(100)
        window_1d = df_1d[df_1d.index <= current_time_1h].tail(50)

        if len(window_1h) < 50 or len(window_4h) < 14 or len(window_1d) < 20:
            continue

        try:
            fusion_result = analyze_fusion(
                window_1h, window_4h, window_1d,
                config={'fusion': {
                    'weights': {'wyckoff': 0.30, 'smc': 0.15, 'liquidity': 0.25, 'momentum': 0.30}
                }}
            )

            wyckoff_scores[i] = fusion_result.wyckoff_score
            smc_scores[i] = fusion_result.smc_score
            hob_scores[i] = fusion_result.hob_score
            momentum_scores[i] = fusion_result.momentum_score

            sampled_count += 1
        except:
            pass

    features['wyckoff'] = wyckoff_scores
    features['smc'] = smc_scores
    features['hob'] = hob_scores
    features['momentum'] = momentum_scores
    features['temporal'] = temporal_scores

    print(f"   ✅ Domain scores computed ({sampled_count} actual fusion calls)")

    # MTF alignment
    print("\n🎯 Computing MTF alignment flags...")
    mtf_aligned = np.zeros(len(df_1h), dtype=bool)

    for i in range(len(df_1h)):
        current_time = df_1h.index[i]
        nearest_4h = df_4h[df_4h.index <= current_time]
        nearest_1d = df_1d[df_1d.index <= current_time]

        if len(nearest_4h) > 50 and len(nearest_1d) > 20:
            h1_aligned = df_1h['close'].iloc[i] > features[f'sma_20'].iloc[i]
            h4_aligned = nearest_4h['close'].iloc[-1] > nearest_4h['close'].rolling(20).mean().iloc[-1]
            d1_aligned = nearest_1d['close'].iloc[-1] > nearest_1d['close'].rolling(20).mean().iloc[-1]

            aligned_count = sum([h1_aligned, h4_aligned, d1_aligned])
            if aligned_count >= 2:
                mtf_aligned[i] = True

    features['mtf_align'] = mtf_aligned
    print("   ✅ MTF alignment computed")

    # Week 1-4 Features
    if include_week1_4:
        print("\n🧠 Computing Week 1-4 knowledge features...")
        print("   This may take a while (computing 66 new feature columns)...")

        config = {}  # Default config for all modules

        # Initialize all feature columns with defaults
        for col, val in get_default_structure_features().items():
            features[col] = val if not isinstance(val, (int, float, bool)) else np.full(len(df_1h), val)

        for col, val in get_default_psychology_features().items():
            features[col] = val if not isinstance(val, (int, float, bool)) else np.full(len(df_1h), val)

        for col, val in get_default_macro_features().items():
            features[col] = val if not isinstance(val, (int, float, bool)) else np.full(len(df_1h), val)

        computed_count = 0
        for i in range(len(df_1h)):
            if i % 50 == 0:
                print(f"   Processing knowledge bar {i}/{len(df_1h)}...")

            current_time = df_1h.index[i]

            # Get causal windows
            window_1h = df_1h.iloc[:i+1].tail(200)
            window_4h = df_4h[df_4h.index <= current_time].tail(100)
            window_1d = df_1d[df_1d.index <= current_time].tail(50)

            # Skip if not enough data
            if len(window_1h) < 50:
                continue

            try:
                # Week 1: Structure
                struct_feats = compute_week1_features(window_1h, window_4h, window_1d, config)
                for col, val in struct_feats.items():
                    features.loc[current_time, col] = val

                # Week 2: Psychology & Volume
                psych_feats = compute_week2_features(window_1h, config)
                for col, val in psych_feats.items():
                    features.loc[current_time, col] = val

                # Week 4: Macro Echo
                macro_feats = compute_week4_features(macro_data, current_time, config)
                for col, val in macro_feats.items():
                    features.loc[current_time, col] = val

                computed_count += 1
            except Exception as e:
                # Keep default values on error
                pass

        print(f"   ✅ Week 1-4 knowledge features computed ({computed_count} bars)")

    # Drop initial NaN rows
    features = features.dropna(subset=['atr_20', 'sma_20'])

    # Save to parquet
    output_dir = Path('data/features_v2')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'{asset}_1H_{start_date}_to_{end_date}.parquet'

    # Add schema version metadata
    features.attrs['schema_version'] = '2.0'
    features.attrs['asset'] = asset
    features.attrs['start_date'] = start_date
    features.attrs['end_date'] = end_date
    features.attrs['include_week1_4'] = include_week1_4

    features.to_parquet(output_path)

    print(f"\n💾 Feature store v2.0 saved:")
    print(f"   {output_path}")
    print(f"   {len(features)} bars × {len(features.columns)} features")
    print(f"   Size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

    # Validate feature contract
    if include_week1_4:
        print(f"\n🔍 Validating feature contract (schema v2.0)...")
        try:
            assert_feature_contract(features, schema_version="2.0")
            print("   ✅ Feature contract validated: All 104 columns present")
        except AssertionError as e:
            print(f"   ❌ Feature contract validation failed:")
            print(f"      {e}")
            return None

    print("\n✅ Feature store v2.0 build complete!")

    return features


def main():
    parser = argparse.ArgumentParser(description='Build feature store v2.0 with Week 1-4 features')
    parser.add_argument('--asset', default='ETH', help='Asset to process (BTC, ETH, SOL)')
    parser.add_argument('--start', default='2024-07-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', default='2024-09-30', help='End date (YYYY-MM-DD)')
    parser.add_argument('--sample', type=int, default=4, help='Sample domain scores every N bars')
    parser.add_argument('--include-week1-4', action='store_true', default=True,
                       help='Include Week 1-4 features (default: True)')

    args = parser.parse_args()

    build_feature_store_v2(args.asset, args.start, args.end, args.sample, args.include_week1_4)


if __name__ == '__main__':
    main()
