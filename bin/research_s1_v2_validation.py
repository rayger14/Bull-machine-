#!/usr/bin/env python3
"""
S1 V2 Features Research & Validation
====================================

OBJECTIVES:
1. Understand why FTX (Nov 9 2022) had lower crisis_composite (0.303) vs LUNA (0.639)
2. Validate V2 features on 2023-2024 capitulation events
3. Analyze feature distributions and refine threshold recommendations
4. Identify optimal threshold combinations with empirical backing

RESEARCH QUESTIONS:
Q1: Why was FTX crisis_composite so low?
Q2: Do 2023-2024 events validate proposed thresholds?
Q3: What's the distribution of V2 features across 2022-2024?
Q4: What thresholds optimize precision/recall?

Author: Claude Code (Research Agent)
Date: 2025-11-23
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.dates import DateFormatter

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.strategies.archetypes.bear.liquidity_vacuum_runtime import (
    apply_liquidity_vacuum_enrichment,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set plotting style
sns.set_style('darkgrid')
plt.rcParams['figure.figsize'] = (14, 8)

# Output directory
OUTPUT_DIR = Path('results/s1_v2_research')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Known Capitulation Events (Ground Truth)
# ============================================================================

CAPITULATION_EVENTS = [
    # 2022 Events
    {
        'date': '2022-05-12',
        'name': 'LUNA Death Spiral',
        'price_range': '$40k → $26k',
        'catalyst': 'UST depeg, LUNA collapse',
        'severity': 'Extreme',
        'notes': 'Multi-day panic, stablecoin crisis'
    },
    {
        'date': '2022-06-18',
        'name': 'LUNA Capitulation Bottom',
        'price_range': '$22k → $17.6k',
        'catalyst': 'Final LUNA washout, 3AC rumors',
        'severity': 'Extreme',
        'notes': 'True capitulation, deep wick, massive volume'
    },
    {
        'date': '2022-11-09',
        'name': 'FTX Collapse',
        'price_range': '$21k → $15.6k',
        'catalyst': 'FTX insolvency, exchange run',
        'severity': 'Severe',
        'notes': 'Fast event, microstructure break'
    },

    # 2023 Events
    {
        'date': '2023-03-10',
        'name': 'SVB Banking Crisis',
        'price_range': '$22k → $19.6k',
        'catalyst': 'Silicon Valley Bank collapse',
        'severity': 'Moderate',
        'notes': 'Tradfi contagion, quick recovery'
    },
    {
        'date': '2023-08-17',
        'name': 'August Flush',
        'price_range': '$29.5k → $25.2k',
        'catalyst': 'China macro concerns',
        'severity': 'Mild',
        'notes': 'Healthy correction, not extreme'
    },

    # 2024 Events
    {
        'date': '2024-08-05',
        'name': 'Japan Carry Trade Unwind',
        'price_range': '$62k → $49k',
        'catalyst': 'BOJ rate hike, carry unwind',
        'severity': 'Severe',
        'notes': 'Global macro shock, violent selloff'
    },
    {
        'date': '2024-09-06',
        'name': 'September Flush',
        'price_range': '$59k → $52.5k',
        'catalyst': 'Labor market concerns',
        'severity': 'Moderate',
        'notes': 'Orderly correction'
    },
]


# ============================================================================
# Data Loading & Enrichment
# ============================================================================

def load_and_enrich_data() -> pd.DataFrame:
    """Load BTC data and apply S1 V2 enrichment"""

    logger.info("Loading BTC 1H data (2022-2024)...")

    # Try different file patterns
    data_files = [
        'data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet',
        'data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31_with_macro.parquet',
    ]

    df = None
    for fpath in data_files:
        if Path(fpath).exists():
            logger.info(f"Loading: {fpath}")
            df = pd.read_parquet(fpath)
            break

    if df is None:
        # Try loading and concatenating separate years
        logger.info("Trying separate year files...")
        df_2022 = pd.read_parquet('data/features_mtf/BTC_1H_2022-01-01_to_2023-12-31.parquet')
        df_2024 = pd.read_parquet('data/features_mtf/BTC_1H_2024-01-01_to_2024-12-31.parquet')
        df = pd.concat([df_2022, df_2024], axis=0)

    logger.info(f"Loaded {len(df)} bars from {df.index.min()} to {df.index.max()}")

    # Filter to 2022-2024
    df = df[(df.index >= '2022-01-01') & (df.index < '2025-01-01')].copy()
    logger.info(f"Filtered to {len(df)} bars (2022-2024)")

    # Check feature availability
    logger.info("\nFeature availability:")
    required_features = ['close', 'high', 'low', 'open', 'volume',
                        'liquidity_score', 'VIX_Z', 'DXY_Z', 'funding_Z', 'rv_20d']
    for feat in required_features:
        status = "✓" if feat in df.columns else "✗"
        logger.info(f"  {status} {feat}")

    # Apply S1 V2 enrichment
    logger.info("\nApplying S1 V2 runtime enrichment...")
    df_enriched = apply_liquidity_vacuum_enrichment(df)

    logger.info(f"Enrichment complete! Added V2 features:")
    v2_features = ['liquidity_drain_pct', 'liquidity_velocity', 'liquidity_persistence',
                   'capitulation_depth', 'crisis_composite', 'volume_climax_last_3b',
                   'wick_exhaustion_last_3b']
    for feat in v2_features:
        if feat in df_enriched.columns:
            logger.info(f"  ✓ {feat}")

    return df_enriched


# ============================================================================
# Q1: FTX Crisis Analysis
# ============================================================================

def analyze_ftx_crisis(df: pd.DataFrame) -> Dict:
    """
    Deep dive into why FTX had low crisis_composite vs LUNA

    Returns:
        Dict with component breakdown and analysis
    """

    logger.info("\n" + "="*80)
    logger.info("Q1: WHY WAS FTX CRISIS_COMPOSITE SO LOW?")
    logger.info("="*80)

    # Extract major events
    events_to_compare = ['2022-06-18', '2022-11-09']  # LUNA vs FTX
    event_names = ['LUNA Capitulation', 'FTX Collapse']

    results = {}

    for date_str, event_name in zip(events_to_compare, event_names):
        try:
            # Get event window (±6 hours for peak detection)
            start_ts = pd.Timestamp(date_str, tz='UTC')
            end_ts = start_ts + pd.Timedelta(hours=6)
            event_window = df.loc[start_ts:end_ts]

            if len(event_window) == 0:
                logger.warning(f"No data for {event_name} ({date_str})")
                continue

            # Find peak crisis_composite in window
            peak_idx = event_window['crisis_composite'].idxmax()
            peak_row = event_window.loc[peak_idx]

            logger.info(f"\n{event_name} ({date_str}):")
            logger.info(f"  Peak timestamp: {peak_idx}")
            logger.info(f"  Price: ${peak_row['close']:.2f}")
            logger.info(f"  Crisis Composite: {peak_row['crisis_composite']:.3f}")
            logger.info(f"  Capitulation Depth: {peak_row['capitulation_depth']:.3f}")

            # Component breakdown
            logger.info(f"\n  Component Breakdown:")
            logger.info(f"    VIX_Z: {peak_row.get('VIX_Z', np.nan):.3f}")
            logger.info(f"    Funding_Z: {peak_row.get('funding_Z', np.nan):.3f}")
            logger.info(f"    RV_20d: {peak_row.get('rv_20d', np.nan):.3f}")
            logger.info(f"    Drawdown: {peak_row['capitulation_depth']:.3f}")

            logger.info(f"\n  V2 Features:")
            logger.info(f"    Liquidity Drain %: {peak_row['liquidity_drain_pct']:.3f}")
            logger.info(f"    Liquidity Score: {peak_row.get('liquidity_score', np.nan):.3f}")
            logger.info(f"    Volume Climax 3B: {peak_row['volume_climax_last_3b']:.3f}")
            logger.info(f"    Wick Exhaustion 3B: {peak_row['wick_exhaustion_last_3b']:.3f}")

            results[event_name] = {
                'date': date_str,
                'peak_timestamp': peak_idx,
                'crisis_composite': peak_row['crisis_composite'],
                'capitulation_depth': peak_row['capitulation_depth'],
                'vix_z': peak_row.get('VIX_Z', np.nan),
                'funding_z': peak_row.get('funding_Z', np.nan),
                'rv_20d': peak_row.get('rv_20d', np.nan),
                'liquidity_drain_pct': peak_row['liquidity_drain_pct'],
                'volume_climax_3b': peak_row['volume_climax_last_3b'],
                'wick_exhaustion_3b': peak_row['wick_exhaustion_last_3b'],
            }

        except Exception as e:
            logger.error(f"Error analyzing {event_name}: {e}")

    # Analysis
    if len(results) == 2:
        luna = results['LUNA Capitulation']
        ftx = results['FTX Collapse']

        logger.info("\n" + "="*80)
        logger.info("COMPARATIVE ANALYSIS:")
        logger.info("="*80)

        logger.info(f"\nCrisis Composite:")
        logger.info(f"  LUNA: {luna['crisis_composite']:.3f}")
        logger.info(f"  FTX:  {ftx['crisis_composite']:.3f}")
        logger.info(f"  Δ:    {luna['crisis_composite'] - ftx['crisis_composite']:.3f} ({(luna['crisis_composite'] - ftx['crisis_composite'])/luna['crisis_composite']*100:.1f}%)")

        logger.info(f"\nKey Differences:")
        logger.info(f"  VIX Stress:      LUNA={luna['vix_z']:.2f}, FTX={ftx['vix_z']:.2f}")
        logger.info(f"  Funding Stress:  LUNA={luna['funding_z']:.2f}, FTX={ftx['funding_z']:.2f}")
        logger.info(f"  Drawdown Depth:  LUNA={luna['capitulation_depth']:.2f}, FTX={ftx['capitulation_depth']:.2f}")

        logger.info(f"\nHYPOTHESIS:")
        logger.info(f"  FTX was a 'fast event' - microstructure break (exchange collapse)")
        logger.info(f"  Macro indicators (VIX, funding) hadn't caught up yet")
        logger.info(f"  LUNA was slower macro stress buildup → higher crisis_composite")
        logger.info(f"  RECOMMENDATION: Lower crisis_composite threshold OR add exchange-specific triggers")

    return results


# ============================================================================
# Q2: 2023-2024 Validation
# ============================================================================

def validate_2023_2024_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze all known capitulation events in 2023-2024

    Returns:
        DataFrame with validation results
    """

    logger.info("\n" + "="*80)
    logger.info("Q2: 2023-2024 VALIDATION")
    logger.info("="*80)

    validation_results = []

    for event in CAPITULATION_EVENTS:
        date_str = event['date']
        event_name = event['name']

        try:
            # Get event window (timezone-aware)
            start_ts = pd.Timestamp(date_str, tz='UTC')
            end_ts = start_ts + pd.Timedelta(hours=12)
            event_window = df.loc[start_ts:end_ts]

            if len(event_window) == 0:
                logger.warning(f"No data for {event_name} ({date_str})")
                validation_results.append({
                    'Date': date_str,
                    'Event': event_name,
                    'Severity': event['severity'],
                    'Data_Available': False
                })
                continue

            # Find peak values in window
            peak_depth_idx = event_window['capitulation_depth'].idxmin()  # Most negative
            peak_crisis_idx = event_window['crisis_composite'].idxmax()
            peak_vol_idx = event_window['volume_climax_last_3b'].idxmax()
            peak_wick_idx = event_window['wick_exhaustion_last_3b'].idxmax()

            # Get peak values
            peak_depth = event_window.loc[peak_depth_idx, 'capitulation_depth']
            peak_crisis = event_window.loc[peak_crisis_idx, 'crisis_composite']
            peak_vol = event_window.loc[peak_vol_idx, 'volume_climax_last_3b']
            peak_wick = event_window.loc[peak_wick_idx, 'wick_exhaustion_last_3b']

            # Proposed thresholds
            DEPTH_MAX = -0.20  # More negative than this
            CRISIS_MIN = 0.40
            VOL_MIN = 0.25
            WICK_MIN = 0.30

            # Check detection
            detected_depth = peak_depth < DEPTH_MAX
            detected_crisis = peak_crisis > CRISIS_MIN
            detected_vol = peak_vol > VOL_MIN
            detected_wick = peak_wick > WICK_MIN
            detected_all = detected_depth and detected_crisis

            logger.info(f"\n{event_name} ({date_str}) - {event['severity']}:")
            logger.info(f"  Capitulation Depth: {peak_depth:.3f} {'✓' if detected_depth else '✗'} (threshold: {DEPTH_MAX})")
            logger.info(f"  Crisis Composite:   {peak_crisis:.3f} {'✓' if detected_crisis else '✗'} (threshold: {CRISIS_MIN})")
            logger.info(f"  Volume Climax 3B:   {peak_vol:.3f} {'✓' if detected_vol else '✗'} (threshold: {VOL_MIN})")
            logger.info(f"  Wick Exhaustion 3B: {peak_wick:.3f} {'✓' if detected_wick else '✗'} (threshold: {WICK_MIN})")
            logger.info(f"  DETECTED: {'YES ✓' if detected_all else 'NO ✗'}")

            validation_results.append({
                'Date': date_str,
                'Event': event_name,
                'Severity': event['severity'],
                'Price_Range': event['price_range'],
                'Depth': peak_depth,
                'Crisis': peak_crisis,
                'Vol_3b': peak_vol,
                'Wick_3b': peak_wick,
                'Detected_Depth': detected_depth,
                'Detected_Crisis': detected_crisis,
                'Detected_All': detected_all,
                'Data_Available': True
            })

        except Exception as e:
            logger.error(f"Error validating {event_name}: {e}")
            import traceback
            traceback.print_exc()

    # Create summary table
    results_df = pd.DataFrame(validation_results)

    logger.info("\n" + "="*80)
    logger.info("VALIDATION SUMMARY:")
    logger.info("="*80)
    logger.info(f"\nTotal events: {len(results_df)}")
    logger.info(f"Data available: {results_df['Data_Available'].sum()}")

    if results_df['Data_Available'].sum() > 0:
        available = results_df[results_df['Data_Available']]
        logger.info(f"\nDetection Rate:")
        logger.info(f"  Depth threshold: {available['Detected_Depth'].sum()}/{len(available)} ({available['Detected_Depth'].mean()*100:.1f}%)")
        logger.info(f"  Crisis threshold: {available['Detected_Crisis'].sum()}/{len(available)} ({available['Detected_Crisis'].mean()*100:.1f}%)")
        logger.info(f"  Both (AND):      {available['Detected_All'].sum()}/{len(available)} ({available['Detected_All'].mean()*100:.1f}%)")

        # By severity
        logger.info(f"\nDetection by Severity:")
        for severity in ['Extreme', 'Severe', 'Moderate', 'Mild']:
            sev_data = available[available['Severity'] == severity]
            if len(sev_data) > 0:
                detected = sev_data['Detected_All'].sum()
                total = len(sev_data)
                logger.info(f"  {severity:10s}: {detected}/{total} ({detected/total*100:.1f}%)")

    return results_df


# ============================================================================
# Q3: Distribution Analysis
# ============================================================================

def analyze_distributions(df: pd.DataFrame):
    """
    Analyze distribution of V2 features across 2022-2024
    """

    logger.info("\n" + "="*80)
    logger.info("Q3: DISTRIBUTION ANALYSIS")
    logger.info("="*80)

    # Key V2 features
    features = [
        'capitulation_depth',
        'crisis_composite',
        'volume_climax_last_3b',
        'wick_exhaustion_last_3b',
        'liquidity_drain_pct'
    ]

    for feat in features:
        if feat not in df.columns:
            logger.warning(f"Feature {feat} not found, skipping")
            continue

        data = df[feat].dropna()

        logger.info(f"\n{feat}:")
        logger.info(f"  Count: {len(data)}")
        logger.info(f"  Mean:  {data.mean():.4f}")
        logger.info(f"  Std:   {data.std():.4f}")
        logger.info(f"  Min:   {data.min():.4f}")
        logger.info(f"  Max:   {data.max():.4f}")

        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        logger.info(f"  Percentiles:")
        for p in percentiles:
            val = np.percentile(data, p)
            logger.info(f"    p{p:>3d}: {val:.4f}")

    # Combined threshold analysis
    logger.info("\n" + "="*80)
    logger.info("THRESHOLD COMBINATIONS:")
    logger.info("="*80)

    depth_col = 'capitulation_depth'
    crisis_col = 'crisis_composite'

    if depth_col in df.columns and crisis_col in df.columns:
        total_bars = len(df)

        # Various threshold combinations
        threshold_combos = [
            # (depth_max, crisis_min, name)
            (-0.15, 0.30, 'Aggressive'),
            (-0.20, 0.35, 'Moderate'),
            (-0.20, 0.40, 'Proposed'),
            (-0.25, 0.40, 'Conservative'),
            (-0.25, 0.45, 'Very Conservative'),
        ]

        for depth_thresh, crisis_thresh, name in threshold_combos:
            depth_match = (df[depth_col] < depth_thresh).sum()
            crisis_match = (df[crisis_col] > crisis_thresh).sum()
            both_match = ((df[depth_col] < depth_thresh) &
                         (df[crisis_col] > crisis_thresh)).sum()

            depth_pct = depth_match / total_bars * 100
            crisis_pct = crisis_match / total_bars * 100
            both_pct = both_match / total_bars * 100

            # Annualized trade estimate (3 years of data)
            trades_per_year = both_match / 3

            logger.info(f"\n{name}: depth<{depth_thresh}, crisis>{crisis_thresh}")
            logger.info(f"  Depth matches:  {depth_match:5d} ({depth_pct:5.2f}%)")
            logger.info(f"  Crisis matches: {crisis_match:5d} ({crisis_pct:5.2f}%)")
            logger.info(f"  Both (AND):     {both_match:5d} ({both_pct:5.2f}%)")
            logger.info(f"  Est. trades/year: {trades_per_year:.1f}")

    # Create distribution plots
    create_distribution_plots(df)


def create_distribution_plots(df: pd.DataFrame):
    """Create distribution visualization plots"""

    logger.info("\nCreating distribution plots...")

    # 1. Histograms for each V2 feature
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('S1 V2 Feature Distributions (2022-2024)', fontsize=16, fontweight='bold')

    features = [
        ('capitulation_depth', 'Capitulation Depth'),
        ('crisis_composite', 'Crisis Composite'),
        ('volume_climax_last_3b', 'Volume Climax (Last 3 Bars)'),
        ('wick_exhaustion_last_3b', 'Wick Exhaustion (Last 3 Bars)'),
        ('liquidity_drain_pct', 'Liquidity Drain %'),
        ('liquidity_persistence', 'Liquidity Persistence (Bars)'),
    ]

    for idx, (feat, label) in enumerate(features):
        ax = axes[idx // 3, idx % 3]

        if feat in df.columns:
            data = df[feat].dropna()
            ax.hist(data, bins=100, alpha=0.7, color='steelblue', edgecolor='black')
            ax.set_xlabel(label, fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.set_title(f'{label}\n(n={len(data):,})', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)

            # Add percentile lines
            for p, color, style in [(5, 'green', '--'), (95, 'red', '--')]:
                val = np.percentile(data, p)
                ax.axvline(val, color=color, linestyle=style, linewidth=2,
                          label=f'p{p}={val:.3f}')
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, f'{feat}\nNot Available',
                   ha='center', va='center', fontsize=12)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 's1_v2_distributions.png', dpi=150, bbox_inches='tight')
    logger.info(f"  Saved: {OUTPUT_DIR / 's1_v2_distributions.png'}")
    plt.close()

    # 2. Scatter: Depth vs Crisis (with events highlighted)
    if 'capitulation_depth' in df.columns and 'crisis_composite' in df.columns:
        fig, ax = plt.subplots(figsize=(14, 10))

        # All data points
        ax.scatter(df['capitulation_depth'], df['crisis_composite'],
                  alpha=0.3, s=10, color='gray', label='All bars')

        # Highlight known events
        for event in CAPITULATION_EVENTS:
            try:
                event_window = df.loc[event['date']:pd.Timestamp(event['date']) + pd.Timedelta(hours=12)]
                if len(event_window) > 0:
                    ax.scatter(event_window['capitulation_depth'],
                             event_window['crisis_composite'],
                             alpha=0.8, s=100, label=f"{event['name']} ({event['date']})")
            except:
                pass

        # Threshold lines
        ax.axvline(-0.20, color='red', linestyle='--', linewidth=2, label='Depth threshold (-0.20)')
        ax.axhline(0.40, color='red', linestyle='--', linewidth=2, label='Crisis threshold (0.40)')

        ax.set_xlabel('Capitulation Depth', fontsize=12, fontweight='bold')
        ax.set_ylabel('Crisis Composite', fontsize=12, fontweight='bold')
        ax.set_title('Capitulation Depth vs Crisis Composite\n(Known Events Highlighted)',
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 's1_v2_depth_vs_crisis.png', dpi=150, bbox_inches='tight')
        logger.info(f"  Saved: {OUTPUT_DIR / 's1_v2_depth_vs_crisis.png'}")
        plt.close()


# ============================================================================
# Q4: Threshold Calibration
# ============================================================================

def calibrate_thresholds(df: pd.DataFrame, validation_df: pd.DataFrame):
    """
    Recommend optimal thresholds based on empirical data
    """

    logger.info("\n" + "="*80)
    logger.info("Q4: THRESHOLD CALIBRATION")
    logger.info("="*80)

    # Extract known events for precision calculation
    known_events = validation_df[validation_df['Data_Available']].copy()

    if len(known_events) == 0:
        logger.warning("No validation events available for calibration")
        return

    # Test various threshold combinations
    logger.info("\nTesting threshold combinations...")
    logger.info("(Goal: Maximize recall on known events while controlling false positives)")

    depth_thresholds = [-0.10, -0.15, -0.20, -0.25, -0.30]
    crisis_thresholds = [0.30, 0.35, 0.40, 0.45, 0.50]

    calibration_results = []

    for depth_t in depth_thresholds:
        for crisis_t in crisis_thresholds:
            # Count matches in full dataset
            total_signals = ((df['capitulation_depth'] < depth_t) &
                           (df['crisis_composite'] > crisis_t)).sum()

            # Count matches in known events
            events_detected = ((known_events['Depth'] < depth_t) &
                             (known_events['Crisis'] > crisis_t)).sum()

            # Calculate metrics
            recall = events_detected / len(known_events)
            false_pos_rate = (total_signals - events_detected) / len(df) * 100
            trades_per_year = total_signals / 3  # 3 years of data

            calibration_results.append({
                'depth_threshold': depth_t,
                'crisis_threshold': crisis_t,
                'total_signals': total_signals,
                'events_detected': events_detected,
                'recall': recall,
                'false_pos_rate': false_pos_rate,
                'trades_per_year': trades_per_year
            })

    calib_df = pd.DataFrame(calibration_results)

    # Sort by recall (descending), then FP rate (ascending)
    calib_df = calib_df.sort_values(['recall', 'false_pos_rate'],
                                     ascending=[False, True])

    logger.info("\nTop 10 Threshold Combinations (by Recall):")
    logger.info(f"{'Depth':>8s} {'Crisis':>8s} {'Signals':>8s} {'Detected':>9s} "
               f"{'Recall':>7s} {'FP Rate':>8s} {'Trades/Yr':>10s}")
    logger.info("-" * 80)

    for idx, row in calib_df.head(10).iterrows():
        logger.info(f"{row['depth_threshold']:>8.2f} {row['crisis_threshold']:>8.2f} "
                   f"{row['total_signals']:>8.0f} {row['events_detected']:>9.0f} "
                   f"{row['recall']:>7.1%} {row['false_pos_rate']:>8.3f}% "
                   f"{row['trades_per_year']:>10.1f}")

    # Recommendations
    logger.info("\n" + "="*80)
    logger.info("THRESHOLD RECOMMENDATIONS:")
    logger.info("="*80)

    # Conservative: High recall (≥90%), low FP
    conservative = calib_df[(calib_df['recall'] >= 0.90) &
                           (calib_df['trades_per_year'] <= 15)]
    if len(conservative) > 0:
        best_conservative = conservative.iloc[0]
        logger.info(f"\n1. CONSERVATIVE (High Precision):")
        logger.info(f"   capitulation_depth_max: {best_conservative['depth_threshold']:.2f}")
        logger.info(f"   crisis_composite_min:   {best_conservative['crisis_threshold']:.2f}")
        logger.info(f"   Expected: {best_conservative['trades_per_year']:.1f} trades/year")
        logger.info(f"   Recall:   {best_conservative['recall']:.1%}")
        logger.info(f"   FP Rate:  {best_conservative['false_pos_rate']:.3f}%")

    # Balanced: Good recall (≥80%), moderate FP
    balanced = calib_df[(calib_df['recall'] >= 0.80) &
                       (calib_df['trades_per_year'] >= 10) &
                       (calib_df['trades_per_year'] <= 20)]
    if len(balanced) > 0:
        best_balanced = balanced.iloc[0]
        logger.info(f"\n2. BALANCED (Recommended):")
        logger.info(f"   capitulation_depth_max: {best_balanced['depth_threshold']:.2f}")
        logger.info(f"   crisis_composite_min:   {best_balanced['crisis_threshold']:.2f}")
        logger.info(f"   Expected: {best_balanced['trades_per_year']:.1f} trades/year")
        logger.info(f"   Recall:   {best_balanced['recall']:.1%}")
        logger.info(f"   FP Rate:  {best_balanced['false_pos_rate']:.3f}%")

    # Aggressive: Maximum recall
    aggressive = calib_df[calib_df['recall'] >= 0.95]
    if len(aggressive) > 0:
        best_aggressive = aggressive.sort_values('trades_per_year').iloc[0]
        logger.info(f"\n3. AGGRESSIVE (Max Recall):")
        logger.info(f"   capitulation_depth_max: {best_aggressive['depth_threshold']:.2f}")
        logger.info(f"   crisis_composite_min:   {best_aggressive['crisis_threshold']:.2f}")
        logger.info(f"   Expected: {best_aggressive['trades_per_year']:.1f} trades/year")
        logger.info(f"   Recall:   {best_aggressive['recall']:.1%}")
        logger.info(f"   FP Rate:  {best_aggressive['false_pos_rate']:.3f}%")

    # Save calibration results
    calib_df.to_csv(OUTPUT_DIR / 's1_v2_threshold_calibration.csv', index=False)
    logger.info(f"\nSaved calibration results to: {OUTPUT_DIR / 's1_v2_threshold_calibration.csv'}")


# ============================================================================
# Risk Assessment
# ============================================================================

def assess_risks(validation_df: pd.DataFrame):
    """
    Assess which events would be missed and estimate false positive rate
    """

    logger.info("\n" + "="*80)
    logger.info("RISK ASSESSMENT")
    logger.info("="*80)

    # Proposed thresholds
    DEPTH_MAX = -0.20
    CRISIS_MIN = 0.40

    available = validation_df[validation_df['Data_Available']].copy()

    if len(available) == 0:
        logger.warning("No events available for risk assessment")
        return

    # Missed events
    missed = available[~available['Detected_All']]

    logger.info(f"\nMISSED EVENTS (with proposed thresholds):")
    logger.info(f"Total missed: {len(missed)}/{len(available)}")

    if len(missed) > 0:
        for _, event in missed.iterrows():
            logger.info(f"\n  {event['Event']} ({event['Date']}) - {event['Severity']}:")
            logger.info(f"    Depth:  {event['Depth']:.3f} (need < {DEPTH_MAX})")
            logger.info(f"    Crisis: {event['Crisis']:.3f} (need > {CRISIS_MIN})")
            logger.info(f"    Price:  {event['Price_Range']}")

            # What threshold would catch it?
            if event['Depth'] >= DEPTH_MAX:
                req_depth = event['Depth'] - 0.01
                logger.info(f"    → Need depth_max ≥ {req_depth:.2f} to catch")
            if event['Crisis'] <= CRISIS_MIN:
                req_crisis = event['Crisis'] + 0.01
                logger.info(f"    → Need crisis_min ≤ {req_crisis:.2f} to catch")
    else:
        logger.info("  No missed events! ✓")

    # Severity breakdown
    logger.info(f"\nDETECTION BY SEVERITY:")
    for severity in ['Extreme', 'Severe', 'Moderate', 'Mild']:
        sev_data = available[available['Severity'] == severity]
        if len(sev_data) > 0:
            detected = sev_data['Detected_All'].sum()
            total = len(sev_data)
            logger.info(f"  {severity:10s}: {detected}/{total} detected ({detected/total*100:.0f}%)")


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Execute full S1 V2 research analysis"""

    logger.info("="*80)
    logger.info("S1 V2 FEATURES RESEARCH & VALIDATION")
    logger.info("="*80)
    logger.info(f"Output directory: {OUTPUT_DIR}")

    try:
        # Load and enrich data
        df = load_and_enrich_data()

        # Q1: FTX Crisis Analysis
        ftx_analysis = analyze_ftx_crisis(df)

        # Q2: 2023-2024 Validation
        validation_df = validate_2023_2024_events(df)

        # Q3: Distribution Analysis
        analyze_distributions(df)

        # Q4: Threshold Calibration
        calibrate_thresholds(df, validation_df)

        # Risk Assessment
        assess_risks(validation_df)

        # Save validation results
        if len(validation_df) > 0:
            validation_df.to_csv(OUTPUT_DIR / 's1_v2_validation_results.csv', index=False)
            logger.info(f"\nSaved validation results to: {OUTPUT_DIR / 's1_v2_validation_results.csv'}")

        logger.info("\n" + "="*80)
        logger.info("RESEARCH COMPLETE!")
        logger.info("="*80)
        logger.info(f"\nResults saved to: {OUTPUT_DIR}")
        logger.info("\nGenerated files:")
        logger.info("  - s1_v2_validation_results.csv")
        logger.info("  - s1_v2_threshold_calibration.csv")
        logger.info("  - s1_v2_distributions.png")
        logger.info("  - s1_v2_depth_vs_crisis.png")

    except Exception as e:
        logger.error(f"Research failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
