#!/usr/bin/env python3
"""
Backfill Wyckoff Event Detection to Feature Store

Adds 18+ event columns to existing feature store:
- wyckoff_sc, wyckoff_bc (Phase A climax events)
- wyckoff_spring_a, wyckoff_spring_b (Phase C springs)
- wyckoff_lps, wyckoff_lpsy (Phase D last points)
- wyckoff_sos, wyckoff_sow (Phase B strength signals)
- wyckoff_ut, wyckoff_utad (Phase C upthrusts)
- Plus confidence scores and PTI confluence flags

Usage:
    python3 bin/backfill_wyckoff_events.py --asset BTC --start 2022-01-01 --end 2024-12-31
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import argparse
import logging
from datetime import datetime

# Import Wyckoff event detector
from engine.wyckoff.events import detect_all_wyckoff_events

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Backfill Wyckoff events to feature store')
    parser.add_argument('--asset', type=str, default='BTC', help='Asset symbol')
    parser.add_argument('--start', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--config', type=str, default='configs/wyckoff_events_config.json', 
                       help='Wyckoff config file')
    parser.add_argument('--dry-run', action='store_true', help='Test without writing')
    args = parser.parse_args()

    # Step 1: Load feature store
    feature_path = f'data/features_mtf/{args.asset}_1H_{args.start}_to_{args.end}.parquet'
    logger.info(f'Loading feature store: {feature_path}')
    
    df = pd.read_parquet(feature_path)
    logger.info(f'Loaded {len(df)} rows, {len(df.columns)} columns')

    # Step 2: Load config
    import json
    with open(args.config, 'r') as f:
        config = json.load(f)

    # Step 3: Detect Wyckoff events
    logger.info('Detecting Wyckoff events (this may take 1-2 minutes)...')

    df_events = detect_all_wyckoff_events(
        df=df,
        cfg=config.get('wyckoff_events', {})
    )
    
    # Step 4: Count events (exclude _confidence columns)
    event_cols = [col for col in df_events.columns
                  if col.startswith('wyckoff_')
                  and not col.endswith('_confidence')
                  and not col.endswith('_phase_abc')
                  and not col.endswith('_sequence_position')
                  and df_events[col].dtype == bool]

    logger.info(f'\nEvent Detection Summary:')
    logger.info(f'{"="*60}')

    total_events = 0
    for col in sorted(event_cols):
        count = df_events[col].sum()
        total_events += count
        event_name = col.replace('wyckoff_', '').upper()
        confidence_col = col + '_confidence'
        avg_conf = df_events[df_events[col]][confidence_col].mean() if count > 0 else 0.0
        logger.info(f'  {event_name:15s}: {count:5d} events (avg confidence: {avg_conf:.2f})')

    logger.info(f'{"="*60}')
    logger.info(f'  TOTAL          : {total_events:5d} events')
    logger.info(f'{"="*60}\n')

    # Step 5: Validate critical events
    critical_events = ['wyckoff_bc', 'wyckoff_spring_a', 'wyckoff_lps']
    for event_col in critical_events:
        if event_col in df_events.columns:
            event_mask = df_events[event_col]
            if event_mask.sum() > 0:
                sample_idx = event_mask[event_mask].head(3).index
                logger.info(f'{event_col.upper()} samples:')
                for idx in sample_idx:
                    row = df_events.loc[idx]
                    logger.info(f'  {row.name} - Price: ${row["close"]:.2f}, '
                               f'Confidence: {row.get(event_col + "_confidence", 0):.2f}')
    
    # Step 6: Write back to feature store
    if args.dry_run:
        logger.info('\nDRY RUN - Not writing to disk')
        logger.info(f'Would write {len(df_events.columns)} columns (added {len(event_cols)} event columns)')
    else:
        backup_path = feature_path.replace('.parquet', '_backup.parquet')
        logger.info(f'\nBacking up original to: {backup_path}')
        df.to_parquet(backup_path)
        
        logger.info(f'Writing updated feature store: {feature_path}')
        df_events.to_parquet(feature_path)
        
        logger.info(f'✅ Backfill complete! Added {len(event_cols)} Wyckoff event columns')


if __name__ == '__main__':
    main()
