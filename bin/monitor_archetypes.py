#!/usr/bin/env python3
"""
Archetype Monitoring Script

Monitors S4 (Funding Divergence), S5 (Long Squeeze), and S1 V2 (Liquidity Vacuum)
archetype conditions in real-time or on historical data.

Usage:
    # Monitor live conditions
    python3 bin/monitor_archetypes.py --mode live

    # Analyze historical period
    python3 bin/monitor_archetypes.py --mode historical --start 2024-01-01 --end 2024-12-31

    # Alert mode (check if conditions are met NOW)
    python3 bin/monitor_archetypes.py --mode alert --archetypes S4,S5
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse
import json
import logging
from typing import Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ArchetypeMonitor:
    """Monitor archetype signal conditions"""

    def __init__(self, feature_store_path: str = "data/feature_store.parquet"):
        self.feature_store_path = feature_store_path
        self.df = None

    def load_data(self, start_date: Optional[str] = None, end_date: Optional[str] = None):
        """Load feature store data"""
        logger.info(f"Loading feature store from {self.feature_store_path}")
        self.df = pd.read_parquet(self.feature_store_path)

        if start_date:
            self.df = self.df[self.df.index >= start_date]
        if end_date:
            self.df = self.df[self.df.index <= end_date]

        logger.info(f"Loaded {len(self.df)} bars")

    def check_s4_conditions(self, row: pd.Series) -> Dict:
        """Check if S4 (Funding Divergence) conditions are met"""
        funding_z = row.get('funding_z', 0)
        liquidity = row.get('liquidity_score', 0.5)
        close = row.get('close', 0)
        sma_20 = row.get('sma_20', close)
        resilience = close / sma_20 if sma_20 > 0 else 0.5

        conditions = {
            'funding_z': funding_z,
            'funding_extreme_negative': funding_z < -1.976,
            'resilience': resilience,
            'resilience_sufficient': resilience > 0.555,
            'liquidity': liquidity,
            'liquidity_thin': liquidity < 0.348
        }

        conditions['signal_active'] = (
            conditions['funding_extreme_negative'] and
            conditions['resilience_sufficient'] and
            conditions['liquidity_thin']
        )

        return conditions

    def check_s5_conditions(self, row: pd.Series) -> Dict:
        """Check if S5 (Long Squeeze) conditions are met"""
        funding_z = row.get('funding_z', 0)
        rsi = row.get('rsi', 50)
        liquidity = row.get('liquidity_score', 0.5)

        conditions = {
            'funding_z': funding_z,
            'funding_extreme_positive': funding_z > 1.5,
            'rsi': rsi,
            'rsi_overbought': rsi > 70,
            'liquidity': liquidity,
            'liquidity_thin': liquidity < 0.20
        }

        conditions['signal_active'] = (
            conditions['funding_extreme_positive'] and
            conditions['rsi_overbought'] and
            conditions['liquidity_thin']
        )

        return conditions

    def alert_current_conditions(self, archetypes: List[str] = None) -> Dict:
        """Check if any archetype conditions are met RIGHT NOW"""
        if self.df is None or len(self.df) == 0:
            return {'error': 'No data loaded'}

        latest = self.df.iloc[-1]
        timestamp = self.df.index[-1]

        if archetypes is None:
            archetypes = ['S4', 'S5', 'S1']

        alerts = {
            'timestamp': timestamp,
            'active_signals': [],
            'conditions': {}
        }

        if 'S4' in archetypes:
            s4_cond = self.check_s4_conditions(latest)
            alerts['conditions']['S4'] = s4_cond
            if s4_cond['signal_active']:
                alerts['active_signals'].append('S4_FUNDING_DIVERGENCE')

        if 'S5' in archetypes:
            s5_cond = self.check_s5_conditions(latest)
            alerts['conditions']['S5'] = s5_cond
            if s5_cond['signal_active']:
                alerts['active_signals'].append('S5_LONG_SQUEEZE')

        alerts['alert_status'] = 'ACTIVE' if len(alerts['active_signals']) > 0 else 'CLEAR'

        return alerts


def main():
    parser = argparse.ArgumentParser(description='Monitor archetype signal conditions')
    parser.add_argument('--mode', choices=['alert', 'historical'], default='alert')
    parser.add_argument('--archetypes', type=str, default='S4,S5')
    parser.add_argument('--feature-store', type=str, default='data/feature_store.parquet')

    args = parser.parse_args()
    archetypes = args.archetypes.upper().split(',')

    monitor = ArchetypeMonitor(feature_store_path=args.feature_store)

    try:
        monitor.load_data()
        alerts = monitor.alert_current_conditions(archetypes=archetypes)

        print("\n" + "=" * 80)
        print("ARCHETYPE ALERT STATUS")
        print("=" * 80)
        print(f"\nTimestamp: {alerts['timestamp']}")
        print(f"Alert Status: {alerts['alert_status']}")
        print(f"\nActive Signals: {', '.join(alerts['active_signals']) if alerts['active_signals'] else 'NONE'}")
        print("\n" + "-" * 80)

        for arch, conditions in alerts['conditions'].items():
            print(f"\n{arch}:")
            print(f"  Signal Active: {conditions['signal_active']}")
            for key, value in conditions.items():
                if key != 'signal_active':
                    if isinstance(value, float):
                        print(f"    {key}: {value:.3f}")
                    else:
                        print(f"    {key}: {value}")

        print("\n" + "=" * 80 + "\n")

    except FileNotFoundError:
        logger.error(f"Feature store not found: {args.feature_store}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
