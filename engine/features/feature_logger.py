#!/usr/bin/env python3
"""
Feature Logger for ML Ensemble Training

Central logging facility for archetype scores, baseline signals, and regime features.
Writes to disk in batched Parquet format for efficient offline ML training.

Architecture:
- Logs archetype outputs as continuous scores (not final signals)
- Logs baseline signals and confidence metrics
- Logs regime classifications and probabilities
- Buffers writes for performance (flush every N samples or on demand)

Version: 1.0
Author: System Architect
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class FeatureLogger:
    """
    Central feature logging for ML ensemble training.

    Responsibilities:
    - Log archetype scores (not final signals)
    - Log baseline signals and metrics
    - Log regime/context features
    - Sync to disk in batched Parquet files

    Usage:
        logger = FeatureLogger(output_path="data/feature_logs")
        logger.log_archetype_features(
            timestamp=pd.Timestamp.now(),
            archetype_name="liquidity_vacuum",
            score=0.85,
            confidence=0.92,
            metadata={"oi_drop": -15.2}
        )
        logger.flush()  # Write to disk
    """

    VERSION = "feature_logger@v1.0"

    def __init__(
        self,
        output_path: str,
        buffer_size: int = 1000,
        auto_flush: bool = True
    ):
        """
        Initialize feature logger.

        Args:
            output_path: Directory to write feature logs
            buffer_size: Number of samples before auto-flush
            auto_flush: If True, flush when buffer full
        """
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)

        self.buffer_size = buffer_size
        self.auto_flush = auto_flush

        # Separate buffers for different feature types
        self.archetype_buffer: List[Dict] = []
        self.baseline_buffer: List[Dict] = []
        self.regime_buffer: List[Dict] = []

        # Track flush count for file naming
        self.flush_count = 0

        logger.info(f"[FeatureLogger] Initialized v{self.VERSION}")
        logger.info(f"[FeatureLogger] Output path: {self.output_path}")
        logger.info(f"[FeatureLogger] Buffer size: {self.buffer_size}")

    def log_archetype_features(
        self,
        timestamp: pd.Timestamp,
        archetype_name: str,
        score: float,
        confidence: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log archetype output as feature (not decision).

        Args:
            timestamp: Bar timestamp
            archetype_name: Archetype identifier (e.g., "liquidity_vacuum")
            score: Archetype score [0-1] (continuous)
            confidence: Setup quality [0-1] (continuous)
            metadata: Optional additional features (dict)
        """
        record = {
            'timestamp': timestamp,
            'feature_type': 'archetype',
            'archetype_name': archetype_name,
            f'{archetype_name}_score': float(score),
            f'{archetype_name}_confidence': float(confidence)
        }

        # Add metadata fields
        if metadata:
            for key, value in metadata.items():
                record[f'{archetype_name}_{key}'] = value

        self.archetype_buffer.append(record)

        # Auto-flush if buffer full
        if self.auto_flush and len(self.archetype_buffer) >= self.buffer_size:
            self.flush()

    def log_baseline_features(
        self,
        timestamp: pd.Timestamp,
        baseline_name: str,
        signal: int,
        confidence: float,
        metrics: Optional[Dict[str, float]] = None
    ):
        """
        Log baseline signal and metrics.

        Args:
            timestamp: Bar timestamp
            baseline_name: Baseline identifier (e.g., "sma_crossover")
            signal: Baseline signal {-1: short, 0: hold, 1: long}
            confidence: Signal confidence [0-1]
            metrics: Optional metrics (e.g., SMA values, RSI)
        """
        record = {
            'timestamp': timestamp,
            'feature_type': 'baseline',
            'baseline_name': baseline_name,
            f'{baseline_name}_signal': int(signal),
            f'{baseline_name}_confidence': float(confidence)
        }

        # Add metric fields
        if metrics:
            for key, value in metrics.items():
                record[f'{baseline_name}_{key}'] = value

        self.baseline_buffer.append(record)

        # Auto-flush if buffer full
        if self.auto_flush and len(self.baseline_buffer) >= self.buffer_size:
            self.flush()

    def log_regime_features(
        self,
        timestamp: pd.Timestamp,
        regime_label: str,
        regime_proba: Dict[str, float],
        macro_features: Optional[Dict[str, float]] = None,
        wyckoff_events: Optional[Dict[str, bool]] = None
    ):
        """
        Log regime classification and context features.

        Args:
            timestamp: Bar timestamp
            regime_label: Regime classification (risk_on, neutral, risk_off, crisis)
            regime_proba: Probability distribution across regimes
            macro_features: Macro context (VIX_Z, DXY_Z, funding_Z, etc.)
            wyckoff_events: Wyckoff event flags (spring_active, lps_active, etc.)
        """
        record = {
            'timestamp': timestamp,
            'feature_type': 'regime',
            'regime_label': regime_label
        }

        # Add regime probabilities
        for regime, prob in regime_proba.items():
            record[f'regime_prob_{regime}'] = float(prob)

        # Add macro features
        if macro_features:
            for key, value in macro_features.items():
                record[key] = value

        # Add Wyckoff event flags
        if wyckoff_events:
            for key, value in wyckoff_events.items():
                record[f'wyckoff_{key}'] = bool(value)

        self.regime_buffer.append(record)

        # Auto-flush if buffer full
        if self.auto_flush and len(self.regime_buffer) >= self.buffer_size:
            self.flush()

    def flush(self):
        """
        Flush all buffers to disk as Parquet files.

        Files are named with timestamp and flush count for easy sorting.
        """
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.flush_count += 1

        # Flush archetype features
        if self.archetype_buffer:
            df = pd.DataFrame(self.archetype_buffer)
            filepath = self.output_path / f"archetype_features_{timestamp_str}_{self.flush_count:04d}.parquet"
            df.to_parquet(filepath, index=False)
            logger.debug(f"[FeatureLogger] Flushed {len(df)} archetype records to {filepath.name}")
            self.archetype_buffer = []

        # Flush baseline features
        if self.baseline_buffer:
            df = pd.DataFrame(self.baseline_buffer)
            filepath = self.output_path / f"baseline_features_{timestamp_str}_{self.flush_count:04d}.parquet"
            df.to_parquet(filepath, index=False)
            logger.debug(f"[FeatureLogger] Flushed {len(df)} baseline records to {filepath.name}")
            self.baseline_buffer = []

        # Flush regime features
        if self.regime_buffer:
            df = pd.DataFrame(self.regime_buffer)
            filepath = self.output_path / f"regime_features_{timestamp_str}_{self.flush_count:04d}.parquet"
            df.to_parquet(filepath, index=False)
            logger.debug(f"[FeatureLogger] Flushed {len(df)} regime records to {filepath.name}")
            self.regime_buffer = []

    def close(self):
        """Flush remaining data and close logger."""
        self.flush()
        logger.info(f"[FeatureLogger] Closed after {self.flush_count} flushes")

    def get_stats(self) -> Dict[str, int]:
        """Get buffer statistics."""
        return {
            'archetype_buffer_size': len(self.archetype_buffer),
            'baseline_buffer_size': len(self.baseline_buffer),
            'regime_buffer_size': len(self.regime_buffer),
            'total_flushes': self.flush_count
        }


class FeatureLoader:
    """
    Load logged features from disk for ML training.

    Responsibilities:
    - Load and merge archetype, baseline, regime features
    - Handle missing values and data validation
    - Create time-aligned feature matrix
    """

    def __init__(self, feature_log_path: str):
        """
        Initialize feature loader.

        Args:
            feature_log_path: Path to feature log directory
        """
        self.feature_log_path = Path(feature_log_path)
        if not self.feature_log_path.exists():
            raise FileNotFoundError(f"Feature log path not found: {self.feature_log_path}")

    def load_archetype_features(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load archetype features from Parquet files.

        Args:
            start_date: Optional start date filter (YYYY-MM-DD)
            end_date: Optional end date filter (YYYY-MM-DD)

        Returns:
            DataFrame with archetype features
        """
        files = sorted(self.feature_log_path.glob("archetype_features_*.parquet"))
        if not files:
            logger.warning("[FeatureLoader] No archetype feature files found")
            return pd.DataFrame()

        # Load all files
        dfs = [pd.read_parquet(f) for f in files]
        df = pd.concat(dfs, ignore_index=True)

        # Filter by date
        if start_date:
            df = df[df['timestamp'] >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df['timestamp'] <= pd.Timestamp(end_date)]

        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)

        logger.info(f"[FeatureLoader] Loaded {len(df)} archetype feature records")
        return df

    def load_baseline_features(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Load baseline features from Parquet files."""
        files = sorted(self.feature_log_path.glob("baseline_features_*.parquet"))
        if not files:
            logger.warning("[FeatureLoader] No baseline feature files found")
            return pd.DataFrame()

        dfs = [pd.read_parquet(f) for f in files]
        df = pd.concat(dfs, ignore_index=True)

        if start_date:
            df = df[df['timestamp'] >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df['timestamp'] <= pd.Timestamp(end_date)]

        df = df.sort_values('timestamp').reset_index(drop=True)

        logger.info(f"[FeatureLoader] Loaded {len(df)} baseline feature records")
        return df

    def load_regime_features(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Load regime features from Parquet files."""
        files = sorted(self.feature_log_path.glob("regime_features_*.parquet"))
        if not files:
            logger.warning("[FeatureLoader] No regime feature files found")
            return pd.DataFrame()

        dfs = [pd.read_parquet(f) for f in files]
        df = pd.concat(dfs, ignore_index=True)

        if start_date:
            df = df[df['timestamp'] >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df['timestamp'] <= pd.Timestamp(end_date)]

        df = df.sort_values('timestamp').reset_index(drop=True)

        logger.info(f"[FeatureLoader] Loaded {len(df)} regime feature records")
        return df

    def load_all_features(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load and merge all feature types into single DataFrame.

        Args:
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            Merged DataFrame with all features aligned by timestamp
        """
        # Load each feature type
        archetype_df = self.load_archetype_features(start_date, end_date)
        baseline_df = self.load_baseline_features(start_date, end_date)
        regime_df = self.load_regime_features(start_date, end_date)

        # Merge on timestamp
        df = pd.DataFrame()

        if not baseline_df.empty:
            df = baseline_df
        if not archetype_df.empty:
            if df.empty:
                df = archetype_df
            else:
                df = df.merge(archetype_df, on='timestamp', how='outer')
        if not regime_df.empty:
            if df.empty:
                df = regime_df
            else:
                df = df.merge(regime_df, on='timestamp', how='outer')

        # Sort and remove duplicates
        df = df.sort_values('timestamp').drop_duplicates(subset='timestamp').reset_index(drop=True)

        logger.info(f"[FeatureLoader] Merged features: {len(df)} rows, {len(df.columns)} columns")
        return df


# Example usage and testing
if __name__ == '__main__':
    import sys

    logging.basicConfig(level=logging.INFO)

    # Test feature logger
    logger = FeatureLogger(output_path="data/test_feature_logs", buffer_size=10)

    # Log some test features
    for i in range(25):
        ts = pd.Timestamp('2024-01-01') + pd.Timedelta(hours=i)

        # Archetype features
        logger.log_archetype_features(
            timestamp=ts,
            archetype_name='liquidity_vacuum',
            score=0.5 + i * 0.01,
            confidence=0.8,
            metadata={'oi_drop': -10.0 - i}
        )

        # Baseline features
        logger.log_baseline_features(
            timestamp=ts,
            baseline_name='sma_crossover',
            signal=1 if i % 2 == 0 else 0,
            confidence=0.9,
            metrics={'sma_50': 50000 + i * 100, 'sma_200': 48000}
        )

        # Regime features
        logger.log_regime_features(
            timestamp=ts,
            regime_label='risk_on',
            regime_proba={'risk_on': 0.7, 'neutral': 0.2, 'risk_off': 0.1},
            macro_features={'vix_z': -0.5, 'dxy_z': 0.3}
        )

    # Close and flush
    logger.close()

    print(f"\n✅ Feature logging test complete")
    print(f"   Flushes: {logger.flush_count}")

    # Test feature loader
    print("\n📂 Testing feature loader...")
    loader = FeatureLoader("data/test_feature_logs")
    df = loader.load_all_features()

    print(f"\n✅ Feature loading test complete")
    print(f"   Rows: {len(df)}")
    print(f"   Columns: {list(df.columns)}")
    print(f"\nSample data:\n{df.head()}")
