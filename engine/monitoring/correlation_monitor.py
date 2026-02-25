"""
Correlation Monitoring System for Archetype Diversification

This module tracks rolling correlation across archetypes to ensure true diversification
and detect redundant strategies. Based on institutional standards:
- Target: Correlation < 0.5 for true diversification
- Alert: Correlation > 0.7 (diversification breakdown)
- Critical: Correlation > 0.8 (redundant strategies)

Key Features:
1. Rolling correlation windows (30/60/90 days)
2. Correlation regime detection (stress vs normal)
3. Redundant strategy identification
4. Integration with portfolio allocator for conflict resolution
5. Heatmap visualization and reporting

Author: Claude Code
Date: 2026-02-04
Spec: Institutional diversification monitoring standards
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Set
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

# Suppress matplotlib font warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')


@dataclass
class CorrelationAlert:
    """Alert for correlation breakdown or redundant strategies."""
    timestamp: datetime
    archetype_pair: Tuple[str, str]
    correlation: float
    window_days: int
    alert_type: str  # 'breakdown' (>0.7) or 'redundant' (>0.8)
    severity: str  # 'warning' (0.7-0.8) or 'critical' (>0.8)
    recommendation: str

    def __str__(self):
        return (
            f"[{self.severity.upper()}] {self.alert_type} @ {self.timestamp}: "
            f"{self.archetype_pair[0]} <-> {self.archetype_pair[1]} = {self.correlation:.3f} "
            f"(window={self.window_days}d) | {self.recommendation}"
        )


@dataclass
class CorrelationMetrics:
    """Summary metrics for correlation health."""
    timestamp: datetime
    window_days: int
    mean_correlation: float
    median_correlation: float
    max_correlation: float
    num_high_corr_pairs: int  # Pairs with corr > 0.7
    num_redundant_pairs: int  # Pairs with corr > 0.8
    diversification_score: float  # 1.0 - mean_correlation (higher is better)
    pairs_above_target: int  # Pairs with corr > 0.5 (institutional target)

    def is_healthy(self) -> bool:
        """Check if diversification is healthy (institutional standard)."""
        return (
            self.mean_correlation < 0.5 and
            self.num_high_corr_pairs == 0 and
            self.num_redundant_pairs == 0
        )

    def __str__(self):
        health = "HEALTHY" if self.is_healthy() else "UNHEALTHY"
        return (
            f"[{health}] Correlation Health @ {self.timestamp} (window={self.window_days}d)\n"
            f"  Mean: {self.mean_correlation:.3f} | Median: {self.median_correlation:.3f} | Max: {self.max_correlation:.3f}\n"
            f"  Diversification Score: {self.diversification_score:.3f}\n"
            f"  High Corr Pairs (>0.7): {self.num_high_corr_pairs}\n"
            f"  Redundant Pairs (>0.8): {self.num_redundant_pairs}\n"
            f"  Pairs Above Target (>0.5): {self.pairs_above_target}"
        )


class CorrelationMonitor:
    """
    Monitor archetype correlation over time to ensure diversification.

    Tracks rolling correlation across multiple windows and detects:
    - Correlation spikes during market stress
    - Redundant strategies (consistently high correlation)
    - Diversification breakdown (corr > 0.7)
    - Regime-dependent correlation patterns

    Usage:
        monitor = CorrelationMonitor(window_days=[30, 60, 90])

        # Update with new returns
        monitor.update(timestamp, archetype_returns)

        # Check for alerts
        alerts = monitor.get_alerts(min_severity='warning')

        # Generate report
        report = monitor.generate_report()

        # Get correlation for portfolio allocator
        corr_matrix = monitor.get_correlation_matrix(window_days=60)
    """

    def __init__(
        self,
        window_days: List[int] = [30, 60, 90],
        alert_threshold: float = 0.7,
        redundant_threshold: float = 0.8,
        target_correlation: float = 0.5,
        min_observations: int = 20,
        output_dir: Optional[Path] = None
    ):
        """
        Initialize correlation monitor.

        Args:
            window_days: List of rolling window sizes (in days)
            alert_threshold: Correlation threshold for alerts (default: 0.7)
            redundant_threshold: Correlation threshold for redundancy (default: 0.8)
            target_correlation: Institutional target for diversification (default: 0.5)
            min_observations: Minimum observations required for correlation calculation
            output_dir: Directory for saving reports and visualizations
        """
        self.window_days = sorted(window_days)
        self.alert_threshold = alert_threshold
        self.redundant_threshold = redundant_threshold
        self.target_correlation = target_correlation
        self.min_observations = min_observations
        self.output_dir = Path(output_dir) if output_dir else Path.cwd() / "results" / "correlation_monitoring"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Data storage: timestamp -> archetype -> return
        self.returns_history: List[Dict[str, float]] = []
        self.timestamps: List[datetime] = []

        # Correlation history: (window, timestamp) -> correlation_matrix
        self.correlation_history: Dict[Tuple[int, datetime], pd.DataFrame] = {}

        # Alert history
        self.alerts: List[CorrelationAlert] = []

        # Metrics history
        self.metrics_history: List[CorrelationMetrics] = []

        # Archetype registry (populated as data comes in)
        self.archetypes: Set[str] = set()

        logger.info(
            f"CorrelationMonitor initialized: "
            f"windows={window_days}, alert_threshold={alert_threshold:.2f}, "
            f"redundant_threshold={redundant_threshold:.2f}, target={target_correlation:.2f}"
        )

    def update(self, timestamp: datetime, archetype_returns: Dict[str, float]) -> None:
        """
        Update monitor with new archetype returns.

        Args:
            timestamp: Current timestamp
            archetype_returns: Dictionary mapping archetype name -> return (%)
                Example: {'spring': 2.5, 'wick_trap': -1.2, 'whipsaw': 0.8}
        """
        # Store returns
        self.returns_history.append(archetype_returns)
        self.timestamps.append(timestamp)

        # Update archetype registry
        self.archetypes.update(archetype_returns.keys())

        # Calculate correlation for all windows if we have enough data
        for window in self.window_days:
            if len(self.returns_history) >= self.min_observations:
                self._calculate_rolling_correlation(window)

        logger.debug(
            f"Updated correlation monitor @ {timestamp}: "
            f"{len(archetype_returns)} archetypes, "
            f"total_observations={len(self.returns_history)}"
        )

    def _calculate_rolling_correlation(self, window_days: int) -> None:
        """
        Calculate rolling correlation for a given window.

        Args:
            window_days: Window size in days
        """
        if len(self.returns_history) < self.min_observations:
            logger.warning(
                f"Insufficient data for correlation calculation: "
                f"{len(self.returns_history)} < {self.min_observations}"
            )
            return

        # Convert to DataFrame for easier manipulation
        df_returns = pd.DataFrame(self.returns_history, index=self.timestamps)

        # Calculate rolling correlation
        # Use last window_days of data (or all data if less than window)
        end_idx = len(df_returns)
        start_idx = max(0, end_idx - window_days)
        window_data = df_returns.iloc[start_idx:end_idx]

        # Need at least min_observations for valid correlation
        if len(window_data) < self.min_observations:
            return

        # Calculate correlation matrix
        corr_matrix = window_data.corr()

        # Store in history
        current_time = self.timestamps[-1]
        self.correlation_history[(window_days, current_time)] = corr_matrix

        # Check for alerts
        self._check_correlation_alerts(corr_matrix, window_days, current_time)

        # Calculate and store metrics
        metrics = self._calculate_metrics(corr_matrix, window_days, current_time)
        self.metrics_history.append(metrics)

    def _check_correlation_alerts(
        self,
        corr_matrix: pd.DataFrame,
        window_days: int,
        timestamp: datetime
    ) -> None:
        """
        Check correlation matrix for alerts (breakdown or redundancy).

        Args:
            corr_matrix: Correlation matrix
            window_days: Window size used for correlation
            timestamp: Current timestamp
        """
        archetypes = list(corr_matrix.columns)

        for i in range(len(archetypes)):
            for j in range(i + 1, len(archetypes)):
                arch1 = archetypes[i]
                arch2 = archetypes[j]
                correlation = corr_matrix.loc[arch1, arch2]

                # Skip NaN correlations
                if pd.isna(correlation):
                    continue

                # Check for redundant strategies (>0.8)
                if abs(correlation) >= self.redundant_threshold:
                    alert = CorrelationAlert(
                        timestamp=timestamp,
                        archetype_pair=(arch1, arch2),
                        correlation=correlation,
                        window_days=window_days,
                        alert_type='redundant',
                        severity='critical',
                        recommendation=f"Consider deactivating {arch2} (redundant with {arch1})"
                    )
                    self.alerts.append(alert)
                    logger.warning(str(alert))

                # Check for diversification breakdown (>0.7)
                elif abs(correlation) >= self.alert_threshold:
                    alert = CorrelationAlert(
                        timestamp=timestamp,
                        archetype_pair=(arch1, arch2),
                        correlation=correlation,
                        window_days=window_days,
                        alert_type='breakdown',
                        severity='warning',
                        recommendation=f"Monitor {arch1}-{arch2} pair for regime-dependent correlation"
                    )
                    self.alerts.append(alert)
                    logger.warning(str(alert))

    def _calculate_metrics(
        self,
        corr_matrix: pd.DataFrame,
        window_days: int,
        timestamp: datetime
    ) -> CorrelationMetrics:
        """
        Calculate summary metrics from correlation matrix.

        Args:
            corr_matrix: Correlation matrix
            window_days: Window size used
            timestamp: Current timestamp

        Returns:
            CorrelationMetrics object
        """
        # Extract upper triangle (avoid diagonal and duplicates)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        upper_triangle = corr_matrix.where(mask)
        correlations = upper_triangle.values.flatten()
        correlations = correlations[~np.isnan(correlations)]

        # Calculate metrics
        mean_corr = np.mean(np.abs(correlations))
        median_corr = np.median(np.abs(correlations))
        max_corr = np.max(np.abs(correlations))

        num_high = np.sum(np.abs(correlations) > self.alert_threshold)
        num_redundant = np.sum(np.abs(correlations) > self.redundant_threshold)
        num_above_target = np.sum(np.abs(correlations) > self.target_correlation)

        diversification_score = 1.0 - mean_corr

        return CorrelationMetrics(
            timestamp=timestamp,
            window_days=window_days,
            mean_correlation=mean_corr,
            median_correlation=median_corr,
            max_correlation=max_corr,
            num_high_corr_pairs=int(num_high),
            num_redundant_pairs=int(num_redundant),
            diversification_score=diversification_score,
            pairs_above_target=int(num_above_target)
        )

    def get_correlation_matrix(
        self,
        window_days: Optional[int] = None,
        timestamp: Optional[datetime] = None
    ) -> Optional[pd.DataFrame]:
        """
        Get correlation matrix for a specific window and timestamp.

        Args:
            window_days: Window size (uses largest window if None)
            timestamp: Timestamp (uses latest if None)

        Returns:
            Correlation matrix or None if not available
        """
        if not self.correlation_history:
            return None

        # Use defaults if not specified
        if window_days is None:
            window_days = self.window_days[-1]  # Largest window
        if timestamp is None:
            timestamp = self.timestamps[-1]  # Latest

        # Find closest timestamp
        key = (window_days, timestamp)
        if key in self.correlation_history:
            return self.correlation_history[key]

        # If exact match not found, get latest for that window
        matching_keys = [k for k in self.correlation_history.keys() if k[0] == window_days]
        if matching_keys:
            latest_key = max(matching_keys, key=lambda k: k[1])
            return self.correlation_history[latest_key]

        return None

    def get_alerts(
        self,
        min_severity: str = 'warning',
        since: Optional[datetime] = None,
        window_days: Optional[int] = None
    ) -> List[CorrelationAlert]:
        """
        Get correlation alerts.

        Args:
            min_severity: Minimum severity ('warning' or 'critical')
            since: Only return alerts after this timestamp
            window_days: Filter by specific window size

        Returns:
            List of alerts matching criteria
        """
        filtered = self.alerts

        # Filter by severity
        if min_severity == 'critical':
            filtered = [a for a in filtered if a.severity == 'critical']

        # Filter by timestamp
        if since:
            filtered = [a for a in filtered if a.timestamp >= since]

        # Filter by window
        if window_days:
            filtered = [a for a in filtered if a.window_days == window_days]

        return filtered

    def get_redundant_pairs(self, threshold: Optional[float] = None) -> List[Tuple[str, str, float]]:
        """
        Get pairs of archetypes with high correlation (redundancy).

        Args:
            threshold: Correlation threshold (uses redundant_threshold if None)

        Returns:
            List of (archetype1, archetype2, correlation) tuples
        """
        if threshold is None:
            threshold = self.redundant_threshold

        # Get latest correlation matrix (largest window)
        corr_matrix = self.get_correlation_matrix()
        if corr_matrix is None:
            return []

        pairs = []
        archetypes = list(corr_matrix.columns)

        for i in range(len(archetypes)):
            for j in range(i + 1, len(archetypes)):
                arch1 = archetypes[i]
                arch2 = archetypes[j]
                correlation = corr_matrix.loc[arch1, arch2]

                if not pd.isna(correlation) and abs(correlation) >= threshold:
                    pairs.append((arch1, arch2, correlation))

        # Sort by correlation (descending)
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)

        return pairs

    def get_diversification_metrics(
        self,
        window_days: Optional[int] = None
    ) -> Optional[CorrelationMetrics]:
        """
        Get latest diversification metrics for a window.

        Args:
            window_days: Window size (uses largest if None)

        Returns:
            CorrelationMetrics or None
        """
        if not self.metrics_history:
            return None

        # Filter by window
        if window_days is None:
            window_days = self.window_days[-1]

        matching = [m for m in self.metrics_history if m.window_days == window_days]
        if matching:
            return matching[-1]  # Latest

        return None

    def plot_correlation_heatmap(
        self,
        window_days: Optional[int] = None,
        save: bool = True,
        show: bool = False
    ) -> Optional[Path]:
        """
        Generate correlation heatmap visualization.

        Args:
            window_days: Window size (uses largest if None)
            save: Save to file
            show: Display plot

        Returns:
            Path to saved file or None
        """
        corr_matrix = self.get_correlation_matrix(window_days=window_days)
        if corr_matrix is None:
            logger.warning("No correlation data available for heatmap")
            return None

        if window_days is None:
            window_days = self.window_days[-1]

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))

        # Plot heatmap
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt='.2f',
            cmap='RdYlGn_r',  # Red for high correlation, green for low
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0.5,
            cbar_kws={'label': 'Correlation'},
            ax=ax
        )

        # Add threshold lines
        n = len(corr_matrix)

        # Add colored boxes for thresholds
        # Target line (0.5)
        ax.axhline(y=0, color='blue', linestyle='--', linewidth=1, alpha=0.3)
        ax.axvline(x=0, color='blue', linestyle='--', linewidth=1, alpha=0.3)

        # Alert line (0.7)
        ax.axhline(y=n, color='orange', linestyle='--', linewidth=1.5, alpha=0.5)
        ax.axvline(x=n, color='orange', linestyle='--', linewidth=1.5, alpha=0.5)

        # Redundant line (0.8)
        ax.axhline(y=n, color='red', linestyle='--', linewidth=2, alpha=0.5)
        ax.axvline(x=n, color='red', linestyle='--', linewidth=2, alpha=0.5)

        ax.set_title(
            f'Archetype Correlation Matrix (Rolling {window_days}-Day Window)\n'
            f'Target: <0.5 (blue) | Alert: >0.7 (orange) | Redundant: >0.8 (red)',
            fontsize=14,
            pad=20
        )

        plt.tight_layout()

        # Save
        output_path = None
        if save:
            timestamp_str = self.timestamps[-1].strftime('%Y%m%d_%H%M%S')
            filename = f'correlation_heatmap_{window_days}d_{timestamp_str}.png'
            output_path = self.output_dir / filename
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved correlation heatmap to {output_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return output_path

    def plot_correlation_timeseries(
        self,
        archetype_pairs: Optional[List[Tuple[str, str]]] = None,
        window_days: Optional[int] = None,
        save: bool = True,
        show: bool = False
    ) -> Optional[Path]:
        """
        Plot correlation over time for specific archetype pairs.

        Args:
            archetype_pairs: List of (arch1, arch2) tuples to plot (plots top 5 if None)
            window_days: Window size (uses largest if None)
            save: Save to file
            show: Display plot

        Returns:
            Path to saved file or None
        """
        if window_days is None:
            window_days = self.window_days[-1]

        # Get all correlation matrices for this window
        matrices = [
            (ts, matrix) for (win, ts), matrix in self.correlation_history.items()
            if win == window_days
        ]

        if not matrices:
            logger.warning("No correlation history for time series plot")
            return None

        matrices.sort(key=lambda x: x[0])  # Sort by timestamp

        # Get pairs to plot
        if archetype_pairs is None:
            # Use top 5 most correlated pairs from latest matrix
            redundant = self.get_redundant_pairs(threshold=0.5)
            archetype_pairs = [(p[0], p[1]) for p in redundant[:5]]

        if not archetype_pairs:
            logger.warning("No archetype pairs to plot")
            return None

        # Build time series for each pair
        fig, ax = plt.subplots(figsize=(14, 8))

        for arch1, arch2 in archetype_pairs:
            timestamps = []
            correlations = []

            for ts, matrix in matrices:
                if arch1 in matrix.columns and arch2 in matrix.columns:
                    corr = matrix.loc[arch1, arch2]
                    if not pd.isna(corr):
                        timestamps.append(ts)
                        correlations.append(corr)

            if timestamps:
                ax.plot(timestamps, correlations, marker='o', label=f'{arch1}-{arch2}')

        # Add threshold lines
        ax.axhline(
            y=self.target_correlation,
            color='blue',
            linestyle='--',
            linewidth=1,
            alpha=0.5,
            label=f'Target ({self.target_correlation})'
        )
        ax.axhline(
            y=self.alert_threshold,
            color='orange',
            linestyle='--',
            linewidth=1.5,
            alpha=0.7,
            label=f'Alert ({self.alert_threshold})'
        )
        ax.axhline(
            y=self.redundant_threshold,
            color='red',
            linestyle='--',
            linewidth=2,
            alpha=0.7,
            label=f'Redundant ({self.redundant_threshold})'
        )

        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Correlation', fontsize=12)
        ax.set_title(
            f'Archetype Correlation Over Time ({window_days}-Day Rolling Window)',
            fontsize=14,
            pad=20
        )
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save
        output_path = None
        if save:
            timestamp_str = self.timestamps[-1].strftime('%Y%m%d_%H%M%S')
            filename = f'correlation_timeseries_{window_days}d_{timestamp_str}.png'
            output_path = self.output_dir / filename
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved correlation time series to {output_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return output_path

    def generate_report(self, save: bool = True) -> str:
        """
        Generate comprehensive correlation monitoring report.

        Args:
            save: Save report to file

        Returns:
            Report text
        """
        lines = []
        lines.append("=" * 80)
        lines.append("ARCHETYPE CORRELATION MONITORING REPORT")
        lines.append("=" * 80)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Archetypes Tracked: {len(self.archetypes)}")
        lines.append(f"Observations: {len(self.returns_history)}")
        lines.append(f"Windows: {self.window_days} days")
        lines.append("")

        # Summary for each window
        for window in self.window_days:
            metrics = self.get_diversification_metrics(window_days=window)
            if metrics:
                lines.append("-" * 80)
                lines.append(f"WINDOW: {window} DAYS")
                lines.append("-" * 80)
                lines.append(str(metrics))
                lines.append("")

        # Alerts summary
        lines.append("-" * 80)
        lines.append("ALERTS SUMMARY")
        lines.append("-" * 80)

        critical_alerts = self.get_alerts(min_severity='critical')
        warning_alerts = [a for a in self.get_alerts(min_severity='warning') if a.severity != 'critical']

        lines.append(f"Critical Alerts (Redundant Strategies): {len(critical_alerts)}")
        lines.append(f"Warning Alerts (Diversification Breakdown): {len(warning_alerts)}")
        lines.append("")

        # Recent critical alerts
        if critical_alerts:
            lines.append("Recent Critical Alerts (Last 5):")
            for alert in critical_alerts[-5:]:
                lines.append(f"  {alert}")
            lines.append("")

        # Recent warnings
        if warning_alerts:
            lines.append("Recent Warning Alerts (Last 5):")
            for alert in warning_alerts[-5:]:
                lines.append(f"  {alert}")
            lines.append("")

        # Redundant pairs
        lines.append("-" * 80)
        lines.append("REDUNDANT ARCHETYPE PAIRS (Correlation > 0.8)")
        lines.append("-" * 80)
        redundant = self.get_redundant_pairs()
        if redundant:
            for arch1, arch2, corr in redundant:
                lines.append(f"  {arch1} <-> {arch2}: {corr:.3f}")
        else:
            lines.append("  None detected (healthy diversification)")
        lines.append("")

        # High correlation pairs
        lines.append("-" * 80)
        lines.append("HIGH CORRELATION PAIRS (Correlation > 0.7)")
        lines.append("-" * 80)
        high_corr = self.get_redundant_pairs(threshold=0.7)
        if high_corr:
            for arch1, arch2, corr in high_corr:
                lines.append(f"  {arch1} <-> {arch2}: {corr:.3f}")
        else:
            lines.append("  None detected (healthy diversification)")
        lines.append("")

        # Recommendations
        lines.append("-" * 80)
        lines.append("RECOMMENDATIONS")
        lines.append("-" * 80)

        latest_metrics = self.get_diversification_metrics()
        if latest_metrics:
            if latest_metrics.is_healthy():
                lines.append("  ✓ Diversification is HEALTHY")
                lines.append(f"  ✓ Mean correlation ({latest_metrics.mean_correlation:.3f}) below target (0.5)")
                lines.append("  ✓ No redundant strategies detected")
            else:
                lines.append("  ✗ Diversification needs attention:")

                if latest_metrics.mean_correlation >= 0.5:
                    lines.append(
                        f"    - Mean correlation ({latest_metrics.mean_correlation:.3f}) "
                        f"exceeds institutional target (0.5)"
                    )

                if latest_metrics.num_redundant_pairs > 0:
                    lines.append(
                        f"    - {latest_metrics.num_redundant_pairs} redundant strategy pairs detected"
                    )
                    lines.append("    - RECOMMENDATION: Deactivate lower-performing archetype in each pair")

                if latest_metrics.num_high_corr_pairs > 0:
                    lines.append(
                        f"    - {latest_metrics.num_high_corr_pairs} high-correlation pairs (>0.7)"
                    )
                    lines.append("    - RECOMMENDATION: Monitor for regime-dependent correlation")

        lines.append("")
        lines.append("=" * 80)

        report = "\n".join(lines)

        # Save
        if save:
            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'correlation_report_{timestamp_str}.txt'
            output_path = self.output_dir / filename
            output_path.write_text(report)
            logger.info(f"Saved correlation report to {output_path}")

        return report

    def get_conflict_resolution_data(self) -> Dict[str, List[str]]:
        """
        Get data for portfolio allocator conflict resolution.

        Returns dictionary mapping each archetype to list of conflicting archetypes
        (those with correlation > alert_threshold).

        This can be used by the portfolio allocator to:
        1. Avoid taking highly-correlated positions simultaneously
        2. Reduce allocation to archetypes in conflict
        3. Prioritize archetypes with lower correlation to existing positions

        Returns:
            Dictionary mapping archetype -> list of conflicting archetypes
        """
        corr_matrix = self.get_correlation_matrix()
        if corr_matrix is None:
            return {}

        conflicts = {}
        archetypes = list(corr_matrix.columns)

        for arch in archetypes:
            conflicting = []
            for other_arch in archetypes:
                if arch != other_arch:
                    corr = corr_matrix.loc[arch, other_arch]
                    if not pd.isna(corr) and abs(corr) >= self.alert_threshold:
                        conflicting.append(other_arch)

            if conflicting:
                conflicts[arch] = conflicting

        return conflicts

    def save_state(self, filepath: Optional[Path] = None) -> Path:
        """
        Save monitor state to disk for persistence.

        Args:
            filepath: Path to save state (uses default if None)

        Returns:
            Path to saved file
        """
        if filepath is None:
            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = self.output_dir / f'correlation_monitor_state_{timestamp_str}.pkl'

        # Package state
        state = {
            'returns_history': self.returns_history,
            'timestamps': self.timestamps,
            'correlation_history': self.correlation_history,
            'alerts': self.alerts,
            'metrics_history': self.metrics_history,
            'archetypes': self.archetypes,
            'config': {
                'window_days': self.window_days,
                'alert_threshold': self.alert_threshold,
                'redundant_threshold': self.redundant_threshold,
                'target_correlation': self.target_correlation,
                'min_observations': self.min_observations
            }
        }

        # Save with pickle
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

        logger.info(f"Saved monitor state to {filepath}")
        return filepath

    @classmethod
    def load_state(cls, filepath: Path) -> 'CorrelationMonitor':
        """
        Load monitor state from disk.

        Args:
            filepath: Path to state file

        Returns:
            CorrelationMonitor instance
        """
        import pickle
        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        # Create monitor with saved config
        config = state['config']
        monitor = cls(
            window_days=config['window_days'],
            alert_threshold=config['alert_threshold'],
            redundant_threshold=config['redundant_threshold'],
            target_correlation=config['target_correlation'],
            min_observations=config['min_observations']
        )

        # Restore state
        monitor.returns_history = state['returns_history']
        monitor.timestamps = state['timestamps']
        monitor.correlation_history = state['correlation_history']
        monitor.alerts = state['alerts']
        monitor.metrics_history = state['metrics_history']
        monitor.archetypes = state['archetypes']

        logger.info(f"Loaded monitor state from {filepath}")
        return monitor
