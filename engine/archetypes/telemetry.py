#!/usr/bin/env python3
"""
Archetype Telemetry System

Tracks archetype detection counts, distributions, and per-archetype performance
metrics for monitoring and analysis.
"""

from collections import defaultdict
from typing import Dict, Optional
import logging


class ArchetypeTelemetry:
    """
    Tracks archetype detection statistics.

    Provides counters for each archetype and summary reports.
    """

    def __init__(self, names: Optional[list] = None):
        """
        Initialize telemetry with archetype names.

        Args:
            names: List of archetype names to track
        """
        if names is None:
            names = [
                'trap_reversal',
                'order_block_retest',
                'fvg_continuation',
                'failed_continuation',
                'liquidity_compression',
                'expansion_exhaustion',
                'reaccumulation',
                'trap_within_trend',
                'wick_trap',
                'volume_exhaustion',
                'ratio_coil_break'
            ]

        self.names = names
        self.counts = defaultdict(int)
        self.total_checks = 0
        self.total_matches = 0

        # Optional: Per-archetype PnL tracking (for future use)
        self.pnl_by_archetype = defaultdict(list)
        self.pf_by_archetype = defaultdict(list)

    def count(self, name: str):
        """
        Increment counter for archetype.

        Args:
            name: Archetype name
        """
        if name:
            self.counts[name] += 1
            self.total_matches += 1

    def check(self):
        """Increment total check counter."""
        self.total_checks += 1

    def record_trade(self, name: str, pnl: float, profit_factor: Optional[float] = None):
        """
        Record trade outcome for archetype (optional future use).

        Args:
            name: Archetype name
            pnl: Trade PnL
            profit_factor: Trade profit factor
        """
        if name:
            self.pnl_by_archetype[name].append(pnl)
            if profit_factor is not None:
                self.pf_by_archetype[name].append(profit_factor)

    def summary(self) -> Dict:
        """
        Get summary statistics.

        Returns:
            Dict with counts, percentages, and optional PnL stats
        """
        summary = {
            'total_checks': self.total_checks,
            'total_matches': self.total_matches,
            'match_rate_pct': (
                100.0 * self.total_matches / self.total_checks
                if self.total_checks > 0 else 0.0
            ),
            'counts': dict(self.counts),
            'percentages': {}
        }

        # Calculate percentages
        if self.total_matches > 0:
            for name in self.names:
                count = self.counts.get(name, 0)
                pct = 100.0 * count / self.total_matches
                summary['percentages'][name] = pct

        # Add PnL stats if available
        if self.pnl_by_archetype:
            summary['pnl_stats'] = {}
            for name in self.names:
                if name in self.pnl_by_archetype:
                    pnls = self.pnl_by_archetype[name]
                    summary['pnl_stats'][name] = {
                        'trades': len(pnls),
                        'total_pnl': sum(pnls),
                        'avg_pnl': sum(pnls) / len(pnls) if pnls else 0.0
                    }

        return summary

    def dump(self, logger: Optional[logging.Logger] = None):
        """
        Pretty print telemetry summary.

        Args:
            logger: Optional logger (uses print if None)
        """
        summary = self.summary()

        def log(msg):
            if logger:
                logger.info(msg)
            else:
                print(msg)

        log("=" * 60)
        log("ARCHETYPE TELEMETRY")
        log("=" * 60)

        log(f"Total Checks: {summary['total_checks']}")
        log(f"Total Matches: {summary['total_matches']}")
        log(f"Match Rate: {summary['match_rate_pct']:.1f}%")
        log("")

        log("Archetype Distribution:")
        log("-" * 60)

        counts = summary['counts']
        percentages = summary['percentages']

        # Sort by count descending
        sorted_archetypes = sorted(
            self.names,
            key=lambda n: counts.get(n, 0),
            reverse=True
        )

        for name in sorted_archetypes:
            count = counts.get(name, 0)
            pct = percentages.get(name, 0.0)
            if count > 0:
                log(f"  {name:25s}: {count:4d} ({pct:5.1f}%)")

        # Print PnL stats if available
        if 'pnl_stats' in summary and summary['pnl_stats']:
            log("")
            log("PnL by Archetype:")
            log("-" * 60)
            for name in sorted_archetypes:
                if name in summary['pnl_stats']:
                    stats = summary['pnl_stats'][name]
                    log(f"  {name:25s}: {stats['trades']:2d} trades, "
                        f"${stats['total_pnl']:8.2f} total, "
                        f"${stats['avg_pnl']:7.2f} avg")

        log("=" * 60)

    def reset(self):
        """Reset all counters."""
        self.counts.clear()
        self.pnl_by_archetype.clear()
        self.pf_by_archetype.clear()
        self.total_checks = 0
        self.total_matches = 0
