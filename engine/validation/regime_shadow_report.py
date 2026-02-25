"""
Shadow Regime Report - Track regime quality metrics without affecting execution.

This runs the live regime model in parallel and reports on its behavior,
but doesn't let it control trades during Phase 2 validation.

Purpose: Quantify regime model issues for future retraining.
"""

import pandas as pd
from typing import Dict
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class RegimeShadowTracker:
    """Track regime model behavior for quality assessment."""

    def __init__(self):
        """Initialize shadow tracker."""
        self.regime_history = []
        self.transition_count = 0
        self.last_regime = None
        self.override_counts = defaultdict(int)
        self.hysteresis_blocks = 0

        # Distribution counters
        self.regime_bar_counts = defaultdict(int)

        # Duration tracking
        self.regime_start_times = {}
        self.regime_durations = defaultdict(list)

    def record(self, timestamp: pd.Timestamp, regime_result: Dict):
        """
        Record regime classification for shadow analysis.

        Args:
            timestamp: Current timestamp
            regime_result: Dict from RegimeService.get_regime()
        """
        regime = regime_result['regime_label']

        # Track distribution
        self.regime_bar_counts[regime] += 1

        # Track transitions
        if self.last_regime and self.last_regime != regime:
            self.transition_count += 1

            # Record duration of previous regime
            if self.last_regime in self.regime_start_times:
                duration_hours = (timestamp - self.regime_start_times[self.last_regime]).total_seconds() / 3600
                self.regime_durations[self.last_regime].append(duration_hours)

        # Track overrides
        if regime_result.get('event_override'):
            event_type = regime_result.get('override_reason', 'unknown')
            self.override_counts[event_type] += 1

        # Update regime start time
        if self.last_regime != regime:
            self.regime_start_times[regime] = timestamp

        self.last_regime = regime
        self.regime_history.append({
            'timestamp': timestamp,
            'regime': regime,
            'confidence': regime_result.get('regime_confidence', 0.0),
            'override': regime_result.get('event_override', False)
        })

    def generate_report(self, total_bars: int) -> str:
        """
        Generate shadow regime quality report.

        Returns:
            Formatted report string
        """
        lines = []
        lines.append("="*80)
        lines.append("SHADOW REGIME REPORT - Model Quality Metrics")
        lines.append("="*80)
        lines.append("")

        # Distribution
        lines.append("Regime Distribution:")
        for regime in ['crisis', 'risk_off', 'neutral', 'risk_on']:
            count = self.regime_bar_counts[regime]
            pct = (count / total_bars * 100) if total_bars > 0 else 0

            # Target ranges
            targets = {
                'crisis': (1, 5),
                'risk_off': (30, 40),
                'neutral': (30, 40),
                'risk_on': (20, 30)
            }
            target_min, target_max = targets[regime]
            status = "✅" if target_min <= pct <= target_max else "⚠️"

            lines.append(f"  {regime:10s}: {count:5d} bars ({pct:5.1f}%) - Target: {target_min}-{target_max}% {status}")

        lines.append("")

        # Transitions
        lines.append(f"Regime Transitions: {self.transition_count}")
        transitions_per_year = self.transition_count * (365*24 / total_bars) if total_bars > 0 else 0
        status = "✅" if 10 <= transitions_per_year <= 40 else "⚠️"
        lines.append(f"  Annualized: {transitions_per_year:.1f}/year - Target: 10-40/year {status}")
        lines.append("")

        # Durations
        lines.append("Average Regime Duration:")
        for regime in ['crisis', 'risk_off', 'neutral', 'risk_on']:
            if regime in self.regime_durations and self.regime_durations[regime]:
                avg_hours = sum(self.regime_durations[regime]) / len(self.regime_durations[regime])
                lines.append(f"  {regime:10s}: {avg_hours:6.1f} hours ({avg_hours/24:.1f} days)")
        lines.append("")

        # Event overrides
        if self.override_counts:
            lines.append("Event Override Triggers:")
            for event_type, count in sorted(self.override_counts.items()):
                lines.append(f"  {event_type:20s}: {count} times")
        else:
            lines.append("Event Override Triggers: None")
        lines.append("")

        # Issues summary
        lines.append("Model Quality Assessment:")
        issues = []

        crisis_pct = (self.regime_bar_counts['crisis'] / total_bars * 100) if total_bars > 0 else 0
        if crisis_pct > 5:
            issues.append(f"- Crisis over-prediction: {crisis_pct:.1f}% (target: 1-5%)")

        risk_on_pct = (self.regime_bar_counts['risk_on'] / total_bars * 100) if total_bars > 0 else 0
        if risk_on_pct < 10:
            issues.append(f"- Risk-on under-prediction: {risk_on_pct:.1f}% (target: 20-30%)")

        if transitions_per_year > 50:
            issues.append(f"- Excessive transitions: {transitions_per_year:.1f}/year (target: <40)")
        elif transitions_per_year < 5:
            issues.append(f"- Too stable: {transitions_per_year:.1f}/year (target: >10)")

        if issues:
            lines.append("  ⚠️ Issues Detected:")
            lines.extend([f"    {issue}" for issue in issues])
            lines.append("")
            lines.append("  Recommendation: Retrain model on 2023-2024 only (out-of-sample for 2022)")
        else:
            lines.append("  ✅ No major issues detected")

        lines.append("")
        lines.append("="*80)

        return "\n".join(lines)
