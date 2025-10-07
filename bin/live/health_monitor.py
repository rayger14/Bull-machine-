#!/usr/bin/env python3
"""
Health Monitor for Bull Machine v1.7.3 Live Feeds
Tracks macro veto rates, SMC hits, HOB relevance, and delta breaches
"""

import sys
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Add project root for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@dataclass
class HealthBands:
    """Health band thresholds for live monitoring."""
    # Veto rates (percentage)
    macro_veto_min: float = 5.0
    macro_veto_max: float = 15.0

    # SMC 2+ hit rate (percentage)
    smc_2hit_min: float = 30.0

    # HOB relevance (percentage)
    hob_relevance_max: float = 30.0

    # Delta breach tolerance
    delta_breaches_max: int = 0

    # Delta caps (absolute values)
    macro_delta_cap: float = 0.10
    momentum_delta_cap: float = 0.06
    hob_delta_cap: float = 0.05
    hps_delta_cap: float = 0.03


@dataclass
class HealthMetrics:
    """Current health metrics snapshot."""
    timestamp: datetime
    macro_veto_rate: float = 0.0
    smc_2hit_rate: float = 0.0
    hob_relevance_rate: float = 0.0
    delta_breaches: int = 0

    # Delta values
    macro_delta: float = 0.0
    momentum_delta: float = 0.0
    hob_delta: float = 0.0
    hps_delta: float = 0.0

    # Health status
    is_healthy: bool = True
    warnings: List[str] = field(default_factory=list)


class HealthMonitor:
    """Monitor system health with rolling windows and breach detection."""

    def __init__(self, window_size: int = 100, health_bands: Optional[HealthBands] = None):
        self.window_size = window_size
        self.health_bands = health_bands or HealthBands()

        # Rolling windows
        self.macro_vetoes = deque(maxlen=window_size)
        self.smc_hits = deque(maxlen=window_size)
        self.hob_relevance = deque(maxlen=window_size)
        self.signals_total = deque(maxlen=window_size)

        # Delta tracking
        self.delta_breaches = 0
        self.health_log: List[HealthMetrics] = []

    def record_signal(self, signal_data: Dict, domains_active: Dict, timestamp: datetime):
        """
        Record signal for health tracking.

        Args:
            signal_data: Signal result with confidence, vetoes, etc.
            domains_active: Dict of domain activity (wyckoff, smc, hob, etc.)
            timestamp: Signal timestamp
        """
        # Track macro vetoes
        macro_vetoed = signal_data.get('macro_vetoed', False)
        self.macro_vetoes.append(1 if macro_vetoed else 0)

        # Track SMC hits (2 or more domains)
        smc_domains = ['liquidity', 'structure', 'orderflow']
        smc_active_count = sum(1 for domain in smc_domains if domains_active.get(domain, False))
        self.smc_hits.append(1 if smc_active_count >= 2 else 0)

        # Track HOB relevance
        hob_active = domains_active.get('hob', False)
        self.hob_relevance.append(1 if hob_active else 0)

        # Track total signals
        self.signals_total.append(1)

        # Check delta breaches
        self._check_delta_breaches(signal_data)

    def _check_delta_breaches(self, signal_data: Dict):
        """Check for delta cap breaches."""
        breaches = 0

        # Extract deltas from signal data
        macro_delta = abs(signal_data.get('macro_delta', 0.0))
        momentum_delta = abs(signal_data.get('momentum_delta', 0.0))
        hob_delta = abs(signal_data.get('hob_delta', 0.0))
        hps_delta = abs(signal_data.get('hps_delta', 0.0))

        # Check each delta cap
        if macro_delta > self.health_bands.macro_delta_cap:
            breaches += 1
        if momentum_delta > self.health_bands.momentum_delta_cap:
            breaches += 1
        if hob_delta > self.health_bands.hob_delta_cap:
            breaches += 1
        if hps_delta > self.health_bands.hps_delta_cap:
            breaches += 1

        if breaches > 0:
            self.delta_breaches += breaches

    def get_current_metrics(self, timestamp: datetime) -> HealthMetrics:
        """Calculate current health metrics."""
        metrics = HealthMetrics(timestamp=timestamp)

        # Calculate rates
        if len(self.signals_total) > 0:
            total_signals = sum(self.signals_total)

            if total_signals > 0:
                metrics.macro_veto_rate = (sum(self.macro_vetoes) / total_signals) * 100
                metrics.smc_2hit_rate = (sum(self.smc_hits) / total_signals) * 100
                metrics.hob_relevance_rate = (sum(self.hob_relevance) / total_signals) * 100

        metrics.delta_breaches = self.delta_breaches

        # Check health status
        metrics.is_healthy = self._assess_health(metrics)
        metrics.warnings = self._generate_warnings(metrics)

        return metrics

    def _assess_health(self, metrics: HealthMetrics) -> bool:
        """Assess overall system health."""
        # Check macro veto rate
        if not (self.health_bands.macro_veto_min <= metrics.macro_veto_rate <= self.health_bands.macro_veto_max):
            return False

        # Check SMC 2+ hit rate
        if metrics.smc_2hit_rate < self.health_bands.smc_2hit_min:
            return False

        # Check HOB relevance
        if metrics.hob_relevance_rate > self.health_bands.hob_relevance_max:
            return False

        # Check delta breaches
        if metrics.delta_breaches > self.health_bands.delta_breaches_max:
            return False

        return True

    def _generate_warnings(self, metrics: HealthMetrics) -> List[str]:
        """Generate specific warnings based on metrics."""
        warnings = []

        # Macro veto rate warnings
        if metrics.macro_veto_rate < self.health_bands.macro_veto_min:
            warnings.append(f"Macro veto rate too low: {metrics.macro_veto_rate:.1f}% (min: {self.health_bands.macro_veto_min}%)")
        elif metrics.macro_veto_rate > self.health_bands.macro_veto_max:
            warnings.append(f"Macro veto rate too high: {metrics.macro_veto_rate:.1f}% (max: {self.health_bands.macro_veto_max}%)")

        # SMC hit rate warning
        if metrics.smc_2hit_rate < self.health_bands.smc_2hit_min:
            warnings.append(f"SMC 2+ hit rate too low: {metrics.smc_2hit_rate:.1f}% (min: {self.health_bands.smc_2hit_min}%)")

        # HOB relevance warning
        if metrics.hob_relevance_rate > self.health_bands.hob_relevance_max:
            warnings.append(f"HOB relevance too high: {metrics.hob_relevance_rate:.1f}% (max: {self.health_bands.hob_relevance_max}%)")

        # Delta breach warning
        if metrics.delta_breaches > self.health_bands.delta_breaches_max:
            warnings.append(f"Delta breaches detected: {metrics.delta_breaches} (max: {self.health_bands.delta_breaches_max})")

        return warnings

    def log_health(self, metrics: HealthMetrics):
        """Log health metrics for tracking."""
        self.health_log.append(metrics)

        # Print warnings if any
        if not metrics.is_healthy:
            print(f"⚠️  Health Warning at {metrics.timestamp}:")
            for warning in metrics.warnings:
                print(f"   - {warning}")

    def get_health_summary(self) -> Dict:
        """Get comprehensive health summary."""
        if not self.health_log:
            return {'status': 'no_data'}

        latest = self.health_log[-1]

        # Calculate average metrics over recent window
        recent_window = min(50, len(self.health_log))
        recent_metrics = self.health_log[-recent_window:]

        avg_macro_veto = np.mean([m.macro_veto_rate for m in recent_metrics])
        avg_smc_hit = np.mean([m.smc_2hit_rate for m in recent_metrics])
        avg_hob_relevance = np.mean([m.hob_relevance_rate for m in recent_metrics])

        return {
            'status': 'healthy' if latest.is_healthy else 'warning',
            'current_metrics': {
                'macro_veto_rate': latest.macro_veto_rate,
                'smc_2hit_rate': latest.smc_2hit_rate,
                'hob_relevance_rate': latest.hob_relevance_rate,
                'delta_breaches': latest.delta_breaches
            },
            'average_metrics': {
                'macro_veto_rate': avg_macro_veto,
                'smc_2hit_rate': avg_smc_hit,
                'hob_relevance_rate': avg_hob_relevance
            },
            'health_bands': {
                'macro_veto_range': f"{self.health_bands.macro_veto_min}-{self.health_bands.macro_veto_max}%",
                'smc_2hit_min': f"{self.health_bands.smc_2hit_min}%",
                'hob_relevance_max': f"{self.health_bands.hob_relevance_max}%"
            },
            'warnings': latest.warnings,
            'total_checks': len(self.health_log)
        }

    def export_health_log(self, filepath: str):
        """Export health log to JSON."""
        import json

        export_data = []
        for metrics in self.health_log:
            export_data.append({
                'timestamp': metrics.timestamp.isoformat(),
                'macro_veto_rate': metrics.macro_veto_rate,
                'smc_2hit_rate': metrics.smc_2hit_rate,
                'hob_relevance_rate': metrics.hob_relevance_rate,
                'delta_breaches': metrics.delta_breaches,
                'is_healthy': metrics.is_healthy,
                'warnings': metrics.warnings
            })

        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)


# VIX Hysteresis Helper
class VIXHysteresis:
    """VIX hysteresis state management."""

    def __init__(self, on_threshold: float = 22.0, off_threshold: float = 18.0):
        self.on_threshold = on_threshold
        self.off_threshold = off_threshold
        self.is_active = False
        self.previous_value = None

    def update(self, vix_value: float) -> bool:
        """
        Update hysteresis state.

        Args:
            vix_value: Current VIX value

        Returns:
            True if hysteresis is active
        """
        # Track previous value before updating
        if self.previous_value is None:
            self.previous_value = vix_value

        if not self.is_active and vix_value >= self.on_threshold:
            self.is_active = True
        elif self.is_active and vix_value <= self.off_threshold:
            self.is_active = False

        # Update previous value for next iteration
        old_value = self.previous_value
        self.previous_value = vix_value

        return self.is_active
