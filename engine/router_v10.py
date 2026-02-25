#!/usr/bin/env python3
"""
Router v10 - Intelligent Config Selector

Switches between bull/bear configs based on regime + confidence + events.

Decision Logic:
    1. Confidence veto: If confidence < threshold → CASH
    2. Event suppression: If event window → CASH
    3. Regime-based selection:
       - risk_off/crisis → Bear config (defensive)
       - risk_on/neutral → Bull config (aggressive)

Usage:
    from engine.router_v10 import RouterV10

    router = RouterV10()

    # Get config decision for single bar
    decision = router.select_config(
        timestamp=timestamp,
        regime_label='risk_on',
        regime_confidence=0.75,
        event_flag=False
    )

    if decision['action'] == 'CASH':
        # No new entries
        pass
    else:
        # Use decision['config_path']
        pass
"""

import json
from pathlib import Path
from typing import Dict, Optional
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


class RouterV10:
    """
    Intelligent config router for all-weather trading.

    Switches between bull/bear configs based on market regime, confidence, and events.
    """

    def __init__(
        self,
        bull_config_path: str = 'configs/v10_bases/btc_bull_v10_best.json',
        bear_config_path: str = 'configs/v10_bases/btc_bear_v10_best.json',
        confidence_threshold: float = 0.60,
        event_suppression: bool = True,
        hysteresis_bars: int = 0
    ):
        """
        Initialize Router v10.

        Args:
            bull_config_path: Path to aggressive bull config
            bear_config_path: Path to defensive bear config
            confidence_threshold: Min confidence for entries (default 0.60)
            event_suppression: Whether to suppress entries during events (default True)
            hysteresis_bars: Consecutive bars required before regime switch (default 0)
        """
        self.bull_config_path = Path(bull_config_path)
        self.bear_config_path = Path(bear_config_path)
        self.confidence_threshold = confidence_threshold
        self.event_suppression = event_suppression
        self.hysteresis_bars = hysteresis_bars

        # Validate config files exist
        if not self.bull_config_path.exists():
            raise FileNotFoundError(f"Bull config not found: {bull_config_path}")
        if not self.bear_config_path.exists():
            raise FileNotFoundError(f"Bear config not found: {bear_config_path}")

        # Load configs to validate JSON structure
        with open(self.bull_config_path) as f:
            self.bull_config = json.load(f)
        with open(self.bear_config_path) as f:
            self.bear_config = json.load(f)

        # Hysteresis state tracking
        self._prev_regime = None
        self._regime_counter = 0

        # Telemetry
        self.decision_history = []

    def select_config(
        self,
        timestamp: pd.Timestamp,
        regime_label: str,
        regime_confidence: float,
        event_flag: bool
    ) -> Dict:
        """
        Select config for given bar.

        Args:
            timestamp: Bar timestamp
            regime_label: One of ['crisis', 'risk_off', 'neutral', 'risk_on']
            regime_confidence: Probability of regime (0.0 to 1.0)
            event_flag: True if in macro event window

        Returns:
            {
                'action': 'BULL' | 'BEAR' | 'CASH',
                'config_path': str or None,
                'config': dict or None,
                'reason': str,
                'timestamp': pd.Timestamp,
                'regime_label': str,
                'regime_confidence': float,
                'event_flag': bool
            }
        """

        # Rule 1: Confidence veto (stand down if uncertain)
        if regime_confidence < self.confidence_threshold:
            decision = self._make_decision(
                action='CASH',
                config_path=None,
                config=None,
                reason=f'low_confidence_{regime_confidence:.2f}',
                timestamp=timestamp,
                regime_label=regime_label,
                regime_confidence=regime_confidence,
                event_flag=event_flag
            )
            self._log_decision(decision)
            return decision

        # Rule 2: Event suppression (stand down during macro events)
        if self.event_suppression and event_flag:
            decision = self._make_decision(
                action='CASH',
                config_path=None,
                config=None,
                reason='event_suppression',
                timestamp=timestamp,
                regime_label=regime_label,
                regime_confidence=regime_confidence,
                event_flag=event_flag
            )
            self._log_decision(decision)
            return decision

        # Rule 3: Regime-based selection (with optional hysteresis)
        effective_regime = self._apply_hysteresis(regime_label)

        if effective_regime in ['risk_off', 'crisis']:
            # Defensive mode - use bear config
            decision = self._make_decision(
                action='BEAR',
                config_path=str(self.bear_config_path),
                config=self.bear_config,
                reason=f'{effective_regime}_defensive',
                timestamp=timestamp,
                regime_label=regime_label,
                regime_confidence=regime_confidence,
                event_flag=event_flag
            )
        else:
            # Aggressive mode - use bull config (risk_on or neutral)
            decision = self._make_decision(
                action='BULL',
                config_path=str(self.bull_config_path),
                config=self.bull_config,
                reason=f'{effective_regime}_aggressive',
                timestamp=timestamp,
                regime_label=regime_label,
                regime_confidence=regime_confidence,
                event_flag=event_flag
            )

        self._log_decision(decision)
        return decision

    def _apply_hysteresis(self, regime_label: str) -> str:
        """
        Apply hysteresis to prevent regime whipsaw.

        Requires N consecutive bars of new regime before switching.

        Args:
            regime_label: Current regime

        Returns:
            Effective regime (may be previous regime if hysteresis not met)
        """
        if self.hysteresis_bars == 0:
            # No hysteresis
            return regime_label

        # First call - initialize previous regime
        if self._prev_regime is None:
            self._prev_regime = regime_label
            self._regime_counter = 0
            return regime_label

        if regime_label == self._prev_regime:
            # Same regime, reset counter
            self._regime_counter = 0
            return regime_label
        else:
            # New regime detected
            self._regime_counter += 1

            if self._regime_counter >= self.hysteresis_bars:
                # Hysteresis threshold met - accept new regime
                self._prev_regime = regime_label
                self._regime_counter = 0
                return regime_label
            else:
                # Hysteresis not met - stick with previous regime
                return self._prev_regime

    def _make_decision(
        self,
        action: str,
        config_path: Optional[str],
        config: Optional[dict],
        reason: str,
        timestamp: pd.Timestamp,
        regime_label: str,
        regime_confidence: float,
        event_flag: bool
    ) -> Dict:
        """Create decision dictionary."""
        return {
            'action': action,
            'config_path': config_path,
            'config': config,
            'reason': reason,
            'timestamp': timestamp,
            'regime_label': regime_label,
            'regime_confidence': regime_confidence,
            'event_flag': event_flag
        }

    def _log_decision(self, decision: Dict):
        """Log decision for telemetry."""
        # Store lightweight version (no full config dict)
        log_entry = {
            'timestamp': decision['timestamp'],
            'action': decision['action'],
            'reason': decision['reason'],
            'regime_label': decision['regime_label'],
            'regime_confidence': decision['regime_confidence'],
            'event_flag': decision['event_flag']
        }
        self.decision_history.append(log_entry)

    def get_stats(self) -> Dict:
        """
        Get router statistics from decision history.

        Returns:
            {
                'total_decisions': int,
                'action_distribution': {'BULL': %, 'BEAR': %, 'CASH': %},
                'reason_distribution': {reason: count},
                'regime_switches': int,
                'confidence_stats': {mean, median, min, max}
            }
        """
        if len(self.decision_history) == 0:
            return {
                'total_decisions': 0,
                'action_distribution': {},
                'reason_distribution': {},
                'regime_switches': 0,
                'confidence_stats': {}
            }

        # Action distribution
        actions = [d['action'] for d in self.decision_history]
        total = len(actions)
        action_dist = {
            'BULL': 100 * actions.count('BULL') / total,
            'BEAR': 100 * actions.count('BEAR') / total,
            'CASH': 100 * actions.count('CASH') / total
        }

        # Reason distribution
        reasons = [d['reason'] for d in self.decision_history]
        reason_dist = {r: reasons.count(r) for r in set(reasons)}

        # Regime switches
        regimes = [d['regime_label'] for d in self.decision_history]
        switches = sum(1 for i in range(1, len(regimes)) if regimes[i] != regimes[i-1])

        # Confidence stats
        confidences = [d['regime_confidence'] for d in self.decision_history]
        conf_stats = {
            'mean': sum(confidences) / len(confidences),
            'median': sorted(confidences)[len(confidences)//2],
            'min': min(confidences),
            'max': max(confidences)
        }

        return {
            'total_decisions': total,
            'action_distribution': action_dist,
            'reason_distribution': reason_dist,
            'regime_switches': switches,
            'confidence_stats': conf_stats
        }

    def export_decision_log(self, output_path: str):
        """
        Export decision history to JSON file.

        Args:
            output_path: Path to output JSON file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert timestamps to strings for JSON serialization
        log_data = []
        for entry in self.decision_history:
            log_entry = entry.copy()
            log_entry['timestamp'] = entry['timestamp'].isoformat()
            log_data.append(log_entry)

        with open(output_path, 'w') as f:
            json.dump(log_data, f, indent=2)

        print(f"Exported {len(log_data)} router decisions to {output_path}")


if __name__ == '__main__':
    """Quick test of router v10."""
    print("\n" + "="*80)
    print("ROUTER V10 - QUICK TEST")
    print("="*80)

    # Initialize router
    router = RouterV10()
    print("\n✅ Initialized RouterV10")
    print(f"   Bull config: {router.bull_config_path}")
    print(f"   Bear config: {router.bear_config_path}")
    print(f"   Confidence threshold: {router.confidence_threshold}")
    print(f"   Event suppression: {router.event_suppression}")

    # Test decision scenarios
    print(f"\n{'='*80}")
    print("TEST SCENARIOS")
    print('='*80)

    test_cases = [
        # (regime, confidence, event_flag, expected_action, description)
        ('risk_on', 0.85, False, 'BULL', 'Clear bull market'),
        ('risk_off', 0.75, False, 'BEAR', 'Clear bear market'),
        ('crisis', 0.70, False, 'BEAR', 'Crisis mode'),
        ('neutral', 0.80, False, 'BULL', 'Neutral regime'),
        ('risk_on', 0.45, False, 'CASH', 'Low confidence veto'),
        ('risk_on', 0.85, True, 'CASH', 'Event suppression'),
        ('risk_off', 0.55, False, 'CASH', 'Low confidence + risk_off'),
    ]

    timestamp = pd.Timestamp('2024-01-15 14:00:00', tz='UTC')

    for regime, conf, event, expected, desc in test_cases:
        decision = router.select_config(
            timestamp=timestamp,
            regime_label=regime,
            regime_confidence=conf,
            event_flag=event
        )

        status = "✅" if decision['action'] == expected else "❌"
        print(f"\n{status} {desc}")
        print(f"   Regime: {regime:10s} | Conf: {conf:.2f} | Event: {event}")
        print(f"   → Action: {decision['action']:5s} (reason: {decision['reason']})")

        if decision['action'] != expected:
            print(f"   ⚠️  Expected: {expected}, got: {decision['action']}")

    # Stats
    print(f"\n{'='*80}")
    print("ROUTER STATISTICS")
    print('='*80)

    stats = router.get_stats()
    print(f"\nTotal decisions: {stats['total_decisions']}")

    print("\nAction distribution:")
    for action, pct in stats['action_distribution'].items():
        bar = '█' * int(pct / 5)
        print(f"  {action:5s}: {pct:5.1f}% {bar}")

    print("\nReason distribution:")
    for reason, count in sorted(stats['reason_distribution'].items()):
        print(f"  {reason:30s}: {count:3d}")

    print("\nConfidence stats:")
    for stat, val in stats['confidence_stats'].items():
        print(f"  {stat:8s}: {val:.3f}")

    print("\n" + "="*80)
    print("✅ ROUTER V10 READY")
    print("="*80)
