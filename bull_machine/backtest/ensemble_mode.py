"""
Bull Machine v1.5.1 - Time-Tolerant Ensemble Mode
Implements realistic multi-timeframe alignment with lead-lag tolerance
"""

import collections
from typing import Dict, Tuple
from bull_machine.core.telemetry import log_telemetry


def _pass_floors(scores: dict, floors: dict) -> bool:
    """Check if layer scores pass quality floors."""
    for k, f in floors.items():
        if scores.get(k, 0.0) < f:
            return False
    return True


class EnsembleAligner:
    """
    Time-tolerant ensemble alignment system.

    Features:
    - min_consensus: require K of 3 TFs (default 2 of 3)
    - rolling_k/N: require K bars aligned within last N bars (per TF)
    - lead_lag_window: HTF (1D, 4H) may confirm within last M bars; 1H can trigger now
    - consensus_penalty: slight score reduction when not all 3 TFs agree
    """

    def __init__(self, cfg: dict):
        ens = cfg.get('ensemble', {})
        self.min_consensus = int(ens.get('min_consensus', 2))
        self.consensus_penalty = float(ens.get('consensus_penalty', 0.02))
        self.rolling_k = int(ens.get('rolling_k', 0))
        self.rolling_n = int(ens.get('rolling_n', 0))
        self.lead_lag_window = int(ens.get('lead_lag_window', 3))
        self.floors = cfg.get('quality_floors', {})
        self.base_threshold = float(cfg.get('entry_threshold', 0.45))
        self.dynamic_thresholds = ens.get('dynamic_thresholds', True)

        # Rolling windows for each timeframe
        self.windows = {
            '1H': collections.deque(maxlen=self.rolling_n or 1),
            '4H': collections.deque(maxlen=self.rolling_n or 1),
            '1D': collections.deque(maxlen=self.rolling_n or 1),
        }

        print(f"🔄 EnsembleAligner initialized:")
        print(f"   Min consensus: {self.min_consensus}/3 timeframes")
        print(f"   Rolling requirement: {self.rolling_k}/{self.rolling_n} bars")
        print(f"   Lead-lag window: {self.lead_lag_window} bars")
        print(f"   Consensus penalty: {self.consensus_penalty}")
        print(f"   Base threshold: {self.base_threshold}")
        print(f"   Dynamic thresholds: {self.dynamic_thresholds}")

    def compute_dynamic_threshold(self, df) -> float:
        """
        Compute dynamic entry threshold based on market volatility.
        Lower threshold in high volatility to allow more trades.
        """
        if not self.dynamic_thresholds:
            return self.base_threshold

        try:
            # Calculate ATR percentage
            high_low_range = (df['high'].rolling(14).max() - df['low'].rolling(14).min()).iloc[-1]
            current_price = df['close'].iloc[-1]
            atr_pct = high_low_range / current_price if current_price > 0 else 0.05

            # Dynamic adjustment based on volatility
            if atr_pct > 0.05:  # High volatility
                adjustment = -0.03  # Lower threshold more aggressively
            elif atr_pct < 0.025:  # Low volatility
                adjustment = 0.015  # Raise threshold for selectivity
            else:
                adjustment = -0.005  # Slight bias towards more trades

            dynamic_threshold = max(0.42, min(0.48, self.base_threshold + adjustment))

            log_telemetry('layer_masks.json', {
                'atr_pct': atr_pct,
                'base_threshold': self.base_threshold,
                'adjustment': adjustment,
                'dynamic_threshold': dynamic_threshold
            })

            return dynamic_threshold

        except Exception as e:
            log_telemetry('layer_masks.json', {
                'dynamic_threshold_error': str(e),
                'fallback_threshold': self.base_threshold
            })
            return self.base_threshold

    def update(self, tf_scores: dict):
        """Update rolling windows with current timeframe scores."""
        for tf, scores in tf_scores.items():
            if tf in self.windows:
                passed = _pass_floors(scores, self.floors)
                self.windows[tf].append(passed)

    def _rolling_pass(self, tf: str) -> bool:
        """Check if timeframe passes rolling requirement."""
        if self.rolling_k == 0 or self.rolling_n == 0:
            # No rolling requirement - just check latest bar
            return len(self.windows[tf]) > 0 and self.windows[tf][-1]

        # Check if K of last N bars passed
        return sum(self.windows[tf]) >= self.rolling_k

    def fire(self, tf_scores: dict, df=None) -> Tuple[bool, float]:
        """
        Determine if ensemble should fire entry signal with dynamic thresholds.

        Args:
            tf_scores: Dictionary of timeframe scores
            df: Price DataFrame for dynamic threshold calculation

        Returns:
            Tuple of (should_fire: bool, ensemble_score: float)
        """
        # Check current and recent passes for each timeframe
        tf_pass = {tf: self._rolling_pass(tf) for tf in ('1H', '4H', '1D')}

        def recent_pass(tf: str) -> bool:
            """Check if TF passed within lead-lag window."""
            w = list(self.windows[tf])
            if not w:
                return False
            # Check last lead_lag_window bars
            window_size = min(self.lead_lag_window, len(w))
            return any(w[-i-1] for i in range(window_size))

        # Count votes with lead-lag tolerance
        votes = 0

        # 1D and 4H can use lead-lag window (HTF confirmation can be slightly delayed)
        votes += 1 if (tf_pass['1D'] or recent_pass('1D')) else 0
        votes += 1 if (tf_pass['4H'] or recent_pass('4H')) else 0

        # 1H must be current (LTF drives timing)
        votes += 1 if tf_pass['1H'] else 0

        # Check minimum consensus
        if votes < self.min_consensus:
            log_telemetry('layer_masks.json', {
                'ensemble_fire': False,
                'ensemble_votes': votes,
                'min_consensus': self.min_consensus,
                'tf_pass': tf_pass,
                'reason': 'insufficient_consensus'
            })
            return False, 0.0

        # Calculate ensemble score (average of all timeframe scores)
        def mean_score(scores: dict) -> float:
            if not scores:
                return 0.0
            return sum(scores.values()) / len(scores)

        # Weight timeframes: 1D highest, 4H medium, 1H lowest
        weights = {'1D': 0.5, '4H': 0.3, '1H': 0.2}
        weighted_score = 0.0
        total_weight = 0.0

        for tf, scores in tf_scores.items():
            if tf in weights:
                weight = weights[tf]
                score = mean_score(scores)
                weighted_score += score * weight
                total_weight += weight

        if total_weight > 0:
            weighted_score /= total_weight

        # Apply consensus penalty if not all 3 TFs agree
        if votes == 2:  # Only 2 of 3 agree
            weighted_score -= self.consensus_penalty

        # Compute dynamic threshold
        entry_threshold = self.compute_dynamic_threshold(df) if df is not None else self.base_threshold

        # Check ensemble threshold
        fire = weighted_score >= entry_threshold

        # Log ensemble decision
        log_telemetry('layer_masks.json', {
            'ensemble_fire': fire,
            'ensemble_votes': votes,
            'weighted_score': weighted_score,
            'entry_threshold': entry_threshold,
            'base_threshold': self.base_threshold,
            'tf_pass': tf_pass,
            'rolling_k': self.rolling_k,
            'rolling_n': self.rolling_n,
            'lead_lag_window': self.lead_lag_window,
            'consensus_penalty_applied': votes == 2
        })

        return fire, weighted_score