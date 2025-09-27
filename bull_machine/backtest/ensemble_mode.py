"""
Bull Machine v1.6.0 - Enhanced Ensemble Mode with Advanced Scoring
Implements volatility-weighted and time-based scoring to prevent clustering
Integrates M1/M2 Wyckoff and hidden Fibonacci signals
"""

import collections
import pandas as pd
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

    def compute_dynamic_threshold(self, df: pd.DataFrame, hour: int = None) -> float:
        """
        Compute dynamic entry threshold based on market volatility and time.
        Enhanced v1.6.0 with time-based adjustments to prevent clustering.
        """
        if not self.dynamic_thresholds:
            return self.base_threshold

        try:
            # Calculate ATR percentage
            high_low_range = (df['high'].rolling(14).max() - df['low'].rolling(14).min()).iloc[-1]
            current_price = df['close'].iloc[-1]
            atr_pct = high_low_range / current_price if current_price > 0 else 0.05

            # Volatility-based adjustment
            if atr_pct > 0.05:  # High volatility
                vol_adjustment = -0.02  # Lower threshold for more opportunities
            elif atr_pct < 0.025:  # Low volatility
                vol_adjustment = 0.015  # Higher threshold for selectivity
            else:
                vol_adjustment = -0.005  # Slight bias towards trades

            # Time-based adjustment (if hour provided)
            time_adjustment = 0.0
            if hour is not None:
                if 2 <= hour < 8:  # Asian session (lower volume)
                    time_adjustment = 0.01  # Higher threshold
                elif 8 <= hour < 16:  # European session (high volume)
                    time_adjustment = -0.01  # Lower threshold
                elif 16 <= hour < 24:  # US session (high volume)
                    time_adjustment = -0.005  # Slightly lower threshold
                # Overlap periods get additional bonus
                if hour in [8, 9, 15, 16]:  # Session overlaps
                    time_adjustment -= 0.005

            dynamic_threshold = max(0.40, min(0.50, self.base_threshold + vol_adjustment + time_adjustment))

            log_telemetry('layer_masks.json', {
                'atr_pct': atr_pct,
                'hour': hour,
                'base_threshold': self.base_threshold,
                'vol_adjustment': vol_adjustment,
                'time_adjustment': time_adjustment,
                'dynamic_threshold': dynamic_threshold
            })

            return dynamic_threshold

        except Exception as e:
            log_telemetry('layer_masks.json', {
                'dynamic_threshold_error': str(e),
                'fallback_threshold': self.base_threshold
            })
            return self.base_threshold

    def compute_weighted_ensemble_score(self, tf_scores: Dict, dfs: Dict) -> float:
        """
        Compute volatility-weighted ensemble score to prevent clustering.
        Enhanced v1.6.0 with M1/M2 and Fibonacci weighting.
        """
        try:
            # Calculate market volatility for weighting
            df_1h = dfs.get('1H')
            if df_1h is None or len(df_1h) < 14:
                return self._simple_average_score(tf_scores)

            atr_pct = (df_1h['high'].rolling(14).max() - df_1h['low'].rolling(14).min()).iloc[-1] / df_1h['close'].iloc[-1]

            # Dynamic weights based on volatility and signal type
            base_weights = {
                'wyckoff': 1.0,
                'liquidity': 1.0,
                'structure': 1.0,
                'momentum': 1.2,  # Always important
                'volume': 1.0,
                'context': 1.0,
                'mtf': 1.2,  # Always important
            }

            # Enhanced weights for v1.6.0 signals
            enhanced_weights = {
                'm1': 1.3 if atr_pct > 0.04 else 1.1,  # M1 springs more valuable in volatility
                'm2': 1.3 if atr_pct > 0.04 else 1.1,  # M2 markups more valuable in volatility
                'fib_retracement': 1.2,  # Hidden fibs are precision tools
                'fib_extension': 1.2,
            }

            # Volatility adjustments for base signals
            if atr_pct > 0.05:  # High volatility market
                base_weights.update({
                    'wyckoff': 1.3,  # Structure becomes more important
                    'liquidity': 0.9,  # Liquidity less reliable in chaos
                    'volume': 1.2,  # Volume confirmation critical
                })
            elif atr_pct < 0.025:  # Low volatility market
                base_weights.update({
                    'structure': 1.2,  # Technical levels more reliable
                    'momentum': 0.9,  # Momentum less important
                })

            # Combine all weights
            all_weights = {**base_weights, **enhanced_weights}

            # Calculate weighted score for each timeframe
            tf_weighted_scores = {}
            for tf, scores in tf_scores.items():
                weighted_sum = 0.0
                total_weight = 0.0

                for signal_type, score in scores.items():
                    if score > 0:  # Only weight active signals
                        weight = all_weights.get(signal_type, 1.0)
                        weighted_sum += score * weight
                        total_weight += weight

                if total_weight > 0:
                    tf_weighted_scores[tf] = weighted_sum / total_weight
                else:
                    tf_weighted_scores[tf] = 0.0

            # Average across timeframes
            ensemble_score = sum(tf_weighted_scores.values()) / len(tf_weighted_scores)

            log_telemetry('layer_masks.json', {
                'atr_pct': atr_pct,
                'tf_weighted_scores': tf_weighted_scores,
                'ensemble_score': ensemble_score,
                'volatility_regime': 'high' if atr_pct > 0.05 else 'low' if atr_pct < 0.025 else 'normal'
            })

            return ensemble_score

        except Exception as e:
            log_telemetry('layer_masks.json', {
                'weighted_scoring_error': str(e)
            })
            return self._simple_average_score(tf_scores)

    def _simple_average_score(self, tf_scores: Dict) -> float:
        """Fallback simple averaging if weighted scoring fails."""
        try:
            all_scores = []
            for tf, scores in tf_scores.items():
                tf_avg = sum(scores.values()) / len(scores) if scores else 0.0
                all_scores.append(tf_avg)
            return sum(all_scores) / len(all_scores) if all_scores else 0.0
        except:
            return 0.0

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

    def fire(self, tf_scores: dict, dfs: dict = None) -> Tuple[bool, float]:
        """
        Determine if ensemble should fire entry signal with enhanced v1.6.0 scoring.

        Args:
            tf_scores: Dictionary of timeframe scores
            dfs: Dictionary of DataFrames for enhanced scoring and dynamic thresholds

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

        # Calculate ensemble score using enhanced v1.6.0 weighting
        if dfs:
            # Use enhanced scoring with volatility and time weighting
            ensemble_score = self.compute_weighted_ensemble_score(tf_scores, dfs)

            # Extract hour from 1H timeframe for time-based threshold
            hour = None
            try:
                if '1H' in dfs and 'timestamp' in dfs['1H'].columns:
                    hour = pd.to_datetime(dfs['1H']['timestamp'].iloc[-1]).hour
            except:
                pass

            # Get dynamic threshold
            threshold_df = dfs.get('1H', list(dfs.values())[0])
            dynamic_threshold = self.compute_dynamic_threshold(threshold_df, hour)
        else:
            # Fallback to simple average
            ensemble_score = self._simple_average_score(tf_scores)
            dynamic_threshold = self.base_threshold

        # Apply consensus penalty if not all timeframes agree
        if votes == 2:  # Only 2 out of 3 timeframes
            ensemble_score -= self.consensus_penalty

        # Check if score meets threshold
        fire_signal = ensemble_score >= dynamic_threshold

        log_telemetry('layer_masks.json', {
            'ensemble_fire': fire_signal,
            'ensemble_score': ensemble_score,
            'dynamic_threshold': dynamic_threshold,
            'ensemble_votes': votes,
            'tf_pass': tf_pass,
            'consensus_penalty_applied': votes == 2,
            'hour': hour
        })

        return fire_signal, ensemble_score

        # OLD SIMPLE SCORING - keeping as reference
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