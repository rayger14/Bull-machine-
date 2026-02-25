"""
FeatureProvider - Hybrid Feature Store and Runtime Computation

This class provides a flexible feature provisioning layer that supports both:
1. Feature store lookups (for precomputed historical features)
2. Runtime computation (for essential indicators when feature store unavailable)

The hybrid approach enables:
- Fast backtesting with precomputed features (when available)
- Live trading compatibility with runtime computation (when needed)
- Graceful degradation (feature store → runtime → minimal defaults)

Architecture:
    Bar → FeatureProvider.get_features()
        ↓
        Try: Feature store lookup by timestamp
        ↓
        Fallback: Runtime computation of essential indicators
        ↓
        Return: Unified feature dict

Essential Features for Runtime Computation:
- ATR (for stop loss calculation)
- Liquidity score (for archetype detection)
- Fusion score (for confidence weighting)
- Basic Wyckoff indicators (for archetype detection)

Author: Claude Code (System Architect)
Date: 2026-01-21
"""

import logging
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from pathlib import Path

from engine.integrations.event_engine import Bar

# Import PTI calculator for runtime computation
try:
    from engine.psychology.pti import calculate_pti
    PTI_AVAILABLE = True
except ImportError:
    PTI_AVAILABLE = False

logger = logging.getLogger(__name__)


class FeatureProvider:
    """
    Hybrid feature provider supporting both feature store and runtime computation.

    Strategy:
    1. If feature_store_path provided and timestamp found → use feature store
    2. Else if enable_runtime_computation → compute essential features
    3. Else → raise error (no features available)

    This enables both:
    - Historical backtesting (fast, uses precomputed features)
    - Live trading (real-time, computes features on the fly)
    """

    def __init__(
        self,
        feature_store_path: Optional[str] = None,
        enable_runtime_computation: bool = True,
        atr_period: int = 14,
        rsi_period: int = 14,
        adx_period: int = 14
    ):
        """
        Initialize feature provider.

        Args:
            feature_store_path: Path to precomputed feature store CSV (if available)
            enable_runtime_computation: Enable runtime computation as fallback
            atr_period: ATR period for runtime computation
            rsi_period: RSI period for runtime computation
            adx_period: ADX period for runtime computation
        """
        self.feature_store_path = feature_store_path
        self.enable_runtime_computation = enable_runtime_computation
        self.atr_period = atr_period
        self.rsi_period = rsi_period
        self.adx_period = adx_period

        # Feature store (loaded on first access)
        self.feature_store: Optional[pd.DataFrame] = None
        self.feature_store_loaded = False

        # Price history buffer (for runtime computation)
        self.price_buffer = []
        self.max_buffer_size = max(atr_period, rsi_period, adx_period) + 50  # Keep extra for smoothing

        # Runtime computation cache
        self.runtime_cache: Dict[str, Any] = {}

        logger.info("FeatureProvider initialized")
        logger.info(f"  Feature store: {feature_store_path or 'None (runtime only)'}")
        logger.info(f"  Runtime computation: {'Enabled' if enable_runtime_computation else 'Disabled'}")
        logger.info(f"  ATR period: {atr_period}")

    def get_features(self, bar: Bar) -> Dict[str, Any]:
        """
        Get features for current bar.

        Hybrid strategy:
        1. Try feature store lookup
        2. Fallback to runtime computation
        3. Raise error if neither available

        Args:
            bar: Current bar (OHLCV data)

        Returns:
            Feature dict with all required features

        Raises:
            RuntimeError: If no features available (neither store nor runtime)
        """
        # Strategy 1: Try feature store
        if self.feature_store_path:
            features = self._lookup_feature_store(bar.timestamp)
            if features is not None:
                # Add bar OHLCV data (not in feature store)
                features['open'] = bar.open
                features['high'] = bar.high
                features['low'] = bar.low
                features['close'] = bar.close
                features['volume'] = bar.volume
                return features

        # Strategy 2: Fallback to runtime computation
        if self.enable_runtime_computation:
            features = self._compute_features_runtime(bar)
            return features

        # Strategy 3: No features available
        raise RuntimeError(
            f"No features available for {bar.timestamp}. "
            f"Feature store: {self.feature_store_path or 'None'}, "
            f"Runtime computation: {self.enable_runtime_computation}"
        )

    def _lookup_feature_store(self, timestamp: pd.Timestamp) -> Optional[Dict[str, Any]]:
        """
        Lookup features from precomputed feature store.

        Args:
            timestamp: Bar timestamp to lookup

        Returns:
            Feature dict if found, None otherwise
        """
        # Lazy load feature store
        if not self.feature_store_loaded:
            self._load_feature_store()

        if self.feature_store is None:
            return None

        # Lookup by timestamp
        try:
            # Handle timestamp alignment (feature store may be indexed differently)
            # Try exact match first
            if timestamp in self.feature_store.index:
                row = self.feature_store.loc[timestamp]
                return row.to_dict()

            # Try nearest timestamp (within 1 hour tolerance)
            nearest_idx = self.feature_store.index.get_indexer([timestamp], method='nearest')[0]
            if nearest_idx >= 0:
                nearest_ts = self.feature_store.index[nearest_idx]
                time_diff = abs((nearest_ts - timestamp).total_seconds())

                if time_diff <= 3600:  # 1 hour tolerance
                    row = self.feature_store.iloc[nearest_idx]
                    logger.debug(f"Feature store: nearest match {nearest_ts} (diff: {time_diff:.0f}s)")
                    return row.to_dict()
                else:
                    logger.warning(f"Feature store: no match within 1h for {timestamp} (nearest: {nearest_ts}, diff: {time_diff:.0f}s)")
                    return None
            else:
                logger.warning(f"Feature store: timestamp {timestamp} not found")
                return None

        except Exception as e:
            logger.error(f"Feature store lookup failed: {e}")
            return None

    def _load_feature_store(self):
        """Load feature store from CSV or parquet."""
        if not self.feature_store_path:
            return

        try:
            path = Path(self.feature_store_path)
            if not path.exists():
                logger.warning(f"Feature store not found: {path}")
                return

            logger.info(f"Loading feature store from {path}...")

            # Auto-detect format
            if str(path).endswith('.parquet'):
                self.feature_store = pd.read_parquet(path)
            elif str(path).endswith('.csv'):
                self.feature_store = pd.read_csv(path, index_col=0, parse_dates=True)
            else:
                raise ValueError(f"Unsupported file format: {path}. Use .csv or .parquet")

            logger.info(f"✓ Loaded {len(self.feature_store)} rows from feature store")
            logger.info(f"  Date range: {self.feature_store.index[0]} to {self.feature_store.index[-1]}")
            logger.info(f"  Columns: {len(self.feature_store.columns)} features")

            self.feature_store_loaded = True

        except Exception as e:
            logger.error(f"Failed to load feature store: {e}")
            self.feature_store = None
            self.feature_store_loaded = True

    def _compute_features_runtime(self, bar: Bar) -> Dict[str, Any]:
        """
        Compute essential features at runtime.

        This computes the MINIMUM set of features required for:
        - Archetype detection (liquidity_score, fusion_score)
        - Risk management (ATR, stop loss calculation)
        - Regime classification (if enabled)

        For a full backtest, use the feature store instead.

        Args:
            bar: Current bar

        Returns:
            Feature dict with essential features
        """
        # Add bar to price buffer
        self.price_buffer.append({
            'timestamp': bar.timestamp,
            'open': bar.open,
            'high': bar.high,
            'low': bar.low,
            'close': bar.close,
            'volume': bar.volume
        })

        # Trim buffer
        if len(self.price_buffer) > self.max_buffer_size:
            self.price_buffer = self.price_buffer[-self.max_buffer_size:]

        # Initialize feature dict with bar data
        features = {
            'timestamp': bar.timestamp,
            'open': bar.open,
            'high': bar.high,
            'low': bar.low,
            'close': bar.close,
            'volume': bar.volume
        }

        # Compute ATR (essential for stop loss)
        features['atr_14'] = self._compute_atr()
        features['atr'] = features['atr_14']  # Alias

        # Compute basic indicators
        features['rsi_14'] = self._compute_rsi()
        features['adx_14'] = self._compute_adx()

        # Compute liquidity score (simplified)
        features['liquidity_score'] = self._compute_liquidity_score_simple()

        # Compute fusion score (simplified)
        features['fusion_score'] = self._compute_fusion_score_simple(features)

        # Add minimal regime features (for RegimeService)
        features.update(self._compute_regime_features_minimal())

        # Add previous close (for regime event detection)
        if len(self.price_buffer) >= 2:
            features['prev_close'] = self.price_buffer[-2]['close']
        else:
            features['prev_close'] = bar.close

        # ========================================================================
        # BACKFILLED FEATURES (P0 Priority) - Post-PR #26 Runtime Integration
        # ========================================================================

        # 1. PTI (Phase Transition Index) - P0
        # Unlocks: spring (A), wick_trap (K), trap_within_trend (H)
        pti_features = self._compute_pti_runtime()
        features.update(pti_features)

        # 2. BOMS (Break of Market Structure) - P0
        # Unlocks: order_block_retest (B)
        features['boms_strength'] = self._compute_boms_strength()

        # 3. Resilience Score - P1
        # Unlocks: funding_divergence (S4)
        features['resilience_score'] = self._compute_resilience_score()

        # 4. Fusion Wyckoff (ensure max() not mean()) - P1
        features['fusion_wyckoff'] = self._compute_fusion_wyckoff()

        # 5. Momentum Score (weighted ADX/MACD/ROC) - P2
        features['momentum_score'] = self._compute_momentum_score(features)

        # Debug logging (every bar)
        logger.debug(
            f"Runtime features: ATR=${features['atr_14']:.2f}, "
            f"RSI={features['rsi_14']:.1f}, "
            f"Liquidity={features['liquidity_score']:.3f}, "
            f"Fusion={features['fusion_score']:.3f}, "
            f"PTI={features.get('tf1h_pti_score', 0.0):.3f}, "
            f"BOMS={features.get('boms_strength', 0.0):.3f}"
        )

        # Feature distribution logging (once per hour for analysis)
        # This helps identify if thresholds are too high/low
        if not hasattr(self, '_last_feature_log_hour'):
            self._last_feature_log_hour = -1

        current_hour = bar.timestamp.hour
        if current_hour != self._last_feature_log_hour:
            self._last_feature_log_hour = current_hour
            logger.info(
                f"[FEATURE_DIST] {bar.timestamp.strftime('%Y-%m-%d %H:00')} | "
                f"pti={features.get('tf1h_pti_score', 0.0):.4f} | "
                f"boms={features.get('boms_strength', 0.0):.4f} | "
                f"resilience={features.get('resilience_score', 0.0):.4f} | "
                f"fusion_wyck={features.get('fusion_wyckoff', 0.0):.4f} | "
                f"wyckoff={features.get('wyckoff_score', 0.0):.4f} | "
                f"liquidity={features['liquidity_score']:.4f} | "
                f"fusion={features['fusion_score']:.4f} | "
                f"adx={features['adx_14']:.2f} | "
                f"rsi={features['rsi_14']:.2f} | "
                f"momentum={features.get('momentum_score', 0.0):.4f}"
            )

        return features

    def _compute_atr(self) -> float:
        """
        Compute Average True Range.

        Returns:
            ATR value
        """
        if len(self.price_buffer) < self.atr_period + 1:
            # Not enough data - use simple range estimate
            if len(self.price_buffer) > 0:
                recent = self.price_buffer[-1]
                return (recent['high'] - recent['low'])
            else:
                return 0.0

        # Calculate true ranges
        true_ranges = []
        for i in range(1, len(self.price_buffer)):
            prev = self.price_buffer[i-1]
            curr = self.price_buffer[i]

            tr = max(
                curr['high'] - curr['low'],
                abs(curr['high'] - prev['close']),
                abs(curr['low'] - prev['close'])
            )
            true_ranges.append(tr)

        # Average over period
        if len(true_ranges) >= self.atr_period:
            atr = np.mean(true_ranges[-self.atr_period:])
        else:
            atr = np.mean(true_ranges)

        return atr

    def _compute_rsi(self) -> float:
        """
        Compute Relative Strength Index.

        Returns:
            RSI value (0-100)
        """
        if len(self.price_buffer) < self.rsi_period + 1:
            return 50.0  # Neutral

        # Calculate price changes
        closes = [bar['close'] for bar in self.price_buffer[-(self.rsi_period+1):]]
        changes = np.diff(closes)

        # Separate gains and losses
        gains = np.where(changes > 0, changes, 0)
        losses = np.where(changes < 0, -changes, 0)

        # Average gains and losses
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)

        # Calculate RS and RSI
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _compute_adx(self) -> float:
        """
        Compute Average Directional Index (simplified).

        Returns:
            ADX value (0-100)
        """
        if len(self.price_buffer) < self.adx_period + 1:
            return 20.0  # Neutral

        # Simplified ADX based on price momentum
        # Full ADX calculation requires DI+/DI- which needs more history
        closes = [bar['close'] for bar in self.price_buffer[-(self.adx_period+1):]]
        highs = [bar['high'] for bar in self.price_buffer[-(self.adx_period+1):]]
        lows = [bar['low'] for bar in self.price_buffer[-(self.adx_period+1):]]

        # Calculate directional movement
        up_moves = []
        down_moves = []
        for i in range(1, len(highs)):
            up_move = highs[i] - highs[i-1]
            down_move = lows[i-1] - lows[i]
            up_moves.append(max(up_move, 0))
            down_moves.append(max(down_move, 0))

        # Average directional movement
        avg_up = np.mean(up_moves)
        avg_down = np.mean(down_moves)

        # Simplified ADX (directional strength)
        if avg_up + avg_down == 0:
            return 20.0
        adx = 100 * abs(avg_up - avg_down) / (avg_up + avg_down)

        return min(adx, 100.0)

    def _compute_liquidity_score_simple(self) -> float:
        """
        Compute simplified liquidity score.

        In production, this uses BOMS/FVG/displacement analysis.
        For runtime, we use volume and price action as proxy.

        Returns:
            Liquidity score (0-1)
        """
        if len(self.price_buffer) < 10:
            return 0.5  # Neutral

        # Volume analysis (relative to recent average)
        recent_volumes = [bar['volume'] for bar in self.price_buffer[-10:]]
        avg_volume = np.mean(recent_volumes)
        current_volume = self.price_buffer[-1]['volume']

        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        volume_score = min(volume_ratio / 2.0, 1.0)  # Normalize to 0-1

        # Price range analysis (relative to ATR)
        atr = self._compute_atr()
        current_range = self.price_buffer[-1]['high'] - self.price_buffer[-1]['low']
        range_ratio = current_range / atr if atr > 0 else 1.0
        range_score = min(range_ratio, 1.0)

        # Combined liquidity score
        liquidity_score = (volume_score + range_score) / 2.0

        return liquidity_score

    def _compute_fusion_score_simple(self, features: Dict[str, Any]) -> float:
        """
        Compute simplified fusion score.

        In production, this uses weighted blend of:
        - Wyckoff signals
        - Liquidity score
        - Momentum indicators
        - Macro regime
        - FRVP analysis

        For runtime, we use simplified momentum + liquidity blend.

        Args:
            features: Current features

        Returns:
            Fusion score (0-1)
        """
        # Momentum component (RSI + ADX)
        rsi = features['rsi_14']
        adx = features['adx_14']

        # RSI momentum (distance from 50)
        rsi_momentum = abs(rsi - 50.0) / 50.0

        # ADX strength (normalized)
        adx_strength = adx / 100.0

        momentum_score = (rsi_momentum + adx_strength) / 2.0

        # Liquidity component
        liquidity_score = features['liquidity_score']

        # Combined fusion score
        fusion_score = (0.5 * momentum_score + 0.5 * liquidity_score)

        # Clip to [0, 1]
        fusion_score = max(0.0, min(1.0, fusion_score))

        return fusion_score

    def _compute_regime_features_minimal(self) -> Dict[str, Any]:
        """
        Compute minimal regime features for RegimeService.

        These are the essential features needed for dynamic regime classification.
        For a full regime model, use the feature store.

        Returns:
            Dict with minimal regime features
        """
        if len(self.price_buffer) < 20:
            # Not enough data - return neutral defaults
            logger.warning(
                "MACRO: BTC.D and USDT.D unavailable (insufficient price buffer). Using NaN."
            )
            return {
                'macro_regime': 'neutral',
                'crash_frequency_7d': 0.0,
                'crisis_persistence': 0.0,
                'aftershock_score': 0.0,
                'rv_20d': 0.3,
                'rv_60d': 0.3,
                'drawdown_persistence': 0.0,
                'funding_Z': 0.0,
                'oi': 1e9,
                'volume_z_7d': 0.0,
                'USDT.D': float('nan'),
                'BTC.D': float('nan'),
                'VIX_Z': 0.0,
                'DXY_Z': 0.0,
                'YIELD_CURVE': 0.0
            }

        # Calculate basic regime indicators
        closes = np.array([bar['close'] for bar in self.price_buffer[-20:]])
        returns = np.diff(np.log(closes))

        # Realized volatility (20-day)
        rv_20d = np.std(returns) * np.sqrt(365 * 24)  # Annualized (assuming hourly)

        # Drawdown
        peak = np.maximum.accumulate(closes)
        drawdown = (closes - peak) / peak
        max_drawdown = np.min(drawdown)

        # Volume z-score (simplified)
        volumes = np.array([bar['volume'] for bar in self.price_buffer[-20:]])
        volume_z = (volumes[-1] - np.mean(volumes)) / (np.std(volumes) + 1e-8)

        logger.warning(
            "MACRO: BTC.D and USDT.D not available in runtime feature provider. Using NaN."
        )

        return {
            'macro_regime': 'neutral',  # Will be overwritten by RegimeService
            'crash_frequency_7d': 0.0,
            'crisis_persistence': 0.0,
            'aftershock_score': 0.0,
            'rv_20d': min(rv_20d, 2.0),  # Cap at 200% annualized
            'rv_60d': min(rv_20d, 2.0),  # Use same as 20d (no 60d history)
            'drawdown_persistence': abs(max_drawdown),
            'funding_Z': 0.0,  # Not available in runtime
            'oi': 1e9,  # Dummy value
            'volume_z_7d': volume_z,
            'USDT.D': float('nan'),  # No hardcoded values — NaN if unavailable
            'BTC.D': float('nan'),   # No hardcoded values — NaN if unavailable
            'VIX_Z': 0.0,  # Neutral
            'DXY_Z': 0.0,  # Neutral
            'YIELD_CURVE': 0.0  # Neutral
        }

    # ============================================================================
    # BACKFILLED FEATURES - Post-PR #26 Runtime Integration
    # ============================================================================

    def _compute_pti_runtime(self) -> Dict[str, Any]:
        """
        Compute PTI (Phase Transition Index) at runtime.

        PTI quantifies psychological trap phases (spring, utad, shakeout).
        This unlocks spring (A), wick_trap (K), and trap_within_trend (H) archetypes.

        Returns:
            Dict with PTI features:
                - tf1h_pti_score: 0-1 trap strength
                - tf1h_pti_trap_type: 'spring', 'utad', 'shakeout', 'none'
                - tf1h_pti_confidence: 0-1 confidence
        """
        if not PTI_AVAILABLE or len(self.price_buffer) < 30:
            # PTI module not available or insufficient data
            return {
                'tf1h_pti_score': 0.0,
                'tf1h_pti_trap_type': 'none',
                'tf1h_pti_confidence': 0.0
            }

        try:
            # Convert price buffer to DataFrame for PTI calculation
            df = pd.DataFrame(self.price_buffer)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')

            # Calculate PTI
            pti_signal = calculate_pti(df, timeframe='1H')

            # Map trap types to archetype-compatible format
            trap_type_map = {
                'bearish_trap': 'spring',  # Retail trapped short → spring long
                'bullish_trap': 'utad',    # Retail trapped long → utad short
                'none': 'none'
            }

            return {
                'tf1h_pti_score': pti_signal.pti_score,
                'tf1h_pti_trap_type': trap_type_map.get(pti_signal.trap_type, 'none'),
                'tf1h_pti_confidence': pti_signal.confidence
            }

        except Exception as e:
            logger.warning(f"PTI calculation failed: {e}")
            return {
                'tf1h_pti_score': 0.0,
                'tf1h_pti_trap_type': 'none',
                'tf1h_pti_confidence': 0.0
            }

    def _compute_boms_strength(self) -> float:
        """
        Compute BOMS (Break of Market Structure) strength.

        BOMS detects volume-weighted price breaks of previous structure.
        Formula: boms_strength = volume * (price_break / range)

        This unlocks order_block_retest (B) archetype.

        Returns:
            BOMS strength (0-1, higher = stronger structural break)
        """
        if len(self.price_buffer) < 20:
            return 0.0

        try:
            # Get recent price action
            recent_bars = self.price_buffer[-20:]
            current_bar = self.price_buffer[-1]

            # Find recent swing high/low
            highs = [bar['high'] for bar in recent_bars]
            lows = [bar['low'] for bar in recent_bars]

            swing_high = max(highs[:-3])  # Exclude last 3 bars
            swing_low = min(lows[:-3])
            range_size = swing_high - swing_low

            if range_size <= 0:
                return 0.0

            # Check for break above swing high (bullish BOS)
            price_break_up = max(0, current_bar['close'] - swing_high)

            # Check for break below swing low (bearish BOS)
            price_break_down = max(0, swing_low - current_bar['close'])

            price_break = max(price_break_up, price_break_down)

            # Volume weight (relative to recent average)
            recent_volumes = [bar['volume'] for bar in recent_bars]
            avg_volume = np.mean(recent_volumes)
            volume_ratio = current_bar['volume'] / avg_volume if avg_volume > 0 else 1.0

            # BOMS strength = volume * (price_break / range)
            boms_strength = min(volume_ratio, 3.0) * (price_break / range_size)

            # Normalize to 0-1
            boms_strength = float(np.clip(boms_strength, 0, 1))

            return boms_strength

        except Exception as e:
            logger.warning(f"BOMS calculation failed: {e}")
            return 0.0

    def _compute_resilience_score(self) -> float:
        """
        Compute resilience score (market bounce strength).

        Resilience measures how quickly price bounces after drops.
        Proxy formula: resilience = 1 - abs(recent_drawdown) * volume_spike

        This unlocks funding_divergence (S4) archetype.

        Returns:
            Resilience score (0-1, higher = stronger resilience)
        """
        if len(self.price_buffer) < 10:
            return 0.5  # Neutral

        try:
            # Get recent price action
            recent_closes = [bar['close'] for bar in self.price_buffer[-10:]]
            recent_volumes = [bar['volume'] for bar in self.price_buffer[-10:]]

            # Calculate recent drawdown
            peak = np.max(recent_closes[:-1])
            current = recent_closes[-1]
            drawdown = (current - peak) / peak if peak > 0 else 0.0

            # Volume spike (relative to average)
            avg_volume = np.mean(recent_volumes[:-1])
            current_volume = recent_volumes[-1]
            volume_spike = current_volume / avg_volume if avg_volume > 0 else 1.0

            # Resilience = ability to absorb drawdown with volume
            # High volume + small drawdown = high resilience
            # High volume + large drawdown = low resilience (panic)
            resilience = 1.0 - (abs(drawdown) * min(volume_spike, 3.0) / 3.0)

            resilience = float(np.clip(resilience, 0, 1))

            return resilience

        except Exception as e:
            logger.warning(f"Resilience calculation failed: {e}")
            return 0.5

    def _compute_fusion_wyckoff(self) -> float:
        """
        Compute fusion_wyckoff score (max of spring/LPS/SOS, NOT mean).

        Historical bug: Used mean() instead of max(), diluting strong signals.
        Correct formula: fusion_wyckoff = max(spring_a, lps, sos)

        Returns:
            Wyckoff fusion score (0-1)
        """
        if len(self.price_buffer) < 20:
            return 0.0

        try:
            # Simplified Wyckoff event detection for runtime
            # In production, this would use full Wyckoff engine

            # Detect spring-like pattern: long lower wick + volume spike
            current = self.price_buffer[-1]
            body = abs(current['close'] - current['open'])
            lower_wick = min(current['open'], current['close']) - current['low']
            total_range = current['high'] - current['low']

            spring_score = 0.0
            if total_range > 0:
                wick_ratio = lower_wick / total_range
                if wick_ratio > 0.5 and body < total_range * 0.3:  # Long wick, small body
                    spring_score = min(wick_ratio, 1.0)

            # Detect volume strength (LPS/SOS proxy)
            recent_volumes = [bar['volume'] for bar in self.price_buffer[-10:]]
            avg_volume = np.mean(recent_volumes[:-1])
            current_volume = current['volume']
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0

            sos_lps_score = min(volume_ratio / 2.0, 1.0)  # Normalize to 0-1

            # CORRECT: Use max() not mean()
            fusion_wyckoff = max(spring_score, sos_lps_score, 0.0)

            return float(np.clip(fusion_wyckoff, 0, 1))

        except Exception as e:
            logger.warning(f"Fusion Wyckoff calculation failed: {e}")
            return 0.0

    def _compute_momentum_score(self, features: Dict[str, Any]) -> float:
        """
        Compute momentum score (weighted ADX/MACD/ROC).

        Formula: momentum_score = 0.4 * adx + 0.3 * macd_momentum + 0.3 * roc_momentum

        Returns:
            Momentum score (0-1)
        """
        try:
            # ADX component (already normalized 0-100)
            adx = features.get('adx_14', 20.0)
            adx_norm = adx / 100.0

            # MACD momentum (simplified - use RSI as proxy)
            rsi = features.get('rsi_14', 50.0)
            macd_momentum = abs(rsi - 50.0) / 50.0

            # ROC (Rate of Change) - use recent price change
            if len(self.price_buffer) >= 5:
                closes = [bar['close'] for bar in self.price_buffer[-5:]]
                roc = (closes[-1] - closes[0]) / closes[0] if closes[0] > 0 else 0.0
                roc_momentum = min(abs(roc) * 10, 1.0)  # Normalize to 0-1
            else:
                roc_momentum = 0.0

            # Weighted combination
            momentum_score = 0.4 * adx_norm + 0.3 * macd_momentum + 0.3 * roc_momentum

            return float(np.clip(momentum_score, 0, 1))

        except Exception as e:
            logger.warning(f"Momentum score calculation failed: {e}")
            return 0.5
