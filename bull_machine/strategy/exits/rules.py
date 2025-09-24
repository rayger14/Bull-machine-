"""
Exit Signal Rules
Core logic for detecting when to exit positions based on various criteria.
"""

import logging
from typing import Any, Dict, Optional

import pandas as pd

from .types import (
    CHoCHContext,
    ExitAction,
    ExitSignal,
    ExitType,
    MomentumContext,
    TimeStopContext,
)


class CHoCHAgainstDetector:
    """
    Detects Change of Character against our position bias.
    CHoCH-Against means market structure is shifting away from our trade direction.
    """

    def __init__(self, config: Dict[str, Any]):
        # STEP 3: Kill silent fallbacks and use sweep parameter names
        # Use bars_confirm (from sweep) not confirmation_bars (legacy name)
        if "bars_confirm" not in config:
            raise ValueError(f"CHOCH missing required key bars_confirm in {list(config.keys())}")

        self.min_break_strength = config.get('min_break_strength', 0.6)
        self.bars_confirm = int(config["bars_confirm"])  # STEP 3: Use sweep parameter name
        self.volume_confirmation_required = config.get('volume_confirmation', True)

        # STEP 2: Echo effective config at init
        effective_config = {
            "min_break_strength": self.min_break_strength,
            "bars_confirm": self.bars_confirm,
            "volume_confirmation": self.volume_confirmation_required
        }
        logging.info("EXIT_EVAL_APPLIED choch_against=%s", effective_config)

    def evaluate(self, symbol: str, position_bias: str,
                 mtf_data: Dict[str, pd.DataFrame],
                 current_bar: pd.Timestamp) -> Optional[ExitSignal]:
        """
        Detect CHoCH-Against pattern.

        Args:
            symbol: Trading symbol
            position_bias: Current position direction ("long" or "short")
            mtf_data: Multi-timeframe OHLCV data
            current_bar: Current timestamp

        Returns:
            ExitSignal if CHoCH-Against detected, None otherwise
        """

        # Check higher timeframes first (4H, then 1D)
        for tf in ['4H', '1D']:
            if tf not in mtf_data:
                continue

            df = mtf_data[tf]
            if len(df) < 20:  # Need sufficient history
                continue

            signal = self._check_choch_on_timeframe(
                symbol, position_bias, df, current_bar, tf
            )

            if signal:
                return signal

        return None

    def _check_choch_on_timeframe(self, symbol: str, position_bias: str,
                                  df: pd.DataFrame, current_bar: pd.Timestamp,
                                  timeframe: str) -> Optional[ExitSignal]:
        """Check for CHoCH pattern on specific timeframe."""

        try:
            # Get recent data (last 10 bars)
            recent_df = df.tail(10).copy()
            if len(recent_df) < 5:
                return None

            # Identify key levels (swing highs/lows)
            recent_df['swing_high'] = self._identify_swing_highs(recent_df)
            recent_df['swing_low'] = self._identify_swing_lows(recent_df)

            # For long positions, look for bearish CHoCH
            if position_bias == 'long':
                return self._detect_bearish_choch(symbol, recent_df, current_bar, timeframe)

            # For short positions, look for bullish CHoCH
            elif position_bias == 'short':
                return self._detect_bullish_choch(symbol, recent_df, current_bar, timeframe)

        except Exception as e:
            logging.error(f"CHoCH detection error on {timeframe}: {e}")

        return None

    def _detect_bearish_choch(self, symbol: str, df: pd.DataFrame,
                              current_bar: pd.Timestamp, timeframe: str) -> Optional[ExitSignal]:
        """Detect bearish CHoCH (against long position)."""

        # Look for break of recent swing low
        swing_lows = df[df['swing_low'] == True]
        if len(swing_lows) < 2:
            return None

        # Get the most recent significant swing low
        recent_low = swing_lows.iloc[-1]
        recent_low_price = recent_low['low']

        # Check if current price is breaking below this level
        current_low = df.iloc[-1]['low']
        current_close = df.iloc[-1]['close']

        if current_low <= recent_low_price:
            # Potential break detected
            break_strength = self._calculate_break_strength(df, recent_low_price, 'bearish')

            if break_strength >= self.min_break_strength:
                # Check volume confirmation if required
                volume_confirmed = True
                if self.volume_confirmation_required:
                    volume_confirmed = self._check_volume_confirmation(df, 'bearish')

                if volume_confirmed:
                    confidence = min(0.9, break_strength * 1.1)
                    urgency = 0.8 if current_close < recent_low_price else 0.6

                    context = CHoCHContext(
                        timeframe=timeframe,
                        direction='bearish',
                        break_price=recent_low_price,
                        confirmation_price=current_close,
                        structure_strength=break_strength,
                        volume_confirmation=volume_confirmed
                    )

                    # STEP 2: Attach effective parameters to signal for irrefutable tracing
                    effective_params = {
                        "min_break_strength": self.min_break_strength,
                        "bars_confirm": self.bars_confirm,
                        "volume_confirmation": self.volume_confirmation_required,
                        "break_strength": break_strength,
                        "tf": timeframe
                    }

                    return ExitSignal(
                        timestamp=current_bar,
                        symbol=symbol,
                        exit_type=ExitType.CHOCH_AGAINST,
                        action=ExitAction.FULL_EXIT,
                        confidence=confidence,
                        urgency=urgency,
                        reasons=[f"Bearish CHoCH on {timeframe}", f"Break strength: {break_strength:.2f}"],
                        context={
                            'choch': context.__dict__,
                            'effective_params': effective_params  # STEP 2: Irrefutable parameter tracing
                        }
                    )

        return None

    def _detect_bullish_choch(self, symbol: str, df: pd.DataFrame,
                              current_bar: pd.Timestamp, timeframe: str) -> Optional[ExitSignal]:
        """Detect bullish CHoCH (against short position)."""

        # Look for break of recent swing high
        swing_highs = df[df['swing_high'] == True]
        if len(swing_highs) < 2:
            return None

        # Get the most recent significant swing high
        recent_high = swing_highs.iloc[-1]
        recent_high_price = recent_high['high']

        # Check if current price is breaking above this level
        current_high = df.iloc[-1]['high']
        current_close = df.iloc[-1]['close']

        if current_high >= recent_high_price:
            # Potential break detected
            break_strength = self._calculate_break_strength(df, recent_high_price, 'bullish')

            if break_strength >= self.min_break_strength:
                # Check volume confirmation if required
                volume_confirmed = True
                if self.volume_confirmation_required:
                    volume_confirmed = self._check_volume_confirmation(df, 'bullish')

                if volume_confirmed:
                    confidence = min(0.9, break_strength * 1.1)
                    urgency = 0.8 if current_close > recent_high_price else 0.6

                    context = CHoCHContext(
                        timeframe=timeframe,
                        direction='bullish',
                        break_price=recent_high_price,
                        confirmation_price=current_close,
                        structure_strength=break_strength,
                        volume_confirmation=volume_confirmed
                    )

                    # STEP 2: Attach effective parameters to signal for irrefutable tracing
                    effective_params = {
                        "min_break_strength": self.min_break_strength,
                        "bars_confirm": self.bars_confirm,
                        "volume_confirmation": self.volume_confirmation_required,
                        "break_strength": break_strength,
                        "tf": timeframe
                    }

                    return ExitSignal(
                        timestamp=current_bar,
                        symbol=symbol,
                        exit_type=ExitType.CHOCH_AGAINST,
                        action=ExitAction.FULL_EXIT,
                        confidence=confidence,
                        urgency=urgency,
                        reasons=[f"Bullish CHoCH on {timeframe}", f"Break strength: {break_strength:.2f}"],
                        context={
                            'choch': context.__dict__,
                            'effective_params': effective_params  # STEP 2: Irrefutable parameter tracing
                        }
                    )

        return None

    def _identify_swing_highs(self, df: pd.DataFrame, window: int = 3) -> pd.Series:
        """Identify swing highs using rolling window."""
        highs = df['high'].rolling(window=window*2+1, center=True).max()
        return df['high'] == highs

    def _identify_swing_lows(self, df: pd.DataFrame, window: int = 3) -> pd.Series:
        """Identify swing lows using rolling window."""
        lows = df['low'].rolling(window=window*2+1, center=True).min()
        return df['low'] == lows

    def _calculate_break_strength(self, df: pd.DataFrame, level: float, direction: str) -> float:
        """Calculate the strength of a level break."""
        current_bar = df.iloc[-1]

        if direction == 'bearish':
            # For bearish break, measure how far below the level we are
            penetration = (level - current_bar['low']) / level
            volume_ratio = current_bar['volume'] / df['volume'].tail(5).mean()
        else:
            # For bullish break, measure how far above the level we are
            penetration = (current_bar['high'] - level) / level
            volume_ratio = current_bar['volume'] / df['volume'].tail(5).mean()

        # Combine penetration depth and volume
        strength = min(1.0, penetration * 100 + min(0.3, volume_ratio * 0.1))
        return max(0.0, strength)

    def _check_volume_confirmation(self, df: pd.DataFrame, direction: str) -> bool:
        """Check if volume confirms the break."""
        current_volume = df.iloc[-1]['volume']
        avg_volume = df['volume'].tail(10).mean()

        # Volume should be above average for confirmation
        return current_volume > avg_volume * 1.2


class MomentumFadeDetector:
    """
    Detects when momentum is fading, suggesting position should be exited.
    Uses RSI divergence, volume decline, and velocity analysis.
    """

    def __init__(self, config: Dict[str, Any]):
        # STEP 3: Kill silent fallbacks and use sweep parameter names
        # Use drop_pct (from sweep) as the main momentum threshold
        if "drop_pct" not in config:
            raise ValueError(f"MOMENTUM missing required key drop_pct in {list(config.keys())}")

        self.rsi_period = config.get('rsi_period', 14)
        self.drop_pct = float(config["drop_pct"])  # STEP 3: Use sweep parameter name
        self.volume_decline_threshold = config.get('volume_decline_threshold', 0.3)
        self.velocity_threshold = config.get('velocity_threshold', 0.4)
        self.lookback = config.get('lookback', 6)
        self.min_bars_in_pos = config.get('min_bars_in_pos', 4)

        # STEP 2: Echo effective config at init
        effective_config = {
            "rsi_period": self.rsi_period,
            "drop_pct": self.drop_pct,
            "volume_decline_threshold": self.volume_decline_threshold,
            "velocity_threshold": self.velocity_threshold,
            "lookback": self.lookback,
            "min_bars_in_pos": self.min_bars_in_pos
        }
        logging.info("EXIT_EVAL_APPLIED momentum_fade=%s", effective_config)

    def evaluate(self, symbol: str, position_bias: str,
                 mtf_data: Dict[str, pd.DataFrame],
                 current_bar: pd.Timestamp) -> Optional[ExitSignal]:
        """
        Detect momentum fade signals.

        Args:
            symbol: Trading symbol
            position_bias: Current position direction ("long" or "short")
            mtf_data: Multi-timeframe OHLCV data
            current_bar: Current timestamp

        Returns:
            ExitSignal if momentum fade detected, None otherwise
        """

        # Start with execution timeframe (1H)
        if '1H' not in mtf_data:
            return None

        df = mtf_data['1H']
        if len(df) < self.rsi_period + 10:
            return None

        try:
            # Calculate momentum indicators
            df = self._add_momentum_indicators(df.copy())

            # Check for momentum fade
            fade_score = self._calculate_fade_score(df, position_bias)

            # STEP 3: Use drop_pct (from sweep) as the main threshold, not rsi_divergence_threshold
            if fade_score >= self.drop_pct:
                confidence = min(0.85, fade_score)
                urgency = 0.6  # Momentum fades are less urgent than structure breaks

                # Determine action based on fade severity
                if fade_score >= 0.8:
                    action = ExitAction.FULL_EXIT
                else:
                    action = ExitAction.PARTIAL_EXIT
                    exit_percentage = 0.5  # Exit half the position

                context = MomentumContext(
                    current_rsi=df.iloc[-1]['rsi'],
                    rsi_divergence=self._check_rsi_divergence(df, position_bias),
                    volume_decline=self._calculate_volume_decline(df),
                    velocity_slowdown=self._calculate_velocity_decline(df),
                    timeframes_affected=['1H']
                )

                # STEP 2: Attach effective parameters to signal for irrefutable tracing
                effective_params = {
                    "rsi_period": self.rsi_period,
                    "drop_pct": self.drop_pct,
                    "volume_decline_threshold": self.volume_decline_threshold,
                    "velocity_threshold": self.velocity_threshold,
                    "lookback": self.lookback,
                    "min_bars_in_pos": self.min_bars_in_pos,
                    "fade_score": fade_score
                }

                signal = ExitSignal(
                    timestamp=current_bar,
                    symbol=symbol,
                    exit_type=ExitType.MOMENTUM_FADE,
                    action=action,
                    confidence=confidence,
                    urgency=urgency,
                    reasons=[f"Momentum fade score: {fade_score:.2f}"],
                    context={
                        'momentum': context.__dict__,
                        'effective_params': effective_params  # STEP 2: Irrefutable parameter tracing
                    }
                )

                if action == ExitAction.PARTIAL_EXIT:
                    signal.exit_percentage = 0.5

                return signal

        except Exception as e:
            logging.error(f"Momentum fade detection error: {e}")

        return None

    def _add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add RSI and other momentum indicators."""
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # Price velocity (rate of change)
        df['velocity'] = df['close'].pct_change(periods=3)

        return df

    def _calculate_fade_score(self, df: pd.DataFrame, position_bias: str) -> float:
        """Calculate overall momentum fade score."""
        scores = []

        # RSI divergence score
        rsi_div_score = self._get_rsi_divergence_score(df, position_bias)
        scores.append(rsi_div_score)

        # Volume decline score
        vol_decline_score = self._get_volume_decline_score(df)
        scores.append(vol_decline_score)

        # Velocity decline score
        vel_decline_score = self._get_velocity_decline_score(df)
        scores.append(vel_decline_score)

        # Weighted average (RSI divergence most important)
        weights = [0.5, 0.3, 0.2]
        return sum(score * weight for score, weight in zip(scores, weights))

    def _get_rsi_divergence_score(self, df: pd.DataFrame, position_bias: str) -> float:
        """Check for RSI divergence against price."""
        if len(df) < 10:
            return 0.0

        recent_df = df.tail(10)

        if position_bias == 'long':
            # For long positions, look for bearish divergence
            # Price making higher highs, RSI making lower highs
            price_trend = recent_df['close'].iloc[-1] > recent_df['close'].iloc[-5]
            rsi_trend = recent_df['rsi'].iloc[-1] < recent_df['rsi'].iloc[-5]

            if price_trend and rsi_trend:
                # Bearish divergence detected
                divergence_strength = abs(recent_df['rsi'].iloc[-5] - recent_df['rsi'].iloc[-1]) / 20
                return min(1.0, divergence_strength)

        else:  # short position
            # For short positions, look for bullish divergence
            # Price making lower lows, RSI making higher lows
            price_trend = recent_df['close'].iloc[-1] < recent_df['close'].iloc[-5]
            rsi_trend = recent_df['rsi'].iloc[-1] > recent_df['rsi'].iloc[-5]

            if price_trend and rsi_trend:
                # Bullish divergence detected
                divergence_strength = abs(recent_df['rsi'].iloc[-1] - recent_df['rsi'].iloc[-5]) / 20
                return min(1.0, divergence_strength)

        return 0.0

    def _get_volume_decline_score(self, df: pd.DataFrame) -> float:
        """Calculate volume decline score."""
        recent_volume = df['volume'].tail(3).mean()
        baseline_volume = df['volume'].tail(10).mean()

        decline = (baseline_volume - recent_volume) / baseline_volume
        return min(1.0, max(0.0, decline / self.volume_decline_threshold))

    def _get_velocity_decline_score(self, df: pd.DataFrame) -> float:
        """Calculate velocity decline score."""
        recent_velocity = abs(df['velocity'].tail(3).mean())
        baseline_velocity = abs(df['velocity'].tail(10).mean())

        if baseline_velocity == 0:
            return 0.0

        decline = (baseline_velocity - recent_velocity) / baseline_velocity
        return min(1.0, max(0.0, decline / self.velocity_threshold))

    def _check_rsi_divergence(self, df: pd.DataFrame, position_bias: str) -> bool:
        """Simple boolean check for RSI divergence."""
        return self._get_rsi_divergence_score(df, position_bias) > 0.5

    def _calculate_volume_decline(self, df: pd.DataFrame) -> float:
        """Calculate percentage volume decline."""
        recent_volume = df['volume'].tail(3).mean()
        baseline_volume = df['volume'].tail(10).mean()
        return (baseline_volume - recent_volume) / baseline_volume

    def _calculate_velocity_decline(self, df: pd.DataFrame) -> float:
        """Calculate velocity decline percentage."""
        recent_velocity = abs(df['velocity'].tail(3).mean())
        baseline_velocity = abs(df['velocity'].tail(10).mean())

        if baseline_velocity == 0:
            return 0.0

        return (baseline_velocity - recent_velocity) / baseline_velocity


class TimeStopEvaluator:
    """
    Implements time-based stops to prevent positions from staying open too long.
    Considers time decay, performance vs time spent, and maximum hold periods.
    """

    def __init__(self, config: Dict[str, Any]):
        # Store config for later access
        self.config = config

        # STEP 2: Remove silent fallbacks and use ONLY backtest-level config
        # No strategy-level shadowing, no legacy fallbacks unless explicitly missing
        if "max_bars_1h" not in config:
            raise ValueError(f"TIME_STOP missing required key max_bars_1h in {list(config.keys())}")

        self.max_bars_1h = int(config["max_bars_1h"])
        self.max_bars_4h = int(config.get("max_bars_4h", 42))
        self.max_bars_1d = int(config.get("max_bars_1d", 10))
        self.performance_threshold = config.get('performance_threshold', 0.1)  # 10% gain to justify time
        self.time_decay_start = config.get('time_decay_start', 0.7)  # Start decay at 70% of max time

        # STEP 2: Echo effective config at init
        effective_config = {
            "max_bars_1h": self.max_bars_1h,
            "max_bars_4h": self.max_bars_4h,
            "max_bars_1d": self.max_bars_1d,
            "performance_threshold": self.performance_threshold,
            "time_decay_start": self.time_decay_start
        }
        logging.info("EXIT_EVAL_APPLIED time_stop=%s", effective_config)

    def evaluate(self, symbol: str, position_data: Dict[str, Any],
                 current_bar: pd.Timestamp) -> Optional[ExitSignal]:
        """
        Evaluate time-based exit conditions.

        Args:
            symbol: Trading symbol
            position_data: Position information including entry time, PnL
            current_bar: Current timestamp

        Returns:
            ExitSignal if time stop triggered, None otherwise
        """

        # Use new position aging system
        bars_held = position_data.get('bars_held', 0)
        timeframe = position_data.get('timeframe', '1H')
        current_pnl_pct = position_data.get('pnl_pct', 0.0)

        # Debug logging for time stop evaluation
        logging.info(f"[TIME_STOP_DEBUG] sym={symbol} bar={current_bar} "
                     f"bars_held={bars_held} timeframe={timeframe} pnl_pct={current_pnl_pct}")

        # Determine max bars based on timeframe (with backward compatibility)
        tf_key_map = {
            "1H": "max_bars_1h",
            "4H": "max_bars_4h",
            "1D": "max_bars_1d",
        }
        tf_key = tf_key_map.get(timeframe)

        if tf_key and tf_key in self.config:
            max_bars = self.config[tf_key]
        else:
            # Fall back to legacy bars_max or defaults
            max_bars = self.config.get("bars_max", self.max_bars_1h)

        # Calculate time decay factor using bars_held
        time_ratio = bars_held / max_bars if max_bars > 0 else 0

        logging.info(f"[TIME_STOP_DEBUG] {symbol}: bars_held={bars_held} "
                     f"max_bars={max_bars} time_ratio={time_ratio:.3f}")

        try:
            if time_ratio >= 1.0:
                # Maximum time exceeded - hard stop
                confidence = 0.9
                urgency = 0.9
                action = ExitAction.FULL_EXIT
                reason = f"Maximum hold time exceeded ({bars_held} bars > {max_bars} bars)"

            elif time_ratio >= self.time_decay_start:
                # In decay zone - evaluate performance vs time
                performance_vs_time = current_pnl_pct / time_ratio if time_ratio > 0 else 0

                if performance_vs_time < self.performance_threshold:
                    # Poor performance relative to time spent
                    confidence = 0.7 + (time_ratio - self.time_decay_start) * 0.3
                    urgency = 0.5
                    action = ExitAction.PARTIAL_EXIT
                    reason = f"Time decay: {time_ratio:.1%} of max time, performance: {performance_vs_time:.1%}"
                else:
                    # Good performance, let it run a bit longer
                    return None
            else:
                # Still within acceptable time range
                return None

            context = TimeStopContext(
                bars_in_trade=bars_held,
                max_bars_allowed=max_bars,
                time_decay_factor=time_ratio,
                performance_vs_time=current_pnl_pct / time_ratio if time_ratio > 0 else 0
            )

            # STEP 2: Attach effective parameters to signal for irrefutable tracing
            effective_params = {
                "max_bars_1h": self.max_bars_1h,
                "max_bars_4h": self.max_bars_4h,
                "max_bars_1d": self.max_bars_1d,
                "performance_threshold": self.performance_threshold,
                "time_decay_start": self.time_decay_start,
                "bars_held": bars_held,
                "limit": max_bars,
                "tf": timeframe
            }

            signal = ExitSignal(
                timestamp=current_bar,
                symbol=symbol,
                exit_type=ExitType.TIME_STOP,
                action=action,
                confidence=confidence,
                urgency=urgency,
                reasons=[reason],
                context={
                    'time_stop': context.__dict__,
                    'effective_params': effective_params  # STEP 2: Irrefutable parameter tracing
                }
            )

            if action == ExitAction.PARTIAL_EXIT:
                signal.exit_percentage = 0.5

            return signal

        except Exception as e:
            logging.error(f"Time stop evaluation error: {e}")

        return None
