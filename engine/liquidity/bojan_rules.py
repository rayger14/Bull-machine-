"""
Bojan Liquidity Engine - Rule-Based Reaction System

Implements Bojan's demand→HOB→reaction logic with institutional
order flow analysis and position management rules.
"""

import pandas as pd
import numpy as np
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import logging

from .hob import HOBSignal, HOBType, HOBQuality, LiquidityLevel

logger = logging.getLogger(__name__)

class ReactionType(Enum):
    """Types of liquidity reactions"""
    DEMAND_REACTION = "demand_reaction"
    SUPPLY_REACTION = "supply_reaction"
    LIQUIDITY_GRAB = "liquidity_grab"
    FALSE_BREAKOUT = "false_breakout"
    CONTINUATION = "continuation"

class ExitTrigger(Enum):
    """Exit trigger types for Bojan rules"""
    PARTIAL_PROFIT = "partial_profit"
    FULL_EXIT = "full_exit"
    STOP_LOSS = "stop_loss"
    TRAIL_STOP = "trail_stop"
    TIME_EXIT = "time_exit"

@dataclass
class LiquidityReaction:
    """Liquidity reaction signal"""
    reaction_type: ReactionType
    timestamp: pd.Timestamp
    entry_price: float
    direction: str  # 'long' or 'short'
    confidence: float
    strength: float
    exit_levels: Dict[str, float]
    hob_signal: Optional[HOBSignal]
    metadata: Dict[str, Any]

@dataclass
class BojanExit:
    """Bojan exit signal"""
    trigger: ExitTrigger
    timestamp: pd.Timestamp
    exit_price: float
    percentage: float  # 0-100, percentage of position to exit
    reason: str
    confidence: float
    metadata: Dict[str, Any]

class BojanEngine:
    """
    Advanced Bojan Liquidity Engine implementing institutional order flow logic.

    Combines HOB detection with reaction analysis for high-probability setups:
    - Demand/supply zone validation
    - HOB quality assessment
    - Institutional reaction detection
    - Dynamic exit management
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.bojan_config = config.get('bojan_engine', {})

        # Reaction detection parameters
        self.min_reaction_strength = self.bojan_config.get('min_reaction_strength', 0.6)
        self.reaction_timeframe = self.bojan_config.get('reaction_timeframe_bars', 12)
        self.volume_confirmation = self.bojan_config.get('require_volume_confirmation', True)

        # Exit management parameters
        self.partial_exit_levels = self.bojan_config.get('partial_exit_levels', [0.5, 1.0, 1.5])
        self.partial_exit_percentages = self.bojan_config.get('partial_exit_percentages', [30, 40, 30])
        self.trail_stop_activation = self.bojan_config.get('trail_stop_activation', 1.5)
        self.trail_stop_distance = self.bojan_config.get('trail_stop_distance', 0.5)

        # Risk management
        self.max_hold_bars = self.bojan_config.get('max_hold_bars', 72)  # 3 days for 1H
        self.adverse_move_threshold = self.bojan_config.get('adverse_move_threshold', 0.3)

        # State tracking
        self.active_positions = {}
        self.reaction_history = []

    def analyze_liquidity_reaction(self, hob_signal: HOBSignal, current_data: Dict[str, pd.DataFrame],
                                  historical_data: Dict[str, pd.DataFrame]) -> Optional[LiquidityReaction]:
        """
        Analyze liquidity reaction following HOB detection.

        Args:
            hob_signal: Detected HOB signal
            current_data: Current timeframe data
            historical_data: Historical data for pattern validation

        Returns:
            LiquidityReaction if valid reaction detected
        """
        try:
            # Get primary timeframe data
            primary_tf = '1H'
            if primary_tf not in current_data:
                logger.warning(f"No data for primary timeframe {primary_tf}")
                return None

            df = current_data[primary_tf]
            current_time = df.index[-1]
            current_price = df['close'].iloc[-1]

            # 1. Validate demand/supply zone strength
            zone_validation = self._validate_demand_supply_zone(hob_signal, df, historical_data)
            if zone_validation['score'] < 0.6:
                return None

            # 2. Detect institutional reaction patterns
            reaction_analysis = self._detect_institutional_reaction(hob_signal, df)
            if not reaction_analysis['detected']:
                return None

            # 3. Analyze order flow confirmation
            order_flow = self._analyze_order_flow_confirmation(hob_signal, df)

            # 4. Calculate reaction strength and confidence
            reaction_strength = self._calculate_reaction_strength(
                zone_validation, reaction_analysis, order_flow
            )

            if reaction_strength < self.min_reaction_strength:
                return None

            # 5. Determine reaction type and direction
            reaction_type, direction = self._classify_reaction(hob_signal, reaction_analysis)

            # 6. Calculate entry and exit levels
            entry_price, exit_levels = self._calculate_bojan_levels(
                hob_signal, current_price, direction, df
            )

            # 7. Calculate final confidence
            confidence = min(0.95, reaction_strength * hob_signal.confidence)

            return LiquidityReaction(
                reaction_type=reaction_type,
                timestamp=current_time,
                entry_price=entry_price,
                direction=direction,
                confidence=confidence,
                strength=reaction_strength,
                exit_levels=exit_levels,
                hob_signal=hob_signal,
                metadata={
                    'zone_validation': zone_validation,
                    'reaction_analysis': reaction_analysis,
                    'order_flow': order_flow,
                    'hob_quality': hob_signal.quality.value,
                    'institutional_score': reaction_analysis.get('institutional_score', 0)
                }
            )

        except Exception as e:
            logger.error(f"Error analyzing liquidity reaction: {e}")
            return None

    def generate_exit_signals(self, position_id: str, current_data: Dict[str, pd.DataFrame],
                             reaction: LiquidityReaction) -> List[BojanExit]:
        """
        Generate dynamic exit signals based on Bojan rules.

        Args:
            position_id: Unique position identifier
            current_data: Current market data
            reaction: Original liquidity reaction

        Returns:
            List of exit signals
        """
        try:
            exits = []
            primary_tf = '1H'

            if primary_tf not in current_data:
                return exits

            df = current_data[primary_tf]
            current_time = df.index[-1]
            current_price = df['close'].iloc[-1]

            # Calculate current P&L
            if reaction.direction == 'long':
                pnl_ratio = (current_price - reaction.entry_price) / reaction.entry_price
            else:
                pnl_ratio = (reaction.entry_price - current_price) / reaction.entry_price

            # 1. Check for partial profit exits
            partial_exits = self._check_partial_exits(pnl_ratio, current_time, current_price, reaction)
            exits.extend(partial_exits)

            # 2. Check for adverse move stop
            adverse_exit = self._check_adverse_move(pnl_ratio, current_time, current_price, reaction)
            if adverse_exit:
                exits.append(adverse_exit)

            # 3. Check for time-based exits
            time_exit = self._check_time_exit(position_id, current_time, current_price, reaction)
            if time_exit:
                exits.append(time_exit)

            # 4. Check for trail stop activation
            trail_exit = self._check_trail_stop(position_id, current_data, reaction)
            if trail_exit:
                exits.append(trail_exit)

            # 5. Check for liquidity grab reversals
            reversal_exit = self._check_liquidity_reversal(current_data, reaction)
            if reversal_exit:
                exits.append(reversal_exit)

            return exits

        except Exception as e:
            logger.error(f"Error generating exit signals: {e}")
            return []

    def _validate_demand_supply_zone(self, hob_signal: HOBSignal, df: pd.DataFrame,
                                   historical_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Validate strength of demand/supply zone"""
        try:
            zone_price = hob_signal.liquidity_levels[0].price
            zone_type = hob_signal.liquidity_levels[0].level_type

            # Historical validation
            historical_score = 0.0
            if '1D' in historical_data:
                daily_data = historical_data['1D']
                historical_score = self._analyze_historical_zone_strength(daily_data, zone_price, zone_type)

            # Current timeframe validation
            current_score = self._analyze_current_zone_strength(df, zone_price, zone_type)

            # Age factor (newer zones often stronger)
            age_hours = hob_signal.liquidity_levels[0].age_hours
            age_factor = max(0.3, 1.0 - (age_hours / 168))  # Decay over 7 days

            # Combine scores
            overall_score = (historical_score * 0.4 + current_score * 0.4 + age_factor * 0.2)

            return {
                'score': overall_score,
                'historical_score': historical_score,
                'current_score': current_score,
                'age_factor': age_factor,
                'zone_touches': hob_signal.liquidity_levels[0].touches
            }

        except Exception as e:
            logger.error(f"Error validating demand/supply zone: {e}")
            return {'score': 0.0}

    def _detect_institutional_reaction(self, hob_signal: HOBSignal, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect institutional reaction patterns"""
        try:
            # Look for reaction in recent bars
            reaction_bars = min(self.reaction_timeframe, len(df))
            recent_data = df.tail(reaction_bars)

            if len(recent_data) < 3:
                return {'detected': False, 'institutional_score': 0.0}

            # Analyze volume surge
            volume_surge = self._analyze_volume_surge(recent_data)

            # Analyze price rejection
            price_rejection = self._analyze_price_rejection(recent_data, hob_signal)

            # Analyze momentum shift
            momentum_shift = self._analyze_momentum_shift(recent_data, hob_signal)

            # Analyze wick formation
            wick_analysis = self._analyze_institutional_wicks(recent_data, hob_signal)

            # Calculate institutional score
            institutional_score = (
                volume_surge * 0.3 +
                price_rejection * 0.25 +
                momentum_shift * 0.25 +
                wick_analysis * 0.2
            )

            detected = institutional_score > 0.6

            return {
                'detected': detected,
                'institutional_score': institutional_score,
                'volume_surge': volume_surge,
                'price_rejection': price_rejection,
                'momentum_shift': momentum_shift,
                'wick_analysis': wick_analysis
            }

        except Exception as e:
            logger.error(f"Error detecting institutional reaction: {e}")
            return {'detected': False, 'institutional_score': 0.0}

    def _analyze_order_flow_confirmation(self, hob_signal: HOBSignal, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze order flow for reaction confirmation"""
        try:
            recent_data = df.tail(5)

            # Simplified order flow analysis using OHLC
            if hob_signal.hob_type == HOBType.BULLISH_HOB:
                # Look for buying pressure
                buying_pressure = self._calculate_buying_pressure(recent_data)
                return {
                    'confirmed': buying_pressure > 0.6,
                    'strength': buying_pressure,
                    'type': 'bullish_flow'
                }
            else:
                # Look for selling pressure
                selling_pressure = self._calculate_selling_pressure(recent_data)
                return {
                    'confirmed': selling_pressure > 0.6,
                    'strength': selling_pressure,
                    'type': 'bearish_flow'
                }

        except Exception as e:
            logger.error(f"Error analyzing order flow: {e}")
            return {'confirmed': False, 'strength': 0.0}

    def _calculate_reaction_strength(self, zone_validation: Dict, reaction_analysis: Dict,
                                   order_flow: Dict) -> float:
        """Calculate overall reaction strength"""
        try:
            zone_score = zone_validation.get('score', 0.0)
            institutional_score = reaction_analysis.get('institutional_score', 0.0)
            flow_score = order_flow.get('strength', 0.0)

            # Weighted combination
            reaction_strength = (
                zone_score * 0.4 +
                institutional_score * 0.4 +
                flow_score * 0.2
            )

            return min(1.0, reaction_strength)

        except Exception:
            return 0.0

    def _classify_reaction(self, hob_signal: HOBSignal, reaction_analysis: Dict) -> Tuple[ReactionType, str]:
        """Classify the type of reaction and determine direction"""
        try:
            if hob_signal.hob_type == HOBType.BULLISH_HOB:
                return ReactionType.DEMAND_REACTION, 'long'
            elif hob_signal.hob_type == HOBType.BEARISH_HOB:
                return ReactionType.SUPPLY_REACTION, 'short'
            else:
                # Default for potential HOB
                return ReactionType.CONTINUATION, 'long'

        except Exception:
            return ReactionType.CONTINUATION, 'long'

    def _calculate_bojan_levels(self, hob_signal: HOBSignal, current_price: float,
                               direction: str, df: pd.DataFrame) -> Tuple[float, Dict[str, float]]:
        """Calculate entry and exit levels using Bojan methodology"""
        try:
            atr = self._calculate_atr(df, 14)
            level_price = hob_signal.liquidity_levels[0].price

            if direction == 'long':
                entry_price = current_price
                stop_loss = level_price - (atr * 1.2)

                # Multiple profit targets
                exit_levels = {
                    'stop_loss': stop_loss,
                    'target_1': entry_price + (atr * self.partial_exit_levels[0]),
                    'target_2': entry_price + (atr * self.partial_exit_levels[1]),
                    'target_3': entry_price + (atr * self.partial_exit_levels[2])
                }
            else:  # short
                entry_price = current_price
                stop_loss = level_price + (atr * 1.2)

                exit_levels = {
                    'stop_loss': stop_loss,
                    'target_1': entry_price - (atr * self.partial_exit_levels[0]),
                    'target_2': entry_price - (atr * self.partial_exit_levels[1]),
                    'target_3': entry_price - (atr * self.partial_exit_levels[2])
                }

            return entry_price, exit_levels

        except Exception as e:
            logger.error(f"Error calculating Bojan levels: {e}")
            return current_price, {}

    def _check_partial_exits(self, pnl_ratio: float, current_time: pd.Timestamp,
                           current_price: float, reaction: LiquidityReaction) -> List[BojanExit]:
        """Check for partial profit exit opportunities"""
        exits = []

        try:
            # Check each profit target
            for i, (target_key, target_price) in enumerate(reaction.exit_levels.items()):
                if not target_key.startswith('target_'):
                    continue

                target_index = int(target_key.split('_')[1]) - 1
                if target_index >= len(self.partial_exit_percentages):
                    continue

                # Check if target reached
                target_reached = False
                if reaction.direction == 'long':
                    target_reached = current_price >= target_price
                else:
                    target_reached = current_price <= target_price

                if target_reached:
                    percentage = self.partial_exit_percentages[target_index]
                    exits.append(BojanExit(
                        trigger=ExitTrigger.PARTIAL_PROFIT,
                        timestamp=current_time,
                        exit_price=current_price,
                        percentage=percentage,
                        reason=f"Partial exit at {target_key}",
                        confidence=0.9,
                        metadata={'target_level': target_price}
                    ))

        except Exception as e:
            logger.error(f"Error checking partial exits: {e}")

        return exits

    def _check_adverse_move(self, pnl_ratio: float, current_time: pd.Timestamp,
                          current_price: float, reaction: LiquidityReaction) -> Optional[BojanExit]:
        """Check for adverse move stop loss"""
        try:
            if pnl_ratio <= -self.adverse_move_threshold:
                return BojanExit(
                    trigger=ExitTrigger.STOP_LOSS,
                    timestamp=current_time,
                    exit_price=current_price,
                    percentage=100,
                    reason=f"Adverse move stop: {pnl_ratio:.2%}",
                    confidence=0.95,
                    metadata={'pnl_ratio': pnl_ratio}
                )
            return None

        except Exception:
            return None

    def _check_time_exit(self, position_id: str, current_time: pd.Timestamp,
                        current_price: float, reaction: LiquidityReaction) -> Optional[BojanExit]:
        """Check for time-based exits"""
        try:
            position_age = (current_time - reaction.timestamp).total_seconds() / 3600
            max_hours = self.max_hold_bars

            if position_age >= max_hours:
                return BojanExit(
                    trigger=ExitTrigger.TIME_EXIT,
                    timestamp=current_time,
                    exit_price=current_price,
                    percentage=100,
                    reason=f"Time exit after {position_age:.1f} hours",
                    confidence=0.8,
                    metadata={'position_age_hours': position_age}
                )
            return None

        except Exception:
            return None

    def _check_trail_stop(self, position_id: str, current_data: Dict[str, pd.DataFrame],
                         reaction: LiquidityReaction) -> Optional[BojanExit]:
        """Check for trailing stop activation"""
        try:
            # Simplified trailing stop logic
            # In production, this would track high-water marks
            return None

        except Exception:
            return None

    def _check_liquidity_reversal(self, current_data: Dict[str, pd.DataFrame],
                                 reaction: LiquidityReaction) -> Optional[BojanExit]:
        """Check for liquidity grab reversals"""
        try:
            # Advanced reversal detection would go here
            # For now, return None
            return None

        except Exception:
            return None

    # Helper methods
    def _analyze_historical_zone_strength(self, daily_data: pd.DataFrame, zone_price: float, zone_type: str) -> float:
        """Analyze historical strength of zone on higher timeframes"""
        try:
            tolerance = 0.01  # 1%

            if zone_type == 'support':
                touches = ((daily_data['low'] <= zone_price * (1 + tolerance)) &
                          (daily_data['low'] >= zone_price * (1 - tolerance))).sum()
            else:
                touches = ((daily_data['high'] >= zone_price * (1 - tolerance)) &
                          (daily_data['high'] <= zone_price * (1 + tolerance))).sum()

            return min(1.0, touches / 5.0)

        except Exception:
            return 0.0

    def _analyze_current_zone_strength(self, df: pd.DataFrame, zone_price: float, zone_type: str) -> float:
        """Analyze current timeframe zone strength"""
        try:
            recent_data = df.tail(50)
            tolerance = 0.005  # 0.5%

            if zone_type == 'support':
                near_touches = ((recent_data['low'] <= zone_price * (1 + tolerance)) &
                               (recent_data['low'] >= zone_price * (1 - tolerance))).sum()
            else:
                near_touches = ((recent_data['high'] >= zone_price * (1 - tolerance)) &
                               (recent_data['high'] <= zone_price * (1 + tolerance))).sum()

            return min(1.0, near_touches / 3.0)

        except Exception:
            return 0.0

    def _analyze_volume_surge(self, recent_data: pd.DataFrame) -> float:
        """Analyze volume surge in reaction"""
        try:
            recent_volume = recent_data['volume'].tail(3).mean()
            baseline_volume = recent_data['volume'].mean()

            surge_ratio = recent_volume / baseline_volume if baseline_volume > 0 else 1.0
            return min(1.0, surge_ratio / 2.0)  # Cap at 2x volume

        except Exception:
            return 0.0

    def _analyze_price_rejection(self, recent_data: pd.DataFrame, hob_signal: HOBSignal) -> float:
        """Analyze price rejection from level"""
        try:
            level_price = hob_signal.liquidity_levels[0].price
            current_price = recent_data['close'].iloc[-1]

            distance = abs(current_price - level_price) / level_price
            rejection_strength = min(1.0, distance / 0.02)  # 2% max distance

            return rejection_strength

        except Exception:
            return 0.0

    def _analyze_momentum_shift(self, recent_data: pd.DataFrame, hob_signal: HOBSignal) -> float:
        """Analyze momentum shift"""
        try:
            if len(recent_data) < 5:
                return 0.0

            # Simple momentum using price changes
            price_changes = recent_data['close'].pct_change().tail(3)
            momentum_direction = 1 if price_changes.sum() > 0 else -1

            # Check if momentum aligns with expected HOB direction
            expected_direction = 1 if hob_signal.hob_type == HOBType.BULLISH_HOB else -1

            if momentum_direction == expected_direction:
                return abs(price_changes.mean()) * 100  # Convert to percentage
            else:
                return 0.0

        except Exception:
            return 0.0

    def _analyze_institutional_wicks(self, recent_data: pd.DataFrame, hob_signal: HOBSignal) -> float:
        """Analyze institutional wick formations"""
        try:
            latest_bar = recent_data.iloc[-1]

            if hob_signal.hob_type == HOBType.BULLISH_HOB:
                # Look for rejection wicks at support
                body_size = abs(latest_bar['close'] - latest_bar['open'])
                lower_wick = min(latest_bar['open'], latest_bar['close']) - latest_bar['low']
                wick_ratio = lower_wick / body_size if body_size > 0 else 0
            else:
                # Look for rejection wicks at resistance
                body_size = abs(latest_bar['close'] - latest_bar['open'])
                upper_wick = latest_bar['high'] - max(latest_bar['open'], latest_bar['close'])
                wick_ratio = upper_wick / body_size if body_size > 0 else 0

            return min(1.0, wick_ratio / 1.5)  # Normalize to 1.5:1 ratio

        except Exception:
            return 0.0

    def _calculate_buying_pressure(self, recent_data: pd.DataFrame) -> float:
        """Calculate buying pressure from OHLC data"""
        try:
            pressure_scores = []

            for _, bar in recent_data.iterrows():
                # Analyze where close is relative to high-low range
                range_size = bar['high'] - bar['low']
                if range_size == 0:
                    continue

                close_position = (bar['close'] - bar['low']) / range_size
                pressure_scores.append(close_position)

            return np.mean(pressure_scores) if pressure_scores else 0.0

        except Exception:
            return 0.0

    def _calculate_selling_pressure(self, recent_data: pd.DataFrame) -> float:
        """Calculate selling pressure from OHLC data"""
        try:
            return 1.0 - self._calculate_buying_pressure(recent_data)

        except Exception:
            return 0.0

    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        try:
            high_low = data['high'] - data['low']
            high_close = (data['high'] - data['close'].shift(1)).abs()
            low_close = (data['low'] - data['close'].shift(1)).abs()

            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(period).mean().iloc[-1]

            return atr if not pd.isna(atr) else data['high'].iloc[-1] - data['low'].iloc[-1]

        except Exception:
            return 0.001