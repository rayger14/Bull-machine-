"""
Macro Context Analysis - SMT Data Processing and Regime Detection

Provides advanced analysis capabilities for Smart Money Theory signals
including regime classification and context filtering.
"""

import pandas as pd
import numpy as np
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

from .signals import SMTSignal, SMTSignalType, HPS_Score

logger = logging.getLogger(__name__)

class MacroRegime(Enum):
    """Macro market regime classifications"""
    ACCUMULATION = "accumulation"
    MARKUP = "markup"
    DISTRIBUTION = "distribution"
    MARKDOWN = "markdown"
    NEUTRAL = "neutral"
    TRANSITION = "transition"

@dataclass
class RegimeSignal:
    """Regime classification signal"""
    regime: MacroRegime
    confidence: float
    timestamp: pd.Timestamp
    duration_hours: int
    strength_indicators: Dict[str, float]

class SMTAnalyzer:
    """
    Advanced SMT analysis for regime detection and context filtering.

    Provides higher-level analysis of SMT signals for:
    - Market regime classification
    - CRT (Composite Reaccumulation Time) detection
    - Premium/discount environment assessment
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.analysis_config = config.get('smt_analysis', {})

        # Regime detection parameters
        self.regime_lookback = self.analysis_config.get('regime_lookback_hours', 168)  # 7 days
        self.regime_confidence_threshold = self.analysis_config.get('regime_confidence', 0.7)

        # CRT detection parameters
        self.crt_min_duration = self.analysis_config.get('crt_min_hours', 72)  # 3 days
        self.crt_max_volatility = self.analysis_config.get('crt_max_volatility', 0.15)

        # State tracking
        self.regime_history = []
        self.current_regime = None

    def classify_regime(self, smt_signals: List[SMTSignal], market_data: Dict[str, pd.DataFrame]) -> RegimeSignal:
        """
        Classify current macro regime based on SMT signals and market structure.

        Args:
            smt_signals: List of current SMT signals
            market_data: Dict with BTC, USDT.D, BTC.D, TOTAL3 data

        Returns:
            RegimeSignal with classification and confidence
        """
        try:
            if not market_data or 'BTC' not in market_data:
                return self._default_regime_signal()

            btc_data = market_data['BTC']
            current_time = btc_data.index[-1]

            # Analyze market structure components
            structure_analysis = self._analyze_market_structure(btc_data)
            dominance_analysis = self._analyze_dominance_trends(market_data)
            liquidity_analysis = self._analyze_liquidity_conditions(market_data)

            # Weight SMT signals
            signal_weight = self._weight_smt_signals(smt_signals)

            # Combine analysis for regime classification
            regime_scores = {
                MacroRegime.ACCUMULATION: self._score_accumulation(
                    structure_analysis, dominance_analysis, liquidity_analysis, signal_weight
                ),
                MacroRegime.MARKUP: self._score_markup(
                    structure_analysis, dominance_analysis, liquidity_analysis, signal_weight
                ),
                MacroRegime.DISTRIBUTION: self._score_distribution(
                    structure_analysis, dominance_analysis, liquidity_analysis, signal_weight
                ),
                MacroRegime.MARKDOWN: self._score_markdown(
                    structure_analysis, dominance_analysis, liquidity_analysis, signal_weight
                )
            }

            # Determine regime with highest score
            best_regime = max(regime_scores.keys(), key=lambda x: regime_scores[x])
            confidence = regime_scores[best_regime]

            # Check for transition state
            if confidence < self.regime_confidence_threshold:
                best_regime = MacroRegime.TRANSITION
                confidence = 1 - confidence

            # Calculate duration
            duration_hours = self._calculate_regime_duration(best_regime, current_time)

            regime_signal = RegimeSignal(
                regime=best_regime,
                confidence=confidence,
                timestamp=current_time,
                duration_hours=duration_hours,
                strength_indicators={
                    'structure': structure_analysis.get('strength', 0),
                    'dominance': dominance_analysis.get('strength', 0),
                    'liquidity': liquidity_analysis.get('strength', 0),
                    'smt_signal': signal_weight
                }
            )

            # Update history
            self.current_regime = regime_signal
            self.regime_history.append(regime_signal)

            return regime_signal

        except Exception as e:
            logger.error(f"Error in regime classification: {e}")
            return self._default_regime_signal()

    def detect_crt_phase(self, market_data: Dict[str, pd.DataFrame]) -> Optional[Dict[str, Any]]:
        """
        Detect Composite Reaccumulation Time (CRT) during SMR phases.

        CRT indicates periods of low volatility accumulation before major moves.
        """
        try:
            if 'BTC' not in market_data:
                return None

            btc_data = market_data['BTC']
            if len(btc_data) < self.crt_min_duration:
                return None

            # Analyze recent volatility
            recent_data = btc_data.tail(self.crt_min_duration)
            returns = recent_data['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(24)  # Annualized for hourly data

            # Check for low volatility conditions
            if volatility <= self.crt_max_volatility:
                # Additional CRT indicators
                range_contraction = self._analyze_range_contraction(recent_data)
                volume_profile = self._analyze_volume_profile(recent_data)
                order_flow = self._analyze_order_flow_balance(recent_data)

                crt_strength = (range_contraction + volume_profile + order_flow) / 3

                if crt_strength > 0.6:  # Strong CRT signal
                    return {
                        'detected': True,
                        'strength': crt_strength,
                        'duration_hours': len(recent_data),
                        'volatility': volatility,
                        'range_contraction': range_contraction,
                        'volume_profile': volume_profile,
                        'order_flow': order_flow,
                        'timestamp': btc_data.index[-1]
                    }

            return None

        except Exception as e:
            logger.error(f"Error in CRT detection: {e}")
            return None

    def assess_premium_discount(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Assess premium/discount environment for better trade timing.

        Analyzes funding rates, futures basis, and options flow.
        """
        try:
            # Simplified premium/discount assessment
            # In production, this would include funding rates, futures basis, etc.

            if 'BTC' not in market_data:
                return {'environment': 'neutral', 'strength': 0.0}

            btc_data = market_data['BTC']
            recent_data = btc_data.tail(168)  # 7 days

            # Price momentum analysis as proxy
            short_ma = recent_data['close'].tail(24).mean()
            long_ma = recent_data['close'].tail(168).mean()
            momentum = (short_ma / long_ma - 1)

            # Volume-weighted assessment
            volume_trend = recent_data['volume'].tail(24).mean() / recent_data['volume'].tail(168).mean()

            if momentum > 0.02 and volume_trend > 1.2:
                environment = 'premium'
                strength = min(0.9, momentum * 10 + (volume_trend - 1))
            elif momentum < -0.02 and volume_trend > 1.2:
                environment = 'discount'
                strength = min(0.9, abs(momentum) * 10 + (volume_trend - 1))
            else:
                environment = 'neutral'
                strength = 0.5

            return {
                'environment': environment,
                'strength': strength,
                'momentum': momentum,
                'volume_trend': volume_trend,
                'timestamp': btc_data.index[-1]
            }

        except Exception as e:
            logger.error(f"Error in premium/discount assessment: {e}")
            return {'environment': 'neutral', 'strength': 0.0}

    def _analyze_market_structure(self, btc_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze market structure for regime classification"""
        try:
            recent_data = btc_data.tail(self.regime_lookback)

            # Higher highs/higher lows analysis
            highs = recent_data['high'].rolling(24).max()
            lows = recent_data['low'].rolling(24).min()

            higher_highs = (highs.diff() > 0).sum()
            higher_lows = (lows.diff() > 0).sum()
            lower_highs = (highs.diff() < 0).sum()
            lower_lows = (lows.diff() < 0).sum()

            total_swings = higher_highs + higher_lows + lower_highs + lower_lows
            if total_swings == 0:
                return {'trend': 'neutral', 'strength': 0.0}

            uptrend_score = (higher_highs + higher_lows) / total_swings
            downtrend_score = (lower_highs + lower_lows) / total_swings

            if uptrend_score > 0.6:
                trend = 'bullish'
                strength = uptrend_score
            elif downtrend_score > 0.6:
                trend = 'bearish'
                strength = downtrend_score
            else:
                trend = 'sideways'
                strength = 1 - max(uptrend_score, downtrend_score)

            return {
                'trend': trend,
                'strength': strength,
                'higher_highs': higher_highs,
                'higher_lows': higher_lows,
                'lower_highs': lower_highs,
                'lower_lows': lower_lows
            }

        except Exception as e:
            logger.error(f"Error in market structure analysis: {e}")
            return {'trend': 'neutral', 'strength': 0.0}

    def _analyze_dominance_trends(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Analyze BTC dominance trends"""
        try:
            if 'BTC.D' not in market_data:
                return {'trend': 'neutral', 'strength': 0.0}

            btc_dom = market_data['BTC.D'].tail(self.regime_lookback)
            dom_change = (btc_dom['close'].iloc[-1] / btc_dom['close'].iloc[0] - 1)

            if dom_change > 0.05:  # 5% increase
                trend = 'btc_gaining'
                strength = min(0.9, dom_change * 5)
            elif dom_change < -0.05:  # 5% decrease
                trend = 'alts_gaining'
                strength = min(0.9, abs(dom_change) * 5)
            else:
                trend = 'stable'
                strength = 0.5

            return {
                'trend': trend,
                'strength': strength,
                'change_pct': dom_change
            }

        except Exception as e:
            logger.error(f"Error in dominance analysis: {e}")
            return {'trend': 'neutral', 'strength': 0.0}

    def _analyze_liquidity_conditions(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Analyze liquidity conditions via USDT.D and volume"""
        try:
            if 'USDT.D' not in market_data:
                return {'condition': 'neutral', 'strength': 0.0}

            usdt_data = market_data['USDT.D'].tail(self.regime_lookback)
            usdt_trend = (usdt_data['close'].iloc[-1] / usdt_data['close'].iloc[0] - 1)

            # Rising USDT.D = liquidity tightening, falling = loosening
            if usdt_trend > 0.02:
                condition = 'tightening'
                strength = min(0.9, usdt_trend * 10)
            elif usdt_trend < -0.02:
                condition = 'loosening'
                strength = min(0.9, abs(usdt_trend) * 10)
            else:
                condition = 'stable'
                strength = 0.5

            return {
                'condition': condition,
                'strength': strength,
                'usdt_trend': usdt_trend
            }

        except Exception as e:
            logger.error(f"Error in liquidity analysis: {e}")
            return {'condition': 'neutral', 'strength': 0.0}

    def _weight_smt_signals(self, smt_signals: List[SMTSignal]) -> float:
        """Weight SMT signals for regime analysis"""
        if not smt_signals:
            return 0.0

        total_weight = 0.0
        for signal in smt_signals:
            # Weight by HPS score and confidence
            hps_weight = signal.hps_score.value / 2.0  # 0-1 scale
            total_weight += signal.confidence * signal.strength * hps_weight

        return min(1.0, total_weight / len(smt_signals))

    def _score_accumulation(self, structure: Dict, dominance: Dict, liquidity: Dict, signals: float) -> float:
        """Score accumulation regime likelihood"""
        score = 0.0

        # Sideways structure favors accumulation
        if structure['trend'] == 'sideways':
            score += 0.3 * structure['strength']

        # BTC gaining dominance during accumulation
        if dominance['trend'] == 'btc_gaining':
            score += 0.2 * dominance['strength']

        # Liquidity tightening during accumulation
        if liquidity['condition'] == 'tightening':
            score += 0.3 * liquidity['strength']

        # Strong SMT signals indicate accumulation setup
        score += 0.2 * signals

        return min(1.0, score)

    def _score_markup(self, structure: Dict, dominance: Dict, liquidity: Dict, signals: float) -> float:
        """Score markup regime likelihood"""
        score = 0.0

        # Bullish structure favors markup
        if structure['trend'] == 'bullish':
            score += 0.4 * structure['strength']

        # Either dominance trend can work in markup
        score += 0.1 * dominance['strength']

        # Liquidity loosening supports markup
        if liquidity['condition'] == 'loosening':
            score += 0.3 * liquidity['strength']

        # Moderate SMT signals
        score += 0.2 * signals

        return min(1.0, score)

    def _score_distribution(self, structure: Dict, dominance: Dict, liquidity: Dict, signals: float) -> float:
        """Score distribution regime likelihood"""
        score = 0.0

        # Sideways at highs or weakening bullish
        if structure['trend'] in ['sideways', 'bearish']:
            score += 0.3 * structure['strength']

        # Alts gaining during distribution
        if dominance['trend'] == 'alts_gaining':
            score += 0.3 * dominance['strength']

        # Liquidity conditions less relevant
        score += 0.1 * liquidity['strength']

        # Strong SMT signals may indicate distribution
        score += 0.3 * signals

        return min(1.0, score)

    def _score_markdown(self, structure: Dict, dominance: Dict, liquidity: Dict, signals: float) -> float:
        """Score markdown regime likelihood"""
        score = 0.0

        # Bearish structure favors markdown
        if structure['trend'] == 'bearish':
            score += 0.4 * structure['strength']

        # Either dominance trend
        score += 0.1 * dominance['strength']

        # Liquidity tightening during markdown
        if liquidity['condition'] == 'tightening':
            score += 0.3 * liquidity['strength']

        # Strong SMT signals
        score += 0.2 * signals

        return min(1.0, score)

    def _calculate_regime_duration(self, regime: MacroRegime, current_time: pd.Timestamp) -> int:
        """Calculate how long current regime has been active"""
        if not self.regime_history:
            return 0

        # Find when this regime started
        for i in range(len(self.regime_history) - 1, -1, -1):
            if self.regime_history[i].regime != regime:
                if i < len(self.regime_history) - 1:
                    start_time = self.regime_history[i + 1].timestamp
                    return int((current_time - start_time).total_seconds() / 3600)
                break

        return 0

    def _default_regime_signal(self) -> RegimeSignal:
        """Return default neutral regime signal"""
        return RegimeSignal(
            regime=MacroRegime.NEUTRAL,
            confidence=0.5,
            timestamp=pd.Timestamp.now(),
            duration_hours=0,
            strength_indicators={}
        )

    def _analyze_range_contraction(self, data: pd.DataFrame) -> float:
        """Analyze price range contraction for CRT detection"""
        try:
            # Calculate ATR contraction
            atr_20 = self._calculate_atr(data, 20)
            atr_5 = self._calculate_atr(data, 5)

            if len(atr_20) < 2 or len(atr_5) < 2:
                return 0.0

            contraction = 1 - (atr_5.iloc[-1] / atr_20.iloc[-1])
            return max(0.0, min(1.0, contraction))

        except Exception:
            return 0.0

    def _analyze_volume_profile(self, data: pd.DataFrame) -> float:
        """Analyze volume profile for CRT detection"""
        try:
            volume_ma_20 = data['volume'].rolling(20).mean()
            volume_ma_5 = data['volume'].rolling(5).mean()

            if len(volume_ma_20) < 1 or len(volume_ma_5) < 1:
                return 0.0

            # Lower volume indicates accumulation
            volume_ratio = volume_ma_5.iloc[-1] / volume_ma_20.iloc[-1]
            return max(0.0, min(1.0, 1 - volume_ratio))

        except Exception:
            return 0.0

    def _analyze_order_flow_balance(self, data: pd.DataFrame) -> float:
        """Analyze order flow balance (simplified)"""
        try:
            # Simplified: use close vs OHLC midpoint
            midpoint = (data['high'] + data['low']) / 2
            close_bias = (data['close'] / midpoint - 1).abs().mean()

            # Lower bias indicates balance
            return max(0.0, min(1.0, 1 - close_bias * 10))

        except Exception:
            return 0.5

    def _calculate_atr(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range"""
        try:
            high_low = data['high'] - data['low']
            high_close = (data['high'] - data['close'].shift(1)).abs()
            low_close = (data['low'] - data['close'].shift(1)).abs()

            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            return true_range.rolling(period).mean()

        except Exception:
            return pd.Series()


class ContextFilter:
    """
    Context-based filtering for trade signals using macro regime analysis.

    Provides filtering logic to suppress or enhance signals based on
    macro context and regime classification.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.filter_config = config.get('context_filter', {})

        # Filter thresholds
        self.regime_confidence_threshold = self.filter_config.get('regime_confidence', 0.7)
        self.crt_enhancement_factor = self.filter_config.get('crt_enhancement', 1.5)
        self.premium_discount_factor = self.filter_config.get('premium_discount_factor', 0.8)

    def filter_signal(self, signal_strength: float, regime: RegimeSignal,
                     crt_data: Optional[Dict], premium_discount: Dict) -> Tuple[float, Dict[str, Any]]:
        """
        Filter and adjust signal strength based on macro context.

        Args:
            signal_strength: Original signal strength (0-1)
            regime: Current regime classification
            crt_data: CRT detection results
            premium_discount: Premium/discount environment

        Returns:
            Tuple of (adjusted_strength, filter_metadata)
        """
        try:
            adjusted_strength = signal_strength
            metadata = {
                'original_strength': signal_strength,
                'regime_effect': 0.0,
                'crt_effect': 0.0,
                'premium_discount_effect': 0.0
            }

            # Regime-based adjustments
            regime_multiplier = self._get_regime_multiplier(regime)
            adjusted_strength *= regime_multiplier
            metadata['regime_effect'] = regime_multiplier - 1.0

            # CRT enhancement
            if crt_data and crt_data.get('detected'):
                crt_multiplier = 1.0 + (crt_data['strength'] * (self.crt_enhancement_factor - 1.0))
                adjusted_strength *= crt_multiplier
                metadata['crt_effect'] = crt_multiplier - 1.0

            # Premium/discount adjustment
            pd_multiplier = self._get_premium_discount_multiplier(premium_discount)
            adjusted_strength *= pd_multiplier
            metadata['premium_discount_effect'] = pd_multiplier - 1.0

            # Ensure bounds
            adjusted_strength = max(0.0, min(1.0, adjusted_strength))
            metadata['final_strength'] = adjusted_strength

            return adjusted_strength, metadata

        except Exception as e:
            logger.error(f"Error in context filtering: {e}")
            return signal_strength, {}

    def _get_regime_multiplier(self, regime: RegimeSignal) -> float:
        """Get regime-based signal multiplier"""
        if regime.confidence < self.regime_confidence_threshold:
            return 1.0  # Neutral for low confidence

        # Adjust based on regime type
        regime_multipliers = {
            MacroRegime.ACCUMULATION: 1.2,  # Favor signals in accumulation
            MacroRegime.MARKUP: 1.1,       # Slightly favor in markup
            MacroRegime.DISTRIBUTION: 0.8,  # Reduce in distribution
            MacroRegime.MARKDOWN: 0.9,     # Reduce in markdown
            MacroRegime.TRANSITION: 0.95,  # Slight reduction in transition
            MacroRegime.NEUTRAL: 1.0       # No adjustment
        }

        base_multiplier = regime_multipliers.get(regime.regime, 1.0)
        confidence_factor = (regime.confidence - self.regime_confidence_threshold) / (1.0 - self.regime_confidence_threshold)

        return 1.0 + (base_multiplier - 1.0) * confidence_factor

    def _get_premium_discount_multiplier(self, premium_discount: Dict) -> float:
        """Get premium/discount environment multiplier"""
        environment = premium_discount.get('environment', 'neutral')
        strength = premium_discount.get('strength', 0.0)

        if environment == 'discount':
            # Favor long signals in discount environment
            return 1.0 + (strength * (1.0 / self.premium_discount_factor - 1.0))
        elif environment == 'premium':
            # Reduce long signals in premium environment
            return 1.0 - (strength * (1.0 - self.premium_discount_factor))
        else:
            return 1.0