"""
Domain Fusion Engine - v1.8.6 "Temporal Intelligence"

Integrates all Bull Machine domain engines:
- Wyckoff (accumulation/distribution phases)
- SMC (BOS/CHOCH/FVG/OB)
- HOB/Liquidity (order blocks, liquidity levels)
- Momentum (RSI/MACD divergence)

v1.8.5 Enhancements:
- Negative Fibonacci levels for trend confluence
- Fourier noise filter for signal/noise separation
- Event-driven analysis (conference tagging, funding/OI)
- Narrative trap detection (HODL traps, distribution)

v1.8.6 Temporal Analysis:
- Gann cycles (30/60/90 day ACF vibrations)
- Square of 9 proximity scoring
- Thermo-floor (mining cost floor)
- Log premium (difficulty-based time multiplier)
- Logistic bid (institutional re-accumulation)
- LPPLS blowoff detection

Returns unified 0-1 score with MTF confluence validation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass
import logging

# Import all domain engines
from engine.wyckoff.wyckoff_engine import detect_wyckoff_phase
from engine.smc.smc_engine import SMCEngine
from engine.liquidity.hob import HOBDetector
from engine.momentum.momentum_engine import calculate_rsi, calculate_macd_norm, momentum_delta

# v1.8.5 imports
from engine.liquidity.fib_levels import calculate_fib_bonus
from engine.noise.fourier_filter import apply_fourier_multiplier
from engine.events.event_engine import tag_events, should_veto_event
from engine.narrative.trap_detector import should_veto_narrative

# v1.8.6 imports
from engine.temporal import temporal_signal

logger = logging.getLogger(__name__)

@dataclass
class FusionSignal:
    """Unified fusion signal output"""
    score: float  # 0-1 overall confidence
    direction: str  # 'long', 'short', 'neutral'
    confidence: float  # 0-1
    
    # Domain scores
    wyckoff_score: float
    smc_score: float
    hob_score: float
    momentum_score: float
    
    # MTF validation
    mtf_aligned: bool
    mtf_confidence: float
    
    # Component signals
    wyckoff_phase: Optional[str]
    smc_bias: Optional[str]
    hob_quality: Optional[str]
    momentum_bias: Optional[str]
    
    # Metadata
    features: Dict[str, Any]
    reasons: list


def _standardize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names to lowercase"""
    cols_map = {}
    for col in df.columns:
        lower = col.lower()
        if lower in ['open', 'high', 'low', 'close', 'volume']:
            cols_map[col] = lower
    
    if cols_map:
        df = df.rename(columns=cols_map)
    
    # Ensure required columns exist
    required = ['open', 'high', 'low', 'close']
    for req in required:
        if req not in df.columns:
            raise ValueError(f"Missing required column: {req}")
    
    # Add volume if missing
    if 'volume' not in df.columns:
        df['volume'] = 1000000.0
    
    return df[['open', 'high', 'low', 'close', 'volume']].copy()


def _wyckoff_to_score(wyck_result: Dict, config: Dict) -> tuple:
    """Convert Wyckoff result to normalized 0-1 score"""
    try:
        phase = wyck_result.get('phase')
        conf = wyck_result.get('confidence', 0.0)

        # TUNED: Lower confidence threshold from 0.3 → 0.2
        if not phase or conf < 0.2:
            return 0.5, 'neutral', []

        # Map phases to directional bias
        bullish_phases = ['accumulation', 'spring', 'markup', 'reaccumulation']
        bearish_phases = ['distribution', 'upthrust', 'markdown', 'redistribution']

        reasons = []
        if phase in bullish_phases:
            # TUNED: Amplify score from 0.5x → 0.7x boost
            score = 0.3 + (conf * 0.7)  # 0.3-1.0 range (was 0.5-1.0)
            direction = 'long'
            reasons.append(f'Wyckoff {phase} (conf={conf:.2f})')
        elif phase in bearish_phases:
            # TUNED: Amplify bearish score symmetrically
            score = 0.7 - (conf * 0.7)  # 0.0-0.7 range (was 0.0-0.5)
            direction = 'short'
            reasons.append(f'Wyckoff {phase} (conf={conf:.2f})')
        else:
            score = 0.5
            direction = 'neutral'

        # TUNED: Increase CRT bonus from 0.05 → 0.10
        if wyck_result.get('crt_active'):
            score += 0.10 if direction == 'long' else -0.10
            reasons.append('CRT active')

        return np.clip(score, 0, 1), direction, reasons

    except Exception as e:
        logger.error(f"Error converting Wyckoff to score: {e}")
        return 0.5, 'neutral', []


def _smc_to_score(smc_engine: SMCEngine, df: pd.DataFrame, config: Dict) -> tuple:
    """Convert SMC signal to normalized 0-1 score"""
    try:
        signal = smc_engine.analyze(df)
        
        # Use confluence_score as base (already 0-1)
        score = signal.confluence_score
        direction = signal.direction
        
        reasons = []
        if signal.hit_counters['ob_hits'] > 0:
            reasons.append(f"OB hits: {signal.hit_counters['ob_hits']}")
        if signal.hit_counters['fvg_hits'] > 0:
            reasons.append(f"FVG hits: {signal.hit_counters['fvg_hits']}")
        if signal.hit_counters['bos_hits'] > 0:
            reasons.append(f"BOS hits: {signal.hit_counters['bos_hits']}")
        if signal.hit_counters['sweep_hits'] > 0:
            reasons.append(f"Sweep hits: {signal.hit_counters['sweep_hits']}")
        
        # Map direction to score range
        if direction == 'long':
            score = 0.5 + (score * 0.5)
        elif direction == 'short':
            score = 0.5 - (score * 0.5)
        else:
            score = 0.5
        
        return np.clip(score, 0, 1), direction, reasons, signal.institutional_bias
        
    except Exception as e:
        logger.error(f"Error converting SMC to score: {e}")
        return 0.5, 'neutral', [], 'neutral'


def _hob_to_score(hob_detector: HOBDetector, df: pd.DataFrame, config: Dict) -> tuple:
    """Convert HOB analysis to normalized 0-1 score"""
    try:
        # Simple HOB scoring based on volume profile and liquidity levels
        v = df['volume'].values
        c = df['close'].values
        h = df['high'].values
        l = df['low'].values

        # Calculate volume surge in last 10 bars
        vol_mean = np.mean(v[-50:]) if len(v) >= 50 else np.mean(v)
        recent_vol = np.mean(v[-10:])
        vol_surge = recent_vol / max(vol_mean, 1e-9)

        # Check for wick presence (institutional absorption)
        last_bar_wick_ratio = (h[-1] - max(c[-1], df['open'].iloc[-1])) / max(h[-1] - l[-1], 1e-9)

        reasons = []
        score = 0.5
        direction = 'neutral'
        quality = 'retail'

        # TUNED: Lower volume surge threshold from 1.5 → 1.2 and amplify boost
        if vol_surge > 1.2:
            score += 0.20  # was 0.15
            reasons.append(f'Volume surge: {vol_surge:.2f}x')
            if vol_surge > 1.8:  # was 2.0
                quality = 'institutional'
                score += 0.15  # was 0.1

        # TUNED: Lower wick threshold from 0.4 → 0.3 and amplify boost
        if last_bar_wick_ratio > 0.3:
            score += 0.15  # was 0.1
            reasons.append(f'Wick absorption: {last_bar_wick_ratio:.2f}')

        # Price action bias
        if c[-1] > c[-5]:
            direction = 'long'
        elif c[-1] < c[-5]:
            direction = 'short'
            score = 1.0 - score

        return np.clip(score, 0, 1), direction, reasons, quality

    except Exception as e:
        logger.error(f"Error converting HOB to score: {e}")
        return 0.5, 'neutral', [], 'invalid'


def _momentum_to_score(df: pd.DataFrame, config: Dict) -> tuple:
    """Convert momentum indicators to normalized 0-1 score"""
    try:
        rsi = calculate_rsi(df, period=config.get('rsi_period', 14))
        macd_n = calculate_macd_norm(df)
        delta = momentum_delta(df, config)

        reasons = []
        score = 0.5
        direction = 'neutral'

        # TUNED: Amplify RSI contribution (was +/-0.15, now +/-0.25)
        if rsi > 70:
            score -= 0.25  # was 0.15
            direction = 'short'
            reasons.append(f'RSI overbought: {rsi:.1f}')
        elif rsi < 30:
            score += 0.25  # was 0.15
            direction = 'long'
            reasons.append(f'RSI oversold: {rsi:.1f}')
        elif rsi > 50:
            # TUNED: Amplify bullish RSI from /100 → /50
            score += (rsi - 50) / 50.0  # was /100
            direction = 'long' if rsi > 55 else 'neutral'
        else:
            # TUNED: Amplify bearish RSI from /100 → /50
            score -= (50 - rsi) / 50.0  # was /100
            direction = 'short' if rsi < 45 else 'neutral'

        # TUNED: Amplify MACD contribution from 0.1 → 0.2
        if abs(macd_n) > 0.001:
            if macd_n > 0:
                score += 0.2  # was 0.1
                reasons.append(f'MACD positive: {macd_n:.4f}')
            else:
                score -= 0.1
                reasons.append(f'MACD negative: {macd_n:.4f}')
        
        return np.clip(score, 0, 1), direction, reasons, rsi
        
    except Exception as e:
        logger.error(f"Error converting momentum to score: {e}")
        return 0.5, 'neutral', [], 50.0


def _check_mtf_alignment(df_1h: pd.DataFrame, df_4h: pd.DataFrame, df_1d: pd.DataFrame, 
                        config: Dict) -> tuple:
    """
    Check multi-timeframe alignment.
    
    Returns:
        (aligned: bool, confidence: float, reasons: list)
    """
    try:
        reasons = []
        alignment_score = 0
        total_checks = 0
        
        # Trend alignment check
        def get_trend(df):
            if len(df) < 20:
                return 'neutral'
            sma20 = df['close'].rolling(20).mean()
            if df['close'].iloc[-1] > sma20.iloc[-1] * 1.01:
                return 'up'
            elif df['close'].iloc[-1] < sma20.iloc[-1] * 0.99:
                return 'down'
            return 'neutral'
        
        trend_1h = get_trend(df_1h)
        trend_4h = get_trend(df_4h)
        trend_1d = get_trend(df_1d)
        
        # Check 4H-1D alignment
        if trend_4h == trend_1d and trend_4h != 'neutral':
            alignment_score += 1
            reasons.append(f'4H-1D aligned: {trend_4h}')
        total_checks += 1
        
        # Check 1H-4H alignment
        if trend_1h == trend_4h and trend_1h != 'neutral':
            alignment_score += 1
            reasons.append(f'1H-4H aligned: {trend_1h}')
        total_checks += 1
        
        # Nested structure check (1H pullback in 4H trend)
        nested_threshold = config.get('mtf', {}).get('nested_threshold', 0.02)
        if trend_4h != 'neutral' and trend_1h != trend_4h:
            # This could be a healthy pullback
            price_1h = df_1h['close'].iloc[-1]
            sma_4h = df_4h['close'].rolling(20).mean().iloc[-1]
            distance_pct = abs(price_1h - sma_4h) / sma_4h
            
            if distance_pct < nested_threshold:
                alignment_score += 0.5
                reasons.append(f'Nested structure: 1H pullback in 4H {trend_4h}trend')
        
        # Calculate alignment confidence
        confidence = alignment_score / max(total_checks, 1)
        aligned = confidence >= 0.5  # At least 50% alignment
        
        return aligned, confidence, reasons
        
    except Exception as e:
        logger.error(f"Error checking MTF alignment: {e}")
        return False, 0.0, ['MTF alignment check failed']


def analyze_fusion(df_1h: pd.DataFrame, df_4h: pd.DataFrame, df_1d: pd.DataFrame, 
                   config: Dict) -> FusionSignal:
    """
    Main fusion analysis - combines all domain engines with MTF validation.
    
    Args:
        df_1h: 1H OHLCV data
        df_4h: 4H OHLCV data  
        df_1d: 1D OHLCV data
        config: Configuration dict
        
    Returns:
        FusionSignal with unified score and metadata
    """
    try:
        # Standardize dataframes
        df_1h = _standardize_df(df_1h)
        df_4h = _standardize_df(df_4h)
        df_1d = _standardize_df(df_1d)
        
        # Initialize engines
        smc_engine = SMCEngine(config)
        hob_detector = HOBDetector(config)
        
        # Run domain analyses
        wyck_result = detect_wyckoff_phase(df_1d, config, usdt_stag_strength=0.5)
        wyck_score, wyck_dir, wyck_reasons = _wyckoff_to_score(wyck_result, config)
        
        smc_score, smc_dir, smc_reasons, smc_bias = _smc_to_score(smc_engine, df_1h, config)
        hob_score, hob_dir, hob_reasons, hob_quality = _hob_to_score(hob_detector, df_1h, config)
        mom_score, mom_dir, mom_reasons, rsi = _momentum_to_score(df_1h, config)
        
        # Get weights from config
        weights = config.get('fusion', {}).get('weights', {
            'wyckoff': 0.30,
            'liquidity': 0.25,
            'momentum': 0.30,
            'smc': 0.15
        })
        
        # Calculate weighted fusion score
        fusion_score = (
            wyck_score * weights['wyckoff'] +
            smc_score * weights['smc'] +
            hob_score * weights['liquidity'] +
            mom_score * weights['momentum']
        )

        # Log domain scores at debug level
        logger.debug(
            f"Fusion domain scores: wyck={wyck_score:.3f}/{wyck_dir}, "
            f"smc={smc_score:.3f}/{smc_dir}, hob={hob_score:.3f}/{hob_dir}, "
            f"mom={mom_score:.3f}/{mom_dir}, fusion={fusion_score:.3f}"
        )

        # MTF alignment check
        mtf_aligned, mtf_conf, mtf_reasons = _check_mtf_alignment(df_1h, df_4h, df_1d, config)

        # Apply MTF penalty if not aligned
        if not mtf_aligned:
            fusion_score *= 0.8  # 20% penalty

        # ═══════════════════════════════════════════════════════════════
        # v1.8.5 ENHANCEMENTS
        # ═══════════════════════════════════════════════════════════════

        # 1. Fibonacci confluence bonus
        fib_bonus = 0.0
        if config.get('liquidity', {}).get('negative_fibs_enabled', False):
            fib_bonus = calculate_fib_bonus(df_1h, config)
            fusion_score = min(1.0, fusion_score + fib_bonus)

        # 2. Fourier noise filter (apply as multiplier)
        if config.get('fusion', {}).get('fourier_enabled', False):
            fusion_score = apply_fourier_multiplier(fusion_score, df_1h, config.get('fusion', {}))

        # 3. Event veto check (conferences, high leverage)
        event_veto_active = False
        event_veto_reason = ""
        if config.get('events', {}).get('calendar_enabled', False):
            # Need macro data for funding/OI checks (will be provided by caller)
            macro_data = config.get('_macro_data_cache', {})
            event_tags = tag_events(df_1h, macro_data, config.get('events', {}))
            event_veto_active, event_veto_reason = should_veto_event({}, event_tags, config.get('events', {}))

            if event_veto_active:
                fusion_score = 0.0  # Hard veto

        # 4. Narrative trap veto (HODL trap, distribution)
        narrative_veto_active = False
        narrative_veto_reason = ""
        if config.get('narrative', {}).get('narrative_enabled', False):
            macro_cache = config.get('_macro_data_cache', {})
            narrative_veto_active, narrative_veto_reason = should_veto_narrative(
                df_1h, macro_cache, config.get('narrative', {})
            )

            if narrative_veto_active:
                fusion_score *= 0.3  # Heavy penalty (not full veto)

        # ═══════════════════════════════════════════════════════════════
        # END v1.8.5 ENHANCEMENTS
        # ═══════════════════════════════════════════════════════════════

        # ═══════════════════════════════════════════════════════════════
        # v1.8.6 TEMPORAL/GANN ANALYSIS
        # ═══════════════════════════════════════════════════════════════

        temporal_bonus = 0.0
        temporal_veto_active = False
        temporal_veto_reason = ""

        if config.get('temporal', {}).get('enabled', False):
            # Get macro data cache for thermo-floor/difficulty calculations
            macro_cache = config.get('_macro_data_cache', {})

            # Calculate temporal signal
            temporal_result = temporal_signal(
                df_1h,
                df_4h,
                df_1d,
                config.get('temporal', {}),
                macro_cache
            )

            # Check for LPPLS veto first
            if temporal_result.get('veto', False):
                temporal_veto_active = True
                temporal_veto_reason = temporal_result.get('veto_reason', 'LPPLS blowoff detected')
                fusion_score = 0.0  # Hard veto on blowoff detection
            else:
                # Apply bounded bonus based on confluence score
                confluence = temporal_result.get('confluence_score', 0.0)
                bonus_cap = config.get('temporal', {}).get('bonus_cap', 0.15)

                # Scale bonus: 0.0-0.5 confluence → negative bonus (caution)
                #              0.5-1.0 confluence → positive bonus (confluence)
                if confluence < 0.5:
                    temporal_bonus = -(0.5 - confluence) * bonus_cap
                else:
                    temporal_bonus = (confluence - 0.5) * 2 * bonus_cap

                # Clip to ±bonus_cap
                temporal_bonus = np.clip(temporal_bonus, -bonus_cap, bonus_cap)

                # Apply bonus
                fusion_score = np.clip(fusion_score + temporal_bonus, 0.0, 1.0)

        # ═══════════════════════════════════════════════════════════════
        # END v1.8.6 TEMPORAL ANALYSIS
        # ═══════════════════════════════════════════════════════════════

        # ═══════════════════════════════════════════════════════════════
        # v1.8.6 MACRO FUSION COMPOSITE
        # ═══════════════════════════════════════════════════════════════

        macro_fusion_adjustment = 0.0
        macro_fusion_enabled = config.get('macro_fusion', {}).get('enabled', False)

        if macro_fusion_enabled:
            # Get macro analysis result from config cache (populated by runner)
            macro_analysis = config.get('_macro_analysis_cache', {})
            fusion_composite = macro_analysis.get('fusion_composite')

            if fusion_composite is not None:
                # Apply fusion composite adjustment (already capped at ±0.10)
                macro_fusion_adjustment = fusion_composite
                fusion_score = np.clip(fusion_score + macro_fusion_adjustment, 0.0, 1.0)
                logger.debug(f"Macro fusion composite: {fusion_composite:+.3f} → new score: {fusion_score:.3f}")

        # ═══════════════════════════════════════════════════════════════
        # END v1.8.6 MACRO FUSION COMPOSITE
        # ═══════════════════════════════════════════════════════════════

        # Determine overall direction
        directions = [wyck_dir, smc_dir, hob_dir, mom_dir]
        long_votes = sum(1 for d in directions if d == 'long')
        short_votes = sum(1 for d in directions if d == 'short')

        if long_votes > short_votes:
            final_direction = 'long'
        elif short_votes > long_votes:
            final_direction = 'short'
        else:
            # TIE or ALL NEUTRAL: Use fusion_score to break tie
            # If fusion_score > 0.5, bias long; if < 0.5, bias short
            if fusion_score > 0.52:  # Slight bullish bias
                final_direction = 'long'
            elif fusion_score < 0.48:  # Slight bearish bias
                final_direction = 'short'
            else:
                # Dead neutral - use highest-weighted domain's direction
                domain_weights_dirs = [
                    (wyck_score * weights['wyckoff'], wyck_dir),
                    (smc_score * weights['smc'], smc_dir),
                    (hob_score * weights['liquidity'], hob_dir),
                    (mom_score * weights['momentum'], mom_dir)
                ]
                # Sort by weighted score descending
                domain_weights_dirs.sort(reverse=True, key=lambda x: x[0])
                # Use direction of highest-weighted domain if not neutral
                for _, direction in domain_weights_dirs:
                    if direction != 'neutral':
                        final_direction = direction
                        break
                else:
                    final_direction = 'neutral'
        
        # Compile all reasons
        all_reasons = wyck_reasons + smc_reasons + hob_reasons + mom_reasons + mtf_reasons
        
        # Build features dict
        features = {
            'wyckoff': {'phase': wyck_result.get('phase'), 'confidence': wyck_result.get('confidence', 0)},
            'smc': {'bias': smc_bias, 'score': smc_score},
            'hob': {'quality': hob_quality, 'score': hob_score},
            'momentum': {'rsi': rsi, 'score': mom_score},
            'mtf': {'aligned': mtf_aligned, 'confidence': mtf_conf}
        }
        
        return FusionSignal(
            score=float(np.clip(fusion_score, 0, 1)),
            direction=final_direction,
            confidence=float(np.clip(fusion_score, 0, 1)),
            wyckoff_score=float(wyck_score),
            smc_score=float(smc_score),
            hob_score=float(hob_score),
            momentum_score=float(mom_score),
            mtf_aligned=mtf_aligned,
            mtf_confidence=float(mtf_conf),
            wyckoff_phase=wyck_result.get('phase'),
            smc_bias=smc_bias,
            hob_quality=hob_quality,
            momentum_bias=mom_dir,
            features=features,
            reasons=all_reasons
        )
        
    except Exception as e:
        # Log detailed error context for debugging
        error_context = {
            'error_type': type(e).__name__,
            'error_message': str(e),
            'df_1h_shape': df_1h.shape if df_1h is not None else None,
            'df_4h_shape': df_4h.shape if df_4h is not None else None,
            'df_1d_shape': df_1d.shape if df_1d is not None else None,
            'config_keys': list(config.keys()) if config else None
        }
        logger.error(
            f"CRITICAL: Fusion analysis failed - {type(e).__name__}: {str(e)}\n"
            f"Context: {error_context}",
            exc_info=True,
            extra={'error_context': error_context}
        )

        # Return neutral signal on error (defensive fallback)
        # NOTE: This prevents trading on corrupted data, but masks bugs
        # Consider fail-fast mode for development/testing
        return FusionSignal(
            score=0.5,
            direction='neutral',
            confidence=0.0,
            wyckoff_score=0.5,
            smc_score=0.5,
            hob_score=0.5,
            momentum_score=0.5,
            mtf_aligned=False,
            mtf_confidence=0.0,
            wyckoff_phase=None,
            smc_bias='neutral',
            hob_quality='invalid',
            momentum_bias='neutral',
            features={'error': str(e), 'error_type': type(e).__name__},
            reasons=[f'FUSION ERROR: {type(e).__name__} - {str(e)[:100]}']
        )
