"""
Bull Machine v1.6.0 - Enhanced Core Trader with M1/M2 and Hidden Fibs
Integrates advanced Wyckoff phase detection and Fibonacci zone analysis
"""

import pandas as pd
from typing import Dict, Any, Optional
from bull_machine.core.telemetry import log_telemetry
from bull_machine.modules.fusion.v151_core_trader import CoreTraderV151
from bull_machine.strategy.wyckoff_m1m2 import compute_m1m2_scores
from bull_machine.strategy.hidden_fibs import compute_hidden_fib_scores, detect_price_time_confluence
from bull_machine.oracle import trigger_whisper, format_whisper_for_log

class CoreTraderV160(CoreTraderV151):
    """
    v1.6.0 Enhanced Core Trader with M1/M2 Wyckoff and Hidden Fibonacci signals.

    New Features:
    - M1 (spring/shakeout) detection at range lows
    - M2 (markup/re-accumulation) detection at range highs
    - Hidden Fibonacci retracement/extension zones
    - Volatility-weighted scoring integration
    """

    def compute_base_scores(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Enhanced scoring with M1/M2 Wyckoff and hidden Fibonacci signals.

        Combines v1.5.1 base scores with new v1.6.0 advanced signal detection.
        """
        # Get base layer scores from v1.5.1
        layer_scores = super().compute_base_scores(df)

        # Determine timeframe for adaptive scoring
        tf = getattr(self, '_current_timeframe', '1H')

        try:
            # Add M1/M2 Wyckoff phase detection
            m1m2_scores = compute_m1m2_scores(df, tf)
            layer_scores.update(m1m2_scores)

            # Add hidden Fibonacci zone analysis
            fib_scores = compute_hidden_fib_scores(df, tf)
            layer_scores.update(fib_scores)

            # Log enhanced scoring
            log_telemetry('layer_masks.json', {
                'module': 'v160_enhanced',
                'timeframe': tf,
                'base_scores_count': len(layer_scores) - len(m1m2_scores) - len(fib_scores),
                'm1m2_scores': m1m2_scores,
                'fib_scores': fib_scores,
                'enhanced_score_total': len(layer_scores)
            })

        except Exception as e:
            log_telemetry('layer_masks.json', {
                'module': 'v160_enhanced',
                'scoring_error': str(e),
                'timeframe': tf,
                'fallback_to_base': True
            })

        return layer_scores

    def set_current_timeframe(self, tf: str):
        """Set current timeframe for adaptive scoring."""
        self._current_timeframe = tf

    def check_enhanced_confluence(self, layer_scores: Dict[str, float], config: Dict) -> bool:
        """
        Enhanced confluence check including M1/M2 and Fibonacci signals.

        Args:
            layer_scores: Complete layer scores including v1.6.0 enhancements
            config: Configuration with quality floors for new signals

        Returns:
            True if enhanced confluence requirements are met
        """
        quality_floors = config.get('quality_floors', {})

        # Check traditional layer floors (from v1.5.1)
        base_layers = ['wyckoff', 'liquidity', 'structure', 'momentum', 'volume', 'context', 'mtf']
        base_passed = all(
            layer_scores.get(layer, 0.0) >= quality_floors.get(layer, 0.0)
            for layer in base_layers
        )

        # Check M1/M2 Wyckoff floors (if enabled)
        m1_floor = quality_floors.get('m1', 0.0)
        m2_floor = quality_floors.get('m2', 0.0)
        m1_score = layer_scores.get('m1', 0.0)
        m2_score = layer_scores.get('m2', 0.0)

        # M1/M2 confluence: At least one must be active and pass floor
        m1_passed = m1_score >= m1_floor if m1_floor > 0 else True
        m2_passed = m2_score >= m2_floor if m2_floor > 0 else True
        wyckoff_enhanced = m1_passed or m2_passed

        # Check Fibonacci floors (if enabled)
        fib_ret_floor = quality_floors.get('fib_retracement', 0.0)
        fib_ext_floor = quality_floors.get('fib_extension', 0.0)
        fib_ret_score = layer_scores.get('fib_retracement', 0.0)
        fib_ext_score = layer_scores.get('fib_extension', 0.0)

        # Fibonacci confluence: At least one level must be active and pass floor
        fib_ret_passed = fib_ret_score >= fib_ret_floor if fib_ret_floor > 0 else True
        fib_ext_passed = fib_ext_score >= fib_ext_floor if fib_ext_floor > 0 else True
        fib_confluence = fib_ret_passed or fib_ext_passed

        # Overall confluence: base + enhanced signals
        confluence_met = base_passed and wyckoff_enhanced and fib_confluence

        log_telemetry('layer_masks.json', {
            'enhanced_confluence_check': {
                'base_layers_passed': base_passed,
                'wyckoff_enhanced_passed': wyckoff_enhanced,
                'fib_confluence_passed': fib_confluence,
                'overall_confluence_met': confluence_met,
                'm1_score': m1_score,
                'm2_score': m2_score,
                'fib_retracement': fib_ret_score,
                'fib_extension': fib_ext_score
            }
        })

        return confluence_met

    def check_entry(self, df: pd.DataFrame, last_trade_bar: int, config: dict, equity: float = 10000) -> Optional[Dict]:
        """
        Enhanced entry check with v1.6.0 M1/M2 and Fibonacci integration.

        Overrides v1.5.1 entry logic to include new signal confluence.
        """
        current_bar = len(df) - 1

        # Set timeframe context for scoring
        tf = config.get("timeframe", "1H")
        self.set_current_timeframe(tf)

        # Cooldown enforcement (from v1.5.1)
        cooldown_bars = int(config.get("cooldown_bars", 0) or 0)
        if cooldown_bars > 0:
            bars_since_trade = current_bar - last_trade_bar
            if bars_since_trade < cooldown_bars:
                log_telemetry("layer_masks.json", {
                    "cooldown_veto": True,
                    "cooldown_bars": cooldown_bars,
                    "bars_since_trade": bars_since_trade,
                    "version": "v1.6.0"
                })
                return None

        # Regime veto (from v1.5.1)
        if config.get("features", {}).get("regime_filter"):
            from bull_machine.modules.regime_filter import regime_ok
            regime_cfg = config.get("regime", {})
            if not regime_ok(df, tf, regime_cfg):
                log_telemetry("layer_masks.json", {
                    "regime_veto": True,
                    "timeframe": tf,
                    "regime_config": regime_cfg,
                    "version": "v1.6.0"
                })
                return None

        # Compute enhanced layer scores (includes M1/M2 and fibs)
        layer_scores = self.compute_base_scores(df)

        # v1.6.1: Price and time symmetry = where structure and vibration align.
        # Check for Fibonacci price-time confluence
        if config.get('features', {}).get('temporal_fib', False):
            confluence_data = detect_price_time_confluence(df, config, current_bar)

            # Enhance scores with cluster confluence
            if confluence_data['confluence_detected']:
                cluster_strength = confluence_data['confluence_strength']

                # Boost Fibonacci scores for price clusters
                if confluence_data['price_cluster']:
                    layer_scores['fib_retracement'] = min(1.0,
                        layer_scores.get('fib_retracement', 0.0) + cluster_strength * 0.5)
                    layer_scores['fib_extension'] = min(1.0,
                        layer_scores.get('fib_extension', 0.0) + cluster_strength * 0.5)

                # Boost ensemble score for time clusters
                if confluence_data['time_cluster']:
                    temporal_weight = config.get('weights', {}).get('temporal', 0.10)
                    layer_scores['ensemble_entry'] = layer_scores.get('ensemble_entry', 0.0) + \
                        confluence_data['time_cluster']['strength'] * temporal_weight

                # Add cluster tags for Oracle whispers
                layer_scores['cluster_tags'] = confluence_data['tags']
                layer_scores['confluence_strength'] = cluster_strength

                # Trigger Oracle whispers for high confluence events
                wyckoff_phase = layer_scores.get('wyckoff_phase', '')
                whispers = trigger_whisper(layer_scores, phase=wyckoff_phase)
                if whispers:
                    whisper_log = format_whisper_for_log(whispers, layer_scores)
                    log_telemetry('oracle_whispers.json', whisper_log)

        # Apply knowledge adapters (from v1.5.1)
        self._apply_knowledge_adapters(df, layer_scores, config)

        # Enhanced confluence check with v1.6.0 signals
        if not self.check_enhanced_confluence(layer_scores, config):
            log_telemetry("layer_masks.json", {
                "enhanced_confluence_veto": True,
                "layer_scores": layer_scores,
                "version": "v1.6.0"
            })
            return None

        # Check traditional vetoes (from v1.5.1)
        veto = self.check_confluence_vetoes(df, layer_scores, config)
        if veto:
            return None

        # Calculate fusion score (will be enhanced by ensemble_mode.py weighting)
        weighted_score = self._calculate_fusion_score(layer_scores, config)
        entry_threshold = config.get("entry_threshold", 0.44)

        if weighted_score < entry_threshold:
            log_telemetry("layer_masks.json", {
                "threshold_veto": True,
                "weighted_score": weighted_score,
                "entry_threshold": entry_threshold,
                "layer_scores": layer_scores,
                "version": "v1.6.0"
            })
            return None

        # Enhanced side determination using M1/M2 context
        side = self._determine_trade_side_enhanced(layer_scores, config)

        # ATR-based position sizing (from v1.5.1)
        if config.get("features", {}).get("atr_sizing"):
            from bull_machine.strategy.position_sizing import atr_risk_size
            risk_cfg = config.get("risk", {})
            quantity = atr_risk_size(df, equity, risk_cfg)
        else:
            quantity = equity * 0.03

        # ATR-based exit levels (from v1.5.1)
        stop_loss = None
        take_profit = None
        if config.get("features", {}).get("atr_exits"):
            from bull_machine.strategy.atr_exits import compute_exit_levels
            risk_cfg = config.get("risk", {})
            stop_loss, take_profit = compute_exit_levels(df, side, risk_cfg)

        # Create enhanced trade plan
        trade_plan = {
            "side": side,
            "quantity": quantity,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "timeframe": tf,
            "layer_scores": layer_scores,
            "ensemble_score": weighted_score,
            "version": "v1.6.0",
            # v1.6.0 specific fields
            "m1_score": layer_scores.get('m1', 0.0),
            "m2_score": layer_scores.get('m2', 0.0),
            "fib_retracement": layer_scores.get('fib_retracement', 0.0),
            "fib_extension": layer_scores.get('fib_extension', 0.0),
            "enhanced_confluence": True
        }

        log_telemetry("layer_masks.json", {
            "trade_entry_success": True,
            "trade_plan": trade_plan,
            "version": "v1.6.0"
        })

        return trade_plan

    def _determine_trade_side_enhanced(self, layer_scores: Dict[str, float], config: Dict) -> str:
        """
        Enhanced trade side determination using M1/M2 Wyckoff context.

        M1 signals (springs) suggest long bias after shakeout.
        M2 signals (markup) suggest continuation of established trend.
        """
        # Traditional momentum-based side determination
        momentum_score = layer_scores.get("momentum", 0.5)
        momentum_floor = config.get("quality_floors", {}).get("momentum", 0.3)
        base_side = "long" if momentum_score >= momentum_floor else "short"

        # M1/M2 bias adjustment
        m1_score = layer_scores.get('m1', 0.0)
        m2_score = layer_scores.get('m2', 0.0)

        # Strong M1 signal (spring) typically suggests long bias after shakeout
        if m1_score > 0.4:
            enhanced_side = "long"
            bias_reason = "M1_spring_detected"

        # Strong M2 signal (markup) suggests trend continuation
        elif m2_score > 0.4:
            # M2 in uptrend = continue long, M2 in downtrend = continue short
            enhanced_side = base_side  # Follow momentum direction
            bias_reason = "M2_markup_continuation"

        else:
            # No strong Wyckoff bias, use traditional momentum
            enhanced_side = base_side
            bias_reason = "traditional_momentum"

        log_telemetry('layer_masks.json', {
            'side_determination': {
                'base_side': base_side,
                'enhanced_side': enhanced_side,
                'bias_reason': bias_reason,
                'momentum_score': momentum_score,
                'm1_score': m1_score,
                'm2_score': m2_score
            }
        })

        return enhanced_side