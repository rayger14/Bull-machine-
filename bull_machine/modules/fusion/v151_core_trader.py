"""
Bull Machine v1.5.1 - Core Trader Enhanced Fusion Engine
Integrates ATR-based position sizing, exits, regime filtering, and ensemble logic
"""

import pandas as pd
from typing import Dict, Any, Optional
from bull_machine.core.telemetry import log_telemetry
from bull_machine.modules.fusion.v150_enhanced import FusionEngineV150
from bull_machine.strategy.position_sizing import atr_risk_size
from bull_machine.strategy.atr_exits import compute_exit_levels, maybe_trail_sl, enhanced_exit_check
from bull_machine.modules.regime_filter import regime_ok
from bull_machine.modules.wyckoff.wyckoff_phase import wyckoff_phase_scores
from bull_machine.modules.liquidity.liquidity_sweep import liquidity_sweep, rank_orderblocks, fvg_quality
from bull_machine.modules.exits.wick_magnet import wick_magnet_distance

class CoreTraderV151(FusionEngineV150):
    """
    v1.5.1 Core Trader with ATR-based risk management and enhanced filtering.
    """

    def check_entry(self, df: pd.DataFrame, last_trade_bar: int, config: dict, equity: float = 10000) -> Optional[Dict]:
        """
        Enhanced entry check with ATR sizing, regime filtering, and ensemble logic.

        Args:
            df: Price data DataFrame
            last_trade_bar: Index of last trade entry
            config: Configuration dict
            equity: Current portfolio equity

        Returns:
            Dict with trade plan or None if vetoed
        """
        current_bar = len(df) - 1

        # Cooldown enforcement
        cooldown_bars = int(config.get("cooldown_bars", 0) or 0)
        if cooldown_bars > 0:
            bars_since_trade = current_bar - last_trade_bar
            if bars_since_trade < cooldown_bars:
                log_telemetry("layer_masks.json", {
                    "cooldown_veto": True,
                    "cooldown_bars": cooldown_bars,
                    "bars_since_trade": bars_since_trade
                })
                return None

        # Regime veto (enhanced volume/volatility filter)
        if config.get("features", {}).get("regime_filter"):
            regime_cfg = config.get("regime", {})
            tf = config.get("timeframe", "")
            if not regime_ok(df, tf, regime_cfg):
                log_telemetry("layer_masks.json", {
                    "regime_veto": True,
                    "timeframe": tf,
                    "regime_config": regime_cfg
                })
                return None

        # Compute base layer scores
        if hasattr(self, '_layer_scores'):
            layer_scores = self._layer_scores.copy()
        else:
            layer_scores = self.compute_base_scores(df)

        # Apply knowledge adapters to layer scores
        self._apply_knowledge_adapters(df, layer_scores, config)

        # Check confluence vetoes with v1.5.0 alphas
        veto = self.check_confluence_vetoes(df, layer_scores, config)
        if veto:
            return None

        # Enhanced MTF ensemble confluence requirement
        if config.get("features", {}).get("ensemble_htf_bias"):
            ctx_floor = config.get("quality_floors", {}).get("context", 0.3)
            mtf_floor = config.get("quality_floors", {}).get("mtf", 0.3)
            ctx_score = layer_scores.get("context", 0.0)
            mtf_score = layer_scores.get("mtf", 0.0)

            # Stricter ensemble requirements for better confluence
            ensemble_threshold = 0.55  # Both context + MTF must average above this
            combined_htf_score = (ctx_score + mtf_score) / 2.0

            # Individual floor checks
            ctx_passed = ctx_score >= ctx_floor
            mtf_passed = mtf_score >= mtf_floor

            # Ensemble confluence check
            ensemble_passed = combined_htf_score >= ensemble_threshold

            # All three must pass for strong HTF bias confirmation
            if not (ctx_passed and mtf_passed and ensemble_passed):
                log_telemetry("layer_masks.json", {
                    "ensemble_veto": True,
                    "context_score": ctx_score,
                    "context_floor": ctx_floor,
                    "context_passed": ctx_passed,
                    "mtf_score": mtf_score,
                    "mtf_floor": mtf_floor,
                    "mtf_passed": mtf_passed,
                    "combined_htf_score": combined_htf_score,
                    "ensemble_threshold": ensemble_threshold,
                    "ensemble_passed": ensemble_passed,
                    "veto_reason": "htf_ensemble_confluence_failed"
                })
                return None

        # Calculate fusion score and check entry threshold
        weighted_score = self._calculate_fusion_score(layer_scores, config)
        entry_threshold = config.get("entry_threshold", 0.45)

        if weighted_score < entry_threshold:
            log_telemetry("layer_masks.json", {
                "threshold_veto": True,
                "weighted_score": weighted_score,
                "entry_threshold": entry_threshold,
                "layer_scores": layer_scores
            })
            return None

        # Determine trade side based on momentum
        momentum_score = layer_scores.get("momentum", 0.5)
        momentum_floor = config.get("quality_floors", {}).get("momentum", 0.3)
        side = "long" if momentum_score >= momentum_floor else "short"

        # ATR-based position sizing
        if config.get("features", {}).get("atr_sizing"):
            risk_cfg = config.get("risk", {})
            quantity = atr_risk_size(df, equity, risk_cfg)
        else:
            # Fallback: fixed percentage sizing
            quantity = equity * 0.03

        # ATR-based exit levels
        stop_loss = None
        take_profit = None
        if config.get("features", {}).get("atr_exits"):
            risk_cfg = config.get("risk", {})
            stop_loss, take_profit = compute_exit_levels(df, side, risk_cfg)

        # Create trade plan
        trade_plan = {
            "side": side,
            "quantity": quantity,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "timeframe": config.get("timeframe", ""),
            "layer_scores": layer_scores,
            "weighted_score": weighted_score,
            "entry_bar": current_bar,
            "entry_price": df['close'].iloc[-1]
        }

        # Log successful entry
        log_telemetry("layer_masks.json", {
            "entry_signal": True,
            "trade_plan": trade_plan,
            "regime_passed": True
        })

        return trade_plan

    def update_stop(self, df: pd.DataFrame, trade_plan: Dict, config: dict) -> Dict:
        """
        Update trailing stop loss if enabled.

        Args:
            df: Price data DataFrame
            trade_plan: Current trade plan
            config: Configuration dict

        Returns:
            Updated trade plan
        """
        if config.get("features", {}).get("atr_exits") and trade_plan.get("stop_loss"):
            risk_cfg = config.get("risk", {})
            side = trade_plan["side"]
            current_sl = trade_plan["stop_loss"]

            new_sl = maybe_trail_sl(df, side, current_sl, risk_cfg)
            if new_sl != current_sl:
                trade_plan["stop_loss"] = new_sl
                log_telemetry("layer_masks.json", {
                    "trailing_stop_update": True,
                    "old_sl": current_sl,
                    "new_sl": new_sl,
                    "side": side
                })

        return trade_plan

    def _calculate_fusion_score(self, layer_scores: Dict[str, float], config: dict) -> float:
        """
        Calculate weighted fusion score.

        Args:
            layer_scores: Dictionary of layer scores
            config: Configuration dict

        Returns:
            Weighted fusion score
        """
        # Default layer weights optimized for real market performance
        default_weights = {
            'wyckoff': 0.20,      # Primary trend analysis
            'liquidity': 0.15,    # Volume confirmation
            'structure': 0.18,    # Support/resistance
            'momentum': 0.15,     # RSI/MACD signals
            'volume': 0.12,       # Volume patterns
            'context': 0.10,      # Market environment
            'mtf': 0.10           # Multi-timeframe
        }

        layer_weights = config.get('layer_weights', default_weights)

        weighted_score = sum(
            layer_scores.get(layer, 0) * layer_weights.get(layer, 0.1)
            for layer in ['wyckoff', 'liquidity', 'structure', 'momentum', 'volume', 'context', 'mtf']
        )

        return weighted_score

    def check_exit(self, df: pd.DataFrame, trade_plan: Dict, config: dict) -> Dict:
        """
        Check if position should be closed using enhanced exit logic.

        Args:
            df: Price data DataFrame
            trade_plan: Current trade plan (treated as position)
            config: Configuration dict

        Returns:
            Dict with exit signal details or simple boolean for compatibility
        """
        # Convert trade_plan to position format for enhanced_exit_check
        position = {
            'side': trade_plan['side'],
            'entry_price': trade_plan['entry_price'],
            'stop_loss': trade_plan.get('stop_loss'),
            'take_profit': trade_plan.get('take_profit'),
            'entry_bar': trade_plan.get('entry_bar', len(df) - 1),
            'closed_percent': trade_plan.get('closed_percent', 0.0),
            'high_price': trade_plan.get('high_price', trade_plan['entry_price']),
            'low_price': trade_plan.get('low_price', trade_plan['entry_price'])
        }

        # Copy any ladder flags that might exist
        for key in trade_plan:
            if key.startswith('ladder_'):
                position[key] = trade_plan[key]

        # Use enhanced exit check with profit ladders and dynamic trailing
        exit_signal = enhanced_exit_check(df, position, config)

        # Update trade_plan with any position changes
        for key in ['closed_percent', 'high_price', 'low_price']:
            if key in position:
                trade_plan[key] = position[key]

        # Copy ladder flags back
        for key in position:
            if key.startswith('ladder_'):
                trade_plan[key] = position[key]

        # Legacy wick magnet check (only if enhanced exit didn't trigger)
        if not exit_signal['close_position'] and config.get("features", {}).get("wick_magnet"):
            if wick_magnet_distance(df):
                current_price = df['close'].iloc[-1]
                log_telemetry("layer_masks.json", {"exit_reason": "wick_magnet", "price": current_price})
                return {
                    'close_position': True,
                    'closed_percent': 1.0,
                    'exit_price': current_price,
                    'exit_reason': 'wick_magnet'
                }

        # Legacy bars held limit (only if enhanced exit didn't trigger)
        if not exit_signal['close_position']:
            bars_held = len(df) - 1 - trade_plan.get("entry_bar", len(df) - 1)
            max_bars = config.get("max_bars_held", 10)
            if bars_held >= max_bars:
                current_price = df['close'].iloc[-1]
                log_telemetry("layer_masks.json", {"exit_reason": "max_bars", "bars_held": bars_held})
                return {
                    'close_position': True,
                    'closed_percent': 1.0,
                    'exit_price': current_price,
                    'exit_reason': 'max_bars'
                }

        return exit_signal

    def _apply_knowledge_adapters(self, df: pd.DataFrame, layer_scores: Dict[str, float], config: dict):
        """
        Apply knowledge adapters to enhance layer scores.

        Args:
            df: Price data DataFrame
            layer_scores: Layer scores dict (modified in-place)
            config: Configuration dict
        """
        timeframe = config.get("timeframe", "")
        features = config.get("features", {})

        # Wyckoff Phase Analysis
        if features.get("wyckoff_phase"):
            wyckoff_score = wyckoff_phase_scores(df, timeframe)
            layer_scores["wyckoff"] += wyckoff_score * 0.5
            log_telemetry("layer_masks.json", {
                "wyckoff_adapter_applied": True,
                "wyckoff_contribution": wyckoff_score * 0.5
            })

        # Liquidity Sweep Analysis
        if features.get("liquidity_sweep"):
            sweep_detected = liquidity_sweep(df)
            adjustment = 0.05 if sweep_detected else -0.05
            layer_scores["liquidity"] += adjustment
            log_telemetry("layer_masks.json", {
                "liquidity_sweep_applied": True,
                "sweep_detected": sweep_detected,
                "liquidity_adjustment": adjustment
            })

        # Order Block Analysis
        if features.get("order_blocks"):
            ob_score = rank_orderblocks(df)
            fvg_score = fvg_quality(df)

            # Combine order block and FVG scores
            liquidity_enhancement = (ob_score * 0.3) + (fvg_score * 0.2)
            layer_scores["liquidity"] += liquidity_enhancement
            layer_scores["structure"] += liquidity_enhancement * 0.5  # Also boost structure

            log_telemetry("layer_masks.json", {
                "order_blocks_applied": True,
                "orderblock_score": ob_score,
                "fvg_score": fvg_score,
                "liquidity_enhancement": liquidity_enhancement
            })

        # Cap all scores to reasonable ranges
        for layer in layer_scores:
            layer_scores[layer] = max(0.1, min(0.9, layer_scores[layer]))