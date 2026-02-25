#!/usr/bin/env python3
"""
Coinbase Paper Trading Runner for Bull Machine

Fetches real BTC-PERP-INTX market data from Coinbase, generates signals
using the full Bull Machine engine, and tracks virtual positions (paper trading).

Replaces the Freqtrade/Kraken spot setup with Coinbase perpetual futures.

Usage:
    # Paper trading (real Coinbase data, virtual fills):
    python3 bin/live/coinbase_runner.py --paper

    # Paper trading with custom config:
    python3 bin/live/coinbase_runner.py --paper --initial-cash 100000 \
        --config configs/bull_machine_isolated_v11_fixed.json

    # Paper trading with Binance data fallback (if Coinbase unavailable):
    python3 bin/live/coinbase_runner.py --paper --fallback-binance

    # Live trading (future - real orders on Coinbase):
    python3 bin/live/coinbase_runner.py --live  # NOT YET IMPLEMENTED
"""

import argparse
import json
import logging
import signal
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("coinbase_runner")

# ---------------------------------------------------------------------------
# Import CoinbaseAdapter (being built in parallel -- graceful fallback)
# ---------------------------------------------------------------------------
COINBASE_AVAILABLE = False
try:
    from bin.live.coinbase_client import CoinbaseAdapter
    COINBASE_AVAILABLE = True
except ImportError:
    logger.debug("CoinbaseAdapter not yet available (coinbase_client.py missing).")

# ---------------------------------------------------------------------------
# Import CoinbaseFundingClient (optional, for funding cost tracking)
# ---------------------------------------------------------------------------
COINBASE_FUNDING_AVAILABLE = False
try:
    from bin.live.coinbase_funding import CoinbaseFundingClient
    COINBASE_FUNDING_AVAILABLE = True
except ImportError:
    logger.debug("CoinbaseFundingClient not yet available (coinbase_funding.py missing).")

# ---------------------------------------------------------------------------
# Import BinanceAdapter (fallback data source)
# ---------------------------------------------------------------------------
BINANCE_AVAILABLE = False
try:
    from bin.live.binance_adapter import BinanceAdapter
    BINANCE_AVAILABLE = True
except ImportError:
    logger.debug("BinanceAdapter not available.")

# ---------------------------------------------------------------------------
# Import core engine components (always required)
# ---------------------------------------------------------------------------
from bin.live.v11_shadow_runner import V11ShadowRunner
from bin.live.live_feature_computer import LiveFeatureComputer

# ---------------------------------------------------------------------------
# Import Cointegration Detector (optional, for mean-reversion analysis)
# ---------------------------------------------------------------------------
COINTEGRATION_AVAILABLE = False
try:
    from bin.live.cointegration_detector import compute_cointegration
    COINTEGRATION_AVAILABLE = True
except ImportError:
    logger.debug("CointegrationDetector not available (cointegration_detector.py missing).")

# ---------------------------------------------------------------------------
# Import StressSimulator (optional, for macro stress scenario analysis)
# ---------------------------------------------------------------------------
STRESS_SIMULATOR_AVAILABLE = False
try:
    from bin.live.stress_simulator import StressSimulator
    STRESS_SIMULATOR_AVAILABLE = True
except ImportError:
    logger.debug("StressSimulator not available (stress_simulator.py missing).")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = "configs/bull_machine_isolated_v11_fixed.json"
DEFAULT_INITIAL_CASH = 100_000.0
DEFAULT_COMMISSION_RATE = 0.0002   # Coinbase 0.02% per side
DEFAULT_SLIPPAGE_BPS = 3.0
WARMUP_CANDLES = 1000              # SMA200 + 800 buffer (41 days for solid 1D Wyckoff)
CANDLE_WAIT_MINUTES = 1
CANDLE_WAIT_SECONDS = 30           # Wait until XX:01:30 UTC
FUNDING_LOG_INTERVAL_HOURS = 6
STATE_SAVE_INTERVAL_BARS = 6       # Save state every 6 bars (~6 hours)
COINTEGRATION_INTERVAL_HOURS = 4   # Run cointegration analysis every 4 hours
STRESS_CHECK_INTERVAL_HOURS = 4    # Check stress scenarios every 4 hours
FEATURE_HISTORY_SIZE = 168          # 1 week of hourly bars (for cointegration)


class CoinbasePaperRunner:
    """
    Main runner that orchestrates:
    1. CoinbaseAdapter for market data (1H candles)
    2. LiveFeatureComputer for feature computation (~240 features)
    3. V11ShadowRunner for signal generation + virtual position tracking
    4. CoinbaseFundingClient for funding rate monitoring
    5. Status dashboard + trade logging
    """

    def __init__(
        self,
        config_path: str = DEFAULT_CONFIG,
        initial_cash: float = DEFAULT_INITIAL_CASH,
        commission_rate: float = DEFAULT_COMMISSION_RATE,
        slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
        use_binance_fallback: bool = False,
        verbose: bool = False,
    ):
        self.config_path = config_path
        self.initial_cash = initial_cash
        self.commission_rate = commission_rate
        self.slippage_bps = slippage_bps
        self.use_binance_fallback = use_binance_fallback
        self.verbose = verbose
        self.running = True

        # Output directory for Coinbase paper trading results
        self.output_dir = PROJECT_ROOT / "results" / "coinbase_paper"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # ---- Data Adapter ----
        self.adapter = self._init_adapter()

        # ---- Feature Computer ----
        self.feature_computer = LiveFeatureComputer(buffer_size=WARMUP_CANDLES)
        logger.info("LiveFeatureComputer initialized (buffer_size=%d)", WARMUP_CANDLES)

        # ---- V11 Shadow Runner (signal generation + position tracking) ----
        self.runner = V11ShadowRunner(
            config_path=config_path,
            initial_cash=initial_cash,
            commission_rate=commission_rate,
            slippage_bps=slippage_bps,
        )
        # Override the shadow runner's output dir to our Coinbase-specific one
        self.runner.output_dir = self.output_dir
        self.runner.signal_log_path = self.output_dir / "signals.csv"
        self.runner._init_signal_log()
        logger.info("V11ShadowRunner initialized with config: %s", config_path)

        # ---- Coinbase Funding Client (optional) ----
        self.funding_client = None
        if COINBASE_FUNDING_AVAILABLE:
            try:
                self.funding_client = CoinbaseFundingClient()
                logger.info("CoinbaseFundingClient initialized for funding cost tracking.")
            except Exception as exc:
                logger.warning("CoinbaseFundingClient init failed: %s", exc)

        # ---- Funding cost tracking state ----
        self.funding_costs: Dict[str, float] = {
            "total_funding_cost_usd": 0.0,
            "total_funding_events": 0,
            "last_funding_rate": 0.0,
            "last_funding_timestamp": None,
            "cost_by_position": {},
        }
        self._load_funding_costs()

        # ---- Feature history for rolling macro correlations + cointegration ----
        self.feature_history: List[Dict] = []  # ring buffer, max FEATURE_HISTORY_SIZE

        # ---- Cointegration analysis state ----
        self.last_cointegration_time: Optional[datetime] = None
        self.last_cointegration_result: Optional[Dict] = None

        # ---- Stress Simulator (optional, precomputes at startup) ----
        self.stress_simulator = None
        self.last_stress_check_time: Optional[datetime] = None
        self.active_stress_scenarios: List[Dict] = []
        if STRESS_SIMULATOR_AVAILABLE:
            try:
                self.stress_simulator = StressSimulator()
                stats = self.stress_simulator.precompute_all()
                if stats:
                    logger.info(
                        "StressSimulator initialized: %d scenarios precomputed",
                        len(stats),
                    )
                else:
                    logger.warning("StressSimulator: no scenario stats computed")
                    self.stress_simulator = None
            except Exception as exc:
                logger.warning("StressSimulator init failed: %s", exc)
                self.stress_simulator = None

        # ---- Last signal narrative (for dashboard display) ----
        self.last_signal_narrative: Optional[dict] = None

        # ---- Wyckoff phase transition tracking ----
        self.wyckoff_phase_history = []  # [{from_phase, to_phase, timestamp, price}]
        self.wyckoff_cycle_start = None  # ISO timestamp string
        self.wyckoff_last_phase = 'neutral'

        # ---- Session tracking ----
        self.session_start = datetime.now(timezone.utc)
        self.bars_processed = 0
        self.last_funding_log_time = self.session_start
        self.last_processed_ts: Optional[pd.Timestamp] = None
        self.adapter_source = "coinbase" if not use_binance_fallback else "binance"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_whale_intelligence(self, features: Dict) -> Dict:
        """Build whale/institutional intelligence payload for heartbeat.

        Surfaces raw derivatives data (OKX/Binance/CoinGlass) alongside
        the engine's derived signals so the dashboard can explain what real
        data means AND how the engine uses it for regime detection and
        archetype gating.
        """
        sf = self._safe_float

        # --- Raw institutional data (from derivatives API) ---
        oi_change_4h = sf(features.get('oi_change_4h'))
        oi_change_24h = sf(features.get('oi_change_24h'))
        oi_value = sf(features.get('oi_value'), scale=1, decimals=0)
        funding_rate = sf(features.get('binance_funding_rate'), decimals=6)
        funding_z = sf(features.get('funding_Z', features.get('funding_z')))
        ls_ratio = sf(features.get('ls_ratio_extreme'))
        taker_imbalance = sf(features.get('taker_imbalance'))

        # --- Engine-derived signals (what the engine computes) ---
        oi_price_div = sf(features.get('oi_price_divergence'))
        funding_oi_div = sf(features.get('funding_oi_divergence'))

        # Derivatives heat (CMI sub-component)
        oi_4h_val = features.get('oi_change_4h')
        has_data = oi_4h_val is not None and not (isinstance(oi_4h_val, float) and oi_4h_val != oi_4h_val)

        if has_data:
            oi_4h = float(oi_4h_val) if oi_4h_val == oi_4h_val else 0.0
            oi_momentum = min(max(oi_4h + 0.5, 0.0), 1.0)
            fr = float(features.get('binance_funding_rate', 0) or 0)
            funding_health = max(1.0 - abs(fr) * 5000.0, 0.0)
            tk = float(features.get('taker_imbalance', 0) or 0)
            taker_conviction = min(max(tk + 0.5, 0.0), 1.0)
            deriv_heat = round(0.40 * oi_momentum + 0.30 * funding_health + 0.30 * taker_conviction, 4)
        else:
            oi_momentum = None
            funding_health = None
            taker_conviction = None
            deriv_heat = None

        # Whale conflict count (mirrors archetype_instance._compute_whale_conflict)
        conflict_count = 0
        direction = "long"  # Default perspective for dashboard
        if funding_z is not None and funding_z > 2.0:
            conflict_count += 1
        if oi_change_4h is not None and oi_change_4h < -0.05:
            conflict_count += 1
        if taker_imbalance is not None and taker_imbalance < -0.5:
            conflict_count += 1
        if ls_ratio is not None and ls_ratio > 2.0:
            conflict_count += 1

        penalty_tiers = [1.00, 0.95, 0.90, 0.85, 0.80]
        whale_penalty = penalty_tiers[min(conflict_count, 4)]

        # Overall institutional sentiment
        if not has_data:
            sentiment = "no_data"
        elif conflict_count >= 3:
            sentiment = "strongly_bearish"
        elif conflict_count >= 2:
            sentiment = "bearish"
        elif oi_change_4h is not None and oi_change_4h > 0.02 and taker_imbalance is not None and taker_imbalance > 0.2:
            sentiment = "bullish"
        elif oi_change_4h is not None and oi_change_4h > 0.05:
            sentiment = "strongly_bullish"
        else:
            sentiment = "neutral"

        return {
            # Raw data from derivatives API (OKX/Binance/CoinGlass)
            "raw": {
                "oi_value": oi_value,
                "oi_change_4h": oi_change_4h,
                "oi_change_24h": oi_change_24h,
                "funding_rate": funding_rate,
                "funding_z": funding_z,
                "ls_ratio_extreme": ls_ratio,
                "taker_imbalance": taker_imbalance,
            },
            # Engine-derived signals
            "derived": {
                "oi_price_divergence": oi_price_div,
                "funding_oi_divergence": funding_oi_div,
                "derivatives_heat": deriv_heat,
                "oi_momentum": round(oi_momentum, 4) if oi_momentum is not None else None,
                "funding_health": round(funding_health, 4) if funding_health is not None else None,
                "taker_conviction": round(taker_conviction, 4) if taker_conviction is not None else None,
            },
            # Whale conflict assessment
            "conflict": {
                "count": conflict_count,
                "penalty_multiplier": whale_penalty,
                "signals": {
                    "funding_overcrowded": funding_z is not None and funding_z > 2.0,
                    "oi_declining": oi_change_4h is not None and oi_change_4h < -0.05,
                    "aggressive_selling": taker_imbalance is not None and taker_imbalance < -0.5,
                    "ls_ratio_extreme": ls_ratio is not None and ls_ratio > 2.0,
                },
            },
            # Overall sentiment
            "sentiment": sentiment,
            "has_data": has_data,
            # CMI integration status
            "cmi_status": {
                "derivatives_heat_weight": 0.0,
                "note": "Disabled pending >2 years of OI data. Computed but not factored into threshold.",
            },
        }

    @staticmethod
    def _safe_float(val, scale=1.0, decimals=4):
        """Safely convert a feature value to float, handling NaN/None."""
        if val is None:
            return None
        try:
            f = float(val)
            if f != f:  # NaN check
                return None
            return round(f * scale, decimals)
        except (TypeError, ValueError):
            return None

    def _enrich_legacy_narrative(self, pos) -> Dict:
        """Enrich a position's entry_narrative that was created before domain_scores existed.

        Rebuilds headline, confluence_factors, risk_factors, and domain_scores
        from the position's stored data. Only called when entry_narrative
        is missing the new fields.
        """
        old = pos.entry_narrative or {}
        gv = old.get('gate_values', {})

        archetype = pos.archetype
        direction = pos.direction
        entry_price = pos.entry_price
        fusion_score = pos.fusion_score
        threshold = pos.threshold_at_entry

        price_str = f"${entry_price:,.0f}"
        margin = fusion_score - threshold
        bypassed = margin < 0

        # Extract gate values (use what was stored, or defaults)
        rsi = gv.get('rsi_14', 50)
        adx = gv.get('adx', 20)
        atr = gv.get('atr_14', 0)
        wick = gv.get('wick_ratio', 1.0)
        boms = gv.get('boms_strength', 0)
        bos_bull = gv.get('bos_bullish', 0)
        bos_bear = gv.get('bos_bearish', 0)
        vol_z = gv.get('volume_z', 0)
        fz = gv.get('funding_z', 0)
        chop = gv.get('chop_score', 0.5)
        fvg = gv.get('fvg_present', 0)
        fg_raw = gv.get('fear_greed', 50)

        # Domain scores from factor_attribution if available
        fa = pos.entry_factor_attribution or {}
        ec = fa.get('entry_conditions', {})
        fw = ec.get('fusion_weights', {})
        w_score = fw.get('wyckoff', 0.0) if fw else 0.0
        l_score = fw.get('liquidity', 0.0) if fw else 0.0
        m_score = fw.get('momentum', 0.0) if fw else 0.0
        s_score = fw.get('smc', 0.0) if fw else 0.0

        # Build headline per archetype
        support_or_resist = 'support' if direction == 'long' else 'resistance'
        headlines = {
            'trap_within_trend': f"False breakdown at {price_str} — wick {wick:.1f}x body rejected {support_or_resist}, trapping {'bears' if direction == 'long' else 'bulls'}.",
            'wick_trap': f"{'Lower' if direction == 'long' else 'Upper'} wick anomaly ({wick:.1f}x body) at {price_str} rejected {support_or_resist}. RSI {rsi:.0f}, BOS {'confirmed' if (bos_bull or bos_bear) else 'pending'}.",
            'liquidity_vacuum': f"Capitulation at {price_str} — volume {vol_z:+.1f}σ exhausted {'sellers' if direction == 'long' else 'buyers'}.",
            'liquidity_sweep': f"Price swept {'below' if direction == 'long' else 'above'} {price_str} triggering stops. BOMS {boms:.3f}. RSI {rsi:.0f}.",
            'retest_cluster': f"Fakeout then real move at {price_str}. Volume {vol_z:+.1f}σ on retest confirms genuine demand.",
            'spring': f"Wyckoff spring at {price_str} — price broke {support_or_resist}, shook out {'bears' if direction == 'long' else 'bulls'}, reversed.",
            'order_block_retest': f"Order block retest at {price_str}. BOMS {boms:.3f}, BOS {'confirmed' if (bos_bull or bos_bear) else 'pending'}.",
            'failed_continuation': f"{'Bearish' if direction == 'long' else 'Bullish'} move failed at {price_str}. RSI {rsi:.0f}, ADX {adx:.0f} — momentum dying.",
            'confluence_breakout': f"Consolidation breakout at {price_str}. ATR ${atr:.0f}, BOMS {boms:.3f}.",
            'fvg_continuation': f"FVG displacement at {price_str}. BOS {'confirmed' if (bos_bull or bos_bear) else 'pending'}.",
            'funding_divergence': f"Funding divergence at {price_str}: Z={fz:+.2f}. Price holding despite negative funding.",
            'long_squeeze': f"Long squeeze at {price_str}: funding Z={fz:+.2f}, RSI {rsi:.0f} overheated.",
            'liquidity_compression': f"Range compression at {price_str}: ATR ${atr:.0f}, chop {chop:.2f}.",
            'exhaustion_reversal': f"Exhaustion at {price_str}: RSI {rsi:.0f}, ATR ${atr:.0f}, vol {vol_z:+.1f}σ.",
            'volume_fade_chop': f"Volume fade at {price_str}. Vol {vol_z:+.1f}σ, RSI {rsi:.0f}, ADX {adx:.0f}.",
            'whipsaw': f"Whipsaw at {price_str}: wick {wick:.1f}x body, vol {vol_z:+.1f}σ, chop {chop:.2f}.",
        }
        headline = headlines.get(archetype, f"{archetype.replace('_', ' ').title()} at {price_str}. Fusion {fusion_score:.3f}.")

        # Build confluence
        confluence = []
        if bos_bull and direction == 'long':
            confluence.append('Break of Structure (bullish)')
        if bos_bear and direction == 'short':
            confluence.append('Break of Structure (bearish)')
        if boms > 0.3:
            confluence.append(f"BOMS displacement {boms:.2f}")
        if adx > 25:
            confluence.append(f"Trending market (ADX {adx:.0f})")
        if fvg:
            confluence.append('FVG present')
        if abs(vol_z) > 1.5:
            confluence.append(f"Volume anomaly ({vol_z:+.1f}σ)")
        if wick > 2.0:
            confluence.append(f"Wick rejection ({wick:.1f}x body)")

        # Build risk
        risk_factors = []
        if abs(fz) > 1.0:
            risk_factors.append(f"Funding {'elevated' if fz > 0 else 'negative'} (Z={fz:+.1f})")
        if chop > 0.7:
            risk_factors.append(f"Choppy market (chop={chop:.2f})")
        if fg_raw < 20:
            risk_factors.append(f"Extreme Fear (F&G={fg_raw:.0f})")
        elif fg_raw > 80:
            risk_factors.append(f"Extreme Greed (F&G={fg_raw:.0f})")
        if adx < 15:
            risk_factors.append(f"No clear trend (ADX {adx:.0f})")
        if bypassed:
            risk_factors.append(f"BYPASSED threshold (fusion {fusion_score:.3f} < threshold {threshold:.3f})")

        # Merge with old narrative, preserving everything else
        enriched = dict(old)
        enriched['headline'] = headline
        enriched['summary'] = headline
        enriched['trigger'] = headline.split('.')[0] + '.'
        enriched['confluence_factors'] = confluence
        enriched['confluence'] = confluence
        enriched['risk_factors'] = risk_factors
        enriched['domain_scores'] = {
            'wyckoff': round(w_score, 3),
            'liquidity': round(l_score, 3),
            'momentum': round(m_score, 3),
            'smc': round(s_score, 3),
        }
        enriched['regime_context'] = (
            f"{pos.regime_at_entry.replace('_', ' ').title()} "
            f"(risk_temp={pos.risk_temp_at_entry:.2f}), "
            f"threshold={threshold:.3f}, "
            f"{'BYPASSED' if bypassed else 'passed'} with "
            f"{'+' if margin >= 0 else ''}{margin:.3f} margin"
        )
        return enriched

    def _serialize_position(self, pos, close_price: float) -> Dict:
        """Serialize a TrackedPosition to a rich dict for heartbeat/dashboard."""
        entry_p = float(pos.entry_price)
        current_p = float(close_price) if close_price > 0 else entry_p
        orig_qty = float(pos.original_quantity)
        cur_qty = float(pos.current_quantity)
        leverage = float(getattr(self.runner, 'leverage', 1.0))

        # P&L calculation
        if pos.direction == 'long':
            unrealized_pnl = (current_p - entry_p) * cur_qty
        else:
            unrealized_pnl = (entry_p - current_p) * cur_qty
        entry_value = entry_p * orig_qty
        unrealized_pnl_pct = (unrealized_pnl / entry_value * 100) if entry_value > 0 else 0.0

        # SL/TP distances
        sl = float(pos.stop_loss) if pos.stop_loss and pos.stop_loss == pos.stop_loss else 0.0
        tp = float(pos.take_profit) if pos.take_profit and pos.take_profit == pos.take_profit else 0.0
        sl_distance_pct = abs(entry_p - sl) / entry_p * 100 if entry_p > 0 and sl > 0 else 0.0
        tp_distance_pct = abs(tp - entry_p) / entry_p * 100 if entry_p > 0 and tp > 0 else 0.0

        # Risk:reward
        risk = abs(entry_p - sl) if sl > 0 else 0.0
        reward = abs(tp - entry_p) if tp > 0 else 0.0
        risk_reward = round(reward / risk, 2) if risk > 0 else 0.0

        # Position size in USD
        position_size_usd = entry_p * orig_qty

        return {
            "id": pos.position_id,
            "archetype": pos.archetype,
            "direction": pos.direction,
            "entry_time": str(pos.entry_time),
            "entry_price": round(entry_p, 2),
            "current_price": round(current_p, 2),
            "unrealized_pnl": round(unrealized_pnl, 2),
            "unrealized_pnl_pct": round(unrealized_pnl_pct, 2),
            "stop_loss": round(sl, 2) if sl > 0 else None,
            "take_profit": round(tp, 2) if tp > 0 else None,
            "position_size_usd": round(position_size_usd, 2),
            "leverage": leverage,
            "risk_reward": risk_reward,
            "sl_distance_pct": round(sl_distance_pct, 2),
            "tp_distance_pct": round(tp_distance_pct, 2),
            "bars_held": pos.bars_held,
            "fusion_score": round(pos.fusion_score, 4),
            "regime_at_entry": pos.regime_at_entry,
            "original_quantity": orig_qty,
            "current_quantity": cur_qty,
            "atr_at_entry": round(float(pos.atr_at_entry), 2) if pos.atr_at_entry and pos.atr_at_entry == pos.atr_at_entry else None,
            "trailing_stop": round(float(pos.trailing_stop), 2) if pos.trailing_stop and pos.trailing_stop == pos.trailing_stop else None,
            "executed_scale_outs": pos.executed_scale_outs or [],
            "total_exits_pct": round(pos.total_exits_pct, 4),
            "threshold_at_entry": round(pos.threshold_at_entry, 4),
            "threshold_margin": round(pos.threshold_margin, 4),
            "would_have_passed": pos.would_have_passed,
            "risk_temp_at_entry": round(pos.risk_temp_at_entry, 4),
            "instability_at_entry": round(pos.instability_at_entry, 4),
            "crisis_prob_at_entry": round(pos.crisis_prob_at_entry, 4),
            "narrative": self._enrich_legacy_narrative(pos) if not (pos.entry_narrative or {}).get('domain_scores') else pos.entry_narrative,
            "factor_attribution": pos.entry_factor_attribution,
        }

    def _build_wyckoff_context(self, features) -> Dict:
        """
        Extract raw market microstructure metrics that explain WHY Wyckoff events
        fired. These are the same features the detection engine evaluates.
        """
        _sf = self._safe_float
        close = features.get('close', 0) or 0
        high = features.get('high', 0) or 0
        low = features.get('low', 0) or 0
        bar_range = high - low if high > low else 1e-10

        # Close position within bar (0=bottom, 1=top)
        close_position = (close - low) / bar_range if bar_range > 0 else 0.5

        # Lower/upper wick quality
        lower_wick = (min(features.get('open', close), close) - low) / bar_range if bar_range > 0 else 0
        upper_wick = (high - max(features.get('open', close), close)) / bar_range if bar_range > 0 else 0

        return {
            'volume_z': _sf(features.get('volume_z', features.get('volume_z_20', None))),
            'close_position': round(close_position, 3),
            'lower_wick_pct': round(max(0, lower_wick), 3),
            'upper_wick_pct': round(max(0, upper_wick), 3),
            'rsi_14': _sf(features.get('rsi_14')),
            'adx': _sf(features.get('adx', features.get('adx_14', None))),
            'range_z': _sf(features.get('range_z', features.get('atr_z', None))),
            'close': round(close, 2),
        }

    def _compute_macro_outlook(self, features, heartbeat_macro):
        """
        Synthesize macro signals into directional BTC outlook at 4 timeframes.
        Uses existing macro knowledge from macro_engine.py and macro_signals.py.

        Returns dict with keys: '1w', '1m', '6m', '1y' — each containing:
          - score, label, regime, states, trader_signals, narrative,
          - bull_case, bear_case, key_movers, factors
        """
        m = heartbeat_macro or {}

        # ---- Extract raw signals ----
        fg = m.get('fear_greed')
        btc_d = m.get('btc_dominance')
        usdt_d = m.get('usdt_dominance')
        usdc_d = m.get('usdc_dominance')
        vix_z = m.get('vix_z')
        dxy_z = m.get('dxy_z')
        gold_z = m.get('gold_z')
        oil_z = m.get('oil_z')
        yc = m.get('yield_curve')
        # Total stablecoin dominance (USDT + USDC)
        total_stable_d = None
        if usdt_d is not None:
            total_stable_d = usdt_d + (usdc_d or 0)
        elif usdc_d is not None:
            total_stable_d = usdc_d
        risk_temp = getattr(self.runner, 'last_risk_temp', features.get('risk_temperature', 0.5))
        crisis = getattr(self.runner, 'last_crisis_prob', features.get('crisis_prob', 0.0))
        funding_z = m.get('funding_z') if m.get('funding_z') is not None else (
            self._safe_float(features.get('funding_Z', None))
        )

        # ---- Per-factor signals: +1 = bullish, -1 = bearish ----
        signals = {}

        if fg is not None:
            if fg <= 20:      signals['fear_greed'] = +0.8
            elif fg <= 35:    signals['fear_greed'] = +0.4
            elif fg >= 80:    signals['fear_greed'] = -0.8
            elif fg >= 65:    signals['fear_greed'] = -0.4
            else:             signals['fear_greed'] = 0.0

        if dxy_z is not None:
            signals['dxy'] = float(np.clip(-dxy_z * 0.5, -1.0, 1.0))

        if vix_z is not None:
            signals['vix'] = +0.3 if vix_z > 2.0 else float(np.clip(-vix_z * 0.4, -1.0, 1.0))

        if gold_z is not None:
            signals['gold'] = float(np.clip(-gold_z * 0.2, -0.5, 0.5))

        if oil_z is not None:
            signals['oil'] = float(np.clip(-oil_z * 0.3, -0.7, 0.7))

        if yc is not None:
            if yc < -0.5:    signals['yield_curve'] = -0.7
            elif yc < 0:     signals['yield_curve'] = -0.3
            elif yc > 1.0:   signals['yield_curve'] = +0.3
            else:             signals['yield_curve'] = +0.1

        if btc_d is not None:
            if btc_d > 55:   signals['btc_dominance'] = +0.3
            elif btc_d > 50: signals['btc_dominance'] = +0.1
            elif btc_d < 40: signals['btc_dominance'] = -0.2
            else:            signals['btc_dominance'] = 0.0

        if total_stable_d is not None:
            if total_stable_d > 10:  signals['usdt_dominance'] = -0.5
            elif total_stable_d > 8: signals['usdt_dominance'] = -0.2
            elif total_stable_d < 5: signals['usdt_dominance'] = +0.3
            else:                    signals['usdt_dominance'] = 0.0

        if funding_z is not None:
            if funding_z > 2.0:   signals['funding'] = -0.6
            elif funding_z > 1.0: signals['funding'] = -0.2
            elif funding_z < -2.0: signals['funding'] = +0.6
            elif funding_z < -1.0: signals['funding'] = +0.2
            else:                  signals['funding'] = 0.0

        signals['risk_temp'] = float(np.clip((risk_temp - 0.5) * 2.0, -1.0, 1.0))
        signals['crisis'] = float(np.clip(-crisis * 2.0, -1.0, 0.0)) if crisis > 0.3 else 0.0

        # ---- Macro state detection (from macro_signals.py logic) ----
        states = {}
        if dxy_z is not None:
            if dxy_z > 1.5:    states['dxy'] = 'breakout'
            elif dxy_z < -1.5: states['dxy'] = 'breakdown'
            else:              states['dxy'] = 'neutral'
        else: states['dxy'] = 'neutral'

        if oil_z is not None:
            if oil_z > 1.5:    states['oil'] = 'hot'
            elif oil_z < -1.5: states['oil'] = 'cool'
            else:              states['oil'] = 'neutral'
        else: states['oil'] = 'neutral'

        if gold_z is not None:
            states['gold'] = 'flight' if gold_z > 1.5 else 'neutral'
        else: states['gold'] = 'neutral'

        if yc is not None:
            if yc < -0.3:   states['yield'] = 'inverted'
            elif yc > 0.5:  states['yield'] = 'steepening'
            else:           states['yield'] = 'neutral'
        else: states['yield'] = 'neutral'

        states['curve'] = states['yield']

        if btc_d is not None:
            if btc_d > 60:   states['breadth'] = 'weak'
            elif btc_d < 50: states['breadth'] = 'strong'
            else:            states['breadth'] = 'neutral'
        else: states['breadth'] = 'neutral'

        if total_stable_d is not None:
            if total_stable_d > 10:   states['usdt_d'] = 'breakout'
            elif total_stable_d < 5:  states['usdt_d'] = 'breakdown'
            else:                     states['usdt_d'] = 'neutral'
        else: states['usdt_d'] = 'neutral'

        # ---- Regime classification (from macro_pulse.py MacroRegime logic) ----
        # Stagflation: Oil hot + DXY breaking out = poison
        if states['oil'] == 'hot' and states['dxy'] == 'breakout':
            regime = 'STAGFLATION'
            regime_label = 'Stagflation'
        # Risk-off: multiple bearish signals
        elif (states['dxy'] == 'breakout' or (vix_z is not None and vix_z > 1.5) or
              states['gold'] == 'flight' or states['usdt_d'] == 'breakout'):
            bearish_count = sum([
                states['dxy'] == 'breakout',
                vix_z is not None and vix_z > 1.5,
                states['gold'] == 'flight',
                states['usdt_d'] == 'breakout',
                states['yield'] == 'inverted',
            ])
            if bearish_count >= 2:
                regime = 'RISK_OFF'
                regime_label = 'Risk-Off'
            else:
                regime = 'NEUTRAL'
                regime_label = 'Neutral'
        # Risk-on: DXY weak + breadth strong + no crisis
        elif states['dxy'] == 'breakdown' and crisis < 0.3:
            regime = 'RISK_ON'
            regime_label = 'Risk-On'
        else:
            regime = 'NEUTRAL'
            regime_label = 'Neutral'

        # ---- Trader signal breakdown (from macro_engine.py) ----
        trader_signals = {'wyckoff': [], 'moneytaur': [], 'zeroika': []}

        # Wyckoff Insider: VIX/MOVE crisis, Gold/DXY fiat rotation, DXY+VIX synergy
        if vix_z is not None:
            if vix_z > 1.5:
                trader_signals['wyckoff'].append(f"VIX elevated ({vix_z:+.1f}σ) - risk-off")
            else:
                trader_signals['wyckoff'].append(f"VIX calm ({vix_z:+.1f}σ)")
        if gold_z is not None and gold_z > 1.0:
            trader_signals['wyckoff'].append(f"Gold flight-to-safety ({gold_z:+.1f}σ)")
        if dxy_z is not None and vix_z is not None and dxy_z > 1.5 and vix_z > 1.5:
            trader_signals['wyckoff'].append("DXY+VIX synergy trap - crisis mode")
        if total_stable_d is not None and total_stable_d > 10:
            trader_signals['wyckoff'].append(f"Stablecoin dominance ({total_stable_d:.1f}% USDT+USDC) - alt bleed")

        # Moneytaur: DXY/Oil rotations, funding/OI leverage, yields
        if dxy_z is not None:
            if dxy_z > 1.0:
                trader_signals['moneytaur'].append(f"Strong dollar ({dxy_z:+.1f}σ) - liquidity drain")
            elif dxy_z < -1.0:
                trader_signals['moneytaur'].append(f"Weak dollar ({dxy_z:+.1f}σ) - favorable")
        if oil_z is not None:
            if oil_z > 1.0:
                trader_signals['moneytaur'].append(f"Oil pressure ({oil_z:+.1f}σ) - inflation risk")
            elif oil_z < -1.0:
                trader_signals['moneytaur'].append(f"Oil relief ({oil_z:+.1f}σ) - easing")
        if funding_z is not None and abs(funding_z) > 1.0:
            trader_signals['moneytaur'].append(
                f"Funding {'overcrowded longs' if funding_z > 0 else 'shorts squeezable'} (Z={funding_z:+.1f})"
            )
        if yc is not None:
            if yc < -0.3:
                trader_signals['moneytaur'].append(f"Yield curve 10Y-5Y inverted ({yc:.2f}%) - recession")
            elif yc > 0.5:
                trader_signals['moneytaur'].append(f"Yield curve 10Y-5Y steepening ({yc:.2f}%) - growth")

        # ZeroIKA: USD strength, dominance coils, structural shifts
        if dxy_z is not None:
            if abs(dxy_z) < 0.5:
                trader_signals['zeroika'].append("USD neutral - no structural signal")
            elif dxy_z > 1.5:
                trader_signals['zeroika'].append("USD strength regime - structural headwind")
            elif dxy_z < -1.5:
                trader_signals['zeroika'].append("USD weakness regime - structural tailwind")
        if btc_d is not None:
            if btc_d > 60:
                trader_signals['zeroika'].append(f"BTC dominance coil ({btc_d:.1f}%) - alt suppression")
            elif btc_d < 45:
                trader_signals['zeroika'].append(f"BTC dominance low ({btc_d:.1f}%) - alt rotation")

        # ---- Build narrative ----
        narrative_parts = []
        if fg is not None:
            fg_desc = 'Extreme Fear' if fg <= 20 else 'Fear' if fg <= 35 else 'Extreme Greed' if fg >= 80 else 'Greed' if fg >= 65 else 'Neutral sentiment'
            narrative_parts.append(f"{fg_desc} (F&G={fg:.0f})")
        if dxy_z is not None and abs(dxy_z) > 1.0:
            narrative_parts.append(f"{'weakening' if dxy_z < 0 else 'strengthening'} dollar ({dxy_z:+.1f}σ)")
        if regime == 'STAGFLATION':
            narrative_parts.append("Oil+DXY both rising = stagflation poison")
        if total_stable_d is not None and total_stable_d > 10:
            narrative_parts.append(f"Stablecoin dominance at {total_stable_d:.1f}% (USDT+USDC) shows capital fleeing to stables")
        if gold_z is not None and gold_z > 1.0:
            narrative_parts.append(f"Gold bid ({gold_z:+.1f}σ) = flight-to-safety active")
        if funding_z is not None and abs(funding_z) > 1.5:
            side = 'longs overcrowded' if funding_z > 0 else 'shorts squeezable'
            narrative_parts.append(f"Funding {side} (Z={funding_z:+.1f})")
        # Add trader perspective summaries
        if trader_signals['wyckoff']:
            narrative_parts.append(f"Wyckoff: {trader_signals['wyckoff'][0]}")
        if trader_signals['moneytaur']:
            narrative_parts.append(f"Moneytaur: {trader_signals['moneytaur'][0]}")

        narrative = '. '.join(narrative_parts) + '.' if narrative_parts else 'Insufficient data for narrative.'

        # ---- Timeframe weights (unchanged) ----
        weight_profiles = {
            '1w': {
                'fear_greed': 0.20, 'funding': 0.15, 'vix': 0.15,
                'risk_temp': 0.20, 'crisis': 0.10,
                'dxy': 0.05, 'oil': 0.05, 'gold': 0.03,
                'yield_curve': 0.02, 'btc_dominance': 0.03, 'usdt_dominance': 0.02,
            },
            '1m': {
                'fear_greed': 0.12, 'funding': 0.08, 'vix': 0.10,
                'risk_temp': 0.15, 'crisis': 0.10,
                'dxy': 0.15, 'oil': 0.08, 'gold': 0.05,
                'yield_curve': 0.07, 'btc_dominance': 0.05, 'usdt_dominance': 0.05,
            },
            '6m': {
                'fear_greed': 0.05, 'funding': 0.03, 'vix': 0.05,
                'risk_temp': 0.10, 'crisis': 0.05,
                'dxy': 0.20, 'oil': 0.10, 'gold': 0.10,
                'yield_curve': 0.15, 'btc_dominance': 0.10, 'usdt_dominance': 0.07,
            },
            '1y': {
                'fear_greed': 0.03, 'funding': 0.02, 'vix': 0.03,
                'risk_temp': 0.05, 'crisis': 0.02,
                'dxy': 0.20, 'oil': 0.12, 'gold': 0.12,
                'yield_curve': 0.20, 'btc_dominance': 0.12, 'usdt_dominance': 0.09,
            },
        }

        outlook = {}
        for tf, weights in weight_profiles.items():
            total_score = 0.0
            total_weight = 0.0
            factors = []
            bull_factors = []
            bear_factors = []
            bull_score = 0.0
            bear_score = 0.0

            for factor_name, weight in weights.items():
                if factor_name in signals:
                    sig = signals[factor_name]
                    contribution = sig * weight
                    total_score += contribution
                    total_weight += weight
                    factor_entry = {
                        'name': factor_name,
                        'signal': round(sig, 3),
                        'weight': round(weight, 3),
                        'contribution': round(contribution, 4),
                    }
                    factors.append(factor_entry)

                    # Separate bull vs bear
                    if sig > 0.05:
                        bull_score += contribution
                        bull_factors.append(f"{factor_name.replace('_', ' ').title()} ({sig:+.2f})")
                    elif sig < -0.05:
                        bear_score += contribution
                        bear_factors.append(f"{factor_name.replace('_', ' ').title()} ({sig:+.2f})")

            normalized_score = total_score / total_weight if total_weight > 0 else 0.0
            normalized_score = float(np.clip(normalized_score, -1.0, 1.0))

            if normalized_score >= 0.4:     label = "Strong Bull"
            elif normalized_score >= 0.15:  label = "Bullish"
            elif normalized_score > -0.15:  label = "Neutral"
            elif normalized_score > -0.4:   label = "Bearish"
            else:                           label = "Strong Bear"

            sorted_factors = sorted(factors, key=lambda x: abs(x['contribution']), reverse=True)

            outlook[tf] = {
                'score': round(normalized_score, 3),
                'label': label,
                'regime': regime,
                'regime_label': regime_label,
                'states': states,
                'trader_signals': trader_signals,
                'narrative': narrative,
                'bull_case': {
                    'score': round(bull_score, 4),
                    'factors': bull_factors[:5],
                },
                'bear_case': {
                    'score': round(bear_score, 4),
                    'factors': bear_factors[:5],
                },
                'key_movers': [f['name'] for f in sorted_factors[:3]],
                'factors': sorted_factors,
            }

        return outlook

    def _compute_capital_flows(self, features, heartbeat_macro):
        """
        Compute intermarket capital flow data for the SVG diagram.
        Based on relationships from engine/context/macro_pulse.py.
        """
        m = heartbeat_macro or {}
        dxy_z = m.get('dxy_z')
        vix_z = m.get('vix_z')
        gold_z = m.get('gold_z')
        oil_z = m.get('oil_z')
        yc = m.get('yield_curve')
        btc_d = m.get('btc_dominance')
        usdt_d = m.get('usdt_dominance')
        usdc_d = m.get('usdc_dominance')
        fg = m.get('fear_greed')
        # Total stablecoin dominance (USDT + USDC)
        total_stable_d = None
        if usdt_d is not None:
            total_stable_d = usdt_d + (usdc_d or 0)
        elif usdc_d is not None:
            total_stable_d = usdc_d
        btc_price = features.get('close', 0)

        def _s(v):
            """Safe float for display."""
            if v is None:
                return 'N/A'
            return f"{v:+.2f}σ" if abs(v) < 100 else f"{v:.1f}"

        # ---- Nodes: 8 asset classes with live readings ----
        nodes = {
            'dollar': {
                'label': 'Dollar (DXY)', 'value': _s(dxy_z),
                'status': 'bearish' if dxy_z and dxy_z > 1.0 else 'bullish' if dxy_z and dxy_z < -1.0 else 'neutral',
                'state': 'breakout' if dxy_z and dxy_z > 1.5 else 'breakdown' if dxy_z and dxy_z < -1.5 else 'neutral',
            },
            'bonds': {
                'label': 'Bonds', 'value': f"{yc:.2f}%" if yc is not None else 'N/A',
                'status': 'bearish' if yc and yc < -0.3 else 'bullish' if yc and yc > 0.5 else 'neutral',
                'state': 'inverted' if yc and yc < -0.3 else 'steepening' if yc and yc > 0.5 else 'neutral',
            },
            'equities': {
                'label': 'Equities (VIX)', 'value': _s(vix_z),
                'status': 'bearish' if vix_z and vix_z > 1.5 else 'bullish' if vix_z and vix_z < -0.5 else 'neutral',
                'state': 'spike' if vix_z and vix_z > 1.5 else 'calm' if vix_z and vix_z < 0 else 'neutral',
            },
            'crypto': {
                'label': 'Crypto (BTC)', 'value': f"${btc_price:,.0f}" if btc_price else 'N/A',
                'status': 'neutral',
                'state': 'neutral',
            },
            'altcoins': {
                'label': 'Altcoins', 'value': f"{btc_d:.1f}%" if btc_d else 'N/A',
                'status': 'bullish' if btc_d and btc_d < 50 else 'bearish' if btc_d and btc_d > 60 else 'neutral',
                'state': 'rotation' if btc_d and btc_d < 50 else 'suppressed' if btc_d and btc_d > 60 else 'neutral',
            },
            'stablecoins': {
                'label': 'Stablecoins', 'value': f"{total_stable_d:.1f}%" if total_stable_d else 'N/A',
                'status': 'bearish' if total_stable_d and total_stable_d > 10 else 'bullish' if total_stable_d and total_stable_d < 5 else 'neutral',
                'state': 'breakout' if total_stable_d and total_stable_d > 10 else 'breakdown' if total_stable_d and total_stable_d < 5 else 'neutral',
                'detail': f"USDT {usdt_d:.1f}% + USDC {usdc_d:.1f}%" if usdt_d and usdc_d else None,
            },
            'gold': {
                'label': 'Gold', 'value': _s(gold_z),
                'status': 'bearish' if gold_z and gold_z > 1.0 else 'neutral',
                'state': 'flight' if gold_z and gold_z > 1.5 else 'neutral',
            },
            'oil': {
                'label': 'Oil/Energy', 'value': _s(oil_z),
                'status': 'bearish' if oil_z and oil_z > 1.5 else 'bullish' if oil_z and oil_z < -1.5 else 'neutral',
                'state': 'hot' if oil_z and oil_z > 1.5 else 'cool' if oil_z and oil_z < -1.5 else 'neutral',
            },
        }

        # ---- Edges: 10 intermarket flows ----
        edges = {
            'liquidity_drain': {
                'from': 'dollar', 'to': 'crypto', 'direction': 'drain',
                'active': bool(dxy_z and dxy_z > 1.0),
                'strength': min(1.0, max(0, (dxy_z or 0) - 0.5) / 2.0),
                'label': 'DXY breakout drains crypto liquidity',
                'source_fn': 'dxy_breakout_strength()',
            },
            'liquidity_flow': {
                'from': 'dollar', 'to': 'crypto', 'direction': 'flow',
                'active': bool(dxy_z and dxy_z < -1.0),
                'strength': min(1.0, max(0, -(dxy_z or 0) - 0.5) / 2.0),
                'label': 'Weak dollar flows capital to crypto',
                'source_fn': '_check_dxy_breakdown()',
            },
            'stagflation': {
                'from': 'oil', 'to': 'dollar', 'direction': 'drain',
                'active': bool(oil_z and oil_z > 1.0 and dxy_z and dxy_z > 0.5),
                'strength': 0.85 if (oil_z and oil_z > 1.0 and dxy_z and dxy_z > 0.5) else 0.0,
                'label': 'Oil + DXY both rising = stagflation poison',
                'source_fn': 'oil_dxy_stagflation()',
            },
            'flight_to_safety': {
                'from': 'equities', 'to': 'gold', 'direction': 'flow',
                'active': bool(gold_z and gold_z > 1.0),
                'strength': min(1.0, max(0, (gold_z or 0) - 0.5) / 2.0),
                'label': 'Gold rising = flight-to-safety from risk assets',
                'source_fn': 'gold_flight_to_safety()',
            },
            'bond_stress': {
                'from': 'bonds', 'to': 'dollar', 'direction': 'drain',
                'active': bool(yc and yc < -0.3),
                'strength': min(1.0, max(0, -(yc or 0)) / 1.0),
                'label': 'Yield curve 10Y-5Y inversion = tightening/recession',
                'source_fn': 'yields_spike()',
            },
            'credit_contagion': {
                'from': 'bonds', 'to': 'equities', 'direction': 'drain',
                'active': bool(yc and yc < -0.5 and vix_z and vix_z > 1.0),
                'strength': 0.7 if (yc and yc < -0.5 and vix_z and vix_z > 1.0) else 0.0,
                'label': 'Bond stress + VIX spike = credit contagion',
                'source_fn': 'hyg_credit_stress()',
            },
            'risk_appetite': {
                'from': 'equities', 'to': 'crypto', 'direction': 'flow',
                'active': bool(vix_z and vix_z < 0),
                'strength': min(1.0, max(0, -(vix_z or 0)) / 2.0),
                'label': 'Low VIX = risk appetite flows to crypto',
                'source_fn': 'vix_move_spike()',
            },
            'crypto_rotation': {
                'from': 'crypto', 'to': 'altcoins', 'direction': 'flow',
                'active': bool(btc_d and btc_d < 50),
                'strength': min(1.0, max(0, 50 - (btc_d or 55)) / 15.0),
                'label': 'BTC dominance falling = alt rotation',
                'source_fn': 'total3_vs_total()',
            },
            'stablecoin_flight': {
                'from': 'crypto', 'to': 'stablecoins', 'direction': 'drain',
                'active': bool(total_stable_d and total_stable_d > 8),
                'strength': min(1.0, max(0, (total_stable_d or 0) - 6) / 8.0),
                'label': 'Capital fleeing to stablecoins (USDT+USDC)',
                'source_fn': 'usdt_sfp_wolfe()',
            },
            'fear_exit': {
                'from': 'stablecoins', 'to': 'dollar', 'direction': 'drain',
                'active': bool(fg is not None and fg < 15),
                'strength': min(1.0, max(0, 20 - (fg or 50)) / 20.0) if fg is not None else 0.0,
                'label': 'Extreme fear = capital exiting crypto entirely',
                'source_fn': 'sentiment_extreme',
            },
        }

        return {'nodes': nodes, 'edges': edges}

    # ------------------------------------------------------------------
    # Stress scenario checking
    # ------------------------------------------------------------------

    def _check_stress_scenarios(self, features) -> List[Dict]:
        """
        Check current macro features against stress scenario thresholds.

        Runs the full check every STRESS_CHECK_INTERVAL_HOURS to save CPU.
        Between checks, returns the cached result.
        """
        if self.stress_simulator is None:
            return []

        now = datetime.now(timezone.utc)

        # Throttle: only run full check every N hours
        if (
            self.last_stress_check_time is not None
            and (now - self.last_stress_check_time).total_seconds()
            < STRESS_CHECK_INTERVAL_HOURS * 3600
        ):
            return self.active_stress_scenarios

        # Convert features to dict if needed
        if hasattr(features, "to_dict"):
            feat_dict = features.to_dict()
        elif isinstance(features, dict):
            feat_dict = features
        else:
            feat_dict = dict(features)

        self.active_stress_scenarios = self.stress_simulator.check_current(feat_dict)
        self.last_stress_check_time = now

        # Log active scenarios
        active_count = sum(1 for s in self.active_stress_scenarios if s.get("active"))
        if active_count > 0:
            active_names = [
                s["name"] for s in self.active_stress_scenarios if s.get("active")
            ]
            logger.info(
                "[STRESS] %d active scenario(s): %s",
                active_count,
                ", ".join(active_names),
            )

        return self.active_stress_scenarios

    # ------------------------------------------------------------------
    # Signal narrative generation
    # ------------------------------------------------------------------

    def _generate_signal_narrative(
        self,
        signal: dict,
        features,
        cmi_breakdown: dict,
    ) -> dict:
        """
        Build a human-readable narrative explaining why a signal was generated.

        Combines information from the V11ShadowRunner's narrative (archetype-
        specific gate conditions, confluence factors) with CMI state and
        position details to produce a concise explanation suitable for display
        on the dashboard.

        Returns a dict with:
          - headline: one-line summary
          - text: 2-3 sentence narrative
          - archetype, direction, entry_price, timestamp
          - cmi: risk_temperature, instability, crisis_prob, dynamic_threshold
          - regime: current regime label
          - position: size_usd, leverage, stop_loss, take_profit
          - raw_narrative: the full narrative dict from _build_signal_narrative
        """
        archetype = signal.get("archetype", "unknown")
        direction = signal.get("direction", "long")
        entry_price = signal.get("entry_price", 0.0)
        fusion_score = signal.get("fusion_score", 0.0)
        timestamp = signal.get("timestamp", "")

        # Pull regime from features
        regime = "neutral"
        if hasattr(features, "get"):
            regime = str(features.get("regime_label", "neutral"))

        # CMI values
        risk_temp = self.runner.last_risk_temp
        instability = self.runner.last_instability
        crisis_prob = self.runner.last_crisis_prob
        threshold = self.runner.last_dynamic_threshold

        # Find the matching last_bar_signal entry which has the full narrative
        raw_narrative = None
        for lbs in self.runner.last_bar_signals:
            if (lbs.get("archetype") == archetype
                    and lbs.get("status") == "allocated"
                    and lbs.get("narrative")):
                raw_narrative = lbs["narrative"]
                break

        # Build headline — prefer archetype-specific headline from raw narrative
        price_str = f"${entry_price:,.0f}" if entry_price else "$?"
        if raw_narrative and raw_narrative.get("headline"):
            headline = raw_narrative["headline"]
        else:
            headline = (
                f"{archetype.replace('_', ' ').title()} "
                f"{direction.upper()} @ {price_str}"
            )

        # Extract confluence/risk from raw narrative for top-level access
        confluence_factors = (raw_narrative or {}).get("confluence_factors", [])
        risk_factors = (raw_narrative or {}).get("risk_factors", [])
        domain_scores = (raw_narrative or {}).get("domain_scores")

        # Build concise text narrative (2-3 sentences)
        parts = []

        # Sentence 1: Market context
        regime_desc = {
            "bull": "bullish",
            "neutral": "neutral",
            "bear": "bearish",
            "crisis": "crisis",
        }.get(regime, regime)
        fg_val = None
        if hasattr(features, "get"):
            fg_raw = features.get(
                "FEAR_GREED", features.get("fear_greed_norm", None)
            )
            if fg_raw is not None:
                try:
                    fg_val = float(fg_raw)
                    if fg_val != fg_val:
                        fg_val = None
                    elif fg_val <= 1.0:
                        fg_val = fg_val * 100
                except (TypeError, ValueError):
                    fg_val = None

        ema_state = ""
        if hasattr(features, "get"):
            p_above_50 = features.get("price_above_ema_50", 0)
            ema_50_200 = features.get("ema_50_above_200", 0)
            try:
                p_above_50 = int(float(p_above_50 or 0))
                ema_50_200 = int(float(ema_50_200 or 0))
            except (TypeError, ValueError):
                p_above_50, ema_50_200 = 0, 0
            if p_above_50 and ema_50_200:
                ema_state = "HTF trend aligned up"
            elif p_above_50:
                ema_state = "price above EMA50"
            elif ema_50_200:
                ema_state = "EMA50 above EMA200"
            else:
                ema_state = "below key EMAs"

        ctx_parts = [f"{regime_desc.title()} regime ({ema_state})"]
        if fg_val is not None:
            if fg_val < 25:
                ctx_parts.append(f"Extreme Fear (F&G={fg_val:.0f})")
            elif fg_val < 40:
                ctx_parts.append(f"Fear (F&G={fg_val:.0f})")
            elif fg_val > 75:
                ctx_parts.append(f"Greed (F&G={fg_val:.0f})")
            elif fg_val > 60:
                ctx_parts.append(f"Mild Greed (F&G={fg_val:.0f})")
        parts.append(", ".join(ctx_parts) + ".")

        # Sentence 2: Why this archetype fired (use raw narrative trigger)
        if raw_narrative and raw_narrative.get("trigger"):
            parts.append(raw_narrative["trigger"])
        elif raw_narrative and raw_narrative.get("summary"):
            first = raw_narrative["summary"].split(".")[0] + "."
            parts.append(first)

        # Sentence 3: Fusion vs threshold
        margin = fusion_score - threshold
        parts.append(
            f"Fusion {fusion_score:.3f} vs threshold {threshold:.3f} "
            f"({'+' if margin >= 0 else ''}{margin:.3f} margin, "
            f"risk_temp={risk_temp:.2f})."
        )

        text = " ".join(parts)

        # Find position details if this signal was just opened
        position_info = {}
        for pos in self.runner.positions.values():
            if pos.archetype == archetype and str(pos.entry_time) == timestamp:
                position_info = {
                    "size_usd": round(
                        pos.original_quantity * pos.entry_price, 2
                    ),
                    "leverage": getattr(self.runner, "leverage", 1.0),
                    "stop_loss": round(pos.stop_loss, 2),
                    "take_profit": round(pos.take_profit, 2),
                    "quantity_btc": round(pos.original_quantity, 8),
                }
                break

        return {
            "headline": headline,
            "text": text,
            "archetype": archetype,
            "direction": direction,
            "entry_price": entry_price,
            "timestamp": str(timestamp),
            "regime": regime,
            "fusion_score": round(fusion_score, 4),
            "threshold": round(threshold, 4),
            "confluence_factors": confluence_factors,
            "risk_factors": risk_factors,
            "domain_scores": domain_scores,
            "cmi": {
                "risk_temperature": round(risk_temp, 4),
                "instability": round(instability, 4),
                "crisis_prob": round(crisis_prob, 4),
                "dynamic_threshold": round(threshold, 4),
            },
            "position": position_info,
            "raw_narrative": raw_narrative,
        }

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _init_adapter(self):
        """Initialize the market data adapter (Coinbase preferred, Binance fallback)."""
        if not self.use_binance_fallback and COINBASE_AVAILABLE:
            try:
                adapter = CoinbaseAdapter()
                # Quick connectivity check
                logger.info("Testing Coinbase connectivity...")
                test_df = adapter.fetch_ohlcv_1h(limit=2)
                if not test_df.empty:
                    logger.info(
                        "Coinbase connected. Latest candle: %s  BTC=$%.0f",
                        test_df.index[-1],
                        test_df["close"].iloc[-1],
                    )
                    return adapter
                else:
                    logger.warning("Coinbase returned empty data. Checking fallback...")
            except Exception as exc:
                logger.warning("Coinbase connection failed: %s", exc)

        # Fallback to Binance
        if BINANCE_AVAILABLE:
            logger.info("Using Binance as data source (fallback).")
            self.adapter_source = "binance"
            adapter = BinanceAdapter()
            return adapter

        # Neither available
        logger.error(
            "No market data adapter available.\n"
            "  - Coinbase: coinbase_client.py not found or connection failed\n"
            "  - Binance:  binance_adapter.py not found or ccxt not installed\n"
            "\n"
            "To fix:\n"
            "  1. Ensure bin/live/coinbase_client.py exists (CoinbaseAdapter class)\n"
            "  2. Or install ccxt for Binance fallback: pip install ccxt\n"
            "  3. Or use --fallback-binance flag with ccxt installed\n"
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Warmup
    # ------------------------------------------------------------------

    def warmup(self):
        """Fetch 500 historical 1H candles from the data source and seed the feature computer."""
        logger.info(
            "Warming up with %d historical candles from %s...",
            WARMUP_CANDLES,
            self.adapter_source,
        )
        t0 = time.time()

        try:
            candles = self.adapter.fetch_ohlcv_1h(limit=WARMUP_CANDLES)
        except Exception as exc:
            logger.error("Failed to fetch warmup candles: %s", exc)
            logger.error("Cannot proceed without historical data for indicator warmup.")
            sys.exit(1)

        if candles.empty:
            logger.error("Warmup returned 0 candles. Check network and adapter.")
            sys.exit(1)

        if len(candles) < 200:
            logger.warning(
                "Only %d warmup candles fetched (need >= 200 for SMA200). "
                "Indicators may be unreliable.",
                len(candles),
            )

        self.feature_computer.ingest_candles(candles)

        # Backfill feature_history from warmup candles + historical macro
        # data so correlation and cointegration are ready immediately.
        # yfinance provides 90 days of VIX/DXY/Gold/Oil z-scores,
        # Alternative.me provides 8 days of F&G history.
        # BTC.D/USDT.D are real-time only (CoinGecko /global endpoint).
        macro_daily = None
        try:
            macro_daily = self.feature_computer._macro_fetcher.get_historical_daily()
        except Exception as exc:
            logger.warning("Failed to fetch historical macro data: %s", exc)

        # Current dominance values (forward-fill for all warmup bars)
        current_btc_d = float("nan")
        current_usdt_d = float("nan")
        current_usdc_d = float("nan")
        if macro_daily is not None:
            dom = self.feature_computer._macro_fetcher._dominance_cache
            if dom.get('BTC.D') is not None:
                current_btc_d = dom['BTC.D']
            if dom.get('USDT.D') is not None:
                current_usdt_d = dom['USDT.D']
            if dom.get('USDC.D') is not None:
                current_usdc_d = dom['USDC.D']

        warmup_tail = candles.tail(FEATURE_HISTORY_SIZE)
        n_macro_matched = 0
        for ts, row in warmup_tail.iterrows():
            close_val = float(row.get("close", row.get("Close", 0)))
            if close_val <= 0:
                continue

            snapshot = {
                "close": close_val,
                "dxy_z": float("nan"),
                "vix_z": float("nan"),
                "gold_z": float("nan"),
                "oil_z": float("nan"),
                "fear_greed_norm": float("nan"),
                "btc_d": current_btc_d,
                "usdt_d": current_usdt_d,
                "usdc_d": current_usdc_d,
            }

            # Look up macro values for this timestamp's date
            if macro_daily is not None and len(macro_daily) > 0:
                candle_date = pd.Timestamp(ts)
                if candle_date.tz is not None:
                    candle_date = candle_date.tz_localize(None)
                # Find the most recent daily macro row <= this candle's date
                mask = macro_daily.index <= candle_date
                if mask.any():
                    macro_row = macro_daily.loc[mask].iloc[-1]
                    matched = False
                    for col, snap_key in [
                        ('VIX_Z', 'vix_z'), ('DXY_Z', 'dxy_z'),
                        ('GOLD_Z', 'gold_z'), ('OIL_Z', 'oil_z'),
                        ('fear_greed_norm', 'fear_greed_norm'),
                    ]:
                        if col in macro_row.index:
                            val = macro_row[col]
                            if val == val:  # not NaN
                                snapshot[snap_key] = float(val)
                                matched = True
                    if 'YIELD_CURVE' in macro_row.index:
                        yc = macro_row['YIELD_CURVE']
                        if yc == yc:
                            snapshot['yield_curve'] = float(yc)
                    if matched:
                        n_macro_matched += 1

            self.feature_history.append(snapshot)

        logger.info(
            "Backfilled feature_history: %d bars (%d with macro data). "
            "Sources: yfinance (VIX/DXY/Gold/Oil z-scores), "
            "alternative.me (F&G 8d), CoinGecko (BTC.D/USDT.D current)",
            len(self.feature_history),
            n_macro_matched,
        )

        elapsed = time.time() - t0
        logger.info(
            "Warmup complete: %d candles ingested in %.1fs  [%s .. %s]",
            len(candles),
            elapsed,
            candles.index[0],
            candles.index[-1],
        )

    # ------------------------------------------------------------------
    # State persistence (funding costs)
    # ------------------------------------------------------------------

    def _funding_costs_path(self) -> Path:
        return self.output_dir / "funding_costs.json"

    def _load_funding_costs(self):
        """Load accumulated funding costs from disk."""
        path = self._funding_costs_path()
        if path.exists():
            try:
                with open(path) as f:
                    saved = json.load(f)
                self.funding_costs.update(saved)
                logger.info(
                    "Loaded funding costs: $%.2f total over %d events",
                    self.funding_costs["total_funding_cost_usd"],
                    self.funding_costs["total_funding_events"],
                )
            except (json.JSONDecodeError, KeyError) as exc:
                logger.warning("Could not load funding costs: %s", exc)

    def _save_funding_costs(self):
        """Save accumulated funding costs to disk."""
        path = self._funding_costs_path()
        try:
            with open(path, "w") as f:
                json.dump(self.funding_costs, f, indent=2, default=str)
        except Exception as exc:
            logger.warning("Failed to save funding costs: %s", exc)

    def _save_wyckoff_state(self):
        """Persist Wyckoff phase tracking to disk."""
        import json
        state = {
            'phase_history': self.wyckoff_phase_history[-50:],
            'cycle_start': self.wyckoff_cycle_start,
            'last_phase': self.wyckoff_last_phase,
        }
        path = self.output_dir / 'wyckoff_state.json'
        try:
            with open(path, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as exc:
            logger.warning("Failed to save wyckoff state: %s", exc)

    def _load_wyckoff_state(self):
        """Load Wyckoff phase tracking from disk."""
        import json
        path = self.output_dir / 'wyckoff_state.json'
        if not path.exists():
            return
        try:
            with open(path) as f:
                state = json.load(f)
            self.wyckoff_phase_history = state.get('phase_history', [])
            self.wyckoff_cycle_start = state.get('cycle_start')
            self.wyckoff_last_phase = state.get('last_phase', 'neutral')
            logger.info("Loaded Wyckoff state: %d phase transitions, cycle_start=%s",
                        len(self.wyckoff_phase_history), self.wyckoff_cycle_start)
        except Exception as exc:
            logger.warning("Failed to load wyckoff state: %s", exc)

    def _track_wyckoff_phase_transition(self, features, timestamp, btc_price):
        """Track phase transitions and cycle starts."""
        current_phase = str(features.get('wyckoff_phase_abc', 'neutral'))
        if current_phase != self.wyckoff_last_phase:
            transition = {
                'from_phase': self.wyckoff_last_phase,
                'to_phase': current_phase,
                'timestamp': str(timestamp),
                'price': round(float(btc_price), 2),
            }
            self.wyckoff_phase_history.append(transition)
            # New cycle: transition from neutral to any phase
            if self.wyckoff_last_phase == 'neutral' and current_phase != 'neutral':
                self.wyckoff_cycle_start = str(timestamp)
            self.wyckoff_last_phase = current_phase
            logger.info("Wyckoff phase: %s -> %s at $%.0f", transition['from_phase'], current_phase, btc_price)

    def _build_event_narratives(self, features) -> dict:
        """For each active event, generate structural narrative describing what the engine saw."""
        TEMPLATES = {
            'sc': "Selling Climax at ${price}: volume spiked to {vz:.1f} sigma with price at range lows. The {lwp:.0f}% lower wick shows institutional absorption of panic selling.",
            'bc': "Buying Climax at ${price}: volume spiked to {vz:.1f} sigma with price at range highs. The {uwp:.0f}% upper wick shows distribution into euphoric buying.",
            'ar': "Automatic Rally from ${price}: declining volume ({vz:.1f} sigma) while price rebounds. Close in upper {cp:.0f}% of bar — buying pressure emerging.",
            'as': "Automatic Reaction from ${price}: declining volume ({vz:.1f} sigma) as price drops. Close at {cp:.0f}% of range confirms selling.",
            'st': "Secondary Test near ${price}: volume much lower ({vz:.1f} sigma) on this retest of support. Sellers are exhausted — no new lows made.",
            'sos': "Sign of Strength at ${price}: volume surged to {vz:.1f} sigma as price broke above range with strong {cp:.0f}% close.",
            'sow': "Sign of Weakness at ${price}: volume surged to {vz:.1f} sigma as price broke below range with weak {cp:.0f}% close.",
            'spring_a': "Deep Spring at ${price}: price broke below support with a {lwp:.0f}% lower wick rejection on {vz:.1f} sigma volume. Classic bear trap.",
            'spring_b': "Shallow Spring at ${price}: mild dip below range with {lwp:.0f}% wick and quick recovery. Close at {cp:.0f}% shows demand.",
            'lps': "Last Point of Support at ${price}: pullback on very low volume ({vz:.1f} sigma) with strong {cp:.0f}% close. Final test before markup.",
            'lpsy': "Last Point of Supply at ${price}: weak rally on low volume ({vz:.1f} sigma), close at only {cp:.0f}%. Final exit before markdown.",
            'ut': "Upthrust at ${price}: high broke above resistance but closed back inside range. Upper wick rejection on {vz:.1f} sigma volume.",
            'utad': "Upthrust After Distribution at ${price}: final breakout trap with extreme momentum. Most dangerous bull trap before markdown.",
        }

        narratives = {}
        close_price = float(features.get('close', 0) or 0)
        high = float(features.get('high', 0) or 0)
        low = float(features.get('low', 0) or 0)
        open_price = float(features.get('open', close_price) or close_price)
        bar_range = high - low if high > low else 1e-10
        # Compute context from OHLCV (same as _build_wyckoff_context)
        vz_raw = features.get('volume_z', features.get('volume_z_20', 0))
        vz = float(vz_raw) if vz_raw == vz_raw and vz_raw is not None else 0.0
        cp = ((close_price - low) / bar_range * 100) if bar_range > 0 else 50.0
        lwp = ((min(open_price, close_price) - low) / bar_range * 100) if bar_range > 0 else 0.0
        uwp = ((high - max(open_price, close_price)) / bar_range * 100) if bar_range > 0 else 0.0

        for event_key, template in TEMPLATES.items():
            if not features.get(f'wyckoff_{event_key}', False):
                continue
            try:
                narratives[event_key] = template.format(
                    price=f"{close_price:,.0f}",
                    vz=vz, cp=cp, lwp=lwp, uwp=uwp,
                )
            except Exception:
                narratives[event_key] = f"{event_key.upper()} active at ${close_price:,.0f}"
        return narratives

    def _save_trades(self):
        """Save completed trades to trades.json for dashboard."""
        trades_data = []
        for t in self.runner.trades:
            # Compute position_size_usd from quantity * entry_price
            pos_size = t.quantity * t.entry_price if t.quantity and t.entry_price else 0.0
            trade_dict = {
                "timestamp_entry": str(t.timestamp_entry),
                "timestamp_exit": str(t.timestamp_exit),
                "archetype": t.archetype,
                "direction": t.direction,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "quantity": t.quantity,
                "pnl": round(t.pnl, 2),
                "pnl_usd": round(t.pnl, 2),  # Alias for dashboard compatibility
                "pnl_pct": round(t.pnl_pct, 4),
                "entry_regime": t.entry_regime,
                "duration_hours": round(t.duration_hours, 1),
                "fusion_score": round(t.fusion_score, 4),
                "exit_reason": t.exit_reason,
                "factor_attribution": t.factor_attribution,
                "stop_loss": round(getattr(t, 'stop_loss', 0.0), 2),
                "take_profit": round(getattr(t, 'take_profit', 0.0), 2),
                "atr_at_entry": round(getattr(t, 'atr_at_entry', 0.0), 4),
                "threshold_at_entry": round(getattr(t, 'threshold_at_entry', 0.0), 4),
                "threshold_margin": round(getattr(t, 'threshold_margin', 0.0), 4),
                "risk_temp_at_entry": round(getattr(t, 'risk_temp_at_entry', 0.0), 4),
                "instability_at_entry": round(getattr(t, 'instability_at_entry', 0.0), 4),
                "crisis_prob_at_entry": round(getattr(t, 'crisis_prob_at_entry', 0.0), 4),
                "leverage_applied": round(getattr(t, 'leverage_applied', 1.0), 2),
                "position_size_usd": round(pos_size, 2),
            }
            trades_data.append(trade_dict)
        path = self.output_dir / "trades.json"
        try:
            with open(path, "w") as f:
                json.dump(trades_data, f, indent=2)
        except Exception as exc:
            logger.warning("Failed to save trades.json: %s", exc)

        # Save phantom trades
        try:
            phantom_summary = self.runner.get_phantom_summary()
            phantom_path = self.output_dir / "phantom_trades.json"
            with open(phantom_path, "w") as f:
                json.dump(phantom_summary, f, indent=2)
        except Exception as exc:
            logger.warning("Failed to save phantom_trades.json: %s", exc)

    def _compute_trade_counterfactual(self, trade):
        """Compute counterfactual for the most recently completed trade."""
        try:
            from engine.analysis.counterfactual import CounterfactualEngine

            if not hasattr(self, '_cf_engine'):
                # Build a minimal DataFrame from recent candles for simulation
                if hasattr(self, 'candle_buffer') and len(self.candle_buffer) > 50:
                    df = pd.DataFrame(self.candle_buffer)
                    if 'timestamp' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        df = df.set_index('timestamp').sort_index()
                    self._cf_engine = CounterfactualEngine(
                        df,
                        commission_rate=0.0002,
                        slippage_bps=3.0,
                    )
                else:
                    return None

            result = self._cf_engine.analyze_trade(trade)
            return result.to_dict()
        except Exception as e:
            logger.warning(f"Counterfactual analysis failed: {e}")
            return None

    def _save_performance_summary(self):
        """Save running performance stats to disk."""
        path = self.output_dir / "performance_summary.json"
        equity = (
            self.runner.equity_curve[-1]
            if self.runner.equity_curve
            else self.initial_cash
        )

        wins = [t for t in self.runner.trades if t.pnl > 0]
        losses = [t for t in self.runner.trades if t.pnl <= 0]
        total_pnl = sum(t.pnl for t in self.runner.trades) if self.runner.trades else 0.0
        gross_profit = sum(t.pnl for t in wins) if wins else 0.0
        gross_loss = abs(sum(t.pnl for t in losses)) if losses else 0.0
        pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        win_rate = len(wins) / len(self.runner.trades) * 100 if self.runner.trades else 0.0

        max_dd = 0.0
        if self.runner.equity_curve:
            peak = self.runner.equity_curve[0]
            for eq in self.runner.equity_curve:
                if eq > peak:
                    peak = eq
                dd = (peak - eq) / peak if peak > 0 else 0.0
                if dd > max_dd:
                    max_dd = dd

        summary = {
            "session_start": self.session_start.isoformat(),
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "adapter_source": self.adapter_source,
            "bars_processed": self.bars_processed,
            "initial_cash": self.initial_cash,
            "current_equity": equity,
            "total_return_pct": (equity - self.initial_cash) / self.initial_cash * 100,
            "total_pnl": total_pnl,
            "completed_trades": len(self.runner.trades),
            "open_positions": len(self.runner.positions),
            "win_rate_pct": win_rate,
            "profit_factor": pf if pf != float("inf") else 999.99,
            "max_drawdown_pct": max_dd * 100,
            "total_signals_generated": self.runner.total_signals,
            "signals_allocated": self.runner.signals_allocated,
            "signals_rejected": self.runner.signals_rejected,
            "total_funding_cost_usd": self.funding_costs["total_funding_cost_usd"],
            "net_pnl_after_funding": total_pnl - self.funding_costs["total_funding_cost_usd"],
        }

        try:
            with open(path, "w") as f:
                json.dump(summary, f, indent=2)
        except Exception as exc:
            logger.warning("Failed to save performance summary: %s", exc)

    # ------------------------------------------------------------------
    # Rolling macro correlations
    # ------------------------------------------------------------------

    def _append_feature_snapshot(self, features):
        """Store macro feature values for rolling correlation computation."""
        def _get(key, *alt_keys):
            val = features.get(key)
            for ak in alt_keys:
                if val is None or (isinstance(val, float) and val != val):
                    val = features.get(ak)
            if val is None:
                return float("nan")
            try:
                f = float(val)
                return f if f == f else float("nan")
            except (TypeError, ValueError):
                return float("nan")

        snapshot = {
            "close": _get("close"),
            "dxy_z": _get("DXY_Z", "dxy_z"),
            "vix_z": _get("VIX_Z", "vix_z"),
            "gold_z": _get("GOLD_Z", "gold_z"),
            "oil_z": _get("OIL_Z", "oil_z"),
            "fear_greed_norm": _get("fear_greed_norm", "FEAR_GREED"),
            "btc_d": _get("BTC.D", "btc_dominance"),
            "usdt_d": _get("USDT.D", "usdt_dominance"),
            "usdc_d": _get("USDC.D", "usdc_dominance"),
        }
        self.feature_history.append(snapshot)
        # Keep last FEATURE_HISTORY_SIZE entries (168 = 1 week for cointegration)
        if len(self.feature_history) > FEATURE_HISTORY_SIZE:
            self.feature_history = self.feature_history[-FEATURE_HISTORY_SIZE:]

    def _compute_macro_correlations(self) -> Dict:
        """
        Compute rolling 20-bar and 60-bar correlations between macro indicators
        and BTC returns. Returns dict suitable for heartbeat.json.
        """
        history = self.feature_history
        if len(history) < 5:
            return {"window_20": {}, "window_60": {}, "regime": "insufficient_data",
                    "n_bars": len(history), "min_bars_20": 21, "min_bars_60": 61}

        pairs = [
            ("dxy_z", "dxy_btc"),
            ("vix_z", "vix_btc"),
            ("gold_z", "gold_btc"),
            ("oil_z", "oil_btc"),
            ("fear_greed_norm", "fg_btc"),
            ("btc_d", "btc_d_btc"),
            ("usdt_d", "usdt_d_btc"),
            ("usdc_d", "usdc_d_btc"),
        ]

        def _rolling_corr(window: int) -> Dict[str, float]:
            if len(history) < window + 1:
                return {label: None for _, label in pairs}

            recent = history[-window:]
            closes = [h["close"] for h in recent]
            # Compute BTC returns (log returns)
            btc_returns = []
            for i in range(1, len(closes)):
                prev_c = closes[i - 1]
                cur_c = closes[i]
                if prev_c > 0 and prev_c == prev_c and cur_c == cur_c:
                    btc_returns.append(cur_c / prev_c - 1.0)
                else:
                    btc_returns.append(float("nan"))

            result = {}
            for feat_key, label in pairs:
                # Use values aligned with returns (skip first)
                feat_vals = [recent[i][feat_key] for i in range(1, len(recent))]

                # Filter out NaN pairs
                valid_x = []
                valid_y = []
                for x, y in zip(feat_vals, btc_returns):
                    if x == x and y == y:  # NaN check
                        valid_x.append(x)
                        valid_y.append(y)

                if len(valid_x) < 5:
                    result[label] = None
                    continue

                x_arr = np.array(valid_x)
                y_arr = np.array(valid_y)

                # Check for zero variance
                if np.std(x_arr) < 1e-10 or np.std(y_arr) < 1e-10:
                    result[label] = 0.0
                    continue

                corr = np.corrcoef(x_arr, y_arr)[0, 1]
                if corr != corr:  # NaN
                    result[label] = None
                else:
                    # USDT.D should be inverted (rising USDT.D = bearish)
                    # but we report raw correlation; the UI can flag it
                    result[label] = round(float(corr), 3)

            return result

        w20 = _rolling_corr(20)
        w60 = _rolling_corr(60)

        # Determine regime: "stressed" if avg absolute correlation > 0.6
        abs_corrs = [abs(v) for v in w20.values() if v is not None]
        avg_abs = sum(abs_corrs) / len(abs_corrs) if abs_corrs else 0.0

        # Count bars with macro data (non-NaN macro fields)
        n_macro_bars = sum(
            1 for h in history
            if h.get("dxy_z", float("nan")) == h.get("dxy_z", float("nan"))
            and h.get("dxy_z") is not None
            and not (isinstance(h.get("dxy_z"), float) and h["dxy_z"] != h["dxy_z"])
        )

        # If we have close data from warmup but no macro data yet
        if n_macro_bars < 5 and len(history) >= 5:
            regime = "awaiting_macro"
        elif avg_abs > 0.6:
            regime = "stressed"
        else:
            regime = "normal"

        return {
            "window_20": w20,
            "window_60": w60,
            "avg_abs_corr_20": round(avg_abs, 3),
            "regime": regime,
            "n_bars": len(history),
            "n_macro_bars": n_macro_bars,
            "min_bars_20": 21,
            "min_bars_60": 61,
        }

    # ------------------------------------------------------------------
    # Cointegration analysis (runs every 4 hours)
    # ------------------------------------------------------------------

    def _get_cointegration_results(self) -> Optional[Dict]:
        """
        Return cointegration analysis results.
        Recomputes every COINTEGRATION_INTERVAL_HOURS to save CPU.
        Between runs, returns the cached result.
        """
        if not COINTEGRATION_AVAILABLE:
            return None

        now = datetime.now(timezone.utc)

        # Check if we need to recompute
        should_recompute = (
            self.last_cointegration_time is None
            or (now - self.last_cointegration_time).total_seconds()
            >= COINTEGRATION_INTERVAL_HOURS * 3600
        )

        if should_recompute:
            try:
                result = compute_cointegration(self.feature_history)
                self.last_cointegration_result = result
                self.last_cointegration_time = now
                n_opp = sum(
                    1 for p in result.get("pairs", [])
                    if p.get("has_opportunity", False)
                )
                logger.info(
                    "[COINTEGRATION] Analysis complete: %d pairs, %d opportunities, %d bars used",
                    len(result.get("pairs", [])),
                    n_opp,
                    result.get("n_bars_used", 0),
                )
            except Exception as exc:
                logger.warning("Cointegration computation failed: %s", exc)
                # Keep stale result if available
                if self.last_cointegration_result is None:
                    return None

        return self.last_cointegration_result

    def _compute_factor_attribution_summary(self) -> dict:
        """
        Compute aggregate factor attribution across all completed trades.

        Returns summary with avg_pct plus winners vs losers breakdown
        for each factor (technical, liquidity, macro, regime).
        """
        trades = self.runner.trades
        if not trades:
            return {}

        factors = ['technical', 'liquidity', 'macro', 'regime']
        all_vals = {f: [] for f in factors}
        win_vals = {f: [] for f in factors}
        loss_vals = {f: [] for f in factors}

        for t in trades:
            attr = t.factor_attribution
            if not attr:
                continue
            for f in factors:
                val = attr.get(f, 0.0)
                all_vals[f].append(val)
                if t.pnl > 0:
                    win_vals[f].append(val)
                else:
                    loss_vals[f].append(val)

        def _avg(lst):
            return round(sum(lst) / len(lst), 4) if lst else 0.0

        summary = {}
        for f in factors:
            summary[f] = {
                'avg_pct': _avg(all_vals[f]),
                'win_contribution': _avg(win_vals[f]),
                'loss_contribution': _avg(loss_vals[f]),
            }

        return summary

    # ------------------------------------------------------------------
    # Oracle Synthesis — master narrative from all data sources
    # ------------------------------------------------------------------

    @staticmethod
    def _whale_macro_summary(whale: dict, whale_has_data: bool) -> dict:
        """Summarize whale intelligence for the macro_summary section."""
        if not whale_has_data:
            return {"state": "Warming Up", "impact": "No data yet", "detail": "Needs ~10 bars after restart"}
        sentiment = whale.get("sentiment", "neutral")
        conflict_count = (whale.get("conflict") or {}).get("count", 0)
        oi_4h = (whale.get("raw") or {}).get("oi_change_4h", 0) or 0
        if conflict_count >= 3:
            return {"state": "Conflicted", "impact": "Strong institutional headwind", "detail": f"{conflict_count}/4 whale conflict signals active"}
        if conflict_count >= 2:
            return {"state": "Cautious", "impact": "Moderate whale conflict", "detail": f"{conflict_count}/4 conflict signals, OI {oi_4h*100:+.1f}% 4h"}
        if sentiment in ("bullish", "strongly_bullish"):
            return {"state": "Accumulating", "impact": "Institutional tailwind", "detail": f"OI rising ({oi_4h*100:+.1f}% 4h), {sentiment}"}
        if sentiment in ("bearish", "strongly_bearish"):
            return {"state": "Distributing", "impact": "Institutional headwind", "detail": f"OI declining ({oi_4h*100:+.1f}% 4h), {sentiment}"}
        return {"state": "Neutral", "impact": "No strong institutional signal", "detail": f"OI {oi_4h*100:+.1f}% 4h, balanced flows"}

    def _build_synthesis_paragraph(self, **kw) -> str:
        """
        Build ONE cohesive paragraph that synthesizes all intelligence:
        Wyckoff structure, whale/institutional flows, macro environment,
        capital flows, and engine posture — weighted by reliability.

        Priority order (what matters most to least):
        1. Market structure (Wyckoff phase) — the skeleton of price
        2. Institutional positioning (whale data) — who's betting what
        3. Macro environment (dollar, vol, sentiment) — the backdrop
        4. Capital flows — where money is moving
        5. Engine state — what our system is doing about it
        """
        btc_price = kw['btc_price']
        bias = kw['bias']
        posture = kw['posture']
        posture_action = kw['posture_action']
        phase = kw['phase']
        risk_temp = kw['risk_temp']
        crisis_prob = kw['crisis_prob']
        threshold = kw['threshold']
        bullish_signals = kw['bullish_signals']
        bearish_signals = kw['bearish_signals']
        total_sources = kw['total_sources']
        whale_has_data = kw['whale_has_data']
        whale_raw = kw['whale_raw']
        whale_derived = kw['whale_derived']
        whale_conflict = kw['whale_conflict']
        whale = kw['whale']
        macro = kw['macro']
        dollar_summary = kw['dollar_summary']
        vol_summary = kw['vol_summary']
        sent_summary = kw['sent_summary']
        flows_summary = kw['flows_summary']
        wyckoff = kw['wyckoff']
        active_stress_count = kw['active_stress_count']

        parts = []

        # --- 1. Market Structure (Wyckoff) — highest weight ---
        phase_desc = {
            "A": "stopping action from the prior trend — the first signs of a potential reversal",
            "B": "building cause within a trading range as supply and demand contest control",
            "C": "a critical spring or upthrust test at the range boundary — the final shakeout zone",
            "D": "a confirmed sign of strength or weakness emerging from the range",
            "E": "active trend continuation in the markup or markdown phase",
            "accumulation": "accumulation by smart money at current levels",
            "markup": "an active markup phase with demand in control",
            "distribution": "distribution by smart money — caution warranted",
            "markdown": "markdown with supply pressure dominating",
        }
        structure_text = phase_desc.get(phase)
        if structure_text:
            parts.append(f"At ${btc_price:,.0f}, Wyckoff structure shows {structure_text}.")
        else:
            parts.append(f"BTC is at ${btc_price:,.0f} with no clear Wyckoff phase.")

        # --- 2. Institutional Positioning (whale data) — second highest weight ---
        if whale_has_data:
            oi_4h = whale_raw.get("oi_change_4h", 0) or 0
            oi_24h = whale_raw.get("oi_change_24h", 0) or 0
            taker = whale_raw.get("taker_imbalance", 0) or 0
            funding_r = whale_raw.get("funding_rate", 0) or 0
            ls = whale_raw.get("ls_ratio_extreme", 0) or 0
            conflict_count = whale_conflict.get("count", 0)
            whale_sent = whale.get("sentiment", "neutral")

            # OI narrative
            if abs(oi_4h) > 0.02 or abs(oi_24h) > 0.03:
                if oi_4h > 0.02:
                    oi_desc = f"OI is rising ({oi_4h*100:+.1f}% over 4h), indicating new positions opening"
                elif oi_4h < -0.03:
                    oi_desc = f"OI is declining ({oi_4h*100:+.1f}% over 4h), signaling institutional unwinding"
                elif oi_24h > 0.05:
                    oi_desc = f"OI has built significantly over 24h ({oi_24h*100:+.1f}%), showing sustained position building"
                elif oi_24h < -0.05:
                    oi_desc = f"OI has dropped over 24h ({oi_24h*100:+.1f}%), indicating broad deleveraging"
                else:
                    oi_desc = f"OI is shifting ({oi_4h*100:+.1f}% 4h, {oi_24h*100:+.1f}% 24h)"
            else:
                oi_desc = "OI is stable"

            # Taker + funding color
            flow_parts = []
            if abs(taker) > 0.2:
                flow_parts.append("aggressive buying" if taker > 0 else "aggressive selling")
            if abs(funding_r) > 0.0003:
                flow_parts.append("longs paying elevated funding" if funding_r > 0 else "shorts paying — bearish crowding")

            if flow_parts:
                whale_text = f"{oi_desc}, with {' and '.join(flow_parts)}."
            else:
                whale_text = f"{oi_desc} with balanced taker flow and neutral funding."

            # Conflict warning
            if conflict_count >= 3:
                whale_text += f" Multiple whale conflicts detected ({conflict_count}/4) — institutional data strongly conflicts with long positioning."
            elif conflict_count >= 2:
                whale_text += f" Whale conflict warning ({conflict_count}/4 signals) suggests caution."

            parts.append(whale_text)
        else:
            parts.append("Institutional flow data is warming up (needs ~10 bars after restart).")

        # --- 3. Macro Environment — third weight ---
        fg = macro.get("fear_greed")
        dxy_state = dollar_summary["state"].lower()
        vol_state = vol_summary["state"].lower()

        macro_pieces = []
        if fg is not None:
            if fg <= 20:
                macro_pieces.append(f"extreme fear (F&G {fg:.0f})")
            elif fg <= 35:
                macro_pieces.append(f"fear (F&G {fg:.0f})")
            elif fg >= 80:
                macro_pieces.append(f"extreme greed (F&G {fg:.0f})")
            elif fg >= 65:
                macro_pieces.append(f"greed (F&G {fg:.0f})")

        if dxy_state not in ("neutral", "unknown"):
            macro_pieces.append(f"a {dxy_state} dollar")
        if vol_state not in ("neutral", "unknown"):
            macro_pieces.append(f"{vol_state} volatility")

        flows_state = flows_summary["state"].lower()
        if flows_state not in ("balanced", "unknown"):
            macro_pieces.append(f"capital {flows_state.lower()}")

        if macro_pieces:
            macro_env = "favorable" if bias == "bullish" else "challenging" if bias == "bearish" else "mixed"
            parts.append(f"The macro backdrop is {macro_env}: {', '.join(macro_pieces)}.")

        # --- 4. Stress / Crisis ---
        if crisis_prob > 0.5:
            parts.append(f"Crisis probability is elevated at {crisis_prob:.0%}, triggering emergency defensive measures.")
        elif active_stress_count >= 2:
            parts.append(f"{active_stress_count} stress scenarios are active, adding tail risk.")

        # --- 5. Engine State + Net Assessment ---
        regime_word = "bear" if risk_temp < 0.35 else "bull" if risk_temp > 0.65 else "neutral"
        parts.append(
            f"With {bullish_signals} bullish vs {bearish_signals} bearish signals across {total_sources} sources "
            f"in a {regime_word} regime (threshold {threshold:.2f}), "
            f"the engine is {posture_action}."
        )

        return " ".join(parts) if parts else "Insufficient data to form a market thesis."

    def _build_oracle_synthesis(self, heartbeat_data: dict) -> dict:
        """
        Synthesize all heartbeat data sources into a single coherent market
        narrative (the "Master Oracle"). Reads macro outlook, stress scenarios,
        cointegration, capital flows, CMI, Wyckoff, and recent trades to
        produce posture, bias, thesis, risks, catalysts, and macro summaries.
        """

        # ---- Safe accessors ----
        def _g(d, *keys, default=None):
            """Nested safe .get() with NaN guard."""
            for k in keys:
                if d is None or not isinstance(d, dict):
                    return default
                d = d.get(k)
            if d is None:
                return default
            if isinstance(d, float) and d != d:  # NaN
                return default
            return d

        macro = heartbeat_data.get("macro") or {}
        outlook = heartbeat_data.get("macro_outlook") or {}
        stress = heartbeat_data.get("active_stress_scenarios") or []
        coint = heartbeat_data.get("cointegration") or {}
        flows = heartbeat_data.get("capital_flows") or {}
        wyckoff = heartbeat_data.get("wyckoff") or {}
        whale = heartbeat_data.get("whale_intelligence") or {}
        risk_temp = heartbeat_data.get("risk_temp", 0.5)
        instability = heartbeat_data.get("instability", 0.5)
        crisis_prob = heartbeat_data.get("crisis_prob", 0.0)
        threshold = heartbeat_data.get("threshold", 0.4)
        btc_price = heartbeat_data.get("btc_price", 0.0)

        # ---- 1. Bias Detection ----
        bullish_signals = 0
        bearish_signals = 0
        total_sources = 0

        # Macro outlook timeframes
        for tf in ["1w", "1m", "6m", "1y"]:
            score = _g(outlook, tf, "score", default=None)
            if score is not None:
                total_sources += 1
                if score > 0.05:
                    bullish_signals += 1
                elif score < -0.05:
                    bearish_signals += 1

        # Wyckoff directional scores (1H, 4H, 1D)
        for prefix, bull_key, bear_key in [
            ("1H", "bullish_1h", "bearish_1h"),
            ("4H", "bullish_4h", "bearish_4h"),
            ("1D", "bullish_1d", "bearish_1d"),
        ]:
            bull_s = wyckoff.get(bull_key)
            bear_s = wyckoff.get(bear_key)
            if bull_s is not None and bear_s is not None:
                total_sources += 1
                if bull_s > bear_s + 0.1:
                    bullish_signals += 1
                elif bear_s > bull_s + 0.1:
                    bearish_signals += 1

        # Capital flows (count active bullish vs bearish edges)
        edges = flows.get("edges") or {}
        for edge_name, edge in edges.items():
            if not edge.get("active"):
                continue
            total_sources += 1
            direction = edge.get("direction", "")
            # "flow" to crypto = bullish, "drain" from crypto = bearish
            to_node = edge.get("to", "")
            from_node = edge.get("from", "")
            if direction == "flow" and to_node == "crypto":
                bullish_signals += 1
            elif direction == "drain" and (from_node == "crypto" or to_node != "crypto"):
                bearish_signals += 1
            elif direction == "flow" and from_node == "crypto":
                bearish_signals += 1

        # Whale / institutional intelligence
        whale_raw = whale.get("raw") or {}
        whale_derived = whale.get("derived") or {}
        whale_conflict = whale.get("conflict") or {}
        whale_has_data = whale.get("has_data", False)
        if whale_has_data:
            total_sources += 1
            whale_sentiment = whale.get("sentiment", "neutral")
            if whale_sentiment in ("bullish", "strongly_bullish"):
                bullish_signals += 1
            elif whale_sentiment in ("bearish", "strongly_bearish"):
                bearish_signals += 1

        # CMI risk_temperature
        if risk_temp is not None:
            total_sources += 1
            if risk_temp > 0.55:
                bullish_signals += 1
            elif risk_temp < 0.45:
                bearish_signals += 1

        # ---- 2. Confidence ----
        n_aligned = max(bullish_signals, bearish_signals)
        n_total = max(total_sources, 1)
        active_stress_count = sum(1 for s in stress if s.get("active"))
        confidence = max(0.0, min(1.0, n_aligned / n_total - active_stress_count * 0.1))

        # ---- 3. Bias ----
        if bullish_signals > bearish_signals + 1:
            bias = "bullish"
        elif bearish_signals > bullish_signals + 1:
            bias = "bearish"
        else:
            bias = "neutral"

        bias_strength = abs(bullish_signals - bearish_signals) / max(n_total, 1)
        bias_strength = min(1.0, bias_strength)

        # ---- 4. Posture ----
        if crisis_prob > 0.5 or active_stress_count >= 3:
            posture = "CRISIS"
        elif bias == "bearish" or crisis_prob > 0.3:
            posture = "DEFENSIVE"
        elif bias == "neutral" or confidence < 0.4:
            posture = "CAUTIOUS"
        elif bias == "bullish" and confidence > 0.6 and active_stress_count == 0:
            posture = "RISK_ON"
        else:
            posture = "CAUTIOUS"

        # ---- 5. Macro Summary ----
        dxy_z = macro.get("dxy_z")
        vix_z = macro.get("vix_z")
        gold_z = macro.get("gold_z")
        oil_z = macro.get("oil_z")
        yc = macro.get("yield_curve")
        fg = macro.get("fear_greed")

        def _classify_z(z, name_strong, name_weak):
            if z is None:
                return {"state": "Unknown", "impact": "No data", "detail": f"{name_strong} data unavailable"}
            if z > 1.5:
                return {"state": "Elevated", "impact": name_strong, "detail": f"Z-score {z:+.1f} (above 1.5 sigma)"}
            if z > 0.5:
                return {"state": "Firm", "impact": f"Mildly {name_strong.lower()}", "detail": f"Z-score {z:+.1f}"}
            if z < -1.5:
                return {"state": "Weak", "impact": name_weak, "detail": f"Z-score {z:+.1f} (below -1.5 sigma)"}
            if z < -0.5:
                return {"state": "Soft", "impact": f"Mildly {name_weak.lower()}", "detail": f"Z-score {z:+.1f}"}
            return {"state": "Neutral", "impact": "Minimal effect", "detail": f"Z-score {z:+.1f}"}

        dollar_summary = _classify_z(dxy_z, "Headwind for BTC", "Tailwind for BTC")
        vol_summary = _classify_z(vix_z, "Risk-off pressure", "Risk-on favorable")
        gold_summary = _classify_z(gold_z, "Flight to safety active", "No safe-haven demand")
        oil_summary = _classify_z(oil_z, "Inflation pressure", "Easing conditions")

        # Sentiment from F&G
        if fg is not None:
            if fg <= 20:
                sent_summary = {"state": "Extreme Fear", "impact": "Contrarian bullish", "detail": f"F&G={fg:.0f} — historically a buying zone"}
            elif fg <= 35:
                sent_summary = {"state": "Fear", "impact": "Slightly bullish", "detail": f"F&G={fg:.0f}"}
            elif fg >= 80:
                sent_summary = {"state": "Extreme Greed", "impact": "Contrarian bearish", "detail": f"F&G={fg:.0f} — historically a distribution zone"}
            elif fg >= 65:
                sent_summary = {"state": "Greed", "impact": "Slightly bearish", "detail": f"F&G={fg:.0f}"}
            else:
                sent_summary = {"state": "Neutral", "impact": "No sentiment signal", "detail": f"F&G={fg:.0f}"}
        else:
            sent_summary = {"state": "Unknown", "impact": "No data", "detail": "Fear & Greed unavailable"}

        # Rates from yield curve
        if yc is not None:
            if yc < -0.5:
                rates_summary = {"state": "Inverted", "impact": "Recession signal", "detail": f"10Y-5Y spread at {yc:.2f}%"}
            elif yc < 0:
                rates_summary = {"state": "Flat", "impact": "Caution", "detail": f"10Y-5Y spread at {yc:.2f}%"}
            elif yc > 1.0:
                rates_summary = {"state": "Steepening", "impact": "Growth signal", "detail": f"10Y-5Y spread at {yc:.2f}%"}
            else:
                rates_summary = {"state": "Normal", "impact": "Neutral", "detail": f"10Y-5Y spread at {yc:.2f}%"}
        else:
            rates_summary = {"state": "Unknown", "impact": "No data", "detail": "Yield curve unavailable"}

        # Capital flows summary
        active_edges = [e for e in edges.values() if e.get("active")]
        n_bull_flows = sum(1 for e in active_edges if e.get("direction") == "flow" and e.get("to") == "crypto")
        n_bear_flows = sum(1 for e in active_edges if e.get("direction") == "drain")
        if n_bull_flows > n_bear_flows:
            flows_summary = {"state": "Inflows", "impact": "Capital entering crypto", "detail": f"{n_bull_flows} inflow vs {n_bear_flows} drain edges active"}
        elif n_bear_flows > n_bull_flows:
            flows_summary = {"state": "Outflows", "impact": "Capital leaving crypto", "detail": f"{n_bear_flows} drain vs {n_bull_flows} inflow edges active"}
        else:
            flows_summary = {"state": "Balanced", "impact": "No clear direction", "detail": f"{len(active_edges)} active flow edges"}

        macro_summary = {
            "dollar": dollar_summary,
            "volatility": vol_summary,
            "sentiment": sent_summary,
            "rates": rates_summary,
            "energy": oil_summary,
            "flows": flows_summary,
            "whale": self._whale_macro_summary(whale, whale_has_data),
        }

        # ---- 6. Thesis Generation (unified narrative paragraph) ----

        phase = wyckoff.get("phase", "neutral")

        # Engine posture descriptions (used in thesis and later)
        posture_desc_map = {
            "RISK_ON": "selectively deploying capital in confirmed setups",
            "CAUTIOUS": "requiring higher conviction before entering positions",
            "DEFENSIVE": "filtering aggressively — only the strongest setups pass",
            "CRISIS": "in protective mode — nearly all signals are being filtered",
        }
        posture_action = posture_desc_map.get(posture, "operating normally")

        # Build a single cohesive paragraph from all intelligence sources,
        # ordered by what matters most: structure > whale flows > macro > engine state
        thesis = self._build_synthesis_paragraph(
            btc_price=btc_price,
            bias=bias,
            confidence=confidence,
            posture=posture,
            posture_action=posture_action,
            phase=phase,
            risk_temp=risk_temp,
            instability=instability,
            crisis_prob=crisis_prob,
            threshold=threshold,
            bullish_signals=bullish_signals,
            bearish_signals=bearish_signals,
            total_sources=total_sources,
            whale=whale,
            whale_has_data=whale_has_data,
            whale_raw=whale_raw,
            whale_derived=whale_derived,
            whale_conflict=whale_conflict,
            macro=macro,
            dollar_summary=dollar_summary,
            vol_summary=vol_summary,
            sent_summary=sent_summary,
            flows_summary=flows_summary,
            wyckoff=wyckoff,
            active_stress_count=active_stress_count,
        )

        # One-liner headline
        if posture == "CRISIS":
            one_liner = f"Crisis conditions detected — engine defensive at ${btc_price:,.0f}."
        elif bias == "bullish" and confidence > 0.6:
            one_liner = f"Bullish alignment across {bullish_signals} of {total_sources} sources at ${btc_price:,.0f}."
        elif bias == "bearish" and confidence > 0.6:
            one_liner = f"Bearish pressure from {bearish_signals} of {total_sources} sources at ${btc_price:,.0f}."
        else:
            one_liner = f"Mixed signals ({bullish_signals}B/{bearish_signals}S) — engine cautious at ${btc_price:,.0f}."

        # ---- 7. Outlook (short/medium/long) ----
        def _build_outlook_entry(tf_key, label_map):
            tf_data = outlook.get(tf_key) or {}
            score = tf_data.get("score", 0.0)
            label = tf_data.get("label", "Neutral")
            movers = tf_data.get("key_movers", [])
            mover_str = ", ".join(m.replace("_", " ") for m in movers[:3]) if movers else "insufficient data"
            return {
                "label": label,
                "confidence": round(abs(score), 2),
                "summary": f"{label} — driven by {mover_str}.",
            }

        outlook_section = {
            "short_term": _build_outlook_entry("1w", {}),
            "medium_term": _build_outlook_entry("1m", {}),
            "long_term": _build_outlook_entry("1y", {}),
        }

        # ---- 8. Aligned / Conflicting Factors ----
        aligned_factors = []
        conflicting_factors = []

        for tf in ["1w", "1m"]:
            tf_data = outlook.get(tf) or {}
            for f in tf_data.get("factors", []):
                name = f.get("name", "").replace("_", " ").title()
                sig = f.get("signal", 0)
                if not name:
                    continue
                if (bias == "bullish" and sig > 0.1) or (bias == "bearish" and sig < -0.1):
                    aligned_factors.append(f"{name} ({sig:+.2f})")
                elif (bias == "bullish" and sig < -0.1) or (bias == "bearish" and sig > 0.1):
                    conflicting_factors.append(f"{name} ({sig:+.2f})")

        # Deduplicate
        aligned_factors = list(dict.fromkeys(aligned_factors))[:8]
        conflicting_factors = list(dict.fromkeys(conflicting_factors))[:8]

        # ---- 9. Risks ----
        risks = []
        for scenario in stress:
            if not scenario.get("active"):
                continue
            risks.append({
                "name": scenario.get("name", "Unknown Stress"),
                "probability": round(scenario.get("probability", 0.5), 2),
                "impact": scenario.get("impact", "Moderate drawdown risk"),
                "status": "ACTIVE",
            })

        # Add structural risks from macro state
        if crisis_prob > 0.3:
            risks.append({
                "name": "Elevated Crisis Probability",
                "probability": round(crisis_prob, 2),
                "impact": "Engine applying emergency sizing and threshold penalties",
                "status": "MONITORING",
            })
        if instability > 0.6:
            risks.append({
                "name": "High Market Instability",
                "probability": round(instability, 2),
                "impact": "Choppy conditions increase false signal rate",
                "status": "MONITORING",
            })

        # ---- 10. Catalysts ----
        catalysts = []

        # Cointegration pairs near +/-2 sigma (mean reversion opportunity)
        coint_pairs = coint.get("pairs") or []
        for pair in coint_pairs:
            if pair.get("has_opportunity"):
                z = pair.get("z_score", 0)
                pair_name = pair.get("pair", "unknown")
                direction = "reversion down" if z > 0 else "reversion up"
                catalysts.append(f"{pair_name} at {z:+.1f} sigma — {direction} expected")

        # Capital flow edges about to flip
        for edge_name, edge in edges.items():
            strength = edge.get("strength", 0)
            if 0.2 < strength < 0.5 and edge.get("active"):
                catalysts.append(f"{edge.get('label', edge_name)} (strength {strength:.0%}, could intensify)")

        # Wyckoff phase transitions
        phase_next = {
            "A": "Watch for AR (Automatic Rally/Reaction) — the first counter-move after climax",
            "B": "Watch for ST (Secondary Test) — volume declining on re-tests confirms the range",
            "C": "Watch for Spring/UT — the final shakeout before the real move",
            "D": "Watch for LPS/LPSY — last pullback before trend continuation",
            "E": "Trend in progress — watch for exhaustion signals at extended levels",
            "accumulation": "Watch for SOS (Sign of Strength) — breakout from range",
            "markup": "Watch for supply entering at higher levels (UT, BC events)",
            "distribution": "Watch for SOW (Sign of Weakness) — breakdown from range",
            "markdown": "Watch for SC (Selling Climax) — capitulation and reversal",
        }
        next_watch = phase_next.get(phase)
        if next_watch:
            catalysts.append(next_watch)

        if not catalysts:
            catalysts.append("No near-term catalysts identified")

        # ---- 11. Engine Status ----
        n_positions = heartbeat_data.get("positions", 0)
        n_trades = heartbeat_data.get("completed_trades", 0)

        # Recent trade performance
        trades = self.runner.trades
        recent = trades[-5:] if len(trades) >= 5 else trades
        if recent:
            wins = sum(1 for t in recent if t.pnl > 0)
            losses = len(recent) - wins
            perf_str = f"Last {len(recent)} trades: {wins}W/{losses}L"
        else:
            perf_str = "No completed trades yet"

        # Threshold context
        if threshold > 0.55:
            thresh_desc = f"Threshold at {threshold:.2f} is very selective — filtering most signals"
        elif threshold > 0.40:
            thresh_desc = f"Threshold at {threshold:.2f} filters roughly half of signals"
        elif threshold > 0.25:
            thresh_desc = f"Threshold at {threshold:.2f} allows moderate signal flow"
        else:
            thresh_desc = f"Threshold at {threshold:.2f} is permissive — most signals pass"

        engine_status = {
            "posture_description": posture_desc_map.get(posture, "Operating normally").capitalize(),
            "active_positions": n_positions,
            "recent_performance": perf_str,
            "threshold_context": thresh_desc,
        }

        # ---- 12. Market Structure ----
        event_narratives = wyckoff.get("event_narratives") or {}
        active_events = [k for k, v in (wyckoff.get("events") or {}).items() if v.get("active")]
        event_detail = ". ".join(event_narratives.get(e, "") for e in active_events if event_narratives.get(e))

        structure_narratives = {
            "A": f"The Wyckoff structure is showing an accumulation signal with high confidence. Smart money appears to be absorbing supply at these levels. {event_detail}".strip(),
            "B": f"Building cause within the trading range. Supply and demand are being tested as the structure develops. {event_detail}".strip(),
            "C": f"Critical test phase — spring or upthrust probing the range boundary. This is the final shakeout before a directional move. {event_detail}".strip(),
            "D": f"Trend emerging from the Wyckoff range. Sign of Strength or Weakness has been confirmed. {event_detail}".strip(),
            "E": f"Trend continuation in progress. Markup or markdown phase active. {event_detail}".strip(),
            "accumulation": f"Smart money appears to be accumulating. {event_detail}".strip(),
            "markup": f"Uptrend in progress with demand in control. {event_detail}".strip(),
            "distribution": f"Smart money may be distributing at these levels. {event_detail}".strip(),
            "markdown": f"Downtrend in progress with supply dominating. {event_detail}".strip(),
            "neutral": f"No clear Wyckoff structure. {event_detail}".strip() if event_detail else "Market is ranging without a clear Wyckoff phase.",
        }

        # Key levels from Wyckoff context
        mkt_ctx = wyckoff.get("market_context") or {}
        close = mkt_ctx.get("close", btc_price)

        market_structure = {
            "summary": structure_narratives.get(phase, "Insufficient data for structural analysis."),
            "phase": phase.title() if phase != "neutral" else "Neutral / Ranging",
            "key_levels": {
                "support": round(close * 0.97, 2) if close else 0,
                "resistance": round(close * 1.03, 2) if close else 0,
                "invalidation": round(close * 0.93, 2) if close else 0,
            },
            "next_expected": phase_next.get(phase, "Continue monitoring for Wyckoff event triggers."),
        }

        # ---- Assemble Oracle ----
        return {
            "posture": posture,
            "confidence": round(confidence, 2),
            "bias": bias,
            "bias_strength": round(bias_strength, 2),
            "thesis": thesis,
            "one_liner": one_liner,
            "outlook": outlook_section,
            "aligned_factors": aligned_factors,
            "conflicting_factors": conflicting_factors,
            "risks": risks,
            "catalysts": catalysts[:6],
            "engine_status": engine_status,
            "market_structure": market_structure,
            "macro_summary": macro_summary,
        }

    def _save_heartbeat(self, timestamp, features, acted_signals):
        """Write heartbeat.json + append equity_history.csv for dashboard."""
        close_price = features.get("close", 0.0)
        if close_price != close_price or close_price is None:  # NaN guard
            close_price = 0.0
        close_price = float(close_price)
        regime = str(features.get("regime_label", "?"))
        equity = (
            self.runner.equity_curve[-1]
            if self.runner.equity_curve
            else self.initial_cash
        )

        heartbeat = {
            "timestamp": str(timestamp),
            "btc_price": float(close_price),
            "regime": regime,
            "threshold": round(self.runner.last_dynamic_threshold, 4),
            "risk_temp": round(self.runner.last_risk_temp, 4),
            "instability": round(self.runner.last_instability, 4),
            "crisis_prob": round(self.runner.last_crisis_prob, 4),
            "equity": round(equity, 2),
            "positions": len(self.runner.positions),
            "leverage": getattr(self.runner, 'leverage', 1.0),
            "open_position_details": [
                self._serialize_position(pos, close_price)
                for pos in self.runner.positions.values()
            ],
            "completed_trades": len(self.runner.trades),
            "signals_this_bar": len(acted_signals),
            "total_signals": self.runner.total_signals,
            "signals_allocated": self.runner.signals_allocated,
            "signals_rejected": self.runner.signals_rejected,
            "phantom_tracker": self.runner.get_phantom_summary(),
            "bars_processed": self.bars_processed,
            "session_start": self.session_start.isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "last_bar_signals": self.runner.last_bar_signals,
            # --- CMI Component Breakdown (Phase 1) ---
            "cmi_breakdown": self.runner.last_cmi_breakdown,
            "cmi_comparison": self.runner.last_cmi_comparison if self.runner.last_cmi_comparison else None,
            # --- Last Signal Narrative ---
            "last_signal_narrative": self.last_signal_narrative,
            # --- Macro Environment Data ---
            "macro": {
                "fear_greed": self._safe_float(features.get("FEAR_GREED", None), scale=1) or self._safe_float(features.get("fear_greed_norm", None), scale=100),
                "fear_greed_label": features.get("FEAR_GREED_LABEL", ""),
                "btc_dominance": self._safe_float(features.get("BTC.D", features.get("btc_dominance", None))),
                "usdt_dominance": self._safe_float(features.get("USDT.D", features.get("usdt_dominance", None))),
                "usdc_dominance": self._safe_float(features.get("USDC.D", features.get("usdc_dominance", None))),
                "vix_z": self._safe_float(features.get("VIX_Z", features.get("vix_z", None))),
                "dxy_z": self._safe_float(features.get("DXY_Z", features.get("dxy_z", None))),
                "gold_z": self._safe_float(features.get("GOLD_Z", features.get("gold_z", None))),
                "oil_z": self._safe_float(features.get("OIL_Z", features.get("oil_z", None))),
                "yield_curve": self._safe_float(features.get("YIELD_CURVE", features.get("yield_curve", None))),
                "eth_btc_ratio": self._safe_float(features.get("eth_btc_ratio", None), decimals=6),
                "total_market_cap": self._safe_float(features.get("total_market_cap", None), scale=1, decimals=0),
                "btc_gold_ratio": round(close_price / features.get("gold_price", 0), 2) if features.get("gold_price") and close_price > 0 else None,
                "btc_oil_ratio": round(close_price / features.get("oil_price", 0), 2) if features.get("oil_price") and close_price > 0 else None,
            },
            # --- Funding Rate Data ---
            "funding": {
                "last_rate_bps": round(self.funding_costs.get("last_funding_rate", 0.0) * 10000, 4),
                "annualized_pct": round(self.funding_costs.get("last_funding_rate", 0.0) * 3 * 365 * 100, 2),
                "total_cost_usd": round(self.funding_costs.get("total_funding_cost_usd", 0.0), 4),
                "total_events": self.funding_costs.get("total_funding_events", 0),
                "funding_z": self._safe_float(features.get("funding_Z", features.get("funding_z", None))),
            },
            # --- Whale / Institutional Intelligence ---
            "whale_intelligence": self._build_whale_intelligence(features),
        }

        # --- Wyckoff enrichment (before building wyckoff dict) ---
        # Extract enrichment data
        event_history = getattr(self.feature_computer, 'last_wyckoff_event_history', []) or []
        conviction_breakdown = getattr(self.feature_computer, 'last_wyckoff_conviction', {}) or {}

        # Track phase transitions
        btc_price = float(features.get('close', 0) or 0)
        self._track_wyckoff_phase_transition(features, timestamp, btc_price)

        # Build structural narratives
        event_narratives = self._build_event_narratives(features)

        # Calculate cycle duration
        cycle_duration_hours = None
        if self.wyckoff_cycle_start:
            try:
                start_ts = pd.Timestamp(self.wyckoff_cycle_start)
                now_ts = pd.Timestamp(timestamp)
                cycle_duration_hours = round((now_ts - start_ts).total_seconds() / 3600, 1)
            except Exception:
                pass

        heartbeat["wyckoff"] = {
                "score": self._safe_float(features.get("wyckoff_score")),
                "event_confidence": self._safe_float(features.get("wyckoff_event_confidence")),
                "tf4h_phase_score": self._safe_float(features.get("tf4h_wyckoff_phase_score")),
                "tf1d_score": self._safe_float(features.get("tf1d_wyckoff_score")),
                "tf1d_m1_signal": int(features.get("tf1d_wyckoff_m1_signal", 0) or 0),
                "tf1d_m2_signal": int(features.get("tf1d_wyckoff_m2_signal", 0) or 0),
                "bullish_1h": self._safe_float(features.get("wyckoff_bullish_score")),
                "bearish_1h": self._safe_float(features.get("wyckoff_bearish_score")),
                "bullish_4h": self._safe_float(features.get("tf4h_wyckoff_bullish_score")),
                "bearish_4h": self._safe_float(features.get("tf4h_wyckoff_bearish_score")),
                "bullish_1d": self._safe_float(features.get("tf1d_wyckoff_bullish_score")),
                "bearish_1d": self._safe_float(features.get("tf1d_wyckoff_bearish_score")),
                "tf1d_bars": int(features.get("tf1d_daily_bars", 0) or 0),
                "phase": features.get("wyckoff_phase_abc", "neutral"),
                "sequence_position": int(features.get("wyckoff_sequence_position", 0) or 0),
                "events": {
                    e: {
                        "active": bool(features.get(f"wyckoff_{e}", False)),
                        "confidence": self._safe_float(features.get(f"wyckoff_{e}_confidence")),
                    }
                    for e in [
                        "sc", "bc", "ar", "as", "st", "st_bc",
                        "sos", "sow", "spring_a", "spring_b",
                        "lps", "lpsy", "ut", "utad",
                    ]
                },
                # Raw market context for detection evidence
                "market_context": self._build_wyckoff_context(features),
                # --- NEW FIELDS: phase tracking, narratives, methodology ---
                "cycle_start": self.wyckoff_cycle_start,
                "cycle_duration_hours": cycle_duration_hours,
                "phase_transitions": self.wyckoff_phase_history[-10:],
                "event_history": event_history[:15] if isinstance(event_history, list) else [],
                "conviction": conviction_breakdown if isinstance(conviction_breakdown, dict) else {},
                "event_narratives": event_narratives,
                "typical_durations": {
                    "A": {"hours": "24-96", "description": "Stopping action: 1-4 days on 1H BTC"},
                    "B": {"hours": "168-672", "description": "Building cause: 1-4 weeks (longest phase)"},
                    "C": {"hours": "24-168", "description": "Testing phase: 1-7 days"},
                    "D": {"hours": "48-168", "description": "Trend emerging: 2-7 days"},
                    "E": {"hours": "168-2016", "description": "Trend continuation: 1-12 weeks"},
                },
                "methodology": {
                    "type": "state_machine_validated",
                    "description": "Sequential state machine validates Wyckoff events in context. Events require proper predecessors (AR needs prior SC, ST needs SC+AR with declining volume). Structures invalidated when key price levels are broken on high volume.",
                    "limitations": [
                        "No machine learning — uses rule-based thresholds",
                        "Structure resets after 500 bars (~3 weeks)",
                        "Single timeframe per call — no cross-timeframe state",
                    ],
                },
        }

        # Compute macro outlook and capital flows
        try:
            heartbeat["macro_outlook"] = self._compute_macro_outlook(
                features, heartbeat["macro"]
            )
        except Exception as exc:
            logger.warning("Failed to compute macro outlook: %s", exc)
            heartbeat["macro_outlook"] = None

        try:
            heartbeat["capital_flows"] = self._compute_capital_flows(
                features, heartbeat["macro"]
            )
        except Exception as exc:
            logger.warning("Failed to compute capital flows: %s", exc)
            heartbeat["capital_flows"] = None

        try:
            heartbeat["macro_correlations"] = self._compute_macro_correlations()
        except Exception as exc:
            logger.warning("Failed to compute macro correlations: %s", exc)
            heartbeat["macro_correlations"] = None

        # Cointegration analysis (every 4 hours to save CPU)
        try:
            heartbeat["cointegration"] = self._get_cointegration_results()
        except Exception as exc:
            logger.warning("Failed to compute cointegration: %s", exc)
            heartbeat["cointegration"] = None

        try:
            heartbeat["factor_attribution_summary"] = self._compute_factor_attribution_summary()
        except Exception as exc:
            logger.warning("Failed to compute factor attribution summary: %s", exc)
            heartbeat["factor_attribution_summary"] = None

        # Stress scenario check (every STRESS_CHECK_INTERVAL_HOURS to save CPU)
        try:
            heartbeat["active_stress_scenarios"] = self._check_stress_scenarios(features)
        except Exception as exc:
            logger.warning("Failed to check stress scenarios: %s", exc)
            heartbeat["active_stress_scenarios"] = []

        # Oracle synthesis — master narrative from all sources
        try:
            heartbeat["oracle"] = self._build_oracle_synthesis(heartbeat)
        except Exception as exc:
            logger.warning("Failed to build oracle synthesis: %s", exc)
            heartbeat["oracle"] = None

        try:
            hb_path = self.output_dir / "heartbeat.json"
            with open(hb_path, "w") as f:
                json.dump(heartbeat, f, indent=2)
        except Exception as exc:
            logger.warning("Failed to write heartbeat.json: %s", exc)

        # Append all signals (including rejected) to signal_log.json (ring buffer, last 200)
        try:
            log_path = self.output_dir / "signal_log.json"
            existing = []
            if log_path.exists():
                try:
                    existing = json.loads(log_path.read_text())
                except (json.JSONDecodeError, IOError):
                    existing = []
            existing.extend(self.runner.last_bar_signals)
            # Keep last 200 entries
            existing = existing[-200:]
            with open(log_path, "w") as f:
                json.dump(existing, f, indent=2)
        except Exception as exc:
            logger.warning("Failed to write signal_log.json: %s", exc)

        # Append to equity history CSV for chart
        try:
            hist_path = self.output_dir / "equity_history.csv"
            write_header = not hist_path.exists()
            with open(hist_path, "a") as f:
                if write_header:
                    f.write("timestamp,btc_price,equity,regime,threshold,risk_temp,instability,crisis_prob\n")
                f.write(
                    f"{timestamp},{close_price:.2f},{equity:.2f},{regime},"
                    f"{self.runner.last_dynamic_threshold:.4f},"
                    f"{self.runner.last_risk_temp:.4f},"
                    f"{self.runner.last_instability:.4f},"
                    f"{self.runner.last_crisis_prob:.4f}\n"
                )
        except Exception as exc:
            logger.warning("Failed to append equity_history.csv: %s", exc)

    # ------------------------------------------------------------------
    # Main paper trading loop
    # ------------------------------------------------------------------

    def run_paper(self):
        """
        Main paper trading loop:
        1. Wait for candle close (XX:01:30 UTC)
        2. Fetch latest completed candle from adapter
        3. Update feature computer
        4. Run process_bar() for signals
        5. Log results
        6. Fetch funding rate for cost tracking
        7. Save state periodically
        8. Sleep until next hour
        """
        # Load saved state if available (positions, cash, etc.)
        loaded = self.runner.load_state()
        if loaded:
            logger.info("Resumed from saved state.")
        else:
            logger.info("No saved state found. Starting fresh.")

        # Load Wyckoff phase tracking state
        self._load_wyckoff_state()

        # Install signal handlers for graceful shutdown
        original_sigint = signal.getsignal(signal.SIGINT)
        original_sigterm = signal.getsignal(signal.SIGTERM)

        def _shutdown_handler(signum, frame):
            sig_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
            logger.info("Received %s. Shutting down gracefully...", sig_name)
            self.running = False

        signal.signal(signal.SIGINT, _shutdown_handler)
        signal.signal(signal.SIGTERM, _shutdown_handler)

        logger.info("=" * 72)
        logger.info("COINBASE PAPER TRADING STARTED")
        logger.info("=" * 72)
        logger.info("  Data source:     %s", self.adapter_source)
        logger.info("  Config:          %s", self.config_path)
        logger.info("  Initial cash:    $%.0f", self.initial_cash)
        logger.info("  Commission:      %.2f bps per side", self.commission_rate * 10000)
        logger.info("  Slippage:        %.1f bps per side", self.slippage_bps)
        logger.info("  Output dir:      %s", self.output_dir)
        logger.info("  Press Ctrl+C to stop gracefully.")
        logger.info("=" * 72)

        try:
            self._main_loop()
        except Exception as exc:
            logger.error("Unexpected error in main loop: %s", exc, exc_info=True)
        finally:
            # Always save state and print summary on exit
            logger.info("Saving final state...")
            self.runner.save_state()
            self._save_funding_costs()
            self._save_performance_summary()
            self._save_trades()
            self._save_wyckoff_state()
            self.runner.print_summary()
            self._print_funding_summary()

            # Restore original signal handlers
            signal.signal(signal.SIGINT, original_sigint)
            signal.signal(signal.SIGTERM, original_sigterm)

    def _main_loop(self):
        """Inner loop: wait for candle close, fetch, process, sleep."""
        while self.running:
            now = datetime.now(timezone.utc)

            # ----- Wait for candle close (XX:01:30 UTC) -----
            # The candle for hour H closes at H+1:00:00. We wait until
            # H+1:01:30 to be safe, ensuring the exchange has finalized it.
            target_second = CANDLE_WAIT_MINUTES * 60 + CANDLE_WAIT_SECONDS  # 90s past the hour
            current_second_in_hour = now.minute * 60 + now.second

            if current_second_in_hour < target_second:
                sleep_secs = target_second - current_second_in_hour
                logger.debug(
                    "Waiting %ds for candle close (target: XX:%02d:%02d)...",
                    sleep_secs,
                    CANDLE_WAIT_MINUTES,
                    CANDLE_WAIT_SECONDS,
                )
                # Sleep in short intervals to respond to shutdown signals
                self._interruptible_sleep(sleep_secs)
                if not self.running:
                    break
                continue

            # ----- Fetch latest completed candle -----
            try:
                latest_candles = self.adapter.fetch_ohlcv_1h(limit=2)
            except Exception as exc:
                logger.error("Failed to fetch candles: %s", exc)
                self._interruptible_sleep(60)
                continue

            if latest_candles.empty:
                logger.warning("No candles returned from %s. Retrying in 60s.", self.adapter_source)
                self._interruptible_sleep(60)
                continue

            # Use the second-to-last candle (the most recently COMPLETED one).
            # The adapter's fetch_ohlcv_1h already drops in-progress candles,
            # but we fetch 2 for safety: the last one is the completed candle.
            if len(latest_candles) >= 2:
                candle_row = latest_candles.iloc[-1]
                candle_ts = latest_candles.index[-1]
            else:
                candle_row = latest_candles.iloc[-1]
                candle_ts = latest_candles.index[-1]

            # Skip if we already processed this candle
            if self.last_processed_ts is not None and candle_ts <= self.last_processed_ts:
                logger.info(
                    "Candle %s already processed. Sleeping until next hour.",
                    candle_ts,
                )
                self._sleep_until_next_hour()
                continue

            # ----- Process the candle -----
            acted_signals = self._process_candle(candle_row, candle_ts)
            self.last_processed_ts = candle_ts
            self.bars_processed += 1

            # ----- Fetch and track funding rate -----
            self._track_funding_costs()

            # ----- Log funding costs periodically -----
            hours_since_funding_log = (
                datetime.now(timezone.utc) - self.last_funding_log_time
            ).total_seconds() / 3600.0
            if hours_since_funding_log >= FUNDING_LOG_INTERVAL_HOURS:
                self._print_funding_summary()
                self.last_funding_log_time = datetime.now(timezone.utc)

            # ----- Save state periodically -----
            if self.bars_processed % STATE_SAVE_INTERVAL_BARS == 0:
                self.runner.save_state()
                self._save_funding_costs()
                self._save_performance_summary()
                self._save_trades()
                self._save_wyckoff_state()
                logger.info("[STATE] Periodic save complete (bar %d).", self.bars_processed)

            # ----- Sleep until next hour -----
            self._sleep_until_next_hour()

    # ------------------------------------------------------------------
    # Candle processing
    # ------------------------------------------------------------------

    def _process_candle(self, candle_row: pd.Series, timestamp: pd.Timestamp) -> List[dict]:
        """
        Process a single candle through the full pipeline:
        1. Feature computation via LiveFeatureComputer
        2. Funding rate injection
        3. Signal generation via V11ShadowRunner.process_bar()
        4. Status logging
        """
        # Convert pandas Series to dict for feature computer
        candle_dict = candle_row.to_dict()
        candle_dict["timestamp"] = timestamp

        # Save candle to history for dashboard chart
        try:
            hist_path = self.output_dir / "candle_history.csv"
            write_header = not hist_path.exists()
            with open(hist_path, "a") as f:
                if write_header:
                    f.write("timestamp,open,high,low,close,volume\n")
                f.write(
                    f"{timestamp},{candle_dict.get('open',0):.2f},"
                    f"{candle_dict.get('high',0):.2f},{candle_dict.get('low',0):.2f},"
                    f"{candle_dict.get('close',0):.2f},{candle_dict.get('volume',0):.2f}\n"
                )
            # Trim to last 200 candles
            if hist_path.stat().st_size > 20000:
                lines_hist = hist_path.read_text().splitlines()
                if len(lines_hist) > 201:
                    hist_path.write_text('\n'.join(lines_hist[:1] + lines_hist[-200:]) + '\n')
        except Exception as exc:
            logger.warning("Failed to save candle history: %s", exc)


        # Compute features (~240 columns)
        try:
            features = self.feature_computer.update(candle_dict)
        except Exception as exc:
            logger.error("Feature computation failed for %s: %s", timestamp, exc)
            return []

        if features is None or (isinstance(features, pd.Series) and features.empty):
            logger.warning("Feature computer returned empty features for %s", timestamp)
            return []

        # Validate critical fields exist
        close_val = features.get("close") if isinstance(features, dict) else features.get("close", None)
        if close_val is None or (isinstance(close_val, float) and close_val != close_val):
            logger.warning("Feature vector missing 'close' for %s — skipping bar", timestamp)
            return []

        # Inject funding rate if available
        funding_rate = self._fetch_funding_rate()
        if funding_rate is not None:
            features["funding_rate"] = funding_rate
            self.funding_costs["last_funding_rate"] = funding_rate
            self.funding_costs["last_funding_timestamp"] = str(timestamp)

        # Track feature snapshot for rolling macro correlations
        self._append_feature_snapshot(features)

        # Run the signal engine
        try:
            acted_signals = self.runner.process_bar(features, timestamp)
        except Exception as exc:
            logger.error("process_bar() failed for %s: %s", timestamp, exc, exc_info=True)
            return []

        # Generate narratives for allocated (ENTRY) signals
        for sig in acted_signals:
            if sig.get("action") == "ENTRY":
                try:
                    narrative = self._generate_signal_narrative(
                        signal=sig,
                        features=features,
                        cmi_breakdown=self.runner.last_cmi_breakdown,
                    )
                    sig["narrative"] = narrative
                    self.last_signal_narrative = narrative
                    logger.info(
                        "[NARRATIVE] %s | %s",
                        narrative.get("headline", ""),
                        narrative.get("text", ""),
                    )
                except Exception as exc:
                    logger.warning("Failed to generate narrative: %s", exc)

        # Print heartbeat status
        self._print_status(timestamp, features, acted_signals)

        return acted_signals

    # ------------------------------------------------------------------
    # Status and logging
    # ------------------------------------------------------------------

    def _print_status(self, timestamp: pd.Timestamp, features: pd.Series, acted_signals: List[dict]):
        """Print heartbeat status line with key metrics."""
        close_price = features.get("close", 0.0)
        regime = features.get("regime_label", "?")
        equity = (
            self.runner.equity_curve[-1]
            if self.runner.equity_curve
            else self.initial_cash
        )
        n_positions = len(self.runner.positions)
        n_trades = len(self.runner.trades)

        # Use actual dynamic threshold from adaptive fusion
        threshold = self.runner.last_dynamic_threshold

        logger.info(
            "[HEARTBEAT] %s | BTC=$%.0f | regime=%s | threshold=%.3f | "
            "risk_temp=%.3f | crisis=%.3f | "
            "positions=%d | equity=$%.0f | trades=%d | signals=%d",
            timestamp,
            close_price,
            regime,
            threshold,
            self.runner.last_risk_temp,
            self.runner.last_crisis_prob,
            n_positions,
            equity,
            n_trades,
            len(acted_signals),
        )

        # Write heartbeat.json for dashboard
        self._save_heartbeat(timestamp, features, acted_signals)

        # Log individual entries/exits prominently
        for sig in acted_signals:
            action = sig.get("action", "?")
            archetype = sig.get("archetype", "?")
            direction = sig.get("direction", "?")
            fusion = sig.get("fusion_score", 0.0)
            price = sig.get("entry_price", 0.0)
            reason = sig.get("reason", "")
            if action == "ENTRY":
                logger.info(
                    ">>> ENTRY: %s %s | fusion=%.3f | price=$%.2f | %s",
                    direction.upper(),
                    archetype,
                    fusion,
                    price,
                    reason,
                )
            elif action == "EXIT":
                logger.info(
                    "<<< EXIT: %s %s | price=$%.2f | %s",
                    direction.upper(),
                    archetype,
                    price,
                    reason,
                )

    # ------------------------------------------------------------------
    # Funding rate tracking
    # ------------------------------------------------------------------

    def _fetch_funding_rate(self) -> Optional[float]:
        """Fetch the current funding rate from the data adapter."""
        # Try CoinbaseFundingClient first (more detailed)
        if self.funding_client is not None:
            try:
                result = self.funding_client.get_current_funding_rate()
                if result and "funding_rate" in result:
                    return float(result["funding_rate"])
            except Exception as exc:
                logger.debug("CoinbaseFundingClient failed: %s", exc)

        # Fall back to adapter's fetch_funding_rate
        try:
            rate = self.adapter.fetch_funding_rate()
            return float(rate)
        except Exception as exc:
            logger.debug("fetch_funding_rate failed: %s", exc)
            return None

    def _track_funding_costs(self):
        """
        Calculate and accumulate funding costs for all open positions.

        Funding cost per hour = position_size_usd * funding_rate * leverage
        (Perpetual futures typically charge funding every 8 hours,
        but we track hourly costs for continuous monitoring.)
        """
        if not self.runner.positions:
            return

        funding_rate = self.funding_costs.get("last_funding_rate", 0.0)
        if funding_rate == 0.0:
            return

        leverage = self.runner.leverage
        if leverage != leverage or leverage is None or leverage <= 0:  # NaN / invalid guard
            leverage = 1.0
        total_hourly_cost = 0.0

        for pos_id, pos in self.runner.positions.items():
            # Position size in USD (with NaN guard)
            qty = pos.current_quantity if pos.current_quantity == pos.current_quantity else 0.0
            price = pos.entry_price if pos.entry_price == pos.entry_price else 0.0
            position_size_usd = qty * price
            if position_size_usd <= 0:
                continue

            # Hourly funding cost.
            # Standard perp funding is every 8h, so hourly fraction = rate / 8.
            # Positive funding rate: longs pay shorts.
            # Negative funding rate: shorts pay longs.
            if pos.direction == "long":
                hourly_cost = position_size_usd * funding_rate * leverage / 8.0
            else:
                # Shorts receive funding when rate is positive
                hourly_cost = -position_size_usd * funding_rate * leverage / 8.0

            total_hourly_cost += hourly_cost

            # Track per-position costs
            if pos_id not in self.funding_costs["cost_by_position"]:
                self.funding_costs["cost_by_position"][pos_id] = 0.0
            self.funding_costs["cost_by_position"][pos_id] += hourly_cost

        self.funding_costs["total_funding_cost_usd"] += total_hourly_cost
        self.funding_costs["total_funding_events"] += 1

        if abs(total_hourly_cost) > 0.01:
            logger.debug(
                "[FUNDING] Hourly cost: $%.4f | Rate: %.6f | Open positions: %d",
                total_hourly_cost,
                funding_rate,
                len(self.runner.positions),
            )

    def _print_funding_summary(self):
        """Print accumulated funding cost summary."""
        total_cost = self.funding_costs["total_funding_cost_usd"]
        n_events = self.funding_costs["total_funding_events"]
        last_rate = self.funding_costs.get("last_funding_rate", 0.0)
        last_ts = self.funding_costs.get("last_funding_timestamp", "N/A")

        logger.info("=" * 60)
        logger.info("[FUNDING SUMMARY]")
        logger.info("  Total accumulated cost:   $%.4f", total_cost)
        logger.info("  Funding events tracked:   %d", n_events)
        logger.info("  Last funding rate:        %.6f (%.4f%%)", last_rate, last_rate * 100)
        logger.info("  Last funding timestamp:   %s", last_ts)

        # Per-position breakdown (top 5 by cost)
        pos_costs = self.funding_costs.get("cost_by_position", {})
        if pos_costs:
            sorted_costs = sorted(pos_costs.items(), key=lambda x: abs(x[1]), reverse=True)
            for pos_id, cost in sorted_costs[:5]:
                logger.info("    %s: $%.4f", pos_id, cost)
        logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Sleep helpers
    # ------------------------------------------------------------------

    def _interruptible_sleep(self, total_seconds: float):
        """Sleep in 1-second intervals so we can respond to shutdown signals."""
        end_time = time.time() + total_seconds
        while self.running and time.time() < end_time:
            remaining = end_time - time.time()
            time.sleep(min(1.0, max(0.0, remaining)))

    def _sleep_until_next_hour(self):
        """Sleep until the next candle processing time (next hour + offset)."""
        now = datetime.now(timezone.utc)
        # Next processing time: next hour + CANDLE_WAIT_MINUTES:CANDLE_WAIT_SECONDS
        next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
        next_process_time = next_hour + timedelta(
            minutes=CANDLE_WAIT_MINUTES, seconds=CANDLE_WAIT_SECONDS
        )
        sleep_secs = (next_process_time - now).total_seconds()
        if sleep_secs > 0:
            logger.debug("Sleeping %.0fs until next candle at %s.", sleep_secs, next_process_time)
            self._interruptible_sleep(sleep_secs)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Coinbase Paper Trading Runner for Bull Machine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Paper trading with Coinbase data:
  python3 bin/live/coinbase_runner.py --paper

  # Paper trading with Binance fallback:
  python3 bin/live/coinbase_runner.py --paper --fallback-binance

  # Custom config and capital:
  python3 bin/live/coinbase_runner.py --paper --initial-cash 25000 \\
      --config configs/bull_machine_isolated_v11_fixed.json
""",
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--paper",
        action="store_true",
        help="Paper trading mode (real market data, virtual fills). Default.",
    )
    mode.add_argument(
        "--live",
        action="store_true",
        help="Live trading mode (NOT YET IMPLEMENTED -- will exit with error).",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG,
        help=f"Path to config JSON (default: {DEFAULT_CONFIG}).",
    )
    parser.add_argument(
        "--initial-cash",
        type=float,
        default=DEFAULT_INITIAL_CASH,
        help=f"Starting capital in USD (default: {DEFAULT_INITIAL_CASH:,.0f}).",
    )
    parser.add_argument(
        "--commission-rate",
        type=float,
        default=DEFAULT_COMMISSION_RATE,
        help=f"Commission per side as decimal (default: {DEFAULT_COMMISSION_RATE} = {DEFAULT_COMMISSION_RATE*10000:.1f} bps).",
    )
    parser.add_argument(
        "--slippage-bps",
        type=float,
        default=DEFAULT_SLIPPAGE_BPS,
        help=f"Slippage per side in basis points (default: {DEFAULT_SLIPPAGE_BPS}).",
    )
    parser.add_argument(
        "--fallback-binance",
        action="store_true",
        help="Use Binance as data source if Coinbase is unavailable.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug-level logging.",
    )

    args = parser.parse_args()

    # ---- Logging setup ----
    log_level = logging.DEBUG if args.verbose else logging.INFO
    log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(level=log_level, format=log_format)

    # Quiet down noisy libraries unless verbose
    if not args.verbose:
        logging.getLogger("ccxt").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("requests").setLevel(logging.WARNING)

    # ---- Live mode guard ----
    if args.live:
        logger.error(
            "Live trading mode is NOT YET IMPLEMENTED.\n"
            "This would place real orders on Coinbase with real money.\n"
            "Use --paper for paper trading with real market data and virtual fills."
        )
        sys.exit(1)

    # ---- Paper trading mode ----
    runner = CoinbasePaperRunner(
        config_path=args.config,
        initial_cash=args.initial_cash,
        commission_rate=args.commission_rate,
        slippage_bps=args.slippage_bps,
        use_binance_fallback=args.fallback_binance,
        verbose=args.verbose,
    )

    # Warmup: fetch historical candles and seed feature computer
    runner.warmup()

    # Start the paper trading loop
    runner.run_paper()


if __name__ == "__main__":
    main()
