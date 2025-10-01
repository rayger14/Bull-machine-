"""Advanced module scaffolds for Bull Machine v1.2.1 / v1.3.

These implement the **interfaces** expected by main_v13.py and friends.
Claude Code should fill in TODOs to make them production-ready.
"""

from typing import Any, Dict, List, Optional

try:
    from bull_machine.core.types import (
        BiasCtx,
        LiquidityResult,
        RangeCtx,
        Series,
        Signal,
        SyncReport,
        WyckoffResult,
    )
except Exception:
    from dataclasses import dataclass, field

    @dataclass
    class WyckoffResult:
        regime: str = "neutral"
        phase: str = "neutral"
        bias: str = "neutral"
        phase_confidence: float = 0.0
        trend_confidence: float = 0.0
        range: Optional[Dict] = None
        notes: List[str] = field(default_factory=list)

        @property
        def confidence(self) -> float:
            return (self.phase_confidence + self.trend_confidence) / 2.0

    @dataclass
    class LiquidityResult:
        score: float = 0.0
        pressure: str = "neutral"
        fvgs: List[Dict] = field(default_factory=list)
        order_blocks: List[Dict] = field(default_factory=list)
        sweeps: List[Dict] = field(default_factory=list)
        phobs: List[Dict] = field(default_factory=list)
        metadata: Dict = field(default_factory=dict)

    @dataclass
    class Signal:
        ts: int = 0
        side: str = "neutral"
        confidence: float = 0.0
        reasons: List[str] = field(default_factory=list)
        ttl_bars: int = 0
        metadata: Dict = field(default_factory=dict)
        mtf_sync: Optional[Any] = None

    @dataclass
    class BiasCtx:
        tf: str = "1H"
        bias: str = "neutral"
        confirmed: bool = False
        strength: float = 0.0
        bars_confirmed: int = 0
        ma_distance: float = 0.0
        trend_quality: float = 0.0

    @dataclass
    class RangeCtx:
        tf: str = "1H"
        low: float = 0.0
        high: float = 0.0
        mid: float = 0.0

    @dataclass
    class SyncReport:
        htf: BiasCtx = BiasCtx()
        mtf: BiasCtx = BiasCtx()
        ltf_bias: str = "neutral"
        nested_ok: bool = False
        eq_magnet: bool = False
        desync: bool = False
        decision: str = "raise"
        threshold_bump: float = 0.0
        alignment_score: float = 0.0
        notes: List[str] = field(default_factory=list)

    @dataclass
    class Series:
        bars: List[Any] = field(default_factory=list)
        timeframe: str = "1H"
        symbol: str = "UNKNOWN"


class FusionEngineV1_3:
    """
    6-layer fusion (v1.2.1) + MTF-gating hook (v1.3).
    Expected API:
      - __init__(config)
      - fuse(modules: dict, sync_report: Optional[Any]) -> Signal|None
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.weights = self.config.get("signals", {}).get(
            "weights",
            {
                "wyckoff": 0.30,
                "liquidity": 0.25,
                "structure": 0.20,
                "momentum": 0.10,
                "volume": 0.10,
                "context": 0.05,
            },
        )
        self.base_threshold = self.config.get("signals", {}).get("enter_threshold", 0.35)

    def _fuse_impl(self, modules: Dict[str, Any], sync_report: Optional[Any] = None) -> Optional[Signal]:
        # Schema validation to catch field drift early
        self._validate_modules(modules)

        wy = modules.get("wyckoff")
        liq = modules.get("liquidity")

        # Safe field access with fallbacks
        side = getattr(wy, "bias", "neutral") if wy else "neutral"
        liq_score = self._safe_get_score(liq) if liq else 0.0

        # Calculate weighted score
        wyckoff_weight = self.weights.get("wyckoff", 0.30)
        liquidity_weight = self.weights.get("liquidity", 0.25)

        wyckoff_score = 0.0
        if wy and side in ("long", "short"):
            # Use phase and trend confidence for wyckoff contribution
            avg_confidence = (
                getattr(wy, "phase_confidence", 0.0) + getattr(wy, "trend_confidence", 0.0)
            ) / 2
            wyckoff_score = avg_confidence * wyckoff_weight

        liquidity_score = liq_score * liquidity_weight if liq_score > 0 else 0.0

        # Simple structure/momentum/volume/context contributions (placeholders)
        other_score = 0.1 * (
            self.weights.get("structure", 0.20)
            + self.weights.get("momentum", 0.10)
            + self.weights.get("volume", 0.10)
            + self.weights.get("context", 0.05)
        )

        total_score = wyckoff_score + liquidity_score + other_score

        # Apply MTF gating if sync_report provided
        if sync_report:
            mtf_boost = getattr(sync_report, "threshold_bump", 0.0)
            total_score += mtf_boost

        # Only signal if above threshold and bias is clear
        if side not in ("long", "short") or total_score < self.base_threshold:
            return None

        # Build detailed reasons
        reasons = []
        if wyckoff_score > 0.1:
            reasons.append(f"wyckoff_{side}")
        if liquidity_score > 0.1:
            reasons.append(f"liquidity_{getattr(liq, 'pressure', 'neutral')}")
        if len(reasons) == 0:
            reasons.append("minimal_signal")

        return Signal(
            ts=0,
            side=side,
            confidence=min(0.95, total_score),
            reasons=reasons,
            ttl_bars=20,
            metadata={
                "wyckoff_score": wyckoff_score,
                "liquidity_score": liquidity_score,
                "total_score": total_score,
                "threshold": self.base_threshold,
            },
        )

    def _validate_modules(self, modules: Dict[str, Any]):
        """Validate module schemas to catch field drift early"""
        wy = modules.get("wyckoff")
        liq = modules.get("liquidity")

        if wy:
            self._require(
                wy,
                ["bias", "phase", "regime", "phase_confidence", "trend_confidence"],
                "WyckoffResult",
            )
        if liq:
            self._require(liq, ["score", "pressure", "fvgs", "order_blocks"], "LiquidityResult")

    def _require(self, obj, keys, label):
        """Ensure required fields exist on object"""
        missing = [k for k in keys if not hasattr(obj, k)]
        if missing:
            raise ValueError(f"{label} missing fields: {missing}")

    def _safe_get_score(self, liq_result):
        """Safe access for liquidity score with fallbacks"""
        return getattr(liq_result, "score", getattr(liq_result, "overall_score", 0.0)) or 0.0


# Backward compatibility alias for tests
from bull_machine.modules.fusion.enhanced import EnhancedFusionEngineV1_4


class AdvancedFusionEngine(EnhancedFusionEngineV1_4):
    """Compatibility wrapper to bridge API differences."""

    def fuse(self, modules: Dict[str, Any], sync_report: Optional[Any] = None) -> Optional[Signal]:
        """Main fusion entry point - delegate to parent's fuse_with_mtf."""
        # Parent class only has fuse_with_mtf, so call that
        return super().fuse_with_mtf(modules, sync_report)
