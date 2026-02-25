"""Test v1.3 MTF sync and fusion features"""

import pytest
from bull_machine.core.types import BiasCtx, WyckoffResult, LiquidityResult
from bull_machine.core.sync import decide_mtf_entry
from bull_machine.fusion.fuse import FusionEngineV1_3


def test_mtf_sync_contract():
    """Test MTF sync decision logic with desync scenario"""
    # HTF and MTF short, LTF long = desync
    htf = BiasCtx(
        tf="1D",
        bias="short",
        confirmed=True,
        strength=0.8,
        bars_confirmed=3,
        ma_distance=0.05,
        trend_quality=0.7,
        ma_slope=-0.02,
    )

    mtf = BiasCtx(
        tf="4H",
        bias="short",
        confirmed=True,
        strength=0.7,
        bars_confirmed=2,
        ma_distance=0.03,
        trend_quality=0.6,
        ma_slope=-0.01,
    )

    # Test with desync (LTF opposite of HTF)
    rep = decide_mtf_entry(
        htf,
        mtf,
        ltf_bias="long",  # Opposite of HTF
        nested_ok=False,
        eq_magnet=False,
        policy={
            "desync_behavior": "raise",
            "desync_bump": 0.10,
            "eq_magnet_gate": True,
            "eq_bump": 0.05,
            "nested_bump": 0.03,
            "alignment_discount": 0.05,
        },
    )

    # Should raise threshold due to desync
    assert rep.decision in ("raise", "veto")
    assert rep.desync is True
    if rep.decision == "raise":
        assert rep.threshold_bump > 0


def test_fusion_no_neutral():
    """Test that fusion never emits 'neutral' side"""
    config = {
        "mtf": {"enabled": False},
        "fusion": {
            "enter_threshold": 0.0,  # Very low to ensure signal
            "weights": {"wyckoff": 0.6, "liquidity": 0.4},
        },
    }

    eng = FusionEngineV1_3(config)

    # Wyckoff neutral but liquidity bearish
    wy = WyckoffResult(
        regime="trending",
        phase="D",
        bias="neutral",
        phase_confidence=0.3,
        trend_confidence=0.3,
        range=None,
    )

    liq = LiquidityResult(score=0.9, pressure="bearish", fvgs=[], order_blocks=[])

    sig = eng.fuse_with_mtf({"wyckoff": wy, "liquidity": liq}, None)

    # Should determine side from liquidity pressure
    assert sig is None or sig.side in ("long", "short")
    if sig:
        assert sig.side == "short"  # Should follow liquidity bearish


def test_eq_magnet_veto():
    """Test EQ magnet veto behavior"""
    htf = BiasCtx(
        tf="1D",
        bias="long",
        confirmed=True,
        strength=0.7,
        bars_confirmed=2,
        ma_distance=0.02,
        trend_quality=0.6,
        ma_slope=0.01,
    )

    mtf = BiasCtx(
        tf="4H",
        bias="long",
        confirmed=True,
        strength=0.6,
        bars_confirmed=2,
        ma_distance=0.02,
        trend_quality=0.5,
        ma_slope=0.01,
    )

    # Test with EQ magnet active and gate enabled
    rep = decide_mtf_entry(
        htf,
        mtf,
        ltf_bias="long",
        nested_ok=True,
        eq_magnet=True,  # In equilibrium zone
        policy={
            "desync_behavior": "raise",
            "eq_magnet_gate": True,  # Should veto
            "eq_bump": 0.05,
        },
    )

    # Should veto due to EQ magnet
    assert rep.decision == "veto"
    assert rep.eq_magnet is True


def test_perfect_alignment_bonus():
    """Test perfect MTF alignment gives threshold discount"""
    htf = BiasCtx(
        tf="1D",
        bias="long",
        confirmed=True,
        strength=0.9,
        bars_confirmed=3,
        ma_distance=0.05,
        trend_quality=0.8,
        ma_slope=0.03,
    )

    mtf = BiasCtx(
        tf="4H",
        bias="long",
        confirmed=True,
        strength=0.85,
        bars_confirmed=3,
        ma_distance=0.04,
        trend_quality=0.75,
        ma_slope=0.025,
    )

    # Perfect alignment
    rep = decide_mtf_entry(
        htf,
        mtf,
        ltf_bias="long",  # All aligned
        nested_ok=True,
        eq_magnet=False,
        policy={"desync_behavior": "raise", "alignment_discount": 0.05},
    )

    # Should allow with discount
    assert rep.decision == "allow"
    assert rep.alignment_score >= 0.8  # Perfect 3-way alignment = 83.33%
    assert rep.threshold_bump < 0  # Negative = discount
