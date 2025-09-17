import pytest
from bull_machine.modules.fusion.advanced import AdvancedFusionEngine
from bull_machine.core.types import Series, Bar, WyckoffResult

def _series(prices, volumes=None, vol=0.01):
    volumes = volumes or [1000] * len(prices)
    bars = [Bar(ts=i, open=p, high=p*(1+vol), low=p*(1-vol), close=p, volume=v)
            for i, (p, v) in enumerate(zip(prices, volumes))]
    return Series(bars=bars, symbol="TEST", timeframe="1h")

def _wy(phase='C', bias='long', phase_conf=0.7, trend_conf=0.7):
    return WyckoffResult(regime='trending', phase=phase, bias=bias,
                        phase_confidence=phase_conf, trend_confidence=trend_conf, range=None)

def test_early_phase_veto_blocks_low_confidence():
    """Early-phase veto should block Phase A/B unless confidence > 0.6"""
    config = {
        "features": {"veto_system": True},
        "fusion": {"enter_threshold": 0.40, "weights": {"wyckoff": 0.5, "liquidity": 0.5}}
    }
    engine = AdvancedFusionEngine(config)

    # Phase A with low confidence - should be vetoed
    modules_data = {
        "wyckoff": _wy(phase='A', phase_conf=0.5),
        "liquidity": {"overall_score": 0.8},
        "series": _series([100, 101, 102])
    }

    vetoes = engine._check_vetoes(modules_data)
    assert 'early_wyckoff_phase' in vetoes

def test_early_phase_veto_allows_high_confidence():
    """Early-phase veto should allow Phase A/B with confidence > 0.6"""
    config = {
        "features": {"veto_system": True},
        "fusion": {"enter_threshold": 0.40, "weights": {"wyckoff": 0.5, "liquidity": 0.5}}
    }
    engine = AdvancedFusionEngine(config)

    # Phase A with high confidence - should be allowed
    modules_data = {
        "wyckoff": _wy(phase='A', phase_conf=0.7),
        "liquidity": {"overall_score": 0.5},
        "series": _series([100, 101, 102])
    }

    vetoes = engine._check_vetoes(modules_data)
    assert 'early_wyckoff_phase' not in vetoes

def test_volatility_veto_normal_move():
    """Volatility veto should NOT trigger for normal market moves"""
    config = {
        "features": {"veto_system": True},
        "fusion": {"volatility_shock_sigma": 4.0, "enter_threshold": 0.40, "weights": {"wyckoff": 0.5, "liquidity": 0.5}}
    }
    engine = AdvancedFusionEngine(config)

    # Normal price series with gradual moves
    normal_prices = [100 + i*0.5 for i in range(20)]  # 0.5% moves
    modules_data = {
        "wyckoff": _wy(),
        "liquidity": {"overall_score": 0.5},
        "series": _series(normal_prices)
    }

    vetoes = engine._check_vetoes(modules_data)
    assert 'volatility_shock' not in vetoes

def test_volatility_veto_extreme_move():
    """Volatility veto should trigger for extreme moves > mu + 4σ"""
    config = {
        "features": {"veto_system": True},
        "fusion": {"volatility_shock_sigma": 4.0, "enter_threshold": 0.40, "weights": {"wyckoff": 0.5, "liquidity": 0.5}}
    }
    engine = AdvancedFusionEngine(config)

    # Create series with extreme final move
    normal_prices = [100 + i*0.1 for i in range(15)]  # Normal 0.1% moves
    extreme_prices = normal_prices + [normal_prices[-1] * 1.15]  # 15% spike

    modules_data = {
        "wyckoff": _wy(),
        "liquidity": {"overall_score": 0.5},
        "series": _series(extreme_prices)
    }

    vetoes = engine._check_vetoes(modules_data)
    assert 'volatility_shock' in vetoes

def test_trend_filter_alignment():
    """Trend filter should allow aligned trades and block counter-trend"""
    config = {
        "features": {"trend_filter": True, "veto_system": True},
        "fusion": {"trend_alignment_threshold": 0.60, "enter_threshold": 0.40,
                   "weights": {"wyckoff": 0.5, "liquidity": 0.5}}
    }
    engine = AdvancedFusionEngine(config)

    # Long bias with sufficient Wyckoff score - should be allowed
    modules_data = {
        "wyckoff": _wy(bias='long', phase_conf=0.8, trend_conf=0.8),  # Combined = 0.8
        "liquidity": {"overall_score": 0.5},
        "series": _series([100, 101, 102])
    }

    result = engine.fuse(modules_data)
    # Should not have trend filter veto since Wyckoff score (0.8) > threshold (0.6)
    assert 'trend_filter_long' not in result.breakdown.get('vetoes', [])