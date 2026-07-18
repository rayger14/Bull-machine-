"""Wiring tests for the resurrected HOB demand-reaction archetype (S9).

Pre-registered 2026-07-17: identity = the ORIGINAL v1.x HOBDetector
(engine/liquidity/hob.py) with untouched defaults, long side only.
No tuning, no hard gates — the detector IS the archetype.
"""
import numpy as np
import pandas as pd
import yaml
from pathlib import Path

from engine.archetypes.structural_check import StructuralChecker, NAME_TO_LETTER

REPO = Path(__file__).resolve().parents[1]


def make_df(n=300, seed=7):
    """Synthetic OHLCV with a support level being retested on the last bar."""
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 0.3, n))
    k = min(40, n // 2)
    close[-k:] = close[-k] + rng.normal(0, 0.05, k)  # consolidation
    df = pd.DataFrame({
        "open": close + rng.normal(0, 0.05, n),
        "close": close,
        "volume": rng.uniform(5, 15, n),
    })
    df["high"] = df[["open", "close"]].max(axis=1) + rng.uniform(0, 0.2, n)
    df["low"] = df[["open", "close"]].min(axis=1) - rng.uniform(0, 0.2, n)
    df.index = pd.date_range("2024-01-01", periods=n, freq="h")
    return df


def test_mapping_registered():
    assert NAME_TO_LETTER.get("hob_reaction") == "S9"


def test_yaml_loads_and_is_preregistered():
    cfg = yaml.safe_load((REPO / "configs/archetypes/hob_reaction.yaml").read_text())
    assert cfg["name"] == "hob_reaction"
    assert cfg["direction"] == "long"
    assert cfg["hard_gates"] == []          # detector IS the gate
    ws = cfg["fusion_weights"]
    assert all(abs(w - 0.25) < 1e-9 for w in ws.values())  # neutral, untuned


def test_check_returns_bool_and_never_errors():
    sc = StructuralChecker(config={}, mode="backtest")
    df = make_df()
    passed, reason = sc.check_structure("hob_reaction", df.iloc[-1], df.iloc[-2],
                                        df, 5000)
    assert isinstance(passed, bool)
    assert "error" not in reason  # identity errors must be clean False, not error-pass


def test_insufficient_history_is_false():
    sc = StructuralChecker(config={}, mode="backtest")
    df = make_df(n=30)  # below the detector's 50-bar minimum
    passed, _ = sc.check_structure("hob_reaction", df.iloc[-1], df.iloc[-2], df, 29)
    assert passed is False


def test_positional_index_handled():
    """Live buffers have positional indexes — the check synthesizes a
    deterministic DatetimeIndex, so verdicts match the datetime-index path."""
    sc1 = StructuralChecker(config={}, mode="backtest")
    sc2 = StructuralChecker(config={}, mode="backtest")
    df = make_df()
    r1, _ = sc1.check_structure("hob_reaction", df.iloc[-1], df.iloc[-2], df, 299)
    df2 = df.reset_index(drop=True)
    r2, _ = sc2.check_structure("hob_reaction", df2.iloc[-1], df2.iloc[-2], df2, 299)
    assert r1 == r2
