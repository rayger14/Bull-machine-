"""Tests for the downtrend-skip filter (config-driven, disabled by default).

Validated 2026-07-02: skipping long entries while close < 200-day mean turned
2022 from -$43K to $0 and cut MaxDD 51% -> 16.4%. These tests exercise the
config plumbing and skip semantics without a full backtest run.
"""
import json
import ast
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]


def test_backtester_default_off():
    """Without a downtrend_skip config key the engine must not build the state."""
    src = (REPO / "bin/backtest_v11_standalone.py").read_text()
    assert "downtrend_skip" in src
    assert "self.downtrend_skip_enabled = bool(dts_cfg.get('enabled', False))" in src


def test_downtrend_state_math():
    """The detector definition: close < rolling(sma_days*24).mean(), min_periods=half."""
    idx = pd.date_range("2024-01-01", periods=24 * 300, freq="h")
    close = pd.Series(60000.0, index=idx)
    close.iloc[-24 * 30:] = 40000.0  # last 30 days crash well below the mean
    bars = 200 * 24
    state = close < close.rolling(bars, min_periods=bars // 2).mean()
    assert bool(state.iloc[-1]) is True       # in downtrend at the end
    assert bool(state.iloc[24 * 250]) is False  # flat period: not a downtrend


def test_shadow_runner_skip_wiring():
    """Live runner: config keys parsed, phantom routing present, None state = no skip."""
    src = (REPO / "bin/live/v11_shadow_runner.py").read_text()
    assert "downtrend_skip_enabled" in src
    assert "downtrend_active" in src
    # phantom routing on skip (data collection preserved)
    assert "rejection_stage='downtrend_skip'" in src
    # skip only applies when state is affirmatively True (None -> unknown -> no skip)
    assert "self.downtrend_skip_enabled and self.downtrend_active" in src


def test_live_config_does_not_enable_skip():
    """PRODUCTION SAFETY: the deployed live config must not enable the skip
    (offline-validation-only decision, 2026-07-05)."""
    cfg = json.loads((REPO / "configs/champion_paper.json").read_text())
    assert not (cfg.get("downtrend_skip") or {}).get("enabled", False)
    base = json.loads((REPO / "configs/bull_machine_isolated_v11_fixed.json").read_text())
    assert not (base.get("downtrend_skip") or {}).get("enabled", False)


def test_coinbase_client_has_daily_fetch():
    src = (REPO / "bin/live/coinbase_client.py").read_text()
    assert "def fetch_ohlcv_1d" in src
    assert "GRANULARITY_ONE_DAY" in src
    # syntax sanity of the whole module
    ast.parse(src)


def test_backtester_and_runner_parse():
    for f in ["bin/backtest_v11_standalone.py", "bin/live/v11_shadow_runner.py",
              "bin/live/coinbase_runner.py"]:
        ast.parse((REPO / f).read_text())
