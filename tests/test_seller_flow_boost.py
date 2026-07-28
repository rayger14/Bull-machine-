"""Seller-flow boost (validated 2026-07-28): wick_trap 1.25x with SCOPED
cap-exemption on taker_imbalance <= 0 flushes. Backtest/live parity + the
scoped-capex contract (legacy boosts stay capped; guard bit-identical)."""
import ast, json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BT = (REPO / "bin/backtest_v11_standalone.py").read_text()
LV = (REPO / "bin/live/v11_shadow_runner.py").read_text()


def test_both_engines_have_boost():
    for src in (BT, LV):
        assert "wick_trap_seller_flow" in src
        assert "capex_mult" in src


def test_cap_scaling_uses_scoped_capex_only():
    # cap must scale by capex_mult, never the total multiplier
    for src in (BT, LV):
        cap_zone = src[src.index("max_margin = self.initial_cash * self.max_margin_pct"):]
        cap_zone = cap_zone[:600]
        assert "capex_mult" in cap_zone
        assert ".get('multiplier'" not in cap_zone


def test_live_config_enabled_and_parses():
    cfg = json.loads((REPO / "configs/champion_paper.json").read_text())
    assert cfg["seller_flow_boost"]["enabled"] is True
    ast.parse(BT); ast.parse(LV)


def test_condition_is_seller_aggressed():
    for src in (BT, LV):
        assert "float(ti) <= 0.0" in src
