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


def test_bojan_wick_boost_parity():
    """Boost 4 (2026-07-28): wick-majority >= 0.5 lower-wick, capex-scoped,
    present in both engines and enabled live."""
    for src in (BT, LV):
        assert "bojan_wick_majority" in src
        assert "0.5" in src[src.index("bojan_wick_boost"):src.index("bojan_wick_majority")]
    cfg = json.loads((REPO / "configs/champion_paper.json").read_text())
    assert cfg["bojan_wick_boost"]["enabled"] is True


def test_live_metadata_enrichment():
    """Live signals must carry taker_imbalance + wick_lower_ratio into
    metadata BEFORE Step 4b, else the boosts silently never fire live."""
    enrich = LV[LV.index("Step 3e"):LV.index("Step 4:")]
    assert "s.metadata['taker_imbalance']" in enrich
    assert "s.metadata['wick_lower_ratio']" in enrich


def test_breadth_boost_parity():
    """Boost 5 (2026-08-01): LOCAL-flush breadth, capex-scoped, both engines,
    enabled live, metadata enrichment present."""
    for src in (BT, LV):
        assert "local_flush_breadth" in src
    enrich = LV[LV.index("Step 3e"):LV.index("Step 4:")]
    assert "alt_basket_ret_4h" in enrich
    cfg = json.loads((REPO / "configs/champion_paper.json").read_text())
    assert cfg["breadth_boost"]["enabled"] is True


def test_exodus_refusal_parity():
    """Refusal rule (2026-08-02): identity gate in logic.py, live feed +
    passthrough present, gate enabled in champion_paper."""
    LG = (REPO / "engine/archetypes/logic.py").read_text()
    assert "wt_no_exodus_K" in LG and "stables_rot_rising" in LG
    RN = (REPO / "bin/live/coinbase_runner.py").read_text()
    assert "_refresh_stables_rotation" in RN
    LF = (REPO / "bin/live/live_feature_computer.py").read_text()
    assert "stables_rot_rising" in LF
    cfg = json.loads((REPO / "configs/champion_paper.json").read_text())
    assert cfg["structural_checks"]["gate_params"]["wt_no_exodus_K"] == 1


def test_wyckoff_phase_boost_parity():
    """Boost 6 (2026-08-04): C_accum long context boost, capex-scoped, present
    in both engines; live metadata enrichment carries wyckoff_phase_dir.
    (Config enablement is asserted separately once the deploy decision lands.)"""
    for src in (BT, LV):
        assert "wyckoff_phase_boost" in src
        assert "wyckoff_phase_C_accum" in src
    enrich = LV[LV.index("Step 3e"):LV.index("Step 4:")]
    assert "wyckoff_phase_dir" in enrich
