"""Regression test: CHoCH flags must survive _extra_archetype_features.

The 2026-07-13 engine-integrity batch fixed CHoCH derivation in
_smc_features, but a leftover "simplified" stub in _extra_archetype_features
(which runs LATER in update()) unconditionally overwrote both flags to 0 —
so CHoCH stayed dead on every live bar until 2026-07-16. These tests pin the
ordering contract: _extra_archetype_features must pass through any CHoCH
values already computed, defaulting to 0 only when SMC never set them.
"""
import ast
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = (REPO / "bin/live/live_feature_computer.py").read_text()


def _fn_body(name: str) -> str:
    m = re.search(rf"def {name}\(.*?(?=\n    def )", SRC, re.S)
    assert m, f"{name} not found"
    return m.group(0)


def test_extra_features_does_not_clobber_choch():
    body = _fn_body("_extra_archetype_features")
    assert "out['tf1h_choch_detected'] = features.get('tf1h_choch_detected', 0)" in body
    assert "out['tf4h_choch_flag'] = features.get('tf4h_choch_flag', 0)" in body
    # the old unconditional zeroing must be gone
    assert "out['tf1h_choch_detected'] = 0" not in body
    assert "out['tf4h_choch_flag'] = 0" not in body


def test_smc_features_still_computes_choch():
    """The real derivation (new_trend != previous_trend on a current-bar
    break) must remain in _smc_features for both timeframes."""
    body = _fn_body("_smc_features")
    assert body.count("new_trend") >= 2 and body.count("previous_trend") >= 2
    assert "out['tf1h_choch_detected'] = 1 if any(" in body
    assert "out['tf4h_choch_flag'] = 1 if any(" in body


def test_update_ordering_smc_before_extra():
    """The pass-through only works if _smc_features runs before
    _extra_archetype_features in update()."""
    smc = SRC.index("features.update(self._smc_features())")
    extra = SRC.index("features.update(self._extra_archetype_features(features))")
    assert smc < extra


def test_module_parses():
    ast.parse(SRC)
