"""Regression test: macro_regime must reflect regime_label, not the default.

_extra_archetype_features (update step Q) copies regime_label before
_regime_detection (step D) has produced it — pinning macro_regime to
'neutral' on every live bar for 33+ days (feature-health sweep,
2026-07-21). update() now re-derives macro_regime after regime detection.
"""
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = (REPO / "bin/live/live_feature_computer.py").read_text()


def test_macro_regime_rederived_after_regime_detection():
    detect = SRC.index("features.update(self._regime_detection(features))")
    rederive = SRC.index(
        "features['macro_regime'] = features.get('regime_label', 'neutral')")
    assert rederive > detect, "macro_regime must be set AFTER regime detection"


def test_early_copy_still_harmless():
    """The step-Q placeholder read may remain, but the post-detection
    assignment must be the LAST write of macro_regime in update()'s flow."""
    last_write = SRC.rindex("'macro_regime'")
    # the final occurrence must be the re-derivation, not the placeholder
    ctx = SRC[last_write - 200:last_write + 120]
    assert "regime_label" in ctx
