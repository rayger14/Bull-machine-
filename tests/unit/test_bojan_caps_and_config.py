import pandas as pd
from bull_machine.modules.fusion.bojan_hook import apply_bojan

def test_bojan_cap_applies_when_po3_present():
    df = pd.DataFrame([dict(open=100, high=110, low=95, close=105)])
    layer_scores = {"structure":0.30, "wyckoff":0.30, "volume":0.30}
    config = {"features":{"bojan":True}, "bojan":{"wick_magnet_threshold":0.0}}  # force wick bonus
    new, tele = apply_bojan(layer_scores, df, tf="1H", config=config, last_hooks={"po3_boost":0.10})
    assert new["structure"] >= layer_scores["structure"]  # boost happened
    # boost should be capped under PO3 overlap
    assert (new["structure"] - layer_scores["structure"]) <= 0.07 * 0.50 + 1e-9