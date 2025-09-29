import pandas as pd
from bull_machine.modules.fusion.bojan_hook import apply_bojan

def test_bojan_feeds_structure_wyckoff_volume_layers():
    df = pd.DataFrame([dict(open=100, high=112, low=98, close=101)])
    base = {"structure":0.25, "wyckoff":0.25, "volume":0.25}
    cfg  = {"features":{"bojan":True}, "bojan":{"wick_magnet_threshold":0.0, "trap_body_min":0.0}}
    new, tele = apply_bojan(base, df, tf="4H", config=cfg, last_hooks={})
    assert new["structure"] > base["structure"]
    assert new["wyckoff"]   > base["wyckoff"]
    assert new["volume"]    > base["volume"]
    assert "bojan_applied" in tele and tele["bojan_applied"] > 0