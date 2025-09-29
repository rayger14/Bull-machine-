import pandas as pd
from bull_machine.modules.bojan.bojan import compute_bojan_score

def test_bojan_basic_smoke():
    """Basic smoke test to confirm Bojan wiring"""
    df = pd.DataFrame([{"open": 100, "high": 110, "low": 95, "close": 105, "volume": 1000}])
    result = compute_bojan_score(df)
    assert isinstance(result, dict)
    assert "bojan_score" in result
    assert "signals" in result
    assert result["bojan_score"] >= 0.0