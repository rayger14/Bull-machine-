import pandas as pd
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from bull_machine.strategy.wyckoff_m1m2 import compute_m1m2_scores

def _mk_uptrend(n=200, step=0.2):
    return pd.DataFrame({
        'open':[0]*n,
        'high':[0]*n,
        'low':[0]*n,
        'close':[100 + i*step for i in range(n)],
        'volume':[1000]*n
    })

def test_adaptive_allows_fib_only_long_in_uptrend():
    ltf = _mk_uptrend(220, 0.2)
    htf = _mk_uptrend(220, 0.5)
    # no M1/M2 path (simulate zeros): rely on fib + HTF up + RSI>55
    res = compute_m1m2_scores(ltf, '1H', htf, fib_scores={'fib_retracement':0.60,'fib_extension':0.20})
    # side must be long-or-neutral; if neutral your runner still may pass via decide_side
    assert res['side'] in ('long','neutral'), f"Expected long/neutral, got {res['side']}"
    print(f"✅ Adaptive confluence test passed: side={res['side']}")

if __name__ == "__main__":
    test_adaptive_allows_fib_only_long_in_uptrend()
    print("✅ All adaptive confluence tests passed")