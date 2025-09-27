import pandas as pd
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from bull_machine.strategy.wyckoff_m1m2 import compute_m1m2_scores

def _mk_series(n=220, start=100, step=0.4):
    return [start + i*step for i in range(n)]

def _mk_df(n=220, step=0.4, vol=1000):
    return pd.DataFrame({
        'open':[0]*n,
        'high':[0]*n,
        'low':[0]*n,
        'close':_mk_series(n, 100, step),
        'volume':[vol]*n
    })

def test_adaptive_bias_and_fib_only_path_allows_long():
    # HTF pronounced uptrend, LTF up with some momentum -> favors LONG + FibR
    ltf = _mk_df(240, step=0.5, vol=1500)
    htf = _mk_df(240, step=0.8, vol=2000)
    res = compute_m1m2_scores(ltf, '1H', htf, fib_scores={'fib_retracement':0.62,'fib_extension':0.20})
    assert res['side'] in ('long','neutral'), f"Expected long/neutral, got {res['side']}"
    print(f"✅ Adaptive Fib-only test passed: side={res['side']}")

if __name__ == "__main__":
    test_adaptive_bias_and_fib_only_path_allows_long()
    print("✅ All adaptive Fib-only tests passed")