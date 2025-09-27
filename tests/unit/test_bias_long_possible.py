import pandas as pd
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from bull_machine.strategy.wyckoff_m1m2 import compute_m1m2_scores

def test_long_side_possible_with_uptrend():
    # simple uptrend HTF, modest LTF up drift
    htf = pd.DataFrame({'close':[100+i*0.5 for i in range(120)],
                        'open':[0]*120,'high':[0]*120,'low':[0]*120,'volume':[1000]*120})
    ltf = pd.DataFrame({'close':[100+i*0.2 for i in range(120)],
                        'open':[0]*120,'high':[0]*120,'low':[0]*120,'volume':[1000]*120})
    scores = compute_m1m2_scores(ltf, '1H', htf, fib_scores={'fib_retracement':0.6,'fib_extension':0.2})
    assert scores['side'] in ('long','neutral')  # must allow long in an uptrend

if __name__ == "__main__":
    test_long_side_possible_with_uptrend()
    print("✅ LONG bias unit test passed")