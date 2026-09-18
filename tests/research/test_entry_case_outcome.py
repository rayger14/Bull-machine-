"""Hand-calculated single-case bracket scorer tests, no model or market fixtures."""
import importlib
from pathlib import Path

import pandas as pd
import pytest


def run(rows, stop=90, horizon=2, **kw):
    assert Path('scripts/research/entry_case_outcome.py').exists(), 'scorer missing'
    m=importlib.import_module('scripts.research.entry_case_outcome')
    bars=pd.DataFrame(rows,columns=['open','high','low','close'],
                      index=pd.date_range('2026-01-01',periods=len(rows),freq='min',tz='UTC'))
    return m.score_case(bars,decision_time=bars.index[0],step_minutes=1,
                        horizon_bars=horizon,stop=stop,notional=1000,cost_bps=10,**kw)


def test_entry_bar_stop():
    r=run([[100,105,89,99],[99,100,98,99],[101,150,1,100]])
    assert r['exit_reason']=='stop' and r['exit_price']==90
    assert r['initial_risk']==100 and r['gross_pnl']==-100 and r['net_pnl']==-101
    assert r['horizon_return']==.01
    assert r['mfe']==.05 and r['mae']==-.11


def test_target_and_conservative_ambiguity():
    r=run([[100,121,95,110],[110,115,105,112],[110,120,90,110]])
    assert r['exit_reason']=='target' and r['exit_price']==120 and r['net_pnl']==199
    r=run([[100,121,89,110],[110,115,105,112],[110,120,90,110]])
    assert r['exit_reason']=='stop' and r['ambiguous_bar'] is True and r['net_pnl']==-101


def test_adverse_and_favorable_gaps():
    r=run([[100,105,95,101],[85,90,80,86],[86,90,85,87]])
    assert r['exit_price']==85 and r['net_pnl']==-151
    r=run([[100,105,95,101],[125,130,124,126],[126,130,120,127]])
    assert r['exit_price']==120 and r['ambiguous_bar'] is False


def test_open_target_gap_precedes_later_same_bar_stop():
    r=run([[100,105,95,101],[125,130,80,126],[126,130,120,127]])
    assert r['exit_price']==120 and r['exit_reason']=='target' and not r['ambiguous_bar']


def test_deadline_open_ignores_deadline_range():
    r=run([[100,105,95,101],[101,110,96,105],[106,140,80,100]])
    assert r['exit_reason']=='deadline' and r['exit_price']==106 and r['net_pnl']==59
    assert r['mfe']==.10 and r['mae']==-.05


def test_deadline_malformed_high_low_close_are_not_consumed():
    r=run([[100,105,95,101],[101,110,96,105],[106,float('nan'),-1,'bad']])
    assert r['exit_price']==106 and r['net_pnl']==59
    assert r['mfe']==.10 and r['mae']==-.05


def test_excursions_include_new_extremes_after_bracket_exit():
    r=run([[100,105,89,99],[99,140,70,110],[110,200,1,100]])
    assert r['exit_reason']=='stop' and r['net_pnl']==-101
    assert r['mfe']==.40 and r['mae']==-.30


def test_invalid_plan_is_retained():
    r=run([[100,105,95,101],[101,110,96,105],[106,140,80,100]],stop=100)
    assert r['status']=='invalid_plan' and r['net_pnl'] is None


def test_missing_tail_rejected():
    with pytest.raises(ValueError): run([[100,105,95,101],[101,110,96,105]])


def test_gap_and_bad_prices_rejected():
    m=importlib.import_module('scripts.research.entry_case_outcome')
    bars=pd.DataFrame([[100,105,95,101]]*3,columns=['open','high','low','close'],
                      index=pd.to_datetime(['2026-01-01T00:00Z','2026-01-01T00:02Z','2026-01-01T00:03Z']))
    with pytest.raises(ValueError):m.score_case(bars,decision_time=bars.index[0],step_minutes=1,horizon_bars=2,stop=90)
    with pytest.raises(ValueError):run([[100,90,95,101],[101,110,96,105],[106,140,80,100]])


@pytest.mark.parametrize('key,value',[('step_minutes',0),('step_minutes',True),('horizon_bars',0),('horizon_bars',1.5),('stop',float('nan')),('cost_bps',-1),('notional',0)])
def test_invalid_parameters(key,value):
    m=importlib.import_module('scripts.research.entry_case_outcome')
    bars=pd.DataFrame([[100,105,95,101]]*3,columns=['open','high','low','close'],
                      index=pd.date_range('2026-01-01',periods=3,freq='min',tz='UTC'))
    kw=dict(decision_time=bars.index[0],step_minutes=1,horizon_bars=2,stop=90)
    kw[key]=value
    with pytest.raises(ValueError):m.score_case(bars,**kw)
