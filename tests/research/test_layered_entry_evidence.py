"""Catch future/partial candle exposure and unsupported detector witnesses."""
import importlib
from pathlib import Path

import pandas as pd
import pytest


def module():
    assert Path('scripts/research/layered_entry_evidence.py').exists(), 'evidence helper missing'
    return importlib.import_module('scripts.research.layered_entry_evidence')


def bars(n=200):
    b=pd.DataFrame(dict(open=[110.]*n,high=[112.]*n,low=[109.]*n,
                        close=[111.]*n,volume=[1.]*n),
                   index=pd.date_range('2026-01-01',periods=n,freq='min',tz='UTC'))
    return b


def setup():
    b=bars()
    for k in ['open','high','low','close']:b[k]+=pd.Series([i*.01 for i in range(len(b))],index=b.index)
    b.loc[b.index[20],'low']=100.
    b.loc[b.index[60],'low']=100.
    b.loc[b.index[80],['open','high','low','close']]=[101.,102.,99.,99.5]
    b.loc[b.index[81],['open','high','low','close']]=[99.5,102.,99.4,101.]
    e=dict(pivot_idx=60,confirmed_idx=75,sweep_idx=80,reclaim_idx=81,
           level=100.,sweep_low=99.,touches=2)
    return b,e


def test_complete_bucket_ignores_developing_and_future_prices():
    m=module();b=bars(20)
    b.loc[b.index[10]:,['high','close']]=999.
    rows=m.completed_candles(b,b.index[12],5,2)
    assert rows==[[b.index[0].isoformat(),110.,112.,109.,111.,5.],
                  [b.index[5].isoformat(),110.,112.,109.,111.,5.]]


def test_exact_close_includes_just_completed_bucket():
    m=module();b=bars(20)
    assert m.completed_candles(b,b.index[10],5,1)[0][0]==b.index[5].isoformat()


@pytest.mark.parametrize('bad',['gap','duplicate','nan_volume','off_grid','naive'])
def test_invalid_selected_history_is_not_filled(bad):
    m=module();b=bars(20);at=b.index[10]
    if bad=='gap':b=b.drop(b.index[2])
    elif bad=='duplicate':b=pd.concat([b.iloc[:3],b.iloc[2:]])
    elif bad=='nan_volume':b.loc[b.index[2],'volume']=float('nan')
    elif bad=='off_grid':b.index=b.index+pd.Timedelta(seconds=1)
    else:b.index=b.index.tz_localize(None)
    with pytest.raises(ValueError):m.completed_candles(b,at,5,2)


def test_witness_reconstructs_pivot_confirmation_and_prior_touch():
    m=module();b,e=setup();w=m.minute_setup_witness(b,e,b.index[82])
    assert w['pivot']['open_time']==b.index[60].isoformat()
    assert w['pivot']['available_at']==b.index[76].isoformat()
    assert w['prior_touches'][0]['open_time']==b.index[20].isoformat()
    assert w['prior_touches'][0]['available_at']==b.index[36].isoformat()
    assert w['first_sweep']['open_time']==b.index[80].isoformat()
    assert w['reclaim']['open_time']==b.index[81].isoformat()
    assert w['sweep_low']==99. and w['touches']==2


def test_witness_never_uses_rows_at_or_after_decision():
    m=module();b,e=setup();a=m.minute_setup_witness(b,e,b.index[82])
    b.loc[b.index[82]:,'low']=float('nan')
    assert m.minute_setup_witness(b,e,b.index[82])==a


@pytest.mark.parametrize('key,value',[('level',100.1),('touches',3),('sweep_idx',79),('confirmed_idx',74),('sweep_low',98.)])
def test_forged_detector_record_rejected(key,value):
    m=module();b,e=setup();e[key]=value
    with pytest.raises(ValueError):m.minute_setup_witness(b,e,b.index[82])


def test_unclosed_reclaim_is_unavailable():
    m=module();b,e=setup()
    with pytest.raises(ValueError):m.minute_setup_witness(b,e,b.index[81])


def test_numeric_strings_must_not_be_aggregated_lexically():
    m=module();b=bars(20)
    for k in ['open','high','low','close']:b[k]=b[k].astype(str)
    with pytest.raises(ValueError):m.completed_candles(b,b.index[10],5,2)


def test_witness_rejects_string_prices_before_detector_arithmetic():
    m=module();b,e=setup()
    for k in ['open','high','low','close']:b[k]=b[k].astype(str)
    with pytest.raises(ValueError):m.minute_setup_witness(b,e,b.index[82])
