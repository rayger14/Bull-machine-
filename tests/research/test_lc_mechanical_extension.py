"""Tests protect causal plan construction, retained failures and book grouping."""
import importlib
from copy import deepcopy

import pandas as pd
import pytest


def api():
    name = 'scripts.research.lc_mechanical_extension'
    assert importlib.util.find_spec(name), 'mechanical extension missing'
    return importlib.import_module(name)


def fixture():
    bars = pd.DataFrame(dict(open=100., high=101., low=99., close=100.),
        index=pd.date_range('2025-12-31T22:00Z', '2026-01-02T00:00Z', freq='min'))
    row = dict(candidate_id='case1', decision_time='2026-01-01T00:00:00+00:00',
               features=dict(open=100.,high=101.,low=99.,close=100.,atr_14=1.),
               previous_features=dict(open=100.,high=101.,low=99.,close=100.),
               native_diagnostic=dict(native_signal=dict(direction='long')))
    return row, bars


def test_freezes_stop_last_five_minutes_and_ignores_future():
    row,bars=fixture(); bars.loc['2025-12-31T23:58Z','high']=103.
    row['features']['high']=103.
    a=api().prepare_cases([row],bars)
    changed=bars.copy(); changed.loc['2026-01-01':,'high']=999.
    assert api().prepare_cases([row],changed)==a
    assert a[0]['plans']['immediate']['stop']==97.3
    assert a[0]['plans']['mechanical_wait']['level']==103.
    assert a[0]['plans']['immediate']['exit_deadline']=='2026-01-02T00:00:00+00:00'


@pytest.mark.parametrize('kind',['gap','mismatch','bad_atr'])
def test_bad_predecision_evidence_retains_unavailable_case(kind):
    row,bars=fixture()
    if kind=='gap': bars=bars.drop(pd.Timestamp('2025-12-31T23:58Z'))
    if kind=='mismatch': row['features']['close']=102.
    if kind=='bad_atr': row['features']['atr_14']=float('nan')
    case=api().prepare_cases([row],bars)[0]
    assert case['candidate_id']=='case1'
    assert case['plans'] is None and case['unavailable_reason']


def test_duplicate_identity_aborts_instead_of_double_counting():
    row,bars=fixture()
    with pytest.raises(ValueError,match='duplicate'):
        api().prepare_cases([row,deepcopy(row)],bars)


@pytest.mark.parametrize('cl,hi,lo,expected',[
    (102.,103.,99.,'upside_expansion_candidate'),
    (98.,101.,97.,'downside_rebound_candidate'),
    (100.,101.,98.,'downside_rebound_candidate'),
    (100.,101.,99.,'unresolved'),
])
def test_existing_subtype_geometry(cl,hi,lo,expected):
    row,bars=fixture()
    bars.loc['2025-12-31T23:00Z':'2025-12-31T23:59Z',['open','high','low','close']]=[cl,hi,lo,cl]
    row['features'].update(open=cl,high=hi,low=lo,close=cl)
    assert api().prepare_cases([row],bars)[0]['subtype']==expected


def test_costs_groups_and_unchanged_input():
    row,bars=fixture(); cases=api().prepare_cases([row],bars); before=deepcopy(cases)
    result=api().score_extension(cases,bars)
    assert cases==before
    primary=result['scenarios']['12bps_90s']
    assert primary['immediate']['summary']['policy_net_pnl']==-60.
    assert primary['mechanical_wait']['summary']['policy_net_pnl']==0.
    assert primary['immediate']['by_quarter']['2026Q1']['known_net_subtotal']==-60.
    assert primary['immediate']['by_subtype']['unresolved']['candidate_count']==1
    assert result['scenarios']['24bps_90s']['immediate']['summary']['policy_net_pnl']==-120.
    assert result['execution_authorized'] is False


def test_missing_case_cannot_be_reported_as_full_zero_policy():
    row,bars=fixture(); cases=api().prepare_cases([row],bars.iloc[1:])
    result=api().score_extension(cases,bars)
    assert result['scenarios']['12bps_90s']['immediate']['summary']['policy_net_pnl'] is None
    assert result['scenarios']['12bps_90s']['immediate']['by_quarter']['2026Q1']['policy_net_pnl'] is None
