"""Literal first-fill/cost/null examples; these are not historical results."""
import importlib
import json
from copy import deepcopy

import pandas as pd
import pytest

from scripts.research.conditional_assessment import digest
from tests.research.test_lc_structure_outcome import case, minute


def api():
    name = 'scripts.research.lc_practice_replay'
    assert importlib.util.find_spec(name), 'practice replay integration missing'
    return importlib.import_module(name)


def scenario(wait=False):
    s,p,policy,a,e,f = case(wait=wait)
    rows = deepcopy(e['completed_minutes']) + deepcopy(f['minutes'])
    return s,p,policy,a,rows


def replay(c, delay=90):
    s,p,policy,a,rows = c
    return api().replay_proposal(s,p,json.dumps(a),policy,rows,delay)


def test_first_fill_is_not_skipped_for_later_better_price():
    c = scenario(); c[4][2]['high'] = 111.
    c[4].append(minute('2026-01-01T04:06:00Z', 104., 111., 103., 110.))
    result = replay(c)
    assert result['status'] == 'filled'
    assert result['outcome']['entry_time'] == '2026-01-01T04:02:00+00:00'
    assert result['outcome']['net_pnl'] == pytest.approx(79.562520404832)
    assert result['outcome']['exit_reason'] == 'target'


@pytest.mark.parametrize('high,low,expected', [(109.,99.,'stop'), (110.,99.,'stop'), (110.,100.,'target')])
def test_entry_bar_uses_real_bracket_and_stop_first_tie(high,low,expected):
    c=scenario(); c[4][2].update(high=high,low=low)
    r=replay(c)
    assert r['status']=='filled' and r['outcome']['exit_reason']==expected
    if expected=='stop': assert r['net_pnl']==pytest.approx(-100.)


def test_wait_only_uses_first_strict_post_arm_close():
    c=scenario(True); c[4][3]['high']=120.
    r=replay(c)
    assert r['status']=='filled'
    assert r['outcome']['entry_time']=='2026-01-01T04:03:00+00:00'
    assert r['outcome']['entry_price']==111.


def test_fill_relative_timeout_and_delayed_availability():
    c=scenario(); rows=c[4]
    rows[-1] = minute('2026-01-01T04:05:00Z',106.,107.,105.,106.)
    rows.append(dict(open_time='2026-01-01T04:06:00Z',open=107.))
    r=replay(c,delay=180)
    assert r['outcome']['entry_time']=='2026-01-01T04:03:00+00:00'
    assert r['outcome']['exit_time']=='2026-01-01T04:06:00+00:00'


@pytest.mark.parametrize('decision,want,pnl', [('reject','rejected',0.), ('insufficient_evidence','unavailable',None)])
def test_deliberate_reject_is_zero_but_insufficient_is_null(decision,want,pnl):
    c=scenario(); a=c[3]; a['decision']=decision; a['plan']=None
    a['unknowns']=[dict(text='Evidence missing', evidence_ids=['limitations'])]
    r=replay(c)
    assert (r['status'],r['net_pnl'])==(want,pnl)


@pytest.mark.parametrize('bad', ['gap_before_fill','gap_after_fill','unknown_delay','invalid_json','unsupported_invalidation'])
def test_unknowns_never_become_successful_flat_trades(bad):
    c=scenario(); delay=90
    if bad=='gap_before_fill': del c[4][0]
    if bad=='gap_after_fill': del c[4][3]
    if bad=='unknown_delay': delay=None
    if bad=='invalid_json': c[3]['case_id']='wrong-case'
    if bad=='unsupported_invalidation': c[3]['plan']['invalidation_level_id']='bar:1h:0:low'
    r=replay(c,delay)
    assert r['net_pnl'] is None and r['status']=='unavailable'


def test_verified_expiry_and_preentry_cancel_have_zero_exposure():
    c=scenario(True)
    c[4][:]=[minute((pd.Timestamp(c[1]['decision_time'])+pd.Timedelta(minutes=i)).isoformat()) for i in range(15)]
    assert replay(c)['status']=='expired' and replay(c)['net_pnl']==0
    c[4][0]['low']=98.
    assert replay(c)['status']=='cancelled' and replay(c)['net_pnl']==0
    del c[4][0]
    assert replay(c)['net_pnl'] is None


def test_unknown_total_and_subtype_isolation_without_portfolio_competition():
    cases=[dict(case_id='A', subtype='downside', arms={'agent':{'net_pnl':10.,'status':'filled'},'mechanical_matched':{'net_pnl':5.,'status':'filled'}}),
           dict(case_id='B', subtype='upside', arms={'agent':{'net_pnl':None,'status':'unavailable'},'mechanical_matched':{'net_pnl':-5.,'status':'filled'}})]
    summary=api().summarize_cases(cases)
    assert summary['overall']['agent']['total_net_pnl'] is None
    assert summary['overall']['agent']['known_subtotal']==10.
    assert summary['by_subtype']['downside']['agent']['total_net_pnl']==10.
    assert summary['matched']['count']==1 and summary['matched']['known_delta']==5.
    assert summary['matched']['total_delta'] is None
    assert summary['attribution']['B']=='unavailable'


def test_each_subtype_has_matched_coverage_and_honest_winner_loser_attribution():
    def item(cid,subtype,a,b,status):
        return dict(case_id=cid,subtype=subtype,arms=dict(agent=dict(status=status,net_pnl=a),
            mechanical_matched=dict(status='filled',net_pnl=b)))
    rows=[item('A','rebound',0.,10.,'rejected'),item('B','rebound',None,-5.,'unavailable'),
          item('C','expansion',0.,-8.,'rejected'),item('D','expansion',6.,4.,'filled'),
          item('E','expansion',0.,3.,'expired')]
    result=api().summarize_cases(rows)
    rebound=result['by_subtype']['rebound']; expansion=result['by_subtype']['expansion']
    assert rebound['matched']==dict(count=1,roster_count=2,known_delta=-10.,total_delta=None)
    assert rebound['attribution_counts']=={'rejected_winner':1,'unavailable':1}
    assert expansion['matched']==dict(count=3,roster_count=3,known_delta=7.,total_delta=7.)
    assert expansion['attribution_counts']=={'rejected_loser':1,'winner_preserved':1,'expired_missed_winner':1}
