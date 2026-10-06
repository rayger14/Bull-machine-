"""Source-only practice preparation; synthetic BTC-like data, no model calls."""
import importlib
import json
from copy import deepcopy

import pandas as pd
import pytest

from scripts.research.lc_structure_packet import build_structure_packet
from scripts.research.lc_structure_proposal import validate_structure_proposal
from tests.research.lc_structure_fixtures import structure_source
from tests.research.test_lc_master_assessment import inputs


def api():
    name = 'scripts.research.lc_practice_sources'
    assert importlib.util.find_spec(name), 'practice source integration missing'
    return importlib.import_module(name)


def test_closed_candles_match_archive_and_policy_is_risk_capped():
    s = structure_source(); p = build_structure_packet(s); bars = inputs()[1]
    a = api()
    checked = a.verify_candles(p, bars.loc[bars.index < pd.Timestamp(p['decision_time'])])
    assert checked == {'status': 'verified', 'candles_compared': 110, 'errors': []}
    policy = a.practice_policy(p)
    assert policy['max_entry_price'] == 105.52
    assert policy['risk_budget_usd'] == 100
    assert policy['horizon_minutes'] == 1440
    assert policy['roundtrip_cost_bps'] == 12


@pytest.mark.parametrize('change', ['gap', 'price', 'volume', 'duplicate', 'nan'])
def test_archive_mismatch_is_not_accepted(change):
    p = build_structure_packet(structure_source()); bars = inputs()[1]
    at = pd.Timestamp('2026-01-01T03:59:00Z')
    if change == 'gap': bars = bars.drop(at)
    if change == 'price': bars.loc[at, 'close'] = 106.
    if change == 'volume': bars.loc[at, 'volume'] = 2.
    if change == 'duplicate': bars = pd.concat([bars, bars.loc[[at]]]).sort_index()
    if change == 'nan': bars.loc[at, 'high'] = float('nan')
    checked = api().verify_candles(p, bars)
    assert checked['status'] == 'unavailable' and checked['errors']


def test_mechanical_control_uses_nearest_higher_timeframe_level_not_distant_parent():
    s = structure_source(); p = build_structure_packet(s); a = api()
    control = a.mechanical_control(s, p, a.practice_policy(p))
    assert control['status'] == 'proposal'
    answer = json.loads(control['raw_response'])
    assert answer['decision'] == 'wait_proposal'
    assert answer['plan']['trigger']['level_id'] == 'bar:5m:11:high'
    assert p['levels'][answer['plan']['destination_level_id']]['price'] == 120.
    assert answer['plan']['stop_level_id'] == answer['plan']['invalidation_level_id']
    assert validate_structure_proposal(s, p, control['raw_response'], a.practice_policy(p))['status'] == 'valid_proposal'


def test_request_contains_no_old_trade_menu_and_supplies_exact_answer_bindings():
    p = build_structure_packet(structure_source()); a = api()
    request = a.structure_request(p, a.practice_policy(p))
    assert request['packet'] == p
    assert request['answer_bindings']['packet_sha256'] == p['seal']
    assert 'plan_menu' not in request and 'source_request' not in request
    assert request['response_schema']['plan']['trigger']['kinds'] == ['immediate', 'close_above']


def test_control_does_not_call_missing_mandatory_data_a_legitimate_no_setup():
    s = structure_source(lambda p: p['current'].update(validated=False))
    p = build_structure_packet(s)
    assert api().mechanical_control(s, p, api().practice_policy(p))['status'] == 'unavailable'
