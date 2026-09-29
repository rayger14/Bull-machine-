"""Catch future evidence, invented levels and silently dropped provenance."""
from copy import deepcopy
import importlib
import json

import pytest

from scripts.research.conditional_assessment import digest
from tests.research.lc_structure_fixtures import structure_source


def api():
    name = 'scripts.research.lc_structure_packet'
    assert importlib.util.find_spec(name), 'structure packet API missing'
    return importlib.import_module(name)


def test_projection_prices_provenance_and_no_mutation():
    source = structure_source(); before = deepcopy(source)
    p = api().build_structure_packet(source)
    assert source == before
    assert p['levels']['bar:5m:11:high']['price'] == 110.
    assert p['levels']['bar:5m:11:low']['price'] == 99.
    assert p['levels']['bar:5m:11:high']['available_at'] == '2026-01-01T04:00:00+00:00'
    assert p['levels']['bar:5m:10:high']['price'] == 110.
    assert p['levels']['bar:5m:11:high']['kind'] == 'candle_extreme'
    assert p['execution_authorized'] is False
    assert not {'plan','plan_menu','indicative_economics'} & p.keys()
    assert 'ceiling_distance_r' not in json.dumps(p)
    assert 'funding_rate' not in json.dumps(p['context'])
    for path in p['citation_catalog'].values():
        v = p
        for key in path: v = v[key]
        assert v is not None
    api().validate_structure_packet(source, p)


def test_resealed_invented_level_fails_source_reconstruction():
    source = structure_source(); p = api().build_structure_packet(source)
    p['levels']['bar:5m:11:high']['price'] = 111.
    p['seal'] = digest({k:v for k,v in p.items() if k != 'seal'})
    with pytest.raises(ValueError): api().validate_structure_packet(source, p)


@pytest.mark.parametrize('mutation', ['future','naive','duplicate','reverse','ohlc','nan',
                                     'bool','unknown_tf','columns','seal'])
def test_invalid_source_or_candles_fail(mutation):
    def change(p):
        rows = p['evidence']['15m']
        if mutation == 'future': rows[-1][0] = '2026-01-01T04:00:00Z'
        if mutation == 'naive': rows[-1][0] = '2026-01-01T03:45:00'
        if mutation == 'duplicate': rows[-1][0] = rows[-2][0]
        if mutation == 'reverse': rows.reverse()
        if mutation == 'ohlc': rows[-1][2] = 1.
        if mutation == 'nan': rows[-1][2] = float('nan')
        if mutation == 'bool': rows[-1][2] = True
        if mutation == 'unknown_tf': p['evidence']['7m'] = []
        if mutation == 'columns': p['candle_columns'].append('extra')
    module = api()
    with pytest.raises(ValueError):
        s = structure_source(change)
        if mutation == 'seal': s['seal'] = '0'*64
        module.build_structure_packet(s)


def test_gaps_and_missing_timeframe_are_reported_without_filling():
    def change(p):
        p['evidence']['15m'].pop(3)
        p['evidence']['1d'] = []
    p = api().build_structure_packet(structure_source(change))
    assert len(p['candles']['15m']) == 15
    assert p['candles']['1d'] == []
    assert 'gap:15m' in p['limitations']['projection']
    assert 'missing:1d' in p['limitations']['projection']


@pytest.mark.parametrize('state,expected', [('absent','absent'),('unknown','unknown'),
                                          ('future','unknown'),('broken','broken_down')])
def test_parent_states_not_conflated(state, expected):
    def change(p):
        parent = p['evidence']['parent_4h']
        if state in ('absent','unknown'):
            parent.update(status='fail' if state=='absent' else 'unknown', bound=None,
                          pivots=[], reasons=['absent_parent' if state=='absent' else 'insufficient_history'],
                          lineage_broken=False)
        if state == 'future': parent['bound']['available_at'] = p['decision_time']
        if state == 'broken':
            parent.update(status='fail', lineage_broken=True, reasons=['bound_lineage_broken'])
            parent['updates'][-1]['source_break_direction'] = 'down'
    p = api().build_structure_packet(structure_source(change))
    assert p['context']['parent_4h']['lifecycle'] == expected
    assert any(k.startswith('parent:4h:') for k in p['levels']) is (state=='broken')


def test_unvalidated_current_stays_unknown_not_a_new_fact():
    p = api().build_structure_packet(structure_source(lambda p:p['current'].update(validated=False)))
    assert p['context']['hourly']['status'] == 'unknown'


@pytest.mark.parametrize('location', ['current','bound','update'])
def test_source_extensions_cannot_leak_into_outcome_hidden_context(location):
    def change(value):
        targets = {'current':value['current']['source_candle'],
                   'bound':value['evidence']['parent_4h']['bound'],
                   'update':value['evidence']['parent_4h']['updates'][0]}
        targets[location].update(future_outcome={'future_high':999}, free_prompt='synthetic extension')
    s=structure_source(change);p=api().build_structure_packet(s)
    # Assert booleans so a failure cannot pretty-print the entire large packet.
    for marker in ('future_outcome','free_prompt','synthetic extension'):
        leaked = marker in json.dumps(p)
        assert leaked is False, 'source extension leaked: ' + marker
    api().validate_structure_packet(s,p)
