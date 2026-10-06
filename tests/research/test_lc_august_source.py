"""A complete August census must not inherit the earlier two-case selector."""
from copy import deepcopy
import importlib

import pandas as pd
import pytest

from scripts.research.lc_campaign_source import EXPECTED_ARCHETYPES, SOURCE_SHA256
from tests.research.test_lc_source_population import row


def api():
    name = 'scripts.research.lc_august_source'
    assert importlib.util.find_spec(name), 'August source adapter missing'
    return importlib.import_module(name)


def inputs():
    clocks = pd.date_range('2026-07-02T01:00Z', '2026-09-01T00:00Z', freq='h')
    rows = [row(d.isoformat(), direction='long' if d in pd.to_datetime(
        ['2026-08-01T01:00Z', '2026-08-17T02:00Z']) else None, selected=False) for d in clocks]
    month = dict(month='2026-08', seed='2026-07-02T00:00Z', start='2026-08-01T00:00Z',
                 end_exclusive='2026-09-01T00:00Z', seed_days=30, hourly_rows=1464,
                 minute_rows=87840, hourly_input_hash='h', atr_contract={'formula_id': 'fixture'},
                 hourly_selected=[{'case_id': 'old-selected-only-one'}],
                 ledgers=[dict(manifest=dict(parameters=dict(anchor_timeframe=tf, pivot_n=3)))
                          for tf in ('4H', '1D')])
    prior = dict(months=[month], source_sha256=SOURCE_SHA256, data_stream_id='fixture')
    replay = dict(rows=rows, contract_id='fixture', blockers=['cold_start'],
                  source_manifest=dict(signals=dict(archetype_order=sorted(EXPECTED_ARCHETYPES))))
    provenance = dict(source_path='/fixture/archive', source_manifest={'files': {}},
                      code_manifest={'files': {}}, config_manifest={'files': {}}, runtime={})
    return prior, replay, provenance


def test_complete_native_population_ignores_old_half_month_selector():
    prior, replay, provenance = inputs()
    before = deepcopy(prior)
    source = api().assemble_august(prior, replay, 'h', provenance)
    assert [c['decision_time'] for c in source['candidates']] == [
        '2026-08-01T01:00:00+00:00', '2026-08-17T02:00:00+00:00']
    assert source['candidate_count'] == 2
    assert all(c['native_emitted'] is False for c in source['candidates'])
    assert set(source['parent_ledgers']) == {'4H_N3', '1D_N3'}
    assert prior == before
    assert source['pristine_holdout'] is False


@pytest.mark.parametrize('bad', ['input_hash', 'month', 'seed', 'row_count', 'clock_gap', 'archetypes', 'parent'])
def test_incompatible_reconstruction_cannot_publish_as_same_source(bad):
    prior, replay, provenance = inputs()
    h = 'h'
    if bad == 'input_hash': h = 'different'
    if bad == 'month': prior['months'][0]['month'] = '2026-09'
    if bad == 'seed': prior['months'][0]['seed'] = '2026-07-03T00:00Z'
    if bad == 'row_count': replay['rows'].pop()
    if bad == 'clock_gap': replay['rows'][1]['decision_time'] = replay['rows'][0]['decision_time']
    if bad == 'archetypes': replay['source_manifest']['signals']['archetype_order'].pop()
    if bad == 'parent': prior['months'][0]['ledgers'][1] = prior['months'][0]['ledgers'][0]
    with pytest.raises(ValueError):
        api().assemble_august(prior, replay, h, provenance)
