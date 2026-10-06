"""Equivalent clock formatting must not invalidate saved-example parity."""
from copy import deepcopy
import importlib

import pytest


def api():
    name = 'scripts.research.lc_august_receipt'
    assert importlib.util.find_spec(name), 'source receipt adapter missing'
    return importlib.import_module(name)


def samples():
    new = dict(candidate_id='hourly-lc:2026-08-10T14:00:00+00:00',
        decision_time='2026-08-10T14:00:00+00:00', setup_open='2026-08-10T13:00:00+00:00',
        track='hourly', features={'close':100.}, native_emitted=True)
    old = deepcopy(new)
    old.update(decision_time='2026-08-10 14:00:00+00:00',
               setup_open='2026-08-10 13:00:00+00:00', case_id=new['candidate_id'], stratum='H1')
    return dict(candidates=[new]), dict(hourly_selected=[old])


def test_equivalent_clocks_do_not_change_source_parity():
    source, month = samples(); before = deepcopy(month)
    assert api().verify_selected(source, month) == 1
    assert month == before


@pytest.mark.parametrize('bad', ['missing', 'price', 'clock', 'naive', 'native'])
def test_real_candidate_drift_still_fails_with_field_evidence(bad):
    source, month = samples()
    if bad == 'missing': source['candidates'] = []
    if bad == 'price': source['candidates'][0]['features']['close'] = 101.
    if bad == 'clock': source['candidates'][0]['decision_time'] = '2026-08-10T15:00:00+00:00'
    if bad == 'naive': source['candidates'][0]['decision_time'] = '2026-08-10T14:00:00'
    if bad == 'native': source['candidates'][0]['native_emitted'] = False
    with pytest.raises(ValueError, match='parity'):
        api().verify_selected(source, month)
