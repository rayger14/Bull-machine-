"""Fixture integrity, not detector recognition or profitable outcomes."""
import copy
import importlib.util
from collections import Counter

import pytest


def api():
    name = 'scripts.research.wyckoff_recognition_cases'
    assert importlib.util.find_spec(name) is not None, 'recognition packet API not built'
    return __import__(name, fromlist=['build_packet'])


def test_twelve_balanced_deterministic_inputs_and_round_trip(tmp_path):
    m = api()
    p = m.build_packet()
    assert Counter(c['category'] for c in p['cases']) == {
        'positive': 4, 'near_miss': 4, 'ambiguous': 4}
    assert len({c['id'] for c in p['cases']}) == 12
    assert m.digest(p) == m.digest(m.build_packet())
    m.validate_packet(p)
    out = tmp_path / 'packet.json'
    m.write_packet(out, p)
    assert m.read_packet(out) == p
    with pytest.raises(FileExistsError):
        m.write_packet(out, p)


def test_mirrored_distribution_preserves_range_and_volume():
    p = api().build_packet()
    for bull, bear in [(p['cases'][0], p['cases'][2]), (p['cases'][1], p['cases'][3])]:
        for a, b in zip(bull['candles'], bear['candles']):
            assert b[1:5] == pytest.approx([210-a[1], 210-a[3], 210-a[2], 210-a[4]])
            assert b[5] == a[5]
        assert bull['direction'] == 'long'
        assert bear['direction'] == 'short'


@pytest.mark.parametrize('mutation', ['bad_ohlc','nan','negative_volume','duplicate',
                                    'future_checkpoint','injected_flag','missing_case'])
def test_packet_rejects_invalid_or_pretagged_inputs(mutation):
    m = api()
    p = copy.deepcopy(m.build_packet())
    c = p['cases'][0]
    if mutation == 'bad_ohlc': c['candles'][0][2] = 0
    if mutation == 'nan': c['candles'][0][4] = float('nan')
    if mutation == 'negative_volume': c['candles'][0][5] = -1
    if mutation == 'duplicate': c['candles'][1][0] = c['candles'][0][0]
    if mutation == 'future_checkpoint': c['checkpoints'][-1]['n'] = len(c['candles'])+1
    if mutation == 'injected_flag': c['candles'][0].append(True)
    if mutation == 'missing_case': p['cases'].pop()
    with pytest.raises(ValueError):
        m.validate_packet(p)
