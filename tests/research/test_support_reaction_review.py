from copy import deepcopy
import importlib
import importlib.util
import json

import pytest

from scripts.research.support_reaction import assess, origin_record
from scripts.research.support_reaction_study import blind_packet
from scripts.research.thesis_contract import signed, verify
from tests.research.support_reaction_fixtures import fixture


def api():
    assert importlib.util.find_spec('scripts.research.support_reaction_review'), 'review module missing'
    return importlib.import_module('scripts.research.support_reaction_review')


def source():
    raw, bars = fixture()
    return blind_packet(assess(origin_record(raw), bars, as_of=raw['deadline']))


def answer(view):
    alias = view['metadata']['origin_candle_id']
    claims = {k: dict(state='known', text='Source fact.', citations=[alias])
              for k in ('location', 'recovery', 'support', 'minute_anchors', 'invalidation', 'room')}
    return dict(packet_seal=view['packet_seal'], cutoff=view['metadata']['cutoff'], status='complete',
                citations=[alias], **claims,
                phase=dict(state='unclassified', text='No complete phase sequence.', citations=[alias]),
                action=dict(state='wait', text='Do not infer a fill.', citations=[alias]))


def test_compact_view_round_trips_every_observation_and_metadata_without_derived_labels():
    p = source()
    v, mapping = api().compact(p)
    assert api().restore(v, mapping) == p
    assert len(mapping['aliases']) == len(p['observations']) + 1  # shared stream
    assert v['metadata']['cutoff'] == '2024-01-02T06:09:00+00:00'
    assert sum(len(t['rows']) for t in v['tables']) == len(p['observations'])
    assert 'decisions' not in v['metadata']


def test_compaction_rejects_future_observation_even_when_packet_is_resigned():
    p = source()
    next(iter(p['observations'].values()))['available_at'] = '2030-01-01T00:00:00Z'
    with pytest.raises(ValueError, match='future'):
        api().compact(signed(p))


@pytest.mark.parametrize('payload', [None, {}])
def test_compaction_preserves_unknown_or_empty_payload_instead_of_dropping_it(payload):
    p = source()
    cid = p['origin_candle_id']
    p['observations'][cid]['payload'] = payload
    p['observations'][cid]['status'] = 'unknown'
    p = signed(p)
    v, mapping = api().compact(p)
    assert api().restore(v,mapping) == p


def test_restoration_rejects_tampered_view_and_foreign_map():
    p = source(); v, mapping = api().compact(p)
    bad = deepcopy(v); bad['metadata']['original_stop'] = 1.
    with pytest.raises(ValueError): api().restore(signed(bad), mapping)
    badmap = deepcopy(mapping)
    badmap['aliases']['O001'] = 'foreign'
    with pytest.raises(ValueError): api().restore(v, signed(badmap))


def test_annotation_aliases_restore_to_real_source_citations():
    p = source(); v, mapping = api().compact(p)
    out = api().translate(p, v, mapping, answer(v))
    assert out['location']['citations'] == [p['origin_candle_id']]
    assert out['citations'] == [p['origin_candle_id']]
    bad = answer(v); bad['location']['citations'] = ['O99999']
    with pytest.raises(ValueError): api().translate(p, v, mapping, bad)


def test_review_gate_requires_each_fixed_case_and_explicit_adjudication():
    p = source(); v, mapping = api().compact(p)
    a = api().translate(p, v, mapping, answer(v))
    finding = dict(packet_seal=p['seal'], classification='judgment_difference',
                   notes='Reviewer waits; a mechanical trigger alone is not trader certification.',
                   citations=[p['origin_candle_id']])
    out = api().review_receipt([p], [a], [finding], source_receipt_sha='test', files={})
    verify(out)
    assert out['status'] == 'exploratory_economics_cleared'
    assert out['edge_demonstrated'] is False
    assert out['trader_certified'] is False
    for annotations, findings in (([], [finding]), ([a,a], [finding]), ([a], [])):
        with pytest.raises(ValueError):
            api().review_receipt([p], annotations, findings, source_receipt_sha='test', files={})
    finding['classification'] = 'material_blocker'
    blocked = api().review_receipt([p], [a], [finding], source_receipt_sha='test', files={})
    assert blocked['status'] == 'blocked'


def test_gate_rejects_empty_notes_foreign_citations_and_auto_schema_approval():
    p = source(); v, mapping = api().compact(p)
    a = api().translate(p, v, mapping, answer(v))
    for f in (dict(packet_seal=p['seal'], classification='valid_schema', notes='OK', citations=[]),
              dict(packet_seal=p['seal'], classification='no_material_issue', notes='', citations=[]),
              dict(packet_seal=p['seal'], classification='no_material_issue', notes='Checked', citations=['foreign'])):
        with pytest.raises(ValueError):
            api().review_receipt([p], [a], [f], source_receipt_sha='test', files={})


def test_review_answers_are_sequential_exclusive_and_chain_checked(tmp_path):
    p = source(); v, mapping = api().compact(p)
    (tmp_path/'locked').mkdir()
    cases = []
    for n in ('01', '02'):
        api()._write(tmp_path/(n+'.json'), v)
        api()._write(tmp_path/(n+'.mapping.json'), mapping)
        cases.append(dict(case=n, packet_seal=p['seal'], view_sha=api().sha(tmp_path/(n+'.json')),
                          mapping_sha=api().sha(tmp_path/(n+'.mapping.json'))))
    api()._write(tmp_path/'manifest.json', signed(dict(cases=cases)))
    draft = tmp_path/'draft.json'; draft.write_text(json.dumps(answer(v)))
    with pytest.raises(FileNotFoundError): api().lock(tmp_path, '02', draft)
    first = api().lock(tmp_path, '01', draft)
    with pytest.raises(FileExistsError): api().lock(tmp_path, '01', draft)
    assert api().lock(tmp_path, '02', draft)['previous'] == first['seal']
    packets,annotations,files=api().collect_locked(tmp_path)
    assert len(packets)==len(annotations)==2
    assert str(tmp_path/'locked'/'01.receipt.json') in files
    (tmp_path/'locked'/'01.json').write_text('{}')
    with pytest.raises(ValueError, match='chain'): api().lock(tmp_path, '02', draft)
    with pytest.raises(ValueError, match='chain'): api().collect_locked(tmp_path)
