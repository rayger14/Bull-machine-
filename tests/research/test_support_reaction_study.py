from copy import deepcopy
import importlib
import importlib.util
import json

import pandas as pd
import pytest

from scripts.research.support_reaction import assess, origin_record
from scripts.research.thesis_contract import signed, verify
from scripts.research.thesis_source import sha
from tests.research.support_reaction_fixtures import fixture


def api():
    assert importlib.util.find_spec('scripts.research.support_reaction_study'), 'source study missing'
    return importlib.import_module('scripts.research.support_reaction_study')


def packet():
    raw, bars = fixture()
    return api().blind_packet(assess(origin_record(raw), bars, as_of=raw['deadline']))


def annotation(p):
    cid = next(iter(p['observations']))
    claims = {k: dict(state='known', text='Literal reviewer observation.', citations=[cid])
              for k in ('location', 'recovery', 'support', 'minute_anchors', 'invalidation', 'room')}
    return dict(packet_seal=p['seal'], cutoff=p['cutoff'], **claims,
                phase=dict(state='unclassified', text='Phase not established.', citations=[cid]),
                action=dict(state='wait', text='More evidence needed.', citations=[cid]),
                status='complete', citations=[cid])


def test_roster_is_twelve_fixed_hash_choices_without_outcome_selection():
    origins = []
    for b, start in enumerate(api().BLOCKS[:-1]):
        for i in range(5):
            origins.append(dict(id=f'raw-{b}-{i}', origin={'available_at': (pd.Timestamp(start)+pd.Timedelta(days=i+1)).isoformat()}))
    roster = api().benchmark_roster(origins)
    assert len(roster) == 12
    assert roster == api().benchmark_roster(list(reversed(origins)))
    assert len({r['id'] for r in roster}) == 12
    assert all(sum(r['block'] == b for r in roster) == 3 for b in range(4))


def test_blind_packet_contains_only_prefix_sources_and_no_policy_verdicts():
    p = packet(); verify(p)
    assert p['cutoff'] == '2024-01-02T06:09:00+00:00'
    text = json.dumps(p)
    for forbidden in ('decisions', 'entry_intents', 'net_pnl', 'supportive', 'contextual_evidence_supported', 'PROJECT.md'):
        assert forbidden not in text
    assert all(pd.Timestamp(e['available_at']) <= pd.Timestamp(p['cutoff']) for e in p['observations'].values())
    assert any(e.get('timeframe') == '1min' for e in p['observations'].values())
    assert not any(e['kind'].startswith('pivot_') for e in p['observations'].values())
    assert api().validate_annotation(p, annotation(p))['status'] == 'valid_schema_not_verified_judgment'


def test_no_recovery_packet_still_contains_prior_twenty_hour_context():
    raw, bars = fixture()
    origin_at = pd.Timestamp(raw['origin']['available_at'])
    bars.loc[origin_at:, ['open', 'high', 'low', 'close']] = [104., 108., 100., 104.]
    r = assess(origin_record(raw), bars, as_of=raw['deadline'])
    assert r['decisions']['B']['reason'] == 'recovery_expired'
    p = api().blind_packet(r)
    hours = [e for e in p['observations'].values() if e.get('timeframe') == '1h' and pd.Timestamp(e['available_at']) <= origin_at]
    assert len(hours) == 20
    assert min(pd.Timestamp(e['start']) for e in hours) == origin_at-pd.Timedelta('20h')
    bars = bars.drop(origin_at-pd.Timedelta('18h'))
    missing = api().blind_packet(assess(origin_record(raw), bars, as_of=raw['deadline']))
    pre = [e for e in missing['observations'].values() if e.get('timeframe') == '1h' and pd.Timestamp(e['available_at']) <= origin_at]
    assert len(pre) == 20
    assert sum(e['status'] == 'unknown' for e in pre) == 1


@pytest.mark.parametrize('bad', ['foreign', 'future', 'accumulation', 'outcome', 'hash', 'no_citation'])
def test_annotations_reject_leaks_and_unsupported_certainty(bad):
    p = packet(); a = annotation(p)
    if bad == 'foreign': a['location']['citations'] = ['missing']
    if bad == 'future':
        cid = next(iter(p['observations'])); p['observations'][cid]['available_at'] = '2030-01-01T00:00:00+00:00'
        p = signed(p); a['packet_seal'] = p['seal']
    if bad == 'accumulation': a['phase']['state'] = 'confirmed_accumulation'
    if bad == 'outcome': a['net_pnl'] = 100.
    if bad == 'hash': a['packet_seal'] = 'foreign'
    if bad == 'no_citation': a['support']['citations'] = []
    with pytest.raises(ValueError): api().validate_annotation(p, a)


def test_source_summary_retains_rejects_and_unknown_denominator():
    raw, bars = fixture()
    r = assess(origin_record(raw), bars, as_of=raw['deadline'])
    raw2, bars2 = fixture(ceiling=130.)
    # Different canonical parent ID is needed for a second independent raw key.
    from scripts.research.thesis_sequence import compile_episode
    raw2['parent']['id'] = 'smaller-room'
    raw2 = compile_episode(raw2, raw2['events'])
    r2 = assess(origin_record(raw2), bars2, as_of=raw2['deadline'])
    s = api().summarize([r, r2], [raw, raw2])
    assert s['raw_episodes'] == 2
    assert s['arms']['B']['statuses'] == {'intent': 1, 'rejected': 1}
    assert s['arms']['C']['intents'] == 1
    assert s['old_A_intents'] == 0
    assert s['independent_annotations_complete'] is False
    assert s['economic_outcomes_computed'] is False


def test_bounded_source_output_exclusive_and_hash_bound(tmp_path):
    dependency = tmp_path/'dependency'; dependency.write_text('input')
    target = tmp_path/'ok'
    out = api().bounded_source(target, {str(dependency): sha(dependency)},
                              lambda: {'summary.json': signed({'scope': 'synthetic_test'})})
    verify(out)
    assert out['status'] == 'completed_source_only'
    assert out['economic_outcomes_computed'] is False
    for name, digest in out['artifacts'].items(): assert sha(target/name) == digest
    with pytest.raises(FileExistsError): api().bounded_source(target, {}, lambda: {})


@pytest.mark.parametrize('failure', ['mutation', 'compute', 'output_cap'])
def test_failure_has_no_success_receipt_or_automatic_retry(tmp_path, failure):
    dependency = tmp_path/'dependency'; dependency.write_text('input')
    target = tmp_path/'failed'; attempts = []
    def compute():
        attempts.append(1)
        if failure == 'mutation': dependency.write_text('changed')
        if failure == 'compute': raise RuntimeError('deliberate')
        return {'huge.json': {'text': 'x'*50000}} if failure == 'output_cap' else {}
    with pytest.raises((ValueError, RuntimeError)):
        api().bounded_source(target, {str(dependency): sha(dependency)}, compute, max_bytes=32768)
    assert len(attempts) == 1
    assert (target/'failure.json').exists()
    assert not (target/'receipt.json').exists()


def test_input_preparation_is_inside_alarm_and_failure_receipt(tmp_path):
    import signal
    target = tmp_path/'prepare_failed'
    def prepare():
        assert 0 < signal.getitimer(signal.ITIMER_REAL)[0] <= api().MAX_SECONDS
        assert (target/'launch.json').exists()
        raise RuntimeError('preflight failed')
    with pytest.raises(RuntimeError, match='preflight failed'):
        api().bounded_source(target, {}, lambda: {}, prepare=prepare)
    assert (target/'failure.json').exists()
    assert not (target/'receipt.json').exists()


def test_cli_exposes_source_only_not_economics(tmp_path):
    assert importlib.util.find_spec('scripts.research.run_support_reaction_study'), 'CLI missing'
    cli = importlib.import_module('scripts.research.run_support_reaction_study')
    with pytest.raises(SystemExit) as error:
        cli.main(['economic', '--output', str(tmp_path/'never')])
    assert error.value.code == 2
    assert not (tmp_path/'never').exists()
