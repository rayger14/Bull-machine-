from copy import deepcopy

import pandas as pd
import pytest

from scripts.research.r3_census import build_r3_census, compile_r3_signals
from tests.research.study_fixtures import minutes, nested_minutes, parent_ledger, with_parent_break


def census(data=None, parents=None, checkpoint=None):
    data = nested_minutes() if data is None else data
    return build_r3_census(data, parents or parent_ledger(), instrument='BTC',
                           data_stream_id='fixture', emit_from='2024-01-01T00:00Z',
                           end_exclusive=data.index[-1] + pd.Timedelta('1min'),
                           checkpoint=checkpoint)


def first_events(result):
    oid = result['opportunities'][0]['id']
    return [e for e in result['events'] if e['opportunity_id'] == oid]


def test_nested_sequence_has_exact_clocks_and_frozen_levels():
    result = census()
    op = result['opportunities'][0]
    assert op['origin_time'] == '2024-01-01T00:30:00+00:00'
    assert (op['child_low'], op['child_high']) == (100., 102.)
    assert op['parent']['id'] == 'parent-v1'
    assert [(e['kind'], e['available_at'][11:16]) for e in first_events(result)] == [
        ('armed', '00:30'), ('breakout', '00:35'), ('retest', '00:40'), ('trigger', '00:41')]
    baseline = compile_r3_signals(result, 'baseline')[0]
    repair = compile_r3_signals(result, 'repair')[0]
    assert baseline['stop'] == 100
    assert repair['stop'] == 101.7
    assert repair['exit_deadline'] == baseline['exit_deadline'] == '2024-01-01T04:35:00+00:00'
    assert result['blockers'] == []


@pytest.mark.parametrize('cut', [30, 35, 40, 41, 59])
def test_restart_is_identical_and_retains_no_raw_history(cut):
    data = nested_minutes()
    full = census(data)
    first = census(data.iloc[:cut])
    resumed = census(data.iloc[cut:], checkpoint=first['checkpoint'])
    for key in ('opportunities', 'events', 'attempts'):
        assert first[key] + resumed[key] == full[key]
    assert resumed['checkpoint'] == full['checkpoint']
    assert len(first['checkpoint']['recent_5m']) <= 6
    assert len(first['checkpoint']['partial_5m']) <= 4
    assert 'minutes' not in first['checkpoint']
    assert not any(e['kind'] == 'right_censored' for e in first['events'])


def test_first_touch_failure_never_accepts_a_later_retest():
    data = nested_minutes()
    data.iloc[39, data.columns.get_loc('close')] = 102.
    result = census(data)
    assert first_events(result)[-1]['reason'] == 'first_retest_failed'
    assert not compile_r3_signals(result, 'repair')


def test_trigger_same_minute_stop_touch_cancels_first():
    data = nested_minutes()
    data.iloc[40, data.columns.get_loc('low')] = 101.7
    result = census(data)
    assert first_events(result)[-1]['reason'] == 'pre_entry_stop_touch'
    assert not compile_r3_signals(result, 'repair')


def test_down_uses_pre_lineage_at_availability_but_up_is_annotation():
    down = census(parents=with_parent_break(parent_ledger(), '2024-01-01T00:41Z'))
    assert first_events(down)[-1]['reason'] == 'parent_down'
    assert not compile_r3_signals(down, 'repair')
    up = census(parents=with_parent_break(parent_ledger(), '2024-01-01T00:41Z', 'up'))
    assert any(e['kind'] == 'parent_up' for e in first_events(up))
    assert compile_r3_signals(up, 'repair')


def test_parent_must_preexist_first_child_open_and_remain_not_down():
    parents = parent_ledger()
    for row in parents['versions'] + parents['transitions']:
        row['available_at'] = '2024-01-01T00:00:00Z'
    result = census(nested_minutes().iloc[:30], parents)
    assert result['opportunities'] == []
    result = census(nested_minutes().iloc[:30], with_parent_break(parent_ledger(), '2024-01-01T00:20Z'))
    assert not result['opportunities']
    assert result['attempts'][-1]['reason'] == 'parent_down_during_construction'


def test_reservation_not_shortened_by_failed_retest():
    data = nested_minutes()
    data.iloc[39, data.columns.get_loc('close')] = 102.
    result = census(data)
    # B=00:35, reservation ends01:15, six wholly new bars finish01:45.
    assert result['opportunities'][1]['origin_time'] == '2024-01-01T01:45:00+00:00'
    assert result['opportunities'][1]['child_first_open'] == '2024-01-01T01:15:00+00:00'


def test_no_break_expires_at_inclusive_final_close():
    result = census(minutes())
    assert [(e['kind'], e['available_at'][11:16]) for e in first_events(result)] == [
        ('armed', '00:30'), ('expired', '01:00')]
    assert result['opportunities'][1]['origin_time'][11:16] == '01:30'


def test_last_breakout_and_last_trigger_closes_are_included():
    data = minutes()
    cols = data.columns.get_indexer(['open', 'high', 'low', 'close'])
    data.iloc[55:60, cols] = [102.5, 103, 102.2, 102.8]
    data.iloc[60:65, cols] = [102.5, 103.1, 101.7, 102.5]
    data.iloc[65:69, cols] = [102.5, 103, 102.2, 102.8]
    data.iloc[69, cols] = [102.5, 103.5, 102.2, 103.2]
    result = census(data)
    signals = compile_r3_signals(result, 'repair')
    assert signals[0]['decision_time'][11:16] == '01:10'
    assert compile_r3_signals(result, 'baseline')[0]['decision_time'][11:16] == '01:00'


def test_gap_is_retained_as_blocker_and_does_not_create_fake_bars():
    result = census(nested_minutes().drop(nested_minutes().index[38]))
    assert result['blockers'][0]['reason'] == 'minute_gap'
    assert first_events(result)[-1]['kind'] == 'unknown'
    assert not compile_r3_signals(result, 'repair')


def test_future_append_cannot_change_prior_events_or_raw_opportunity():
    data = nested_minutes()
    first = census(data.iloc[:41])
    full = census(data)
    assert first['opportunities'] == full['opportunities'][:len(first['opportunities'])]
    assert first['events'] == [e for e in full['events'] if e['available_at'] <= '2024-01-01T00:41:00+00:00']


def test_last_retest_close_included_and_no_later_trigger_allowed():
    data = nested_minutes()
    cols = data.columns.get_indexer(['open', 'high', 'low', 'close'])
    data.iloc[35:60, cols] = [103., 103.3, 102.2, 103.1]
    data.iloc[60:65, cols] = [102.5, 103.1, 101.7, 102.5]
    data.iloc[65:71, cols] = [102.5, 103.0, 102.2, 102.8]
    data.iloc[70, cols] = [102.5, 103.5, 102.2, 103.2]
    result = census(data)
    assert [(e['kind'], e['available_at'][11:16]) for e in first_events(result)][-2:] == [
        ('retest', '01:05'), ('expired', '01:10')]
    assert not compile_r3_signals(result, 'repair')


def test_month_boundary_resume_and_frozen_parent_version():
    data = nested_minutes()
    data.index += pd.Timedelta('30d23h30min')
    parents = parent_ledger()
    version = deepcopy(parents['versions'][0])
    version.update(id='parent-v2', available_at='2024-02-01T00:05:00Z', range_high=109.)
    parents['versions'].append(version)
    parents['transitions'].append({'id': 'parent-tightened', 'available_at': '2024-02-01T00:05:00Z',
                                  'pre_state': 'active', 'post_state': 'active',
                                  'pre_version_id': 'parent-v1', 'post_version_id': 'parent-v2',
                                  'pre_lineage_id': 'lineage-1', 'post_lineage_id': 'lineage-1',
                                  'source_break_direction': None})
    full = census(data, parents)
    first = census(data.iloc[:30], parents)
    resumed = census(data.iloc[30:], parents, first['checkpoint'])
    assert first['events'] + resumed['events'] == full['events']
    assert full['opportunities'][0]['parent']['range_high'] == 110.
    assert full['opportunities'][0]['parent_version_id'] == 'parent-v1'


def test_parent_coverage_cannot_silently_end_during_setup():
    parents = parent_ledger()
    parents['coverage']['query_exclusive_end'] = '2024-01-01T00:40Z'
    with pytest.raises(ValueError, match='out_of_coverage'):
        census(parents=parents)


def test_foreign_or_malformed_parent_references_fail_closed():
    parents = parent_ledger()
    parents['manifest']['data_stream_id'] = 'foreign'
    with pytest.raises(ValueError, match='stream'):
        census(parents=parents)
    parents = parent_ledger()
    parents['transitions'][0]['post_version_id'] = 'absent'
    with pytest.raises(ValueError, match='version'):
        census(parents=parents)
    parents = parent_ledger()
    parents['versions'][0]['range_low'] = True
    with pytest.raises(ValueError):
        census(parents=parents)


def test_restart_rejects_overlap_changed_parent_prefix_or_stream():
    data = nested_minutes()
    first = census(data.iloc[:40])
    with pytest.raises(ValueError, match='contiguous'):
        census(data.iloc[39:], checkpoint=first['checkpoint'])
    parents = deepcopy(parent_ledger())
    parents['versions'][0]['range_high'] = 111.
    with pytest.raises(ValueError, match='prefix'):
        census(data.iloc[40:], parents, first['checkpoint'])


def test_restart_consumers_use_verified_merged_census():
    from scripts.research.r3_census import merge_censuses
    from scripts.research.study_cases import project_case
    data = nested_minutes()
    first = census(data.iloc[:40])
    suffix = census(data.iloc[40:], checkpoint=first['checkpoint'])
    merged = merge_censuses(first, suffix)
    full = census(data)
    assert compile_r3_signals(merged, 'repair') == compile_r3_signals(full, 'repair')
    oid = first['opportunities'][0]['id']
    assert project_case(merged, oid, '2024-01-01T00:41Z') == project_case(full, oid, '2024-01-01T00:41Z')
    with pytest.raises(ValueError, match='contiguous'):
        merge_censuses(first, full)
    cold_suffix = census(data.iloc[40:])
    with pytest.raises(ValueError, match='checkpoint'):
        merge_censuses(first, cold_suffix)
