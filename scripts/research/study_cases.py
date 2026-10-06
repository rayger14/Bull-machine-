"""Outcome-hidden, as-of projections of qualified source evidence (not assessors)."""
from copy import deepcopy
import json
import pandas as pd

from scripts.research.study_contract import finite_number, stable_id, utc_minute
from scripts.research.r3_census import TRANSITION_FIELDS
from scripts.research.replay_clock import digest

FORBIDDEN = {'pnl', 'net_pnl', 'mfe', 'mae', 'outcome', 'profit_factor', 'winner', 'loser', 'future_return'}
OP_FIELDS = ('id', 'family', 'instrument', 'data_stream_id', 'origin_time', 'child_first_open',
             'child_last_close', 'child_low', 'child_high', 'parent_lineage_id', 'parent_version_id')
PARENT_FIELDS = ('id', 'lineage_id', 'available_at', 'range_low', 'range_high', 'low_pivot_id', 'high_pivot_id',
                 'predecessor_version_id', 'creation_reason', 'formation_hour')
BAR_FIELDS = ('id', 'open_time', 'available_at', 'open', 'high', 'low', 'close', 'volume')
EVENT_BASE = {'id', 'opportunity_id', 'available_at', 'kind'}
EVENT_EXTRA = {'armed': {'parent_version_id'}, 'breakout': {'close', 'stop'},
               'retest': {'high', 'low', 'close'}, 'trigger': {'close', 'stop', 'breakout_time'},
               'cancelled': {'reason'}, 'expired': {'reason'}, 'unknown': {'reason'},
               'parent_up': {'transition_id', 'source_transition'}}
PIVOT_FIELDS = ('id', 'side', 'level', 'pivot_open', 'pivot_close', 'confirming_close', 'available_at',
                'instrument', 'data_stream_id', 'anchor_timeframe', 'pivot_n', 'evidence_id', 'supporting_anchor_ids')
TOP_FIELDS = {'schema', 'as_of', 'execution_authorized', 'opportunity', 'parent', 'child_bars',
              'events', 'unknown_context', 'availability_basis', 'source_citations'}
UNKNOWN = ['daily_context', 'fib_time_price', 'gann_timing', 'derivatives_receipts']


def project_case(census, opportunity_id, as_of):
    clock = utc_minute(as_of)
    op = next((o for o in census['opportunities'] if o['id'] == opportunity_id), None)
    if op is None or utc_minute(op['origin_time']) > clock:
        raise ValueError('opportunity not available as of requested clock')
    # Whitelist projection: never copy arbitrary controller or outcome metadata.
    event_fields = EVENT_BASE | set().union(*EVENT_EXTRA.values())
    evidence = op['source_evidence']
    pivots = [{k: deepcopy(p[k]) for k in PIVOT_FIELDS} for p in evidence['pivots']]
    missing = sorted({op['parent']['low_pivot_id'], op['parent']['high_pivot_id']} - {p['id'] for p in pivots})
    packet = {'schema': 'study-case-v1', 'as_of': clock.isoformat(), 'execution_authorized': False,
              'opportunity': {k: deepcopy(op[k]) for k in OP_FIELDS},
              'parent': {k: deepcopy(op['parent'][k]) for k in PARENT_FIELDS},
              'child_bars': [{k: deepcopy(b[k]) for k in BAR_FIELDS} for b in op['child_bars']],
              'events': [{k: deepcopy(e[k]) for k in event_fields if k in e}
                         for e in census['events'] if e['opportunity_id'] == opportunity_id
                         and utc_minute(e['available_at']) <= clock],
              'unknown_context': UNKNOWN.copy(),
              'source_citations': {'status': 'incomplete' if missing or not evidence['contract_id'] else 'resolved',
                                   'contract_id': evidence['contract_id'], 'missing_ids': missing, 'pivots': pivots},
              'availability_basis': 'historical_bar_close_assumption'}
    errors = validate_case(packet)
    if errors:
        raise ValueError('; '.join(errors))
    return packet


def validate_case(packet):
    errors = []

    def walk(value, path='packet'):
        if isinstance(value, dict):
            for key, item in value.items():
                if key.lower() in FORBIDDEN:
                    errors.append('forbidden outcome field: ' + path + '.' + key)
                walk(item, path + '.' + key)
        elif isinstance(value, list):
            for item in value:
                walk(item, path)

    walk(packet)
    try:
        def exact(obj, fields, name):
            if set(obj) != set(fields):
                errors.append('undeclared_or_missing_fields:' + name)

        exact(packet, TOP_FIELDS, 'packet')
        asof = utc_minute(packet['as_of'])
        op, parent = packet['opportunity'], packet['parent']
        exact(op, OP_FIELDS, 'opportunity')
        exact(parent, PARENT_FIELDS, 'parent')
        if packet['schema'] != 'study-case-v1' or op['family'] != 'R3':
            errors.append('invalid_schema_version_or_family')
        if packet['unknown_context'] != UNKNOWN or packet['availability_basis'] != 'historical_bar_close_assumption':
            errors.append('undeclared_context')
        if packet['execution_authorized'] is not False:
            errors.append('execution_must_be_disabled')
        if utc_minute(op['origin_time']) > asof:
            errors.append('future_opportunity')
        if utc_minute(parent['available_at']) >= utc_minute(op['child_first_open']):
            errors.append('parent_not_preexisting')
        if parent['id'] != op['parent_version_id'] or parent['lineage_id'] != op['parent_lineage_id']:
            errors.append('parent_reference_mismatch')
        raw_id = stable_id('r3-box', {k: op[k] for k in ('instrument', 'parent_lineage_id', 'parent_version_id',
                                                        'origin_time', 'child_first_open', 'child_last_close')})
        if raw_id != op['id']:
            errors.append('forged_opportunity_id')
        if len(packet['child_bars']) != 6:
            errors.append('six_complete_child_bars_required')
        previous = utc_minute(op['child_first_open'])
        for bar in packet['child_bars']:
            exact(bar, BAR_FIELDS, 'child_bar')
            opened, closed = utc_minute(bar['open_time']), utc_minute(bar['available_at'])
            if opened != previous or closed != opened + pd.Timedelta('5min') or opened.minute % 5:
                errors.append('incomplete_or_noncontiguous_child_bars')
            previous = closed
            if stable_id('r3-child-bar', {k: bar[k] for k in BAR_FIELDS if k != 'id'}) != bar['id']:
                errors.append('forged_child_bar_id')
            for key in ('open', 'high', 'low', 'close'):
                finite_number(bar[key], positive=True)
            if (not bar['low'] <= min(bar['open'], bar['close']) <= max(bar['open'], bar['close']) <= bar['high']
                    or finite_number(bar['volume']) < 0):
                errors.append('invalid_child_ohlcv')
        if packet['child_bars']:
            if (previous != utc_minute(op['child_last_close']) or previous != utc_minute(op['origin_time'])
                    or min(b['low'] for b in packet['child_bars']) != op['child_low']
                    or max(b['high'] for b in packet['child_bars']) != op['child_high']):
                errors.append('child_window_or_bounds_mismatch')
        if not (parent['range_low'] <= op['child_low'] < op['child_high'] <= parent['range_high']
                and op['child_high'] - op['child_low'] <= (parent['range_high'] - parent['range_low']) / 4):
            errors.append('invalid_nested_geometry')
        for row in packet['child_bars'] + packet['events']:
            if utc_minute(row['available_at']) > asof:
                errors.append('future_observation')
        for event in packet['events']:
            extra = EVENT_EXTRA.get(event['kind'])
            if extra is None:
                errors.append('unknown_event_kind')
                continue
            if event.get('reason') == 'parent_down':
                extra = extra | {'transition_id', 'source_transition'}
            exact(event, EVENT_BASE | extra, 'event')
            if stable_id('r3-event', {k: v for k, v in event.items() if k != 'id'}) != event['id']:
                errors.append('forged_event_id')
            if event['opportunity_id'] != op['id']:
                errors.append('foreign_event')
            if utc_minute(event['available_at']) < utc_minute(op['origin_time']):
                errors.append('event_precedes_opportunity')
            if 'source_transition' in event:
                transition = event['source_transition']
                exact(transition, TRANSITION_FIELDS, 'source_transition')
                if (transition['id'] != event['transition_id']
                        or transition['pre_lineage_id'] != op['parent_lineage_id']
                        or utc_minute(transition['available_at']) != utc_minute(event['available_at'])):
                    errors.append('invalid_transition_citation')
        citations = packet['source_citations']
        exact(citations, {'status', 'contract_id', 'missing_ids', 'pivots'}, 'source_citations')
        expected_missing = sorted({parent['low_pivot_id'], parent['high_pivot_id']} - {p['id'] for p in citations['pivots']})
        expected_status = 'incomplete' if expected_missing or not citations['contract_id'] else 'resolved'
        if citations['missing_ids'] != expected_missing or citations['status'] != expected_status:
            errors.append('incorrect_citation_qualification')
        if utc_minute(parent['available_at']) != utc_minute(parent['formation_hour']) + pd.Timedelta('1h'):
            errors.append('incorrect_parent_availability_clock')
        if citations['contract_id']:
            causal_parent = {'kind': 'parent_version', 'contract_id': citations['contract_id'],
                             'data_stream_id': op['data_stream_id'], 'source_hour': parent['formation_hour'],
                             **{k: parent[k] for k in ('lineage_id', 'predecessor_version_id', 'creation_reason',
                                                       'range_low', 'range_high', 'low_pivot_id', 'high_pivot_id')}}
            if digest(causal_parent) != parent['id']:
                errors.append('forged_parent_version_id')
            for event in packet['events']:
                if 'source_transition' not in event:
                    continue
                transition = event['source_transition']
                causal_transition = {'kind': 'parent_transition', 'contract_id': citations['contract_id'],
                                     'data_stream_id': op['data_stream_id'], **{k: transition[k] for k in
                                     ('source_hour', 'pre_version_id', 'post_version_id', 'post_state',
                                      'source_sweep_low', 'source_sweep_high', 'source_break_direction')}}
                if digest(causal_transition) != transition['id']:
                    errors.append('forged_parent_transition_id')
                if utc_minute(transition['available_at']) != utc_minute(transition['source_hour']) + pd.Timedelta('1h'):
                    errors.append('incorrect_parent_transition_clock')
                if transition['source_break_direction'] != ('up' if event['kind'] == 'parent_up' else 'down'):
                    errors.append('incorrect_parent_transition_direction')
        for pivot in citations['pivots']:
            exact(pivot, PIVOT_FIELDS, 'pivot')
            if (pivot['id'] not in {parent['low_pivot_id'], parent['high_pivot_id']}
                    or pivot['instrument'] != op['instrument'] or pivot['data_stream_id'] != op['data_stream_id']
                    or utc_minute(pivot['available_at']) > utc_minute(parent['available_at'])):
                errors.append('foreign_or_future_pivot')
            if citations['contract_id']:
                causal = {'kind': 'anchor_pivot', 'contract_id': citations['contract_id'],
                          **{k: pivot[k] for k in ('data_stream_id', 'instrument', 'side', 'pivot_open',
                                                   'pivot_close', 'confirming_close', 'level', 'evidence_id')}}
                if digest(causal) != pivot['id']:
                    errors.append('forged_pivot_id')
        json.dumps(packet, allow_nan=False)
    except (KeyError, TypeError, ValueError) as exc:
        errors.append('invalid_schema: ' + str(exc))
    return errors


def render_case(packet):
    errors = validate_case(packet)
    if errors:
        raise ValueError('; '.join(errors))
    op, parent = packet['opportunity'], packet['parent']
    events = '\n'.join('- ' + e['available_at'] + ' ' + e['kind'] + ' [' + e['id'] + ']' for e in packet['events'])
    return (f"R3 case {op['id']} as of {packet['as_of']}\n"
            f"Parent {parent['id']}: {parent['range_low']}–{parent['range_high']}\n"
            f"Child: {op['child_low']}–{op['child_high']}\n{events}\n"
            'Unknown: ' + ', '.join(packet['unknown_context']) + '\nExecution authorized: false\n')
