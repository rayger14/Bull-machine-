"""Causal, source-bound LC evidence projection; never an execution authority.

The source is a caller-verified reconstruction, not authenticated live receipts.
Catalog membership is factual provenance, not proof of a trading interpretation.
"""
from copy import deepcopy
import math

import pandas as pd

from scripts.research.conditional_assessment import digest
from scripts.research.lc_context_assessment import validate_context_request


VERSION = 'lc_structure_packet_v1'
RESPONSE_KEYS = frozenset(('version', 'contract_sha256', 'case_id', 'packet_sha256',
    'curriculum_sha256', 'policy_sha256', 'decision', 'thesis', 'parent_child',
    'sequence', 'supporting', 'opposing', 'competing_explanation', 'unknowns',
    'plan', 'execution_authorized'))
PLAN_KEYS = frozenset(('trigger', 'confirmation_evidence_ids', 'invalidation_level_id',
    'invalidation_operator', 'stop_level_id', 'destination_level_id',
    'obstacle_level_ids', 'horizon_minutes', 'expiry_minutes'))
POLICY_KEYS = frozenset(('version', 'instrument', 'data_stream_id', 'max_entry_price',
    'entry_expiry_minutes', 'horizon_minutes', 'minimum_net_rr', 'risk_budget_usd',
    'max_notional_usd', 'equity_usd', 'max_leverage', 'roundtrip_cost_bps',
    'processing_seconds', 'routing_seconds', 'tick_size'))
DECISIONS = ('enter_proposal', 'wait_proposal', 'reject', 'insufficient_evidence')
THESES = ('downside_rebound', 'upside_expansion', 'unresolved')
CONTRACT = dict(versions=[VERSION, 'lc_structure_proposal_v1', 'lc_structure_policy_v1',
                         'lc_structure_preentry_v1'],
    response_keys=sorted(RESPONSE_KEYS), plan_keys=sorted(PLAN_KEYS),
    policy_keys=sorted(POLICY_KEYS), triggers=['immediate', 'close_above'],
    decisions=list(DECISIONS), theses=list(THESES))
CONTRACT_SHA256 = digest(CONTRACT)
PERIODS = {'1d':1440, '4h':240, '1h':60, '15m':15, '5m':5, '1m':1}
COLUMNS = ('open_time', 'open', 'high', 'low', 'close', 'volume')


def number(value, zero=False):
    try:
        return (type(value) in (int, float) and math.isfinite(value)
                and (value >= 0 if zero else value > 0))
    except OverflowError:
        return False


def clock(value):
    if not isinstance(value, str):
        raise ValueError('timestamp must be a timezone-aware string')
    try:
        t = pd.Timestamp(value)
        if pd.isna(t) or t.tzinfo is None:
            raise ValueError('naive or missing timestamp')
        return t.tz_convert('UTC')
    except (TypeError, OverflowError) as exc:
        raise ValueError('invalid timestamp') from exc


def prices(row):
    return (all(number(row.get(k)) for k in ('open', 'high', 'low', 'close'))
            and number(row.get('volume'), zero=True)
            and row['low'] <= min(row['open'], row['close'])
            and row['high'] >= max(row['open'], row['close']))


def resolve(packet, evidence_id):
    value = packet
    for key in packet['citation_catalog'][evidence_id]:
        value = value[key]
    return value


def _fields(value, names):
    """Nested source extensions are not part of the assessor contract."""
    if value is None:
        return None
    return {name: deepcopy(value[name]) for name in names.split()}


def _context(source):
    raw = source['context']
    context = _fields(raw, 'version case_id candidate_id source_packet_sha256 '
        'decision_time native_long execution_authorized source_authentication')
    context['hourly'] = _fields(raw['hourly'], 'status close_relation swept_prior_low '
        'swept_prior_high reclaimed_prior_low rejected_prior_high inside_bar')
    for tf in ('1m', '5m'):
        key = 'last_two_' + tf
        context[key] = _fields(raw[key], 'status higher_low higher_close last_body '
            'last_close_location postdecision_confirmation_evaluated')
        context[key]['candles'] = [_fields(row, 'open_time open high low close volume')
                                    for row in raw[key]['candles']]
    for key in ('parent_1d', 'parent_4h'):
        parent = raw[key]
        view = _fields(parent, 'evidence_status pre_setup lifecycle lifecycle_scope '
            'decision_state child_nested position_in_bound distance_is_unobstructed_room')
        view['bound'] = _fields(parent['bound'], 'id lineage_id predecessor_version_id '
            'creation_reason range_low range_high low_pivot_id high_pivot_id formation_hour available_at')
        view['updates'] = [_fields(row, 'id available_at pre_lineage_id post_lineage_id '
            'post_state source_break_direction') for row in parent['updates']]
        context[key] = view
    context['current'] = _fields(source['source_packet']['current']['source_candle'],
                                  'open_time open high low close volume')
    context['current_validated'] = source['source_packet']['current']['validated'] is True
    return context


def _project(source):
    raw = source['source_packet']
    setup, decision = clock(raw['setup_open']), clock(raw['decision_time'])
    if setup != setup.floor('h') or decision != setup + pd.Timedelta(hours=1):
        raise ValueError('LC decision clocks')
    if (set(raw['candle_columns']) != set(COLUMNS)
            or len(raw['candle_columns']) != len(COLUMNS)
            or set(raw['evidence']) - (set(PERIODS) | {'parent_4h', 'parent_1d'})):
        raise ValueError('unsupported candle schema')
    provenance = raw['provenance']
    for field in ('instrument', 'data_stream_id'):
        if not isinstance(provenance[field], str) or not provenance[field].strip():
            raise ValueError('missing source identity')
    context = _context(source)
    curriculum = dict(records=deepcopy(raw['curriculum']), brief=deepcopy(source['master_brief']))
    p = dict(version=VERSION, contract_sha256=CONTRACT_SHA256, case_id=raw['case_id'],
        setup_open=setup.isoformat(), decision_time=decision.isoformat(),
        source_binding=dict(version=source['version'], sha256=digest(source)),
        curriculum_sha256=digest(curriculum), instrument=provenance['instrument'],
        data_stream_id=provenance['data_stream_id'], context=context,
        candles={}, levels={}, citation_catalog={}, curriculum=curriculum,
        limitations=dict(source=deepcopy(raw['limitations']), projection=[
            'caller-verified reconstruction; not live receipt authentication',
            'optional fusion, derivatives and Fibonacci features omitted',
            'candle extrema are not confirmed pivots or proven support']),
        execution_authorized=False)

    def add_level(identity, price, kind, tf, locator, start, end, available, lifecycle):
        if not number(price) or available > decision or start > end or end > available:
            raise ValueError('invalid level availability or price')
        p['levels'][identity] = dict(id=identity, price=price, kind=kind, timeframe=tf,
            instrument=p['instrument'], data_stream_id=p['data_stream_id'],
            source_locator=locator, observation_start=start.isoformat(),
            observation_end=end.isoformat(), available_at=available.isoformat(),
            lifecycle=lifecycle,
            relation_to_setup='pre_setup' if available < setup else 'within_setup')
        p['citation_catalog'][identity] = ['levels', identity]

    for tf, minutes in PERIODS.items():
        rows = raw['evidence'].get(tf, [])
        if not isinstance(rows, list):
            raise ValueError('candles must be lists')
        p['candles'][tf] = []
        if not rows:
            p['limitations']['projection'].append('missing:' + tf)
        previous = None
        period = pd.Timedelta(minutes=minutes)
        for i, values in enumerate(rows):
            if not isinstance(values, list) or len(values) != len(COLUMNS):
                raise ValueError('invalid candle row')
            row = dict(zip(raw['candle_columns'], values))
            opened = clock(row['open_time']); end = opened + period
            if (not prices(row) or opened != opened.floor(str(minutes) + 'min')
                    or end > decision or (previous is not None and opened <= previous)):
                raise ValueError('invalid or future candle')
            if previous is not None and opened != previous + period:
                p['limitations']['projection'].append('gap:' + tf)
            previous = opened
            row.update(open_time=opened.isoformat(), observation_end=end.isoformat(),
                       available_at=end.isoformat())
            p['candles'][tf].append(row)
            p['citation_catalog'][f'candle:{tf}:{i}'] = ['candles', tf, i]
            for side in ('high', 'low'):
                add_level(f'bar:{tf}:{i}:{side}', row[side], 'candle_extreme', tf,
                    ['source_packet', 'evidence', tf, i, raw['candle_columns'].index(side)],
                    opened, end, end, 'observed')

    for tf in ('1d', '4h'):
        view = context['parent_' + tf]
        if view['evidence_status'] != 'known':
            p['limitations']['projection'].append('unknown:parent_' + tf)
            continue
        bound = view['bound']
        if bound is None:
            continue
        available = clock(bound['available_at'])
        if available >= setup:
            raise ValueError('parent must predate setup')
        start = clock(bound['formation_hour'])
        for side in ('low', 'high'):
            add_level(f'parent:{tf}:{bound["id"]}:{side}', bound['range_' + side],
                'parent_boundary', tf, ['context', 'parent_' + tf, 'bound', 'range_' + side],
                start, available, available, view['lifecycle'])

    for group in ('hourly', 'current', 'parent_1d', 'parent_4h'):
        p['citation_catalog'][group] = ['context', group]
    p['citation_catalog']['limitations'] = ['limitations']
    p['citation_catalog']['curriculum:brief'] = ['curriculum', 'brief']
    for i in range(len(curriculum['records'])):
        p['citation_catalog'][f'curriculum:{i}'] = ['curriculum', 'records', i]
    p['limitations']['projection'] = sorted(set(p['limitations']['projection']))
    p['seal'] = digest(p)
    return p


def build_structure_packet(source_request: dict) -> dict:
    """Allowlisted projection of a validated legacy request; inputs are unchanged."""
    try:
        validate_context_request(source_request)
        return _project(source_request)
    except (KeyError, TypeError, IndexError, AttributeError, OverflowError) as exc:
        raise ValueError('invalid structure source') from exc


def validate_structure_packet(source_request: dict, packet: dict) -> None:
    # Canonical comparison distinguishes False/0 and does not trust a reseal.
    expected = build_structure_packet(source_request)
    if not isinstance(packet, dict) or digest(packet) != digest(expected):
        raise ValueError('structure packet differs from validated source')
