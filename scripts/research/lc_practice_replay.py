"""Terminal-locked, independent-case replay. Not a portfolio or live executor."""
from collections import Counter
from copy import deepcopy
from decimal import Decimal, ROUND_FLOOR
import math
from numbers import Real

import pandas as pd

from scripts.research.conditional_entry import score_conditional
from scripts.research.lc_judgment_runner import _load, _sha, _verify_lock
from scripts.research.lc_structure_packet import clock, number, prices
from scripts.research.lc_structure_proposal import validate_structure_proposal
from scripts.research.lc_structure_preentry import check_structure_preentry
from scripts.research.lc_structure_outcome import score_structure_outcome
from scripts.research.lc_practice_runtime import read_archive, publish, sealed


def _result(status, reason, pnl=None, **extra):
    return dict(status=status, reason=reason, net_pnl=pnl, execution_authorized=False, **extra)


def replay_proposal(source, packet, raw_response, policy, rows, elapsed_seconds):
    checked = validate_structure_proposal(source, packet, raw_response, policy)
    if checked['status'] == 'valid_reject': return _result('rejected', 'deliberate_reject', 0.)
    if checked['status'] != 'valid_proposal': return _result('unavailable', checked['status'])
    if not number(elapsed_seconds, zero=True): return _result('unavailable', 'timing_unknown')
    plan = checked['proposal']['plan']; levels = packet['levels']
    tick = Decimal(str(policy['tick_size']))
    stop = (Decimal(str(levels[plan['stop_level_id']]['price']))/tick).to_integral_value(rounding=ROUND_FLOOR)*tick
    if Decimal(str(levels[plan['invalidation_level_id']]['price'])) != stop:
        return _result('unavailable', 'separate_postentry_invalidation')
    decision = clock(packet['decision_time']); minute = pd.Timedelta(minutes=1)
    try:
        response = decision + pd.Timedelta(seconds=elapsed_seconds)
        arm = (max(response, decision+pd.Timedelta(seconds=policy['processing_seconds']))
               +pd.Timedelta(seconds=policy['routing_seconds'])).ceil('min')
    except (OverflowError, ValueError): return _result('unavailable', 'unrepresentable_timing')
    completed = []
    for i in range(policy['entry_expiry_minutes']):
        at = decision + i*minute
        if i >= len(rows): return _result('unavailable', 'missing_preentry_minute')
        row = rows[i]
        try:
            if clock(row['open_time']) != at or not number(row['open']):
                return _result('unavailable', 'preentry_clock_or_open')
        except (ValueError, KeyError, TypeError): return _result('unavailable', 'preentry_clock_or_open')
        if at >= arm:
            execution = dict(response_available_at=response.isoformat(), proposed_fill_at=at.isoformat(),
                             fill_open=row['open'], completed_minutes=completed)
            entry = check_structure_preentry(source, packet, raw_response, policy, execution)
            if entry['status'] == 'eligible_hypothetical':
                scored = score_structure_outcome(source, packet, raw_response, policy, execution,
                    dict(instrument=packet['instrument'], data_stream_id=packet['data_stream_id'], minutes=rows[i:]))
                if scored['status'] != 'scored':
                    return _result('unavailable', scored['reason'], scored=scored, execution=execution)
                return _result('filled', scored['outcome']['exit_reason'], scored['outcome']['net_pnl'],
                    outcome=scored['outcome'], scored=scored, execution=execution)
            if entry['status'] == 'cancelled': return _result('cancelled', entry['reason'], 0., preentry=entry)
            if entry['status'] == 'invalid' or entry['reason'] not in ('before_arm','trigger_not_met'):
                return _result('unavailable', entry['reason'], preentry=entry)
        if not isinstance(row, dict) or not prices(row): return _result('unavailable', 'invalid_completed_minute')
        # No consumption of the next bar if a completed observation already cancels.
        if Decimal(str(row['low'])) <= stop:
            return _result('cancelled', 'preentry_invalidation', 0.)
        completed.append({k:row[k] for k in ('open_time','open','high','low','close','volume')})
    return _result('expired', 'entry_expiry', 0.)


def _attribution(agent, baseline):
    a = agent.get('net_pnl'); b = baseline.get('net_pnl')
    if a is None or b is None: return 'unavailable'
    if baseline['status'] != 'filled': return 'baseline_nonentry'
    if agent['status'] == 'rejected':
        return 'rejected_winner' if b > 0 else 'rejected_loser' if b < 0 else 'rejected_flat'
    if agent['status'] != 'filled': return agent['status'] + ('_missed_winner' if b > 0 else '_baseline_nonpositive')
    if b > 0: return 'winner_preserved' if a > 0 else 'winner_not_preserved'
    return 'loser_turned_profitable' if a > 0 else 'loser_still_taken'


def summarize_cases(cases):
    arms = sorted(set(k for c in cases for k in c['arms']))
    def comparison(rows):
        pairs=[(c['arms']['agent']['net_pnl'],c['arms']['mechanical_matched']['net_pnl']) for c in rows]
        deltas=[a-b for a,b in pairs if a is not None and b is not None]
        attribution={c['case_id']:_attribution(c['arms']['agent'],c['arms']['mechanical_matched']) for c in rows}
        return dict(matched=dict(count=len(deltas),roster_count=len(rows),known_delta=math.fsum(deltas),
            total_delta=math.fsum(deltas) if len(deltas)==len(rows) else None),
            attribution=attribution,attribution_counts=dict(Counter(attribution.values())))
    def group(rows):
        result = {}
        for arm in arms:
            values = [c['arms'].get(arm, {}).get('net_pnl') for c in rows]
            known = [v for v in values if v is not None]
            total = math.fsum(known)
            result[arm] = dict(count=len(values), known_count=len(known), unknown_count=len(values)-len(known),
                known_subtotal=total, total_net_pnl=total if len(known)==len(values) else None,
                known_mean=total/len(known) if known else None,
                statuses=dict(Counter(c['arms'].get(arm,{}).get('status','unavailable') for c in rows)))
        return result
    return dict(overall=group(cases),
        by_subtype={k:dict(group([c for c in cases if c['subtype']==k]),
            **comparison([c for c in cases if c['subtype']==k])) for k in sorted({c['subtype'] for c in cases})},
        **comparison(cases),
        interpretation='Independent case sums, not account returns, equity curve, Sharpe or portfolio drawdown')


def _safe_json(value):
    if isinstance(value, float) and not math.isfinite(value): return str(value)
    if isinstance(value, dict): return {k:_safe_json(v) for k,v in value.items()}
    if isinstance(value, list): return [_safe_json(v) for v in value]
    return value


def _rows(frame):
    # Preserve a bad future cell as unknown; only the chronological consumer
    # decides whether it matters. Do not crash on pd.NA or reject an unused tail.
    def numeric(value):
        if isinstance(value, Real) and not isinstance(value, bool):
            value = float(value)
            if math.isfinite(value): return value
        return None
    return [dict(open_time=at.isoformat(), **{k:numeric(row[k]) for k in ('open','high','low','close','volume')})
            for at,row in frame.iterrows()]


def score_run(run):
    """First verify all terminals, then (and only then) read each future interval."""
    locked = run.assert_reveal_allowed(); manifest = run.verify()
    existing = run.root/'case_results.json'
    if existing.exists():
        result = _verify_lock(existing)
        if result['terminal_lock_sha256'] != locked['sha256'] or result['manifest_sha256'] != manifest['sha256']:
            raise ValueError('result binding changed')
        return result
    cases = []
    for case in manifest['cases']:
        folder = run.root/case['folder']; stored = _load(folder/'input.json')
        source, packet, policy = (stored[k] for k in ('source','packet','policy'))
        t = locked['terminals'][case['case_id']]; elapsed = t['elapsed_seconds']
        raw = None; capture = None
        if (folder/'capture.json').exists():
            capture = _load(folder/'capture.json')
            raw = bytes.fromhex(capture['raw_hex']).decode('utf-8', errors='replace')
        arms = {}; rows = []; validation = None
        if case['source_status'] != 'verified':
            arms = {k:_result('unavailable','source_unavailable') for k in
                    ('agent','mechanical_matched','mechanical_90s','legacy_immediate','stay_flat')}
        else:
            decision = clock(packet['decision_time'])
            end = decision + pd.Timedelta(minutes=policy['entry_expiry_minutes']+policy['horizon_minutes']+1)
            frame = read_archive(manifest['archive'], decision, end); rows = _rows(frame)
            if t['status'] in ('valid_reject','valid_proposal','insufficient_evidence'):
                validation = validate_structure_proposal(source,packet,raw,policy)
                arms['agent'] = replay_proposal(source,packet,raw,policy,rows,elapsed)
            else: arms['agent'] = _result('unavailable',t['status'])
            control = stored['control']
            for name, delay in [('mechanical_matched',elapsed),('mechanical_90s',90.)]:
                if delay is None: arms[name] = _result('unavailable','timing_unknown')
                elif control['status']=='no_setup': arms[name]=_result('no_setup',control['reason'],0.)
                elif control['status']!='proposal': arms[name]=_result('unavailable',control['reason'])
                else: arms[name]=replay_proposal(source,packet,control['raw_response'],policy,rows,delay)
            legacy = deepcopy(source['plan_menu']['plans']['enter'])
            parameters = deepcopy(legacy['parameters'])
            parameters.update(processing_seconds=90, routing_seconds=0)
            try:
                scored = score_conditional(frame, as_of=parameters['exit_deadline'],
                                          notional=50000., cost_bps=12., **parameters)
                out = scored['outcome']; resolved = scored['resolution']['status']
                arms['legacy_immediate'] = _result('filled' if resolved=='entry_ready' and out['net_pnl'] is not None
                    else 'unavailable' if out['net_pnl'] is None else resolved,
                    'different fixed-notional/2R/deadline reference; not matched agent value',out['net_pnl'],legacy=scored)
            except (ValueError,TypeError,KeyError) as exc:
                arms['legacy_immediate']=_result('unavailable','legacy_reference:'+str(exc))
            arms['stay_flat']=_result('rejected','flat_reference',0.)
        cases.append(dict(case_id=case['case_id'], decision_time=case['decision_time'], subtype=case['subtype'],
            source_status=case['source_status'], terminal_status=t['status'], elapsed_seconds=elapsed,
            metadata=capture['metadata'] if capture else None, raw_response=raw, validation=validation,
            packet=packet, policy=policy, control=stored['control'], future_minutes=_safe_json(rows), arms=arms))
    overlaps = {}
    for arm in ('agent','mechanical_matched','mechanical_90s'):
        intervals=[]
        for c in cases:
            out=c['arms'][arm].get('outcome')
            if out: intervals.append((clock(out['entry_time']),clock(out['exit_observed_at'])))
        overlaps[arm]=sum(max(a[0],b[0]) < min(a[1],b[1]) for i,a in enumerate(intervals) for b in intervals[i+1:])
    result=sealed(dict(version='lc_practice_results_v1', manifest_sha256=manifest['sha256'],
        terminal_lock_sha256=locked['sha256'], archive_sha256=manifest['archive_sha256'],
        files={str(run.root/'manifest.json'):_sha(run.root/'manifest.json'),
               str(run.root/'terminal_lock.json'):_sha(run.root/'terminal_lock.json')},
        cases=cases, summary=summarize_cases(cases), overlap_pair_counts=overlaps,
        exposure=manifest['exposure'], costs='after flat modeled costs; excludes funding/impact',
        execution_authorized=False))
    publish(existing,result)
    return result
