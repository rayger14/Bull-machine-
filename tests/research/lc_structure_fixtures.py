"""Synthetic offline inputs only; no market archive or agent calls."""
from scripts.research.conditional_assessment import digest
from scripts.research.lc_context_assessment import build_context_request
from tests.research.test_lc_context_assessment import empty_snapshot, context_brief, SETTINGS
from tests.research.test_lc_master_assessment import packet


def structure_source(mutator=None):
    value = packet()
    if mutator:
        mutator(value)
    value.pop('seal', None)
    value['seal'] = digest(value)
    return build_context_request(value, empty_snapshot(), context_brief(), SETTINGS)


def structure_policy():
    return dict(version='lc_structure_policy_v1', instrument='BTC-USD', data_stream_id='same-stream',
        max_entry_price=106., entry_expiry_minutes=15, horizon_minutes=60,
        minimum_net_rr=.5, risk_budget_usd=100., max_notional_usd=2000., equity_usd=1000.,
        max_leverage=2., roundtrip_cost_bps=12., processing_seconds=90., routing_seconds=0., tick_size=.1)


def structure_answer(p, policy, decision='enter_proposal'):
    def claim(ids):
        return dict(text='Synthetic schema fixture; not a trading judgment.', evidence_ids=ids)
    stop = 'bar:5m:11:low'; target = 'bar:5m:11:high'
    close = p['context']['current']['close']
    plan = dict(trigger={'kind':'immediate'}, confirmation_evidence_ids=['candle:5m:11'],
        invalidation_level_id=stop, invalidation_operator='touch_or_below', stop_level_id=stop,
        destination_level_id=target,
        obstacle_level_ids=sorted(k for k,v in p['levels'].items() if close < v['price'] < 110.),
        horizon_minutes=60, expiry_minutes=15)
    if decision == 'wait_proposal':
        # A different synthetic policy can target the parent ceiling at120.
        target = next(k for k,v in p['levels'].items() if v['price'] == 120.)
        plan.update(trigger={'kind':'close_above','level_id':'bar:5m:11:high'},
                    destination_level_id=target, confirmation_evidence_ids=[],
                    obstacle_level_ids=sorted(k for k,v in p['levels'].items() if close < v['price'] < 120.))
    return dict(version='lc_structure_proposal_v1', contract_sha256=p['contract_sha256'],
        case_id=p['case_id'], packet_sha256=p['seal'], curriculum_sha256=p['curriculum_sha256'],
        policy_sha256=digest(policy) if policy is not None else None, decision=decision,
        thesis='downside_rebound', parent_child=claim(['parent_4h','current']),
        sequence=[dict(claim(['candle:5m:10']), observation_end='2026-01-01T03:55:00Z'),
                  dict(claim(['candle:5m:11']), observation_end='2026-01-01T04:00:00Z')],
        supporting=[claim(['current'])], opposing=[claim(['parent_1d'])],
        competing_explanation=claim(['parent_4h']),
        unknowns=[claim(['limitations'])] if decision=='insufficient_evidence' else [],
        plan=plan if decision in ('enter_proposal','wait_proposal') else None,
        execution_authorized=False)
