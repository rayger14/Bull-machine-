"""Pure H1/H2 permission audit; never enforce gates or simulate candidate trades.

Evidence is explicitly ATTESTED, not inferred from finite features or observation
IDs. Each field record supplies field, instrument, value, value_digest (the
replay_clock.digest of value), origin, source, units, formula_id, nonempty mappings
of observation_hashes/dependency_hashes, event_time, available_at, received_at,
valid_until, consumed_at and consumed_path. Native paths are features.<field>;
taker raw paths are features.taker_imbalance.inputs.<field>. Taker output evidence
declares branch=volumes|ratio. Raw branch inputs require their own attestations.
Callers separately freeze expected_field_contracts for every required native/raw
field, mapping source, units and formula_id. Missing expectations are UNKNOWN;
identities are never inferred from the evidence being checked.

Consumption must equal feature_available_at, availability is max(source, receipt),
and valid_until is EXCLUSIVE: decision >= valid_until is expired, matching the
observation replay clock. The source feature hour must be aligned, timezone-aware
and completed by feature_available_at <= decision. No TTL is invented.
Opaque source/dependency hashes are required references, NOT acquisition proof.
H1 changes no numeric gate comparisons. H2 is independent of H1 and uses exactly
the previous completed hourly row. Annotation never changes native features,
cooldown, dedup, selection, positions or outcomes. Full candidate displacement
requires a separately authorized native replay; this module does not do it.
"""
from copy import deepcopy
import hashlib
import math
from numbers import Real
from pathlib import Path
import re

import pandas as pd

from scripts.research.replay_clock import digest, json_safe, utc

OI = 'oi_observed_evidence_v1'
LC = 'lc_observed_evidence_v1'
H2 = 'lc_prior_compression_v1'
POLICIES = (OI, LC, H2)
FIELDS = {OI: ('oi_change_4h', 'oi_change_24h', 'taker_imbalance'),
          LC: ('volume_zscore', 'rsi_14', 'bb_width', 'chop_score', 'adx')}
LIMITATIONS = ('attested_evidence_contract_not_acquisition_verification',
               'audit_only_not_candidate_execution_or_displacement',
               'unknown_evidence_never_grants_permission')
_MISSING = object()
_IDENTITY = ('field', 'instrument', 'source', 'units', 'formula_id', 'consumed_path', 'value_digest')
_TIMES = ('event_time', 'available_at', 'received_at', 'valid_until', 'consumed_at')


def _finite(value):
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def _status(rejected, unknown):
    return 'reject' if rejected else 'unknown' if unknown else 'pass'


def _field(name, native, record, *, instrument, decision, consumed_target,
           expected_contract=None, raw=False):
    rejected, unknown = [], []
    expected_contract = expected_contract if isinstance(expected_contract, dict) else {}
    for key in ('source', 'units', 'formula_id'):
        if not isinstance(expected_contract.get(key), str) or not expected_contract[key]:
            unknown.append('missing_expected_'+key)
    if consumed_target is None:
        unknown.append('feature_consumption_target_not_supplied')
    if native is _MISSING:
        unknown.append('native_value_not_supplied' if not raw else 'raw_value_not_supplied')
    elif not _finite(native):
        rejected.append('nonfinite_or_nonnumeric_value')
    if not isinstance(record, dict):
        return dict(status=_status(rejected, unknown+['missing_attestation']),
                    reasons=rejected+unknown+['missing_attestation'], value=None if native is _MISSING else json_safe(native),
                    origin='unknown', evidence_refs={})
    origin = record.get('origin', 'unknown')
    if origin in ('defaulted', 'invalid'):
        rejected.append('origin_'+origin)
    elif origin not in ('observed', 'computed'):
        unknown.append('origin_unknown')
    for key in _IDENTITY:
        if not isinstance(record.get(key), str) or not record[key]:
            unknown.append('missing_'+key)
    for key in ('source', 'units', 'formula_id'):
        if (isinstance(expected_contract.get(key), str) and expected_contract[key]
                and isinstance(record.get(key), str) and record[key]
                and record[key] != expected_contract[key]):
            rejected.append(key+'_contract_mismatch')
    for key in ('observation_hashes', 'dependency_hashes'):
        hashes = record.get(key)
        if not isinstance(hashes, dict) or not hashes:
            unknown.append('missing_'+key)
        elif any(not isinstance(ref, str) or not ref or not isinstance(value, str)
                 or re.fullmatch('[0-9a-f]{64}', value) is None for ref, value in hashes.items()):
            rejected.append('invalid_'+key)
    expected_path = ('features.taker_imbalance.inputs.' if raw else 'features.')+name
    for key, expected in (('field', name), ('instrument', instrument), ('consumed_path', expected_path)):
        if record.get(key) and record[key] != expected:
            rejected.append(key+'_mismatch')
    attested_value = record.get('value', _MISSING)
    if attested_value is _MISSING:
        unknown.append('missing_value')
    elif not _finite(attested_value):
        rejected.append('invalid_attested_value')
    else:
        actual_digest = digest(attested_value)
        if record.get('value_digest') and record['value_digest'] != actual_digest:
            rejected.append('value_digest_mismatch')
        if native is not _MISSING and _finite(native) and actual_digest != digest(native):
            rejected.append('native_value_mismatch')
    times = {}
    for key in _TIMES:
        if record.get(key) is None:
            unknown.append('missing_'+key)
        else:
            try:
                times[key] = utc(record[key], key)
            except (ValueError, TypeError, OverflowError):
                rejected.append('invalid_'+key)
    if 'event_time' in times and 'available_at' in times and times['event_time'] > times['available_at']:
        rejected.append('event_after_availability')
    if all(key in times for key in ('available_at', 'received_at', 'consumed_at')):
        if max(times['available_at'], times['received_at']) > times['consumed_at']:
            rejected.append('input_late_for_consumption')
    if 'consumed_at' in times:
        if consumed_target is not None and times['consumed_at'] != consumed_target:
            rejected.append('consumption_target_mismatch')
        if times['consumed_at'] > decision:
            rejected.append('consumed_after_decision')
    if 'valid_until' in times and decision >= times['valid_until']:
        rejected.append('expired_at_decision')
    refs = {key: deepcopy(record[key]) for key in (*_IDENTITY, *_TIMES, 'observation_hashes',
                                                  'dependency_hashes', 'branch') if key in record}
    return dict(status=_status(rejected, unknown), reasons=rejected+unknown,
                value=None if native is _MISSING else json_safe(native), origin=origin,
                evidence_refs=json_safe(refs))


def _prior_compression(features, previous, decision):
    reasons, values = [], {}
    if previous is None:
        return ['prior_row_not_supplied'], values
    try:
        opened = utc(features['timestamp'])
        prior = previous['features']
        prior_open = utc(prior['timestamp'])
        available = utc(previous['available_at'])
        if opened != opened.floor('1h') or prior_open != opened-pd.Timedelta('1h'):
            reasons.append('prior_not_immediately_preceding_hour')
        if previous.get('completed') is not True or available < prior_open+pd.Timedelta('1h'):
            reasons.append('prior_not_completed')
        if available > decision:
            reasons.append('prior_not_available_at_decision')
        if decision < opened+pd.Timedelta('1h'):
            reasons.append('current_hour_not_completed')
        width = prior.get('bb_width', _MISSING)
        values['previous_bb_width'] = None if width is _MISSING else json_safe(width)
        if width is _MISSING or not _finite(width):
            reasons.append('prior_bb_width_missing_or_nonfinite')
        elif width > .06:
            reasons.append('prior_bb_width_above_0.06')
    except (KeyError, ValueError, TypeError, OverflowError):
        reasons.append('invalid_prior_hour_context')
    return reasons, values


def evaluate_policy(policy_id, features, *, decision_time, instrument,
                    evidence=None, previous=None, feature_available_at=None,
                    expected_field_contracts=None):
    """Return independent pass/reject/unknown permission with inspectable reasons."""
    if policy_id not in POLICIES:
        raise ValueError('Unknown audit policy')
    if not isinstance(instrument, str) or not instrument:
        raise ValueError('Instrument identity required')
    decision = utc(decision_time)
    consumed_target = utc(feature_available_at) if feature_available_at is not None else None
    if consumed_target is not None and consumed_target > decision:
        raise ValueError('Feature cannot be available after its decision')
    if policy_id == H2:
        reasons, values = _prior_compression(features, previous, decision)
        return dict(policy_id=policy_id, status='reject' if reasons else 'pass',
                    would_allow=not reasons, reasons=reasons, inspected_values=values,
                    evidence_refs={}, fields={}, certified=False, limitations=list(LIMITATIONS))
    evidence = evidence if isinstance(evidence, dict) else {}
    contracts = expected_field_contracts if isinstance(expected_field_contracts, dict) else {}
    extra_rejects, extra_unknown = [], []
    if features.get('timestamp') is None:
        extra_unknown.append('feature_timestamp_not_supplied')
    else:
        try:
            opened = utc(features['timestamp'])
            if opened != opened.floor('1h'):
                extra_rejects.append('feature_hour_not_aligned')
            if opened+pd.Timedelta('1h') > decision:
                extra_rejects.append('feature_hour_not_completed_at_decision')
            if consumed_target is not None and opened+pd.Timedelta('1h') > consumed_target:
                extra_rejects.append('feature_hour_not_completed_at_consumption')
        except (ValueError, TypeError, OverflowError):
            extra_rejects.append('invalid_feature_timestamp')
    checked = {name: _field(name, features.get(name, _MISSING), evidence.get(name),
                instrument=instrument, decision=decision, consumed_target=consumed_target,
                expected_contract=contracts.get(name))
               for name in FIELDS[policy_id]}
    if policy_id == OI:
        taker = evidence.get('taker_imbalance')
        branch = taker.get('branch') if isinstance(taker, dict) else None
        raw_names = ('taker_buy_vol_1h', 'taker_sell_vol_1h') if branch == 'volumes' else (
            ('taker_buy_sell_ratio',) if branch == 'ratio' else ())
        if not raw_names:
            extra_unknown.append('taker_branch_not_attested')
        for name in raw_names:
            record = evidence.get(name)
            value = record.get('value', _MISSING) if isinstance(record, dict) else _MISSING
            checked[name] = _field(name, value, record, raw=True,
                instrument=instrument, decision=decision, consumed_target=consumed_target,
                expected_contract=contracts.get(name))
        if raw_names and all(checked[name]['status'] == 'pass' for name in raw_names):
            raw_values = [evidence[name]['value'] for name in raw_names]
            if branch == 'volumes':
                buy, sell = raw_values
                denominator = buy+sell
                sensible = buy >= 0 and sell >= 0 and _finite(denominator) and denominator > 0
                computed = (buy-sell)/denominator if sensible else None
            else:
                ratio = raw_values[0]
                denominator = ratio+1
                sensible = ratio > 0 and _finite(denominator) and denominator > 0
                computed = (ratio-1)/denominator if sensible else None
            if not sensible:
                extra_rejects.append('invalid_taker_denominator_or_raw_values')
            elif _finite(features.get('taker_imbalance')) and computed != features['taker_imbalance']:
                extra_rejects.append('taker_formula_mismatch')
    rejected = [name+':'+reason for name, field in checked.items() if field['status'] == 'reject'
                for reason in field['reasons']] + extra_rejects
    unknown = [name+':'+reason for name, field in checked.items() if field['status'] == 'unknown'
               for reason in field['reasons']] + extra_unknown
    status = _status(rejected, unknown)
    return dict(policy_id=policy_id, status=status, would_allow=status == 'pass',
                reasons=rejected+unknown, inspected_values={name: field['value'] for name, field in checked.items()},
                evidence_refs={name: field['evidence_refs'] for name, field in checked.items()},
                fields=checked, certified=False, limitations=list(LIMITATIONS))


class ContextAuditor:
    """Audit every completed hour; reconstruct by fresh whole-prehistory replay."""
    def __init__(self, instrument, *, expected_field_contracts=None):
        if not isinstance(instrument, str) or not instrument:
            raise ValueError('Instrument identity required')
        self.instrument = instrument
        self.expected_field_contracts = deepcopy(expected_field_contracts or {})
        self.previous = None
        self.hours_observed = 0
        self.manifest = dict(policy_ids=list(POLICIES), source_sha256=hashlib.sha256(
            Path(__file__).read_bytes()).hexdigest(), instrument=instrument,
            h2_prior_hour_bb_width_max=.06, freshness='explicit valid_until exclusive; no TTL',
            expected_field_contracts=deepcopy(self.expected_field_contracts),
            expected_field_contracts_digest=digest(self.expected_field_contracts),
            evidence_scope=LIMITATIONS[0], mode='audit_only_independent_arms')
        self.contract_id = digest(self.manifest)

    def observe(self, output, decision_time, evidence=None):
        if not output.get('hourly_updated'):
            return None
        features = output['features']
        opened = utc(features['timestamp'])
        available = utc(output['features_available_at'])
        decision = utc(decision_time)
        if opened != opened.floor('1h') or available < opened+pd.Timedelta('1h') or available > decision:
            raise ValueError('Completed as-of-available hourly feature row required')
        if self.previous is not None and opened != utc(self.previous['features']['timestamp'])+pd.Timedelta('1h'):
            raise ValueError('Duplicate/out-of-order/gapped hourly audit history')
        policies = {policy: evaluate_policy(policy, features, decision_time=decision,
            instrument=self.instrument, evidence=evidence, previous=self.previous,
            feature_available_at=available,
            expected_field_contracts=self.expected_field_contracts) for policy in POLICIES}
        native = output.get('engine_signal') or {}
        book = output.get('native_book') or {}
        result = dict(feature_open_time=str(opened), decision_time=str(decision),
                      policies=policies, certified=False,
                      native_bar_index=book.get('bar_index', native.get('bar_index')),
                      native_archetypes={name: deepcopy(native.get('archetypes', {}).get(name))
                                         for name in ('oi_divergence', 'liquidity_compression')},
                      native_entries=[deepcopy(signal) for signal in book.get('acted_signals', [])
                                      if signal.get('action') == 'ENTRY'
                                      and signal.get('archetype') in ('oi_divergence', 'liquidity_compression')],
                      limitations=list(LIMITATIONS))
        self.previous = dict(features=deepcopy(features), available_at=available, completed=True)
        self.hours_observed += 1
        return result

    def snapshot(self):
        return deepcopy(dict(previous=self.previous, hours_observed=self.hours_observed,
                             instrument=self.instrument, contract_id=self.contract_id))


def annotate_pipeline(result, *, instrument, evidence_by_hour=None, expected_field_contracts=None):
    """Produce a separate audit sidecar from supplied rows; never claim missing
    audit-start context means the actual engine had no earlier history.
    """
    auditor = ContextAuditor(instrument, expected_field_contracts=expected_field_contracts)
    attestations = {str(utc(key)): value for key, value in (evidence_by_hour or {}).items()}
    rows = []
    for row in result['rows']:
        output = row['output']
        key = str(utc(output['features']['timestamp'])) if output.get('hourly_updated') else None
        annotation = auditor.observe(output, row['decision_time'], attestations.get(key))
        if annotation is not None:
            rows.append(annotation)
    summary = {policy: dict(pass_count=0, reject=0, unknown=0, native_selected=0,
                           selected_without_permission=0) for policy in POLICIES}
    for row in rows:
        for policy, evaluation in row['policies'].items():
            summary[policy]['pass_count' if evaluation['status'] == 'pass' else evaluation['status']] += 1
            name = 'oi_divergence' if policy == OI else 'liquidity_compression'
            selected = bool((row['native_archetypes'][name] or {}).get('selected', False))
            summary[policy]['native_selected'] += int(selected)
            summary[policy]['selected_without_permission'] += int(selected and not evaluation['would_allow'])
    first = rows[0]['native_bar_index'] if rows else None
    return dict(rows=rows, summary=summary, state=json_safe(auditor.snapshot()), certified=False,
                native_contract_id=result.get('contract_id'), policy_contract_id=auditor.contract_id,
                manifest=auditor.manifest, limitations=list(LIMITATIONS),
                coverage=dict(hours_audited=len(rows), first_native_bar_index=first,
                              starts_after_native_history=first > 1 if first is not None else None,
                              initial_prior_context='not supplied to sidecar; actual prior history not inferred'))
