"""Permission-only H1/H2 witnesses; no candidate trades or native mutation."""
from copy import deepcopy
import importlib
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from scripts.research.replay_clock import digest

ROOT = Path(__file__).resolve().parents[2]
OPEN = pd.Timestamp('2026-01-01T00:00Z')
AT = OPEN + pd.Timedelta('1h')
INSTRUMENT = 'BTC-fixture'
LC = 'lc_observed_evidence_v1'
OI = 'oi_observed_evidence_v1'
H2 = 'lc_prior_compression_v1'


def features(hour=0, **changes):
    result = dict(timestamp=OPEN+pd.Timedelta(hours=hour), volume_zscore=3.5,
                  rsi_14=25., bb_width=.04, chop_score=.2, adx=40.,
                  oi_change_4h=0., oi_change_24h=0., taker_imbalance=0.)
    result.update(changes)
    return result


def attestation(field, value, at=AT, origin='observed', raw=False):
    return dict(field=field, value=value, value_digest=digest(value), origin=origin,
                source='fixture-source', instrument=INSTRUMENT, units='declared-fixture-units',
                formula_id='fixture-formula-v1', observation_hashes={'observation-id': 'a'*64},
                dependency_hashes={'input-window': 'b'*64}, event_time=at-pd.Timedelta('1min'),
                available_at=at, received_at=at, valid_until=at+pd.Timedelta('1h'),
                consumed_at=at, consumed_path=('features.taker_imbalance.inputs.' if raw else 'features.')+field)


def evidence(values, branch='volumes'):
    at = values['timestamp'] + pd.Timedelta('1h')
    result = {key: attestation(key, value, at) for key, value in values.items() if key != 'timestamp'}
    result['taker_imbalance']['branch'] = branch
    raw = {'taker_buy_vol_1h': 10., 'taker_sell_vol_1h': 10.} if branch == 'volumes' else {'taker_buy_sell_ratio': 1.}
    result.update({key: attestation(key, value, at, raw=True) for key, value in raw.items()})
    return result


def expected_contracts():
    return {field: dict(source='fixture-source', units='declared-fixture-units',
                        formula_id='fixture-formula-v1') for field in (
        'volume_zscore', 'rsi_14', 'bb_width', 'chop_score', 'adx',
        'oi_change_4h', 'oi_change_24h', 'taker_imbalance',
        'taker_buy_vol_1h', 'taker_sell_vol_1h', 'taker_buy_sell_ratio')}


def output(hour=0, **changes):
    values = features(hour, **changes)
    return dict(hourly_updated=True, features=values,
                features_available_at=str(values['timestamp']+pd.Timedelta('1h')),
                engine_signal=dict(bar_index=hour+1, archetypes={}),
                native_book=dict(bar_index=hour+1, acted_signals=[], last_bar_signals=[]))


class ArchetypeContextPolicyTests(unittest.TestCase):
    def module(self):
        self.assertTrue((ROOT/'scripts/research/archetype_context_policy.py').exists(),
                        'Standalone H1/H2 audit policy evaluator is missing')
        return importlib.import_module('scripts.research.archetype_context_policy')

    def evaluate(self, policy=LC, values=None, records=None, **kwargs):
        values = features() if values is None else values
        return self.module().evaluate_policy(policy, values, decision_time=AT,
            instrument=INSTRUMENT, evidence=records, feature_available_at=AT,
            expected_field_contracts=expected_contracts(), **kwargs)

    def test_observed_zero_is_valid_but_defaulted_zero_rejects(self):
        values = features()
        records = evidence(values)
        observed = self.evaluate(OI, values, records)
        self.assertEqual(observed['status'], 'pass')
        self.assertTrue(observed['would_allow'])
        self.assertEqual(observed['inspected_values']['oi_change_4h'], 0.)
        records['oi_change_4h']['origin'] = 'defaulted'
        defaulted = self.evaluate(OI, values, records)
        self.assertEqual(defaulted['status'], 'reject')
        self.assertFalse(defaulted['would_allow'])

    def test_missing_metadata_stays_unknown_despite_finite_values_and_observation_ids(self):
        values = features(feature_observation_ids={'derivatives_snapshot': 'a'},
                          feature_observation_visible_at={'derivatives_snapshot': str(AT)})
        result = self.evaluate(OI, values)
        self.assertEqual(result['status'], 'unknown')
        self.assertFalse(result['would_allow'])
        self.assertIn('attested_evidence_contract_not_acquisition_verification', result['limitations'])
        self.assertFalse(result['certified'])

    def test_each_unknown_identity_consumption_or_freshness_field_is_unavailable(self):
        values = features()
        base = evidence(values)
        for key in ('source', 'instrument', 'units', 'formula_id', 'observation_hashes',
                    'dependency_hashes', 'event_time', 'available_at', 'received_at',
                    'valid_until', 'consumed_at', 'consumed_path', 'value_digest', 'field'):
            records = deepcopy(base)
            records['volume_zscore'].pop(key)
            with self.subTest(missing=key):
                result = self.evaluate(LC, values, records)
                self.assertEqual(result['status'], 'unknown')
                self.assertFalse(result['would_allow'])

    def test_missing_actual_feature_consumption_target_cannot_be_inferred_from_decision(self):
        values = features()
        result = self.module().evaluate_policy(LC, values, decision_time=AT,
                    instrument=INSTRUMENT, evidence=evidence(values),
                    expected_field_contracts=expected_contracts())
        self.assertEqual(result['status'], 'unknown')
        self.assertFalse(result['would_allow'])

    def test_invalid_nonfinite_late_or_expired_inputs_reject(self):
        for value in (np.nan, np.inf):
            values = features(volume_zscore=value)
            self.assertEqual(self.evaluate(LC, values, evidence(values))['status'], 'reject')
        values = features()
        for field, change in (('origin', 'invalid'), ('available_at', AT+pd.Timedelta('1s')),
                              ('received_at', AT+pd.Timedelta('1s')),
                              ('valid_until', AT-pd.Timedelta('1s')),
                              ('event_time', AT+pd.Timedelta('1s'))):
            records = evidence(values)
            records['volume_zscore'][field] = change
            with self.subTest(field=field):
                self.assertEqual(self.evaluate(LC, values, records)['status'], 'reject')
        records = evidence(values)
        records['volume_zscore']['valid_until'] = AT
        self.assertEqual(self.evaluate(LC, values, records)['status'], 'reject')
        records['volume_zscore']['valid_until'] = AT+pd.Timedelta('1ns')
        self.assertEqual(self.evaluate(LC, values, records)['status'], 'pass')

    def test_h1_without_expected_field_identity_contract_is_unknown(self):
        values = features()
        result = self.module().evaluate_policy(LC, values, decision_time=AT,
                    instrument=INSTRUMENT, evidence=evidence(values), feature_available_at=AT)
        self.assertEqual(result['status'], 'unknown')
        self.assertFalse(result['would_allow'])

    def test_h1_rejects_wrong_expected_source_units_or_formula_for_native_and_raw_fields(self):
        module = self.module()
        values = features()
        for policy, field in ((LC, 'volume_zscore'), (OI, 'taker_buy_vol_1h')):
            for identity in ('source', 'units', 'formula_id'):
                records = evidence(values)
                records[field][identity] = 'wrong-contract'
                with self.subTest(policy=policy, field=field, identity=identity):
                    result = module.evaluate_policy(policy, values, decision_time=AT,
                        instrument=INSTRUMENT, evidence=records, feature_available_at=AT,
                        expected_field_contracts=expected_contracts())
                    self.assertEqual(result['status'], 'reject')

    def test_public_h1_cannot_pass_missing_future_or_uncompleted_feature_hour(self):
        records = evidence(features())
        for stamp, wanted in ((None, 'unknown'), (OPEN.tz_localize(None), 'reject'),
                              (OPEN+pd.Timedelta('1min'), 'reject'),
                              (AT, 'reject'), (AT+pd.Timedelta('1d'), 'reject')):
            values = features()
            if stamp is None:
                values.pop('timestamp')
            else:
                values['timestamp'] = stamp
            with self.subTest(timestamp=stamp):
                result = self.evaluate(LC, values, records)
                self.assertEqual(result['status'], wanted)
                self.assertFalse(result['would_allow'])

    def test_native_entries_exclude_exit_and_unspecified_actions(self):
        auditor = self.module().ContextAuditor(INSTRUMENT)
        values = output()
        values['native_book']['acted_signals'] = [
            dict(archetype='liquidity_compression', action='ENTRY', id='entry'),
            dict(archetype='liquidity_compression', action='EXIT', id='exit'),
            dict(archetype='liquidity_compression', id='unknown'),
            dict(archetype='wick_trap', action='ENTRY', id='other')]
        row = auditor.observe(values, AT)
        self.assertEqual([record['id'] for record in row['native_entries']], ['entry'])

    def test_expected_field_contracts_are_propagated_and_bound_in_audit_manifest(self):
        module = self.module()
        values = features()
        native = dict(rows=[dict(decision_time=str(AT), output=output())], contract_id='reference')
        expected = expected_contracts()
        first = module.annotate_pipeline(native, instrument=INSTRUMENT,
            evidence_by_hour={str(OPEN): evidence(values)}, expected_field_contracts=expected)
        self.assertEqual(first['rows'][0]['policies'][LC]['status'], 'pass')
        self.assertEqual(first['manifest']['expected_field_contracts'], expected)
        changed = deepcopy(expected)
        changed['adx']['formula_id'] = 'changed-formula'
        second = module.annotate_pipeline(native, instrument=INSTRUMENT,
            evidence_by_hour={str(OPEN): evidence(values)}, expected_field_contracts=changed)
        self.assertEqual(second['rows'][0]['policies'][LC]['status'], 'reject')
        self.assertNotEqual(first['policy_contract_id'], second['policy_contract_id'])

    def test_value_field_instrument_and_consumption_path_are_bound(self):
        values = features()
        for key, wrong in (('field', 'bb_width'), ('instrument', 'ETH-fixture'),
                           ('value', 123.), ('value_digest', 'c'*64),
                           ('consumed_path', 'unrelated.cache.volume_zscore'),
                           ('consumed_at', AT-pd.Timedelta('1s'))):
            records = evidence(values)
            records['volume_zscore'][key] = wrong
            with self.subTest(key=key):
                self.assertEqual(self.evaluate(LC, values, records)['status'], 'reject')

    def test_computed_zero_requires_complete_adx_dependency_not_warmup_default(self):
        values = features(volume_zscore=0., rsi_14=0., bb_width=0., chop_score=0., adx=50.)
        records = evidence(values)
        for record in records.values():
            record['origin'] = 'computed'
        self.assertEqual(self.evaluate(LC, values, records)['status'], 'pass')
        records['adx']['origin'] = 'defaulted'
        self.assertEqual(self.evaluate(LC, values, records)['status'], 'reject')
        records.pop('adx')
        self.assertEqual(self.evaluate(LC, values, records)['status'], 'unknown')

    def test_taker_branch_requires_raw_inputs_and_preserves_observed_balanced_zero(self):
        values = features()
        ratio = evidence(values, 'ratio')
        self.assertEqual(self.evaluate(OI, values, ratio)['status'], 'pass')
        ratio.pop('taker_buy_sell_ratio')
        self.assertEqual(self.evaluate(OI, values, ratio)['status'], 'unknown')
        volumes = evidence(values)
        for field in ('taker_buy_vol_1h', 'taker_sell_vol_1h'):
            volumes[field].update(value=0., value_digest=digest(0.))
        self.assertEqual(self.evaluate(OI, values, volumes)['status'], 'reject')
        mismatch = evidence(values)
        mismatch['taker_buy_vol_1h'].update(value=30., value_digest=digest(30.))
        self.assertEqual(self.evaluate(OI, values, mismatch)['status'], 'reject')

    def test_finite_raw_taker_values_cannot_pass_with_overflowed_denominator(self):
        values = features()
        records = evidence(values)
        for name in ('taker_buy_vol_1h', 'taker_sell_vol_1h'):
            records[name].update(value=1e308, value_digest=digest(1e308))
        self.assertEqual(self.evaluate(OI, values, records)['status'], 'reject')

    def test_h1_is_permission_only_without_new_sign_or_lc_threshold_comparisons(self):
        values = features(oi_change_4h=.1, oi_change_24h=.1, volume_zscore=0.,
                          rsi_14=50., bb_width=.5, chop_score=.9)
        records = evidence(values)
        self.assertTrue(self.evaluate(OI, values, records)['would_allow'])
        self.assertTrue(self.evaluate(LC, values, records)['would_allow'])

    def test_h2_only_uses_exact_previous_completed_hour_and_includes_boundary(self):
        current = features(bb_width=.9)
        prior = dict(features=features(-1, bb_width=.06), completed=True, available_at=OPEN)
        self.assertEqual(self.evaluate(H2, current, previous=prior)['status'], 'pass')
        prior['features']['bb_width'] = .060001
        self.assertEqual(self.evaluate(H2, current, previous=prior)['status'], 'reject')
        prior['features']['bb_width'] = np.nan
        self.assertEqual(self.evaluate(H2, current, previous=prior)['status'], 'reject')
        self.assertEqual(self.evaluate(H2, features(bb_width=.01))['status'], 'reject')

    def test_h2_rejects_current_future_incomplete_or_nonadjacent_prior(self):
        for hour, completed, availability in ((0, True, AT), (-2, True, OPEN),
                                             (-1, False, OPEN), (-1, True, AT+pd.Timedelta('1s'))):
            prior = dict(features=features(hour, bb_width=.01), completed=completed, available_at=availability)
            with self.subTest(hour=hour, completed=completed):
                self.assertEqual(self.evaluate(H2, previous=prior)['status'], 'reject')

    def test_auditor_ignores_minute_carry_and_keeps_arms_independent(self):
        auditor = self.module().ContextAuditor(INSTRUMENT)
        first = auditor.observe(output(bb_width=.04), AT)
        self.assertEqual(first['policies'][H2]['status'], 'reject')
        before = digest(auditor.snapshot())
        carry = output(1, bb_width=.9)
        carry['hourly_updated'] = False
        self.assertIsNone(auditor.observe(carry, AT+pd.Timedelta('1min')))
        self.assertEqual(before, digest(auditor.snapshot()))
        second = auditor.observe(output(1, bb_width=.9), AT+pd.Timedelta('1h'))
        self.assertEqual(second['policies'][LC]['status'], 'unknown')
        self.assertEqual(second['policies'][H2]['status'], 'pass')
        with self.assertRaises(ValueError):
            auditor.observe(output(3), AT+pd.Timedelta('3h'))

    def test_future_append_and_fresh_whole_history_restart_are_deterministic(self):
        module = self.module()
        first, fresh = module.ContextAuditor(INSTRUMENT), module.ContextAuditor(INSTRUMENT)
        prefix = [first.observe(output(i, bb_width=.04 if i == 0 else .1), AT+pd.Timedelta(hours=i)) for i in range(2)]
        original_prefix = deepcopy(prefix)
        tail = first.observe(output(2), AT+pd.Timedelta('2h'))
        replayed = [fresh.observe(output(i, bb_width=.04 if i == 0 else .1) if i < 2 else output(i),
                                 AT+pd.Timedelta(hours=i)) for i in range(3)]
        self.assertEqual(digest(prefix), digest(original_prefix))
        self.assertEqual(digest(prefix+[tail]), digest(replayed))
        self.assertEqual(digest(first.snapshot()), digest(fresh.snapshot()))

    def test_sidecar_does_not_mutate_native_rows_and_marks_truncated_audit_start(self):
        rows = [dict(decision_time=str(AT+pd.Timedelta(hours=i)), output=output(i)) for i in (4, 5)]
        native = dict(rows=rows, contract_id='native-reference', certified=False)
        frozen = digest(native)
        sidecar = self.module().annotate_pipeline(native, instrument=INSTRUMENT)
        self.assertEqual(digest(native), frozen)
        self.assertTrue(sidecar['coverage']['starts_after_native_history'])
        self.assertEqual(sidecar['rows'][0]['policies'][H2]['reasons'], ['prior_row_not_supplied'])
        self.assertEqual(sidecar['summary'][LC]['unknown'], 2)
        self.assertFalse(sidecar['certified'])
        self.assertEqual(sidecar['native_contract_id'], 'native-reference')


if __name__ == '__main__':
    unittest.main()
