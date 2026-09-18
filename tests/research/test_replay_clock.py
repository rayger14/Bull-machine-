import importlib.util
from pathlib import Path
import unittest

import pandas as pd


class Accumulator:
    contract_id = 'accumulator-v1'

    def __init__(self):
        self.total = 0
        self.funding = []

    def update(self, candle, observations):
        self.total += candle['close']
        self.funding.append(observations.get('funding'))
        return {'total': self.total, 'funding': observations.get('funding')}

    def snapshot(self):
        return {'total': self.total, 'funding': self.funding}


def bars(n=6, freq='1h'):
    return pd.DataFrame({'open':100.,'high':101.,'low':99.,'close':100.,'volume':1.},
                        index=pd.date_range('2026-01-01',periods=n,freq=freq,tz='UTC'))


class ReplayClockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=Path(__file__).resolve().parents[2]/'scripts/research/replay_clock.py'
        if not path.exists():
            raise AssertionError('Replay clock implementation missing')
        spec=importlib.util.spec_from_file_location('replay_clock',path)
        cls.m=importlib.util.module_from_spec(spec)
        import sys
        sys.modules[spec.name]=cls.m
        spec.loader.exec_module(cls.m)

    def observation(self, **kwargs):
        data=dict(id='f1',feature='funding',value=.01,instrument='BTC',source='fixture',
                  units='rate',version='v1',event_time='2026-01-01T00:00Z',
                  available_at='2026-01-01T02:00Z')
        data.update(kwargs)
        return self.m.Observation(**data)

    def run_replay(self, frame, observations=(), **kwargs):
        return self.m.replay(frame,observations,Accumulator,instrument='BTC',timeframe='1h',**kwargs)

    def test_release_not_event_label_controls_visibility(self):
        r=self.run_replay(bars(3),[self.observation()])
        self.assertEqual([x['output']['funding'] for x in r['rows']],[None,.01,.01])

    def test_prefix_and_multiple_resume_cuts_preserve_state(self):
        frame=bars();obs=[self.observation()]
        full=self.run_replay(frame,obs)
        for cut in [1,2,5]:
            prefix=self.run_replay(frame.iloc[:cut],obs)
            resumed=self.run_replay(frame,obs,checkpoint=prefix['checkpoint'])
            self.assertEqual(prefix['rows'],full['rows'][:cut])
            self.assertEqual(prefix['rows']+resumed['rows'],full['rows'])
            self.assertEqual(resumed['state'],full['state'])
        self.assertEqual(full['state']['total'],600.)

    def test_warmup_precedes_first_emitted_bar_without_duplication(self):
        r=self.run_replay(bars(),emit_from='2026-01-01T03:00Z')
        self.assertEqual(len(r['rows']),3)
        self.assertEqual(r['rows'][0]['output']['total'],400.)

    def test_changed_pre_checkpoint_input_is_rejected(self):
        frame=bars();cp=self.run_replay(frame.iloc[:3])['checkpoint']
        frame.iloc[0,frame.columns.get_loc('volume')]=2.
        with self.assertRaisesRegex(ValueError,'checkpoint'):
            self.run_replay(frame,checkpoint=cp)

    def test_unknown_and_future_received_availability_cannot_certify(self):
        with self.assertRaisesRegex(ValueError,'available_at'):
            self.run_replay(bars(),[self.observation(available_at=None)])
        obs=self.observation(received_at='2026-01-01T04:00Z')
        r=self.run_replay(bars(),[obs])
        self.assertIsNone(r['rows'][2]['output']['funding'])
        self.assertEqual(r['rows'][3]['output']['funding'],.01)

    def test_duplicate_order_instrument_and_version_fail(self):
        a=self.observation();b=self.observation(id='f2',available_at='2026-01-01T03:00Z')
        cases=[[a,a],[b,a],[self.observation(instrument='ETH')],
               [a,self.observation(id='f2',available_at='2026-01-01T03:00Z',version='v2')]]
        for case in cases:
            with self.subTest(case=case),self.assertRaises(ValueError):
                self.run_replay(bars(),case)

    def test_stale_and_invalid_inputs_are_not_certified(self):
        obs=self.observation(valid_until='2026-01-01T03:00Z')
        r=self.run_replay(bars(),[obs])
        self.assertTrue(r['rows'][1]['certified'])
        self.assertFalse(r['rows'][2]['certified'])
        self.assertIn('stale',r['rows'][2]['issues'][0])
        r=self.run_replay(bars(),[self.observation(value=float('nan'),status='invalid')])
        self.assertFalse(r['certified'])

    def test_bad_bars_and_empty_evidence_fail(self):
        for frame in [bars().iloc[::2],bars().iloc[::-1],pd.concat([bars(),bars()]),bars(0)]:
            with self.subTest(rows=len(frame)),self.assertRaises(ValueError):
                self.run_replay(frame)

    def test_completed_and_developing_context_use_only_elapsed_bars(self):
        frame=bars(24)
        r=self.m.context_at(frame,'2026-01-01T02:00Z','1h','4h')
        self.assertEqual(r['completed'],[])
        self.assertEqual(r['developing'][0]['volume'],2.)
        self.assertEqual(r['developing'][0]['constituents'],2)
        r=self.m.context_at(frame,'2026-01-01T04:00Z','1h','4h')
        self.assertEqual(r['completed'][0]['volume'],4.)
        self.assertEqual(r['developing'],[])
        r=self.m.context_at(frame,'2026-01-02T00:00Z','1h','1d')
        self.assertEqual(r['completed'][0]['volume'],24.)

    def test_minute_context_has_identical_close_availability_rule(self):
        r=self.m.context_at(bars(60,'1min'),'2026-01-01T00:59Z','1min','1h')
        self.assertEqual(r['completed'],[])
        self.assertEqual(r['developing'][0]['volume'],59.)

    def test_invalid_warmup_taints_later_retained_state(self):
        a=self.observation(value=float('nan'),status='invalid',available_at='2026-01-01T01:00Z')
        b=self.observation(id='f2')
        r=self.run_replay(bars(3),[a,b],emit_from='2026-01-01T01:00Z')
        self.assertFalse(r['rows'][0]['certified'])
        self.assertFalse(r['certified'])
        self.assertTrue(r['issues'])

    def test_mutable_processor_outputs_cannot_rewrite_history(self):
        class Mutable(Accumulator):
            def update(self,candle,observations):
                self.funding.append(candle['close'])
                return {'history':self.funding}
        r=self.m.replay(bars(3),[],Mutable,instrument='BTC',timeframe='1h')
        self.assertEqual(r['rows'][0]['output']['history'],[100.])
        self.assertEqual(r['rows'][1]['output']['history'],[100.,100.])

    def test_hash_cannot_confuse_nan_with_literal_marker_dictionary(self):
        a=self.observation(value=float('nan'))
        cp=self.run_replay(bars(3),[a])['checkpoint']
        changed=self.observation(value={'__nonfinite__':'nan'})
        with self.assertRaisesRegex(ValueError,'checkpoint'):
            self.run_replay(bars(),[changed],checkpoint=cp)


if __name__=='__main__':
    unittest.main()
