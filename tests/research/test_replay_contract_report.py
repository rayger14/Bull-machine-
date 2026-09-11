import importlib.util
import json
from pathlib import Path
import unittest
import pandas as pd


class ReplayReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        p=Path(__file__).resolve().parents[2]/'scripts/research/replay_contract_report.py'
        if not p.exists(): raise AssertionError('Replay report implementation missing')
        s=importlib.util.spec_from_file_location('replay_contract_report',p)
        cls.m=importlib.util.module_from_spec(s);s.loader.exec_module(cls.m)

    def fixture(self):
        index=pd.date_range('2026-01-01',periods=3,freq='h',tz='UTC')
        store=pd.DataFrame({'tf1h_fvg_present':0,'tf4h_fvg_present':0,
                            'fvg_present':float('nan')},index=index)
        records=[dict(timestamp=str(t),tf1h_fvg_present=0,tf4h_fvg_present=0) for t in index]
        cfg={f'a{i}':{'name':f'a{i}','hard_gates':[]} for i in range(17)}
        cfg['a0']['hard_gates']=[{'feature':'derived:any_fvg','op':'bool_true'}]
        return store,records,cfg

    def test_missing_provenance_cannot_certify_and_roster_is_preserved(self):
        s,l,c=self.fixture();r=self.m.build_report(s,l,c,{})
        self.assertEqual(len(r['archetypes']),17)
        self.assertFalse(r['certified'])
        self.assertIn('missing_available_at',r['blockers'])
        self.assertIn('missing_source_version',r['blockers'])
        self.assertEqual(r['archetypes'][0]['gates'][0]['store_candidate_changes'],3)

    def test_all_duplicate_rows_excluded_without_arbitrary_last_wins(self):
        s,l,c=self.fixture();l.append(dict(l[0],tf1h_fvg_present=1))
        r=self.m.build_report(s,l,c,{})
        self.assertEqual(r['coverage']['paired_rows'],2)
        self.assertEqual(r['coverage']['excluded_duplicate_rows'],2)

    def test_empty_evidence_never_passes(self):
        s,l,c=self.fixture();r=self.m.build_report(s.iloc[:0],[],c,{})
        self.assertFalse(r['certified'])
        self.assertEqual(len(r['archetypes']),17)
        self.assertIn('no_paired_evidence',r['blockers'])

    def test_explicit_version_mismatch_is_reported(self):
        s,l,c=self.fixture();s.attrs['source_version']='store-v2'
        for row in l:
            row['source_version']='live-v1';row['available_at']=row['timestamp']
        r=self.m.build_report(s,l,c,{})
        self.assertIn('source_version_mismatch',r['blockers'])
        self.assertFalse(r['certified'])

    def test_manifest_is_json_safe_and_boundary_is_separate(self):
        s,l,c=self.fixture();r=self.m.build_report(s,l,c,{'fixture':'hash'})
        json.dumps(r,allow_nan=False)
        self.assertEqual(r['hashes']['fixture'],'hash')
        self.assertEqual(r['boundary']['cases'],192)
        self.assertEqual(r['boundary']['mismatches'],48)
        self.assertIn('not_trade_counts',r['limitations'])

    def test_sparse_live_rows_preserve_absent_keys_in_reference(self):
        s,l,c=self.fixture()
        l[1]['fvg_present']=False
        r=self.m.build_report(s,l,c,{})
        gate=r['archetypes'][0]['gates'][0]
        self.assertEqual(gate['statuses']['live_reference'],{'fail':3})
        self.assertEqual(gate['live_candidate_changes'],0)

    def test_known_instrument_or_venue_mismatch_is_excluded(self):
        for instrument,venue in [('ETH','venue-a'),('BTC','venue-b')]:
            s,l,c=self.fixture();s.attrs.update(instrument='BTC',venue='venue-a')
            for row in l:row.update(instrument=instrument,venue=venue)
            r=self.m.build_report(s,l,c,{})
            self.assertIn('instrument_or_venue_mismatch',r['blockers'])
            self.assertEqual(r['coverage']['paired_rows'],0)
            self.assertEqual(r['coverage']['excluded_identity_rows'],3)


if __name__=='__main__': unittest.main()
