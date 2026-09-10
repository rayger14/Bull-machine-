import importlib.util
from pathlib import Path
import unittest


ROOT=Path(__file__).resolve().parents[2]


def features(oi=0.):
    return dict(volume_zscore=0.,atr_percentile=.5,tf1h_fvg_present=0,
                tf4h_fvg_present=0,fvg_present=False,oi_change_4h=oi,
                liquidity_score=.125,rsi_14=50.,adx=0.,tf1h_bos_detected=0,
                tf4h_bos_bullish=0,tf4h_bos_bearish=0)


class ReplayFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        p=ROOT/'scripts/research/replay_features.py'
        if not p.exists(): raise AssertionError('Feature contract missing')
        s=importlib.util.spec_from_file_location('replay_features',p)
        cls.m=importlib.util.module_from_spec(s);s.loader.exec_module(cls.m)

    def test_nan_fvg_pass_is_reference_only_without_mutating_input(self):
        f=features();f['fvg_present']=float('nan')
        r=self.m.compare_features(f)
        self.assertTrue(r['reference']['any_fvg'])
        self.assertFalse(r['candidate']['any_fvg'])
        self.assertNotEqual(f['fvg_present'],f['fvg_present'])

    def test_oi_overlay_recomputes_liquidity_and_all_declared_fusion_descendants(self):
        r=self.m.compare_features(features(.02))
        f=r['candidate']['features']
        self.assertAlmostEqual(f['liquidity_score'],.165)
        self.assertAlmostEqual(f['fusion_liquidity'],.165)
        for k in ['tf1h_fusion_score','tf4h_fusion_score','tf1d_fusion_score','fusion_total']:
            self.assertAlmostEqual(f[k],.05775)
        self.assertEqual(r['reference']['features']['liquidity_score'],.125)

    def test_missing_oi_invalidates_scores_instead_of_defaulting_to_zero(self):
        f=features();del f['oi_change_4h'];f['fusion_total']=.9
        r=self.m.compare_features(f)['candidate']
        self.assertNotIn('liquidity_score',r['features'])
        self.assertNotIn('fusion_total',r['features'])
        self.assertIn('oi_change_4h',r['missing_dependencies'])

    def test_context_and_funding_not_falsely_marked_refreshed(self):
        f=features(.02);f.update(risk_temperature=.5,funding_Z=0.)
        r=self.m.compare_features(f)['candidate']
        self.assertIn('risk_temperature',r['invalidated'])
        self.assertNotIn('risk_temperature',r['features'])
        self.assertIn('funding_Z',r['uncertified'])

    def test_gate_penalty_and_boundary_behavior_are_recorded_separately(self):
        cfg={'name':'fixture','direction':'long','gate_mode':'soft','hard_gates':[
            {'feature':'derived:any_fvg','op':'bool_true'},
            {'feature':'adx','op':'min','value':0}]}
        case=dict(score=.2,threshold=.3,global_bypass=True,archetype_bypass=False,
                  enforce=True,opt_out=False,gates_passed=True)
        f=features();f['fvg_present']=float('nan')
        r=self.m.decision_probe(f,cfg,case,ROOT)
        self.assertTrue(r['reference']['live']['accepted'])
        self.assertFalse(r['reference']['backtest']['accepted'])
        self.assertEqual(r['candidate']['gate_penalty'],.5)
        self.assertEqual(r['candidate']['post_gate_score'],.1)
        self.assertFalse(r['candidate']['live']['accepted'])

    def test_processor_replay_resume_uses_actual_selected_features(self):
        from scripts.research.replay_clock import replay,Observation
        import pandas as pd
        frame=pd.DataFrame({'open':100.,'high':101.,'low':99.,'close':100.,'volume':1.},
                           index=pd.date_range('2026-01-01',periods=4,freq='h',tz='UTC'))
        obs=[Observation(id=k,feature=k,value=v,instrument='BTC',source='fixture',units='scalar',
                         version='v1',event_time=frame.index[0],available_at=frame.index[0])
             for k,v in sorted(features(.02).items())]
        kw=dict(instrument='BTC',timeframe='1h')
        full=replay(frame,obs,self.m.SelectedFeatureProcessor,**kw)
        pre=replay(frame.iloc[:2],obs,self.m.SelectedFeatureProcessor,**kw)
        post=replay(frame,obs,self.m.SelectedFeatureProcessor,checkpoint=pre['checkpoint'],**kw)
        self.assertEqual(pre['rows']+post['rows'],full['rows'])
        self.assertAlmostEqual(full['rows'][0]['output']['candidate']['features']['liquidity_score'],.165)


if __name__=='__main__': unittest.main()
