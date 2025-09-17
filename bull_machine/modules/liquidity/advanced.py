
from typing import List, Dict, Optional
from ...core.types import Series, WyckoffResult

class AdvancedLiquidityAnalyzer:
    """v1.2.1 Liquidity Analyzer with core patches applied"""
    def __init__(self, config: dict):
        self.config = config
        self.liquidity_cfg = config.get('liquidity', {})
        self.sweep_penetration = self.liquidity_cfg.get('sweep_penetration', 0.002)
        self.sweep_reclaim_bars = self.liquidity_cfg.get('sweep_reclaim_bars', 3)

    # ----- The following helpers are assumed to exist in your repo; present as stubs here -----
    def _detect_fvgs_with_gates(self, series: Series) -> List[Dict]:
        return getattr(self, "_impl_detect_fvgs_with_gates", lambda s: [])(series)

    def _detect_order_blocks(self, series: Series) -> List[Dict]:
        return getattr(self, "_impl_detect_order_blocks", lambda s: [])(series)

    def _calculate_premium_discount_context(self, series: Series) -> Dict:
        return getattr(self, "_impl_calculate_pd_context", lambda s: {"zone": "neutral", "value": 0.0})(series)

    def _determine_pressure(self, fvgs: List[Dict], obs: List[Dict], bias: str) -> str:
        return getattr(self, "_impl_determine_pressure", lambda a,b,c: "neutral")(fvgs, obs, bias)
    # -----------------------------------------------------------------------------------------

    def _detect_liquidity_sweeps(self, series: Series) -> List[Dict]:
        """Detect liquidity sweeps with tick size guard & recency window"""
        sweeps: List[Dict] = []
        if len(series.bars) < 20:
            return sweeps

        tick_size = max(self.liquidity_cfg.get('tick_size', 0.01), 1e-12)
        sweep_recent = self.liquidity_cfg.get('sweep_recent_bars', 5)

        lookback = min(50, len(series.bars))
        bars = series.bars[-lookback:]

        highs, lows = {}, {}
        for i, bar in enumerate(bars):
            high_key = round(bar.high / tick_size) * tick_size
            low_key = round(bar.low / tick_size) * tick_size
            highs.setdefault(high_key, []).append(i)
            lows.setdefault(low_key, []).append(i)

        start = max(0, len(bars) - sweep_recent)
        for i in range(start, len(bars)):
            bar = bars[i]
            # highs
            for level, idxs in highs.items():
                if len(idxs) >= 2 and bar.high > level * (1 + self.sweep_penetration):
                    reclaimed = any(
                        bars[j].close < level
                        for j in range(i+1, min(i+1+self.sweep_reclaim_bars, len(bars)))
                    )
                    if reclaimed:
                        sweeps.append({
                            'type': 'sweep', 'direction': 'bearish',
                            'index': i, 'level': level,
                            'penetration': bar.high - level, 'reclaimed': True
                        })
            # lows
            for level, idxs in lows.items():
                if len(idxs) >= 2 and bar.low < level * (1 - self.sweep_penetration):
                    reclaimed = any(
                        bars[j].close > level
                        for j in range(i+1, min(i+1+self.sweep_reclaim_bars, len(bars)))
                    )
                    if reclaimed:
                        sweeps.append({
                            'type': 'sweep', 'direction': 'bullish',
                            'index': i, 'level': level,
                            'penetration': level - bar.low, 'reclaimed': True
                        })
        return sweeps[-5:]

    def _detect_phobs(self, order_blocks: List[Dict], series: Series) -> List[Dict]:
        """Detect unmitigated HOBs with correct percentage handling"""
        phobs: List[Dict] = []
        mitigation_pct = self.liquidity_cfg.get('phob_mitigation_pct', 75)
        if mitigation_pct > 1:
            mitigation_pct /= 100.0

        for ob in order_blocks:
            mitigated = False
            for i in range(ob['index'] + 1, len(series.bars)):
                bar = series.bars[i]
                if ob['direction'] == 'bullish':
                    if bar.close < ob['bottom'] * (1 - mitigation_pct):
                        mitigated = True
                        break
                else:
                    if bar.close > ob['top'] * (1 + mitigation_pct):
                        mitigated = True
                        break
            if not mitigated:
                last = series.bars[-1].close
                phobs.append({
                    **ob,
                    'type': 'phob',
                    'distance_from_price': abs(last - ob['mid']) / max(last, 1e-12)
                })
        return phobs

    def _calculate_folp_score(self, fvgs: List[Dict], order_blocks: List[Dict],
                              pd_context: Dict, bias: str,
                              phobs: List[Dict] = None, sweeps: List[Dict] = None) -> float:
        """Dynamic liquidity scoring including PHOB proximity & sweep reclaims"""
        fvg_score = max((f['strength'] for f in fvgs
                         if (bias == 'long' and f['direction'] == 'bullish') or
                            (bias == 'short' and f['direction'] == 'bearish')), default=0)
        ob_score = max((o['strength'] for o in order_blocks
                        if (bias == 'long' and o['direction'] == 'bullish') or
                           (bias == 'short' and o['direction'] == 'bearish')), default=0)

        liquidity_score = 0.1
        if phobs and any(p.get('distance_from_price', 1) < 0.01 for p in phobs):
            liquidity_score += 0.2
        if sweeps:
            recent = sweeps[-2:] if len(sweeps) >= 2 else sweeps
            if any(s.get('reclaimed') for s in recent):
                liquidity_score += 0.3
        liquidity_score = min(liquidity_score, 1.0)

        pd_score = 0
        if (bias == 'long' and pd_context['zone'] == 'discount') or \
           (bias == 'short' and pd_context['zone'] == 'premium'):
            pd_score = pd_context['value']

        w = {'fvg': 0.4, 'orderblock': 0.3, 'liquidity': 0.2, 'premium_discount': 0.1}
        total = fvg_score*w['fvg'] + ob_score*w['orderblock'] + liquidity_score*w['liquidity'] + pd_score*w['premium_discount']
        return min(total, 1.0)

    def _find_best_candidate(self, fvgs: List[Dict], order_blocks: List[Dict],
                             phobs: List[Dict], wyckoff_result: WyckoffResult,
                             pd_context: Dict) -> Optional[Dict]:
        """Strict P/D gating; choose highest-score aligned candidate"""
        cands: List[Dict] = []
        for f in fvgs:
            if (wyckoff_result.bias == 'long' and f['direction'] == 'bullish') or \
               (wyckoff_result.bias == 'short' and f['direction'] == 'bearish'):
                cands.append({**f, 'candidate_type': 'fvg', 'score': f['strength']})
        for p in phobs:
            if (wyckoff_result.bias == 'long' and p['direction'] == 'bullish') or \
               (wyckoff_result.bias == 'short' and p['direction'] == 'bearish'):
                cands.append({**p, 'candidate_type': 'phob', 'score': p['strength'] * 1.2})

        filtered: List[Dict] = []
        for c in cands:
            if wyckoff_result.bias == 'long' and pd_context['zone'] == 'discount':
                filtered.append(c)
            elif wyckoff_result.bias == 'short' and pd_context['zone'] == 'premium':
                filtered.append(c)
            elif pd_context['zone'] == 'neutral' and c['score'] > 0.8:
                c = {**c, 'score': c['score'] * 0.8}
                filtered.append(c)

        return max(filtered, key=lambda x: x['score']) if filtered else None

    def analyze(self, series: Series, wyckoff_result: WyckoffResult) -> Dict:
        """Main liquidity analysis; returns dict consumed by fusion/risk"""
        fvgs = self._detect_fvgs_with_gates(series)
        obs = self._detect_order_blocks(series)
        phobs = self._detect_phobs(obs, series)
        sweeps = self._detect_liquidity_sweeps(series)
        pd_context = self._calculate_premium_discount_context(series)
        pressure = self._determine_pressure(fvgs, obs, wyckoff_result.bias)
        overall = self._calculate_folp_score(fvgs, obs, pd_context, wyckoff_result.bias, phobs, sweeps)
        best = self._find_best_candidate(fvgs, obs, phobs, wyckoff_result, pd_context)
        return {
            'fvgs': fvgs, 'order_blocks': obs, 'phobs': phobs, 'sweeps': sweeps,
            'premium_discount': pd_context, 'pressure': pressure,
            'overall_score': overall, 'best_candidate': best
        }
