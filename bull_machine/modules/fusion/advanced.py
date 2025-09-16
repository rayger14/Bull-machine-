
from typing import Dict, List, Optional
from ...core.types import Signal, FusionResult, WyckoffResult

class AdvancedFusionEngine:
    """v1.2.1 Fusion Engine with minimal effective vetoes & trend blocks"""
    def __init__(self, config: dict):
        self.config = config
        self.fusion_cfg = config.get('fusion', {})
        self.enter_threshold = self.fusion_cfg.get('enter_threshold', 0.74)

    # Stubs (assuming real implementations exist in your repo)
    def _calculate_module_scores(self, modules_data: Dict) -> Dict[str, float]:
        return {
            "wyckoff": getattr(modules_data.get("wyckoff"), "trend_confidence", 0.0),
            "liquidity": modules_data.get("liquidity", {}).get("overall_score", 0.0),
            "structure": modules_data.get("structure", {}).get("bos_strength", 0.0),
            "momentum": modules_data.get("momentum", {}).get("score", 0.0),
            "volume": modules_data.get("volume", {}).get("score", 0.0),
            "context": modules_data.get("context", {}).get("score", 0.0),
        }

    def _calculate_fusion_score(self, scores: Dict[str, float]) -> float:
        w = self.fusion_cfg.get("weights", {
            "wyckoff": 0.30, "liquidity": 0.25, "structure": 0.20,
            "momentum": 0.10, "volume": 0.10, "context": 0.05
        })
        return sum(scores.get(k, 0.0) * w.get(k, 0.0) for k in w.keys())

    def _create_breakdown(self, scores: Dict[str, float], vetoes: List[str]) -> Dict:
        return {"scores": scores, "vetoes": vetoes}

    def _build_signal_reasons(self, modules_data: Dict, fusion_score: float) -> List[str]:
        wy = modules_data.get("wyckoff")
        wy_txt = f"Wyckoff {getattr(wy, 'phase', '?')} {getattr(wy, 'bias', '?')}" if wy else "Wyckoff n/a"
        return [wy_txt, f"Fusion score {fusion_score:.2f}"]

    def _check_vetoes(self, modules_data: Dict) -> List[str]:
        vetoes: List[str] = []
        wy = modules_data.get('wyckoff')
        liq = modules_data.get('liquidity', {})

        # Early Wyckoff phase veto (A/B) unless strong confluence
        if isinstance(wy, WyckoffResult) and wy.phase in ['A', 'B']:
            allow = (wy.phase_confidence > 0.8) or (liq.get('overall_score', 0) > 0.7) \
                    or (liq.get('sweeps') and any(s.get('reclaimed') for s in liq['sweeps']))
            if not allow:
                vetoes.append('early_wyckoff_phase')

        # Trend filter: block opposite direction unless highly aligned with Wyckoff
        if self.config.get('features', {}).get('trend_filter', False) and isinstance(wy, WyckoffResult):
            if wy.bias == 'long':
                modules_data['_trend_block'] = {'short_blocked': True}
            elif wy.bias == 'short':
                modules_data['_trend_block'] = {'long_blocked': True}

        # Volatility shock veto using recent bar distribution
        series = modules_data.get('series')
        if series and hasattr(series, 'bars') and len(series.bars) > 15:
            moves = []
            for j in range(1, min(15, len(series.bars))):
                prev, curr = series.bars[-j-1].close, series.bars[-j].close
                if prev > 0:
                    moves.append(abs(curr - prev)/prev)
            if moves:
                mu = sum(moves)/len(moves)
                var = sum((m - mu)**2 for m in moves)/len(moves)
                sigma = var ** 0.5
                if len(series.bars) >= 2:
                    last = abs(series.bars[-1].close - series.bars[-2].close) / max(series.bars[-2].close, 1e-12)
                    if sigma > 0 and last > mu + self.fusion_cfg.get('volatility_shock_sigma', 3.0)*sigma:
                        vetoes.append('volatility_shock')

        # Range suppressor (optional if available)
        try:
            from ..suppressors.range import RangeSuppressor
            rs = RangeSuppressor(self.config)
            if rs.should_suppress(wy):
                vetoes.append('range_suppressed')
        except Exception:
            pass

        return vetoes

    def fuse(self, modules_data: Dict) -> 'FusionResult':
        scores = self._calculate_module_scores(modules_data)
        vetoes = self._check_vetoes(modules_data)

        tb = modules_data.get('_trend_block', {})
        if tb:
            wy_score = scores.get('wyckoff', 0.5)
            align = self.fusion_cfg.get('trend_alignment_threshold', 0.85)
            if tb.get('long_blocked') and wy_score < align:
                vetoes.append('trend_filter_long')
            if tb.get('short_blocked') and wy_score < align:
                vetoes.append('trend_filter_short')

        fusion_score = self._calculate_fusion_score(scores)

        signal: Optional[Signal] = None
        if fusion_score >= self.enter_threshold and not vetoes:
            signal = self._generate_signal(modules_data, fusion_score)

        breakdown = self._create_breakdown(scores, vetoes)
        if vetoes:
            breakdown['veto_reason'] = vetoes[0]
        return FusionResult(signal=signal, breakdown=breakdown, raw_scores=scores)

    def _generate_signal(self, modules_data: Dict, fusion_score: float) -> Optional[Signal]:
        wy = modules_data.get('wyckoff')
        if not wy or wy.bias == 'neutral':
            return None
        return Signal(ts=0, side=wy.bias, confidence=fusion_score,
                      reasons=self._build_signal_reasons(modules_data, fusion_score), ttl_bars=20)
