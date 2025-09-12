import logging
from typing import List, Dict
from ...core.types import Series, LiquidityResult

def analyze(series: Series, bias: str, cfg: dict) -> LiquidityResult:
    """Detect FVGs and Order Blocks; compute bias-aligned score + pressure."""
    try:
        fvgs = _detect_fvgs(series)
        order_blocks = _detect_order_blocks(series)
        score = _calculate_liquidity_score(fvgs, order_blocks, bias)
        pressure = _determine_pressure(fvgs, order_blocks, bias)
        return LiquidityResult(score=score, pressure=pressure, fvgs=fvgs, order_blocks=order_blocks)
    except Exception as e:
        logging.error(f"Liquidity analysis error: {e}")
        return LiquidityResult(score=0.0, pressure='neutral', fvgs=[], order_blocks=[])

def _detect_fvgs(series: Series) -> List[Dict]:
    fvgs: List[Dict] = []
    if len(series.bars) < 3: return fvgs
    for i in range(2, len(series.bars)):
        b1, b2, b3 = series.bars[i-2], series.bars[i-1], series.bars[i]
        # Bullish FVG: gap between b1.low and b3.high
        if b1.low > b3.high:
            gap = b1.low - b3.high
            if b3.close and gap / b3.close > 0.001:
                fvgs.append({'type':'fvg','direction':'bullish','start_idx':i-2,'end_idx':i,
                             'top': b1.low, 'bottom': b3.high, 'size': gap,
                             'strength': min((gap / b3.close)*100, 0.9)})
        # Bearish FVG: gap between b3.low and b1.high
        elif b1.high < b3.low:
            gap = b3.low - b1.high
            if b3.close and gap / b3.close > 0.001:
                fvgs.append({'type':'fvg','direction':'bearish','start_idx':i-2,'end_idx':i,
                             'top': b3.low, 'bottom': b1.high, 'size': gap,
                             'strength': min((gap / b3.close)*100, 0.9)})
    return fvgs[-10:]

def _detect_order_blocks(series: Series) -> List[Dict]:
    obs: List[Dict] = []
    if len(series.bars) < 5: return obs
    for i in range(3, len(series.bars)):
        start_price = series.bars[i-3].close
        end_price = series.bars[i].close
        if not start_price: 
            continue
        move = (end_price - start_price) / start_price
        if move > 0.02:
            for j in range(i-1, max(0, i-10), -1):
                bar = series.bars[j]
                if bar.close < bar.open:
                    obs.append({'type':'order_block','direction':'bullish','idx':j,
                                'top':bar.high,'bottom':bar.low,
                                'strength': min(abs(move)*5, 0.9),
                                'impulse_strength': abs(move)})
                    break
        elif move < -0.02:
            for j in range(i-1, max(0, i-10), -1):
                bar = series.bars[j]
                if bar.close > bar.open:
                    obs.append({'type':'order_block','direction':'bearish','idx':j,
                                'top':bar.high,'bottom':bar.low,
                                'strength': min(abs(move)*5, 0.9),
                                'impulse_strength': abs(move)})
                    break
    return obs[-8:]

def _calculate_liquidity_score(fvgs: List[Dict], obs: List[Dict], bias: str) -> float:
    if not fvgs and not obs: return 0.0
    total_score = 0.0
    total_weight = 0.0
    for f in fvgs:
        if (bias == 'long' and f['direction']=='bullish') or (bias=='short' and f['direction']=='bearish'):
            w = 0.6; total_score += f['strength'] * w; total_weight += w
    for ob in obs:
        if (bias == 'long' and ob['direction']=='bullish') or (bias=='short' and ob['direction']=='bearish'):
            w = 0.8; total_score += ob['strength'] * w; total_weight += w
    return (total_score / total_weight) if total_weight > 0 else 0.0

def _determine_pressure(fvgs: List[Dict], obs: List[Dict], bias: str) -> str:
    bull = len([x for x in fvgs + obs if x['direction']=='bullish'])
    bear = len([x for x in fvgs + obs if x['direction']=='bearish'])
    if bull > bear and bias in ['long','neutral']: return 'bullish'
    elif bear > bull and bias in ['short','neutral']: return 'bearish'
    else: return 'neutral'
