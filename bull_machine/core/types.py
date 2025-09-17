from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class Bar:
    ts: int
    open: float
    high: float
    low: float
    close: float
    volume: float

@dataclass
class Series:
    bars: List[Bar]
    timeframe: str
    symbol: str

@dataclass
class WyckoffResult:
    regime: str                # accumulation|distribution|ranging|neutral|trending
    phase: str                 # A|B|C|D|E|none|neutral
    bias: str                  # long|short|neutral
    phase_confidence: float    # 0..1
    trend_confidence: float    # 0..1
    range: Optional[Dict]      # {low, high, mid, height, within_range, penetration} or None

@dataclass
class LiquidityResult:
    score: float               # 0..1 (bias-aligned weighting)
    pressure: str              # bullish|bearish|neutral
    fvgs: List[Dict]
    order_blocks: List[Dict]

@dataclass
class Signal:
    ts: int
    side: str                  # long|short
    confidence: float          # fusion output 0..1
    reasons: List[str]         # e.g., ["phase B, OB aligned, confidence 0.76"]
    ttl_bars: int

@dataclass
class RiskPlan:
    entry: float
    stop: float
    size: float
    tp_levels: List[Dict]      # [{'name':'tp1','r':1.0,'price':..., 'pct':33, 'action':'...'}, ...]
    rules: Dict = None         # {'be_at':'tp1','trail_at':'tp2','trail_mode':'swing','ttl_bars':18}
    stop_type: str = 'atr'     # Additional fields for v1.2.1
    risk_amount: float = 0.0
    risk_percent: float = 0.0
    profile: str = 'swing'
    expected_r: float = 2.5

@dataclass
class FusionResult:
    signal: Optional[Signal]
    breakdown: Dict            # {'scores': {...}, 'vetoes': [...], 'veto_reason': '...'}
    raw_scores: Dict          # individual module scores before weighting
