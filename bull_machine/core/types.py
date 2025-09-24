from dataclasses import dataclass
from typing import Dict, List, Optional


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
    risk_amount: float = 0.0   # Dollar risk
    risk_percent: float = 0.0  # Percent of account
    profile: str = ''          # Risk profile
    expected_r: float = 0.0    # Expected R-multiple

# ==== v1.3 MTF Sync Types ====

@dataclass
class BiasCtx:
    """Bias context for a specific timeframe"""
    tf: str                   # timeframe string
    bias: str                 # 'long'|'short'|'neutral'
    confirmed: bool           # 2-bar confirm (or chosen rule)
    strength: float           # 0..1 composite
    bars_confirmed: int
    ma_distance: float        # rel distance to key MA
    trend_quality: float      # fraction of bars trending
    ma_slope: float = 0.0    # MA slope for trend strength

@dataclass
class RangeCtx:
    """Range context with premium/discount zones"""
    tf: str
    low: float
    high: float
    mid: float
    time_in_range: int = 0    # bars inside range
    compression: float = 0.0  # 0..1, 1=max compress
    tests_high: int = 0
    tests_low: int = 0
    last_test: str = 'none'   # 'high'|'low'|'none'
    breakout_potential: float = 0.0

    @property
    def height(self) -> float:
        return self.high - self.low

    @property
    def premium_zone(self) -> tuple:
        return (self.mid + 0.35 * self.height, self.high)

    @property
    def discount_zone(self) -> tuple:
        return (self.low, self.mid - 0.35 * self.height)

    @property
    def equilibrium_zone(self) -> tuple:
        return (self.mid - 0.2 * self.height, self.mid + 0.2 * self.height)

@dataclass
class SyncReport:
    """Multi-timeframe sync decision report"""
    htf: BiasCtx
    mtf: BiasCtx
    ltf_bias: str             # 'long'|'short'|'neutral' from LTF analyzer
    nested_ok: bool
    eq_magnet: bool
    desync: bool
    decision: str             # 'allow'|'raise'|'veto'
    threshold_bump: float     # e.g. +0.05
    alignment_score: float    # 0..1
    notes: List[str]

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'htf_bias': self.htf.bias,
            'mtf_bias': self.mtf.bias,
            'ltf_bias': self.ltf_bias,
            'nested_ok': self.nested_ok,
            'eq_magnet': self.eq_magnet,
            'desync': self.desync,
            'decision': self.decision,
            'threshold_bump': self.threshold_bump,
            'alignment_score': self.alignment_score,
            'notes': self.notes
        }

@dataclass
class FusionResult:
    signal: Optional[Signal]
    breakdown: Dict            # {'scores': {...}, 'vetoes': [...], 'veto_reason': '...'}
    raw_scores: Dict          # individual module scores before weighting
