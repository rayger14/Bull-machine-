
import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


def config_hash(cfg: dict) -> str:
    blob = json.dumps(cfg, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]

@dataclass
class TradeLogRow:
    ts: object
    symbol: str
    tf: str
    action: str
    price: float
    size: float
    fee: float = 0.0
    slippage: float = 0.0
    pnl: float = 0.0
    r_multiple: float = 0.0
    wy_conf: float = 0.0
    liq_score: float = 0.0
    struct_conf: float = 0.0
    mom_conf: float = 0.0
    vol_conf: float = 0.0
    mtf_align: float = 0.0
    reasons: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['reasons'] = json.dumps(self.reasons, sort_keys=True)
        return d

@dataclass
class Summary:
    run_id: str
    cfg_hash: str
    seed: int
    commit: str = "unknown"
    metrics: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps({
            "run_id": self.run_id,
            "cfg_hash": self.cfg_hash,
            "seed": self.seed,
            "commit": self.commit,
            "metrics": self.metrics,
            "notes": self.notes
        }, indent=2, sort_keys=True)
