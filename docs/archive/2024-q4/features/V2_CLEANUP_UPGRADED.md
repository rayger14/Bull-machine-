# Bull Machine v2.0 - Cleanup Plan (UPGRADED)

## Executive Summary

**Problem**: Layered advanced knowledge onto older paths → duplicated logic, partial rewires, multiple modules computing the same concepts.

**Goal**: One canonical domain pipeline per concept, one MTF builder, one fusion path, main branch that reproduces Q3 optimizer win.

**Status**: Phase 2 detector wiring complete (Wyckoff M1/M2, Macro VIX fix). Commits frozen. Ready for baseline builds.

**Upgrades from Original Plan**:
1. ✅ HOB vs OrderBlocks decision made: Same family, unified API
2. ✅ Macro taxonomy decision made: Single enum (risk_on/risk_off/neutral/crisis)
3. ✅ Schema versioning added to feature stores
4. ✅ Config migration map for v1→v2 translation
5. ✅ Deprecation policy with CI enforcement
6. ✅ Smoke-test matrix (4 assets × 2 periods × 1 trial)
7. ✅ Domain telemetry (compute time, NaN%, unique states)

---

## Key Decisions (RESOLVED)

### 1. HOB vs OrderBlocks → Same Family ✅

**Decision**: Treat as same concept family with refined taxonomy.

**Unified API** (`engine/domains/smc_hob/order_blocks.py`):
```python
@dataclass
class OrderBlock:
    """Unified order block (visible/hidden/partial)"""
    kind: Literal['visible', 'hidden', 'partial_hidden']  # HOB is 'hidden' or 'partial_hidden'
    role: Literal['supply', 'demand']                      # Bullish/bearish
    confirmation: Literal['none', 'mitigation', 'displacement']

    # Standard fields
    timestamp: pd.Timestamp
    price_high: float
    price_low: float
    volume: float
    strength: float  # 0-1

    # Context
    swing_point: Optional[SwingPoint]
    fvg: Optional[FairValueGap]
```

**Migration Strategy**:
- Keep module name `order_blocks.py` (not `hob.py`)
- All 50+ HOB references map to `OrderBlock(kind='hidden', ...)`
- `HOBDetector` becomes `OrderBlockDetector` with `detect(kind='all' | 'hidden' | 'visible')`
- Adapter in `engine/_deprecated/hob_adapter.py` for legacy callers

**Benefits**:
- Retires 50+ scattered HOB references cleanly
- Single detector for all order block types
- Easy to extend (e.g., add `'breaker_block'` kind later)

### 2. Macro Regime Taxonomy → Single Enum ✅

**Decision**: Keep `risk_on | risk_off | neutral | crisis` as the ONE regime enum.

**Implementation**:
```python
# engine/domains/macro/api.py
class MacroRegime(Enum):
    """Macro regime classification (canonical)"""
    RISK_ON = "risk_on"          # DXY down, yields stable, VIX <15
    RISK_OFF = "risk_off"        # DXY up, yields up, VIX >20
    NEUTRAL = "neutral"          # Mixed signals
    CRISIS = "crisis"            # VIX >30, systemic stress

def map_wyckoff_phase_to_macro(wyckoff_phase: str) -> MacroRegime:
    """Legacy adapter for Wyckoff-style regime names"""
    mapping = {
        'accumulation': MacroRegime.RISK_ON,
        'markup': MacroRegime.RISK_ON,
        'distribution': MacroRegime.RISK_OFF,
        'markdown': MacroRegime.RISK_OFF,
        'transition': MacroRegime.NEUTRAL,
    }
    return mapping.get(wyckoff_phase.lower(), MacroRegime.NEUTRAL)
```

**Migration Strategy**:
- Rename old `engine/context/analysis.py:MacroRegime` → `WyckoffMarketPhase` (internal use only)
- Keep `engine/context/macro_pulse.py:MacroRegime` as canonical
- Add `map_wyckoff_phase_to_macro()` for legacy callers
- MTF builder uses only `MacroRegime` (canonical)

**Benefits**:
- Single source of truth for macro regime
- No enum name collision
- Legacy code can still use Wyckoff-style names via adapter

### 3. Schema Versioning → Feature Store Metadata ✅

**Implementation**:
```python
# bin/build_mtf_feature_store.py (after parquet save)
def write_schema_metadata(parquet_path: str, metadata: Dict):
    """Write schema version + stats to adjacent JSON file"""
    meta_path = parquet_path.replace('.parquet', '_metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

# Usage after df.to_parquet(output_path)
write_schema_metadata(output_path, {
    'schema_version': 'mtf-2.0.0',
    'created_at': datetime.now().isoformat(),
    'asset': asset,
    'start_date': start_date,
    'end_date': end_date,
    'bars': len(df),
    'features': len(df.columns),
    'domains': {
        'wyckoff': {'m1_signals': m1_count, 'm2_signals': m2_count},
        'macro': {'vix_level_unique': vix_unique_count},
        'boms': {'signals': boms_count}
    }
})
```

**Validation** (`scripts/verify_features.py`):
```python
def verify_schema_version(parquet_path: str) -> bool:
    """Check schema version matches expected"""
    meta_path = parquet_path.replace('.parquet', '_metadata.json')
    if not os.path.exists(meta_path):
        print(f"⚠️  No metadata file: {meta_path}")
        return False

    with open(meta_path) as f:
        meta = json.load(f)

    expected = 'mtf-2.0.0'
    actual = meta.get('schema_version', 'unknown')

    if actual != expected:
        print(f"❌ Schema version mismatch: expected {expected}, got {actual}")
        return False

    print(f"✅ Schema version: {actual}")
    return True
```

### 4. Config Migration Map → v1 to v2 Translation ✅

**File**: `configs/migrations/v1_to_v2.json`

```json
{
  "version": "1.0.0",
  "description": "v1.x → v2.0 config migration map",
  "flag_renames": {
    "use_hob": "use_order_blocks",
    "hob_weight": "order_block_weight",
    "macro_regime_mode": "macro_regime_style"
  },
  "param_renames": {
    "hob_volume_threshold": "order_block_volume_threshold",
    "hob_confirmation_bars": "order_block_confirmation_bars"
  },
  "enum_mappings": {
    "macro_regime": {
      "accumulation": "risk_on",
      "markup": "risk_on",
      "distribution": "risk_off",
      "markdown": "risk_off",
      "transition": "neutral"
    }
  },
  "removed_flags": [
    "use_legacy_wyckoff",
    "enable_old_macro_engine"
  ],
  "added_flags": {
    "use_m1_m2_signals": true,
    "schema_version": "mtf-2.0.0"
  }
}
```

**Auto-migration** (`bin/optimize_v2_cached.py`):
```python
def migrate_config_v1_to_v2(config: Dict) -> Dict:
    """Auto-translate v1 config to v2"""
    with open('configs/migrations/v1_to_v2.json') as f:
        migration = json.load(f)

    migrated = config.copy()

    # Rename flags
    for old, new in migration['flag_renames'].items():
        if old in migrated:
            migrated[new] = migrated.pop(old)

    # Rename params
    for old, new in migration['param_renames'].items():
        if old in migrated:
            migrated[new] = migrated.pop(old)

    # Map enum values
    for key, mapping in migration['enum_mappings'].items():
        if key in migrated and migrated[key] in mapping:
            migrated[key] = mapping[migrated[key]]

    # Remove deprecated flags
    for flag in migration['removed_flags']:
        migrated.pop(flag, None)

    # Add new defaults
    migrated.update(migration['added_flags'])

    return migrated
```

### 5. Deprecation Policy → CI Enforcement ✅

**File**: `engine/_deprecated/README.md`

```markdown
# Deprecated Code - Do Not Import

All code in this directory is deprecated and scheduled for removal.

## Policy

- **Retention**: Kept for 2 releases (e.g., v2.0.0 → removed in v2.2.0)
- **Usage**: Only importable from test code (blocked by CI for production)
- **Migration**: See `configs/migrations/` for auto-translation tools

## Files

### `hob_adapter.py`
- **Deprecated in**: v2.0.0
- **Remove in**: v2.2.0
- **Replacement**: `engine/domains/smc_hob/order_blocks.OrderBlockDetector`
- **Migration**: Use `kind='hidden'` for HOB patterns

### `macro_engine.py`
- **Deprecated in**: v2.0.0
- **Remove in**: v2.2.0
- **Replacement**: `engine/domains/macro/pulse.MacroPulseEngine`

### `fusion.py`
- **Deprecated in**: v2.0.0
- **Remove in**: v2.2.0
- **Replacement**: `engine/fusion/domain_fusion.py`

## Testing

Old integration tests may import from `_deprecated/` to verify parity.
All new code MUST use canonical APIs.
```

**CI Rule** (`.github/workflows/ci.yml`):
```yaml
- name: Check for deprecated imports in production code
  run: |
    # Fail if production code imports from _deprecated
    if grep -r "from engine._deprecated" bin/ engine/ --include="*.py" --exclude-dir=tests; then
      echo "❌ Production code imports from engine._deprecated/ (use canonical APIs)"
      exit 1
    fi
    echo "✅ No deprecated imports in production code"
```

**Ruff Rule** (`.ruff.toml`):
```toml
[lint.per-file-ignores]
# Allow deprecated imports only in tests
"tests/**" = ["F401"]  # Unused import (for deprecated modules)

[lint]
# Ban deprecated imports in production
banned-from = ["engine._deprecated"]
banned-from-msg = "Use canonical APIs from engine/domains/ instead"
```

### 6. Smoke-Test Matrix → Fast Pre-Check ✅

**Purpose**: Cheap and fast sanity check before long optimizer runs.

**Implementation** (`scripts/smoke_test.sh`):
```bash
#!/usr/bin/env bash
set -euo pipefail

echo "================================================================================"
echo "Smoke Test Matrix: 4 assets × 2 periods × 1 trial"
echo "================================================================================"

ASSETS=("BTC" "ETH" "SPY" "TSLA")
PERIODS=(
  "2024-07-01 2024-09-30"  # Q3 2024 (in-sample)
  "2025-01-01 2025-03-31"  # Q1 2025 (out-of-sample)
)

for asset in "${ASSETS[@]}"; do
  for period in "${PERIODS[@]}"; do
    read -r start end <<< "$period"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Testing: $asset | $start to $end"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Build feature store (if not exists)
    FEATURE_STORE="data/features_v2/${asset}_1H_${start}_to_${end}.parquet"
    if [[ ! -f "$FEATURE_STORE" ]]; then
      python3 bin/build_mtf_feature_store.py \
        --asset "$asset" \
        --start "$start" \
        --end "$end" \
        --out "$FEATURE_STORE"
    fi

    # Run 1 optimizer trial (smoke test)
    python3 bin/optimize_v2_cached.py \
      --asset "$asset" \
      --start "$start" \
      --end "$end" \
      --features "$FEATURE_STORE" \
      --trials 1 \
      --tag "smoke_test" || {
        echo "❌ Smoke test failed: $asset $start to $end"
        exit 1
      }
  done
done

echo ""
echo "================================================================================"
echo "✅ All Smoke Tests Passed"
echo "================================================================================"
```

**Usage**:
```bash
bash scripts/smoke_test.sh  # 8 quick tests (4 assets × 2 periods)
```

**Pass Criteria**:
- All 8 feature stores build without error
- All 8 optimizer trials complete (no crashes)
- PnL is non-zero (validates trade execution)

### 7. Domain Telemetry → Logging & Debugging ✅

**Implementation** (example for Wyckoff):

```python
# engine/domains/wyckoff/api.py
import logging
import time
from collections import defaultdict

logger = logging.getLogger(__name__)

class DomainTelemetry:
    """Track domain computation stats"""
    def __init__(self, domain_name: str):
        self.domain = domain_name
        self.stats = defaultdict(int)
        self.compute_times = []

    def record(self, bars_processed: int, compute_time: float,
               nan_count: int, unique_states: int):
        """Record batch computation stats"""
        self.stats['total_bars'] += bars_processed
        self.stats['total_nan'] += nan_count
        self.stats['total_unique_states'] += unique_states
        self.compute_times.append(compute_time)

    def summary(self) -> Dict:
        """Get summary stats"""
        return {
            'domain': self.domain,
            'total_bars': self.stats['total_bars'],
            'nan_pct': self.stats['total_nan'] / self.stats['total_bars'] * 100,
            'unique_states': self.stats['total_unique_states'],
            'avg_compute_time_ms': sum(self.compute_times) / len(self.compute_times) * 1000,
            'total_compute_time_s': sum(self.compute_times)
        }

# Global telemetry tracker
_telemetry = DomainTelemetry('wyckoff')

def detect_wyckoff(df_1d: pd.DataFrame, df_4h: Optional[pd.DataFrame] = None,
                   config: Optional[Dict] = None) -> pd.DataFrame:
    """Detect Wyckoff patterns with telemetry"""

    start_time = time.time()

    # ... actual detection logic ...

    # Record telemetry
    compute_time = time.time() - start_time
    nan_count = result['wyckoff_score'].isna().sum()
    unique_states = result['wyckoff_phase'].nunique()

    _telemetry.record(
        bars_processed=len(result),
        compute_time=compute_time,
        nan_count=nan_count,
        unique_states=unique_states
    )

    logger.info(f"Wyckoff: {len(result)} bars, {unique_states} phases, "
                f"{compute_time*1000:.1f}ms")

    return result

def get_telemetry() -> Dict:
    """Get Wyckoff domain telemetry"""
    return _telemetry.summary()
```

**Integration** (`scripts/verify_features.py`):
```python
# After loading parquet
from engine.domains.wyckoff.api import get_telemetry as get_wyckoff_telemetry
from engine.domains.smc_hob.api import get_telemetry as get_smc_telemetry
from engine.domains.macro.api import get_telemetry as get_macro_telemetry

print("\n" + "=" * 80)
print("Domain Telemetry")
print("=" * 80)

for get_telem, name in [
    (get_wyckoff_telemetry, 'Wyckoff'),
    (get_smc_telemetry, 'SMC/HOB'),
    (get_macro_telemetry, 'Macro')
]:
    telem = get_telem()
    print(f"\n{name}:")
    print(f"  Total bars: {telem['total_bars']}")
    print(f"  NaN rate: {telem['nan_pct']:.2f}%")
    print(f"  Unique states: {telem['unique_states']}")
    print(f"  Avg compute time: {telem['avg_compute_time_ms']:.2f}ms/bar")
    print(f"  Total time: {telem['total_compute_time_s']:.2f}s")
```

**Benefits**:
- Fast debugging (see which domain is slow/broken)
- Performance monitoring (track compute time per bar)
- Data quality checks (NaN rate, unique states)
- Easy to add to CI (fail if NaN rate > 10%)

---

## Canonical Domain Structure (FINAL)

```
engine/
├─ domains/                          # Canonical APIs (v2.0.0+)
│  ├─ __init__.py
│  ├─ wyckoff/
│  │  ├─ __init__.py
│  │  ├─ api.py                      # detect_wyckoff() + telemetry
│  │  └─ engine.py                   # WyckoffEngine (advanced M1/M2)
│  ├─ smc_hob/                       # SMC + OrderBlocks (unified)
│  │  ├─ __init__.py
│  │  ├─ api.py                      # analyze_smc_hob() + telemetry
│  │  ├─ smc.py                      # SMCEngine (BOS/CHOCH/FVG)
│  │  ├─ order_blocks.py             # OrderBlockDetector (visible/hidden/partial)
│  │  └─ boms.py                     # detect_boms() (rare signal)
│  ├─ psychology/
│  │  ├─ __init__.py
│  │  ├─ api.py                      # compute_pti(), compute_fakeout() + telemetry
│  │  ├─ pti.py                      # PTI (RSI divergence + volume exhaustion)
│  │  └─ fakeout.py                  # Fakeout (wick traps + failed breakouts)
│  ├─ volume/
│  │  ├─ __init__.py
│  │  ├─ api.py                      # compute_frvp() + telemetry
│  │  └─ frvp.py                     # Fixed Range Volume Profile
│  ├─ momentum/
│  │  ├─ __init__.py
│  │  └─ api.py                      # compute_momentum() + telemetry
│  └─ macro/
│     ├─ __init__.py
│     ├─ api.py                      # compute_macro_echo(), compute_macro_pulse() + telemetry
│     ├─ echo.py                     # MacroEchoSignal (7-day correlation for exits)
│     └─ pulse.py                    # MacroPulse (regime: risk_on/risk_off/neutral/crisis)
├─ fusion/
│  ├─ domain_fusion.py               # Base fusion (calls domain APIs)
│  └─ knowledge_hooks.py             # v2 hooks (structure, psych, volume, macro deltas)
├─ _deprecated/                      # OLD code (CI blocks imports from production)
│  ├─ README.md                      # Deprecation policy
│  ├─ hob_adapter.py                 # Legacy HOB → OrderBlock adapter
│  ├─ macro_engine.py                # Old macro engine
│  └─ fusion.py                      # Old fusion API
```

**MTF Builder Refactor**:

```python
# bin/build_mtf_feature_store.py

# BEFORE (scattered imports)
from engine.wyckoff.wyckoff_engine import WyckoffEngine
from engine.smc.smc_engine import SMCEngine
from engine.liquidity.hob import HOBDetector
from engine.structure.boms_detector import detect_boms
from engine.psychology.pti import calculate_pti
from engine.psychology.fakeout_intensity import detect_fakeout_intensity
from engine.volume.frvp import calculate_frvp
from engine.exits.macro_echo import analyze_macro_echo

# AFTER (canonical APIs)
from engine.domains.wyckoff.api import detect_wyckoff
from engine.domains.smc_hob.api import analyze_smc_hob
from engine.domains.psychology.api import compute_pti, compute_fakeout
from engine.domains.volume.api import compute_frvp
from engine.domains.momentum.api import compute_momentum
from engine.domains.macro.api import compute_macro_echo
```

---

## Do-Now Checklist (Execution Order)

### ✅ Step 1: Freeze Current State
```bash
# DONE
git add -A
git commit -m "fix(mvp): Phase 2 complete - Wyckoff M1/M2 + Macro VIX fix"
git commit -m "chore: add v2.0 cleanup plan + audit tools"
bash scripts/audit_repo.sh > reports/audit_$(date +%Y%m%d).txt
```

### Step 2: Build Baseline Feature Stores (2024 Full Year)
```bash
# Create output directory
mkdir -p data/features_v2 reports/baselines_2024

# Build all asset stores
bash scripts/build_all_stores.sh 2024

# Verify each store
for asset in BTC ETH SPY TSLA; do
  python3 scripts/verify_features.py data/features_v2/${asset}_1H_2024.parquet
done
```

**Expected Output**:
- `data/features_v2/BTC_1H_2024.parquet` (~8760 bars, 69 features)
- `data/features_v2/ETH_1H_2024.parquet` (~8760 bars, 69 features)
- `data/features_v2/SPY_1H_2024.parquet` (~1600 bars RTH, 69 features)
- `data/features_v2/TSLA_1H_2024.parquet` (~1600 bars RTH, 69 features)

### Step 3: Run Baseline Optimizers (200 Trials Each)
```bash
# BTC (24/7)
python3 bin/optimize_v2_cached.py \
  --asset BTC \
  --start 2024-01-01 \
  --end 2024-12-31 \
  --features data/features_v2/BTC_1H_2024.parquet \
  --trials 200 \
  --tag baseline_2024 \
  > reports/baselines_2024/BTC_200trials.log 2>&1

# ETH (24/7)
python3 bin/optimize_v2_cached.py \
  --asset ETH \
  --start 2024-01-01 \
  --end 2024-12-31 \
  --features data/features_v2/ETH_1H_2024.parquet \
  --trials 200 \
  --tag baseline_2024 \
  > reports/baselines_2024/ETH_200trials.log 2>&1

# SPY (RTH only)
python3 bin/optimize_v2_cached.py \
  --asset SPY \
  --start 2024-01-01 \
  --end 2024-12-31 \
  --features data/features_v2/SPY_1H_2024.parquet \
  --trials 200 \
  --tag baseline_2024 \
  --rth \
  > reports/baselines_2024/SPY_200trials.log 2>&1

# TSLA (RTH only)
python3 bin/optimize_v2_cached.py \
  --asset TSLA \
  --start 2024-01-01 \
  --end 2024-12-31 \
  --features data/features_v2/TSLA_1H_2024.parquet \
  --trials 200 \
  --tag baseline_2024 \
  --rth \
  > reports/baselines_2024/TSLA_200trials.log 2>&1

# Archive best configs
cp reports/optuna_results/BTC_best_configs.json reports/baselines_2024/BTC_baseline_configs.json
cp reports/optuna_results/ETH_best_configs.json reports/baselines_2024/ETH_baseline_configs.json
cp reports/optuna_results/SPY_best_configs.json reports/baselines_2024/SPY_baseline_configs.json
cp reports/optuna_results/TSLA_best_configs.json reports/baselines_2024/TSLA_baseline_configs.json
```

**Expected Runtime**: 2-4 hours per asset (can run in parallel)

**Pass Criteria**:
- BTC: Best PF ≥ 2.0, trades ≥ 30
- ETH: Best PF ≥ 1.8, trades ≥ 40
- SPY: Best PF ≥ 1.5, trades ≥ 20
- TSLA: Best PF ≥ 1.5, trades ≥ 15

### Step 4: Create Integration Branch
```bash
git checkout main
git pull
git checkout -b integration/knowledge-v2
git merge --no-ff feature/phase2-regime-classifier -m "Merge Phase 2 detector wiring"
git merge --no-ff feature/ml-meta-optimizer -m "Merge knowledge v2 hooks"
```

### Step 5: Create Canonical Domain APIs (Thin Wrappers)
```bash
# Create directory structure
mkdir -p engine/domains/{wyckoff,smc_hob,psychology,volume,momentum,macro}
mkdir -p engine/_deprecated
mkdir -p configs/migrations

# Create __init__.py files
touch engine/domains/__init__.py
touch engine/domains/{wyckoff,smc_hob,psychology,volume,momentum,macro}/__init__.py
```

**Wyckoff API** (`engine/domains/wyckoff/api.py`):
```python
"""Wyckoff Domain - Canonical API (v2.0.0+)"""
from typing import Dict, Optional
import pandas as pd
from engine.wyckoff.wyckoff_engine import WyckoffEngine, detect_wyckoff_phase

def detect_wyckoff(df_1d: pd.DataFrame, df_4h: Optional[pd.DataFrame] = None,
                   config: Optional[Dict] = None) -> pd.DataFrame:
    """
    Detect Wyckoff patterns across 1D/4H timeframes.

    Returns 1D-indexed DataFrame with columns:
    - wyckoff_phase: str (accumulation, markup, distribution, markdown, spring, upthrust, transition)
    - wyckoff_score: float (0-1)
    - wyckoff_m1: float (spring strength 0-1)
    - wyckoff_m2: float (markup continuation strength 0-1)
    - wyckoff_reasons: str (JSON list of detection reasons)

    Uses advanced M1/M2 detection (requires 150-300 days context).
    """
    # Thin wrapper - delegates to existing working implementation
    engine = WyckoffEngine(config or {})

    # ... implementation (forward to current wyckoff_engine.py)

    return result
```

**SMC/HOB API** (`engine/domains/smc_hob/api.py`):
```python
"""SMC & Order Blocks Domain - Canonical API (v2.0.0+)"""
from typing import Dict, Optional, List
import pandas as pd
from engine.smc.smc_engine import SMCEngine
from engine.smc.order_blocks import OrderBlockDetector
from engine.structure.boms_detector import detect_boms

def analyze_smc_hob(df: pd.DataFrame, timeframe: str = '4H',
                    config: Optional[Dict] = None) -> Dict:
    """
    Analyze SMC + Order Blocks (visible/hidden/partial).

    Returns dict:
    {
        'bos_detected': bool,
        'choch_detected': bool,
        'fvg_present': bool,
        'boms_signal': bool,
        'order_blocks': List[OrderBlock],  # kind ∈ {visible, hidden, partial_hidden}
        'smc_score': float (0-1)
    }
    """
    # ... implementation
```

**Continue for all 6 domains...**

### Step 6: Refactor MTF Builder to Use Canonical APIs
```bash
# Backup current builder
cp bin/build_mtf_feature_store.py bin/build_mtf_feature_store_v1_backup.py

# Edit imports to use canonical APIs
# (Manual edit or script)
```

**Run Parity Gate**:
```bash
# Rebuild Q3 2024 BTC with canonical APIs
python3 bin/build_mtf_feature_store.py \
  --asset BTC \
  --start 2024-07-01 \
  --end 2024-09-30 \
  --out data/features_v2/BTC_1H_Q3_2024_canonical.parquet

# Verify feature variation
python3 scripts/verify_features.py data/features_v2/BTC_1H_Q3_2024_canonical.parquet

# Run 20-trial optimizer
python3 bin/optimize_v2_cached.py \
  --asset BTC \
  --start 2024-07-01 \
  --end 2024-09-30 \
  --features data/features_v2/BTC_1H_Q3_2024_canonical.parquet \
  --trials 20 \
  --tag canonical_api_test

# Compare to baseline (+$433 PNL, 16 trades, PF 2.69)
# Pass if: PnL $425-$442, trades 13-19, PF ≥ 2.5
```

### Step 7: Consolidate Domains (One at a Time)

#### 7.1 Wyckoff
```bash
# Move current implementation
mv engine/wyckoff/* engine/domains/wyckoff/

# Update api.py to use local imports
# Move duplicates to _deprecated/
mv bin/build_wyckoff_cache.py engine/_deprecated/
# (Add adapter for _mock_wyckoff_signal, map_wyckoff_to_phase)

# Run parity gate
bash scripts/smoke_test.sh

# Commit
git add engine/domains/wyckoff engine/_deprecated
git commit -m "refactor(wyckoff): consolidate to engine/domains/wyckoff"
```

#### 7.2 SMC/HOB
```bash
# Move SMC
mv engine/smc/* engine/domains/smc_hob/

# Move order blocks (rename HOBDetector → OrderBlockDetector)
mv engine/liquidity/hob.py engine/domains/smc_hob/order_blocks.py
# Edit to use OrderBlock(kind='visible'|'hidden'|'partial_hidden')

# Move BOMS
mv engine/structure/boms_detector.py engine/domains/smc_hob/boms.py

# Create legacy adapter
# engine/_deprecated/hob_adapter.py

# Run parity gate
bash scripts/smoke_test.sh

# Commit
git add engine/domains/smc_hob engine/_deprecated
git commit -m "refactor(smc_hob): unify SMC + OrderBlocks + BOMS"
```

#### 7.3 Macro
```bash
# Move macro echo
mv engine/exits/macro_echo.py engine/domains/macro/echo.py

# Move macro pulse
mv engine/context/macro_pulse.py engine/domains/macro/pulse.py

# Deprecate old macro engine
mv engine/context/macro_engine.py engine/_deprecated/

# Create api.py with single MacroRegime enum

# Run parity gate
bash scripts/smoke_test.sh

# Commit
git add engine/domains/macro engine/_deprecated
git commit -m "refactor(macro): consolidate echo + pulse, single MacroRegime enum"
```

#### 7.4 Psychology & Volume (Already Clean)
```bash
# Just relocate
mv engine/psychology/* engine/domains/psychology/
mv engine/volume/* engine/domains/volume/

# Create api.py wrappers

# Run parity gate
bash scripts/smoke_test.sh

# Commit
git add engine/domains/{psychology,volume}
git commit -m "refactor(psychology,volume): relocate to engine/domains"
```

### Step 8: Wire CI + Pre-commit Hooks
```bash
# Create .pre-commit-config.yaml (see earlier section)
# Create .github/workflows/ci.yml (see earlier section)
# Create .ruff.toml (see earlier section)

pip install pre-commit ruff black mypy pytest
pre-commit install

# Test locally
pre-commit run --all-files

# Commit
git add .pre-commit-config.yaml .github/workflows/ci.yml .ruff.toml
git commit -m "ci: add pre-commit hooks + GitHub Actions"
```

### Step 9: Run All Parity Gates
```bash
# Gate A - Parity (Baseline)
python3 bin/optimize_v2_cached.py \
  --asset BTC \
  --start 2024-07-01 \
  --end 2024-09-30 \
  --features data/features_v2/BTC_1H_Q3_2024_canonical.parquet \
  --trials 20

# Compare to baseline: +$433 PNL, 16 trades, PF 2.69
# Pass: PnL $425-$442, trades 13-19, PF ≥ 2.5

# Gate B - Optimizer (Multi-Asset)
bash scripts/smoke_test.sh  # 4 assets × 2 periods × 1 trial

# Gate C - Sanity (Feature Variation)
for asset in BTC ETH SPY TSLA; do
  python3 scripts/verify_features.py data/features_v2/${asset}_1H_2024.parquet
done
```

**All gates must pass before merging to main.**

### Step 10: Tag RC & Shadow Trade Plan
```bash
# Open PR
git push -u origin integration/knowledge-v2
gh pr create \
  --title "v2.0.0: Canonical domain APIs + Phase 2 detector wiring" \
  --body "$(cat <<EOF
## Summary
- Consolidated 6 domains to engine/domains/ with canonical APIs
- Phase 2 detector wiring complete (Wyckoff M1/M2, Macro VIX fix)
- All parity gates passed (baseline, multi-asset, sanity)
- CI/pre-commit hooks enforcing deprecation policy

## Acceptance Gates
- ✅ Gate A (Parity): BTC Q3 +$433 PNL, 16 trades, PF 2.69 (±2%)
- ✅ Gate B (Multi-Asset): 4 assets × 2 periods smoke tests pass
- ✅ Gate C (Sanity): All domain features varying (no constants)

## Breaking Changes
- Import paths changed: \`engine/wyckoff\` → \`engine/domains/wyckoff/api\`
- HOBDetector → OrderBlockDetector (kind='hidden' for HOB patterns)
- MacroRegime unified (risk_on/risk_off/neutral/crisis)

## Migration
- v1 configs auto-translate via \`configs/migrations/v1_to_v2.json\`
- Deprecated code in \`engine/_deprecated/\` (2 release retention)
EOF
)"

# After PR approval, merge to main
git checkout main
git merge --no-ff integration/knowledge-v2
git tag -a v2.0.0-rc1 -m "v2.0.0 Release Candidate 1"
git push --tags

# Shadow trade (1 week, paper only)
# Use best configs from baselines_2024/
python3 bin/live/hybrid_runner.py \
  --asset BTC \
  --start $(date -d "yesterday" +%Y-%m-%d) \
  --end $(date -d "+7 days" +%Y-%m-%d) \
  --config reports/baselines_2024/BTC_baseline_configs.json \
  --mode paper \
  --out logs/shadow_trading/BTC_week1/

# Repeat for ETH, SPY, TSLA
```

---

## Timeline (4 Weeks)

### Week 1: Build Baselines (THIS WEEK)
- ✅ Day 1: Freeze current state, commit Phase 2 work
- Day 2-3: Build all asset feature stores (2024 full year)
- Day 4-5: Run 200-trial baseline optimizers (parallel)
- Archive: `reports/baselines_2024/`

### Week 2: Create Canonical APIs
- Day 1: Create integration branch, merge features
- Day 2-3: Create `engine/domains/` with thin wrapper APIs
- Day 4: Refactor MTF builder imports, run parity gate
- Day 5: Smoke tests, commit canonical APIs

### Week 3: Domain Consolidation
- Day 1-2: Consolidate Wyckoff, run parity gate
- Day 3: Consolidate SMC/HOB, run parity gate
- Day 4: Consolidate Macro, run parity gate
- Day 5: Relocate Psychology/Volume, run all gates

### Week 4: CI + Shadow Trading
- Day 1: Wire CI/pre-commit hooks
- Day 2: Final parity gates, open PR
- Day 3: Review, merge to main, tag v2.0.0-rc1
- Day 4-5: Shadow trade 4 assets (paper), monitor parity

---

## Success Criteria

✅ One canonical domain API per concept (6 domains)

✅ One MTF feature store builder (imports only canonical APIs)

✅ One fusion path (domain_fusion + knowledge_hooks)

✅ Main branch reproduces baseline optimizer wins (parity gate)

✅ CI blocks re-introduction of duplicates (ruff + pre-commit)

✅ Schema versioning in all feature stores (`_metadata.json`)

✅ Config migration v1→v2 auto-translation

✅ Deprecation policy enforced (2 release retention)

✅ All tests passing (smoke, unit, parity)

✅ Shadow trade parity (backtest vs live match within 5%)

---

## Files to Create

### Configuration
1. `configs/migrations/v1_to_v2.json` - Config migration map
2. `.pre-commit-config.yaml` - Pre-commit hooks
3. `.github/workflows/ci.yml` - GitHub Actions CI
4. `.ruff.toml` - Ruff linter config

### Documentation
5. `engine/_deprecated/README.md` - Deprecation policy
6. `engine/domains/README.md` - Domain API documentation

### Scripts
7. `scripts/smoke_test.sh` - Smoke test matrix (existing: audit, verify, build_all)

### Canonical APIs (to be created)
8. `engine/domains/wyckoff/api.py`
9. `engine/domains/smc_hob/api.py`
10. `engine/domains/psychology/api.py`
11. `engine/domains/volume/api.py`
12. `engine/domains/momentum/api.py`
13. `engine/domains/macro/api.py`

---

## Direct Answers to Key Questions

**"Are feature stores properly built for MTF?"**
✅ Yes - Phase 2 detector wiring complete. Wyckoff M1/M2 varying, Macro VIX varying. After canonical API refactor, same logic just cleaner imports.

**"Did the optimizer use all domains?"**
✅ Yes - Uses Wyckoff (M1/M2), SMC (BOS/CHOCH/FVG), BOMS, FRVP, PTI, Fakeout, Macro Echo. Canonical APIs won't change domain set, just import paths.

**"Should we clean repo before live?"**
✅ Yes, but AFTER baselines locked. Timeline: Build first (Week 1) → Clean (Weeks 2-3) → Shadow trade (Week 4) → Tiny live (if parity passes).

**"What's the risk?"**
✅ Low - Parity gates at every step. Canonical APIs are thin wrappers (delegate to current working code). Can rollback to Phase 2 state if gates fail.

**"How long will cleanup take?"**
✅ 3 weeks after baselines complete (4 weeks total). Week 1 can start NOW (build stores).

---

**Document Version**: 2.0 (UPGRADED)
**Created**: October 18, 2025
**Author**: Bull Machine Team
**Status**: READY FOR EXECUTION - START WITH BASELINES
