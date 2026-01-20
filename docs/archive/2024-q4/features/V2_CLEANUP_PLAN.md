# Bull Machine v2.0 - Cleanup & Consolidation Plan

## Executive Summary

**Problem**: Layered advanced knowledge onto older paths → duplicated logic, partial rewires, multiple modules computing the same concepts.

**Goal**: One canonical domain pipeline per concept, one MTF builder, one fusion path, main branch that reproduces Q3 optimizer win.

**Status**: Phase 2 detector wiring complete (Wyckoff M1/M2, Macro VIX fix). Ready for spring-clean before live.

---

## Audit Findings

### Current State

**Branches**:
- `feature/phase2-regime-classifier` (current, 23 commits ahead of main)
- `feature/ml-meta-optimizer` (knowledge v2 hooks)
- `feature/macro-fusion-v186` (macro fusion)
- `feature/v1.7` (older feature branch)

**Modified Files** (uncommitted):
- `bin/build_mtf_feature_store.py` (macro extraction fix lines 240-272)
- `bin/optimize_v2_cached.py` (short logic disable, exit condition)
- `engine/wyckoff/wyckoff_engine.py` (M1/M2 integration, None returns fix)
- `engine/structure/squiggle_pattern.py` (undefined variable fix)

**Test/Doc Files** (untracked):
- 9 MVP Phase 2 markdown docs
- 7 test scripts (BOMS, macro, fusion, feature store)
- `download_vix_2024.py`, `bin/build_wyckoff_cache.py`

### Duplicate Logic Detected

#### 1. **Wyckoff** (3 implementations)

**Canonical Path**: `engine/wyckoff/wyckoff_engine.py`
- `WyckoffEngine.analyze()` - Advanced M1/M2 detection
- `detect_wyckoff_phase()` - Function API
- `WyckoffSignal` dataclass (phase, score, m1, m2, reasons)

**Duplicates to Deprecate**:
- `engine/fusion/domain_fusion.py:102` - `_wyckoff_to_score()` (converter, keep)
- `engine/timeframes/mtf_alignment.py:453` - `_mock_wyckoff_signal()` (test stub, remove)
- `engine/structure/internal_external.py:122` - `map_wyckoff_to_phase()` (old mapper, remove)
- `bin/build_wyckoff_cache.py` - Separate cache builder (redundant with MTF builder)
- `bin/run_adaptive_backtest.py:156` - `analyze_wyckoff_patterns()` (old backtest)
- `bin/production_backtest.py:266` - `_generate_wyckoff_signal()` (mock)
- `bin/live/live_mock_feed.py:281` - `_analyze_wyckoff()` (duplicate logic)

**Consolidation**: Keep `engine/wyckoff/*` as canonical. Remove all duplicates except `_wyckoff_to_score()` (needed for fusion).

#### 2. **SMC/HOB/BOMS** (Scattered across 4 modules)

**Canonical Paths**:
- `engine/smc/smc_engine.py` - SMCEngine, BOS/CHOCH detection
- `engine/smc/order_blocks.py` - OrderBlockDetector
- `engine/liquidity/hob.py` - HOBDetector (Hands-on-Back patterns)
- `engine/structure/boms_detector.py` - BOMS detection (Break of Market Structure)

**Duplicates/Confusion**:
- HOB referenced in 50+ locations across bin/, engine/fusion/, engine/liquidity/
- Multiple "HOB score" calculations:
  - `engine/fusion/domain_fusion.py:215` - `_hob_to_score()`
  - `engine/liquidity/hob.py:431` - `calculate_hob_volume_delta()`
  - `engine/liquidity/bojan_rules.py` - Bojan's demand→HOB→reaction logic
- SMC/OrderBlock overlap: Are order blocks the same as HOB?

**Consolidation Needed**:
- **Decision**: Are HOB and OrderBlocks the same concept?
  - If YES: Merge `engine/smc/order_blocks.py` → `engine/liquidity/hob.py`
  - If NO: Clarify distinction in docs, keep separate
- Rename `HOBDetector` to `LiquidityHOBDetector` for clarity
- BOMS stays separate (rare high-conviction signal using SMC + volume + FVG)

#### 3. **PTI/Psychology** (2 modules, scattered hooks)

**Canonical Path**: `engine/psychology/pti.py`
- `calculate_pti()` - Psychology Trap Index (RSI div + volume exhaustion)
- `PTISignal` dataclass

**Also**: `engine/psychology/fakeout_intensity.py`
- `detect_fakeout_intensity()` - Wick trap + failed breakout detection
- `FakeoutSignal` dataclass

**Fusion Hooks**: `engine/fusion/knowledge_hooks.py`
- `apply_pti()` (line 245)
- `apply_fakeout_intensity()` (line 298)

**Consolidation**: Already clean! PTI and Fakeout are separate concepts. No duplicates found.

**Action**: Create single `engine/psychology/api.py` that exports both.

#### 4. **FRVP/Volume** (Clean)

**Canonical Path**: `engine/volume/frvp.py`
- `calculate_frvp()` - Fixed Range Volume Profile
- `FRVPProfile` dataclass

**Fusion Hook**: `engine/fusion/knowledge_hooks.py:349` - `apply_frvp()`

**Consolidation**: Already clean! No duplicates.

**Action**: Create `engine/volume/api.py` that exports FRVP.

#### 5. **Macro** (4 overlapping modules!)

**Current Modules**:
1. `engine/context/macro_engine.py` - MacroContextEngine (old)
2. `engine/context/macro_pulse.py` - MacroPulseEngine (newer, complex regime detection)
3. `engine/exits/macro_echo.py` - `analyze_macro_echo()` (7-day correlation-based)
4. `engine/ml/macro_signals_enhanced.py` - MacroSignalsEnhanced (ML version)

**Duplicate `MacroRegime` Enums**:
- `engine/context/analysis.py:19` - MacroRegime (ACCUMULATION, MARKUP, DISTRIBUTION, MARKDOWN, TRANSITION, NEUTRAL)
- `engine/context/macro_pulse.py:21` - MacroRegime (RISK_ON, RISK_OFF, STAGFLATION, NEUTRAL)

**Confusion**: Two different regime taxonomies!

**Consolidation Needed**:
1. **Decision**: Which regime taxonomy to keep?
   - **Wyckoff-style** (ACCUMULATION/MARKUP/DISTRIBUTION/MARKDOWN) - aligns with Wyckoff phases
   - **Risk-style** (RISK_ON/RISK_OFF/STAGFLATION/NEUTRAL) - aligns with macro indicators

2. **Recommendation**: Keep BOTH but clarify:
   - `analyze_macro_echo()` (exits/macro_echo.py) → For 7-day exit timing
   - `MacroPulseEngine` (context/macro_pulse.py) → For regime classification
   - Deprecate old `macro_engine.py`

3. **Canonical API**:
   - `engine/macro/api.py` exports both `analyze_macro_echo()` and `MacroPulseEngine`

#### 6. **Fusion** (Knowledge Hooks Integration)

**Current Structure**:
- `engine/fusion/domain_fusion.py` - Base fusion (computes domain scores)
- `engine/fusion/knowledge_hooks.py` - v2 hooks (structure, psych, volume, macro deltas)
- `engine/fusion/advanced_fusion.py` - Delta channels (momentum + macro + HOB boosts)
- `engine/fusion.py` - Old fusion (legacy?)

**Consolidation**:
- Keep `domain_fusion.py` + `knowledge_hooks.py`
- Deprecate `fusion.py` (old API)
- Merge `advanced_fusion.py` → `knowledge_hooks.py` (delta channels are hooks)

---

## Cleanup Plan

### Phase 1: Create Canonical Domain APIs

Create single entry point per domain under `engine/domains/`:

```
engine/
├─ domains/              # NEW - Canonical domain engines
│  ├─ __init__.py
│  ├─ wyckoff/
│  │  ├─ __init__.py
│  │  ├─ api.py           # detect_wyckoff(df_1d, df_4h) -> pd.DataFrame
│  │  └─ engine.py        # WyckoffEngine (advanced M1/M2)
│  ├─ smc_hob/
│  │  ├─ __init__.py
│  │  ├─ api.py           # analyze_smc_hob(df) -> pd.DataFrame
│  │  ├─ smc.py           # SMCEngine (BOS/CHOCH/FVG)
│  │  ├─ hob.py           # HOBDetector
│  │  └─ boms.py          # detect_boms()
│  ├─ psychology/
│  │  ├─ __init__.py
│  │  ├─ api.py           # compute_pti(), compute_fakeout()
│  │  ├─ pti.py
│  │  └─ fakeout.py
│  ├─ volume/
│  │  ├─ __init__.py
│  │  ├─ api.py           # compute_frvp()
│  │  └─ frvp.py
│  ├─ momentum/
│  │  ├─ __init__.py
│  │  └─ api.py           # compute_momentum()
│  └─ macro/
│     ├─ __init__.py
│     ├─ api.py           # compute_macro_echo(), compute_macro_pulse()
│     ├─ echo.py          # 7-day correlation (for exits)
│     └─ pulse.py         # Regime classification
├─ fusion/
│  ├─ domain_fusion.py    # Base fusion (calls domain APIs)
│  └─ knowledge_hooks.py  # v2 hooks (structure, psych, volume, macro)
├─ exits/
│  └─ multi_modal_exits.py
├─ _deprecated/           # NEW - Old code moved here
│  ├─ fusion.py           # Old fusion API
│  ├─ macro_engine.py     # Old macro engine
│  └─ README.md           # Explains why deprecated
```

### Phase 2: Refactor MTF Feature Store Builder

**File**: `bin/build_mtf_feature_store.py`

**Current Imports** (scattered):
```python
from engine.wyckoff.wyckoff_engine import WyckoffEngine
from engine.smc.smc_engine import SMCEngine
from engine.liquidity.hob import HOBDetector
from engine.structure.boms_detector import detect_boms
from engine.psychology.pti import calculate_pti
from engine.psychology.fakeout_intensity import detect_fakeout_intensity
from engine.volume.frvp import calculate_frvp
from engine.exits.macro_echo import analyze_macro_echo
```

**New Imports** (canonical APIs):
```python
from engine.domains.wyckoff.api import detect_wyckoff
from engine.domains.smc_hob.api import analyze_smc_hob
from engine.domains.psychology.api import compute_pti, compute_fakeout
from engine.domains.volume.api import compute_frvp
from engine.domains.momentum.api import compute_momentum
from engine.domains.macro.api import compute_macro_echo
```

**Benefits**:
- Single source of truth per domain
- Easy to swap implementations (e.g., ML-based PTI vs heuristic)
- No more scattered imports
- Contract enforced: all APIs return DataFrames with expected columns

### Phase 3: Consolidation Steps

#### 3.1 Wyckoff
1. Move `engine/wyckoff/*` → `engine/domains/wyckoff/`
2. Create `engine/domains/wyckoff/api.py`:
   ```python
   def detect_wyckoff(df_1d: pd.DataFrame, df_4h: Optional[pd.DataFrame] = None,
                      config: Optional[Dict] = None) -> pd.DataFrame:
       """
       Returns 1D-indexed DataFrame with columns:
       - wyckoff_phase: str
       - wyckoff_score: float
       - wyckoff_m1: float
       - wyckoff_m2: float
       - wyckoff_reasons: list[str] (as JSON)
       """
   ```
3. Move deprecated files to `engine/_deprecated/`:
   - `timeframes/mtf_alignment.py:453` (_mock_wyckoff_signal)
   - `structure/internal_external.py:122` (map_wyckoff_to_phase)
4. Remove `bin/build_wyckoff_cache.py` (redundant with MTF builder)

#### 3.2 SMC/HOB/BOMS
1. Create `engine/domains/smc_hob/`
2. Move files:
   - `engine/smc/*` → `engine/domains/smc_hob/smc.py`
   - `engine/liquidity/hob.py` → `engine/domains/smc_hob/hob.py`
   - `engine/structure/boms_detector.py` → `engine/domains/smc_hob/boms.py`
3. Create `engine/domains/smc_hob/api.py`:
   ```python
   def analyze_smc_hob(df: pd.DataFrame, timeframe: str = '4H',
                       config: Optional[Dict] = None) -> Dict:
       """
       Returns dict with:
       - bos_detected: bool
       - choch_detected: bool
       - fvg_present: bool
       - boms_signal: bool
       - hob_score: float
       - order_blocks: List[OrderBlock]
       """
   ```

#### 3.3 Psychology (PTI + Fakeout)
1. Create `engine/domains/psychology/`
2. Move files:
   - `engine/psychology/pti.py` → `engine/domains/psychology/pti.py`
   - `engine/psychology/fakeout_intensity.py` → `engine/domains/psychology/fakeout.py`
3. Create `engine/domains/psychology/api.py`:
   ```python
   def compute_pti(df: pd.DataFrame, timeframe: str = '4H',
                   config: Optional[Dict] = None) -> PTISignal:
       """PTI detection (RSI divergence + volume exhaustion)"""

   def compute_fakeout(df: pd.DataFrame, lookback: int = 30,
                       config: Optional[Dict] = None) -> FakeoutSignal:
       """Fakeout detection (wick traps + failed breakouts)"""
   ```

#### 3.4 Volume (FRVP)
1. Create `engine/domains/volume/`
2. Move `engine/volume/frvp.py` → `engine/domains/volume/frvp.py`
3. Create `engine/domains/volume/api.py`:
   ```python
   def compute_frvp(df: pd.DataFrame, lookback: int = 100,
                    config: Optional[Dict] = None) -> FRVPProfile:
       """Fixed Range Volume Profile"""
   ```

#### 3.5 Macro
1. Create `engine/domains/macro/`
2. Move files:
   - `engine/exits/macro_echo.py` → `engine/domains/macro/echo.py`
   - `engine/context/macro_pulse.py` → `engine/domains/macro/pulse.py`
3. Deprecate:
   - `engine/context/macro_engine.py` → `engine/_deprecated/macro_engine.py`
4. Create `engine/domains/macro/api.py`:
   ```python
   def compute_macro_echo(macro_data: Dict[str, pd.Series], lookback: int = 7,
                          config: Optional[Dict] = None) -> MacroEchoSignal:
       """7-day correlation-based macro exit signal"""

   def compute_macro_pulse(macro_data: Dict[str, pd.DataFrame],
                           config: Optional[Dict] = None) -> MacroPulse:
       """Macro regime classification (RISK_ON/RISK_OFF/STAGFLATION/NEUTRAL)"""
   ```

### Phase 4: Branch Consolidation

1. Create integration branch:
   ```bash
   git checkout main
   git pull
   git checkout -b integration/knowledge-v2
   ```

2. Merge in order (least conflicts → most):
   ```bash
   git merge --no-ff feature/phase2-regime-classifier  # Current work (detector wiring)
   git merge --no-ff feature/ml-meta-optimizer         # Knowledge hooks
   git merge --no-ff feature/macro-fusion-v186         # Macro fusion (if needed)
   ```

3. Run parity gate (see Phase 7)

4. Open PR to main:
   - Title: "v2.0.0: Unify domains + MTF builder + knowledge hooks"
   - Description: Canonical domain APIs, Phase 2 detector wiring complete, VIX fix

### Phase 5: CI & Pre-commit Hooks

**Install**:
```bash
pip install pre-commit ruff black mypy pytest
pre-commit install
```

**`.pre-commit-config.yaml`**:
```yaml
repos:
- repo: https://github.com/psf/black
  rev: 24.8.0
  hooks: [{id: black}]
- repo: https://github.com/astral-sh/ruff-pre-commit
  rev: v0.6.8
  hooks: [{id: ruff, args: [--fix]}]
- repo: https://github.com/pre-commit/mirrors-mypy
  rev: v1.10.0
  hooks: [{id: mypy, additional_dependencies: [pandas-stubs]}]
```

**`.github/workflows/ci.yml`**:
```yaml
name: ci
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.11' }
      - run: pip install -r requirements.txt
      - run: pip install -r requirements-dev.txt || true
      - run: black --check .
      - run: ruff check .
      - run: mypy engine bin --ignore-missing-imports || true
      - run: pytest tests/ -q --tb=short
```

### Phase 6: Acceptance Gates

#### Gate A: Parity (Baseline)

**Goal**: Ensure cleanup doesn't break existing optimizer results.

**Baseline**: Q3 2024 BTC - Best trial +$433 PNL, 16 trades, PF 2.69, Sharpe 1.41

**Test**:
```bash
# Before cleanup (current feature store)
python3 bin/optimize_v2_cached.py --asset BTC --start 2024-07-01 --end 2024-09-30 --trials 20

# After cleanup (rebuilt feature store with canonical APIs)
python3 bin/build_mtf_feature_store.py --asset BTC --start 2024-07-01 --end 2024-09-30
python3 bin/optimize_v2_cached.py --asset BTC --start 2024-07-01 --end 2024-09-30 --trials 20
```

**Pass Criteria**:
- PnL ±2% ($425-$442)
- Trade count ±20% (13-19 trades)
- PF ≥ 2.5

#### Gate B: Optimizer (Multi-Asset)

**Goal**: Validate optimizer works on multiple assets with canonical APIs.

**Test**:
```bash
python3 bin/build_mtf_feature_store.py --asset BTC --start 2024-07-01 --end 2024-09-30
python3 bin/build_mtf_feature_store.py --asset ETH --start 2024-07-01 --end 2024-09-30

python3 bin/optimize_v2_cached.py --asset BTC --start 2024-07-01 --end 2024-09-30 --trials 50
python3 bin/optimize_v2_cached.py --asset ETH --start 2024-07-01 --end 2024-09-30 --trials 50
```

**Pass Criteria**:
- BTC: Best trial PF ≥ 2.5, trades ≥ 12
- ETH: Best trial PF ≥ 2.0, trades ≥ 20

#### Gate C: Sanity (Feature Variation)

**Goal**: Ensure no domain columns are constant/empty in rebuilt stores.

**Test**:
```bash
python3 scripts/verify_features.py data/features_mtf/BTC_1H_2024-07-01_to_2024-09-30.parquet
python3 scripts/verify_features.py data/features_mtf/ETH_1H_2024-07-01_to_2024-09-30.parquet
```

**Pass Criteria**:
- Wyckoff: 8+ unique phases, m1/m2 varying
- Macro: VIX level varying (4 unique), correlation varying
- BOMS: Can be all False (legitimately rare)
- PTI/Fakeout: ≥2 unique values
- FRVP: hvn_count varying

---

## Success Criteria

### "Done" Looks Like:

✅ One canonical domain pipeline per concept (Wyckoff, SMC/HOB, Momentum, Psychology, Volume, Macro)

✅ One MTF feature store builder calling only canonical APIs

✅ One fusion path (`domain_fusion.py` + `knowledge_hooks.py`)

✅ Main branch reproduces Q3 optimizer win (parity gate passed)

✅ CI blocks re-introduction of duplicates (ruff, black, mypy, pytest)

✅ All tests passing (unit, scenario, parity)

✅ Documentation updated (README, domain API docs)

---

## Timeline

### Week 1: Domain Consolidation
- Day 1-2: Create `engine/domains/` structure, move Wyckoff
- Day 3-4: Consolidate SMC/HOB/BOMS, Psychology, Volume
- Day 5: Consolidate Macro

### Week 2: Integration & Testing
- Day 1-2: Refactor MTF builder to use canonical APIs
- Day 3: Create integration branch, merge features
- Day 4: Run acceptance gates (parity, optimizer, sanity)
- Day 5: Fix any gate failures, retest

### Week 3: CI & Docs
- Day 1: Set up pre-commit hooks, GitHub Actions
- Day 2-3: Update documentation (README, API docs)
- Day 4: Final testing, create PR to main
- Day 5: Review, merge to main

### Week 4: Production Readiness
- Day 1-3: Build all asset feature stores (BTC, ETH, SPY, TSLA)
- Day 4-5: Run 200-trial optimizer sweeps
- Tag v2.0.0-rc1, shadow trade for 1 week

---

## Risk Mitigation

### What Could Go Wrong:

1. **Breaking parity gate**: Cleanup changes optimizer results
   - Mitigation: Keep old feature store, compare before/after
   - Rollback: Revert to `feature/phase2-regime-classifier` if parity fails

2. **Merge conflicts**: Three feature branches diverged
   - Mitigation: Merge one at a time, run tests after each
   - Resolution: Manual conflict resolution, prefer newest code

3. **CI failures**: Code doesn't pass linting/type checks
   - Mitigation: Run ruff/black/mypy locally before committing
   - Fix: Auto-fix with `ruff check --fix`, `black .`

4. **Feature variation regression**: Consolidated APIs return constant values
   - Mitigation: Run `verify_features.py` after each consolidation
   - Debug: Add logging to domain APIs to trace computation

---

## Next Steps (After Cleanup)

### Immediate:
1. Build multi-asset feature stores (BTC, ETH, SPY, TSLA) - 2024 full year
2. Run 200-trial optimizer sweeps per asset
3. Tag v2.0.0-rc1 on integration branch

### Short-Term:
4. Phase 3: Fast vectorized backtest (cached features)
5. Phase 4: Live shadow runner (paper trading)

### Medium-Term:
6. ML meta-optimizer (learn domain weights dynamically)
7. Multi-asset portfolio optimization
8. Live execution (tiny capital)

---

## Appendix: File Inventory

### Modified Files (Must Commit Before Cleanup)
- `bin/build_mtf_feature_store.py` (macro extraction fix)
- `bin/optimize_v2_cached.py` (short logic disable)
- `engine/wyckoff/wyckoff_engine.py` (M1/M2 integration)
- `engine/structure/squiggle_pattern.py` (undefined variable fix)

### Documentation Files (Keep)
- `MVP_PHASE2_VIX_FIX_COMPLETE.md` (VIX data download)
- `MVP_PHASE2_WIRING_COMPLETE.md` (detector investigation)
- `MVP_PHASE2_SUCCESS.md` (optimizer results)

### Test Files (Keep for Regression)
- `test_boms_diagnostic.py` (BOMS condition funnel)
- `test_macro_extraction.py` (macro extraction validation)
- `test_feature_store_scores.py` (domain variation check)

### Scripts (Move to scripts/)
- `download_vix_2024.py` → `scripts/download_vix_2024.py`
- `bin/build_wyckoff_cache.py` → DELETE (redundant with MTF builder)

---

**Document Version**: 1.0
**Created**: October 18, 2025
**Author**: Bull Machine Team
**Status**: READY FOR EXECUTION
