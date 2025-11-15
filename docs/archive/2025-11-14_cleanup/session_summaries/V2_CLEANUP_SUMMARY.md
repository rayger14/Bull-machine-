# Bull Machine v2.0 - Cleanup Summary & Next Steps

## What We Found

### Audit Complete ✅

Ran comprehensive repo audit (`reports/repo_audit_v2_cleanup.txt`) and found:

**Duplicate Logic** (Classic "Many Paths" Problem):
- **Wyckoff**: 3 implementations across engine/fusion/timeframes/structure + duplicate builders
- **SMC/HOB**: Scattered across 4 modules (engine/smc, engine/liquidity, engine/structure) with 50+ HOB references
- **Macro**: 4 overlapping modules with 2 different regime taxonomies (!)
- **PTI/Fakeout**: Clean (already consolidated)
- **FRVP**: Clean (already consolidated)

**Branch Situation**:
- 4 unmerged feature branches (phase2-regime-classifier, ml-meta-optimizer, macro-fusion-v186, v1.7)
- Current branch: `feature/phase2-regime-classifier` (23 commits ahead of main)
- Modified files: 4 (MTF builder macro fix, optimizer tweaks, Wyckoff M1/M2, squiggle fix)
- Untracked: 9 MVP Phase 2 docs + 7 test scripts

**Key Finding**: We've been layering advanced knowledge (M1/M2, knowledge hooks) onto older paths without deprecating the old ones → duplicated logic that will cause maintenance hell.

---

## What We Created

### 1. Comprehensive Cleanup Plan

**File**: `V2_CLEANUP_PLAN.md` (27KB, 690 lines)

**Contents**:
- Detailed audit findings (duplicate symbols per domain)
- Target repo layout (`engine/domains/` canonical structure)
- Phase-by-phase consolidation steps (6 domains)
- Branch merge strategy (integration/knowledge-v2)
- CI/pre-commit setup (.pre-commit-config.yaml, .github/workflows/ci.yml)
- 3 acceptance gates (parity, optimizer, sanity)
- Timeline (3 weeks to v2.0.0-rc1)
- Risk mitigation strategies

### 2. Helper Scripts

**Created**:
1. `scripts/audit_repo.sh` - Repo audit (git status, branches, duplicate symbols, import graph)
2. `scripts/verify_features.py` - Feature store validation (checks for constant columns, domain stats)
3. `scripts/build_all_stores.sh` - Build all asset feature stores (BTC, ETH, SPY, TSLA)

**Usage**:
```bash
# Run audit
bash scripts/audit_repo.sh > reports/audit_$(date +%Y%m%d).txt

# Verify feature store
python3 scripts/verify_features.py data/features_mtf/BTC_1H_2024-07-01_to_2024-09-30.parquet

# Build all stores (2024 full year)
bash scripts/build_all_stores.sh 2024
```

### 3. Audit Report

**File**: `reports/repo_audit_v2_cleanup.txt`

**Key Findings**:
- Wyckoff: 13 locations (canonical: engine/wyckoff/wyckoff_engine.py)
- HOB: 50+ references (canonical: engine/liquidity/hob.py, but overlaps with SMC order_blocks.py)
- Macro: 4 modules with 2 different `MacroRegime` enums (analysis.py vs macro_pulse.py)
- Import graph shows scattered dependencies (no clear canonical API)

---

## Proposed Canonical Structure

### Target Layout

```
engine/
├─ domains/              # NEW - One canonical path per domain
│  ├─ wyckoff/
│  │  ├─ api.py          # detect_wyckoff(df_1d, df_4h) -> pd.DataFrame
│  │  └─ engine.py       # WyckoffEngine (advanced M1/M2)
│  ├─ smc_hob/
│  │  ├─ api.py          # analyze_smc_hob(df) -> Dict
│  │  ├─ smc.py          # SMCEngine (BOS/CHOCH/FVG)
│  │  ├─ hob.py          # HOBDetector
│  │  └─ boms.py         # detect_boms() (rare high-conviction signal)
│  ├─ psychology/
│  │  ├─ api.py          # compute_pti(), compute_fakeout()
│  │  ├─ pti.py
│  │  └─ fakeout.py
│  ├─ volume/
│  │  ├─ api.py          # compute_frvp()
│  │  └─ frvp.py
│  ├─ momentum/
│  │  └─ api.py          # compute_momentum()
│  └─ macro/
│     ├─ api.py          # compute_macro_echo(), compute_macro_pulse()
│     ├─ echo.py         # 7-day correlation (exits)
│     └─ pulse.py        # Regime classification
├─ fusion/
│  ├─ domain_fusion.py   # Base fusion (calls domain APIs)
│  └─ knowledge_hooks.py # v2 hooks
├─ _deprecated/          # OLD code moved here with README
```

### MTF Builder Refactor

**Before** (scattered imports):
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

**After** (canonical APIs):
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
- Clear contract (all APIs return expected schema)
- Easy to swap implementations (ML vs heuristic)
- No accidental divergence

---

## Key Decisions Needed

### 1. HOB vs OrderBlocks - Are They the Same?

**Current State**:
- `engine/smc/order_blocks.py` - OrderBlockDetector (BOS + strong volume bar)
- `engine/liquidity/hob.py` - HOBDetector (Hands-on-Back pattern, 50+ references)

**Question**: Same concept or different?

**Options**:
- **A**: Same → Merge `order_blocks.py` into `hob.py`, rename to `LiquidityHOBDetector`
- **B**: Different → Keep separate, clarify distinction in docs

**Recommendation**: Needs user clarification. Likely Option A (merge).

### 2. Macro Regime Taxonomy - Which One?

**Current State**:
- `engine/context/analysis.py:19` - MacroRegime (ACCUMULATION, MARKUP, DISTRIBUTION, MARKDOWN, TRANSITION, NEUTRAL)
- `engine/context/macro_pulse.py:21` - MacroRegime (RISK_ON, RISK_OFF, STAGFLATION, NEUTRAL)

**Question**: Two different taxonomies for same concept!

**Recommendation**: Keep BOTH but clarify:
- **Wyckoff-style** (ACCUMULATION/MARKUP/...) → aligns with Wyckoff phases
- **Risk-style** (RISK_ON/RISK_OFF/...) → aligns with macro indicators (DXY, VIX, yields)
- Rename one to avoid confusion (e.g., `WyckoffRegime` vs `MacroRiskRegime`)

### 3. When to Execute Cleanup?

**Options**:
- **A**: Before multi-asset feature stores (this week) → Clean house first
- **B**: After multi-asset feature stores (next week) → Get data first, then clean
- **C**: Incremental (parallel) → Clean one domain at a time while building stores

**Recommendation**: **Option C** (incremental)
- Week 1: Build Q3 2024 stores for all assets (BTC, ETH, SPY, TSLA) with CURRENT code
- Week 2: Create `engine/domains/` structure, consolidate Wyckoff first (biggest impact)
- Week 3: Consolidate remaining domains, run parity gates
- Week 4: Merge to main, tag v2.0.0-rc1

**Rationale**: De-risk by validating current code works across all assets BEFORE refactoring.

---

## Immediate Next Steps (This Week)

### Option 1: Build First, Clean Later (Recommended)

**Goal**: Get multi-asset feature stores working with CURRENT code to establish baseline.

**Steps**:
1. ✅ Commit current changes (MTF builder macro fix, Wyckoff M1/M2)
   ```bash
   git add bin/build_mtf_feature_store.py bin/optimize_v2_cached.py engine/wyckoff/wyckoff_engine.py engine/structure/squiggle_pattern.py
   git commit -m "fix(mvp): Phase 2 complete - Wyckoff M1/M2 + macro VIX fix"
   ```

2. Build Q3 2024 feature stores for all 4 assets:
   ```bash
   bash scripts/build_all_stores.sh  # Builds BTC, ETH, SPY, TSLA for 2024
   ```

3. Run 200-trial optimizer sweeps per asset:
   ```bash
   python3 bin/optimize_v2_cached.py --asset BTC --start 2024-07-01 --end 2024-09-30 --trials 200
   python3 bin/optimize_v2_cached.py --asset ETH --start 2024-07-01 --end 2024-09-30 --trials 200
   python3 bin/optimize_v2_cached.py --asset SPY --start 2024-07-01 --end 2024-09-30 --trials 200
   python3 bin/optimize_v2_cached.py --asset TSLA --start 2024-07-01 --end 2024-09-30 --trials 200
   ```

4. THEN start cleanup (next week)

**Timeline**: 3-4 days for builds + optimizations, THEN 2-3 weeks for cleanup.

### Option 2: Clean First, Build Later

**Goal**: Get canonical structure in place BEFORE building multi-asset stores.

**Steps**:
1. Create `engine/domains/` structure
2. Consolidate Wyckoff (move engine/wyckoff → engine/domains/wyckoff)
3. Refactor MTF builder to use canonical Wyckoff API
4. Run parity gate (Q3 BTC should still produce +$433 PNL)
5. Repeat for remaining domains
6. THEN build multi-asset stores with clean code

**Timeline**: 2-3 weeks for cleanup, THEN 3-4 days for builds.

---

## Recommendation

### Go with Option 1: Build First, Clean Later

**Reasoning**:
1. **De-risk**: Validate CURRENT code works across all assets before refactoring
2. **Baseline**: Establish optimizer results for BTC/ETH/SPY/TSLA to compare against after cleanup
3. **Parallel work**: Can start domain consolidation WHILE optimizer trials run (they take hours)
4. **User feedback**: Get multi-asset results this week (as requested), cleanup next week

**This Week**:
- Day 1-2: Build all asset feature stores (2024 full year)
- Day 3-5: Run 200-trial optimizer sweeps per asset (can run in parallel)
- Save best configs per asset to `configs/v2/`

**Next Week**:
- Start domain consolidation (Wyckoff first)
- Create `engine/domains/` structure
- Run parity gates after each consolidation
- Integration branch + PR to main

**Week 3**:
- Complete remaining domain consolidations
- CI/pre-commit setup
- Final parity gates
- Tag v2.0.0-rc1

**Week 4**:
- Shadow trade best configs (logs only, 1 week)
- If parity passes, flip to tiny capital

---

## Files Created

### Documentation
1. `V2_CLEANUP_PLAN.md` - Comprehensive cleanup plan (27KB, 690 lines)
2. `V2_CLEANUP_SUMMARY.md` - This file (executive summary + next steps)
3. `reports/repo_audit_v2_cleanup.txt` - Full audit report

### Scripts
4. `scripts/audit_repo.sh` - Repo audit tool
5. `scripts/verify_features.py` - Feature store validator
6. `scripts/build_all_stores.sh` - Multi-asset builder

**All scripts are executable and ready to use.**

---

## Questions for User

1. **HOB vs OrderBlocks**: Same concept or different? (affects consolidation strategy)

2. **Macro Regime Taxonomy**: Keep both (Wyckoff-style + Risk-style) or merge? (rename to avoid collision)

3. **Timing**: Build first (Option 1 - recommended) or Clean first (Option 2)?

4. **Scope**: Full 2024 year for all assets, or Q3 2024 only? (full year = better data, Q3 = faster)

5. **Assets**: BTC, ETH, SPY, TSLA confirmed? Any others (AAPL, QQQ, GLD)?

---

## Current Status

✅ **Phase 2 Detector Wiring**: Complete (Wyckoff M1/M2, Macro VIX fix, BOMS verified)

✅ **Audit**: Complete (duplicate logic mapped, cleanup plan documented)

✅ **Scripts**: Created (audit, verify, build_all_stores)

⏳ **Next**: Awaiting user decision on Option 1 (build first) vs Option 2 (clean first)

---

**Document Version**: 1.0
**Created**: October 18, 2025
**Author**: Bull Machine Team
**Status**: READY FOR USER DECISION
