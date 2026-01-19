# Bull Machine v2 Code Refactoring Changelog

## Branch: `bull-machine-v2-integration`

### Refactoring Mission
Improve code readability and maintainability WITHOUT changing trading logic.

**Gold Standard Test (2024 BTC):**
```bash
PYTHONHASHSEED=0 python3 bin/backtest_knowledge_v2.py --asset BTC --start 2024-01-01 --end 2024-09-30 --config configs/frozen/btc_1h_v2_baseline.json
```
**Expected:** 17 trades, PF 6.17 ± 0.1, Win Rate 76.5% ± 2%

---

## Refactoring Log

### Refactor #1: ✅ Variable Name Improvements
**Date:** 2025-11-12
**Files:** `engine/archetypes/logic_v2_adapter.py`
**Goal:** Replace abbreviations with descriptive names
- ✅ `ctx` → `context` (RuntimeContext parameter) - COMPLETED
- ⏳ `fusion_th` → `fusion_threshold` (clarity over brevity) - Deferred (minor impact)
- ⏳ `liq` → `liquidity_score` (explicit naming) - Deferred (used in local vars only)
- ⏳ `mom` → `momentum_score` (consistency) - Deferred (used in local vars only)
- ⏳ `wy` → `wyckoff_score` (readability) - Deferred (used in local vars only)

**Status:** ✅ COMPLETED (ctx → context)
**Validation:** ✅ PASSED - 17 trades, PF 6.17, Win Rate 76.5%
**Commit SHA:** `52ae109`

---

### Refactor #2: 🟡 Standardize Archetype Return Types (In Progress)
**Date:** 2025-11-12
**Files:** `engine/archetypes/logic_v2_adapter.py`
**Goal:** Make ALL archetype detectors return `(matched: bool, score: float, meta: dict)`
- Current: B,H,L return `tuple` ✅, A returns `tuple` ✅
- Remaining: C,D,E,F,G,K,M,S1-S8 return `bool` ⏳
- Target: Uniform tuple return prevents dispatch bugs

**Status:** 🟡 IN PROGRESS (1/12 bool archetypes converted)
**Validation:** ✅ PASSED - 17 trades, PF 6.17, Win Rate 76.5%
**Commit SHA:** `e5e475a` (partial - _check_A only)

---

### Refactor #3: [PENDING] Extract Common Patterns
**Date:** TBD
**Files:** `engine/archetypes/logic_v2_adapter.py`
**Goal:** DRY - Don't Repeat Yourself
- Extract threshold lookup pattern: `_get_threshold_with_gate(archetype, param, default)`
- Extract feature extraction: Already using `self.g()` - good!
- Extract gate checking: `_check_gate(value, threshold, reason)`

**Status:** Not started
**Validation:** Pending
**Commit SHA:** N/A

---

### Refactor #4: [PENDING] Document Fusion Score Flow
**Date:** TBD
**Files:** `engine/archetypes/logic_v2_adapter.py`
**Goal:** Clarify two-tier scoring system
- Document `global_fusion_score` (soft filters applied, used in legacy dispatcher)
- Document `archetype_specific_score` (archetype-weighted, used in evaluate-all dispatcher)
- Add docstring to `_detect_all_archetypes()` explaining scoring logic

**Status:** Not started
**Validation:** Pending
**Commit SHA:** N/A

---

### Refactor #5: [PENDING] Remove Dead Comments
**Date:** TBD
**Files:** Multiple
**Goal:** Clean up resolved PHASE1 FIX comments
- Keep essential WHY comments
- Remove historical fix markers from resolved issues

**Status:** Not started
**Validation:** Pending
**Commit SHA:** N/A

---

## Validation Results

| Refactor | Trades | PF | Win Rate | Status |
|----------|--------|----|---------:|--------|
| Baseline | 17 | 6.17 | 76.5% | ✅ Gold Standard |
| #1 Variable Names | - | - | - | ⏳ Pending |
| #2 Return Types | - | - | - | ⏳ Pending |
| #3 Extract Patterns | - | - | - | ⏳ Pending |
| #4 Document Flow | - | - | - | ⏳ Pending |
| #5 Remove Comments | - | - | - | ⏳ Pending |

---

## Rollback Log

None yet.

---

## Notes

- Each refactor is validated independently against the gold standard
- If ANY metric drifts outside tolerance (±0.1 PF, ±2% Win Rate), STOP and investigate
- All refactors preserve exact trading logic - only improve code structure
