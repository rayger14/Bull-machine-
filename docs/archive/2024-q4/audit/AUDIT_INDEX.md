# Archetype Parameter System Audit - Document Index

**Audit Date:** 2025-11-08  
**Status:** CRITICAL - Blocking optimization  
**Scope:** Complete architectural review of parameter flow

---

## READ THESE IN ORDER

### 1. START HERE: AUDIT_EXECUTIVE_SUMMARY.md (8 pages)
**Quick overview of the problem and what needs to be fixed**

Contains:
- The core problem (2 incompatible code paths)
- Top 5 critical issues with evidence
- Immediate action items (3 phases)
- Files that need changes
- Estimated effort: 3-7 days

**Time to read:** 10 minutes

---

### 2. THEN: AUDIT_QUICK_REFERENCE.md (4 pages)
**Quick lookup guide for daily reference during fixes**

Contains:
- Critical files table
- Zero-variance bug diagram
- 6 immediate fixes (priority order)
- Parameter flow checklist
- Quick reference format

**Time to read:** 5 minutes

---

### 3. DEEP DIVE: COMPREHENSIVE_ARCHETYPE_AUDIT.md (35 pages)
**Complete architectural analysis with all details**

Contains:
- Complete parameter flow diagram
- All 5 critical inconsistencies documented
- 70+ hardcoded values listed by location
- 20+ parameter name mismatches cataloged
- All file:line references
- Full root cause analysis
- Config reading paths explained
- Naming mismatch catalog
- Dead code inventory

**Time to read:** 30-45 minutes

---

## QUICK STATS

| Metric | Value |
|--------|-------|
| Total pages of audit | 47 |
| Critical issues found | 5 |
| Hardcoded values | 70+ |
| Parameter mismatches | 20+ |
| Naming conflicts | 11 archetypes |
| Lines of duplicate code | 1,597 |
| Estimated fix time | 3-7 days |
| Files that need changes | 5-6 |

---

## KEY ISSUES AT A GLANCE

### Issue #1: ZERO-VARIANCE BUG
- **Location:** optuna_trap_v2.py:118, logic.py:46
- **Impact:** BLOCKS OPTIMIZATION - all trials identical
- **Root cause:** Optimizer writes to config['archetypes']['trap_within_trend'], code reads from config['thresholds']['H']
- **Fix time:** 2-3 hours

### Issue #2: DUPLICATE CODE
- **Location:** logic.py (961 lines) vs logic_v2_adapter.py (636 lines)
- **Impact:** Code conflict, unclear which is used
- **Root cause:** Incomplete migration from old to new archetype system
- **Fix time:** 1-2 days (after Issue #1 resolved)

### Issue #3: 70+ HARDCODED VALUES
- **Location:** logic_v2_adapter.py lines 385-623
- **Impact:** Config values ignored, defaults always used
- **Root cause:** Every _check_X method has hardcoded thresholds
- **Fix time:** 1-2 days

### Issue #4: NAMING CHAOS
- **Location:** All archetype files
- **Impact:** Lookup failures across all layers
- **Root cause:** 3 different naming systems (letter codes, slugs, registry names)
- **Fix time:** 2-3 days

### Issue #5: PARAMETER MISMATCHES
- **Location:** All config files
- **Impact:** Config parameter names don't match code reads
- **Root cause:** Code reads 'fusion_threshold', config has 'fusion'
- **Fix time:** 1-2 days

---

## FILE LOCATIONS (QUICK REFERENCE)

```
AUDIT REPORTS (3 documents):
  /Bull-machine-/AUDIT_EXECUTIVE_SUMMARY.md ← START HERE
  /Bull-machine-/AUDIT_QUICK_REFERENCE.md ← DAILY REFERENCE
  /Bull-machine-/COMPREHENSIVE_ARCHETYPE_AUDIT.md ← DEEP DIVE

CODE THAT NEEDS FIXING:
  /engine/archetypes/logic.py ← DELETE or REFACTOR
  /engine/archetypes/logic_v2_adapter.py ← REMOVE HARDCODING
  /engine/archetypes/threshold_policy.py ← ADD LOGGING
  /engine/runtime/context.py ← ADD VALIDATION
  /engine/archetypes/param_accessor.py ← EXTEND TO ALL

OPTIMIZERS:
  /bin/optuna_trap_v2.py:118 ← WRITES PARAMETERS
  /bin/optuna_wick_trap_v2.py:113 ← WRITES PARAMETERS

CONFIG:
  /configs/baseline_btc_bull_pf20.json ← NEEDS UPDATE
  /configs/btc_v8_adaptive*.json ← NEEDS UPDATE
```

---

## IMMEDIATE ACTION PLAN

### TODAY (2-3 hours) - CRITICAL:
1. Add logging to ThresholdPolicy._build_base_map() (threshold_policy.py:154-177)
2. Verify config['archetypes']['trap_within_trend'] reaches context
3. Check _check_K slug name (logic_v2_adapter.py:578)

### THIS WEEK (1-2 days) - HIGH:
4. Remove hardcoded defaults (logic_v2_adapter.py:385-623)
5. Update all configs to canonical slug format

### NEXT WEEK (3-5 days) - MEDIUM:
6. Delete logic.py duplicate code
7. Update registry mappings

---

## WHAT TO EXPECT AFTER FIXES

**Before:**
- Optimizer trial #1: parameter_value=0.5 → PNL = 1000
- Optimizer trial #2: parameter_value=0.7 → PNL = 1000 (SAME!)
- Conclusion: Parameter has no effect

**After:**
- Optimizer trial #1: parameter_value=0.5 → PNL = 850 (different!)
- Optimizer trial #2: parameter_value=0.7 → PNL = 1200 (different!)
- Conclusion: Parameter affects results, optimization works!

---

## HOW TO USE THIS AUDIT

### For Project Managers:
- Read: AUDIT_EXECUTIVE_SUMMARY.md
- Time: 10 minutes
- Then: Allocate 3-7 days for fixes

### For Developers:
- Read: AUDIT_QUICK_REFERENCE.md
- Reference: COMPREHENSIVE_ARCHETYPE_AUDIT.md (for details)
- Time: 5 minutes to read, then do fixes using reference docs

### For Code Review:
- Check: All items in AUDIT_QUICK_REFERENCE.md "Parameter Flow Checklist"
- Verify: Each archetype follows the correct pattern
- Test: Wire tests from optimizer → config → context → archetype

---

## DOCUMENT STRUCTURE

```
AUDIT_EXECUTIVE_SUMMARY.md
├─ Core problem statement
├─ Top 5 critical issues (with evidence)
├─ Immediate action items (3 phases)
├─ Files that need changes
├─ Current vs fixed parameter flow diagrams
└─ Estimated effort breakdown

AUDIT_QUICK_REFERENCE.md
├─ Critical files table
├─ Zero-variance bug explanation
├─ 6 immediate fixes (priority order)
├─ Correct vs incorrect patterns
├─ Parameter flow checklist
├─ Success criteria
└─ Key stats

COMPREHENSIVE_ARCHETYPE_AUDIT.md
├─ Complete parameter flow diagram (ASCII art)
├─ All 5 inconsistencies with examples
├─ 70+ hardcoded values by archetype
├─ 20+ parameter name mismatches
├─ All file:line references
├─ Config reading paths (3 different paths!)
├─ Fallback/default mechanisms
├─ Dead code inventory
├─ Naming mismatch catalog
├─ Registry mapping confusion
└─ Detailed recommendations
```

---

## SUCCESS CHECKLIST

After all fixes, verify:

- [ ] Optimizer variance restored (different parameters → different results)
- [ ] Single code path (only logic_v2_adapter.py)
- [ ] All configs use canonical slugs (no letter codes)
- [ ] All parameters use LONG names (fusion_threshold not fusion)
- [ ] No hardcoded thresholds (all from config)
- [ ] Parameter flow traced (optimizer → config → context → archetype)
- [ ] Registry fully mapped (no naming conflicts)
- [ ] Tests pass (new parameter flow works)

---

## CONTACT / QUESTIONS

This audit was generated from:
- Complete code review of 5 archetype/parameter files
- Analysis of 11 archetype implementations
- Mapping of config structure
- Optimization code review

All findings backed by specific file:line references in the detailed audit.

