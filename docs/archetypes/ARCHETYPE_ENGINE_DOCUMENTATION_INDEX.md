# Archetype Engine Documentation Index

**Master index for all archetype engine fix documentation**

---

## Quick Access

**Need to fix and deploy NOW?** → `ARCHETYPE_ENGINE_QUICK_START.md`

**Want full technical details?** → `ARCHETYPE_ENGINE_FIX_COMPLETE.md`

**Validation failing?** → `TROUBLESHOOTING_GUIDE.md`

**Need to understand domain engines?** → `DOMAIN_ENGINE_GUIDE.md`

---

## Documentation Suite

### 1. ARCHETYPE_ENGINE_FIX_COMPLETE.md
**Complete technical report on what was broken and how we fixed it**

- **Purpose:** Comprehensive "what we fixed" documentation
- **Audience:** Technical stakeholders, developers, quant team
- **Length:** ~3000 lines
- **Read time:** 30-45 minutes

**Contents:**
- Executive summary of problems and solutions
- Detailed breakdown of 5 critical bugs
- Performance impact projections (+525-733% improvement)
- Complete file list (27 files created/modified)
- Step-by-step next actions
- 9-step validation checklist
- Maintenance schedule and support info

**Use when:** You need full context on the archetype engine fix

---

### 2. ARCHETYPE_ENGINE_QUICK_START.md
**5-minute guide to fix and validate**

- **Purpose:** Fast execution guide for stakeholders
- **Audience:** Anyone who needs to apply fixes quickly
- **Length:** ~200 lines
- **Read time:** 5 minutes

**Contents:**
- What's wrong (4 bullet summary)
- One-command fix
- 30-minute validation
- 2-hour test
- Deploy decision tree

**Use when:** You need to fix the system RIGHT NOW

---

### 3. FEATURE_MAPPING_REFERENCE.md
**Complete canonical name → feature store mapping**

- **Purpose:** Reference guide for all feature name translations
- **Audience:** Developers working with archetype logic
- **Length:** ~800 lines
- **Read time:** 15 minutes (reference document)

**Contents:**
- Critical mappings table
- Funding/OI features
- Volume/exhaustion features
- Macro/regime features
- SMC features
- Multi-timeframe features
- Wyckoff features
- Temporal features
- Technical indicators
- FeatureMapper API usage
- Validation commands

**Use when:** You need to look up how a feature name maps to the store

---

### 4. DOMAIN_ENGINE_GUIDE.md
**Understanding the 6 domain engines**

- **Purpose:** Explains what each engine does and why it matters
- **Audience:** Quant team, ML engineers, developers
- **Length:** ~1200 lines
- **Read time:** 25 minutes

**Contents:**
- Overview of all 6 engines
- Detailed breakdown:
  - Wyckoff Engine (accumulation/distribution)
  - SMC Engine (smart money concepts)
  - Temporal Engine (Fib time, Gann cycles)
  - HOB Engine (proprietary patterns)
  - Fusion Engine (multi-domain synthesis)
  - Macro Engine (regime classification)
- Features provided by each engine
- Archetypes using each engine
- Configuration examples
- Impact when engines disabled
- Engine interaction map
- Performance impact analysis

**Use when:** You need to understand what domain engines do

---

### 5. VALIDATION_QUICK_REFERENCE.md
**1-page validation checklist**

- **Purpose:** Print-friendly validation reference
- **Audience:** Anyone validating archetype engine
- **Length:** ~350 lines
- **Read time:** 5 minutes (reference)

**Contents:**
- 9-step validation protocol table
- Quick validation (5 min)
- Full validation (30 min)
- Per-archetype validation
- Critical thresholds
- Validation decision matrix
- Common failure patterns
- Pre-deployment checklist
- Validation frequency schedule
- Quick diagnostic commands

**Use when:** You need to validate that fixes worked

---

### 6. TROUBLESHOOTING_GUIDE.md
**Common issues and quick fixes**

- **Purpose:** Diagnostic and fix guide for validation failures
- **Audience:** Anyone encountering validation errors
- **Length:** ~900 lines
- **Read time:** 20 minutes (or search for specific issue)

**Contents:**
- Quick diagnostic command
- 10 common issues with diagnosis and fixes:
  1. High Tier-1 fallback rate
  2. Low test performance
  3. Missing OI/Funding data
  4. Domain engines not enabled
  5. Feature store missing critical features
  6. Wrong calibrations applied
  7. Chaos windows not firing
  8. Regime filter too restrictive
  9. Identical trades across archetypes
  10. Validation passes but production fails
- Quick reference: diagnostic commands
- Prevention best practices
- Weekly maintenance checklist

**Use when:** Validation is failing and you need to debug

---

### 7. CALIBRATION_GUIDE.md
**How to sync with Optuna optimization**

- **Purpose:** Extract and apply optimized parameters
- **Audience:** Quant team, optimization engineers
- **Length:** ~700 lines
- **Read time:** 15 minutes

**Contents:**
- Quick start (one command)
- Optuna database locations
- 3 methods for extracting parameters
- 3 methods for applying calibrations
- Parameter mappings (S1, S4, S5)
- Verification procedures
- Re-optimization process
- Calibration best practices
- Troubleshooting
- Advanced: custom optimization
- Maintenance schedule

**Use when:** You need to apply or update optimized parameters

---

### 8. bin/apply_all_fixes.sh
**Master execution script**

- **Purpose:** One-command application of all fixes
- **Audience:** Operators, developers
- **Length:** ~400 lines
- **Execution time:** 4 hours

**What it does:**
1. Creates FeatureMapper
2. Enables all 6 domain engines
3. Creates calibration templates
4. Backfills OI data (optional)
5. Runs quick validation (optional)

**Usage:**
```bash
# Apply all fixes
./bin/apply_all_fixes.sh

# Skip OI backfill (faster)
./bin/apply_all_fixes.sh --skip-oi-backfill

# Skip validation
./bin/apply_all_fixes.sh --skip-validation
```

**Use when:** You're ready to apply all fixes in one go

---

## Document Relationships

```
ARCHETYPE_ENGINE_QUICK_START.md
    ↓ (need details?)
ARCHETYPE_ENGINE_FIX_COMPLETE.md
    ↓ (validation failing?)
VALIDATION_QUICK_REFERENCE.md
    ↓ (specific error?)
TROUBLESHOOTING_GUIDE.md
    ↓ (need feature mapping?)
FEATURE_MAPPING_REFERENCE.md
    ↓ (need engine info?)
DOMAIN_ENGINE_GUIDE.md
    ↓ (need calibrations?)
CALIBRATION_GUIDE.md
```

---

## Recommended Reading Order

### For Stakeholders
1. `ARCHETYPE_ENGINE_QUICK_START.md` (5 min)
2. `ARCHETYPE_ENGINE_FIX_COMPLETE.md` Executive Summary (10 min)
3. `VALIDATION_QUICK_REFERENCE.md` (5 min)

**Total:** 20 minutes to understand the fix and validation process

### For Developers
1. `ARCHETYPE_ENGINE_FIX_COMPLETE.md` (45 min)
2. `FEATURE_MAPPING_REFERENCE.md` (15 min)
3. `DOMAIN_ENGINE_GUIDE.md` (25 min)
4. `TROUBLESHOOTING_GUIDE.md` (skim, 10 min)

**Total:** 95 minutes to understand the full technical context

### For Operators
1. `ARCHETYPE_ENGINE_QUICK_START.md` (5 min)
2. `VALIDATION_QUICK_REFERENCE.md` (5 min)
3. `TROUBLESHOOTING_GUIDE.md` (skim, 10 min)
4. Review `bin/apply_all_fixes.sh` (5 min)

**Total:** 25 minutes to execute and validate

### For Quant Team
1. `ARCHETYPE_ENGINE_FIX_COMPLETE.md` (45 min)
2. `CALIBRATION_GUIDE.md` (15 min)
3. `DOMAIN_ENGINE_GUIDE.md` (25 min)

**Total:** 85 minutes to understand performance implications

---

## Execution Workflow

```
START
  ↓
Read: ARCHETYPE_ENGINE_QUICK_START.md
  ↓
Execute: ./bin/apply_all_fixes.sh
  ↓
Validate: ./bin/validate_archetype_engine.sh --full
  ↓
Pass? → YES → Deploy to paper trading
      → NO  → Check: TROUBLESHOOTING_GUIDE.md
                ↓
              Fix issues
                ↓
              Re-validate
                ↓
              Pass? → YES → Deploy
                    → NO  → Review: ARCHETYPE_ENGINE_FIX_COMPLETE.md
                              ↓
                            Deep dive technical issues
```

---

## File Locations

All documentation in project root:
```
/Users/raymondghandchi/Bull-machine-/Bull-machine-/
├── ARCHETYPE_ENGINE_FIX_COMPLETE.md
├── ARCHETYPE_ENGINE_QUICK_START.md
├── FEATURE_MAPPING_REFERENCE.md
├── DOMAIN_ENGINE_GUIDE.md
├── VALIDATION_QUICK_REFERENCE.md
├── TROUBLESHOOTING_GUIDE.md
├── CALIBRATION_GUIDE.md
└── bin/apply_all_fixes.sh
```

---

## Version Control

All documentation is version controlled:

```bash
# View documentation history
git log --oneline ARCHETYPE_ENGINE_*.md

# Check current version
head -n 1 ARCHETYPE_ENGINE_FIX_COMPLETE.md
# Shows: # Archetype Engine Fix - Complete Report

# Check last updated
tail -n 5 ARCHETYPE_ENGINE_FIX_COMPLETE.md
# Shows: Last Updated: 2025-12-08
```

---

## Maintenance

**Documentation owner:** Archetype Engine Team

**Review frequency:** After each major archetype change

**Update triggers:**
- New archetype added
- New domain engine added
- Feature store schema changes
- Optimization framework changes
- Production deployment (update with actual results)

**Next review scheduled:** After first production deployment

---

## Quick Command Reference

```bash
# Execute fixes
./bin/apply_all_fixes.sh

# Validate
./bin/validate_archetype_engine.sh --full

# Troubleshoot
./bin/diagnose_archetype_issues.sh --all

# Check calibrations
python bin/verify_calibrations.py --all

# Test performance
python bin/run_archetype_suite.py --archetypes s1,s4,s5 --periods test

# Deploy
python bin/deploy_to_paper_trading.py --systems s1,s4,s5
```

---

## Support

**For questions about:**
- Fix implementation → Read `ARCHETYPE_ENGINE_FIX_COMPLETE.md`
- Validation failures → Check `TROUBLESHOOTING_GUIDE.md`
- Feature mappings → Reference `FEATURE_MAPPING_REFERENCE.md`
- Domain engines → Read `DOMAIN_ENGINE_GUIDE.md`
- Calibrations → Check `CALIBRATION_GUIDE.md`

**For urgent issues:**
1. Run: `./bin/diagnose_archetype_issues.sh`
2. Check: `logs/archetype_validation/`
3. Review: `TROUBLESHOOTING_GUIDE.md`

---

## Document Stats

| Document | Lines | Pages | Read Time | Complexity |
|----------|-------|-------|-----------|------------|
| ARCHETYPE_ENGINE_FIX_COMPLETE.md | ~900 | 15 | 45 min | High |
| ARCHETYPE_ENGINE_QUICK_START.md | ~200 | 3 | 5 min | Low |
| FEATURE_MAPPING_REFERENCE.md | ~800 | 13 | 15 min | Medium |
| DOMAIN_ENGINE_GUIDE.md | ~1200 | 20 | 25 min | Medium |
| VALIDATION_QUICK_REFERENCE.md | ~350 | 6 | 5 min | Low |
| TROUBLESHOOTING_GUIDE.md | ~900 | 15 | 20 min | Medium |
| CALIBRATION_GUIDE.md | ~700 | 12 | 15 min | Medium |

**Total:** ~5,050 lines, ~84 pages, ~130 minutes reading time

---

## Printable Versions

For stakeholder meetings:

**1-page summary:**
```bash
# Print pages 1-2 of Quick Start
head -80 ARCHETYPE_ENGINE_QUICK_START.md | lpr
```

**Validation checklist:**
```bash
# Print validation reference (6 pages)
lpr VALIDATION_QUICK_REFERENCE.md
```

**Executive summary:**
```bash
# Print first 200 lines of complete report
head -200 ARCHETYPE_ENGINE_FIX_COMPLETE.md | lpr
```

---

**Document Version:** 1.0
**Last Updated:** 2025-12-08
**Status:** Complete
**Next Review:** After production deployment
