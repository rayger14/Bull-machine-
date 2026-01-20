# Commit Strategy Execution Guide

**Status:** Ready to execute
**Date:** 2026-01-15
**Task:** Commit Phase 1 & 2 confidence calibration work

---

## Overview

This guide walks through committing the confidence calibration and hybrid integration work using a **three-commit strategy** following Conventional Commits standard.

**Total time:** 15-30 minutes

---

## Pre-Commit Checklist

Before executing commits, verify:

- [ ] All agents completed successfully (backtest, explore, research, architect)
- [ ] Integration tests pass (`bin/test_hybrid_confidence_integration.py`)
- [ ] Backtest validation complete (`bin/backtest_hybrid_confidence_validation.py`)
- [ ] Working directory clean (no uncommitted changes you want to keep)

---

## Execution Steps

### Step 1: Organize Files (5-10 minutes)

**Run cleanup script:**
```bash
# Make script executable
chmod +x bin/cleanup_and_organize_docs.sh

# Execute cleanup
./bin/cleanup_and_organize_docs.sh

# Expected output:
# - docs/regime/confidence_calibration/ created
# - docs/regime/ contains hybrid guides
# - docs/archive/2026-01-15_*/ contains research files
# - Root directory cleaned
```

**Verify cleanup worked:**
```bash
# Check root directory (should be ~10 MD files)
ls -1 *.md 2>/dev/null | wc -l

# Check docs structure
ls -R docs/regime/
ls -R docs/archive/2026-01-15*/
```

---

### Step 2: Review Changes (2-5 minutes)

**Check git status:**
```bash
git status

# Expected output:
# - Modified: engine/context/regime_service.py
# - Modified: engine/context/confidence_calibrator.py (if modified)
# - New files: bin/build_confidence_calibrator.py
# - New files: bin/extract_confidence_calibration_data.py
# - New files: bin/test_hybrid_confidence_integration.py
# - New files: bin/backtest_hybrid_confidence_validation.py
# - New files: models/confidence_calibrator_v1.pkl
# - New files: docs/regime/HYBRID_CONFIDENCE_GUIDE.md
# - Many untracked files in docs/archive/
```

**Review specific changes:**
```bash
# Check regime_service.py changes (Layer 1.5b addition)
git diff engine/context/regime_service.py | head -100

# Verify new scripts exist
ls -lh bin/*confidence*.py
ls -lh models/confidence_calibrator_v1.pkl
```

---

### Step 3: Commit 1 - Core Feature (3-5 minutes)

**Stage core feature files:**
```bash
# Python code
git add engine/context/regime_service.py
git add engine/context/confidence_calibrator.py

# Scripts
git add bin/build_confidence_calibrator.py
git add bin/extract_confidence_calibration_data.py
git add bin/test_hybrid_confidence_integration.py
git add bin/backtest_hybrid_confidence_validation.py

# Model artifact
git add models/confidence_calibrator_v1.pkl

# Data (if you want to commit training data)
# git add data/confidence_calibration_data.parquet  # Optional - may be large
```

**Review staged changes:**
```bash
git diff --cached --stat

# Should show:
# - engine/context/regime_service.py modified
# - bin/*.py added
# - models/*.pkl added
```

**Commit with template:**
```bash
git commit -F commit_message_1_feature.txt

# Alternative: Review in editor first
git commit --verbose -F commit_message_1_feature.txt
```

**Verify commit:**
```bash
git log -1 --stat

# Should show:
# - Conventional commit format (feat(regime):)
# - All core files included
# - Co-Authored-By footer present
```

---

### Step 4: Commit 2 - Documentation (2-3 minutes)

**Stage documentation files:**
```bash
# Regime documentation
git add docs/regime/HYBRID_CONFIDENCE_GUIDE.md
git add docs/regime/HYBRID_CONFIDENCE_BACKTEST_REPORT.md
git add docs/regime/HYBRID_CONFIDENCE_VALIDATION_SUMMARY.md
git add docs/regime/HYBRID_CONFIDENCE_QUICK_REF.md
git add docs/regime/confidence_calibration/

# Archive folders
git add docs/archive/2026-01-15_confidence_calibration/
```

**Review staged docs:**
```bash
git diff --cached --stat

# Should show:
# - docs/regime/ files added
# - docs/archive/ files added
```

**Commit with template:**
```bash
git commit -F commit_message_2_docs.txt
```

**Verify:**
```bash
git log -1 --stat
```

---

### Step 5: Commit 3 - Cleanup (3-5 minutes)

**Update .gitignore (if needed):**
```bash
# Check if patterns are already present
grep "BENCHMARK_" .gitignore
grep "BEFORE_AFTER_" .gitignore

# If missing, add them:
cat >> .gitignore << 'EOF'

# Research and analysis reports (keep in docs/archive/)
/BENCHMARK_*.md
/BEFORE_AFTER_*.md
/*_DELIVERABLES*.md
/*_INDEX.md
/*_QUICK_START.md
/*_QUICK_REF*.md
/*_SUMMARY.md
/*_COMPLETE*.md

# Allow specific permanent docs
!README.md
!ARCHITECTURE.md
!CHANGELOG.md
EOF

git add .gitignore
```

**Remove tracked files that should be ignored:**
```bash
# Archive remaining root-level reports
git add docs/archive/2026-01-15_cleanup/

# Remove from git tracking (but keep locally in archive)
git rm --cached PHASE*.md 2>/dev/null || true
git rm --cached *_REPORT.md 2>/dev/null || true
git rm --cached *_QUICK_REF*.md 2>/dev/null || true
git rm --cached *_DELIVERABLES*.md 2>/dev/null || true
git rm --cached *_INDEX.md 2>/dev/null || true

# Note: These files are now in docs/archive/ so they won't be lost
```

**Commit cleanup:**
```bash
git commit -F commit_message_3_cleanup.txt
```

**Verify:**
```bash
git log -3 --oneline

# Should show:
# - chore: clean root directory
# - docs(regime): add confidence calibration documentation
# - feat(regime): add confidence calibration with hybrid integration
```

---

### Step 6: Final Verification (2-3 minutes)

**Check clean state:**
```bash
git status

# Should show:
# - On branch feature/ghost-modules-to-live-v2
# - Your branch is ahead of 'origin/...' by 3 commits
# - Untracked files: docs/archive/ files (.gitignored)
```

**Verify root directory is clean:**
```bash
ls -1 *.md 2>/dev/null

# Should show only essentials:
# - README.md
# - ARCHITECTURE.md
# - CHANGELOG.md
# - CONTRIBUTING.md (if exists)
# - LICENSE.md (if exists)
```

**Check commit history:**
```bash
git log --oneline -5

# Recent commits should show:
# - Conventional commit format
# - Clear scopes (regime)
# - Co-Authored-By footers
```

**Review file changes:**
```bash
# Show all files changed in last 3 commits
git diff HEAD~3 --stat

# Show detailed changes to regime_service.py
git show HEAD~2:engine/context/regime_service.py | grep -A 10 "Layer 1.5b"
```

---

### Step 7: Push to Remote (1 minute)

**Push commits:**
```bash
# Push to feature branch
git push origin feature/ghost-modules-to-live-v2

# If branch doesn't exist yet:
# git push -u origin feature/ghost-modules-to-live-v2
```

**Verify on GitHub:**
- Check commit messages render correctly
- Verify "BREAKING CHANGE" is highlighted
- Confirm all files present

---

## Post-Commit Actions

### Immediate

- [ ] Update CHANGELOG.md with Phase 1 & 2 work (optional)
- [ ] Create PR if ready for review (optional)
- [ ] Tag release if deploying (optional): `git tag v2.1.0-confidence`

### Next Phase (Week 1)

According to strategic guidance, next steps are:

1. **Fix position sizing** (30 min)
   - Reduce from 20% to 12% per position
   - Re-run backtest to verify <35% max DD

2. **Fix S5 short bug** (2-4 hours)
   - Debug why long_squeeze executes longs
   - Target: 30-40% of trades should be short

3. **Fix regime metadata saving** (30 min)
   - Update Trade dataclass to save regime labels
   - Enable regime-specific performance analysis

4. **Walk-forward validation** (4-8 hours) ← **CRITICAL GATE**
   - Run bin/walk_forward_validation.py
   - GO/NO-GO: If degradation >20%, fix overfitting first

---

## Rollback Procedures

### If Commit 1 (Feature) Has Issues

```bash
# Undo last commit but keep changes
git reset --soft HEAD~1

# Review changes
git diff --cached

# Make fixes
# ...

# Re-commit
git commit -F commit_message_1_feature.txt
```

### If All 3 Commits Need Rollback

```bash
# Undo last 3 commits but keep changes
git reset --soft HEAD~3

# Verify changes still staged
git status

# Rework and re-commit
# ...
```

### If Already Pushed to Remote

```bash
# WARNING: Only if no one else has pulled!
git reset --hard HEAD~3
git push --force origin feature/ghost-modules-to-live-v2

# Safer: Revert commits (preserves history)
git revert HEAD~2..HEAD
git push origin feature/ghost-modules-to-live-v2
```

---

## Troubleshooting

### "Too many files to commit"

```bash
# Check number of files
git diff --cached --stat | wc -l

# If >100 files, something went wrong
# Reset and review:
git reset

# Stage files incrementally
git add engine/
git add bin/
# etc.
```

### ".gitignore patterns not working"

```bash
# Check if files already tracked
git ls-files | grep BENCHMARK_

# If tracked, must remove first
git rm --cached BENCHMARK_*.md

# Then .gitignore will work
```

### "Commit message too long"

```bash
# Git might warn if subject line >72 chars
# Current subject: "feat(regime): add confidence calibration with hybrid integration"
# Length: 67 chars ✓ OK

# If body too long, git will still accept it
# No action needed
```

---

## Success Criteria

After completing all steps, verify:

- ✅ Three commits created with Conventional Commits format
- ✅ Root directory has <15 MD files
- ✅ Documentation organized in docs/regime/
- ✅ Research archived in docs/archive/2026-01-15_*/
- ✅ All integration tests still pass
- ✅ Pushed to remote successfully

---

## Summary

**What we committed:**
- Phase 1 & 2 confidence calibration work
- Hybrid approach (raw agreement + calibrated stability)
- Integration tests and validation
- Comprehensive documentation

**What we cleaned up:**
- 500+ root-level MD files → organized in docs/
- .gitignore patterns updated
- Repository navigation improved

**Next steps:**
- Week 1: Foundation fixes (20 hours)
- Week 2-3: Complete regime detection (15 hours)
- Week 4-8: Paper trading (required before archetype tuning)

**Time to production-ready:** 4-8 weeks following strategic roadmap

---

**Ready to execute?** Start with Step 1: Run cleanup script.
