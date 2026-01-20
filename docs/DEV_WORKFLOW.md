# Development Workflow & Branching Strategy

**Version:** 2.0
**Last Updated:** 2025-11-19
**Status:** Active - Ghost to Live v2 Upgrade

---

## Overview

Bull Machine follows a structured branching strategy for feature development, testing, and production deployment. This document defines the workflow for the Ghost → Live v2 upgrade and future development.

---

## Branch Hierarchy

```
main (production baseline)
 │
 └─── feature/ghost-modules-to-live-v2 (integration branch)
       │
       ├─── feature/tier1-core-modules (Tier 1 features)
       ├─── feature/tier2-enhanced-modules (Tier 2 features)
       └─── feature/tier3-experimental-modules (Tier 3 features)
```

---

## Branch Types

### 1. main
- **Purpose:** Production-ready code baseline
- **Protection:** Protected, requires PR approval
- **Merge Policy:** Only from validated integration branches
- **Testing:** Full integration test suite must pass

### 2. Integration Branches (feature/ghost-modules-to-live-v2)
- **Purpose:** Consolidate related features before merging to main
- **Lifetime:** Duration of multi-phase upgrade
- **Validation:** Gold standard backtests + regression tests
- **Merge to main:** Only after Phase 4 validation passes

### 3. Feature Branches (feature/tier1-core-modules)
- **Purpose:** Isolated development of specific tiers
- **Naming:** `feature/<tier>-<description>`
- **Base:** Always branch from integration branch
- **Merge back:** To integration branch after tier validation

### 4. Experimental Branches (experiment/<name>)
- **Purpose:** Research, prototyping, parameter tuning
- **Lifetime:** Temporary, deleted after merge or rejection
- **Merge Policy:** Only proven experiments merge to feature branches

---

## Ghost → Live v2 Workflow

### Phase 0: Planning & Setup ✓
**Branch:** `feature/ghost-modules-to-live-v2`
**Status:** In Progress

**Deliverables:**
- [x] Create integration branch
- [x] Brain blueprint snapshot
- [x] Architecture design documents
- [x] Risk mitigation plan
- [x] CI/CD guardrails spec
- [ ] Review and approval

**Duration:** 1-2 days

---

### Phase 1: Tier 1 Core Modules (LIVE → Production)
**Branch:** `feature/tier1-core-modules` (from integration branch)
**Modules:** 48 LIVE modules requiring production-grade validation

**Workflow:**
1. Checkout from integration branch
2. Review all Tier 1 LIVE modules for production readiness
3. Add unit tests for any missing coverage
4. Validate feature store columns (no NaNs, correct ranges)
5. Run integration tests
6. Merge to integration branch

**Validation Criteria:**
- All unit tests pass
- Integration tests pass
- Feature store validation clean
- No performance degradation vs baseline

**Duration:** 3-5 days

---

### Phase 2: Tier 2 Enhanced Modules (PARTIAL → LIVE)
**Branch:** `feature/tier2-enhanced-modules` (from integration branch)
**Modules:** 25 PARTIAL modules requiring completion

**Workflow:**
1. Checkout from integration branch
2. Complete each PARTIAL module (add missing logic, tests, docs)
3. Add feature store columns if needed
4. Add unit tests for new logic
5. Integration test each module
6. Merge to integration branch

**Validation Criteria:**
- Module transitions from PARTIAL → LIVE
- Feature coverage increases
- Tests pass
- Documentation complete

**Duration:** 5-7 days

---

### Phase 3: Tier 3 Experimental Modules (IDEA → PARTIAL/LIVE)
**Branch:** `feature/tier3-experimental-modules` (from integration branch)
**Modules:** 16 IDEA ONLY modules requiring implementation

**Workflow:**
1. Checkout from integration branch
2. Implement each IDEA module from specification
3. Add feature store columns
4. Add comprehensive tests
5. Validate with backtest
6. Merge to integration branch

**Validation Criteria:**
- Module transitions from IDEA → PARTIAL or LIVE
- New features validated
- Tests pass
- Performance impact documented

**Duration:** 7-10 days

---

### Phase 4: Integration & Validation
**Branch:** `feature/ghost-modules-to-live-v2`
**Purpose:** Final validation before merging to main

**Workflow:**
1. Merge all tier branches into integration branch
2. Run full regression test suite
3. Run gold standard backtests (2022-2024)
4. Performance comparison vs baseline
5. Create PR to main

**Validation Criteria:**
- All tests pass
- Gold standard metrics within ±5%
- No new bugs introduced
- Feature store integrity validated
- Documentation complete

**Pass/Fail Gates:**
- Profit Factor: ±5% of baseline
- Trade Count: ±10% of baseline
- Max Drawdown: ±10% of baseline
- Win Rate: ±5% of baseline

**Duration:** 2-3 days

---

## Commit Message Standards

### Format
```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types
- `feat`: New feature
- `fix`: Bug fix
- `refactor`: Code refactoring
- `test`: Adding tests
- `docs`: Documentation
- `chore`: Maintenance tasks

### Examples
```
feat(tier1): add production-grade SMC order block detection

- Validate all order block calculations
- Add unit tests for edge cases
- Update feature store schema

Closes #123
```

```
fix(feature-store): resolve NaN values in tf4h_fusion_score

- Add fallback logic for missing macro data
- Validate column ranges
- Add integration test

Fixes #456
```

---

## Pull Request Process

### 1. Pre-PR Checklist
- [ ] All tests pass locally
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] Commit messages follow standards
- [ ] Branch is up to date with base branch

### 2. PR Template
```markdown
## Description
Brief description of changes

## Changes
- List of key changes
- Files modified
- New features added

## Testing
- Unit tests: PASS/FAIL
- Integration tests: PASS/FAIL
- Backtest results: Link to results

## Validation
- Gold standard: PASS/FAIL (metrics)
- Performance impact: +/- X%
- Feature coverage: +X features

## Checklist
- [ ] Tests pass
- [ ] Documentation updated
- [ ] No breaking changes
- [ ] Backward compatible
```

### 3. Review Process
- Minimum 1 approval required
- CI/CD checks must pass
- Gold standard validation for integration branches
- No merge if tests fail

---

## Testing Strategy

### Local Testing (Before Commit)
```bash
# Run unit tests
pytest tests/unit/

# Run integration tests
pytest tests/integration/

# Run smoke tests
pytest tests/smoke/
```

### CI/CD Testing (Automated)
```bash
# Blueprint consistency test
python bin/validate_blueprint_consistency.py

# Dead feature detector
python bin/detect_dead_features.py

# Config consistency test
python bin/validate_config_consistency.py

# Backtest expectations test
pytest tests/integration/test_gold_standard.py
```

---

## Rollback Strategy

### If Phase Validation Fails

**Step 1: Identify Issue**
- Review test failures
- Check backtest metrics
- Inspect logs for errors

**Step 2: Rollback Options**

**Option A: Revert Last Commit**
```bash
git revert HEAD
git push origin feature/ghost-modules-to-live-v2
```

**Option B: Reset to Last Good State**
```bash
git reset --hard <last-good-commit>
git push --force origin feature/ghost-modules-to-live-v2
```

**Option C: Restore from Backup**
```bash
# Restore feature store
cp data/features_mtf/*.parquet.bak data/features_mtf/

# Reset branch
git fetch origin
git reset --hard origin/feature/ghost-modules-to-live-v2
```

### If Production Issue Detected

**Step 1: Hotfix Branch**
```bash
git checkout main
git checkout -b hotfix/critical-issue
# Fix issue
git commit -m "hotfix: critical issue description"
git push origin hotfix/critical-issue
```

**Step 2: Emergency PR**
- Fast-track approval process
- Deploy immediately after tests pass

**Step 3: Backport to Integration**
```bash
git checkout feature/ghost-modules-to-live-v2
git cherry-pick <hotfix-commit>
```

---

## Branch Cleanup

### After Successful Merge to Main
```bash
# Archive feature branch (don't delete)
git tag archive/ghost-modules-to-live-v2 feature/ghost-modules-to-live-v2
git push origin archive/ghost-modules-to-live-v2

# Delete local branch
git branch -d feature/ghost-modules-to-live-v2

# Delete remote branch (optional, keep for history)
# git push origin --delete feature/ghost-modules-to-live-v2
```

### After Failed Branch
```bash
# Tag for historical reference
git tag rejected/<branch-name> <branch-name>
git push origin rejected/<branch-name>

# Delete branch
git branch -D <branch-name>
```

---

## Development Environment

### Required Tools
- Python 3.9+
- Git 2.30+
- pytest 7.0+
- pandas 1.5+
- numpy 1.23+

### Setup
```bash
# Clone repository
git clone <repo-url>
cd Bull-machine-

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run tests to validate setup
pytest tests/smoke/
```

---

## Best Practices

### 1. Incremental Development
- Small, focused commits
- Commit after each logical change
- Test before committing

### 2. Feature Flags
- Use feature flags for risky changes
- Enable/disable features via config
- Gradual rollout strategy

### 3. Backward Compatibility
- Never break existing APIs
- Deprecate before removing
- Maintain frozen baselines

### 4. Documentation
- Update docs with code changes
- Add inline comments for complex logic
- Maintain architecture diagrams

### 5. Testing
- Write tests before code (TDD)
- Aim for 80%+ code coverage
- Test edge cases and error paths

---

## Emergency Procedures

### Production Failure
1. Immediately notify team
2. Create hotfix branch from main
3. Fix and test rapidly
4. Fast-track PR approval
5. Deploy and monitor

### Data Corruption
1. Stop all processes
2. Restore from backup
3. Validate data integrity
4. Document incident
5. Update backup procedures

### CI/CD Pipeline Failure
1. Check CI/CD logs
2. Validate local tests pass
3. Fix pipeline configuration
4. Re-run pipeline
5. Document issue and fix

---

## References

- Architecture: `docs/ARCHITECTURE.md`
- Testing Guide: `docs/guides/testing.md`
- Feature Store Schema: `docs/FEATURE_STORE_SCHEMA_v2.md`
- CI/CD Guardrails: `docs/CI_GUARDRAILS_SPEC.md`

---

## Version History

- **v2.0** (2025-11-19): Ghost → Live v2 workflow
- **v1.0** (2025-11-14): Initial workflow documentation
