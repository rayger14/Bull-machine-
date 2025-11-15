# Repository Cleanup - Quick Start Guide

**Status:** Ready to execute
**Estimated Time:** 3.5 hours
**Space Recovery:** ~1.25 GB (66% reduction)

---

## Pre-Flight Checklist

- [ ] Read `CLEANUP_PLAN.md` for full details
- [ ] Ensure you're on the correct branch
- [ ] No uncommitted changes (or stash them)
- [ ] Have 30 minutes of uninterrupted time

---

## Quick Execution (Recommended)

### 1. Dry Run First (5 minutes)
```bash
# Preview what will be done without making changes
./cleanup_repository.sh --dry-run --verbose
```

Review the output carefully. If anything looks wrong, STOP and review `CLEANUP_PLAN.md`.

### 2. Execute Cleanup (30 minutes)
```bash
# Run full cleanup
./cleanup_repository.sh --verbose
```

This will:
- Create git tag: `pre-cleanup-2025-11-14` (for rollback)
- Execute all 7 phases automatically
- Commit each phase separately
- Show progress and summary

### 3. Validate (5 minutes)
```bash
# Verify nothing broke
./validate_cleanup.sh
```

This checks:
- Essential files present
- Production code intact
- Imports working
- Configs accessible
- Git status clean

### 4. Test Suite (10 minutes)
```bash
# Run tests to ensure nothing broke
pytest
```

### 5. Manual Verification (10 minutes)
```bash
# Check key functionality
python -c "import bull_machine; import engine"
python bin/backtest_knowledge_v2.py --help

# Review changes
git status
git log --oneline -10
```

### 6. Commit & Push (5 minutes)
```bash
# If everything looks good
git push origin $(git branch --show-current)
```

---

## Phase-by-Phase Execution (Alternative)

If you prefer to run phases individually:

```bash
# Phase 1: Enhance .gitignore
./cleanup_repository.sh --phase 1
git status  # Review
git push    # Optional: push after each phase

# Phase 2: Archive documentation
./cleanup_repository.sh --phase 2
git status

# Phase 3: Clean results & logs
./cleanup_repository.sh --phase 3
git status

# Phase 4: Curate bin scripts
./cleanup_repository.sh --phase 4
git status

# Phase 5: Organize configs
./cleanup_repository.sh --phase 5
git status

# Phase 6: Remove build artifacts
./cleanup_repository.sh --phase 6
git status

# Phase 7: Clean root files
./cleanup_repository.sh --phase 7
git status

# Validate
./validate_cleanup.sh
```

---

## What Gets Deleted

### Results Directory (~750 MB)
- All `optuna_*` experiment folders
- All `bench_v2_frontier`, `macro_fix_validation`, etc.
- All timestamped signal files (`hybrid_signals_*.jsonl`)
- All test artifacts (`health_summary_*.json`, `portfolio_summary_*.json`)
- Debug files (`fusion_debug.jsonl`, `fusion_validation.jsonl`)

**KEPT:** `results/bench_v2/`, `results/bear_patterns/`, `results/archive/`

### Logs Directory (~350 MB)
- All `bear_archetypes_*.log` files (83 MB each)
- All debug and validation logs
- All timestamped backtest logs

**KEPT:** `logs/paper_trading/`, `logs/archive/`

### Root Directory (~400 MB)
- 100+ markdown files → moved to `docs/archive/2024-q4/`
- 21 Python scripts → deleted or moved to bin/archive/
- 50+ JSON result files → deleted
- Shell scripts, logs, CSV files → deleted

**KEPT:** README.md, CHANGELOG.md, setup.py, requirements.txt, pytest.ini, etc.

### Build Artifacts (~2 MB)
- `dist/`, `.mypy_cache/`, `.ruff_cache/`
- `__pycache__/`, `*.pyc`
- `.DS_Store`

---

## What Gets Moved/Reorganized

### Documentation
```
docs/archive/2024-q4/
├── mvp_phases/          # All MVP_PHASE*.md
├── pull_requests/       # All PR*.md
├── sessions/            # All SESSION*.md
├── archetype_work/      # All ARCHETYPE*.md
├── optimization/        # All optimization reports
├── implementations/     # All implementation plans
├── cleanup/             # Cleanup reports
├── features/            # Feature documentation
├── validation/          # Validation reports
├── status/              # Status updates
├── optuna/              # Optuna documentation
├── phases/              # Phase progress
├── paper_trading/       # Paper trading docs
└── audit/               # System audits
```

### Bin Scripts
```
bin/archive/
├── experimental/        # Backfill scripts, fix scripts
└── diagnostics/         # Debug and diagnostic scripts
```

### Configs
```
configs/
├── production/
│   ├── frozen/
│   ├── live/
│   ├── paper_trading/
│   └── mvp_*.json
├── experimental/
│   ├── adaptive/
│   ├── sweep/
│   └── knowledge_v2/
└── archive/
    └── v*/              # All version directories
```

---

## Rollback Instructions

If something goes wrong:

### Option 1: Git Tag Rollback (Recommended)
```bash
# Rollback to pre-cleanup state
git reset --hard pre-cleanup-2025-11-14

# If you already pushed
git push --force origin $(git branch --show-current)
```

### Option 2: Revert Specific Phase
```bash
# See commits
git log --oneline -10

# Revert specific phase
git revert <commit-hash>
```

### Option 3: Cherry-pick Good Changes
```bash
# Start fresh
git reset --hard pre-cleanup-2025-11-14

# Re-apply specific phases
git cherry-pick <phase1-commit>
git cherry-pick <phase2-commit>
```

---

## Troubleshooting

### "File not found" during cleanup
**Cause:** File already deleted or path changed
**Solution:** Safe to ignore if file was experimental

### Import errors after cleanup
**Cause:** Production file accidentally moved/deleted
**Solution:** Check validation output, rollback if needed

### Configs not found
**Cause:** Config path changed during reorganization
**Solution:** Check `configs/production/` and `configs/experimental/`

### Large repo size still
**Cause:** Git history contains large files
**Solution:** Use `git gc --aggressive --prune=now` (advanced)

---

## Post-Cleanup Best Practices

### Prevent Future Bloat

1. **Use .gitignore patterns**
   - Experimental results go in `results/YYYYMMDD_experiment_name/`
   - All gitignored by default

2. **Documentation policy**
   - Session summaries → `docs/archive/YYYY-QQ/sessions/`
   - Keep only current roadmap in root

3. **Bin scripts policy**
   - One-time migrations → `bin/archive/experimental/`
   - Production utilities → `bin/`

4. **Monthly audits**
   ```bash
   # Check for new bloat
   find . -name "*.md" -type f -mtime -30  # Recent markdown
   du -sh results/ logs/                    # Directory sizes
   git status --short | wc -l               # Untracked count
   ```

---

## FAQ

**Q: Will this break my production code?**
A: No. All production code, configs, and dependencies are preserved. Only experimental outputs and old documentation are removed/archived.

**Q: Can I run this on main branch?**
A: Recommended to run on a feature branch first, test, then merge to main.

**Q: What if I need an archived document?**
A: All moved to `docs/archive/2024-q4/` with organized structure. Easy to find and reference.

**Q: Will results/bench_v2 be deleted?**
A: No, it's explicitly preserved as a production benchmark.

**Q: Can I customize what gets deleted?**
A: Yes, edit the script functions for each phase. The script is well-commented.

**Q: How long until I can use the repo again?**
A: Immediately after cleanup. No rebuild or reinstall needed.

---

## Support

If you encounter issues:

1. Check validation output: `./validate_cleanup.sh`
2. Review `CLEANUP_PLAN.md` for details
3. Check git log: `git log --oneline -20`
4. Rollback if needed: `git reset --hard pre-cleanup-2025-11-14`

---

## Success Criteria

After cleanup, you should have:

- ✅ Root directory has ~5 markdown files (down from 116)
- ✅ Root directory has 0 Python scripts (down from 21)
- ✅ Repository size ~650 MB (down from 1.9 GB)
- ✅ Organized docs in `docs/archive/2024-q4/`
- ✅ Clean bin/ with only production scripts
- ✅ Organized configs in `production/`, `experimental/`, `archive/`
- ✅ Enhanced .gitignore preventing future bloat
- ✅ All tests pass
- ✅ Imports work
- ✅ Production functionality intact

---

**Ready to clean up? Start with the dry run!**

```bash
./cleanup_repository.sh --dry-run --verbose
```
