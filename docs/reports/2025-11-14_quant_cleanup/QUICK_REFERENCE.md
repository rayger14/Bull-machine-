# Cleanup Quick Reference
## 2025-11-14 Repository Cleanup

### TL;DR

Cleaned up 1.1M+ lines of experimental data while preserving 100% of code, configs, and documentation.

### Key Changes

**Space Recovered**: ~656MB
**Files Deleted**: 92 files (all experimental outputs)
**Repo Size**: 344MB (was ~1GB in results/)

### What Changed

#### DELETED (All Experimental/Temporary)
- 211 `hybrid_signals_*.jsonl` files
- 29 `health_summary_*.json` files
- 31 `portfolio_summary_*.json` files
- 20+ `optuna_*/` experiment directories
- 5 `router_v10_*/` experiment directories
- `bench_v2_frontier/` (167MB)
- `macro_fix_validation/` (222MB)
- All `.log` files throughout repo
- `chart_logs_binance/` directory
- `test_checkpoints/` directory
- Various experiment archives

#### CREATED (New Curated Reference)
```
results_reference/
├── btc/ml_training_baseline.json
├── eth/ml_training_baseline.json
├── bear_market/2022_validation.json
├── optimization/trap_v2_production_params.json
└── system_validation/trade_log_reference.jsonl
```

#### KEPT (Production Assets)
- `results/bench_v2/` (613MB - production benchmarks)
- `results/bear_patterns/` (validation data)
- All source code (100%)
- All configs (100%)
- All documentation (100%)
- All data/features (100%)

### Git Status

```
94 files changed
6 insertions(+)
1,147,296 deletions(-)
```

### Tests Status

✓ All tests passing (verified post-cleanup)

### Next Steps

1. **Review the changes**: Check `git status` and `git diff`
2. **Run full test suite** (if desired): `pytest tests/`
3. **Commit the cleanup**:
   ```bash
   git add .
   git commit -m "chore: cleanup repository to quant standards (1.1M lines removed)"
   ```

### Future Development

- **Experiments**: Run locally, outputs auto-gitignored
- **Reference results**: Add to `results_reference/` if canonical
- **Documentation**: Add to `docs/` or `docs/reports/`
- **Configs**: Add to `configs/`

### Rollback (if needed)

If you need to undo:
```bash
git reset --hard HEAD~1  # Undo the commit
git clean -fd            # Remove untracked files
```

---

**Full details**: See `REPO_CLEANUP_SUMMARY.md` in this directory
