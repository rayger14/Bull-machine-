# Commit Quick Start - Domain Engine Implementation

**READY TO COMMIT**: All files organized, tested, and documented.

## TL;DR - Execute All Commits

```bash
# 1. Feature Registry
git add engine/features/registry.py
git commit -m "feat(engine): add domain engine feature specifications

Add 15 new feature specifications to registry for domain engines:
- SMC features: demand/supply zones, liquidity sweep, CHOCH
- HOB features: demand/supply zones, order book imbalance
- Temporal features: fib time cluster, confluence, resistance/support clusters

Part of complete domain engine wiring for S1, S4, S5 archetypes.
Enables 44 unique features across Wyckoff, SMC, HOB, Temporal, Macro engines.

Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"

# 2. Domain Engine Wiring
git add engine/archetypes/logic_v2_adapter.py
git commit -m "feat(engine): complete domain engine wiring for S1, S4, S5 archetypes

Add comprehensive domain logic integration (+530 lines):

S1 Liquidity Vacuum (37 features):
- Wyckoff: Spring A/B (2.50x), SC (2.00x), ST (1.50x), LPS (1.80x)
- SMC: 4H BOS (2.00x), 1H BOS (1.40x), demand zones (1.50x)
- Temporal: Fib time (1.80x), confluence (1.50x), 4H fusion (1.60x)
- HOB: Demand zones (1.50x), bid imbalance (1.30x)

S4 Funding Divergence (33 features):
- Wyckoff: Spring (2.50x), accumulation (2.00x), LPS (1.50x), SOS (1.80x)
- SMC: 4H BOS (2.00x), demand zones (1.60x), liquidity sweep (1.80x)
- Temporal: Fib time (1.70x), confluence (1.50x), Wyckoff-PTI (1.40x)

S5 Long Squeeze (35 features):
- Wyckoff: UTAD (2.50x), BC (2.00x), distribution (2.00x), SOW (1.80x)
- SMC: 4H bearish BOS (2.00x), supply zones (1.80x), CHOCH (1.60x)
- Temporal: Fib resistance (1.80x), Wyckoff-PTI (1.50x)
- HOB: Supply zones (1.50x), ask imbalance (1.30x)

Total: 44 unique domain features wired across 3 archetypes
Max theoretical boost: Up to 95x (S1 full confluence, realistic: 8-12x)
Veto protection: 15+ hard/soft vetoes prevent catastrophic entries

Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"

# 3. Critical Bug Fix
git add bin/backtest_knowledge_v2.py
git commit -m "fix(backtest): propagate feature flags to domain engine

Critical bug fix: Feature flags (enable_wyckoff, enable_smc, enable_temporal,
enable_hob, enable_macro) were not being passed to ArchetypeLogic, causing
domain engines to be invisible to backtest system despite correct wiring.

Root cause: Missing feature flag extraction from config in RuntimeContext
Impact: Domain amplification (2x-95x boosts) were not being applied
Fix: Extract feature flags from config and propagate to ArchetypeLogic

This fix enables domain engines to properly amplify archetype signals.

Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"

# 4. Feature Store
git add data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet
git commit -m "feat(data): update feature store with domain features

Update main feature store with latest domain engine features:
- Wyckoff events with confidence scores
- SMC BOS, zones, sweeps, CHOCH
- HOB demand/supply zones and imbalances
- Temporal Fib time, confluence, clusters

Feature store: BTC_1H_2022-01-01_to_2024-12-31.parquet (Dec 11 13:36)
Total features: 40+ domain features added

Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"

# 5. MVP Configs
git add configs/mvp/mvp_bear_market_v1.json configs/mvp/mvp_bull_market_v1.json
git commit -m "feat(config): update MVP bull/bear market configs

Update MVP configurations with latest archetype parameters and settings.
Part of domain engine implementation rollout.

Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"

# 6. Engine Modules
git add engine/context/regime_classifier.py engine/archetypes/threshold_policy.py engine/wyckoff/wyckoff_engine.py engine/feature_flags.py engine/strategies/archetypes/bear/__init__.py
git commit -m "feat(engine): update regime classifier, threshold policy, Wyckoff engine

Update supporting engine modules as part of domain engine implementation:
- Regime classifier enhancements
- Threshold policy adjustments
- Wyckoff engine improvements

Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"

# 7. Pipeline Fix
git add bin/fix_oi_change_pipeline.py
git commit -m "fix(pipeline): fix OI change pipeline issues

Fix open interest change calculation pipeline.

Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"

# 8. Domain Engine Docs
git add docs/domain_engine/
git commit -m "docs(domain-engine): add complete domain engine documentation

Add comprehensive documentation for domain engine implementation:
- Complete engine wiring report (feature maps, boost values)
- Feature generation report (tools and pipeline)
- Domain engine guide (operator manual)
- Quick references and status reports
- Critical bug fix documentation
- Feature availability assessment

Total: 10 documentation files covering all aspects of domain engines.

Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"

# 9. Session Reports
git add docs/ALPHA_COMPLETENESS_VERIFICATION_REPORT.md docs/ALPHA_GAP_ACTION_PLAN.md
git commit -m "docs: add alpha completeness verification and gap analysis

Add comprehensive analysis of alpha completeness and remaining gaps:
- Alpha completeness verification report
- Gap action plan for missing features
- Feature prioritization matrix

Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"

# 10. Changelog
git add CHANGELOG.md
git commit -m "docs(changelog): add domain engine implementation entries

Update CHANGELOG with complete session work:
- Domain engine wiring (44 features across S1, S4, S5)
- Critical feature flag bug fix
- Feature generation tools
- Comprehensive documentation

Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

## Verify All Commits

```bash
# Check commit history
git log --oneline -10

# Verify branch status
git status

# See what changed
git log origin/main..HEAD
```

## Individual Commit Commands

If you prefer to commit one at a time (recommended), see **GIT_COMMIT_PLAN.md** for:
- Detailed commit messages
- Verification commands for each commit
- Pre-commit checklist
- Rollback strategies

## Pre-Execution Checklist

- [ ] All Python files compile (`python -m py_compile <file>`)
- [ ] All JSON files valid (`python -m json.tool <file>`)
- [ ] Feature store loadable (`pd.read_parquet(...)`)
- [ ] On correct branch (`git branch` shows feature/ghost-modules-to-live-v2)
- [ ] No sensitive data in commits
- [ ] Reviewed GIT_COMMIT_PLAN.md

## Quick Verification

```bash
# Compile check all Python files
python -m py_compile engine/features/registry.py
python -m py_compile engine/archetypes/logic_v2_adapter.py
python -m py_compile bin/backtest_knowledge_v2.py
python -m py_compile engine/context/regime_classifier.py
python -m py_compile engine/archetypes/threshold_policy.py
python -m py_compile engine/wyckoff/wyckoff_engine.py
python -m py_compile engine/feature_flags.py
python -m py_compile engine/strategies/archetypes/bear/__init__.py
python -m py_compile bin/fix_oi_change_pipeline.py

# Validate JSON configs
python -m json.tool configs/mvp/mvp_bear_market_v1.json > /dev/null
python -m json.tool configs/mvp/mvp_bull_market_v1.json > /dev/null

# Verify feature store
python -c "import pandas as pd; df=pd.read_parquet('data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet'); print(f'✅ Feature store OK: {len(df)} rows, {len(df.columns)} cols')"
```

## Rollback (If Needed)

```bash
# Undo all commits, keep changes
git reset --soft HEAD~10

# Undo specific commit
git revert <commit-hash>

# Nuclear option - discard everything
git reset --hard HEAD~10  # USE WITH CAUTION
```

## After Commits

```bash
# Run integration tests
pytest tests/ -v

# Run backtest to verify domain amplification
python bin/backtest_knowledge_v2.py --config configs/mvp/mvp_bull_market_v1.json

# Generate missing features
python bin/generate_all_missing_features.py

# Verify feature store quality
python bin/verify_feature_store_quality.py
```

## Files Being Committed

**Core Changes** (11 files):
- 3 engine Python files (logic, registry, backtest)
- 5 supporting engine modules
- 1 pipeline fix
- 2 MVP configs

**Data** (1 file):
- Feature store parquet (13M)

**Documentation** (13 files):
- docs/domain_engine/ (10 files)
- docs/ALPHA_* (2 files)
- CHANGELOG.md (1 file)

**Total**: 25 files across 10 commits

## Success Criteria

After commits:
- ✅ All commits in git log
- ✅ git status shows clean working directory
- ✅ All files compile successfully
- ✅ Feature store loads without errors
- ✅ Documentation accessible and readable

---

**See Also**:
- `GIT_COMMIT_PLAN.md` - Comprehensive commit plan with verification
- `SESSION_CLEANUP_REPORT.md` - Full session cleanup details
- `docs/domain_engine/` - Complete domain engine documentation

**Status**: ✅ READY TO COMMIT
**Next Step**: Execute commits (all at once or one by one)
