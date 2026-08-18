# Bull Machine v17 — Development Guidelines

## Current Architecture (2026-02-25)

**v17 Whale Footprint**: 16+1 archetypes + YAML hard_gates + 301-col feature store + Optuna ATR + signal dedup + heuristic fixes + whale conflict penalty + Smart Exits V2 + $100K capital

### Key Files
| Purpose | File |
|---------|------|
| **Backtester** | `bin/backtest_v11_standalone.py` (--commission-rate 0.0002, --slippage-bps 3, --start-date, --initial-cash 100000) |
| **Config** | `configs/bull_machine_isolated_v11_fixed.json` (dynamic threshold adaptive fusion) |
| **Feature Store** | `data/features_mtf/BTC_1H_FEATURES_V12_ENHANCED.parquet` (61,306 bars x 301 cols, 2018-2024) |
| **Archetype YAMLs** | `configs/archetypes/*.yaml` (hard_gates, fusion_weights, regime_preferences, exit params) |
| **Exit Logic** | `engine/archetypes/exit_logic.py` (Smart Exits V2: composite invalidation, distress half-exit, chop trailing) |
| **Archetype Engine** | `engine/archetypes/archetype_instance.py` (whale conflict penalty, hard gates, fusion scoring) |
| **Live Runner** | `bin/live/coinbase_runner.py` (Coinbase BTC-PERP-INTX paper trading) |
| **Dashboard** | `dashboard/` (React 19 + TypeScript + Vite + Tailwind CSS 4, served at port 8081) |

---

## Quick Validation

```bash
# Full backtest (2020-2024)
python3 bin/backtest_v11_standalone.py --start-date 2020-01-01 --commission-rate 0.0002 --slippage-bps 3

# Expect: ~915 trades, PF ~1.58, PnL ~$223K
```

---

## Design Constraints (rationale the code can't explain)

- **CMI regime system must stay orthogonal to archetype fusion** — no double-counting (`engine/context/regime_service.py`; weights/thresholds live in `configs/bull_machine_isolated_v11_fixed.json`)
- **Regime exit scaling is DISABLED** (all factors = 1.0) — tested net negative
- **derivatives_heat is DISABLED** (CMI weight = 0%) pending 3+ years of OI data
- **Whale conflict penalty + hard gates** run BEFORE fusion scoring (`engine/archetypes/archetype_instance.py`, per-archetype YAMLs)

---

## Common Gotchas

### NaN Guards
```python
# Always check for NaN in feature extraction
val = bar.get('some_feature', 0.0)
if val is not None and val == val:  # val != val is True for NaN
    use_value(val)
```

### Float Comparison
```python
# WRONG
if position.quantity == 0: ...

# CORRECT
if abs(position.quantity) < 1e-8: ...
```

### Exit Percentage Calculations
```python
# WRONG - compounds
exit_qty = current_qty * pct

# CORRECT - from original
exit_qty = original_qty * pct
```

---

## Deployment

Use the `/deploy-server` skill (server, SSH, deploy.sh, monitoring). **Never deploy without an explicit go from the user in the current exchange.**

---

## Git Workflow (MANDATORY)

**ALWAYS follow this workflow. Do NOT work directly on `main`.**

### Starting Any Task
1. **Create a feature branch** before writing any code:
   ```bash
   git checkout main && git pull
   git checkout -b feat/<short-description>   # e.g. feat/optuna-atr-optimization
   ```
2. Branch naming: `feat/`, `fix/`, `refactor/`, `chore/` prefixes

### During Work
3. **Commit frequently** — at least every meaningful change (not just at the end):
   - After wiring a new feature: commit
   - After fixing a bug: commit
   - After updating configs: commit
   - After adding a new script: commit
   - Rule of thumb: if you'd be upset losing the work, commit it
4. **Push to remote** after each commit session: `git push -u origin <branch>`

### Finishing a Task
5. **Push + create PR** when the task is complete:
   ```bash
   git push -u origin feat/<branch>
   gh pr create --title "feat: short description" --body "..."
   ```
6. **Merge to main** only after user approval (or PR review)
7. After merge, clean up: `git checkout main && git pull && git branch -d feat/<branch>`

### Commit Message Format
```
<type>: <short description>

<optional body with details>

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
```
Types: `feat`, `fix`, `refactor`, `chore`, `docs`, `perf`

### Never Do
- Never work directly on `main` with uncommitted changes
- Never force-push to `main`
- Never leave >5 modified files uncommitted
- Never deploy without committing first

---

## Testing Checklist

Before committing code changes:

- [ ] Run backtest: `python3 bin/backtest_v11_standalone.py --start-date 2020-01-01 --commission-rate 0.0002 --slippage-bps 3`
- [ ] Check PF >= 1.50 (floor), PnL >= $100K (floor)
- [ ] Check MaxDD <= -10.0% (ceiling)
- [ ] Validate config JSON: `python3 -m json.tool < configs/bull_machine_isolated_v11_fixed.json`
- [ ] Check git diff is reasonable (not 2,000+ lines)

---

## Optuna Optimization (MANDATORY)

**ALWAYS use the `/optuna-optimize` skill before running any Optuna optimization.** This skill enforces walk-forward validation, CPCV, and anti-overfit checks. Never run Optuna on the full date range without WFO/CPCV validation.

The skill contains the run commands and full rule set (WFO/CPCV modes, ≤15 params per group, baseline comparison, trade-count and importance checks).

---

## Project Knowledge Base

All learned knowledge, findings, and feedback are stored in `docs/knowledge/`. This directory is version-controlled and travels with the repo.

- **MEMORY.md** — Index of all knowledge files
- **project_emergent_knowledge.md** — Validated trading hypotheses (EMA/Wyckoff divergence REJECTED, OI during distribution CONFIRMED inverted)
- **project_confluence_breakout_investigation.md** — FRVP bug, bypass_fusion_threshold, gate tightening TODO
- **feedback_never_sed_patch.md** — Never patch server directly, always commit first
- **feedback_backtest_output.md** — Always show date range, starting equity, avg risk per trade

Read `docs/knowledge/MEMORY.md` at the start of any session for full context.
