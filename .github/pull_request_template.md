## Summary
Brief description of changes and motivation.

---

## 🔄 Version & Release Checklist

### Version Consistency
- [ ] Version in `bull_machine/version.py` is correct
- [ ] No hardcoded version strings (use `from bull_machine.version import __version__`)
- [ ] README rendered from template (no manual v1.X.Y edits)
- [ ] Version Sync CI check ✅

### Testing & Quality
- [ ] All tests passing locally
- [ ] CI checks passing (ruff, mypy, pytest)
- [ ] Performance regression tests completed (if applicable)
- [ ] Breaking changes documented (if any)

### Documentation
- [ ] CHANGELOG.md entry added (if user-facing changes)
- [ ] Performance charts regenerated (if applicable)
- [ ] Screenshots updated (if UI changes)

### Release Artifacts
- [ ] Key results attached (CSV, JSON reports)
- [ ] Performance plots generated with correct version in title
- [ ] Backtest data validated

---

## 🎯 Key Changes
- [ ] New feature
- [ ] Bug fix
- [ ] Performance improvement
- [ ] Refactoring
- [ ] Documentation
- [ ] CI/CD improvement

**Details:**
[Describe the key technical changes]

---

## 📊 Performance Impact (if applicable)
- **Win Rate**: N/A
- **PnL**: N/A
- **Max Drawdown**: N/A
- **Trade Frequency**: N/A

---

## 🧪 Testing
- [ ] Unit tests added/updated
- [ ] Integration tests passing
- [ ] Manual testing completed
- [ ] Smoke tests passing

**Test Results:**
[Summary of test results or link to CI]

---

## 🚨 Breaking Changes
- [ ] No breaking changes
- [ ] Breaking changes documented below

**Migration Required:**
[If breaking changes, provide migration guide]

---

## 📎 Additional Notes
[Any additional context, deployment notes, or follow-up items]

---

**Auto-generated checklist reminder**: Run `python scripts/render_readme.py` if you modified version info.