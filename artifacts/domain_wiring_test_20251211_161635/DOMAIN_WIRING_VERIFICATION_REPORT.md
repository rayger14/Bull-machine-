# Domain Engine Wiring Verification Report

**Test Date:** 2025-12-11 16:21:48
**Test Period:** 2022-01-01 to 2022-12-31
**Asset:** BTC

## Executive Summary

**Status:** 1/3 archetypes verified

❌ **DOMAIN WIRING ISSUES** - Some archetypes not showing expected behavior

## Results Summary

| Archetype | Core PF | Full PF | Improvement | Core Trades | Full Trades | Wiring Status |
|-----------|---------|---------|-------------|-------------|-------------|---------------|
| S1 | 0.32 | 0.32 | +0.0% | 110 | 110 | ❌ |
| S4 | 0.36 | 0.50 | +38.9% | 122 | 156 | ✅ |
| S5 | 0.32 | 0.32 | +0.0% | 110 | 110 | ❌ |

## Detailed Results

### S1

**Core Variant:**
- Trades: 110
- Profit Factor: 0.32
- Win Rate: 31.8%
- Sharpe: -0.70

**Full Variant:**
- Trades: 110
- Profit Factor: 0.32
- Win Rate: 31.8%
- Sharpe: -0.70

**Verification Checks:**
- ❌ trades_differ: False
- ❌ pf_improved: False
- ❌ full_better: False
- ✅ config_flags_correct: True

**Status:** ❌ FAILED

---

### S4

**Core Variant:**
- Trades: 122
- Profit Factor: 0.36
- Win Rate: 34.4%
- Sharpe: -0.59

**Full Variant:**
- Trades: 156
- Profit Factor: 0.50
- Win Rate: 39.1%
- Sharpe: -0.35

**Verification Checks:**
- ✅ trades_differ: True
- ✅ pf_improved: True
- ✅ full_better: True
- ✅ config_flags_correct: True

**Status:** ✅ VERIFIED

---

### S5

**Core Variant:**
- Trades: 110
- Profit Factor: 0.32
- Win Rate: 31.8%
- Sharpe: -0.70

**Full Variant:**
- Trades: 110
- Profit Factor: 0.32
- Win Rate: 31.8%
- Sharpe: -0.70

**Verification Checks:**
- ❌ trades_differ: False
- ❌ pf_improved: False
- ❌ full_better: False
- ✅ config_flags_correct: True

**Status:** ❌ FAILED

---

## Recommendations

⚠️ Domain engine wiring needs attention:

**S1:**
- Fix: trades_differ
- Fix: pf_improved
- Fix: full_better

**S5:**
- Fix: trades_differ
- Fix: pf_improved
- Fix: full_better
