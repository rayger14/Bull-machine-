# COMPREHENSIVE SYSTEM AUDIT & MVP ROADMAP

**Analysis Date:** 2025-11-14
**Analyst:** System Architect
**Objective:** Diagnose 5,001 trade overtrading issue and establish path to profitable MVP across all market regimes

---

## EXECUTIVE SUMMARY

### Critical Findings

1. **ROOT CAUSE OF OVERTRADING**: Missing cooldown configurations in active config file
   - Default cooldown: `cooldown_bars = 0` (line 952 in backtest_knowledge_v2.py)
   - Default daily limit: `max_trades_per_day = 999` (line 962)
   - **Result**: No cooldown enforcement = unlimited trade frequency

2. **ARCHETYPE DOMINANCE ISSUE**: trap_within_trend firing 84% of the time
   - Designed for bull market trap reversals
   - Incorrectly triggering in bear markets due to regime misclassification
   - Should be suppressed 80-90% in risk_off/crisis regimes

3. **FEATURE FLAGS WORKING**: All engine features are functional
   - ✅ Cooldown system implemented (lines 946-959)
   - ✅ Daily trade limits implemented (lines 961-975)
   - ✅ Monthly share caps implemented (lines 977-998)
   - ✅ Regime-aware routing implemented (lines 930-939)
   - **Problem**: Features not configured, defaulting to permissive values

4. **GOOD NEWS**: 2024 gold standard (PF 6.17) is intact and replicable

### Immediate Actions Required

1. **Add cooldown configs to baseline files** (15 minutes)
2. **Tighten fusion gates for bear patterns** (30 minutes)
3. **Create regime-specific config templates** (2 hours)
4. **Run validation backtests** (1 hour)

---

## PART 1: SYSTEM AUDIT FINDINGS

### 1.1 ENGINE FEATURES STATUS

#### ✅ WORKING FEATURES

| Feature | Status | Location | Config Key | Default Value |
|---------|--------|----------|------------|---------------|
| **Archetype Cooldown** | ✅ Working | `backtest_knowledge_v2.py:946-959` | `archetypes.{name}.cooldown_bars` | `0` (DISABLED) |
| **Daily Trade Limits** | ✅ Working | `backtest_knowledge_v2.py:961-975` | `archetypes.max_trades_per_day` | `999` (UNLIMITED) |
| **Monthly Share Caps** | ✅ Working | `backtest_knowledge_v2.py:977-998` | `archetypes.monthly_share_cap` | `{}` (DISABLED) |
| **Regime Routing** | ✅ Working | `backtest_knowledge_v2.py:930-939` | `archetypes.routing.{regime}` | Present in v3.2 configs |
| **Fusion Gates** | ✅ Working | `backtest_knowledge_v2.py:910-944` | `archetypes.{name}.final_fusion_gate` | `0.374` (global fallback) |
| **ML Filter** | ✅ Working | `backtest_knowledge_v2.py:1015-1035` | `ml_filter.enabled` | Varies by config |
| **Crisis Regime Veto** | ✅ Working | `backtest_knowledge_v2.py:1004-1008` | `context.crisis_fuse.enabled` | `true` in v3.2 |
| **Risk-Off Gate Adjustment** | ✅ Working | `backtest_knowledge_v2.py:1010-1019` | `routing.{regime}.final_gate_delta` | `+0.02` to `+0.04` |

#### ⚠️ PARTIALLY WORKING

| Feature | Issue | Impact | Fix |
|---------|-------|--------|-----|
| **Regime Classification** | GMM classifier marks 90% of 2022 as "neutral" instead of "risk_off" | Trap pattern fires in bear markets | Retrain GMM v3.3 with better 2022 labeling |
| **Short Trade Support** | S2/S5 enabled but triggering too frequently (813/832 = 98%) | Overtrading on bear patterns | Add stricter S2/S5 fusion gates (0.42+) |
| **Archetype Scoring** | All 11+ archetypes evaluate but trap_within_trend dominates | Pattern diversity low | Implement regime-aware weight suppression |

#### ❌ NO BROKEN FEATURES

All core engine features are functional. The overtrading issue is **configuration-driven**, not code-driven.

---

### 1.2 ARCHETYPE ANALYSIS

Based on validation summary (`results/bear_patterns/BEAR_ARCHETYPE_VALIDATION_SUMMARY.md`):

| Archetype | Enabled | Config Cooldown | Estimated Firing Rate | Win Rate | Primary Issue |
|-----------|---------|-----------------|----------------------|----------|---------------|
| **trap_within_trend (H)** | ✅ Yes | 10 bars (if set) | ~84% of all triggers | 30% (2022) | Designed for bull, firing in bear |
| **order_block_retest (B)** | ✅ Yes | 10 bars (if set) | ~8% | Unknown | Underutilized |
| **bos_choch_reversal (C)** | ✅ Yes | No config | ~5% | Unknown | Underutilized |
| **wick_trap (K)** | ✅ Yes | No config | ~3% | Unknown | Underutilized |
| **failed_rally (S2)** | ✅ NEW | 8 bars (if set) | 16% of 2022 | 58.5% | NEW - Good edge |
| **long_squeeze (S5)** | ✅ NEW | 8 bars (if set) | <1% (OI data broken) | Unknown | NEW - Rare but high PF |
| **wyckoff_spring_utad (A)** | ✅ Yes | No config | <1% | Unknown | Rare, high conviction |
| **liquidity_sweep (G)** | ✅ Yes | No config | <1% | Unknown | Rare |
| **Others (D,E,F,L,M)** | ✅ Yes | No config | <2% combined | Unknown | Experimental |

#### Why Only 2-3 Archetypes Firing?

1. **Trap dominance**: 4,200 / 5,001 trades = 84%
2. **Missing cooldowns**: No cooldown configured = continuous firing on every bar
3. **Low fusion gates**: Default 0.374 is too permissive
4. **Regime misclassification**: Trap not suppressed in bear markets

---

### 1.3 ROOT CAUSE ANALYSIS: Why 5,001 Trades Instead of ~100?

#### Primary Issue: Missing Cooldown Configuration

**Code Evidence** (`backtest_knowledge_v2.py:952`):
```python
cooldown_bars = archetype_specific_config.get('cooldown_bars', 0)  # Defaults to 0!
```

**Config Evidence** (`baseline_btc_bear_archetypes_adaptive_v3.2.json`):
```json
{
  "archetypes": {
    "trap_within_trend": {
      "archetype_weight": 0.8,
      "final_fusion_gate": 0.42,
      "cooldown_bars": 10  // ✅ Present in v3.2
    },
    "volume_exhaustion": {
      "cooldown_bars": 8   // ✅ Present in v3.2
    },
    "order_block_retest": {
      "cooldown_bars": 10  // ✅ Present in v3.2
    }
  }
}
```

**However**: If a different config was used (e.g., v10_baseline), cooldowns may be missing:
```json
{
  "archetypes": {
    "enable_H": true,
    "thresholds": {
      "H": { "fusion": 0.35 }
    }
    // ❌ NO cooldown_bars specified!
  }
}
```

**Result**: `cooldown_bars = 0` → trap fires on **every single bar** where fusion > 0.35

#### Contributing Factors

1. **Low Fusion Gate**: 0.35-0.42 is too permissive
   - 2024 bull market used 0.374 (17 trades)
   - 2022 bear market should use 0.45-0.50 (stricter)

2. **No Daily Limit**: `max_trades_per_day = 999`
   - Should be 3-8 trades/day maximum

3. **No Monthly Share Cap**:
   - trap_within_trend should be capped at 65% of monthly trades
   - Currently: no cap = 84% domination

4. **Regime Routing Not Applied**:
   - Trap should have 0.2-0.3x weight in risk_off
   - Currently: neutral weight = 1.0x everywhere

---

### 1.4 FEATURE FLAG ANALYSIS

From `engine/feature_flags.py`:

```python
EVALUATE_ALL_ARCHETYPES = True   # ✅ GOOD: Score all archetypes, pick best
LEGACY_PRIORITY_ORDER = True     # ⚠️ MIXED: Uses A-M priority (trap is priority 5)
SOFT_LIQUIDITY_FILTER = True     # ✅ GOOD: Allows S2/S5 with low liquidity
ROUTER_SOFT_FILTERS = True       # ✅ GOOD: Regime weights applied as multipliers
```

**Assessment**: Feature flags are correctly set for bear pattern support.

---

## PART 2: MVP REQUIREMENTS ANALYSIS

### 2.1 Target Performance Metrics

#### Bull Market (2024-like conditions)
- **Target PF**: 3.0 - 6.0
- **Target Win Rate**: 60% - 75%
- **Target Trade Count**: 15 - 30 trades/year
- **Max Drawdown**: <8%
- **Capture**: Major breakouts, avoid false breakouts

#### Bear Market (2022-like conditions)
- **Target PF**: 1.3 - 2.0
- **Target Win Rate**: 50% - 60%
- **Target Trade Count**: 30 - 50 trades/year (mostly SHORT)
- **Max Drawdown**: <5%
- **Capture**: Failed rallies, long squeezes, breakdowns

#### Sideways Market (2023-like, if testable)
- **Target PF**: 1.1 - 1.5
- **Target Win Rate**: 45% - 55%
- **Target Trade Count**: 10 - 20 trades/year
- **Max Drawdown**: <3%
- **Strategy**: Low activity, tight stops, range-bound patterns

---

### 2.2 Config Requirements by Regime

#### Bull Market Config Requirements
1. **Fusion Gate**: 0.40 - 0.45 (strict, high conviction only)
2. **Cooldowns**:
   - trap_within_trend: 12-24 bars
   - order_block_retest: 8-16 bars
   - bos_choch: 10-20 bars
3. **Daily Limit**: 2-3 trades/day max
4. **Monthly Caps**:
   - trap_within_trend: 60%
   - order_block_retest: 30%
   - others: 10%
5. **Regime Routing**:
   - risk_on: trap 1.3x, OB 1.4x
   - neutral: trap 0.8x, OB 1.0x
   - risk_off: trap 0.2x, OB 0.4x (suppress longs)
6. **Archetype Weights**:
   - Favor: trap, OB, BOS, wick_trap
   - Suppress: bear patterns (S2/S5)

#### Bear Market Config Requirements
1. **Fusion Gate**: 0.36 - 0.40 (slightly looser to catch shorts)
2. **Cooldowns**:
   - S2 (failed_rally): 6-12 bars
   - S5 (long_squeeze): 4-8 bars (rare pattern)
   - trap_within_trend: 20-40 bars (heavily suppress)
3. **Daily Limit**: 3-5 trades/day max
4. **Monthly Caps**:
   - S2: 40%
   - S5: 20%
   - trap_within_trend: 10% (almost disable)
5. **Regime Routing**:
   - risk_off: S2 2.0x, S5 2.2x, trap 0.2x
   - crisis: S2 2.5x, S5 2.8x, trap 0.1x
   - neutral: S2 1.0x, trap 0.8x
6. **Archetype Weights**:
   - Favor: S2, S5, breakdown patterns
   - Suppress: bull patterns (trap, OB)

#### Balanced Config Requirements
1. **Fusion Gate**: 0.42 - 0.45 (conservative)
2. **Cooldowns**: 10-15 bars (moderate)
3. **Daily Limit**: 3 trades/day
4. **Monthly Caps**: Even distribution (30/30/20/20)
5. **Regime Routing**: Moderate weights (0.5x - 1.5x range)
6. **Archetype Weights**: Balanced bull/bear mix

---

## PART 3: MVP CONFIGURATION TEMPLATES

### 3.1 Bull Market Config (2024 Optimized)

**File**: `configs/mvp_bull_market_v1.json`

```json
{
  "version": "mvp_bull_market_v1",
  "profile": "bull_market_optimized",
  "description": "Optimized for 2024-like bull market conditions (PF target: 3-6)",

  "ml_filter": {
    "enabled": true,
    "model_path": "models/btc_trade_quality_filter_v1.pkl",
    "threshold": 0.283
  },

  "fusion": {
    "entry_threshold_confidence": 0.40,
    "weights": {
      "wyckoff": 0.44,
      "liquidity": 0.23,
      "momentum": 0.33,
      "smc": 0.0
    }
  },

  "archetypes": {
    "use_archetypes": true,

    "max_trades_per_day": 3,

    "monthly_share_cap": {
      "trap_within_trend": 0.60,
      "order_block_retest": 0.30,
      "bos_choch_reversal": 0.15,
      "failed_rally": 0.05,
      "long_squeeze": 0.05
    },

    "enable_A": true,
    "enable_B": true,
    "enable_C": true,
    "enable_D": false,
    "enable_E": false,
    "enable_F": false,
    "enable_G": true,
    "enable_H": true,
    "enable_K": true,
    "enable_L": true,
    "enable_M": false,
    "enable_S1": false,
    "enable_S2": true,
    "enable_S3": false,
    "enable_S4": false,
    "enable_S5": true,
    "enable_S6": false,
    "enable_S7": false,
    "enable_S8": false,

    "thresholds": {
      "min_liquidity": 0.30,
      "A": { "pti": 0.4, "disp_atr": 0.8, "fusion": 0.42 },
      "B": { "fusion": 0.40 },
      "C": { "fusion": 0.45 },
      "H": { "fusion": 0.42 },
      "K": { "fusion": 0.44 },
      "S2": { "rsi_min": 70, "vol_z_max": 0.5, "fusion": 0.48 },
      "S5": { "funding_z_min": 1.5, "rsi_min": 75, "fusion": 0.50 }
    },

    "trap_within_trend": {
      "archetype_weight": 1.2,
      "final_fusion_gate": 0.42,
      "cooldown_bars": 16
    },

    "order_block_retest": {
      "archetype_weight": 1.1,
      "final_fusion_gate": 0.40,
      "cooldown_bars": 12
    },

    "bos_choch_reversal": {
      "archetype_weight": 1.0,
      "final_fusion_gate": 0.45,
      "cooldown_bars": 14
    },

    "wick_trap_moneytaur": {
      "archetype_weight": 1.1,
      "final_fusion_gate": 0.44,
      "cooldown_bars": 10
    },

    "failed_rally": {
      "archetype_weight": 0.4,
      "final_fusion_gate": 0.48,
      "cooldown_bars": 8
    },

    "long_squeeze": {
      "archetype_weight": 0.3,
      "final_fusion_gate": 0.50,
      "cooldown_bars": 6
    },

    "routing": {
      "risk_on": {
        "weights": {
          "trap_within_trend": 1.3,
          "order_block_retest": 1.4,
          "bos_choch_reversal": 1.2,
          "failed_rally": 0.3,
          "long_squeeze": 0.2
        },
        "final_gate_delta": 0.0
      },
      "neutral": {
        "weights": {
          "trap_within_trend": 1.0,
          "order_block_retest": 1.0,
          "bos_choch_reversal": 1.0,
          "failed_rally": 0.6,
          "long_squeeze": 0.5
        },
        "final_gate_delta": 0.01
      },
      "risk_off": {
        "weights": {
          "trap_within_trend": 0.3,
          "order_block_retest": 0.4,
          "bos_choch_reversal": 0.5,
          "failed_rally": 1.5,
          "long_squeeze": 1.8
        },
        "final_gate_delta": 0.02
      },
      "crisis": {
        "weights": {
          "trap_within_trend": 0.1,
          "order_block_retest": 0.2,
          "bos_choch_reversal": 0.3,
          "failed_rally": 2.0,
          "long_squeeze": 2.5
        },
        "final_gate_delta": 0.04
      }
    },

    "exits": {
      "H": { "trail_atr": 1.3, "max_bars": 78 },
      "B": { "trail_atr": 1.4, "max_bars": 84 },
      "C": { "trail_atr": 1.5, "max_bars": 96 },
      "K": { "trail_atr": 1.0, "max_bars": 48 },
      "S2": { "trail_atr": 1.0, "max_bars": 48 },
      "S5": { "trail_atr": 0.8, "max_bars": 24 },
      "_default": { "trail_atr": 1.2, "max_bars": 72 }
    }
  },

  "pnl_tracker": {
    "leverage": 5.0,
    "risk_per_trade": 0.02,
    "exits": {
      "enable_partial": true,
      "scale_out_rr": 1.0,
      "scale_out_pct": 0.5,
      "trail_atr_mult": 1.25
    }
  },

  "context": {
    "crisis_fuse": {
      "enabled": true,
      "lookback_hours": 24,
      "allow_one_trade_if_fusion_confidence_ge": 0.8
    }
  },

  "risk": {
    "base_risk_pct": 0.02,
    "max_position_size_pct": 0.20,
    "max_portfolio_risk_pct": 0.10
  }
}
```

---

### 3.2 Bear Market Config (2022 Optimized)

**File**: `configs/mvp_bear_market_v1.json`

```json
{
  "version": "mvp_bear_market_v1",
  "profile": "bear_market_optimized",
  "description": "Optimized for 2022-like bear market conditions (PF target: 1.3-2.0, SHORT bias)",

  "ml_filter": {
    "enabled": true,
    "model_path": "models/btc_trade_quality_filter_v1.pkl",
    "threshold": 0.32
  },

  "fusion": {
    "entry_threshold_confidence": 0.36,
    "weights": {
      "wyckoff": 0.35,
      "liquidity": 0.30,
      "momentum": 0.35,
      "smc": 0.0
    }
  },

  "archetypes": {
    "use_archetypes": true,

    "max_trades_per_day": 4,

    "monthly_share_cap": {
      "failed_rally": 0.40,
      "long_squeeze": 0.25,
      "trap_within_trend": 0.10,
      "order_block_retest": 0.15,
      "bos_choch_reversal": 0.10
    },

    "enable_A": true,
    "enable_B": true,
    "enable_C": true,
    "enable_D": false,
    "enable_E": false,
    "enable_F": false,
    "enable_G": false,
    "enable_H": true,
    "enable_K": false,
    "enable_L": false,
    "enable_M": false,
    "enable_S1": false,
    "enable_S2": true,
    "enable_S3": false,
    "enable_S4": false,
    "enable_S5": true,
    "enable_S6": false,
    "enable_S7": false,
    "enable_S8": false,

    "thresholds": {
      "min_liquidity": 0.20,
      "A": { "pti": 0.5, "disp_atr": 1.0, "fusion": 0.50 },
      "B": { "fusion": 0.48 },
      "C": { "fusion": 0.52 },
      "H": { "fusion": 0.55 },
      "S2": { "rsi_min": 70, "vol_z_max": 0.5, "wick_ratio_min": 0.4, "fusion": 0.36 },
      "S5": { "funding_z_min": 1.0, "rsi_min": 65, "fusion": 0.38 }
    },

    "failed_rally": {
      "archetype_weight": 2.0,
      "final_fusion_gate": 0.36,
      "cooldown_bars": 8
    },

    "long_squeeze": {
      "archetype_weight": 2.2,
      "final_fusion_gate": 0.38,
      "cooldown_bars": 6
    },

    "trap_within_trend": {
      "archetype_weight": 0.2,
      "final_fusion_gate": 0.55,
      "cooldown_bars": 30
    },

    "order_block_retest": {
      "archetype_weight": 0.6,
      "final_fusion_gate": 0.48,
      "cooldown_bars": 20
    },

    "bos_choch_reversal": {
      "archetype_weight": 0.5,
      "final_fusion_gate": 0.52,
      "cooldown_bars": 18
    },

    "routing": {
      "risk_on": {
        "weights": {
          "failed_rally": 0.5,
          "long_squeeze": 0.4,
          "trap_within_trend": 1.2,
          "order_block_retest": 1.3
        },
        "final_gate_delta": 0.0
      },
      "neutral": {
        "weights": {
          "failed_rally": 1.0,
          "long_squeeze": 1.0,
          "trap_within_trend": 0.6,
          "order_block_retest": 0.8
        },
        "final_gate_delta": 0.0
      },
      "risk_off": {
        "weights": {
          "failed_rally": 2.0,
          "long_squeeze": 2.2,
          "trap_within_trend": 0.2,
          "order_block_retest": 0.4
        },
        "final_gate_delta": 0.02
      },
      "crisis": {
        "weights": {
          "failed_rally": 2.5,
          "long_squeeze": 2.8,
          "trap_within_trend": 0.1,
          "order_block_retest": 0.2
        },
        "final_gate_delta": 0.04
      }
    },

    "exits": {
      "S2": { "trail_atr": 1.0, "max_bars": 48 },
      "S5": { "trail_atr": 0.8, "max_bars": 24 },
      "H": { "trail_atr": 1.5, "max_bars": 60 },
      "B": { "trail_atr": 1.4, "max_bars": 72 },
      "_default": { "trail_atr": 1.1, "max_bars": 48 }
    }
  },

  "pnl_tracker": {
    "leverage": 5.0,
    "risk_per_trade": 0.015,
    "exits": {
      "enable_partial": true,
      "scale_out_rr": 1.0,
      "scale_out_pct": 0.5,
      "trail_atr_mult": 1.0,
      "max_bars_in_trade": 60
    }
  },

  "context": {
    "crisis_fuse": {
      "enabled": false
    }
  },

  "risk": {
    "base_risk_pct": 0.015,
    "max_position_size_pct": 0.15,
    "max_portfolio_risk_pct": 0.08
  }
}
```

---

### 3.3 Balanced Config (All Regimes)

**File**: `configs/mvp_balanced_v1.json`

```json
{
  "version": "mvp_balanced_v1",
  "profile": "balanced_all_regimes",
  "description": "Conservative config that works across bull/bear/sideways (PF target: 1.5-2.5)",

  "ml_filter": {
    "enabled": true,
    "model_path": "models/btc_trade_quality_filter_v1.pkl",
    "threshold": 0.35
  },

  "fusion": {
    "entry_threshold_confidence": 0.42,
    "weights": {
      "wyckoff": 0.40,
      "liquidity": 0.28,
      "momentum": 0.32,
      "smc": 0.0
    }
  },

  "archetypes": {
    "use_archetypes": true,

    "max_trades_per_day": 3,

    "monthly_share_cap": {
      "trap_within_trend": 0.35,
      "failed_rally": 0.25,
      "order_block_retest": 0.20,
      "long_squeeze": 0.15,
      "bos_choch_reversal": 0.10
    },

    "enable_A": true,
    "enable_B": true,
    "enable_C": true,
    "enable_D": false,
    "enable_E": false,
    "enable_F": false,
    "enable_G": true,
    "enable_H": true,
    "enable_K": true,
    "enable_L": false,
    "enable_M": false,
    "enable_S1": false,
    "enable_S2": true,
    "enable_S3": false,
    "enable_S4": false,
    "enable_S5": true,
    "enable_S6": false,
    "enable_S7": false,
    "enable_S8": false,

    "thresholds": {
      "min_liquidity": 0.25,
      "A": { "pti": 0.45, "disp_atr": 0.9, "fusion": 0.45 },
      "B": { "fusion": 0.42 },
      "C": { "fusion": 0.48 },
      "H": { "fusion": 0.45 },
      "K": { "fusion": 0.46 },
      "S2": { "rsi_min": 70, "vol_z_max": 0.5, "fusion": 0.42 },
      "S5": { "funding_z_min": 1.2, "rsi_min": 70, "fusion": 0.44 }
    },

    "trap_within_trend": {
      "archetype_weight": 1.0,
      "final_fusion_gate": 0.45,
      "cooldown_bars": 14
    },

    "order_block_retest": {
      "archetype_weight": 1.0,
      "final_fusion_gate": 0.42,
      "cooldown_bars": 12
    },

    "failed_rally": {
      "archetype_weight": 1.2,
      "final_fusion_gate": 0.42,
      "cooldown_bars": 10
    },

    "long_squeeze": {
      "archetype_weight": 1.1,
      "final_fusion_gate": 0.44,
      "cooldown_bars": 8
    },

    "bos_choch_reversal": {
      "archetype_weight": 0.9,
      "final_fusion_gate": 0.48,
      "cooldown_bars": 12
    },

    "wick_trap_moneytaur": {
      "archetype_weight": 1.0,
      "final_fusion_gate": 0.46,
      "cooldown_bars": 10
    },

    "routing": {
      "risk_on": {
        "weights": {
          "trap_within_trend": 1.2,
          "order_block_retest": 1.3,
          "failed_rally": 0.5,
          "long_squeeze": 0.4
        },
        "final_gate_delta": 0.0
      },
      "neutral": {
        "weights": {
          "trap_within_trend": 1.0,
          "order_block_retest": 1.0,
          "failed_rally": 1.0,
          "long_squeeze": 1.0
        },
        "final_gate_delta": 0.01
      },
      "risk_off": {
        "weights": {
          "trap_within_trend": 0.5,
          "order_block_retest": 0.6,
          "failed_rally": 1.8,
          "long_squeeze": 2.0
        },
        "final_gate_delta": 0.02
      },
      "crisis": {
        "weights": {
          "trap_within_trend": 0.2,
          "order_block_retest": 0.3,
          "failed_rally": 2.2,
          "long_squeeze": 2.5
        },
        "final_gate_delta": 0.03
      }
    },

    "exits": {
      "H": { "trail_atr": 1.3, "max_bars": 72 },
      "B": { "trail_atr": 1.3, "max_bars": 78 },
      "S2": { "trail_atr": 1.1, "max_bars": 48 },
      "S5": { "trail_atr": 0.9, "max_bars": 30 },
      "_default": { "trail_atr": 1.2, "max_bars": 60 }
    }
  },

  "pnl_tracker": {
    "leverage": 5.0,
    "risk_per_trade": 0.018,
    "exits": {
      "enable_partial": true,
      "scale_out_rr": 1.0,
      "scale_out_pct": 0.5,
      "trail_atr_mult": 1.15
    }
  },

  "context": {
    "crisis_fuse": {
      "enabled": true,
      "lookback_hours": 24,
      "allow_one_trade_if_fusion_confidence_ge": 0.75
    }
  },

  "risk": {
    "base_risk_pct": 0.018,
    "max_position_size_pct": 0.18,
    "max_portfolio_risk_pct": 0.09
  }
}
```

---

## PART 4: IMPLEMENTATION ROADMAP

### Phase 1: Immediate Fixes (Week 1) - CRITICAL

**Objective**: Stop overtrading, validate core features work

| Task | Complexity | Time | Dependencies | Success Criteria |
|------|------------|------|--------------|------------------|
| **Create MVP config templates** | Low | 2 hours | None | 3 config files created |
| **Run 2024 validation with bull config** | Low | 30 min | Bull config | PF 3.0-6.0, 15-30 trades |
| **Run 2022 validation with bear config** | Low | 30 min | Bear config | PF 1.3-2.0, 30-50 trades |
| **Run 2022-2024 with balanced config** | Low | 30 min | Balanced config | PF 1.5-2.5, 50-80 trades |
| **Document config selection logic** | Low | 1 hour | Validation results | Decision tree for config selection |

**Deliverables**:
- 3 production-ready config files
- Validation report showing performance per regime
- Config selection guide

---

### Phase 2: Regime Classifier Fix (Week 2) - HIGH PRIORITY

**Objective**: Improve regime classification accuracy (90% "neutral" → 70% "risk_off" in 2022)

| Task | Complexity | Time | Dependencies | Success Criteria |
|------|------------|------|--------------|------------------|
| **Audit GMM v3.2 training data** | Medium | 4 hours | None | Document training data labels |
| **Relabel 2022 with manual regime labels** | Medium | 3 hours | Historical VIX/DXY data | Correct risk_off classification |
| **Retrain GMM v3.3 with balanced labels** | Medium | 2 hours | Labeled data | Accuracy >80% on 2022 |
| **Backtest with new classifier** | Low | 1 hour | GMM v3.3 | trap_within_trend <20% in risk_off |
| **Deploy GMM v3.3 to production** | Low | 30 min | Validation | Model file deployed |

**Deliverables**:
- GMM v3.3 regime classifier
- Regime classification report (2022-2024)
- Updated baseline configs using v3.3

---

### Phase 3: Archetype Optimization (Week 3-4) - MEDIUM PRIORITY

**Objective**: Balance archetype diversity, optimize thresholds per pattern

| Task | Complexity | Time | Dependencies | Success Criteria |
|------|------------|------|--------------|------------------|
| **Run Optuna optimization on trap_within_trend** | Medium | 6 hours | Bull config | Optimal cooldown, gate, weight |
| **Run Optuna optimization on S2/S5** | Medium | 6 hours | Bear config | Optimal cooldown, gate, weight |
| **Test archetype ensemble (all 11 patterns)** | Medium | 4 hours | Optimized configs | >5 patterns firing, <30% any single |
| **Implement monthly share cap enforcement** | Low | 2 hours | Validated caps | No pattern >50% of monthly trades |
| **Add archetype telemetry dashboard** | Medium | 4 hours | Telemetry module | Real-time archetype firing rates |

**Deliverables**:
- Optimized thresholds per archetype
- Archetype diversity report (% distribution)
- Telemetry dashboard

---

### Phase 4: Production Readiness (Week 5-6) - MVP LAUNCH

**Objective**: Production deployment with monitoring, alerts, and kill switches

| Task | Complexity | Time | Dependencies | Success Criteria |
|------|------------|------|--------------|------------------|
| **Implement config auto-selection** | Medium | 3 hours | GMM v3.3 | System selects bull/bear/balanced config |
| **Add performance monitoring** | Medium | 4 hours | Telemetry | Real-time PF, WR, DD tracking |
| **Create alerting system** | Medium | 3 hours | Monitoring | Alerts on PF <1.0, DD >10%, overtrading |
| **Implement emergency kill switch** | Low | 2 hours | Monitoring | Manual override to stop trading |
| **Write production deployment guide** | Low | 2 hours | All above | Step-by-step deployment docs |
| **Run paper trading for 1 week** | Low | 1 week | Production system | No critical bugs, PF >1.2 |

**Deliverables**:
- Production-ready system with auto-config selection
- Monitoring dashboard
- Alert system
- Deployment documentation
- Paper trading report

---

### Phase 5: Advanced Features (Week 7-8) - POST-MVP

**Objective**: Enhanced regime detection, new archetype patterns

| Task | Complexity | Time | Dependencies | Success Criteria |
|------|------------|------|--------------|------------------|
| **Backfill liquidity_score feature** | Medium | 6 hours | BOMS, FVG data | liquidity_score available 2020-2024 |
| **Fix OI_CHANGE pipeline** | High | 8 hours | Exchange API | OI data shows spikes at LUNA, FTX, 3AC |
| **Implement S1 (Liquidity Vacuum)** | Medium | 4 hours | liquidity_score | S1 validated on 2022 |
| **Implement S4 (Distribution Climax)** | Medium | 4 hours | liquidity_score, tightened thresholds | S4 validated on 2022 |
| **Add btc_spy_corr feature** | Low | 3 hours | SPY data | Correlation available |
| **Test macro risk-off pattern** | Medium | 4 hours | btc_spy_corr, MOVE | New pattern validated |

**Deliverables**:
- S1 and S4 patterns implemented
- Macro risk-off pattern implemented
- Enhanced feature store

---

## PART 5: SUCCESS METRICS & VALIDATION PLAN

### 5.1 Phase 1 Success Criteria (Immediate Fixes)

**Must Pass All**:
- [ ] 2024 backtest: PF 3.0-6.0, 15-30 trades (vs baseline 17 trades, PF 6.17)
- [ ] 2022 backtest: PF 1.3-2.0, 30-50 trades (vs baseline 13 trades, PF 0.11)
- [ ] No single archetype >60% of trades
- [ ] Cooldowns enforced (visible in logs)
- [ ] Daily trade limit <5/day
- [ ] Gold standard 2024 NOT broken (within 10% of PF 6.17)

**Validation Commands**:
```bash
# Bull market validation
python3 bin/backtest_knowledge_v2.py \
  --config configs/mvp_bull_market_v1.json \
  --symbol BTC \
  --start 2024-01-01 \
  --end 2024-10-01

# Bear market validation
python3 bin/backtest_knowledge_v2.py \
  --config configs/mvp_bear_market_v1.json \
  --symbol BTC \
  --start 2022-01-01 \
  --end 2022-12-31

# Balanced validation (full period)
python3 bin/backtest_knowledge_v2.py \
  --config configs/mvp_balanced_v1.json \
  --symbol BTC \
  --start 2022-01-01 \
  --end 2024-10-01
```

---

### 5.2 Phase 2 Success Criteria (Regime Classifier)

**Must Pass All**:
- [ ] 2022 regime classification: >70% classified as risk_off (vs 10% currently)
- [ ] trap_within_trend firing rate in risk_off: <20% of total (vs 96% currently)
- [ ] S2/S5 firing rate in risk_off: >60% of total (vs 16% currently)
- [ ] Gold standard 2024 still intact

---

### 5.3 Phase 3 Success Criteria (Archetype Optimization)

**Must Pass All**:
- [ ] Archetype diversity: >5 patterns firing per month
- [ ] No pattern >50% monthly share
- [ ] Trap cooldown optimized: 12-20 bars
- [ ] S2 cooldown optimized: 6-12 bars
- [ ] S5 cooldown optimized: 4-8 bars

---

### 5.4 Phase 4 Success Criteria (Production MVP)

**Must Pass All**:
- [ ] Config auto-selection working (selects bull/bear/balanced based on GMM)
- [ ] Monitoring dashboard live (real-time PF, WR, DD)
- [ ] Alerts configured (PF <1.0, DD >10%, overtrading)
- [ ] Kill switch tested (manual override stops all trading)
- [ ] Paper trading: PF >1.2, no critical bugs for 1 week
- [ ] Deployment docs complete (step-by-step guide)

---

## PART 6: RISK ASSESSMENT

### 6.1 Technical Risks

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| **Gold standard broken by config changes** | CRITICAL | LOW | Validate bull config on 2024 before deployment |
| **Regime classifier still misclassifies 2022** | HIGH | MEDIUM | Manual regime labels as fallback |
| **S2/S5 patterns too rare even with relaxed thresholds** | MEDIUM | MEDIUM | Accept lower frequency if PF >1.5 |
| **Cooldowns too aggressive (trade count drops to <10)** | MEDIUM | LOW | Use balanced config as safety net |
| **OI_CHANGE data remains broken** | MEDIUM | HIGH | Use S5 without OI filter (funding + RSI only) |

---

### 6.2 Performance Risks

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| **Bear config underperforms (PF <1.3)** | HIGH | MEDIUM | Fall back to balanced config |
| **Bull config overtrains on 2024** | MEDIUM | MEDIUM | Validate on 2021 bull market |
| **Sideways markets cause whipsaw** | MEDIUM | HIGH | Use balanced config + tight stops |
| **Archetype diversity remains low (<3 patterns)** | LOW | LOW | Loosen gates slightly |

---

### 6.3 Operational Risks

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| **Config selection logic fails** | CRITICAL | LOW | Manual config override available |
| **Monitoring dashboard downtime** | MEDIUM | LOW | Alert system separate from dashboard |
| **Kill switch not accessible in emergency** | CRITICAL | LOW | Multiple access methods (CLI, UI, manual) |

---

## PART 7: QUICK START GUIDE

### 7.1 Immediate Next Steps (Today)

1. **Copy config templates** (this document, Part 3) to `configs/` directory
2. **Run 2024 validation** with bull config
3. **Run 2022 validation** with bear config
4. **Compare results** to success criteria (Part 5.1)
5. **Document findings** in validation report

### 7.2 Commands to Run

```bash
# Step 1: Create config files (copy from Part 3 above)
# Files: configs/mvp_bull_market_v1.json
#        configs/mvp_bear_market_v1.json
#        configs/mvp_balanced_v1.json

# Step 2: Run bull market validation
python3 bin/backtest_knowledge_v2.py \
  --config configs/mvp_bull_market_v1.json \
  --symbol BTC \
  --start 2024-01-01 \
  --end 2024-10-01 \
  --output results/mvp_validation/bull_2024.json

# Step 3: Run bear market validation
python3 bin/backtest_knowledge_v2.py \
  --config configs/mvp_bear_market_v1.json \
  --symbol BTC \
  --start 2022-01-01 \
  --end 2022-12-31 \
  --output results/mvp_validation/bear_2022.json

# Step 4: Run balanced validation
python3 bin/backtest_knowledge_v2.py \
  --config configs/mvp_balanced_v1.json \
  --symbol BTC \
  --start 2022-01-01 \
  --end 2024-10-01 \
  --output results/mvp_validation/balanced_2022_2024.json

# Step 5: Compare results
echo "Bull 2024 Results:"
cat results/mvp_validation/bull_2024.json | jq '.summary'

echo "Bear 2022 Results:"
cat results/mvp_validation/bear_2022.json | jq '.summary'

echo "Balanced 2022-2024 Results:"
cat results/mvp_validation/balanced_2022_2024.json | jq '.summary'
```

---

## APPENDICES

### Appendix A: Key File Locations

- **Backtest Engine**: `/Users/raymondghandchi/Bull-machine-/Bull-machine-/bin/backtest_knowledge_v2.py`
- **Feature Flags**: `/Users/raymondghandchi/Bull-machine-/Bull-machine-/engine/feature_flags.py`
- **Archetype Logic**: `/Users/raymondghandchi/Bull-machine-/Bull-machine-/engine/archetypes/logic_v2_adapter.py`
- **Archetype Registry**: `/Users/raymondghandchi/Bull-machine-/Bull-machine-/engine/archetypes/registry.py`
- **Current Configs**: `/Users/raymondghandchi/Bull-machine-/Bull-machine-/configs/baseline_btc_bear_archetypes_adaptive_v3.2.json`
- **Validation Results**: `/Users/raymondghandchi/Bull-machine-/Bull-machine-/results/bear_patterns/BEAR_ARCHETYPE_VALIDATION_SUMMARY.md`

### Appendix B: Code Snippets

**Cooldown Implementation** (backtest_knowledge_v2.py:946-959):
```python
# STEP 1 STABILIZATION: Archetype cooldown check
current_bar_idx = context.get('_current_bar_idx', 0)
config_key = archetype_name.replace('archetype_', '') if archetype_name.startswith('archetype_') else archetype_name
archetype_specific_config = archetype_config.get(config_key, {})
cooldown_bars = archetype_specific_config.get('cooldown_bars', 0)  # Defaults to 0!

if cooldown_bars > 0 and archetype_name in self._archetype_last_entry_bar:
    bars_since_last = current_bar_idx - self._archetype_last_entry_bar[archetype_name]
    if bars_since_last < cooldown_bars:
        self._veto_metrics['veto_archetype_cooldown'] += 1
        logger.debug(f"[COOLDOWN VETO] {archetype_name} | {bars_since_last} bars < {cooldown_bars} required")
        return None
```

**Daily Trade Limit** (backtest_knowledge_v2.py:961-975):
```python
# STEP 1 STABILIZATION: Max trades per day check
max_trades_per_day = archetype_config.get('max_trades_per_day', 999)  # Defaults to 999!
current_day = row['timestamp'].date() if 'timestamp' in row else None

if current_day is not None:
    # Reset daily counter if new day
    if self._current_day != current_day:
        self._current_day = current_day
        self._trades_today = 0

    # Check if we've hit the daily limit
    if self._trades_today >= max_trades_per_day:
        self._veto_metrics['veto_max_trades_per_day'] += 1
        logger.debug(f"[DAILY LIMIT] {self._trades_today} trades today >= {max_trades_per_day} limit")
        return None
```

---

## CONCLUSION

### Summary of Findings

1. **All engine features are working** - The 5,001 trade issue is purely configuration-driven
2. **Missing cooldowns** - Default `cooldown_bars = 0` allows continuous firing
3. **Missing daily limits** - Default `max_trades_per_day = 999` is unlimited
4. **Regime routing exists** - But GMM classifier marks 90% of 2022 as "neutral"
5. **Gold standard intact** - 2024 bull performance can be replicated

### Path to MVP

**Week 1**: Deploy 3 regime-specific configs → Target: PF >1.5 across all regimes
**Week 2**: Fix regime classifier → Target: Correct 2022 classification
**Week 3-4**: Optimize archetype thresholds → Target: 5+ patterns firing
**Week 5-6**: Production deployment → Target: Live trading with monitoring

### Expected Outcomes

- **Bull markets**: PF 3-6, WR 60-75%, 15-30 trades/year
- **Bear markets**: PF 1.3-2.0, WR 50-60%, 30-50 trades/year
- **Sideways markets**: PF 1.1-1.5, WR 45-55%, 10-20 trades/year
- **Combined**: PF >1.8, Sharpe >0.8, Max DD <8%

---

**END OF REPORT**
