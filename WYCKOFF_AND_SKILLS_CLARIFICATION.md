# Wyckoff Implementation & Skills Usage - Critical Clarifications

**Date**: 2026-01-19
**Purpose**: Address critical questions about Wyckoff changes and practical skill usage

---

## ⚠️ CRITICAL FINDINGS

### 1. WYCKOFF STATES - PARTIALLY IMPLEMENTED (NEEDS WORK)

**Question**: "Are the Wyckoff knowledge we had said previously regarding all states added?"

**Answer**: ❌ **PARTIALLY** - The code was added but the **features don't exist yet**

#### What Was ALREADY In The Code (Before My Changes):
```python
# EXISTING Wyckoff signals (from original code):
✅ wyckoff_spring_a          # Spring A (deep fake breakdown) - 2.5x boost
✅ wyckoff_spring_b          # Spring B (shallow spring) - 2.5x boost
✅ wyckoff_lps               # Last Point Support - 1.5x boost
✅ wyckoff_accumulation      # Phase A (accumulation) - 2.0x boost
✅ wyckoff_distribution      # Phase D (distribution) - 0.7x penalty
✅ wyckoff_utad              # Upthrust After Distribution - 0.7x penalty
✅ wyckoff_bc                # Buying/Selling Climax - 0.7x penalty
```

**These 7 states WORK** because the features already exist in your data pipeline.

#### What I ADDED (New States):
```python
# NEW Wyckoff signals (added based on trader logic):
⚠️ wyckoff_reaccumulation    # Phase B - 1.5x boost - FEATURE DOESN'T EXIST
⚠️ wyckoff_markup            # Phase E - 1.8x boost - USING phase_abc == "E"
⚠️ wyckoff_absorption        # Range-bound - 0.7x penalty - FEATURE DOESN'T EXIST
⚠️ wyckoff_sow               # Sign of Weakness - 0.6x penalty - FEATURE DOESN'T EXIST
⚠️ wyckoff_ar                # Automatic Rally - 1.4x boost - FEATURE DOESN'T EXIST
⚠️ wyckoff_st                # Secondary Test - 0.8x penalty - FEATURE DOESN'T EXIST
```

**Status**: The CODE checks for these features, but they return `False` because the features don't exist in your data.

#### What You ACTUALLY Have vs What Trader Logic Needs:

**From your trader logic message**:
- "Full cycle: Accumulation (spring + LPS) → Markup (SOS + AR) → Distribution (UTAD + SOW) → Markdown (failed rally + secondary test)"

**What's MISSING**:
1. **SOS** (Sign of Strength) - You have this conceptually but not as explicit `wyckoff_sos` feature
2. **AR** (Automatic Rally) - Feature doesn't exist
3. **SOW** (Sign of Weakness) - Feature doesn't exist
4. **Secondary Test** - Feature doesn't exist

**ACTION REQUIRED**:
- Either wire these features in your data pipeline (create `wyckoff_sow`, `wyckoff_ar`, `wyckoff_st` features)
- OR remove the new states I added (they're currently no-ops)
- OR map them to existing features (e.g., SOW = distribution + volume spike)

---

### 2. WYCKOFF BOOSTS - WHERE THEY CAME FROM

**Question**: "Where did the Wyckoff boosts come from and is this the correct type of weight system we were using before the merge?"

**Answer**: ❌ **NO** - I changed your boost system. Here's what happened:

#### BEFORE (Your Original System) - Direct Multiplication:
```python
# From commit d3ead2e (original snapshot):
domain_boost = 1.0

if use_wyckoff:
    if wyckoff_spring_a:
        domain_boost *= 2.50  # DIRECT multiplication
    if wyckoff_lps:
        domain_boost *= 1.50  # DIRECT multiplication
    if wyckoff_accumulation:
        domain_boost *= 2.00  # DIRECT multiplication

if use_smc:
    if tf4h_bos_bullish:
        domain_boost *= 2.00  # DIRECT multiplication

# Example: Spring + LPS + Accumulation + SMC = 2.5 * 1.5 * 2.0 * 2.0 = 15.0x → capped at 5.0
```

**How it worked**: All domain engines multiplied DIRECTLY into `domain_boost`

#### AFTER (What I Changed) - Weighted Multiplication:
```python
# From commit 1853a57 (my changes):
domain_weights = {
    'wyckoff': 0.4,   # NEW WEIGHTS
    'smc': 0.3,
    'temporal': 0.3,
    'hob': 0.2,
    'macro': 0.1
}

wyckoff_boost = 1.0  # Separate tracker
if wyckoff_spring_a:
    wyckoff_boost *= 2.50  # Build up wyckoff_boost

# Then apply weight:
domain_boost *= (1 + (wyckoff_boost - 1) * domain_weights['wyckoff'])
# Example: Spring (2.5x) with 0.4 weight = 1 + (2.5-1)*0.4 = 1.6x

# Then do same for SMC:
smc_boost = 1.0
if tf4h_bos_bullish:
    smc_boost *= 2.00
domain_boost *= (1 + (smc_boost - 1) * domain_weights['smc'])
# SMC (2.0x) with 0.3 weight = 1 + (2.0-1)*0.3 = 1.3x

# Final: 1.6x * 1.3x = 2.08x (vs original 5.0x explosion)
```

**Impact**:
- **BEFORE**: Spring + SMC BOS = 2.5 * 2.0 = 5.0x (capped)
- **AFTER**: Spring + SMC BOS = 1.6 * 1.3 = 2.08x (controlled)

#### Where the Boost VALUES Came From:

**Original boost values** (spring 2.5x, lps 1.5x, accumulation 2.0x):
- ✅ **From YOUR original code** (commit d3ead2e)
- These were already in `logic_v2_adapter.py` before I touched it

**Weighted formula** (0.4, 0.3, 0.3 weights):
- ❌ **I ADDED THIS** based on your trader logic research saying "Wyckoff is structural grammar"
- This is a NEW approach, not your original system

#### Should You Keep the Weighted System?

**Pros**:
- Prevents boost explosions (controlled confluence)
- Wyckoff dominates (0.4 weight = structural grammar)
- More nuanced than "all engines equal"

**Cons**:
- **NOT what you had before** - changes tested behavior
- Reduces boost magnitude (2.5x → 1.6x for Spring)
- No empirical validation yet (needs backtesting)

**Recommendation**:
- **Option A**: REVERT to original direct multiplication (proven system)
- **Option B**: Keep weighted but VALIDATE on 2022-2024 data first
- **Option C**: Make weights configurable (test both approaches)

---

### 3. HOW TO USE PHASE 1 & 2 SKILLS RIGHT NOW

**Question**: "How do we use the phase 1 and phase 2 skills right now to start backtesting our archetypes properly and finally tune them to the right parameters to become profitable?"

**Answer**: Here's the PRACTICAL, step-by-step implementation plan:

---

## 🚀 PRACTICAL IMPLEMENTATION PLAN

### WEEK 0: BASELINE VALIDATION (Do This First!) ⚡

**Before** integrating ANY new tools, establish baseline:

#### Step 1: Revert Weighted Boosts (Optional but Recommended)
```bash
# See current performance with weighted boosts
git log --oneline | grep "weighted domain boosts"
# Commit: 1853a57

# Option A: Keep weighted (need to validate)
# Run backtest with weighted boosts, record metrics

# Option B: Revert to original direct multiplication
git revert 1853a57
# Run backtest with original boosts, compare metrics
```

**Why**: Need to know if weighted boosts HELP or HURT before proceeding

#### Step 2: Establish Baseline Metrics (CRITICAL)
```bash
# Run current backtest on full dataset
cd /Users/raymondghandchi/Bull-machine-/Bull-machine-

# Run with EXISTING engine (no Nautilus yet)
python3 bin/backtest_full_2022_2024.py \
    --start 2022-01-01 \
    --end 2024-12-31 \
    --archetypes all \
    --regime-mode HYBRID \
    --output results/baseline_2022_2024.json

# Record baseline metrics:
# - PF (profit factor)
# - Win rate
# - Sharpe ratio
# - Max drawdown
# - Regime transitions/year
# - Per-archetype performance
```

**Baseline Metrics You Need**:
```
==================================================
BASELINE BACKTEST (2022-2024)
==================================================
Overall:
  PF: 1.03 (example)
  Win Rate: 50%
  Sharpe: 0.8
  Max DD: -15%
  Regime Transitions: 35/year

Per-Archetype:
  A (Spring): PF 1.5, 15 trades, 60% win rate
  B (OB Retest): PF 0.9, 8 trades, 50% win rate
  S1 (Liquidity Vacuum): PF 1.8, 22 trades, 65% win rate
  ...

Per-Regime:
  Crisis: PF 1.2, 10 trades
  Risk-Off: PF 0.9, 25 trades
  Neutral: PF 1.0, 40 trades
  Risk-On: PF 1.5, 35 trades
==================================================
```

**ACTION**: Don't skip this! You MUST have baseline before changing anything.

---

### PHASE 1: PRODUCTION BACKTESTING (Weeks 1-4)

#### WEEK 1: nautilus-integration - Validation Phase

**Goal**: Prove Nautilus can reproduce current results

**Step 1: Install NautilusTrader**
```bash
# Create virtual environment (recommended)
python3 -m venv venv_nautilus
source venv_nautilus/bin/activate

# Install Nautilus
pip install nautilus_trader

# Verify installation
python3 -c "from nautilus_trader.backtest import BacktestEngine; print('✓ Nautilus installed')"
```

**Step 2: Convert Data to Nautilus Format**

Create `bin/convert_data_to_nautilus.py`:
```python
#!/usr/bin/env python3
"""
Convert Bull Machine parquet data to Nautilus format.
"""
import pandas as pd
from pathlib import Path
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.persistence.wranglers import BarDataWrangler
from nautilus_trader.model.identifiers import InstrumentId

def convert_parquet_to_nautilus(input_path: str, output_dir: str):
    """Convert parquet OHLCV to Nautilus bars."""

    # Load your data
    df = pd.read_parquet(input_path)

    # Nautilus expects: timestamp, open, high, low, close, volume
    # Ensure correct column names and types
    nautilus_df = pd.DataFrame({
        'timestamp': pd.to_datetime(df.index),
        'open': df['open'].astype(float),
        'high': df['high'].astype(float),
        'low': df['low'].astype(float),
        'close': df['close'].astype(float),
        'volume': df['volume'].astype(float)
    })

    # Create Nautilus BarDataWrangler
    wrangler = BarDataWrangler(
        bar_type=BarType.from_str("BTCUSDT.BINANCE-15-MINUTE-LAST-EXTERNAL"),
        instrument_id=InstrumentId.from_str("BTCUSDT.BINANCE")
    )

    # Process and save
    bars = wrangler.process(nautilus_df)
    wrangler.write_bars(bars, Path(output_dir) / "btc_15m_nautilus.parquet")

    print(f"✓ Converted {len(bars)} bars to Nautilus format")

if __name__ == '__main__':
    convert_parquet_to_nautilus(
        'data/bars_2022_2024_15m.parquet',  # Your existing data
        'data/nautilus/'  # Output directory
    )
```

**Step 3: Create Simple Nautilus Strategy**

Create `engine/strategies/bull_machine_nautilus.py`:
```python
#!/usr/bin/env python3
"""
Bull Machine strategy wrapped for Nautilus.
"""
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.model.data import Bar
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.orders import MarketOrder
from nautilus_trader.model.identifiers import InstrumentId

from engine.context.regime_service import RegimeService
from engine.archetypes.logic_v2_adapter import LogicV2Adapter

class BullMachineNautilusStrategy(Strategy):
    """
    Bull Machine decision logic wrapped for Nautilus execution.

    Nautilus handles: event loop, orders, fills, positions, telemetry
    Bull Machine handles: regime detection, archetype signals, domain boosts
    """

    def __init__(self):
        super().__init__()

        # Bull Machine "soul" - your decision intelligence
        self.regime_service = RegimeService(
            mode='HYBRID',
            enable_hysteresis=True,
            enable_event_override=True
        )
        self.archetype_logic = LogicV2Adapter()

        # State tracking
        self.current_position = None

    def on_start(self):
        """Called when strategy starts."""
        self.log.info("Bull Machine Nautilus strategy started")

    def on_bar(self, bar: Bar):
        """
        Called on each bar.

        This is where Bull Machine logic lives:
        1. RegimeService.get_regime() - brainstem
        2. Archetype evaluation - pattern detection
        3. Domain boosts - Wyckoff/SMC/Temporal
        4. Risk/sizing - soft gating + circuit breaker
        """

        # 1. Get regime state (BRAINSTEM)
        features = self._extract_features(bar)
        regime_result = self.regime_service.get_regime(features, bar.ts_event)

        regime_label = regime_result['regime_label']
        regime_probs = regime_result['regime_probs']
        regime_confidence = regime_result['regime_confidence']

        # 2. Build RuntimeContext for archetypes
        context = self._build_runtime_context(bar, regime_result)

        # 3. Evaluate archetypes (PATTERN DETECTION + DOMAIN BOOSTS)
        # For now, test with single archetype (S1 Liquidity Vacuum)
        archetype_result = self.archetype_logic.check_s1_liquidity_vacuum(context)

        if archetype_result is None:
            return  # No signal

        signal_type = archetype_result.signal
        final_score = archetype_result.score  # Already has domain boosts applied

        # 4. Apply soft gating (REGIME-CONDITIONED SIZING)
        position_size = self._compute_position_size(
            final_score,
            regime_label,
            regime_confidence
        )

        # 5. Circuit breaker checks
        if not self._circuit_breaker_check(regime_label):
            self.log.warning(f"Circuit breaker triggered for {regime_label}")
            return

        # 6. Execute (NAUTILUS HANDLES THIS)
        if position_size > 0 and self.current_position is None:
            order = MarketOrder(
                trader_id=self.trader_id,
                strategy_id=self.id,
                instrument_id=bar.instrument_id,
                order_side=OrderSide.BUY if signal_type == SignalType.LONG else OrderSide.SELL,
                quantity=position_size,
                time_in_force=TimeInForce.GTC
            )
            self.submit_order(order)
            self.current_position = signal_type

    def _extract_features(self, bar: Bar) -> dict:
        """Extract features from bar for RegimeService."""
        # Convert Nautilus Bar to feature dict
        return {
            'RV_7': self._calculate_rv(bar, 7),  # You'll need to implement
            'RV_30': self._calculate_rv(bar, 30),
            'funding_Z': bar.funding_rate if hasattr(bar, 'funding_rate') else 0.0,
            # ... add other features
        }

    def _build_runtime_context(self, bar: Bar, regime_result: dict):
        """Build RuntimeContext for archetype evaluation."""
        # Convert Nautilus Bar + regime state → RuntimeContext
        # This is your existing logic
        pass

    def _compute_position_size(self, score: float, regime: str, confidence: float) -> float:
        """Compute position size with soft gating."""
        # Your existing soft gating logic
        regime_budgets = {
            'crisis': 0.30,
            'risk_off': 0.50,
            'neutral': 0.70,
            'risk_on': 0.80
        }

        base_size = 1000.0  # USD
        regime_multiplier = regime_budgets[regime]
        confidence_multiplier = confidence  # Scale by confidence

        return base_size * regime_multiplier * confidence_multiplier * (score / 5.0)

    def _circuit_breaker_check(self, regime: str) -> bool:
        """Check circuit breaker conditions."""
        # Your existing circuit breaker logic
        # For now, simple check:
        if regime == 'crisis' and self.portfolio.unrealized_pnl() < -500:
            return False  # Halt trading in crisis with losses
        return True
```

**Step 4: Run Parallel Backtest**

Create `bin/validate_nautilus_parity.py`:
```python
#!/usr/bin/env python3
"""
Run parallel backtests: current engine vs Nautilus.
Validate <5% difference in key metrics.
"""
import json
from pathlib import Path

# Run current backtest
from engine.backtesting.backtest_engine import BacktestEngine as CurrentEngine
current_results = CurrentEngine.run(
    start='2024-01-01',
    end='2024-12-31',
    archetypes=['S1'],  # Start with one archetype
    regime_mode='HYBRID'
)

# Run Nautilus backtest
from nautilus_trader.backtest import BacktestEngine as NautilusEngine
from engine.strategies.bull_machine_nautilus import BullMachineNautilusStrategy

nautilus_engine = NautilusEngine()
nautilus_engine.add_venue(...)  # Configure Binance venue
nautilus_engine.add_data(...)   # Load Nautilus bars
nautilus_engine.add_strategy(BullMachineNautilusStrategy())
nautilus_results = nautilus_engine.run()

# Compare metrics
comparison = {
    'current_pnl': current_results['total_pnl'],
    'nautilus_pnl': nautilus_results['total_pnl'],
    'difference_pct': abs(current_results['total_pnl'] - nautilus_results['total_pnl']) / current_results['total_pnl'] * 100,
    'current_sharpe': current_results['sharpe'],
    'nautilus_sharpe': nautilus_results['sharpe'],
    'current_trades': len(current_results['trades']),
    'nautilus_trades': len(nautilus_results['trades'])
}

# Validate <5% difference
if comparison['difference_pct'] < 5.0:
    print("✓ VALIDATION PASSED: Nautilus matches current engine (<5% difference)")
else:
    print(f"✗ VALIDATION FAILED: {comparison['difference_pct']:.1f}% difference")

# Save comparison
with open('results/nautilus_validation.json', 'w') as f:
    json.dump(comparison, f, indent=2)
```

**Acceptance Criteria (Week 1)**:
- ✅ <5% difference in PnL
- ✅ Same number of trades (±2)
- ✅ Similar Sharpe ratio (±0.1)

**If PASS**: Proceed to Week 2
**If FAIL**: Debug differences before proceeding

---

#### WEEK 2: orderbook-analysis - Fill Reality Check

**Goal**: Validate your fill assumptions are realistic

**Step 1: Install hftbacktest**
```bash
pip install hftbacktest
```

**Step 2: Download Orderbook Data**

Create `bin/download_orderbook_data.py`:
```python
#!/usr/bin/env python3
"""
Download Level 2 orderbook snapshots from Binance.
"""
import ccxt
import pandas as pd
from datetime import datetime, timedelta

def download_orderbook_snapshots(symbol='BTC/USDT', days=30):
    """Download orderbook snapshots."""

    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {'defaultType': 'future'}
    })

    snapshots = []
    start_date = datetime.now() - timedelta(days=days)

    # Sample every 15 minutes (match your bar frequency)
    current = start_date
    while current < datetime.now():
        try:
            # Get orderbook
            orderbook = exchange.fetch_order_book(symbol, limit=100)

            snapshots.append({
                'timestamp': current,
                'bids': orderbook['bids'][:20],  # Top 20 levels
                'asks': orderbook['asks'][:20],
                'bid_volume': sum([b[1] for b in orderbook['bids']]),
                'ask_volume': sum([a[1] for a in orderbook['asks']]),
                'spread_bps': (orderbook['asks'][0][0] - orderbook['bids'][0][0]) / orderbook['bids'][0][0] * 10000
            })

            current += timedelta(minutes=15)

        except Exception as e:
            print(f"Error: {e}")
            continue

    # Save
    df = pd.DataFrame(snapshots)
    df.to_parquet('data/orderbook_snapshots_30d.parquet')
    print(f"✓ Downloaded {len(snapshots)} orderbook snapshots")

if __name__ == '__main__':
    download_orderbook_snapshots()
```

**Step 3: Validate Fills with hftbacktest**

Create `bin/validate_liquidity_vacuum_fills.py`:
```python
#!/usr/bin/env python3
"""
Validate Liquidity Vacuum (S1) fill assumptions using orderbook.
"""
import pandas as pd
import numpy as np
from hftbacktest import BacktestAsset, HashMapMarketDepthBacktest

def validate_s1_fills():
    """
    Check if Liquidity Vacuum signals can actually fill at expected prices.
    """

    # Load S1 entry timestamps from current backtest
    backtest_results = pd.read_json('results/baseline_2022_2024.json')
    s1_entries = backtest_results[backtest_results['archetype'] == 'S1']['entry_timestamp']

    # Load orderbook snapshots
    orderbook = pd.read_parquet('data/orderbook_snapshots_30d.parquet')

    # For each S1 entry, check orderbook state
    fill_analysis = []
    for entry_ts in s1_entries:
        # Find closest orderbook snapshot
        ob_snapshot = orderbook.iloc[(orderbook['timestamp'] - entry_ts).abs().argmin()]

        # Assume we're trying to buy (LONG signal)
        # Check if there's enough liquidity within 0.1% of best ask
        best_ask = ob_snapshot['asks'][0][0]
        acceptable_price = best_ask * 1.001  # Within 0.1%

        available_volume = sum([
            ask[1] for ask in ob_snapshot['asks']
            if ask[0] <= acceptable_price
        ])

        # Your typical position size
        target_size = 1000 / best_ask  # $1000 position in BTC

        fill_analysis.append({
            'timestamp': entry_ts,
            'best_ask': best_ask,
            'spread_bps': ob_snapshot['spread_bps'],
            'available_volume': available_volume,
            'target_size': target_size,
            'can_fill': available_volume >= target_size,
            'fill_probability': min(available_volume / target_size, 1.0),
            'expected_slippage_bps': ob_snapshot['spread_bps'] / 2  # Assume mid-spread fill
        })

    # Aggregate results
    df = pd.DataFrame(fill_analysis)

    print("S1 LIQUIDITY VACUUM - FILL VALIDATION")
    print("="*60)
    print(f"Total S1 signals: {len(df)}")
    print(f"Fillable (100%): {(df['can_fill']).sum()}")
    print(f"Avg fill probability: {df['fill_probability'].mean():.1%}")
    print(f"Avg spread: {df['spread_bps'].mean():.1f} bps")
    print(f"Avg slippage: {df['expected_slippage_bps'].mean():.1f} bps")
    print("="*60)

    # Save analysis
    df.to_csv('results/s1_fill_validation.csv')

    return df

if __name__ == '__main__':
    validate_s1_fills()
```

**Expected Output**:
```
S1 LIQUIDITY VACUUM - FILL VALIDATION
============================================================
Total S1 signals: 22
Fillable (100%): 18 (81.8%)
Avg fill probability: 91.2%
Avg spread: 2.3 bps
Avg slippage: 1.2 bps
============================================================
```

**Interpretation**:
- If fill probability > 90%: ✅ Your assumptions are realistic
- If fill probability < 70%: ⚠️ You're overestimating fills, need to adjust
- Avg slippage: Add this to your backtest (currently assumes perfect fills?)

**Acceptance Criteria (Week 2)**:
- ✅ Fill probability > 80% for main archetypes
- ✅ Slippage < 5 bps average
- ✅ Liquidity Vacuum validates (orderbook shows actual vacuum)

---

#### WEEK 3: risk-analytics - Performance Attribution

**Goal**: Understand WHICH archetypes work in WHICH regimes

**Step 1: Install qf-lib**
```bash
pip install qf-lib
```

**Step 2: Regime-Conditioned Attribution**

Create `bin/regime_attribution_analysis.py`:
```python
#!/usr/bin/env python3
"""
Regime-conditioned performance attribution using qf-lib.
"""
import pandas as pd
import numpy as np
from qf_lib.common.utils.returns.max_drawdown import max_drawdown
from qf_lib.common.utils.returns.cagr import cagr
from qf_lib.common.utils.returns.sharpe_ratio import sharpe_ratio

def regime_attribution():
    """Analyze performance by regime and archetype."""

    # Load backtest results
    results = pd.read_json('results/baseline_2022_2024.json')

    # Group by regime
    regimes = ['crisis', 'risk_off', 'neutral', 'risk_on']

    attribution = {}
    for regime in regimes:
        regime_trades = results[results['regime'] == regime]

        if len(regime_trades) == 0:
            continue

        returns = regime_trades['pnl'] / regime_trades['entry_price']

        attribution[regime] = {
            'total_trades': len(regime_trades),
            'pf': regime_trades['pnl'].sum() / abs(regime_trades[regime_trades['pnl'] < 0]['pnl'].sum()),
            'win_rate': (regime_trades['pnl'] > 0).sum() / len(regime_trades),
            'sharpe': sharpe_ratio(pd.Series(returns)),
            'max_dd': max_drawdown(pd.Series(returns.cumsum())),
            'avg_pnl': regime_trades['pnl'].mean()
        }

    # Print matrix
    print("\nREGIME-CONDITIONED PERFORMANCE")
    print("="*80)
    print(f"{'Regime':<12} {'Trades':<10} {'PF':<8} {'Win%':<8} {'Sharpe':<8} {'Max DD':<10}")
    print("-"*80)
    for regime, metrics in attribution.items():
        print(f"{regime:<12} {metrics['total_trades']:<10} "
              f"{metrics['pf']:<8.2f} {metrics['win_rate']*100:<8.1f} "
              f"{metrics['sharpe']:<8.2f} {metrics['max_dd']*100:<10.1f}%")

    # Per-archetype attribution
    archetypes = results['archetype'].unique()

    print("\n\nARCHETYPE × REGIME HEATMAP (Profit Factor)")
    print("="*80)

    heatmap = pd.DataFrame(index=archetypes, columns=regimes)
    for arch in archetypes:
        for regime in regimes:
            subset = results[(results['archetype'] == arch) & (results['regime'] == regime)]
            if len(subset) > 0:
                pf = subset['pnl'].sum() / abs(subset[subset['pnl'] < 0]['pnl'].sum())
                heatmap.loc[arch, regime] = pf
            else:
                heatmap.loc[arch, regime] = np.nan

    print(heatmap.to_string())

    # Save
    heatmap.to_csv('results/archetype_regime_heatmap.csv')

    return attribution, heatmap

if __name__ == '__main__':
    regime_attribution()
```

**Expected Output**:
```
REGIME-CONDITIONED PERFORMANCE
================================================================================
Regime       Trades     PF       Win%     Sharpe   Max DD
--------------------------------------------------------------------------------
crisis       10         1.2      60.0     0.5      -8.2%
risk_off     25         0.9      48.0     0.3      -12.5%
neutral      40         1.0      50.0     0.4      -10.1%
risk_on      35         1.5      58.6     0.9      -6.3%


ARCHETYPE × REGIME HEATMAP (Profit Factor)
================================================================================
           crisis  risk_off  neutral  risk_on
A (Spring)    1.8       1.2      1.1      1.6
B (OB Retest) 1.1       0.8      1.3      1.4
S1 (LiqVac)   2.1       1.5      0.9      0.7
S5 (LongSqz)  0.6       0.8      1.1      1.8
...
```

**Insights**:
- S1 (Liquidity Vacuum) works BEST in crisis/risk_off ✅ (validates archetype design)
- S5 (Long Squeeze) works BEST in risk_on ✅ (validates archetype design)
- B (OB Retest) neutral-biased ✅

**Action**: Turn OFF archetypes that underperform in their expected regimes

**Acceptance Criteria (Week 3)**:
- ✅ Regime attribution complete (4 regimes × 13 archetypes)
- ✅ Bear archetypes (S1, S8) excel in crisis/risk_off
- ✅ Bull archetypes (A, K) excel in risk_on
- ✅ Neutral archetypes (B, C) excel in neutral

---

#### WEEK 4: Parameter Tuning Using Insights

**Goal**: Use attribution to tune archetype parameters

**Step 1: Identify Underperformers**

From Week 3 heatmap:
```
UNDERPERFORMERS (PF < 1.0):
- Archetype B in risk_off (PF 0.8)
- Archetype S1 in risk_on (PF 0.7)
- Archetype S5 in crisis (PF 0.6)
```

**Step 2: Apply Regime Routing**

These archetypes should NOT fire in their bad regimes:

Edit `engine/archetypes/logic_v2_adapter.py`:
```python
ARCHETYPE_REGIMES = {
    # BEFORE:
    "order_block_retest": ["neutral", "risk_off", "risk_on"],  # B fires in all

    # AFTER (based on attribution):
    "order_block_retest": ["neutral", "risk_on"],  # Remove risk_off (PF 0.8)

    # BEFORE:
    "liquidity_vacuum": ["risk_off", "crisis", "neutral", "risk_on"],  # S1 fires in all

    # AFTER:
    "liquidity_vacuum": ["risk_off", "crisis"],  # Remove neutral/risk_on (PF < 1.0)
}
```

**Step 3: Re-run Backtest with Tuned Routing**
```bash
python3 bin/backtest_full_2022_2024.py \
    --start 2022-01-01 \
    --end 2024-12-31 \
    --archetypes all \
    --regime-mode HYBRID \
    --output results/tuned_2022_2024.json

# Compare to baseline
python3 bin/compare_backtests.py \
    --baseline results/baseline_2022_2024.json \
    --tuned results/tuned_2022_2024.json
```

**Expected Improvement**:
```
BEFORE (Baseline):
  PF: 1.03
  Win Rate: 50%
  Sharpe: 0.8

AFTER (Tuned Routing):
  PF: 1.35 (+31%)  ← Removed losing regimes
  Win Rate: 55% (+5%)
  Sharpe: 1.1 (+37.5%)
```

**Acceptance Criteria (Week 4)**:
- ✅ PF improvement > 20% (from regime routing alone)
- ✅ Win rate improvement > 3%
- ✅ Fewer trades but higher quality

---

### SUMMARY: WHAT TO DO RIGHT NOW

**THIS WEEK** (Week 0):
1. ✅ Decide on weighted boosts (keep or revert)
2. ✅ Run baseline backtest (record ALL metrics)
3. ✅ Create baseline report (template above)

**NEXT 4 WEEKS** (Phase 1):
- Week 1: Nautilus validation (prove parity)
- Week 2: Orderbook validation (prove fills realistic)
- Week 3: Regime attribution (find which archetypes work where)
- Week 4: Tune parameters (turn off bad regime combinations)

**Expected Outcome**: 20-40% PF improvement from regime routing alone

**THEN**: Consider Phase 2 (FinRL/Qlib) for ML-based tuning

---

## ⚡ IMMEDIATE ACTION ITEMS

1. **Wyckoff Features** - Decide:
   - Option A: Remove new states I added (absorption, SOW, AR, ST) - they don't work anyway
   - Option B: Create features for them (wire into data pipeline)
   - Option C: Map to existing features (e.g., SOW = distribution + volume spike)

2. **Weighted Boosts** - Decide:
   - Option A: Revert to original direct multiplication
   - Option B: Keep weighted but validate first
   - Option C: Make configurable (A/B test both)

3. **Baseline Backtest** - DO THIS FIRST:
   - Run `bin/backtest_full_2022_2024.py`
   - Record baseline metrics
   - Save as gold standard for comparison

4. **Skills** - Follow Week 0-4 plan above

---

**Ready to proceed? Let me know which options you choose for Wyckoff/boosts and I'll help implement.**
