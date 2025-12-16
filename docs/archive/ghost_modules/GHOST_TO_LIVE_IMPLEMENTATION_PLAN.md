# GHOST TO LIVE: IMPLEMENTATION PLAN
**Systematic Feature Completion Roadmap**

Based on: FEATURE_STORE_REALITY_CHECK.md
Date: 2025-12-11

---

## EXECUTIVE SUMMARY

**Reality Check Results:**
- ✅ 34/49 features (69.4%) already LIVE with real data
- ⚠️ 3/49 features (6.1%) exist but broken (constant values)
- ❌ 12/49 features (24.5%) truly missing

**Work Required:** Fix 3 broken + Implement 12 missing = **15 features total**

---

## PHASE 1: FIX BROKEN FEATURES (CRITICAL)
**Timeline: 1-2 days**

These features exist but are stuck at constant values. They block usage of otherwise working systems.

### 1.1 wyckoff_spring_b (CONSTANT - All False)

**Current State:**
- Exists in feature store
- Always returns False
- Never triggers

**Investigation Required:**
```python
# Check detection logic
grep -r "spring_b" engine/wyckoff/

# Expected location
engine/wyckoff/wyckoff_engine.py
engine/wyckoff/events.py
```

**Likely Issues:**
- Threshold too strict
- Condition logic error
- Missing dependency feature

**Fix Strategy:**
1. Review Wyckoff Spring B definition (second spring after initial reversal)
2. Check if thresholds are realistic for crypto volatility
3. Validate against known market examples (2022 bear bottom, 2024 corrections)
4. Adjust detection sensitivity

**Acceptance Criteria:**
- Feature triggers True at least 0.5% of time (>130 instances in 26K rows)
- Triggers align with visual chart analysis of spring patterns
- No false positives during strong trends

---

### 1.2 wyckoff_pti_confluence (CONSTANT - All False)

**Current State:**
- PTI score exists and works (3,955 unique values)
- Confluence flag never activates

**Investigation Required:**
```python
# Check PTI confluence logic
grep -r "pti_confluence" engine/

# Expected calculation
# Confluence = multiple PTI signals align temporally
```

**Likely Issues:**
- Temporal alignment window too narrow
- Requires multiple simultaneous PTI events (too strict)
- Missing cross-timeframe integration

**Fix Strategy:**
1. Review confluence definition (what constitutes "multiple" signals?)
2. Check temporal window for alignment (currently likely 1 bar = too strict)
3. Expand window to 3-5 bars
4. Consider cross-timeframe PTI alignment (1H + 4H)

**Acceptance Criteria:**
- Activates during known trap clusters (funding divergence + liquidity vacuum)
- Triggers 1-3% of time (high signal, not noise)
- Correlates with archetype win rates

---

### 1.3 temporal_confluence (CONSTANT - All False)

**Current State:**
- Individual temporal features work:
  - `temporal_support_cluster`: ✅
  - `temporal_resistance_cluster`: ✅
  - `fib_time_cluster`: ✅
- Confluence flag stuck at False

**Investigation Required:**
```python
# Check temporal confluence logic
grep -r "temporal_confluence" engine/temporal/

# Expected location
engine/temporal/temporal_fusion.py
```

**Likely Issues:**
- Requires ALL clusters to align (too strict)
- Missing time-based alignment logic
- Not using existing cluster features

**Fix Strategy:**
1. Define confluence as ANY 2+ clusters within N bars
2. Implement sliding window check (5-10 bars)
3. Weight by cluster strength
4. Add multi-timeframe confluence (1H + 4H support/resistance)

**Acceptance Criteria:**
- Activates 5-10% of time (meaningful but not rare)
- Aligns with visual S/R cluster zones
- Provides predictive value for reversals

---

## PHASE 2: IMPLEMENT MISSING WYCKOFF FEATURES
**Timeline: 3-5 days**

### 2.1 Wyckoff Phase Classification (5 features)

**Missing Features:**
```python
wyckoff_phase          # Main phase: accumulation, distribution, markup, markdown
wyckoff_accumulation   # Boolean: in accumulation phase
wyckoff_distribution   # Boolean: in distribution phase
wyckoff_markup         # Boolean: in markup phase
wyckoff_markdown       # Boolean: in markdown phase
```

**Implementation Location:**
```
engine/wyckoff/wyckoff_engine.py
  → Add classify_phase() method
```

**Logic Design:**

```python
def classify_phase(events: dict, price_trend: str, volume_trend: str) -> str:
    """
    Classify current Wyckoff phase based on event sequence

    Accumulation Phase:
    - PS or SC detected
    - AR following SC
    - ST (secondary tests) present
    - Volume declining
    - Price range-bound

    Distribution Phase:
    - BC (buying climax) detected
    - UTAD present
    - LPSY (last point of supply)
    - Volume declining after BC
    - Price topping

    Markup Phase:
    - SOS detected
    - LPS holds
    - Price in uptrend
    - Higher highs, higher lows

    Markdown Phase:
    - SOW detected
    - Failed rallies
    - Price in downtrend
    - Lower highs, lower lows
    """

    # Implementation:
    # 1. Check recent events (last 20-50 bars)
    # 2. Analyze price structure
    # 3. Validate with volume behavior
    # 4. Return phase label
```

**Data Required:**
- Existing: `wyckoff_ps`, `wyckoff_sc`, `wyckoff_ar`, `wyckoff_st`, `wyckoff_bc`, `wyckoff_utad`, `wyckoff_lpsy`, `wyckoff_sos`, `wyckoff_sow`
- Need: Price trend classification (can derive from existing price data)
- Need: Volume trend (can derive from existing volume data)

**Acceptance Criteria:**
- Phase labels change 10-20 times per year (stable phases)
- Accumulation/Distribution phases precede major moves
- Markup/Markdown align with bull/bear trends
- Phase transitions are logical (Accumulation → Markup, Distribution → Markdown)

---

### 2.2 Wyckoff PTI Trap Type Classification

**Missing Feature:**
```python
wyckoff_pti_trap_type  # "liquidity_vacuum" | "failed_rally" | "funding_divergence" | None
```

**Implementation Location:**
```
engine/wyckoff/events.py
  → Add classify_pti_trap() method
```

**Logic Design:**

```python
def classify_pti_trap(
    pti_score: float,
    funding_div: bool,
    liquidity_sweep: bool,
    failed_rally: bool
) -> Optional[str]:
    """
    Classify the type of PTI trap based on market conditions

    Liquidity Vacuum:
    - High PTI score
    - Liquidity sweep detected
    - Rapid price reversal

    Failed Rally:
    - Medium PTI score
    - Rally exhaustion signs
    - Volume divergence

    Funding Divergence:
    - PTI score > 0.3
    - Funding rate diverges from price
    - Sentiment extreme
    """
```

**Data Required:**
- Existing: `wyckoff_pti_score`
- Existing: `smc_liquidity_sweep`
- Existing: `funding_rate`, `funding_reversal`
- Need: Failed rally detection logic

**Acceptance Criteria:**
- Each trap type represents 20-40% of total PTI events
- Trap types have distinct win rate profiles
- Classifications stable (don't flip every bar)

---

### 2.3 Wyckoff Event Confidence & Strength Scores

**Missing Features:**
```python
wyckoff_confidence  # 0.0-1.0: How confident is event detection
wyckoff_strength    # 0.0-1.0: How strong is the event signal
```

**Implementation Location:**
```
engine/wyckoff/wyckoff_engine.py
  → Add calculate_event_confidence() method
  → Add calculate_event_strength() method
```

**Logic Design:**

```python
def calculate_event_confidence(
    event_type: str,
    volume_confirmation: float,
    price_confirmation: float,
    context_alignment: float
) -> float:
    """
    Calculate confidence in event detection

    Factors:
    - Volume confirmation (0-1): Is volume pattern correct?
    - Price confirmation (0-1): Does price structure match?
    - Context alignment (0-1): Does macro context support?

    Returns: 0.0-1.0 confidence score
    """
    return (volume_confirmation * 0.4 +
            price_confirmation * 0.4 +
            context_alignment * 0.2)

def calculate_event_strength(
    event_type: str,
    magnitude: float,
    follow_through: float,
    historical_significance: float
) -> float:
    """
    Calculate strength of detected event

    Factors:
    - Magnitude (0-1): Size of price/volume move
    - Follow-through (0-1): Sustained action after event
    - Historical significance (0-1): Event relative to recent history

    Returns: 0.0-1.0 strength score
    """
    return (magnitude * 0.5 +
            follow_through * 0.3 +
            historical_significance * 0.2)
```

**Acceptance Criteria:**
- Scores distributed across full 0-1 range
- High confidence events have better win rates
- High strength events predict larger moves
- Scores stable (not noisy)

---

## PHASE 3: IMPLEMENT MISSING SMC FEATURES
**Timeline: 2-3 days**

### 3.1 Fair Value Gap (FVG) Detection

**Missing Features:**
```python
smc_fvg_bear  # Bearish fair value gap detected
smc_fvg_bull  # Bullish fair value gap detected
```

**Implementation Location:**
```
engine/features/smc_features.py
  → Add detect_fvg() method
```

**Logic Design:**

```python
def detect_fvg(high: pd.Series, low: pd.Series, close: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    Detect Fair Value Gaps (FVG)

    Bullish FVG:
    - Bar 1 Low > Bar 3 High
    - Gap between bar 1 low and bar 3 high = bullish FVG
    - Represents unfilled buying imbalance

    Bearish FVG:
    - Bar 1 High < Bar 3 Low
    - Gap between bar 1 high and bar 3 low = bearish FVG
    - Represents unfilled selling imbalance

    Returns:
    - bullish_fvg: Boolean series
    - bearish_fvg: Boolean series
    """

    # 3-bar pattern
    bullish_fvg = (low.shift(2) > high)  # Bar -2 low > current bar high
    bearish_fvg = (high.shift(2) < low)  # Bar -2 high < current bar low

    return bullish_fvg, bearish_fvg
```

**Data Required:**
- Existing: `high`, `low`, `close` (in feature store)

**Acceptance Criteria:**
- FVGs detected 5-15% of time
- FVG zones act as support/resistance when price returns
- Bullish FVGs in uptrends, bearish FVGs in downtrends
- Integration with existing SMC framework

---

## PHASE 4: IMPLEMENT MISSING HOB FEATURES
**Timeline: 1-2 days**

### 4.1 HOB Zone Quality & Strength Scores

**Missing Features:**
```python
hob_strength  # Zone strength based on order book depth
hob_quality   # Zone quality based on historical respect
```

**Implementation Location:**
```
engine/features/hob_features.py
  → Add calculate_zone_strength() method
  → Add calculate_zone_quality() method
```

**Logic Design:**

```python
def calculate_zone_strength(
    zone_price: float,
    order_book_depth: pd.Series,
    volume_profile: pd.Series
) -> float:
    """
    Calculate strength of HOB zone

    Factors:
    - Order book depth at zone price
    - Volume traded at zone historically
    - Distance from current price

    Returns: 0.0-1.0 strength score
    """
    # Strength = depth × volume × recency

def calculate_zone_quality(
    zone_price: float,
    historical_touches: int,
    respect_rate: float,
    zone_age: int
) -> float:
    """
    Calculate quality of HOB zone

    Factors:
    - Historical touch count (more = better)
    - Respect rate (% of times held as S/R)
    - Zone age (fresher zones = higher quality)

    Returns: 0.0-1.0 quality score
    """
    # Quality = touches × respect_rate × (1 / age)
```

**Data Required:**
- Existing: `hob_demand_zone`, `hob_supply_zone`, `hob_imbalance`
- Need: Historical zone touch tracking
- Need: S/R respect calculation

**Acceptance Criteria:**
- High strength zones predict reversals
- High quality zones used by archetypes
- Scores distributed across range (not clustered)

---

## IMPLEMENTATION PRIORITIES

### Critical Path (Blocks Production)
1. ⚠️ Fix `wyckoff_spring_b` - **Day 1**
2. ⚠️ Fix `wyckoff_pti_confluence` - **Day 1-2**
3. ⚠️ Fix `temporal_confluence` - **Day 2**

### High Value (Expand Capabilities)
4. 📊 Wyckoff Phase Classification (5 features) - **Day 3-5**
5. 📊 SMC FVG Detection (2 features) - **Day 6-7**
6. 📊 Wyckoff PTI Trap Type - **Day 7-8**

### Quality Enhancement (Nice to Have)
7. 📈 Wyckoff Confidence/Strength (2 features) - **Day 9-10**
8. 📈 HOB Quality/Strength (2 features) - **Day 10-11**

---

## TESTING STRATEGY

### Unit Tests
- Each feature has isolated test
- Known market patterns used as fixtures
- Edge cases validated

### Integration Tests
- Features work together (e.g., phase + events)
- Multi-timeframe consistency
- No conflicts with existing features

### Historical Validation
- Backtest on 2022-2024 data
- Visual spot-checks on charts
- Statistical distribution analysis

### Production Validation
- Deploy to test environment first
- Monitor for 24-48 hours
- Compare with previous feature store version

---

## SUCCESS METRICS

**Feature Completion:**
- [ ] 0 constant features (all 3 fixed)
- [ ] 0 missing features (all 12 implemented)
- [ ] 49/49 features live (100% coverage)

**Feature Quality:**
- [ ] All features have >2 unique values
- [ ] Boolean features trigger 0.1-20% of time
- [ ] Continuous features use >50% of their range
- [ ] No correlation >0.95 between features

**System Impact:**
- [ ] Archetype win rates improve 2-5%
- [ ] Signal quality scores increase
- [ ] No degradation in existing features
- [ ] Feature engineering pipeline <10 seconds

---

## ROLLOUT PLAN

### Week 1: Fix Broken (Phase 1)
- Days 1-2: Fix 3 broken constant features
- Day 3: Validation and testing

### Week 2: Core Implementation (Phase 2 + 3)
- Days 4-8: Implement Wyckoff + SMC features (10 features)
- Days 9-10: Integration testing

### Week 3: Enhancement + QA (Phase 4)
- Days 11-12: Implement HOB features (2 features)
- Days 13-15: Full system validation
- Deploy to production

---

## RESOURCES REQUIRED

**Code Files to Modify:**
```
engine/wyckoff/wyckoff_engine.py      # Phase classification, confidence/strength
engine/wyckoff/events.py              # PTI trap type, event enhancements
engine/features/smc_features.py       # FVG detection
engine/features/hob_features.py       # Quality/strength scores
engine/temporal/temporal_fusion.py    # Confluence logic fix
```

**Testing Files to Create:**
```
tests/unit/wyckoff/test_phase_classification.py
tests/unit/wyckoff/test_event_scoring.py
tests/unit/smc/test_fvg_detection.py
tests/unit/hob/test_zone_scoring.py
tests/integration/test_feature_completeness.py
```

**Documentation to Update:**
```
docs/FEATURE_STORE_SCHEMA_v2.md      # Add new feature specs
docs/WYCKOFF_INTEGRATION_COMPLETE.md  # Update status
FEATURE_STORE_REALITY_CHECK.md        # Final validation report
```

---

## NEXT STEPS

**Immediate (Today):**
1. Review this plan with stakeholders
2. Set up feature branch: `feature/ghost-to-live-completion`
3. Begin Phase 1: Fix `wyckoff_spring_b`

**Tomorrow:**
4. Complete Phase 1 (all 3 broken features)
5. Create test fixtures for Phase 2

**This Week:**
6. Implement Wyckoff phase classification
7. Daily validation of new features

---

## CONCLUSION

**This is NOT a massive rebuild.**

We have 69.4% of features already working. We need to:
- Fix 3 broken features (logic adjustments)
- Add 12 new features (clean implementations)

Total work: **~2 weeks for 1 developer**

**Status: READY TO EXECUTE**

---

*For detailed verification data, see: FEATURE_STORE_REALITY_CHECK.md*
*For current system state, see: git status on branch feature/ghost-modules-to-live-v2*
