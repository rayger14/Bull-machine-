# THE ONE STRATEGY — Bull Machine's surviving synthesis (WI's M2 mechanized)

**Status:** STUDY ONLY. Nothing shipped. Branch `study/one-strategy` off `study/trend-continuation`.
**Date:** 2026-08-11. **Author context:** synthesis of the wyckoff campaign (audit add.44–53).
**Verdict on fresh un-mined crypto:** PORTABILITY CONFIRMED (basket PF ~2.5, above-EMA200
concentration, bear-leak reproduced as a minority drag). See Part 2.

This is the one strategy that survived months of adversarial testing. It is **not** re-derived
here — it is the settled residue of everything that failed and the one thing that kept surfacing.

---

## 0. What died, so this could live (settled verdicts — do NOT re-attempt)

- **Dip-buying / springs at discount** — DEAD cross-asset (add.45/47). The M1 RTZ-spring is a
  falling-knife: it dies in every markdown on every asset class (BTC OOS-B 5/5 stops; SPX bears
  0% WR; Gold −$1,596). Intrinsic, not BTC-specific.
- **Fusion scoring** — negative predictive power (Lesson #54). Never a filter.
- **Filters-as-edge** — 0/10 lifetime. Dedup reshuffles blocked bars; system-PnL-up +
  target-down is a false win.
- **Fine Wyckoff phase detection** — un-buyable, un-trainable (label-starved), un-LLM-able, and
  genuinely absent at crypto tops (add.49). Desks don't use it; they use coarse causal regime.
- **Speeding up the cadence** — the 1H/LTF-trigger expansion was REJECTED (add.52–53): the add.52
  "77% win" was tightest-slice luck (n=5 under strict closed-bar P0), dying to the EMA-200 control
  at tradeable n. **Do not re-attempt any faster-timeframe variant.** Rarity is the signal's
  nature; breadth (more assets), not speed, is the fix.

## 1. What survived — THE DOOR (WI's Model-2 continuation, mechanized)

WI's actual edge is **not** buying the spring; it is buying the **back-up that holds after the
jump across the creek** — trend continuation. Formalized price-only (add.48), it validated on
4 assets with IDENTICAL params including uncorrelated Gold (PF 2.56/3.53/2.82/3.51 on
BTC/SPX/NDX/GOLD). It self-regime-filters (needs an up-break → cannot fire deep in a bear) and
sits 92–100% above the 200-EMA. Its one defect is a bear-market-**rally** breakout leak
(~7–14% of gross, worst on Gold) — a watch-item, **not** something to threshold-fish.

Conviction sizing works (boosts 6/6 lifetime): sizing multipliers on already-firing entries,
never gates.

---

## 2. THE STRATEGY SPEC (the deliverable centerpiece)

Named in WI terms. **Long-only** (distribution/short = closed by evidence). Daily execution,
weekly-anchored structure. All params IDENTICAL to add.48 — no retuning.

### PERMISSION (HTF state — the entire regime filter, stand-down by construction)
An active **weekly-anchored structural range** with a **CONFIRMED body-close up-break**
(the door's T1–T3): two consecutive daily closes above the live rolling weekly range ceiling.
`break_level` = that ceiling. **No confirmed up-break ⇒ flat.** There is no separate regime
veto — the break requirement *is* the regime filter. Also carried: the weak price-regime
protection `close > ema_200 OR struct_range_state != broken_down` (crypto rotation flag = 0).

### ENTRY (the retest / LPS that holds — T4–T5, daily close)
- T1 `eye_dir == 'bull'` (bias set bull only by a confirmed up-break).
- T2 `eye_state ∈ {IN_RANGE, MANIPULATION}` (non-extension — enter the pullback, never the
  breakout extension; MODEL_FORMING is inert price-only).
- T3 a bull CONFIRMED_BREAK occurred within `M2_SOS_WIN` bars → `break_level`.
- T4 `close ≥ break_level` (the up-break was NOT given back = HOLD).
- T5 `low ≤ break_level + RTZ_ATR·ATR` (pulled back INTO the retest zone).
- **Enter LONG at that daily close.** Price-only: no volume anywhere on the entry path.

### CONVICTION SIZE (boost stack — multiplicative, capped; NEVER gates)
Base risk **1%**. Pre-registered multipliers, applied to already-firing entries only:

| axis | multiplier | evidence | computable on fresh vendor OHLCV? |
|------|-----------:|----------|-----------------------------------|
| fib-time confluence tier | ×1.25 | validated 3/3 (price-only) | **Yes** — computed on every asset |
| eq_magnet proximity | ×1.25 | add.53 gem #3 (store-proxy, weak) | **Inert** at verbatim 0.1% tol on daily pivots (see Part 2) |
| ob_quality top-half | ×1.25 | add.53 (store-proxy, weak) | **No** — needs full 5-comp HOB pipeline; RESERVED |
| **cap** | **×2.0 total** | | |

The headline result is reported at **rmult = 1.0 (boost-immune)** — the boost is a sizing tilt,
never part of the verdict. Pre-registration is fixed BEFORE measuring; sizing never changes which
bars fire.

### EXITS (WI banked-and-derisked)
- **TP1 40%** at `struct_range_high` (else `swing_high_50`, else `entry+1R`; must clear ≥0.5R).
- On TP1 fill → **stop to breakeven**.
- **Runner 60%** to the **measured move** `struct_range_high + (high−low)`, floored at `entry+2R`.
- Initial stop = `created_LPS_low − 0.25·ATR14(entry)` (under the pullback leg's low).
- `max_hold = 168` daily bars. Dedup: no re-entry within 3 bars of the last entry.

### PARAMS (identical to add.48; no retuning)
```
M2_SOS_WIN=360   LPS_LOOKBACK=48   RTZ_ATR=0.5   STOP_BUF_ATR=0.25
DEDUP_K=3   MAX_HOLD=168   HTF_N=5 (weekly struct range)
eye: N_RANGE_1D=40, ACCEPT_CONSEC=2, TREND_CONSEC=5
conviction: FIB_MULT=1.25, EQ_MULT=1.25, STACK_CAP=2.0
```
Daily-rescale caveat (flagged, unchanged from add.48): on daily exec these bar counts scale up
(MAX_HOLD 168d ≈ 8mo; M2_SOS_WIN 360d ≈ 17mo). Identical across assets by design — the point is
cross-asset consistency of an unchanged model, not per-asset optimality.

---

## 3. FRESH-ASSET VALIDATION — results (the real, un-mined evidence)

**Data:** daily spot from **Coinbase Exchange public candles** for the un-mined universe. The
Yahoo v8 chart endpoint (the requested source) returned persistent **HTTP 429** from this
environment across multi-minute exponential backoff on both hosts; Stooq gates behind a JS
proof-of-work; Binance is geo-blocked (451). Coinbase responds cleanly and is the venue Part 4
targets, so its spot daily series is the most decision-relevant reference. **The un-mined assets
are unchanged; only the vendor differs.** Costs 2bps + 3bps/side, 1% risk, $100k start.
Self-test parity vs the textbook referee = **0.00% on every asset**.

### Per-asset (headline struct/flat, rmult=1.0, IDENTICAL add.48 params)

| asset | n | WR | PF | PnL | MaxDD | above-EMA200 | bear fires | stand-down | stack PnL (fib×1.25) |
|-------|---:|---:|---:|----:|------:|-----:|---:|:---:|---:|
| ETH-USD | 12 | 58% | 1.93 | +4,816 | −2.0% | 100% | 1 | 2/3 | +6,289 |
| SOL-USD | 3 | 100% | inf | +3,829 | 0.0% | 100% | 1 | 2/3 | +4,334 |
| LTC-USD | 9 | 56% | 3.26 | +7,840 | −2.3% | 89% | 2 | 2/3 | +9,443 |
| XRP-USD | 4 | 25% | **0.08** | −2,748 | −2.8% | 75% | 0 | 3/3 | −2,691 |
| ADA-USD | 3 | 33% | 0.28 | −1,104 | −1.5% | 100% | 2 | 2/3 | −1,487 |
| DOGE-USD | 2 | 0% | 0.00 | −1,233 | −1.2% | 50% | 2 | 1/3 | −1,483 |
| DOT-USD | 1 | 100% | inf | +491 | 0.0% | 100% | 0 | 3/3 | +491 |
| AVAX-USD | 1 | 100% | inf | +2,213 | 0.0% | 0% | 0 | 3/3 | +2,213 |
| LINK-USD | 5 | 80% | 9.89 | +8,902 | −1.0% | 80% | 0 | 3/3 | +8,627 |
| **BTC-USD (ref)** | 10 | 60% | 2.42 | +4,532 | −1.2% | 90% | 2 | 2/3 | +4,129 |

- **PF ≥ 1.5 on 6/9 fresh assets** (ETH, SOL, LTC, DOT, AVAX, LINK). The 3 that fail (XRP, ADA,
  DOGE) are **all n < 5 (directional only)**; XRP additionally has a ~2.5-yr Coinbase-suspension
  data gap (1,820 bars over 7.5 yr) → its PF 0.08 is a data-integrity artifact, not a model
  verdict. Restricting to the statistically-separable n ≥ 5 assets: **3/3 pass** (ETH, LTC, LINK).
- **Above-EMA200 concentration** pooled across fresh assets: **88%** (35 above / 5 below by count).
- **Bear-window fires (the add.48 leak):** pooled fresh n=8, PF **0.58**, **−$1,610** — net loser,
  a minority (8/40 = 20% of trades), ~6% of gross. **Reproduced exactly as add.48 described:** the
  door leaks on bear-market-rally breakouts; it is a modest drag, not a survival threat.

### Episode-adjusted & basket (BTC-correlation honesty)

| metric | value |
|--------|------:|
| RAW fresh trades | 40 |
| INDEPENDENT episodes (±5-day cross-asset clustering) | **32** |
| raw trades / basket-year | ~4.4 (over 9.1-yr span) |
| episode-additive PF (fixed 1% risk) | **2.68**, +$22,667 |
| basket PF (fixed 1% risk, exit-ordered) | **2.50**, +$22,667, MaxDD **−5.79%** |
| basket CPCV (K=6, m=2) | mean PF 2.84, frac PF>1 **100%**, frac PF≥1.5 **87%** |

### Verdict vs the pre-registered rule
Pre-registered (fixed before running): *PF ≥ 1.5 on a majority of fresh assets AND aggregate
fresh episode PF ≥ 1.5 AND bear stand-down consistent with add.48 AND no asset catastrophically
negative (PF < 0.5, n ≥ 5).*

- **C1 majority PF ≥ 1.5:** PASS — 6/9 all-assets, 3/3 among n ≥ 5.
- **C2 aggregate episode PF ≥ 1.5:** PASS — 2.68.
- **C3 bear stand-down consistent w/ add.48:** PASS(qualified) — fires concentrate above-EMA200
  (88%); bear fires are a minority net-losing leak (−$1,610), exactly the add.48 signature.
- **C4 no catastrophic asset (n ≥ 5):** PASS — the only PF < 0.5 cases are all n < 5 (XRP data-gap,
  ADA, DOGE), flagged directional.

**→ PORTABILITY CONFIRMED.** The door generalizes to fresh, never-touched crypto with identical
params. The dead-spring baseline stayed the behavioral opposite (mixed/negative on aggregate,
catastrophic on DOT −$7,949 / AVAX −$2,991).

### Honesty caveats (binding)
- **Cryptos are heavily BTC-correlated.** 40 raw → 32 episodes (only −20%) says the fires are
  somewhat time-spread, but the assets are one macro factor. **The independent cross-asset
  evidence for the door remains SPX / NDX / Gold** (uncorrelated, add.48). Fresh cryptos test
  **portability + build the tradeable basket** — they are not independent confirmation of the edge.
- **Rarity is real and binding:** most assets fire 1–4 times over 5–10 yr (WI's true cadence).
  Only 3 fresh assets reach n ≥ 5. Single-asset PFs of "inf" (SOL/DOT/AVAX) are n≤3 — directional.
  **The basket is the unit of evidence, not any single asset.**
- **eq_magnet leg is inert here:** at the verbatim 0.1% equal-level tolerance (add.53 gem #3),
  **zero** bars show a ≥3-pivot cluster on daily crypto (the original was a 1H/intrabar store
  proxy). Loosening the tolerance to force fires would be tuning — **forbidden**. So the boost
  stack on fresh assets reduces to **fib-time ×1.25**, which is mildly net-positive on the basket
  (~+12% PnL) and slightly negative on 2 assets — a sizing tilt, not an edge.
- **ob_quality leg not computed:** needs the full 5-component HOB pipeline (vol-surge / level-
  strength / reaction-speed) from raw bars — the dedicated quality-axis study (add.53 honest-next),
  not this one.
- **XRP data gap** and **all n < 5 assets** are directional only, not statistically separable.
- This is history, and history **cannot prove** a ~1–4/yr signal. The only remaining honest
  validation is **forward paper-collection of the basket** (Part 4).

---

## 4. DEPLOYMENT PACKAGE (proposal only — NOT deployed; needs explicit user go)

### Tradeable universe on Coinbase INTX perps (plausible)
BTC-PERP, ETH-PERP, SOL-PERP, LTC-PERP, XRP-PERP, DOGE-PERP, ADA-PERP, AVAX-PERP, LINK-PERP,
DOT-PERP (availability shifts; confirm live product list). BNB is **not** on Coinbase → excluded.
Recommended live basket = the assets with a real edge signature + adequate history:
**BTC, ETH, LTC, LINK** (n ≥ 5, PF ≥ 1.5), with **SOL/DOT/AVAX/ADA/DOGE/XRP** collected in
paper as breadth (they are the whole point — rarity is fixed by breadth, not by trusting any one).

### How it coexists with the live 1H archetype engine
**Recommendation: a SEPARATE daily-cadence strategy runner**, NOT an 18th archetype.
Rationale: (a) the door executes on **daily closes with weekly-anchored structure** — a different
clock from the 1H fusion/dedup engine; folding it in as an archetype would subject it to fusion
scoring (Lesson #54), dedup reshuffling (Rule 7), and the CMI threshold path it must stay
orthogonal to. (b) The door is **price-only and self-contained** (no store features, no volume) —
it has no dependency on the 16-archetype plumbing. (c) Clean attribution: a standalone book lets
the door's rare, high-PF fires be measured without crowd-out (the add.18/22 crowd-out lesson).
The two books share only the account and a portfolio-level risk cap.

### Config draft (daily runner)
```yaml
wi_m2_continuation:
  cadence: daily                     # decide on daily close; weekly-anchored structure
  universe: [BTC, ETH, SOL, LTC, XRP, ADA, DOGE, DOT, AVAX, LINK]   # PERP-INTX
  entry:  {door: trend_continuation, m2_mode: broad, price_only: true}
  params: {M2_SOS_WIN: 360, LPS_LOOKBACK: 48, RTZ_ATR: 0.5, STOP_BUF_ATR: 0.25,
           DEDUP_K: 3, MAX_HOLD: 168, HTF_N: 5, N_RANGE_1D: 40}
  size:   {base_risk_pct: 1.0, fib_mult: 1.25, eq_mult: 1.25, ob_mult: 1.25, cap: 2.0}
  exits:  {tp1_frac: 0.40, tp1_level: struct_range_high, after_tp1: breakeven,
           runner_frac: 0.60, runner_level: measured_move, runner_floor_R: 2.0}
  portfolio: {max_concurrent: unbounded_across_assets, per_trade_risk: 1.0pct}
  leak_watch: {flag: bear_rally_breakout, action: LOG_ONLY}    # NOT a filter
```

### Expected cadence & risk footprint
- **~4–5 trades/year for the 10-asset basket** (rarity confirmed). This is WI's real tempo.
- Basket historical MaxDD −5.8% at 1% risk; WR ~55–60% on the meaningful-n assets; PF ~2.5.
- Footprint is small and lumpy — this is a **patience strategy**; most weeks it is flat by
  construction (no up-break = no trade).

### Leak watch-item (logged flag, NOT a filter)
Log every fire's regime context; flag bear-market-rally breakouts (fires while a higher-timeframe
bear/markdown is in force). Do **not** gate on it (filters 0/10; add.50 showed a naive EMA-slope
gate is asset-inconsistent — fixes SPX, hurts NDX/Gold). Evaluate a real CMI-regime overlay only
on **forward** fires, once CMI labels are materialized live (add.50 correction).

### The forward-proof plan (the only remaining honest validation)
Paper-trade the basket on the daily runner; collect every fire with full context (regime, EMA200
side, fib/eq/ob axes, exit path). Re-evaluate after ≥1 year / ≥1 bear episode. Success = the
above-EMA200 concentration and basket PF hold out-of-sample, and the bear leak stays a minor drag.

---

## 5. Files (all study-only; production untouched)
- Spec: `docs/knowledge/ONE_STRATEGY.md` (this file).
- Implementation: `idea_lab/run_one_strategy.py` (fresh-asset harness + boost stack + episode
  clustering + basket + CPCV), `idea_lab/fetch_fresh_crypto.py` (Coinbase daily fetcher).
- Reuses the vetted add.48 door: `idea_lab/trend_continuation_door.py`,
  `idea_lab/unified_archetype_v2.py`, `idea_lab/backtester.py`, `idea_lab/xasset_spx_port.py`.
- Data cache (re-fetchable, not committed): `<scratchpad>/one_strategy/*.parquet`.
- **No production code, config, live, or deploy touched. No bypass/archetype/threshold changes.**
