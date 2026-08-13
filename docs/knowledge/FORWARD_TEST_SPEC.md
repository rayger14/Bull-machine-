# FORWARD-TEST SPEC — The ONE Strategy (trend-continuation door), wide basket
**Pre-registered 2026-08-13 · study/wide-basket · wyckoff_audit add.62 · SPEC ONLY, nothing deployed**

This document locks the forward paper-collection test that ENDS the trend-continuation-door
campaign. It supersedes add.60's single-basket M3 finish line by widening the universe to a
25-market, 4-family basket so the finish line arrives in ~2 years instead of ~7.

Base rule inherited verbatim from **add.60 M3**: forward paper-collection of a fixed-1%-risk
basket is the ONLY remaining honest validation of a ~1-4-trade/yr/asset signal; history cannot
prove it. Nothing about the door changes — identical `idea_lab/trend_continuation_door.py`,
identical params, headline struct/flat rmult=1.0, costs 2bps+3bps/side, 1% risk.

---

## 1. Forward paper universe (families that PASSED wide-basket Part 1)

Run ALL 29 markets in the backtest; the **forward basket is the 4 families that passed**
(crypto, equity-index, single-stock, metal). **FX and energy are DROPPED** from the forward
basket (they refuted the door — see add.62 Part 2) but are **still LOGGED as a shadow book** so
the domain boundary keeps being tested forward rather than assumed.

| Family (in basket) | Symbols | Data source (daily close) |
|---|---|---|
| crypto (10) | BTC ETH SOL LTC XRP ADA DOGE DOT AVAX LINK (USD) | Coinbase Exchange candles (INTX perps for live PnL) |
| equity-index (7) | ^GSPC ^NDX ^DJI ^RUT ^N225 ^GDAXI ^FTSE | yfinance daily (or vendor equiv.) |
| single-stock (3) | AAPL MSFT NVDA | yfinance daily |
| metal (4) | GC=F (gold) SI=F (silver) HG=F (copper) PL=F (platinum) | yfinance daily |
| **BASKET TOTAL** | **25 markets** | |
| shadow (logged, NOT in verdict) | CL=F (oil); EURUSD GBPUSD USDJPY AUDUSD | yfinance daily |

Rationale for the drop: energy PF 0.59 (n=13, directional-negative) and FX family PF 0.57
(n=63, 3/4 majors negative). The door needs a **trending** regime; FX majors mean-revert. The
one trending FX pair (USDJPY, carry-driven, PF 1.50) is kept in the shadow book as the live test
of that thesis, not promoted.

---

## 2. Expected cadence and time-to-verdict

| Basket | in-sample cadence | recent-era (2021-2026, all live) |
|---|---|---|
| FULL 29-asset | 14.0 raw/yr (36yr span) | 24.2/yr |
| **PASS 25-asset (the forward basket)** | 11.9 raw/yr | **20.0/yr** |

Honest forward cadence estimate = **12-20 at-size trades/yr** (full-history rate is the floor;
recent-era rate is elevated by crypto activity and should not be over-extrapolated). Episode-
adjusted (±5d cross-asset clustering, ~73% of raw across the full basket) the **independent**
cadence is ~9-15 episodes/yr.

**Time to n≥30 verdict: ~1.5-2.5 years** (30 / 12-20). Compared with add.60's single-basket
~4.5/yr → ~7yr, the wide basket cuts the finish line by ~3-4×. This is the deliverable's whole
point and it is achieved.

---

## 3. Acceptance test (LOCKED — do not tune after data starts)

Unit = the **25-market PASS basket at fixed 1% risk**, ordered by exit time.
Statistic = pooled per-trade R (episode-clustered for the CI; raw for PF).

- **CONFIRM** iff forward basket **PF ≥ 1.5** AND 1-sample **meanR bootstrap 95% CI-lo > 0** at **n ≥ 30**.
- **REFUTE** iff forward basket **PF < 1.0** with **meanR CI-hi < 0** (at any n ≥ 30).
- **KEEP COLLECTING** otherwise (the marginal zone between deflated-1.6 and 1.0).

Deflation context (add.60 M3, carried forward and re-checked on the wide basket): honest forward
expectation ≈ **half** the in-sample edge. In-sample PASS-basket meanR +0.431 → deflated ~+0.22R;
PF 2.29 → deflated forward PF ~**1.6-1.7**. The deflated point estimate clears CONFIRM's 1.5 floor
but only marginally — **the forward test is a genuine test, not a formality.** Family selection
(dropping fx/energy on the same history) adds forking; the forward data re-adjudicates it, so the
4-family domain claim is itself a forward hypothesis, not a settled result.

Secondary (non-gating) reads to publish at each checkpoint: per-family forward PF (does the domain
claim hold live?), above-EMA200 share (expect ~90%+), bear-window fire tally (the known leak),
shadow-book (fx+energy) PF (is the door's domain boundary where we drew it?).

---

## 4. Minimal runner spec (SPEC ONLY — not built, not deployed)

**A separate, lightweight, daily-cadence process — NOT the 1H archetype engine.** Keeps the door
off fusion / dedup / CMI / the 16-archetype book entirely (per add.54 deployment note).

- **Cadence:** one evaluation per UTC calendar day, after the daily close prints for each family
  (staggered: crypto 00:00 UTC; US equities/indices/metals after the 21:00 UTC session close;
  FX/energy on the shadow book, same daily bar). Idempotent: re-running a day must not double-fire.
- **Per asset, each day:** pull daily OHLCV → `build_daily_sensors` (weekly N=5 reanchor) →
  evaluate `TrendContinuationDoor` at the latest closed daily bar. If it returns a plan, record a
  paper entry at that bar's close with 1% risk; manage with the frozen struct/flat exit chain
  (TP1 40% at struct high → BE → 60% runner to measured move; max_hold 168d).
- **Data sources:** as in §1. Coinbase candles for crypto (matches the intended INTX live venue);
  yfinance (or a paid equiv.) for indices/stocks/metals/fx/energy daily bars.
- **Costs modelled:** commission 0.0002/side + slippage 3bps/side (verified cost-insensitive in
  add.60 M2b at 3× costs). Volume is never read by the door path (confirmed: `build_structural_poc`
  is orthogonal and uncalled; FX fired normally on 100%-zero-volume series) → FX/index volume noise
  is irrelevant.
- **State:** append-only trade ledger (entry_time, asset, family, entry, stop, exit_time, exit,
  R, pnl_fixed, exit_reason) + a live basket-equity curve at fixed 1% risk.

### Flags logged alongside EVERY fire (for later overlay evaluation, never as a filter)
1. `CMI_regime_label` — the live CMI v0 regime label at the fire bar (bull/bear/neutral/crisis),
   for the bear-overlay study (add.48/49/50 open thread: a causal regime gate is the door's one
   missing organ). Logged, **not** gated.
2. `bear_window_flag` — boolean: is the asset in a defined bear window (crypto calendar / equity
   dotcom-GFC-COVID-2022 / else below-EMA200 proxy)? This is the known **bear-rally-breakout leak**;
   every historical bear fire lost. Watch-item, logged never filtered.
3. `above_ema200` — boolean regime context (universal, defined on every asset).
4. `family`, `R_outcome`, `episode_id` (±5d cross-asset cluster id) — for family and episode
   attribution at each checkpoint.

These flags let a future study evaluate a bear-regime overlay **on real forward fires** (paired
channel, the only powered channel per add.60 M1) without ever having gated the collection.

---

## 5. What this spec deliberately does NOT do
- Does not deploy anything, touch production, or add an 18th archetype.
- Does not filter on any flag (CMI/bear/EMA) — those are logged for a later, separately-registered
  overlay test.
- Does not re-open fusion, dip-buying, speed-up, exit-toolkit, or campaign/Gann topology
  (add.60: decisive rejections / structural detector limits).
- Does not promote fx/energy into the verdict basket; they remain a shadow book that keeps the
  door's domain boundary under live test.

**Finish line:** first checkpoint at n=30 basket trades (est. ~2027-Q4 to 2028 at 12-20/yr), then
rolling. CONFIRM → the door is a real, deployable cross-asset daily edge. REFUTE → the in-sample
edge was forking/optimism and the campaign closes. Either way, a verdict arrives in ~2 years.
