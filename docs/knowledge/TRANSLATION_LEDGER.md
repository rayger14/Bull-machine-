# THE TRANSLATION LEDGER — "he-said vs we-coded"
### A fidelity audit of Wyckoff-Insider's methodology as mechanized in Bull Machine, written FOR WI to review

**Status:** STUDY ONLY. Nothing shipped, no production code/config/live/deploy touched.
**Branch:** `study/translation-ledger` off `study/wi-batch7`. Companion: `wyckoff_audit.md` addendum 66.
**Date:** 2026-08-15.

---

## 0. Why this document exists

Across this campaign we mechanized WI's discretionary method into a frozen, testable "door"
(the trend-continuation breakout-retest) and then ran ~18 add-on experiments against it — of which
**0 adopted**. The most recent (batch-7, add.65) tested WI's own core claim — *"the naked retest
underperforms the confluent retest"* — and **refuted it**: on 8,601 stock fires the naked retest
was as good as or better than every confluence tier.

WI reasonably asks the fair question: **four confirmed translation bugs already happened this
campaign** —

1. **Mislocated springs** (add.31/34 era): springs keyed off `rolling(20).min` instead of a drawn
   structural range → they fired in the wrong place. FIXED by `structural_range.py`.
2. **4H eye resample bug** (add.63 lineage): fractal-cadence mismatch when the eye was pushed to 4H.
3. **Gann anchor saturation** (add.58/59): mechanizing "major pivots" as *every* weekly N=5 fractal
   blanketed >50% of the calendar → the time layer was useless *by construction*, not by absence of edge.
4. **Disjoint M2 states** (add.45): the strict-M2 door fired **0 trades** because `MODEL_FORMING` and
   `CONFIRMED_BREAK` are temporally disjoint in the state machine — an architectural mistranslation,
   not a market fact.

So: **is batch-7's refutation a fifth translation failure, or a real null?** This ledger is the
honest answer. It lays out, layer by layer, **his verbatim words**, **our exact code (with every
number and file:line)**, **every interpretation choice that is OURS not his**, **a concrete dated
chart case he can adjudicate from his own charts**, and a **translation-risk grade**. It then
reports two empirical checks built specifically to catch a tilted testing frame (Deliverables 2 & 3).

**How to read the risk grades:**
- **LOW** — our code is a near-verbatim transcription of a specific WI statement; little room to diverge.
- **MED** — the object is WI's but one or more *magnitudes/windows are ours*; a reasonable person could
  have picked differently, and the choice could move outcomes.
- **HIGH** — the mechanization makes a *structural* choice WI never specified (or that we already know
  diverges from his practice), such that it could plausibly change a verdict. These are the rows most
  likely to be *our* fault.

**Provenance rule used throughout** (add.35): the *objects* (range, break, LPS, RTZ, Bojan, TT, Gann
counts) are WI-sourced; almost all *magnitudes* (thresholds, windows, tolerances) are OURS. WI publishes
very few numbers. Every "OURS" below is a place we filled a number he did not give.

---

## 1. THE LEDGER — layer by layer

Each row: **(a) HIS WORDS** · **(b) OUR CODE** · **(c) OURS-interpretation choices** · **(d) chart case** · **(e) risk**.

---

### LAYER 1 — Structural range (the drawn box) + premium/discount location

**(a) His words** (add.11, add.13): *"HTF range until proven otherwise."* The HTF range is a **FIXED
structural object** (W/M swing-anchored, redrawn ONLY on a body-close MSS break); *"premium/discount
measured against THAT."* He explicitly warned our earlier rolling-window location was mis-built:
*"in trends a rolling range trails price so 'premium' degrades to 'market has been rising' = regime beta."*

**(b) Our code:**
- Range object: `idea_lab/structural_range.py::build_structural_range`. A persistent drawn box
  `(range_low, range_high, formed_at)` that holds bar-to-bar and redraws only on a **body-close** break
  (`close > range_high` / `close < range_low`, `structural_range.py:170-196`). Wick-beyond-close-back-inside
  = a **sweep**, not a break (`:198-202`). Initialize from the first pair of confirmed pivots that bracket
  price and are `≥ MIN_WIDTH_ATR` apart (`:136-145`). Optional higher-low floor-tighten (`:204-211`).
- Anchors: **weekly N=5 fractal pivots**, `xasset_spx_port.py::reanchor_frame_weekly` (`:218-241`), pivots
  confirmed only after `p+N` weekly bars close (causal, ~35-day lag).
- Location: `struct_range_pos = (close-low)/(high-low)` (`structural_range.py:163`), 0=discount 1=premium.
- Separately, the **door's break/retest** uses the DAILY *eye* range boundary (`eye_state.py`), a rolling
  40-day **body**-extreme range shifted 1 bar (`eye_state.py:128-133`, `N_RANGE_1D=40`).

**(c) OURS (not his):** `MIN_WIDTH_ATR = 1.5` (noise floor); `TIGHTEN_MIN_WIDTH_ATR = 0.75`; **N=5** weekly
fractal half-width (he never published a pivot lookback — we pre-registered N=3 and N=5); **weekly** as the
anchor TF (he says his true scale is **Weekly/Monthly** — 1D→1W may still be a compromise, add.35 flagged);
the eye's **40-day** rolling boundary and **body-close** semantics (his "body close" is our max/min of open,close).

**(d) Chart case — GOLD:** our eye marked a confirmed bull break at **2024-09-16** (break_level = **$2,551.20**)
and the door bought the retest on **2024-09-18**; it ran to a measured-move runner target (+1.67R). *Is that
$2,551 ceiling the structural level you'd have drawn on Gold in Sep-2024, and is a body-close break of it what
you'd call confirmed?*

**(e) Risk: MED.** The drawn-box object and body-close-break rule are faithful. The **rolling-40-day eye
boundary that the door actually breaks against is NOT your fixed W/M drawn range** — it is a compromise we
adopted for causality/portability. This is the same disease you flagged in add.13 (rolling ≈ regime beta),
partially re-introduced at the eye layer. If your "range" is materially different from a 40-day rolling body
box, our *break location* differs from yours — a plausible source of divergence.

---

### LAYER 2 — Break confirmation / acceptance ("jump across the creek")

**(a) His words** (add.7, add.6): *"A valid PO3 needs three things: A defined range. An aggressive break of
that range. A Wyckoff inside the manipulation phase."* And on confirmation being **acceptance, not the event
bar**: *"Market structure break on LPS. That is the trade… waits for closes, sometimes days"* — which he noted
explains why our detector fired days before his label.

**(b) Our code:** `eye_state.py::_run_state_machine`. Break acceptance = **2 consecutive daily closes** beyond
the live boundary (`ACCEPT_CONSEC = 2`, `:59`, `:210-217`) → `CONFIRMED_BREAK`; 5 consecutive → `TRENDING`
(`TREND_CONSEC = 5`). A single close beyond is a *pending* break, not a bias flip (`:218-221`). The struct-range
object uses `break_confirm_bars = 1` (a single body close, `structural_range.py:80`) for its own state.

**(c) OURS:** **2 closes** as the operationalization of "acceptance" (he said "sometimes days" — we picked 2);
**5 closes** for TRENDING (entirely ours); the struct-range's **1-close** break vs the eye's **2-close** break
(two different acceptance definitions in the same door — a latent inconsistency, ours).

**(d) Chart case — SPX:** break bar **2024-01-25** at **4,864.60** confirmed after two daily closes above;
door entered the retest **2024-02-01** (+1.61R). *Two daily closes — is that "acceptance" to you, or would you
require a weekly-close acceptance (which would delay/skip this)?*

**(e) Risk: MED.** "2 closes" is a defensible reading of "waits for closes" but you may mean **weekly** closes
or a fuller MSS sequence. If your acceptance is slower/weekly, our door fires **earlier and more often** than you
would — directly relevant to the "we fire on structure alone" critique.

---

### LAYER 3 — The retest / LPS entry (the whole trade)

**(a) His words** (add.7): *"Return to zone inside the range. LPS + Bojan. Market structure break on LPS. That
is the trade."* *"MS break > Time displacement"* — prefers the **return after MSS**, no chase of unreturned
displacement. Enter the **back-up that holds**, not the raw reclaim (add.45 framing of his failing-spring rule).

**(b) Our code:** `unified_archetype_v2.py::_m2` (m2_mode="broad"), the door's five checks (`:259-291`):
- **T1** `eye_dir == 'bull'` (bias set bull only by a confirmed up-break)
- **T2** `eye_state ∈ {MODEL_FORMING, IN_RANGE, MANIPULATION}` (non-extension — enter the pullback, not the extension)
- **T3** a bull `CONFIRMED_BREAK` within `M2_SOS_WIN=360` bars → `break_level = range_upper` at that break
- **T4** `close[i] ≥ break_level` (the up-break was NOT given back = HELD)
- **T5** `low[i] ≤ break_level + RTZ_ATR·ATR` (pulled back INTO the retest zone)
Entry at `close[i]`, daily.

**(c) OURS:** `RTZ_ATR = 0.5` (the retest-zone depth — he never quantified fill depth; add.8: *"fill depth
unquantified"*); `M2_SOS_WIN = 360` daily bars (~17 months) as the validity window; the **{MODEL_FORMING,
IN_RANGE, MANIPULATION}** state set (MODEL_FORMING is **inert** price-only because it needs a `C_accum` phase
label we drop); **no `+Bojan` requirement** in the mechanized entry (his "LPS + Bojan" — we test Bojan only as a
sizing tilt, add.10, not as an entry gate). Entry at the **daily close**, not at a resting limit in the zone.

**(d) Chart case — AAPL:** break **2025-09-23** ($245.50), retest-hold entry **2025-10-13** (freshness 14 days,
+1.09R). *Is buying the daily close on the hold your fill, or would you have rested a limit inside the zone and
required the LPS+Bojan candle specifically?*

**(e) Risk: HIGH** *(elevated from MED-HIGH on external review, 2026-08-15)*. The break→pullback→hold shape is faithful and is the layer that carries the edge. But
**"LPS + Bojan" is reduced to "close held above the level + dipped into an ATR-scaled zone"** — the *Bojan
candle*, the *inducement/sweep of the LPS*, and the *fill-in-zone* are all dropped or approximated. If your entry
is specifically the LPS-Bojan candle (not any hold), our fire set is a **superset** of yours — which is exactly
what would make our door "fire on structure alone." **This is the single most likely place the mechanization is
looser than you.**

---

### LAYER 4 — Stops (per-model, under the created LPS)

**(a) His words** (add.6, add.7): TP1 partial, then *"SL movement under created LPS. Position fully derisked
1/10."* Stop is the **model's** invalidation — under the newly-created LPS structure — **not** breakeven and
**not** the deep range low. *"SL under 45M Bojan low."*

**(b) Our code:** `_plan` (`unified_archetype_v2.py:294-344`): `created_low = min(low)` over the pullback leg
(`LPS_LOOKBACK = 48` bars, `_m2:287-288`); `stop = created_low − STOP_BUF_ATR·ATR14(entry)` (`:300`,
`STOP_BUF_ATR = 0.25`). On TP1 fill the stop moves to **breakeven** (`move_stop_to_after_first_tp = entry`, `:341`).

**(c) OURS:** `LPS_LOOKBACK = 48` bars (what "the created LPS low" means mechanically); `STOP_BUF_ATR = 0.25`
(buffer below it); **the post-TP1 stop is breakeven, NOT "under the created LPS"** as you describe — a known
divergence from your recipe (add.6 explicitly: *"under created LPS, NOT breakeven"*).

**(d) Chart case — GOLD 2026-01-26 break / 2026-02-04 entry ($4,908.80):** this one **stopped out (−1.00R)**.
Our stop sat `min-low-over-48-bars − 0.25·ATR` below entry. *Would your "under the created LPS" stop have been
tighter (cutting the loss sooner) or wider (surviving the shake-out)?* This trade's outcome may hinge entirely on
the stop translation.

**(e) Risk: MED.** Initial stop is a faithful "under the LPS structure." But **post-TP1 → breakeven** contradicts
your **post-TP1 → under created LPS** rule. We tested your exact structure-trail once (add.56, Moneytaur-style) and
it was net-harmful *on our single-position engine* — but that engine can't express your re-entry topology, so the
test was not a fair adjudication of your stop.

---

### LAYER 5 — Targets / derisk / runner (TP1, TT, the 1/10 core)

**(a) His words** (add.6/7/8): TT defined **in advance** = measured-move / "2X" projection, often **beyond** the
plain range high, precise to decimals (e.g. 74,384.7). **TP1 ~40%** at the **first opposing supply zone / Bojan
high** (structural, not an R-multiple), derisk to **~1/10**, **core runner held to full TT**, does **not** trail
LTF once TT is mapped. UNF (unfinished W/M/5D) candle levels act as magnets/targets.

**(b) Our code:** `_plan` struct variant (`:327-344`): **TP1 40%** at `struct_range_high` (else `swing_high_50`,
else `entry+1R`; must clear `≥ MIN_TP1_R=0.5R`); runner **60%** to the **measured move** `struct_range_high +
(range_high − range_low)`, floored at `entry+2R`; `max_hold = 168` bars.

**(c) OURS:** **60% runner** vs your **~1/10 (≈10%) core** — a large divergence in position topology; TP1 anchor =
`struct_range_high`/`swing_high_50` **proxy** for "first opposing supply / Bojan high" (add.9: crude proxies); the
**measured-move formula** `high + (high−low)` is our reconstruction — your exact TT formula is **not publicly
recoverable** (add.8); `max_hold = 168` daily bars (~8 months) is ours; **no UNF-magnet target ladder**.

**(d) Chart case — SPX 2024-02-26 break / 2024-08-05 entry:** ran to a measured-move runner (+3.32R) over a long
hold. *Is `range_high + range_width` your TT here, or would your "2X"/UNF projection have banked earlier or later?*

**(e) Risk: MED-HIGH.** The **derisk-to-1/10 vs our 60%-runner** difference means our P&L distribution (and MaxDD,
and how a runner gives back) is **structurally different from yours** even on identical entries. Batch-7 measured
**entry-quality R**, so this exit divergence does not by itself explain the confluence null — but it means our
headline PnL is not your PnL.

---

### LAYER 6 — Campaign topology (1–3 sized entries per model; bank → re-enter)

**(a) His words** (add.57 verbatim): does **not** buy once and sit — banks TP1 40%, moves stop under the **newly
created** LPS/Bojan, fully derisks to ~1/10, takes **later LPS/RTZ/Bojan entries inside the same larger structure**.
Typically **1–3 (sometimes more) sized entries per major model**; major completed HTF models = 2–5/yr.

**(b) Our code:** `campaign_strategy.py::CampaignV2Door` (add.58) — multi-entry engine with `E≤3` higher-break
re-entries **while flat**; `CampaignV2bDoor` (add.59) — gentler LPS-hold re-entry. Both reuse the door's
entries/stops; only the management differs. `campaign_backtester.py` (referee parity 0.00%).

**(c) OURS:** "new higher structure" pre-registered as a **higher `break_level`** (add.58) then relaxed to a
**higher LPS low** (add.59); **re-enter only when FLAT** (our engine is one-position) — your **pyramiding while
positioned** is UNEXPRESSED (add.58 explicit caveat); `E≤3`, dedup 3 bars.

**(d) Chart case:** across 12 assets / 85 campaigns only **~3 second-entries ever fired** in the strict
definition; the gentler version fired 8 (BTC 1.29 entries/campaign). *On a single BTC accumulation you'd trade,
how many separate sized entries do you actually take — and are they while still holding the first, or only after
banking it flat?*

**(e) Risk: HIGH.** This is a **known structural mistranslation of your cadence**. Your "1–3 entries per model"
requires **pyramiding while positioned**; our engine only re-enters when flat, so on our detector the topology
*lowers* cadence (0.76–0.85×) instead of raising it (add.58/59). **If any part of your extra cadence survives
mechanization, it lives in concurrent adds we did not build.** This does not touch batch-7 (a per-entry-quality
test), but it is the clearest place we know we did not build what you do.

---

### LAYER 7 — Bojan / wick anatomy

**(a) His words** (add.8): *"Bojan = multi-timeframe precision"* — wick-tip zones tagging liquidity/supply;
highest-confluence stack = **M2 (3rd candle) + M1 (4th candle) + unfinished-candle push into liquidity + POC/Wyckoff
+ 0.5–0.618 fib**; invalidation = close beyond → converts to an unfinished candle. **No public wick-% rule** — he
said explicitly our legacy 70% threshold is *our* invention.

**(b) Our code:** batch-7 **N2b candle anatomy** (`run_wi_batch7.py:233-238`): entry-bar **lower wick ≥ 0.55** of
range AND close in **top 40%** (`(close−low)/range ≥ 0.60`). Bojan-zone detector: `bojan_detector.py` (wick-tip
zones), tested only as a **sizing tilt** (add.10).

**(c) OURS:** **0.55** wick fraction ("55–60%", we took the low end); **0.60** close-position (top-40%, ours);
the *single-candle anatomy* rule (your Bojan is a **multi-candle** M1/M2 sequence + a persistent **zone**, not one
candle's wick %); we **do not** implement the 3rd+4th candle sequence or the timeframe-scoped zone as an entry.

**(d) Chart case:** batch-7 N2b covered only **10.6%** of fires and its flagged trades (+0.280R) barely beat
unflagged (+0.279R), and the wide basket **inverted** it (0.284 < 0.451). *Is a single lower-wick candle even the
right object, or is your Bojan fundamentally the M1/M2 candle **pair** into a pre-drawn zone — i.e. did we test the
wrong thing entirely?*

**(e) Risk: HIGH (for "did we test your Bojan?").** Our N2b is a **crude single-candle proxy** for a multi-candle,
multi-timeframe, zone-based concept you consider central. Its null in batch-7 is honestly **weak evidence about
your Bojan** — it is strong evidence only that *this particular one-candle anatomy* adds nothing. This row is a
prime candidate for "WRONG — that's not my Bojan."

---

### LAYER 8 — Sweep-retest (the liquidity grab into the level)

**(a) His words** (add.7/8): inducement = engineered trap level; the entry is the **return after** the sweep of
liquidity at the LPS/zone; *"no chase of unreturned displacement."* (Sweep = wick beyond, close back — add.13:
manipulation is a wick deviation, **never** a bias flip.)

**(b) Our code:** batch-7 **N2a sweep-retest** (`run_wi_batch7.py:224-232`): on entry bar `i` or `i−1`, the low
undercuts `break_level` by **0.1–0.4 ATR** then that bar **closes back above** `break_level`. Struct-range sweeps:
`struct_sweep_low/high` (`structural_range.py:198-202`).

**(c) OURS:** the **0.1–0.4 ATR** undercut band (he described the *behavior* verbatim — "spring-style/liquidity
grab retest" — we picked the depth band); the **1-bar** sweep window (`i` or `i−1`), ours.

**(d) Chart case:** N2a covered **27.8%**; flagged vs unflagged meanR **identical (0.279 vs 0.279)**; full-history
flag **worse**, wide **better** (inconsistent). *When you take the sweep-retest, is the grab typically 0.1–0.4 ATR
below the level, deeper, or is depth irrelevant and it's the reclaim-speed that matters?*

**(e) Risk: MED.** The behavior is verbatim-sourced; only the depth band and window are ours. But the add.35/
archaeology note (*sweep reclaim-speed in hours, golden-pocket location, TTL decay* lost in translation) means our
N2a captures the **geometry** of the grab but not its **speed/location quality** — which may be where your edge is.

---

### LAYER 9 — Freshness (how soon after the break the retest is valid)

**(a) His words** (add.11): a setup carries a **validity window** — *"time-based expiry of the expected window"* is
itself an invalidation; *"data points alone are not a trade."* (He gave no bar count.)

**(b) Our code:** batch-7 **N1 freshness** (`:220-222`): `bars_from_break_to_retest ≤ F`, **F = 10**. Door validity
window `M2_SOS_WIN = 360` bars (the outer expiry).

**(c) OURS:** **F = 10** ("8–12", we took the midpoint); the **360-bar** outer window (ours). Both are pure
interpolations of "the window expires."

**(d) Chart case — the freshness result is inverted:** across the primary set the **freshest** retests (0–10 bars)
were the **worst** (meanR +0.215, PF 1.66) and staler retests held up fine (181–360 bars: +0.369, PF 1.81). GOLD's
four recent winners had freshness 2, 3, 3, 7 days; but SPX's biggest winner (**+3.32R**) had freshness **111 days**.
*Is "fresh = better" actually your rule, or does a valid structure stay valid for months until a body-close
invalidates it?* Our data says the latter.

**(e) Risk: MED.** F=10 is our number and it **loses** — but the finding (freshness doesn't help, staleness doesn't
hurt within the window) is robust across primary/full/wide. If your freshness intuition is real, our mechanization
(a hard bar count) is the wrong shape for it. Worth your correction: is it calendar-freshness or *structure-still-valid*?

---

### LAYER 10 — Origin location (breakout from discount, not premium)

**(a) His words** (add.13): *"Deep in premium is where you start prepping longs, not chasing fear"*; breakouts
from **discount/EQ** of the HTF range, not extreme premium; premium/discount measured against the **fixed** range.

**(b) Our code:** batch-7 **N4 origin** (`:248-257`): at the break bar, `break_level` sits in the **lower 60%** of
the trailing **365-calendar-day** high-low span.

**(c) OURS:** **0.60** discount threshold (ours); **365-day** trailing span as the "HTF range" proxy (ours — you
measure against your **drawn** range, not a rolling 365d window).

**(d) Chart case:** N4 covered 20.7%; flagged trades were **worse** than unflagged on both full and wide (agree
flag-worse). *When you say "discount," do you mean the lower third of **your drawn accumulation range**, or the
lower 60% of the last year's price? These can be very different levels.*

**(e) Risk: MED-HIGH.** The **365-day rolling span is not your drawn range** — same rolling-vs-structural disease as
Layer 1. Your "discount" is location within a *specific drawn box*; ours is location within a *rolling year*. If your
box is much smaller/higher than the trailing year, our N4 flags the wrong trades — a plausible mistranslation.

---

### LAYER 11 — Chop-count veto (stand down when the range keeps failing)

**(a) His words** (add.11/15): stands down on **dominance/structure** developing against him; *"no confirmation =
no trade"*; against-HTF → main size ~zero. (He did **not** give a "count 3 failed breakouts in 90 days" rule — this
is our operationalization of "choppy/failing regime.")

**(b) Our code:** batch-7 **N5 chop-veto** (`run_wi_batch7.py:147-163, 259-266`): count **failed breakouts** (a bull
CONFIRMED_BREAK whose close drops back below its level within **15 bars**, resolution ≤ entry) in the trailing
**90 days**; `≥ 3` = "chop regime," tested as removal and inverse-size.

**(c) OURS:** the **entire definition** — 90-day window, 15-bar failure horizon, **3-fail** threshold — is OURS.
WI gave a *concept* (stand down in chop), not this counter.

**(d) Chart case:** N5 covered **2.4%** (degenerate); in bears it MISSED the target — 2018Q4 and 2022 chop-flagged
trades **outperformed** calm ones; COVID had 1 flag. *Is "3 failed breakouts in 90 days" anything like how you
sense chop, or do you read it off dominance-structure / a failing model directly?*

**(e) Risk: HIGH.** This is a **fully-invented mechanization** of a concept you hold, and it is the spec you cared
most about (the bear-rally leak). Its null is honestly **not a test of your judgment** — it's a test of *our guess
at your judgment*, and our guess degenerated (2.4% coverage). The real missing organ remains a **causal bear-regime
gate**; a chop-count is not it. Strong candidate for "WRONG — here's how I actually read chop."

---

### LAYER 12 — Climax / blow-off veto

**(a) His words:** *(none specific)* — WI describes skipping the chase/exhaustion but gave no ATR rule.

**(b) Our code:** batch-7 **N6 climax** (`:268-274`): `atr_14` at the break bar `> 1.8×` its trailing-90-bar median → skip.

**(c) OURS:** **1.8×** and **90-bar** median — both ours, no WI number.

**(d) Chart case:** covered 2.4% (degenerate), removal-delta ≈ 0. *Do you skip breakouts on abnormally wide/volatile
candles, and if so by what feel?*

**(e) Risk: MED (but low-stakes).** Fully ours, but it barely fires; its null is uninformative either way.

---

### LAYER 13 — Gann time (windows, anchors, danger zones)

**(a) His words** (add.57 verbatim): *"Gann timed. Wyckoff mapped."* Never trade Gann alone — react only when a
Gann window **and** Wyckoff confirmation align. Counts **90/180/360/540/720/1080/1440 days** from major highs/lows
or the **halving**; **144** is the recurring hidden number; turns land within **±1–3 bars**. *"Don't be in a swing
long in the final Gann time window"* (Very High tier).

**(b) Our code:** `idea_lab/gann_time.py`. Anchors = weekly N=5 pivots + halvings (**2016-07-09 / 2020-05-11 /
2024-04-19**); counts as verbatim + 144d + 144w; `entry_window` = within **±3 days** of any count; `danger_window`
= halving+{720,1080,1440} OR major-high+{360,540,720,1080,1440}. add.59 added a `major_only` trailing-365-extreme
anchor filter.

**(c) OURS:** **±3 days** tolerance (from "±1–3 bars", widest); the **danger-window count set** (our interpretation
of "final window"); **"major" mechanized as N=5 pivots** (then trailing-365 extreme) — the anchor-selection bug you
already saw: your "major" pivots are **discretionary and few**; ours over-produce.

**(d) Chart case:** the verbatim anchor set gave **50–65% calendar coverage**; even the major filter only reached
~25% (BTC/GOLD/NDX still saturated). *A ~55%-coverage "time window" can't discriminate — is that because "major
anchor" is inherently a discretionary call only you can make?*

**(e) Risk: HIGH (already a confirmed bug, add.58/59).** We **cannot mechanize "major anchor" causally without
discretion** — that is the finding, and it's an honest admission that this layer is un-translatable as-is, not that
your Gann is wrong. If Gann has edge for you, it is gated by a human anchor selection our code cannot reproduce.

---

### LAYER 14 — Dominance / cross-market bias

**(a) His words** (add.13/15): *"We're using Dominance as the main driver… the dominance chart gives the blueprint,
entries come from a valid Wyckoff on the crypto charts."* Same toolkit as price (fixed HTF ranges, M2, Bojans,
body-close breaks) applied to **W/M dominance**; stands down on dominance **structure**, not multi-day rises alone.

**(b) Our code:** dominance reader = `engine/features/stables_rotation.py` → `stables_rot_rising` (0/1), used in the
door's R0 (`_regime_ok`, `unified_archetype_v2.py:211-213`: requires `stables[i]==0`). The structural dominance
reader was **rejected** (add.14, commit 6e518c0: 0/3 tier ordering, 94.7% NEUTRAL, redundant with rot_rising).

**(c) OURS:** dominance reduced to a **binary 3-day rot-rising flag** (add.15: **shorter** than your structural
horizon); off-crypto assets get `stables = 0` (always-true) — so **dominance is entirely absent** from the
stock/metal/index door that batch-7 primarily tested.

**(d) Chart case:** on all 476 stocks + Gold + indices, `stables = 0` always → the door has **no cross-market input
at all** on the primary batch-7 universe. *Your dominance "main driver" simply does not exist in the equity/metal
version of this door — does that invalidate testing your confluence critique on stocks?*

**(e) Risk: HIGH (for crypto), N/A-but-important (for the batch-7 universe).** Your **core bias input is missing**
from the universe on which we refuted your confluence claim. A fair objection: "you tested my method with my main
driver removed." Counter: the door **validated cross-asset without dominance** (add.48/62, incl. uncorrelated Gold),
so dominance is not *necessary* for the edge — but its absence is a real fidelity gap you may legitimately contest.

---

### LAYER 15 — The M1 spring (dip-buy at discount)

**(a) His words** (add.6/7): the spring alone is a **data point**; the **jump + the back-up that holds** is the
trade. He buys the reclaim that holds, not the raw capitulation.

**(b) Our code:** `_m1` (`unified_archetype_v2.py:222-256`): struct_sweep_low spring within `SPRING_WIN=72`, SOS
reclaim (`SOS_BUF_ATR=0.10`), RTZ hold (`RTZ_ATR=0.5`) above the creek. **Not used** by the trend-continuation door
(M2-broad only); tested standalone as the `DeadSpringDoor` baseline.

**(c) OURS:** `SPRING_WIN=72`, `SOS_BUF_ATR=0.10`, `DISCOUNT_MAX=0.4` — all ours.

**(d) Chart case:** the M1 dip-buyer **died cross-asset** (BTC OOS-B 5/5 stops; SPX bears 0% WR; Gold −$1,596) — it
catches falling knives in markdowns. *Do you actually buy springs at discount, or is your real entry always the
continuation back-up after the break (Layer 3)?* Our evidence says the continuation is the edge and the discount
dip-buy is not.

**(e) Risk: LOW (as a translation — we correctly concluded it's not your edge).** We faithfully built and then
**retired** the dip-buyer; add.45/47/ONE_STRATEGY concluded your real edge is the continuation. If anything this is
a translation *success*: we discovered your method is not "buy the spring."

---

## 2. DELIVERABLE 2 — THE FORMING-BAR GAP (do we penalize you by waiting for completed bars?)

**The concern:** you trade the **forming** weekly candle (mid-week rejection reads); our sensors use only
**completed** bars. Where does that bite?

**Precise inventory (verified in code):** the door's **entry** path is **100% daily-completed-bar**:
- Break confirmation = the **daily** eye (2 consecutive **daily** closes vs a rolling-40-**day** boundary,
  `eye_state.py`). Evaluated at **every daily close** — it does **not** wait for the week to end.
- Retest-hold (T4) and proximity (T5) = **daily** close/low reads.
- The **only** completed-**weekly**-bar object is the struct-range's weekly N=5 pivots, and those feed **only**
  (a) the TP anchor `struct_range_high` and (b) the rarely-binding R0 `struct_range_state != broken_down` veto
  (`broken_down` is ~1% of bars, and R0's `close>ema200` branch dominates since 88–100% of fires are above EMA200).

**So the door is already finer-grained than your weekly forming-candle reads** — it reacts at each daily close.
The one place a forming-week read could still move an *entry* is the retest-**hold**: today it needs a completed
daily close **back above** the level.

**The one pre-registered variant** (`idea_lab/forming_bar_sensor.py`): relax **only** T4 to also confirm when the
**week-to-date high** has already reclaimed `break_level` and today's close is within `RTZ_ATR·ATR` of it — i.e.
accept the "mid-week rejection that holds." Week-to-date high is causal (running max within the current ISO week,
bars ≤ i). **Truncation no-repaint check: 1,124,972 points, 0 mismatches.** Referee parity **0.00%**.

**Results (status-quo A vs forming B):**

| set | A n | A PF | A meanR | B n | B PF | B meanR | matched | B-earlier | median shift | added fires |
|-----|----:|-----:|--------:|----:|-----:|--------:|--------:|----------:|-------------:|------------:|
| 50-stock sample (2018+) | 289 | 1.83 | +0.313 | 295 | 1.90 | +0.347 | 284 | 73/284 | **0 d** | 11 (PF 1.12) |
| wide basket (trending) | 432 | 2.29 | +0.431 | 444 | 2.36 | +0.472 | 418 | 115/418 | **0 d** | 26 (PF 1.63) |

**Findings:**
1. **~74% of entries are unchanged** (same-day) — the door already captures the retest at daily resolution.
2. On the **~26%** where forming-week evidence bites, entries move **earlier** (up to 39–60 days on stale setups),
   and outcomes are **mildly better** (paired ΔR +0.03 to +0.04, PF 1.83→1.90 / 2.29→2.36).
3. **The forming-bar gap is REAL but SMALL for this door**, and it does **not** manufacture the batch-7 result — if
   anything a forming-week door is marginally *better*, still with confluence adding nothing.

**Honest status:** this is **one in-sample diagnostic variant**, not an adoption. The mild positive is a **watch-item**
for a pre-registered forward look (train/OOS co-movement not yet established), not a change. The decision-grade point:
**the door does not penalize you by waiting for completed weekly bars — its entry clock is daily.**

---

## 3. DELIVERABLE 3 — SELECTION-BIAS AUDIT (did batch-7's frame manufacture the null?)

**The concern (legitimate):** batch-7 tested confluence as AND-overlays on fires **already selected** by the five
structural checks — conditioning on a collider, a frame that tilts toward null. The check the frame missed: build
the **near-miss** population — retest events that failed **exactly one** of the five checks — and ask whether high
confluence **rescues** them (confluent near-misses good, non-confluent near-misses bad). If yes, your confluence is
a **complement to marginal structure** (a *widening* tool) and batch-7 genuinely missed it.

**Method** (`idea_lab/run_near_miss.py`): on the **476-stock recent 2018+ window**, enumerate every bar (passing R0,
with a causal break_level) and count how many of {C1 dir, C2 state, C4 hold, C5 proximity} fail. `n_fail=0` = REAL
fire; `n_fail=1` = near-miss (tagged by which check). Each event simulated in **isolation** with the door's own
plan/exits/costs (an inline simulator **asserted bit-identical to `run_backtest`** — **sim parity 0.00% on 2,555
real events**). Confluence = the add.65 causal combined score {N1,N2a,N2b,N4}; "confluent" = score ≥ 2 (pre-registered).

**Event census:** REAL 15,123 · missC5 (didn't pull into zone) 80,175 · missC4 (retest didn't hold) 38,050 ·
missC1 (dir not bull) 4,750 · missC2 (extension state) 1,297. *(The C3-miss family = "no break in window" is not a
retest event and is excluded by construction, stated plainly.)*

**Results:**

| population | n | meanR [95% CI] | PF | confluent meanR (CI-lo) | non-confluent meanR | R2 (conf>nonconf) |
|------------|--:|:--------------:|---:|:-----------------------:|:-------------------:|:-----------------:|
| **REAL fires** (isolated) | 15,123 | +0.348 [+0.323,+0.374] | 1.91 | — | — | — |
| missC1 (dir) | 4,750 | +0.462 | 1.93 | +0.518 (+0.309) | +0.458 | ✓ (tiny) |
| missC2 (state) | 1,297 | +0.258 | 1.79 | +0.173 (+0.071) | +0.288 | ✗ (hurts) |
| missC4 (hold) | 38,050 | +0.448 | 1.94 | +0.361 (+0.297) | +0.453 | ✗ (hurts) |
| missC5 (proximity) | 80,175 | +0.283 | 1.73 | +0.307 (+0.258) | +0.282 | ✓ (tiny) |
| **POOLED near-miss** | 124,272 | +0.340 | 1.81 | +0.336 (+0.297) | +0.340 | **✗** |

Near-miss meanR by confluence score: s0 +0.327 · s1 +0.369 · s2 +0.339 · s3 +0.291 · s4 +0.303 — **flat / not monotonic.**

**VERDICT: NO RESCUE — the batch-7 null stands FRAME-INDEPENDENT.** Two things, both decisive:
1. **The near-misses are not a degraded population.** Failing any single structural check still yields a
   **profitable** book (meanR +0.26 to +0.46, PF 1.73–1.94), comparable to — sometimes better than — the real fires.
   So the door's five checks are **not sharply discriminating winners from losers** (caveat below); marginal
   structure is about as good as clean structure.
2. **Confluence does not rescue them.** Pooled, confluent near-misses (+0.336) are **no better** than non-confluent
   (+0.340); on the two largest families (missC4, missC5) confluence is flat-to-negative; the by-score curve is flat.
   The two "R2 ✓" families (missC1, missC5) show differences of +0.02–0.06R inside wide CIs, contradicted by the flat
   score curve — the pre-registered global rule (pooled R1&R2 **and** ≥2 clean families) returns **NO RESCUE**.

**Why this is the stronger result:** batch-7 showed confluence doesn't sort *selected* fires. The near-miss audit
shows confluence doesn't sort *marginal* fires either — the exact frame-independence WI's fairness objection asked
us to test. **Confluence is neither a narrowing tool (batch-7) nor a widening tool (this audit).**

**Binding honesty caveats:**
- **Survivorship + trend beta** (add.64): 476 *current* S&P constituents are survivors; the isolated single-position
  sim enumerates 124k overlapping entries in 2018–2026 uptrends. Much of the uniform +0.28–0.45R is **long-in-a-
  survivor-uptrend beta**, not door edge. But that cuts *against* confluence even harder: if everything drifts up,
  confluence still fails to sort it. It does **not** rescue the confluence hypothesis.
- **Isolation inflates counts** (15k "real" vs the door's 2,622 one-position fires) and removes crowd-out; the unit
  here is per-event R comparability, not a tradeable book.
- **R1 (CI-lo>0) is a weak bar** on a net-positive book (the add.61 lesson) — R2 and the flat score curve are the
  real evidence, and they fail.
- **Compensatory-logic flag (Lesson #54):** had this passed, adoption would have required a strictly-bounded
  two-condition rule (exactly-one-check-missed AND score≥2), never a weighted score. It did not pass; no such rule
  is proposed.

---

## 4. FOR WI — please mark each layer

For every layer in §1, please mark one of:
- **AGREE** — our rule matches your intent closely enough that a null on it is a null on *your* method.
- **WRONG** — our mechanization diverges from what you do; **please state the right rule** (the number, the object,
  the sequence) so we can re-test the corrected version.
- **CAN'T SAY WITHOUT CHART** — you need to see the specific dated case (we will send the chart) before judging.

| # | Layer | Our risk grade | Most likely to be OUR fault? |
|---|-------|:--------------:|:----------------------------:|
| 1 | Structural range + location | MED | rolling-40d eye box ≠ your drawn W/M range |
| 2 | Break/acceptance | MED | "2 daily closes" vs your (weekly?) acceptance |
| 3 | Retest / LPS entry | **HIGH** | **"LPS+Bojan candle + minor structure break" reduced to "any hold in an ATR zone" — our fire set is a SUPERSET of yours. Sharpened question: is the entry specifically the LPS+Bojan candle (rejection wick + close location) that ALSO produces a minor market-structure break, or is any daily close that holds above the broken level acceptable?** |
| 4 | Stops | MED | post-TP1 → breakeven vs your "under created LPS" |
| 5 | TP/derisk/runner | MED-HIGH | 60% runner vs your 1/10 core; our TT formula |
| 6 | Campaign topology | **HIGH** | **no pyramiding-while-positioned (your 1–3 entries/model)** |
| 7 | Bojan/wick anatomy | **HIGH** | **1-candle wick% ≠ your M1/M2 candle-pair + zone** |
| 8 | Sweep-retest | MED | depth band ours; reclaim-speed/location lost |
| 9 | Freshness | MED | F=10 (and it loses; is it calendar or structure-validity?) |
| 10 | Origin/discount | MED-HIGH | 365d rolling span ≠ your drawn range's discount |
| 11 | Chop-count veto | **HIGH** | **fully-invented "3 fails / 90d"; how do you read chop?** |
| 12 | Climax veto | MED | fully ours; barely fires |
| 13 | Gann time | **HIGH** | **"major anchor" un-mechanizable without your discretion** |
| 14 | Dominance | **HIGH** | **your "main driver" is ABSENT on the stock/metal test universe** |
| 15 | M1 spring | LOW | correctly retired; likely not your edge anyway |

---

## 5. Bottom line

**The single highest-probability place we are still looser than you is the entry definition itself
(Layer 3 + Layer 7). If your real trigger is specifically the LPS+Bojan candle (with a minor market
structure break) rather than any hold above the level, then batch-7 tested confluence on a broader
fire set than you would take.**

- **The highest-risk (most-likely-our-fault) translation rows are #3 (LPS+Bojan entry reduced to "any hold"),
  #6 (no pyramiding), #7 (single-candle Bojan), #11 (invented chop-count), #13 (un-mechanizable Gann anchor), and
  #14 (dominance absent on the batch-7 universe).** These are the rows where a "WRONG — here's the real rule" from
  you could change what we test. Nulls on #7/#11/#13/#14 are honestly **weak evidence about your method** — they are
  tests of *our guess* at your judgment.
- **But the batch-7 confluence null is NOT one of those weak cases.** The forming-bar diagnostic shows the door's
  entry clock is daily (not penalized by completed-bar waits), and the near-miss audit shows the confluence null is
  **frame-independent**: confluence sorts neither selected nor marginal retests, and the marginal retests are already
  as good as the clean ones. **On this mechanized door, additional confluence does not select better trades — the
  structural break-retest is already the whole edge.**
- **The one thing that could still overturn the null** is a corrected translation of the *entry object itself*
  (row #3 / #7): if your real entry is specifically the LPS-Bojan candle into a pre-drawn zone (not "any daily close
  that holds"), then we tested confluence on a **looser** fire set than yours, and the right experiment is to first
  tighten the entry to your definition and re-test confluence on **that**. That requires your row-by-row correction
  above. Everything else — cadence topology, Gann, dominance-on-stocks, single-candle Bojan — is either a known
  structural gap we can name, or a layer we cannot mechanize without your discretion.

*Files (all study-only; production untouched): this ledger; `idea_lab/forming_bar_sensor.py`,
`idea_lab/run_forming_bar.py`, `idea_lab/run_near_miss.py`; result JSONs `forming_bar_results.json`,
`near_miss_results.json`; `wyckoff_audit.md` addendum 66.*
