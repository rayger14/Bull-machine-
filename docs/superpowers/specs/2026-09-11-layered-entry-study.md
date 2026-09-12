# Layered entry study v1 — frozen selection before outcomes

User approved broader chronological sampling and emphasized minute data within
larger structure. Research only; no live orders, production changes, optimizer,
new libraries or provider setup. Previous four cases/responses remain unchanged.

## Selection and sources

Use complete March, May and July 2026 from the existing Binance minute archive
SHA256 `5b8a4533f70b8ccd0dc533984469e6886ec73ce315d48f150c667ccb97e84035`.
These alternating months span the recorded live period and have complete prior
history/tails; chosen before candidate outcomes, not by performance. They are
reused research, not pristine holdout or certified regime coverage.

For each month independently replay from exactly 30 days before month start,
through month end exclusive. Derive complete hourly bars from that SAME minute
stream and run existing guarded LFC/native signal replay with all 17 archetypes.
No derivative or macro observations are invented. This is source-code signal
replay under a cold-start convention, not the original live state or book.

In each half-month [1st,15th), [15th,next month), select the first hourly LC
candidate that passes LC identity, finite numeric hard gates (volume z>=3,
RSI>65 or <35, BB<=.06, chop<=.5), and immediately prior completed-hour BB<=.06.
Do not condition selection on native emission or future price. Independently,
select the first chronological minute detector event in each half-month using
the existing causal detector on that month's 30-day-seeded stream. Do not filter
minute selection by parent permission. At most six hourly and six minute cases.
Membership is decision timestamp; absent strata stay absent, never replaced by
outcome. Invalid/missing evidence stays reported and is not silently replaced.
Tie order is decision timestamp then stable event ID; minute detector internally
orders reclaim index then pivot index. LC identity is the existing native
structural `_check_E` evaluation used by the replay. Each monthly detector starts
fresh with its original empty spacing state at the 30-day seed; prefix checks
use exactly that same seed/state, not a cropped local detector restart.

## Layered packet and separation of duties

Every selected decision receives only completed same-stream candles: last30 1m,
last12 5m, last16 15m, last24 1h, last18 4h and last10 daily candles. Require exact
grid/constituent completeness; never backfill a partial higher-timeframe bar.
Native hourly feature timestamps/values retain their own provenance. Numeric
gate checks are supplied as evidence, not a claim of profitable entry.

Minute packets include deterministic, timestamped witnesses for confirmed
31-bar pivots, prior matching touches, first sweep, first reclaim, sweep minimum
and chronological detector spacing. Compute witnesses from predecision data;
verify selected membership by an independent prefix detector run. Include local
OHLC values and availability times, not just pass labels. A confirmed pivot
requires the fifteenth right bar's CLOSE, not its open.

Provide fixed 4H/N3 and daily/N3 parent snapshots strictly before setup (hourly:
setup hour open; minute:first sweep open), plus lifecycle updates through
decision. Do not pick the better parent variant. Daily parent is contextual;
minute mechanical H3 baseline remains 4H/N3. Explicitly distinguish verified
absent parent from unknown/missing evidence. Preserve source update cadence:
hourly parent ledger is not a claim of intraminute structural survival.

Separate rulecards: hourly LC hard gates/H2 only apply hourly. Minute detector
does NOT require RSI, BB compression, volume z or H2. Its sweep/reclaim identity
is distinct from optional/contextual parent filtering. Agents must cite larger
structure, location, sequence, entry, fixed invalidation and room/obstacles.
No live receipt certificate is required for a conditional research judgment;
absent macro/flow inputs must not become invented evidence.

One fresh context-free assessor per CASE, at most twelve calls, with responses
capped at 300 words. A later packet can reveal an earlier outcome through its
historical layers, so no assessor sees multiple cases. No critique/retry before
or after outcomes. Accept/reject/insufficient are all allowed, no forced mix.
Give exact rulecards and pertinent code, withhold native emissions, mechanical
parent masks and all outcomes. Remove outcome/performance-bearing comments,
metadata and filenames from visible code/packets. Lock all responses before any
outcome reveal.
Report scope mistakes and abstentions, not just schema validity.

## Frozen scoring

Reuse `entry_case_outcome.score_case` unchanged. BOTH tracks score on same-source
1m OHLC for finer barrier ordering. Entry at decision-minute open, zero added
model delay, no assumed obtainable live fill. Hourly stop = prior setup close
minus 2.7 native replay ATR; minute stop = sweep minimum*0.9985. Target = actual
entry +2*(entry-stop). Horizons 1440min hourly /240min minute, $50k independent
notional, $60 round-trip costs, unspecified account equity; no book, funding,
impact or compounding. Invalid stop>=entry stays invalid with no bracket PnL.
Known opening gaps precede intrabar extremes; otherwise both-hit stop-first and
ambiguous. Deadline OPEN only; full-horizon MFE/MAE continue after exit.

Show every case including rejected/abstained counterfactuals. Compare hourly
native signal emission, numeric selection baseline, and agent; minute detector,
4H/N3 H3, and agent. Hourly H3 floor-reclaim subtype is NOT a universal gate.
Show per-track/per-month coverage, abstentions, dollars and average initial risk;
no pooled portfolio/Sharpe/significance or superiority claim from tiny counts.
Hourly numeric checks define the selected universe and therefore accept all
selected hourly cases; their agreement is not independent corroboration. Report
available strata, selected cases, assessable cases and accepted valid plans as
distinct denominators. Missing input evidence, verified parent absence, invalid
plans and missing outcome data remain separate statuses. Missing minute bars or
deadline tails block affected scoring, never shorten a horizon or replace a case.

## Verification and September 8

Test exact close boundaries, gaps, no future influence, malformed witness values
and prefix detector identity before historical packets. Freeze source/code,
protocol, candidate IDs, packets and hidden masks; lock model responses before
revealing prices. Record model snapshot/usage as unavailable if not exposed.
Reproduce packets/outcomes and independently review claims before handoff.

September8 LC is a separate known-success diagnostic, excluded from blinded
comparison. Search existing related folders for minute coverage. If unavailable,
report the missing layer honestly; do not fabricate minute bars, splice CME
futures into Binance, or pass hourly-only evidence off as multilayer validation.
