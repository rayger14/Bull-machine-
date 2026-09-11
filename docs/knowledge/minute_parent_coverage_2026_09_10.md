# Minute parent coverage and retrospective schedule

## Frozen scope

Coverage-only research probe: no new trade simulation, PnL, risk allocation or strategy selection. Starting equity, average trade risk and trading costs are not applicable to this annotation audit. It follows the independently reviewed [availability-versus-geometry comparison](minute_parent_availability_2026_09_10.md).

Before new parent counts, the delegated independent quant approved four deterministic windows: the latest four fully available quarterly-start months. All use the same **30 calendar days** of independent parent prehistory; they will not be replaced because a variant has sparse coverage. This is not the previous-calendar-month convention, demonstrated sufficient warmup, or a continuous-history replay.

| Candidate window, UTC/end exclusive | Parent cold start | Parent input hours | Frozen candidates |
|---|---|---:|---:|
| October 1–November 1, 2025 | September 1, 2025 | 1,464 | 236 |
| January 1–February 1, 2026 | December 2, 2025 | 1,464 | 230 |
| April 1–May 1, 2026 | March 2, 2026 | 1,440 | 242 |
| July 1–August 1, 2026 | June 1, 2026 | 1,464 | 233 |

Total: **941 frozen candidates**, not 941 executed positions. Month membership uses reclaim-bar-open in the half-open window. Child events retain their full-history selection, original indices, values and 60-minute spacing. They are not redetected inside these windows.

Parent construction uses complete same-source hours through the candidate month's final hour close, below the existing 2,048-hour cap, with unchanged recovered-source hashes and parameters. TA-Lib ATR14 starts at the declared seed; the first 14 unknown ATR values remain unfilled. The new experiment/ATR identity explicitly records independent 30-day cold starts. July shares the physical June 1 seed with the older reference, but the new convention is separately identified.

All four 4H/1D × N3/N5 variants retain nested controls: A validated strict pre-sweep binding; L A plus no bound-lineage break through decision; G the unchanged frozen geometry/lifecycle policy. Unknowns remain nonpermitted and visible. Sparse or absent parents do not remove candidates from coverage denominators. Receipt authenticity and trader fidelity remain uncertified.

## Whole-source inventory

Saved minute source spans **January 1, 2021 00:00 through August 31, 2026 23:59 UTC**, with 2,979,360 unique, ordered, contiguous minute rows. All 68 calendar months are complete. Native numeric OHLC and finite nonnegative volume pass validation.

The **15,910** frozen candidates reconcile completely: unique identities, original source values, pivot/confirmation/sweep/reclaim chronology, first sweep/reclaim conditions and selector spacing. This does not rerun the full detector or independently certify all original clustering predicates.

Sixty-seven months have the conservative complete post-month tail for the frozen next-open/240-minute contract. August 2026 does not: two actual candidates lack a deadline bar at `reclaim_idx + 241`. They remain reported, not silently removed. A complete calendar month is not necessarily a complete execution tail. Every one of the four selected months has full candidate and conservative tails.

The other **64 months have source/candidate/tail inventory only**, not deep parent annotation. The selected months are a bounded feasibility/stability schedule, not demonstrated representation of all regimes and not pristine holdout.

Private calendar inventory: `results/research_validation_2026_09_10/minute_parent_coverage/calendar_inventory.json`, SHA256 `2e6627de97ca1df24dcc913ce9eef6099397f5f7981eae8b414349d048956ceb`. Full source SHA256 `5b8a4533f70b8ccd0dc533984469e6886ec73ce315d48f150c667ccb97e84035`; frozen baseline SHA256 `6ed3caee375d8a023dd29173d92e3863b6e355dbd5591a7f8a9899fbbe74ea8a`.

## Parent coverage results

All four month runs completed with **3,764 annotations across 941 unique candidate events**, zero unknowns and no malformed-evidence flags. A/L/G counts below are candidate permissions, not trades.

| Month | Variant | A available | L unbroken | G full rule | Absent parent | Bound lineages | G-permitted days |
|---|---|---:|---:|---:|---:|---:|---:|
| Oct 2025 | 4H/N3 | 179 | 179 | 116 | 57 | 15 | 25 |
| Oct 2025 | 4H/N5 | 177 | 177 | 109 | 59 | 17 | 19 |
| Oct 2025 | 1D/N3 | 174 | 174 | 158 | 62 | 3 | 21 |
| Oct 2025 | 1D/N5 | 163 | 163 | 147 | 73 | 2 | 20 |
| Jan 2026 | 4H/N3 | 174 | 174 | 136 | 56 | 20 | 24 |
| Jan 2026 | 4H/N5 | 158 | 158 | 110 | 72 | 8 | 19 |
| Jan 2026 | 1D/N3 | 173 | 172 | 107 | 57 | 12 | 19 |
| Jan 2026 | 1D/N5 | 138 | 137 | 39 | 92 | 15 | 8 |
| Apr 2026 | 4H/N3 | 178 | 178 | 120 | 64 | 15 | 21 |
| Apr 2026 | 4H/N5 | 179 | 179 | 138 | 63 | 11 | 22 |
| Apr 2026 | 1D/N3 | 142 | 142 | 70 | 100 | 7 | 10 |
| Apr 2026 | 1D/N5 | 129 | 129 | 63 | 113 | 1 | 9 |
| Jul 2026 | 4H/N3 | 163 | 163 | 71 | 70 | 24 | 19 |
| Jul 2026 | 4H/N5 | 173 | 173 | 77 | 60 | 15 | 16 |
| Jul 2026 | 1D/N3 | 214 | 214 | 26 | 19 | 3 | 7 |
| Jul 2026 | 1D/N5 | 232 | 232 | 34 | 1 | 1 | 9 |

Unlike the prior ten-day June sample, daily geometry now changes selection in every scheduled month. Its coverage varies greatly: July daily/N5 binds 232 of 233 candidates but permits only 34. Neither a high nor a low permission rate indicates profit.

Daily/N5 April and July each have just **one bound lineage**, and all their G-permitted events remain within that lineage. Large event counts do not supply correspondingly many independent higher-timeframe contexts. This is a reason to report month/parent concentration explicitly, not to discard or replace the periods.

One unique child event, reclaim January 25 at 00:05 UTC, has a bound-lineage break under both daily variants. Thus January A and L differ by one candidate for each. Two annotations of that event are not two independent break examples. Three 4H/N3 annotations have supersession (October 25 and January 1/28); supersession alone does not veto: the October event fails geometry, while both January events pass the frozen geometry. No rule was changed to accommodate these cases.

## Strict availability reporting correction

The quant reviewer identified that an hourly post-transition snapshot is not the same as a parent visible **strictly before** a minute-open decision. The separate lightweight auditor now evaluates every minute-open query in each candidate month with `searchsorted(transition_available_at, minute, side='left') - 1`, including the preceding transition at month start. It also reconciles every candidate binding against that strict transition lookup.

This changes reported available-parent day counts in two cases: October daily/N3 is **24**, not 23; January 4H/N5 is **27**, not 26. At a midnight break the prior parent can still be strictly bindable at exactly 00:00. These counts are minute-grid query points/dates, not duration-integrated continuous-time exposure. The original raw fields are retained for provenance, but `coverage_audit.json` supersedes their interpretation: `available_parent_state_hours` becomes explicitly **active hourly transition snapshots**, not elapsed availability hours.

The final daily tables include every calendar date, even zero-candidate dates, with candidate/A/L/G/absent/unknown counts, exact strict available-minute counts, and bound/G-permitted version and lineage diversity. Available-parent days and bound-candidate days are separately defined; the latter follows the candidate's reclaim date even though binding occurs at first sweep.

## Verification and artifacts

Root fresh full research suite: **359 passed**, one existing urllib3/LibreSSL warning, 13.99 seconds. No production `engine/`, `bin/`, `configs/`, reusable research source or tests changed from phase base `9663134`. All new scripts are throwaway private audit reproductions, not supported engine features.

Each month verifies exact complete-hour inputs, frozen source/runtime/ATR versions, unique frozen event joins and causal clocks, strict parent availability, G⊆L⊆A, exact repeated parent-build and complete repeated annotation equality, input nonmutation, and two event checks per configuration against a freshly rebuilt 15-day prefix. Prefix checks are representative, not an exhaustive all-prefix or continuous-restart proof. The lightweight auditor independently reconciles all 3,764 annotation/control/binding records, full-calendar daily/month totals, and diversity counts. Unknowns remain fail-closed; none occurred.

Private artifacts under `results/research_validation_2026_09_10/minute_parent_coverage/`:

| Artifact | SHA256 |
|---|---|
| `2025-10_parent_coverage.json` | `7a2b96a67a3963d8e45adc8a8473e15bc893980bdf6e3cdd63694078e39ef871` |
| `2026-01_parent_coverage.json` | `5c2319bca46e60509b3be863bc35635e5fc31f6fff7d666a85806a1fa6f0ebeb` |
| `2026-04_parent_coverage.json` | `c45a1f13e144b8cd4d1ee84756fa5436ac801300f92c3aa23cddd78278e2a15d` |
| `2026-07_parent_coverage.json` | `a7ec5f65b22307c004811400226da30ac386d678a45db285a9236b97fdc0c192` |
| `coverage_audit.json` | `a26ba777e01bfb13800b8245adb60865dbc3d29dddad36fe07944ad95498800b` |

These retain manifests, original indices, event/evidence/contract IDs, complete parent ledgers and annotations, control masks, daily diversity and script hashes. `inventory.py`, `parent_probe.py` and `audit.py` reproduce the respective artifacts from repository root with `PYTHONPATH=.`; parent probe additionally takes one of the four fixed `YYYY-MM` arguments. Existing unequal artifacts are not overwritten.

## Review and next step

Independent final result review **approved local coverage research with no blocking findings**. The reviewer reproduced the exact auditor hash and independently verified all 3,764 bindings, referenced geometry, lifecycle/supersession intervals and nested masks. A separate interval-intersection algorithm reproduced every daily strict-minute availability count, including the corrected boundary dates. Root reran the auditor with identical output and checked the reported unique lifecycle/supersession cases against retained records.

No variant is promoted, no broader PnL was computed, and no months were selected/replaced based on these coverage results. The quant approved readiness for a separately frozen economic experiment on these exact four months and configurations: **one baseline plus twelve A/L/G filtered sleeves per month, 52 reports total**. Before execution, materialize/hash the control maps and freeze the unchanged next-open/stop/240-minute/cost contract. Each month/sleeve starts flat without inherited lockout; retain full tails and replay independently after candidate filtering. No stitching into a continuous equity curve or pristine-holdout claim. Report month-level results and parent concentration even if descriptive aggregates are also shown.

The concurrent hourly read-only audit identified a candidate fixture-only outer-cutoff intervention, summarized in [hourly cutoff readiness](hourly_cutoff_experiment_readiness_2026_09_10.md). It is advisory, not implemented or a substitute for exact trace/executable-fill validation. Hourly and minute remain separate, equally important tracks.
