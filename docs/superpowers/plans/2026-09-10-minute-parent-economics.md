# Minute Parent Economics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compare baseline and four frozen parent permissions using independent reference minute sleeves, then execute the registered June experiment.

**Architecture:** One pure comparison wrapper reuses the unchanged simulator, preserving its fills/accounting. Parent reconstruction and historical input acquisition stay outside the wrapper and are independently verified by the controller before the economic run.

**Tech Stack:** Existing Python, pandas/numpy, pytest; standard-library JSON/statistics. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-10-minute-parent-economics-design.md`

## Global Constraints

- No production `engine/`, `bin/`, or `configs/` changes, live settings/orders/deployments, public pushes, optimizer, or dependency installation.
- Keep the existing local research branch; preserve unrelated graph outputs.
- Compare baseline and all four existing parent policies: `4H:3`, `4H:5`, `1D:3`, `1D:5`. No variant is selected or modified from counts or outcomes.
- Long next-open reference, zero delay, $50,000 fixed notional, 0.0015 stop buffer, 240-minute fixed holding/lockout and 12bps round-trip cost. Starting equity unspecified.
- Unknown permission cannot enter. Complete event identity joins required; no silent missing/extra evidence or truthiness.
- Output deterministic JSON-safe, certified false. No inferential interval, economic-readiness claim or cross-timeframe profit ranking.

---

### Task 1: Pure independent-sleeve comparator

**Files:**
- Create: `scripts/research/minute_parent_comparison.py`
- Test: `tests/research/test_minute_parent_comparison.py`
- Reuse unchanged: `scripts/research/minute_sweep_validation.py`

**Interfaces:**
- Consumes: minute OHLC frame, raw locally indexed selector events, complete four-variant event-ID permission maps and aware aligned window bounds.
- Produces: `compare_minute_parent_arms(bars, events, permissions, *, window_start, window_end)` with five arms, each containing `simulation`, complete `event_ledger`, `summary`, and `comparison`. Comparison keys: `shared_entered_ids`, `baseline_only_entered_ids`, `arm_only_entered_ids`, `status_transitions` (list of `{baseline_status, arm_status, count}` records). Summary extends unchanged simulator summary with `completed_fees`, `completed_gross_pnl`, `mean_net_pnl_over_initial_risk`, `median_net_pnl_over_initial_risk`. Top-level `parameters`, `coverage`, `arms`, `limitations`, `certified`.

- [ ] **Step 1: Write behavior-first fixtures and tests**

Use real simulator calls, not a mocked simulator. A flat 720-minute synthetic frame allows early and later candidates; returned trades and event statuses make filtering-vs-lockout order observable.

```python
import pandas as pd
from scripts.research.minute_parent_comparison import compare_minute_parent_arms

def fixture():
    bars = pd.DataFrame({'open': 100., 'high': 101., 'low': 99., 'close': 100.},
                        index=pd.date_range('2026-01-01', periods=720, freq='min', tz='UTC'))
    events = [dict(pivot_idx=1, confirmed_idx=16, sweep_idx=18, reclaim_idx=20,
                   level=98., sweep_low=95., touches=2),
              dict(pivot_idx=80, confirmed_idx=95, sweep_idx=98, reclaim_idx=100,
                   level=98., sweep_low=95., touches=2)]
    ids = ['reclaim:' + bars.index[e['reclaim_idx']].isoformat()
           + '|pivot:' + bars.index[e['pivot_idx']].isoformat() for e in events]
    permissions = {name: {key: True for key in ids}
                   for name in ('4H:3', '4H:5', '1D:3', '1D:5')}
    return bars, events, ids, permissions

def test_permission_rejection_frees_later_busy_candidate():
    bars, events, ids, permissions = fixture()
    permissions['4H:3'][ids[0]] = False
    result = compare_minute_parent_arms(bars, events, permissions,
        window_start='2026-01-01T00:00:00Z', window_end='2026-01-01T12:00:00Z')
    base = result['arms']['baseline']
    arm = result['arms']['4H:3']
    assert [row['status'] for row in base['event_ledger']] == ['completed', 'skipped_busy']
    assert [row['status'] for row in arm['event_ledger']] == ['permission_rejected', 'completed']
    assert arm['comparison']['arm_only_entered_ids'] == [ids[1]]
    assert arm['comparison']['baseline_only_entered_ids'] == [ids[0]]
```

Add concrete tests for all-permit whole-simulator equality; unknown/no-permit/empty arms; hand-derived $60 fee and risk ratio; same-entry-bar and gap stops; early-stop fixed lockout; censored/unfilled tails; rejected out-of-window events; missing/extra variant or event identity; bool/string/numeric permission strictness; malformed index/clock/container; copied-input repeatability/nonmutation and strict JSON. Each names the implementation error it catches. Do not test asserted mock interactions or only duplicate construction details.

- [ ] **Step 2: Observe RED**

Run `python3 -m pytest tests/research/test_minute_parent_comparison.py -q -o addopts=''`. First missing import establishes absent API; after adding only a minimal importable stub, rerun and record assertion RED before implementing the comparison.

- [ ] **Step 3: Implement the minimal wrapper**

Validate all identities/windows/permission maps before economic calls. Work on copies and derive event IDs exactly as the simulator does. Execute baseline once and each arm on its permitted event list:

```python
simulation = simulate_events(
    bars, selected_events, entry_mode='next_open', notional=50000.,
    stop_buffer=.0015, hold_minutes=240, cost_bps=12., decision_delay_seconds=0)
```

Join simulator ledger back onto every original ID, using explicit permission statuses for omitted events. Build entered-ID sets from completed/open_censored states; compare all event status pairs. Compute new completed-only summary fields from actual simulator trades: sum fees, net plus fees, mean/median `pnl / initial_risk`; empty ratios are null. Preserve simulation objects semantically, including all censored positions. Output fixed parameters and explicit unfunded/conditional/flat-start/tail limitations. `json.dumps(result, allow_nan=False)` must succeed or raise ValueError for nonrepresentable output. No CLI, parent recomputation, new exit policy, source mutation, optimizer, or extra simulator is needed.

- [ ] **Step 4: Observe GREEN and review**

Run focused command above, then `python3 -m pytest tests/research -q -o addopts=''`, then `git diff --check`. Record exact RED/GREEN evidence in the task report. Independently review spec compliance and code quality before running historical outcomes.

- [ ] **Step 5: Commit the owned source/test pair**

`git add scripts/research/minute_parent_comparison.py tests/research/test_minute_parent_comparison.py` then `git commit -m "Add frozen minute parent sleeve comparison"`.

## Controller experiment and handoff

After Task 1 review: load and rehash the four frozen sources specified in the design; join all76 original events/annotations using exact IDs, clocks and raw values; slice same minute stream from June1 through June20 04:00 inclusive and rebase all four index fields. Run the wrapper, direct baseline and direct filtered candidate simulations; require complete equality, fresh-copy equality and nonmutation. Save private JSON with hashes, checks and all five arms, and document every arm's outcomes without selecting a winner. Independently review economic interpretation and then run fresh full research tests. Keep the branch local. Archive only this plan's workspace recoverably.
