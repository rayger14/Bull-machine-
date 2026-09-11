# Fusion Live Scorecard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reproducibly describe logged fusion/threshold associations and evidence gaps without calling selected recorded exits a strategy backtest.

**Architecture:** One pure module validates JSON snapshots, retains explicit-ID exit groups, separates open inventory and exclusions, and returns descriptive summaries and rank correlations. Controller owns acquisition, frozen report generation and research interpretation.

**Tech Stack:** Python3.9, existing pandas, pytest, standard library. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-10-fusion-live-scorecard-design.md`

## Global Constraints

- No production `engine/`, `bin/`, or `configs/` changes, live setting changes, orders, deployments, public pushes, weight optimization, or new execution dependencies.
- Keep the existing local research branch in place. Private snapshots and generated reports stay under ignored results.
- Preserve all 17 archetypes; hourly and minute research remain separate and equally important.
- No claim of broker-verified fills, full net returns, complete candidate capture, or historical source-version parity.
- Never derive account drawdown, equity return or Sharpe from these rows.
- `certified=False` and `completion_certified=False`; descriptive correlations, no learned cutoff or inferential intervals in this unit.

---

### Task 1: Pure recorded-exit fusion scorecard

**Files:**
- Create `scripts/research/fusion_live_scorecard.py`
- Create `tests/research/test_fusion_live_scorecard.py`
- Read spec above in full; it defines every required field, quarantine/coverage rule and metric.
- Read `scripts/research/replay_clock.py` only if reusing its `utc`, `json_safe`, `digest` helpers.

**Interfaces:**

```python
def build_fusion_scorecard(trades, *, open_positions, signal_rows, archetypes, snapshot_meta):
    """Return deterministic JSON-safe descriptive evidence; no side effects."""
```

`snapshot_meta` requires aware `server_time` and `heartbeat_updated_at`, plus caller-supplied `source_hashes` mapping retained in output. `archetypes` is a nonempty list of unique nonempty strings. Other inputs are lists. Return keys: `certified`, `limitations`, `snapshot_meta`, `coverage`, `groups`, `quarantined_groups`, `excluded_rows`, `open_inventory`, `summary`, `by_archetype`, `by_direction`, `by_month`, `by_source_version`, `signal_coverage`. Metric/subrecord names may follow the spec vocabulary; document exact names in report so controller can consume them. Include original row indices for all grouped/quarantined/excluded rows, with no row in two categories. A nonempty ID with any invalid exit row quarantines every row for that ID. Open inventory may have missing score metadata, which stays unavailable. No CLI/network/loading pipeline is part of this task.

- [x] **Step 1: Write hand-derived failing tests.** Begin with the public interface and a concrete partial-exit grouping test; implement only after RED. Use a complete synthetic exit fixture with position ID p1, archetype test_long, direction long, entry2026-01-01T00:00Z at100, stop90, score0.30, threshold0.25, margin0.05, source_version epoch1. Two exits at01:00Z and02:00Z, each quantity1, pnl_usd/pnl10 and-5. Literal expected group subtotal5, quantity2, risk proxy20, ratio0.25, duration2h, one winning group (not two observations). Snapshot server2026-01-02T01:00Z, heartbeat2026-01-02T00:00Z. Add a second group p2: short entry100 stop110 quantity1 pnl-5, score0.2 threshold0.3 margin-0.1. Overall two groups, subtotal0, recorded-subtotalPF1, one positive/one negative; each margin cohort n1. Assert zero counts for an additional supplied archetype.

```python
def test_partial_exits_are_one_observation():
    # Use complete literal fixtures described above, not production expected-value helpers.
    report = build_fusion_scorecard(two_exits_p1, open_positions=[], signal_rows=[],
                                  archetypes=['test_long', 'unused'], snapshot_meta=meta)
    assert report['certified'] is False
    assert len(report['groups']) == 1
    assert report['groups'][0]['recorded_exit_pnl_usd'] == 5.0
    assert report['groups'][0]['quantity_sum'] == 2.0
    assert report['groups'][0]['recorded_pnl_over_displayed_stop_risk_proxy'] == 0.25
```

Before each added behavior, add and observe its failing test: blank IDs, duplicates/group conflicts, invalid row contamination, clocks and UTC equivalence, numeric/sentinel handling, stored-vs-display threshold distinction, margin-rounding ambiguity, pnl alias mismatch, open matching/closure limitations, malformed inventory, short risk, source version coverage, tie/constant correlations, empty groups/zero losses, reproducibility/nonmutation. Tied-rank fixture scores[1,1,2] and outcomes[1,2,3] has Spearman sqrt(3)/2, not1. Test a negative monotonic relation=-1 and constant operands=None. Tests must exercise public output; do not assert private helper structure.

- [x] **Step 2: Observe RED.**

```bash
python3 -m pytest tests/research/test_fusion_live_scorecard.py -q -o addopts=''
```

Record initial import failure, then behavior failures after the interface exists. Import failure alone is insufficient proof of numeric/validation branches.

- [x] **Step 3: Implement pure behavior.** Decompose within the module into validation/grouping, group derivation, summary/rank metrics and signal coverage. Use finite type checks (excluding booleans/strings), copied records, aware UTC clocks, deterministic sorting and JSON-safe nulls. Never replace entry threshold with factor display threshold or derive missing score from margin. Preserve the exact contracts in the spec. Keep implementation scope to this source/test pair; if spec ambiguity is load-bearing, ask controller before inventing behavior.

- [x] **Step 4: Verify GREEN and self-review.**

```bash
python3 -m pytest tests/research/test_fusion_live_scorecard.py -q -o addopts=''
python3 -m pytest tests/research -q -o addopts=''
git diff --check
```

Run focused suite while iterating and full research suite once before commit. Baseline271passed with one existing urllib3/LibreSSL warning. No need to read actual outcome rankings; real snapshot run is controller-owned after independent review.

- [x] **Step 5: Commit only owned source/tests and report.**

```bash
git add scripts/research/fusion_live_scorecard.py tests/research/test_fusion_live_scorecard.py
git commit -m "Add descriptive live fusion scorecard"
```

Write full TDD/evidence report at the supplied SDD task-report path; return short status/commit/test counts/concerns. Include public output keys/field naming for controller. Do not spawn subagents. Controller handles independent task/final review, actual report generation and documentation.
