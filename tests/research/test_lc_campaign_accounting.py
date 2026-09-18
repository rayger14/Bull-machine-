from copy import deepcopy
import hashlib
import json

import pytest
import pandas as pd


def clock(minutes):
    return (pd.Timestamp("2026-01-31T23:58:00Z") + pd.Timedelta(minutes=minutes)).isoformat()


def bars(count=8):
    return pd.DataFrame(
        {"open": [100.0] * count, "high": [101.0] * count,
         "low": [99.0] * count, "close": [100.0] * count},
        index=pd.date_range(clock(0), periods=count, freq="min"),
    )


def full_plan(decision=0, **changes):
    plan = {
        "decision_time": clock(decision), "action": "enter", "stop": 95.0,
        "level": None, "entry_expiry": clock(decision + 15),
        "exit_deadline": clock(decision + 1440), "processing_seconds": 0,
        "routing_seconds": 0, "notional": 50000.0, "cost_bps": 12,
    }
    plan.update(changes)
    return plan


def replay(one_bars, plan, cutoff=5):
    from scripts.research.conditional_occupancy import replay_sleeve

    candidate = {"candidate_id": "case", "track": "hourly",
                 "decision_time": plan["decision_time"], "plan": plan,
                 "unavailable_reason": None}
    return replay_sleeve(one_bars, [candidate], track="hourly", as_of=clock(cutoff))


def campaign_case(identity="case", decision=0):
    immediate = full_plan(decision, processing_seconds=90)
    wait = full_plan(decision, action="wait_close_above", level=101.0, processing_seconds=90)
    reject = full_plan(decision, action="reject", processing_seconds=90)
    return {
        "case_id": identity, "track": "hourly", "decision_time": clock(decision),
        "source_close": 100.0, "atr14": 5.0 / 2.7,
        "plans": {"immediate": immediate, "wait_5m_high": wait, "reject": reject},
        "facts": {"hourly_subtype": "synthetic"},
    }


def contract_manifest(ids):
    return {
        "schema_version": "lc_campaign_contract_v1", "campaign_id": "synthetic",
        "cases": [{"case_id": identity, "job_directory": "/synthetic/" + identity,
                   "source_request_sha256": "1" * 64,
                   "role_request_sha256": "2" * 64} for identity in ids],
    }


def accounting_manifest(cases, matched_ids=None, as_of=1440):
    matched_ids = [case["case_id"] for case in cases] if matched_ids is None else matched_ids
    return {
        "schema_version": "lc_campaign_accounting_v1",
        "campaign_manifest": contract_manifest(matched_ids),
        "as_of": clock(as_of), "broad_cases": cases, "matched_case_ids": matched_ids,
    }


def terminal_state(manifest, terminal_by_id, elapsed_seconds=90, timing_valid=True):
    from scripts.research.assessment_evidence_guard import _canonical

    def digest(value):
        return hashlib.sha256(_canonical(value).encode("ascii")).hexdigest()

    cases = {}
    for identity in terminal_by_id:
        attempt_id = digest({"case_id": identity, "role": "specialist", "started_at": 0})
        path_body = {
            "schema_version": "lc_campaign_decision_path_v1", "manifest_sha256": digest(manifest),
            "case_id": identity, "source_request_sha256": "1" * 64,
            "role_request_sha256": "2" * 64, "specialist_attempt_id": attempt_id,
            "start": {"runtime_id": "run", "wall_ns": 0, "monotonic_ns": 0},
            "end": {"runtime_id": "run" if timing_valid else "restarted",
                    "wall_ns": elapsed_seconds * 10**9,
                    "monotonic_ns": elapsed_seconds * 10**9},
            "terminal_sha256": digest(terminal_by_id[identity]), "terminal_valid": True,
            "reason": None if timing_valid else "timer_runtime_changed",
            "timing_valid": timing_valid,
            "elapsed_ns": elapsed_seconds * 10**9 if timing_valid else None,
        }
        path = dict(path_body, path_sha256=digest(path_body))
        cases[identity] = {"roles": {
            "specialist": {"status": "delivered", "attempt_id": attempt_id,
                           "started_at": 0, "deadline_at": 600, "finished_at": 1,
                           "result": {"kind": "delivered",
                                      "job_directory": "/synthetic/" + identity,
                                      "raw_response_sha256": "8" * 64,
                                      "capture_sha256": "9" * 64}},
            "reviewer": {"status": "not_invoked", "attempt_id": None,
                         "started_at": None, "deadline_at": None, "finished_at": None,
                         "result": {"kind": "not_invoked"}},
        }, "decision_path": path}
    body = {
        "schema_version": "lc_campaign_contract_v1", "manifest": manifest,
        "cases": cases, "budgets": {"specialist": len(cases), "reviewer": 0},
        "late_deliveries": [], "terminals": terminal_by_id,
        "terminals_locked": True,
    }
    return dict(body, state_sha256=digest(body))


class SnapshotJob:
    def __init__(self, path, terminal):
        self.path = path
        self.terminal = terminal

    def campaign_binding(self):
        return {"source_request_sha256": "1" * 64, "role_request_sha256": "2" * 64}

    def capture_binding(self, role):
        return {"raw_response_sha256": "8" * 64, "capture_sha256": "9" * 64}

    def grade_binding(self):
        assert self.terminal["kind"] == "published_grade"
        return {key: deepcopy(self.terminal[key])
                for key in ("grade_sha256", "status", "research_plan")}


def synthetic_score(scorer, source, manifest, state, loader=None):
    terminals = state["terminals"]
    loader = loader or (lambda path: SnapshotJob(str(path), terminals[str(path).rsplit("/", 1)[-1]]))
    return scorer(source, manifest, state, job_loader=loader)


def sole_metrics(arm):
    assert arm["combined_net_dollars"] is None
    assert arm["combined_dollars_per_candidate"] is None
    assert arm["combined_mtm"] is None
    assert len(arm["silos"]) == 1
    return arm["silos"][0]["metrics"]


def sole_contrast(contrasts):
    assert contrasts["combined_C_operational_minus_A"] is None
    assert contrasts["combined_C_operational_minus_B"] is None
    assert contrasts["combined_denominator"] is None
    assert len(contrasts["silos"]) == 1
    return contrasts["silos"][0]


def test_controller_fallback_does_not_become_agent_reject():
    from scripts.research.lc_campaign_accounting import operational_plan

    reject = {"action": "reject"}
    result = operational_plan(None, reject, "specialist_timeout")
    assert result == {
        "plan": reject,
        "origin": "controller_fallback",
        "reason": "specialist_timeout",
    }
    result["plan"]["action"] = "enter"
    assert reject == {"action": "reject"}


def test_fallback_requires_explicit_controller_reason():
    from scripts.research.lc_campaign_accounting import operational_plan

    with pytest.raises(ValueError):
        operational_plan(None, {"action": "reject"}, None)


def test_reviewed_plan_is_copied_without_relabelling():
    from scripts.research.lc_campaign_accounting import operational_plan

    reviewed = {"action": "enter", "nested": {"value": 1}}
    before = deepcopy(reviewed)
    result = operational_plan(reviewed, {"action": "reject"}, None)
    assert result == {"plan": reviewed, "origin": "reviewed_judgment", "reason": None}
    result["plan"]["nested"]["value"] = 2
    assert reviewed == before


def test_scenario_plan_changes_only_registered_cost_and_processing():
    from scripts.research.lc_campaign_accounting import scenario_plan

    plan = {
        "decision_time": "2026-01-01T00:00:00+00:00",
        "action": "enter",
        "stop": 95.0,
        "level": None,
        "entry_expiry": "2026-01-01T00:15:00+00:00",
        "exit_deadline": "2026-01-02T00:00:00+00:00",
        "processing_seconds": 90,
        "routing_seconds": 0,
        "notional": 50000.0,
        "cost_bps": 12,
    }
    before = deepcopy(plan)
    result = scenario_plan(plan, 24, 901)
    assert result == dict(plan, cost_bps=24, processing_seconds=901)
    assert plan == before


@pytest.mark.parametrize(
    "cost,seconds", [(True, 90), (-1, 90), (12, True), (12, -1), (12.5, 90)]
)
def test_scenario_plan_rejects_noninteger_or_negative_scenario_values(cost, seconds):
    from scripts.research.lc_campaign_accounting import scenario_plan

    with pytest.raises(ValueError):
        scenario_plan({"action": "reject"}, cost, seconds)


def test_mtm_charges_fee_once_and_gap_stop_is_literal_negative_3060():
    from scripts.research.lc_campaign_accounting import mtm_curve

    source = bars(); source.loc[source.index[1]] = [94.0, 95.0, 93.0, 94.0]
    result = mtm_curve(source, replay(source, full_plan()))
    assert result["status"] == "available"
    assert result["points"][-1] == {"observed_at": clock(1), "dollars": -3060.0}
    assert result["max_drawdown_dollars"] == 3060.0
    assert result["worst_closed_loss"] == -3060.0


def test_mtm_stop_first_ambiguity_and_deadline_open_are_final_observations():
    from scripts.research.lc_campaign_accounting import mtm_curve

    ambiguous = bars(); ambiguous.loc[ambiguous.index[1], ["low", "high"]] = [94.0, 111.0]
    stopped = mtm_curve(ambiguous, replay(ambiguous, full_plan()))
    assert stopped["ambiguous_bar_count"] == 1
    assert stopped["points"][-1] == {"observed_at": clock(2), "dollars": -2560.0}

    deadline = bars(1441); deadline.loc[deadline.index[1440], "open"] = 90.0
    closed = mtm_curve(deadline, replay(deadline, full_plan(), cutoff=1440))
    assert closed["points"][-1] == {"observed_at": clock(1440), "dollars": -5060.0}


def test_mtm_missing_required_open_position_mark_is_unknown_not_zero():
    from scripts.research.lc_campaign_accounting import mtm_curve

    source = bars(); book = replay(source, full_plan(), cutoff=3)
    source = source.drop(source.index[1])
    result = mtm_curve(source, book)
    assert result == {
        "status": "unavailable", "reason": "missing_required_mark",
        "points": None, "max_drawdown_dollars": None,
        "worst_closed_loss": None, "ambiguous_bar_count": None,
    }


def test_flat_reject_curve_reads_no_prices_and_has_no_fictitious_trade():
    from scripts.research.lc_campaign_accounting import mtm_curve

    reject = full_plan(action="reject")
    book = replay(None, reject)
    assert mtm_curve(None, book) == {
        "status": "available", "reason": None, "points": [],
        "max_drawdown_dollars": 0.0, "worst_closed_loss": None,
        "ambiguous_bar_count": 0,
    }


def test_missing_entry_price_is_unknown_and_never_flat_zero():
    from scripts.research.lc_campaign_accounting import mtm_curve

    result = mtm_curve(None, replay(None, full_plan()))
    assert result["status"] == "unavailable"
    assert result["reason"] == "book_incomplete"
    assert result["points"] is None


def test_score_separates_controller_fallback_from_null_judgment_and_pairs_n():
    from scripts.research.lc_campaign_accounting import score_campaign

    case = campaign_case(); manifest = accounting_manifest([case])
    terminal = {"case": {"kind": "external_failure", "reason": "specialist_timeout"}}
    state = terminal_state(manifest["campaign_manifest"], terminal)
    source = bars(1441); source.loc[source.index[3]] = [94.0, 95.0, 93.0, 94.0]
    result = synthetic_score(score_campaign, source, manifest, state)
    s0 = result["matched"]["scenarios"]["S0"]
    assert result["matched"]["silo_denominators"] == [
        {"archetype": "liquidity_compression", "track": "hourly", "denominator": 1}]
    assert result["matched"]["combined_denominator"] is None
    assert sole_metrics(s0["A"])["net_dollars"] == -3060.0
    assert sole_metrics(s0["C_operational"])["net_dollars"] == 0.0
    assert sole_metrics(s0["C_operational"])["win_rate"] is None
    assert sole_metrics(s0["C_judgment"])["available"] is False
    assert sole_contrast(s0["contrasts"]) == {
        "archetype": "liquidity_compression", "track": "hourly",
        "C_operational_minus_A": 3060.0, "C_operational_minus_B": 0.0,
        "paired_denominator": 1}
    assert result["reliability"]["controller_fallback_reasons"] == {"specialist_timeout": 1}
    assert result["reliability"]["deliberate_reviewed_rejects"] == 0


def test_fee_only_scenario_preserves_fill_and_changes_literal_sixty_dollars():
    from scripts.research.lc_campaign_accounting import score_campaign

    case = campaign_case(); manifest = accounting_manifest([case])
    terminal = {"case": {"kind": "external_failure", "reason": "timeout"}}
    source = bars(1441); source.loc[source.index[3]] = [94.0, 95.0, 93.0, 94.0]
    result = synthetic_score(score_campaign, source, manifest,
                            terminal_state(manifest["campaign_manifest"], terminal))
    s0 = sole_metrics(result["broad"]["scenarios"]["S0"]["A"])
    s1 = sole_metrics(result["broad"]["scenarios"]["S1"]["A"])
    assert (s0["net_dollars"], s1["net_dollars"]) == (-3060.0, -3120.0)
    assert s0["admission_order"] == s1["admission_order"] == ["case"]


def test_measured_901_seconds_expires_enter_without_mutating_grade():
    from scripts.research.lc_campaign_accounting import score_campaign

    case = campaign_case(); manifest = accounting_manifest([case])
    plan = deepcopy(case["plans"]["immediate"])
    terminal = {"case": {"kind": "published_grade", "job_directory": "/synthetic/case",
                         "grade_sha256": "7" * 64, "status": "research_ready",
                         "research_plan": plan}}
    before = deepcopy(terminal)
    source = bars(1441)
    result = synthetic_score(score_campaign, source, manifest, terminal_state(
        manifest["campaign_manifest"], terminal, elapsed_seconds=901))
    assert sole_metrics(result["matched"]["scenarios"]["S0"]["C_operational"])["trades"] == 1
    measured = sole_metrics(result["matched"]["scenarios"]["S4"]["C_operational"])
    assert measured["trades"] == 0
    assert measured["entry_reasons"] == {"expired": 1}
    assert terminal == before


def test_invalid_measured_timing_nulls_only_s4_s5():
    from scripts.research.lc_campaign_accounting import score_campaign

    case = campaign_case(); manifest = accounting_manifest([case])
    terminal = {"case": {"kind": "published_grade", "job_directory": "/synthetic/case",
                         "grade_sha256": "7" * 64, "status": "research_ready",
                         "research_plan": deepcopy(case["plans"]["immediate"])}}
    state = terminal_state(manifest["campaign_manifest"], terminal, timing_valid=False)
    result = synthetic_score(score_campaign, bars(1441), manifest, state)
    assert sole_metrics(result["matched"]["scenarios"]["S0"]["C_operational"])["available"] is True
    measured_op = result["matched"]["scenarios"]["S4"]["C_operational"]
    measured_judgment = result["matched"]["scenarios"]["S4"]["C_judgment"]
    assert measured_op["silos"][0]["metrics"] is None
    assert measured_judgment["silos"][0]["metrics"] is None
    assert sole_contrast(result["matched"]["scenarios"]["S4"]["contrasts"])[
        "C_operational_minus_A"] is None


def test_decision_month_contribution_survives_cross_month_exit_and_lomo_is_subtraction():
    from scripts.research.lc_campaign_accounting import score_campaign

    case = campaign_case(); manifest = accounting_manifest([case])
    terminal = {"case": {"kind": "external_failure", "reason": "timeout"}}
    source = bars(1441); source.loc[source.index[3]] = [94.0, 95.0, 93.0, 94.0]
    result = synthetic_score(score_campaign, source, manifest,
                            terminal_state(manifest["campaign_manifest"], terminal))
    a = sole_metrics(result["matched"]["scenarios"]["S0"]["A"])
    assert a["decision_month_contributions"] == {"2026-01": -3060.0}
    assert a["leave_one_month_out"] == {"2026-01": 0.0}
    assert result["matched"]["episodes"]["silos"][0]["episodes"] == [["case"]]


def test_empty_matched_cohort_has_no_fabricated_combined_denominator():
    from scripts.research.lc_campaign_accounting import score_campaign

    manifest = accounting_manifest([], matched_ids=[], as_of=0)
    state = terminal_state(manifest["campaign_manifest"], {})
    result = synthetic_score(score_campaign, None, manifest, state)
    assert result["matched"]["silo_denominators"] == []
    assert result["matched"]["combined_denominator"] is None
    assert result["matched"]["scenarios"]["S0"]["C_judgment"]["silos"] == []
    assert result["matched"]["scenarios"]["S0"]["contrasts"]["silos"] == []
    assert result["matched"]["episodes"] == {"silos": [], "combined": None}


def test_wait_is_post_arm_exclusive_expiry_and_pending_stop_is_causal():
    from scripts.research.lc_campaign_accounting import score_campaign

    case = campaign_case(); manifest = accounting_manifest([case])
    terminal = {"case": {"kind": "external_failure", "reason": "timeout"}}
    state = terminal_state(manifest["campaign_manifest"], terminal)

    confirmed = bars(1441)
    confirmed.loc[confirmed.index[2], ["high", "close"]] = [102.0, 102.0]
    result = synthetic_score(score_campaign, confirmed, manifest, state)
    b = sole_metrics(result["matched"]["scenarios"]["S0"]["B"])
    assert b["trades"] == 1 and b["admission_order"] == ["case"]

    at_expiry = bars(1441)
    at_expiry.loc[at_expiry.index[14], ["high", "close"]] = [102.0, 102.0]
    expired = synthetic_score(score_campaign, at_expiry, manifest, state)
    assert sole_metrics(expired["matched"]["scenarios"]["S0"]["B"])["entry_reasons"] == {"expired": 1}

    stopped = bars(1441); stopped.loc[stopped.index[1], "low"] = 94.0
    cancelled = synthetic_score(score_campaign, stopped, manifest, state)
    assert sole_metrics(cancelled["matched"]["scenarios"]["S0"]["B"])["entry_reasons"] == {"cancelled": 1}


def test_same_open_capacity_release_and_busy_candidate_never_retries():
    from scripts.research.lc_campaign_accounting import score_campaign

    first = campaign_case("first", 0)
    second = campaign_case("second", 1)
    third = campaign_case("third", 2)
    for case in (second, third):
        case["atr14"] = 10.0 / 2.7
        for plan in case["plans"].values(): plan["stop"] = 90.0
    cases = [first, second, third]; manifest = accounting_manifest(cases, ["first", "second", "third"], 1442)
    terminal = {identity: {"kind": "external_failure", "reason": "timeout"}
                for identity in ("first", "second", "third")}
    source = bars(1443); source.loc[source.index[3]] = [94.0, 95.0, 93.0, 94.0]
    result = synthetic_score(score_campaign, source, manifest,
                            terminal_state(manifest["campaign_manifest"], terminal))
    a = sole_metrics(result["matched"]["scenarios"]["S0"]["A"])
    assert a["admission_order"] == ["first", "second"]
    assert a["entry_reasons"] == {"admitted": 2, "skipped_busy": 1}
    assert result["matched"]["episodes"]["silos"][0]["episodes"] == [["first", "second", "third"]]


def test_nonentry_attribution_uses_a_net_sign_and_separates_reason():
    from scripts.research.lc_campaign_accounting import score_campaign

    case = campaign_case(); manifest = accounting_manifest([case])
    terminal = {"case": {"kind": "external_failure", "reason": "specialist_timeout"}}
    source = bars(1441); source.loc[source.index[3]] = [94.0, 95.0, 93.0, 94.0]
    result = synthetic_score(score_campaign, source, manifest,
                            terminal_state(manifest["campaign_manifest"], terminal))
    attribution = result["matched"]["nonentry_attribution_S0"]
    assert attribution["combined"] is None
    assert attribution["silos"][0]["attribution"] == {
        "specialist_timeout": {"missed_winner": 0, "avoided_loser": 1,
                               "breakeven": 0, "unknown": 0}}


def test_output_has_occupied_periods_and_json_safe_exploratory_denominators():
    from scripts.research.lc_campaign_accounting import score_campaign

    case = campaign_case(); manifest = accounting_manifest([case])
    terminal = {"case": {"kind": "external_failure", "reason": "timeout"}}
    source = bars(1441); source.loc[source.index[3]] = [94.0, 95.0, 93.0, 94.0]
    result = synthetic_score(score_campaign, source, manifest,
                            terminal_state(manifest["campaign_manifest"], terminal))
    a = sole_metrics(result["matched"]["scenarios"]["S0"]["A"])
    assert a["occupied_months"] == ["2026-02"]
    assert a["occupied_weeks"] == ["2026-W05"]
    pattern_report = result["matched"]["exploratory_patterns_S0"]
    assert pattern_report["combined"] is None
    subtype = pattern_report["silos"][0]["patterns"]["fields"]["hourly_subtype"]
    assert subtype == {"missing": 0, "groups": {
        '"synthetic"': {"candidates": 1, "a_winners": 0, "a_losers": 1,
                        "a_breakeven": 0, "a_unknown": 0}}}
    json.dumps(result, allow_nan=False)


def test_reliability_separates_capture_grade_fallback_and_latency_counts():
    from scripts.research.lc_campaign_accounting import score_campaign

    case = campaign_case(); manifest = accounting_manifest([case])
    terminal = {"case": {"kind": "external_failure", "reason": "specialist_timeout"}}
    result = synthetic_score(score_campaign, bars(1441), manifest, terminal_state(
        manifest["campaign_manifest"], terminal, elapsed_seconds=91))
    assert result["reliability"] == {
        "cases": 1, "specialist_invoked": 1, "specialist_captured": 1,
        "reviewer_invoked": 0, "reviewer_captured": 0,
        "published_grades": 0, "valid_reviewed": 0, "uncertain": 0,
        "material_or_incomplete_review": 0, "invalid_grade": 0,
        "external_failure_terminals": 1, "role_timeouts": 0,
        "role_status_counts": {"delivered": 1, "not_invoked": 1},
        "grade_status_counts": {},
        "controller_fallback_reasons": {"specialist_timeout": 1},
        "deliberate_reviewed_rejects": 0, "timing_valid": 1,
        "timing_invalid": 0, "latency_seconds": [91],
        "latency_distribution_seconds": {"min": 91, "median": 91, "max": 91},
        "measured_expiries_S4": 0,
    }


def test_deliberate_reviewed_reject_is_not_counted_as_fallback():
    from scripts.research.lc_campaign_accounting import score_campaign

    case = campaign_case(); manifest = accounting_manifest([case])
    terminal = {"case": {"kind": "published_grade", "job_directory": "/synthetic/case",
                         "grade_sha256": "7" * 64, "status": "research_ready",
                         "research_plan": deepcopy(case["plans"]["reject"])}}
    result = synthetic_score(score_campaign, bars(1441), manifest,
                            terminal_state(manifest["campaign_manifest"], terminal))
    s0 = result["matched"]["scenarios"]["S0"]
    assert sole_metrics(s0["C_operational"])["net_dollars"] == 0.0
    assert sole_metrics(s0["C_judgment"])["net_dollars"] == 0.0
    assert result["reliability"]["deliberate_reviewed_rejects"] == 1
    assert result["reliability"]["controller_fallback_reasons"] == {}


def test_score_rejects_rehashed_impossible_decision_path():
    from scripts.research.assessment_evidence_guard import _canonical
    from scripts.research.lc_campaign_accounting import score_campaign

    case = campaign_case(); manifest = accounting_manifest([case])
    terminal = {"case": {"kind": "external_failure", "reason": "timeout"}}
    state = terminal_state(manifest["campaign_manifest"], terminal)
    path = state["cases"]["case"]["decision_path"]
    path["timing_valid"] = False
    path["reason"] = "timer_runtime_changed"
    path["path_sha256"] = hashlib.sha256(_canonical(
        {key: value for key, value in path.items() if key != "path_sha256"}
    ).encode("ascii")).hexdigest()
    state["state_sha256"] = hashlib.sha256(_canonical(
        {key: value for key, value in state.items() if key != "state_sha256"}
    ).encode("ascii")).hexdigest()
    with pytest.raises(ValueError, match="timing"):
        synthetic_score(score_campaign, bars(1441), manifest, state)


def test_score_uses_authoritative_snapshot_validator_and_job_loader():
    from scripts.research.lc_campaign_accounting import score_campaign

    case = campaign_case(); manifest = accounting_manifest([case])
    terminal = {"case": {"kind": "external_failure", "reason": "timeout"}}
    state = terminal_state(manifest["campaign_manifest"], terminal)
    bad = SnapshotJob("/synthetic/case", terminal["case"])
    bad.campaign_binding = lambda: {
        "source_request_sha256": "0" * 64, "role_request_sha256": "2" * 64}
    with pytest.raises(ValueError, match="manifest|binding"):
        synthetic_score(score_campaign, bars(1441), manifest, state,
                        loader=lambda path: bad)


def test_mixed_tracks_are_independent_silos_with_no_aggregate_pnl_or_contrast():
    from scripts.research.lc_campaign_accounting import score_campaign

    hourly = campaign_case("hourly", 0)
    minute = campaign_case("minute", 0); minute["track"] = "minute"
    cases = [hourly, minute]
    manifest = accounting_manifest(cases, ["hourly", "minute"])
    terminal = {identity: {"kind": "external_failure", "reason": "timeout"}
                for identity in ("hourly", "minute")}
    result = synthetic_score(score_campaign, bars(1441), manifest,
                             terminal_state(manifest["campaign_manifest"], terminal))
    arm = result["matched"]["scenarios"]["S0"]["A"]
    assert arm["combined_net_dollars"] is None
    assert arm["combined_dollars_per_candidate"] is None
    assert arm["combined_mtm"] is None
    assert {(item["identity"]["track"], tuple(item["metrics"]["admission_order"]))
            for item in arm["silos"]} == {("hourly", ("hourly",)), ("minute", ("minute",))}
    assert {item["identity"]["track"]
            for item in result["broad"]["scenarios"]["S3"]["B"]["silos"]} == {
        "hourly", "minute"}
    contrasts = result["matched"]["scenarios"]["S0"]["contrasts"]
    assert contrasts["combined_C_operational_minus_A"] is None
    assert contrasts["combined_C_operational_minus_B"] is None
    assert {(item["track"], item["paired_denominator"])
            for item in contrasts["silos"]} == {("hourly", 1), ("minute", 1)}
    assert {item["track"] for item in result["matched"]["episodes"]["silos"]} == {
        "hourly", "minute"}
    assert result["matched"]["episodes"]["combined"] is None
    attribution = result["matched"]["nonentry_attribution_S0"]
    patterns = result["matched"]["exploratory_patterns_S0"]
    assert attribution["combined"] is None and patterns["combined"] is None
    assert {item["identity"]["track"] for item in attribution["silos"]} == {
        "hourly", "minute"}
    assert {item["identity"]["track"] for item in patterns["silos"]} == {
        "hourly", "minute"}


def test_invalid_measured_timing_nulls_only_its_own_track_silo():
    from scripts.research.assessment_evidence_guard import _canonical
    from scripts.research.lc_campaign_accounting import score_campaign

    hourly = campaign_case("hourly", 0)
    minute = campaign_case("minute", 0); minute["track"] = "minute"
    manifest = accounting_manifest([hourly, minute], ["hourly", "minute"])
    terminal = {identity: {"kind": "external_failure", "reason": "timeout"}
                for identity in ("hourly", "minute")}
    state = terminal_state(manifest["campaign_manifest"], terminal)
    path = state["cases"]["minute"]["decision_path"]
    path.update(timing_valid=False, elapsed_ns=None, reason="timer_runtime_changed")
    path["end"]["runtime_id"] = "restarted"
    path["path_sha256"] = hashlib.sha256(_canonical(
        {key: value for key, value in path.items() if key != "path_sha256"}
    ).encode("ascii")).hexdigest()
    state["state_sha256"] = hashlib.sha256(_canonical(
        {key: value for key, value in state.items() if key != "state_sha256"}
    ).encode("ascii")).hexdigest()
    result = synthetic_score(score_campaign, bars(1441), manifest, state)
    silos = {item["identity"]["track"]: item
             for item in result["matched"]["scenarios"]["S4"]["C_operational"]["silos"]}
    assert silos["hourly"]["metrics"] is not None
    assert silos["minute"]["metrics"] is None
    assert silos["minute"]["reason"] == "invalid_measured_timing"


def test_episode_regression_does_not_duplicate_final_group():
    from scripts.research.lc_campaign_accounting import _episodes

    cases = [campaign_case("a", 0), campaign_case("b", 60), campaign_case("c", 3000)]
    assert _episodes(cases) == [["a", "b"], ["c"]]
