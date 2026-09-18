"""Offline accounting for the fixed LC consolidated campaign.

This module does not read sources, outcomes, jobs, or network state.  Callers
must pass already-authorized outcome bars and immutable campaign records.

Task 4 interface (all values JSON-safe except ``bars``, a UTC DataFrame):

``manifest`` is exactly ``{"schema_version": "lc_campaign_accounting_v1",
"campaign_manifest": <the Task-2 frozen manifest>, "as_of": <UTC minute>,
"broad_cases": [case...], "matched_case_ids": [str...]}``.  A case has exactly
``case_id, track, decision_time, source_close, atr14, plans, facts``. ``plans``
has the exact keys ``immediate, wait_5m_high, reject`` and each value is the
strict 10-key plan accepted by ``replay_isolated_sleeves``.  ``facts`` is a
JSON-safe predecision mapping used only for descriptive grouping.

``terminals`` is the *full object* returned by ``CampaignLedger.state()``; it
is not a second terminal schema.  Its ``manifest`` must equal
``campaign_manifest``, grades are read from ``terminals``, and measured timing
is read only from ``cases[case_id].decision_path``.

``score_campaign`` returns ``lc_campaign_accounting_result_v1`` with independent
``broad`` and ``matched`` books.  Every arm contains ``silos`` of
``{identity, denominator, metrics}`` and explicit null ``combined_*`` fields;
paired contrasts, episodes, attribution, and patterns are likewise per matching
archetype/track and never summed across tracks. Reliability remains campaign
level. Unknown books and invalid measured timing remain null, never zero.
"""
from __future__ import annotations

from copy import deepcopy
from collections import Counter
import math
import statistics

import pandas as pd

from scripts.research.conditional_entry import MINUTE, _clock
from scripts.research.conditional_occupancy import PLAN_KEYS, replay_isolated_sleeves
from scripts.research.lc_campaign_contract import ceil_elapsed_seconds, validate_state_snapshot
from scripts.research.assessment_evidence_guard import _canonical


ACCOUNTING_VERSION = "lc_campaign_accounting_v1"
RESULT_VERSION = "lc_campaign_accounting_result_v1"
SCENARIOS = {
    "S0": (12, 90), "S1": (24, 90),
    "S2": (12, 300), "S3": (24, 300),
}


def operational_plan(reviewed_plan: dict | None, reject_plan: dict,
                     reason: str | None) -> dict:
    """Return an execution plan while preserving judgment provenance."""
    if not isinstance(reject_plan, dict):
        raise ValueError("reject execution plan must be a dictionary")
    if reviewed_plan is not None:
        if not isinstance(reviewed_plan, dict):
            raise ValueError("reviewed execution plan must be a dictionary")
        return {"plan": deepcopy(reviewed_plan),
                "origin": "reviewed_judgment", "reason": None}
    if not isinstance(reason, str) or not reason:
        raise ValueError("controller fallback needs an explicit reason")
    return {"plan": deepcopy(reject_plan),
            "origin": "controller_fallback", "reason": reason}


def scenario_plan(plan: dict, cost_bps: int, processing_seconds: int) -> dict:
    """Clone a locked menu plan with one registered scenario's assumptions."""
    if (type(cost_bps) is not int or cost_bps < 0
            or type(processing_seconds) is not int or processing_seconds < 0):
        raise ValueError("scenario cost and processing must be nonnegative integers")
    if not isinstance(plan, dict):
        raise ValueError("plan must be a dictionary")
    result = deepcopy(plan)
    result["cost_bps"] = cost_bps
    result["processing_seconds"] = processing_seconds
    return result


def _mtm_unavailable(reason):
    return {"status": "unavailable", "reason": reason, "points": None,
            "max_drawdown_dollars": None, "worst_closed_loss": None,
            "ambiguous_bar_count": None}


def mtm_curve(bars, book: dict) -> dict:
    """Build a causal, unfunded dollar MTM curve from a sleeve replay.

    Points are observed times.  A minute close at ``t`` is represented at
    ``t + 1 minute``; an open-phase exit is represented at its open.  When both
    occur together the exit is applied first, so the stale close cannot mark an
    already-closed position.  Full round-trip fees are charged exactly once at
    admission and closed gross P&L is never charged again.
    """
    if not isinstance(book, dict) or not isinstance(book.get("ledger"), list):
        raise ValueError("sleeve replay with ledger required")
    resolved_statuses = {"admitted", "skipped_busy", "rejected", "cancelled", "expired"}
    if any(row.get("status") not in resolved_statuses for row in book["ledger"]):
        return _mtm_unavailable("book_incomplete")
    positions = [row.get("position") for row in book["ledger"] if row.get("position")]
    if any(not isinstance(position, dict) or position.get("status") == "unknown"
           for position in positions):
        return _mtm_unavailable("unknown_position_state")
    if not positions:
        return {"status": "available", "reason": None, "points": [],
                "max_drawdown_dollars": 0.0, "worst_closed_loss": None,
                "ambiguous_bar_count": 0}
    try:
        cutoff = _clock(book["as_of"])
        admissions = []
        exits = []
        observation_times = set()
        for position in positions:
            entry = _clock(position["entry_time"])
            fees = float(position["fees"])
            quantity = float(position["quantity"])
            entry_price = float(position["entry_price"])
            if not all(math.isfinite(value) for value in (fees, quantity, entry_price)):
                return _mtm_unavailable("invalid_position_economics")
            admissions.append((entry, fees))
            if position["status"] == "closed":
                exit_observed = _clock(position["exit_observed_at"])
                gross = float(position["gross_pnl"])
                exits.append((exit_observed, gross))
                end = exit_observed
            elif position["status"] == "open":
                end = cutoff
            else:
                return _mtm_unavailable("unknown_position_state")
            at = entry + MINUTE
            while at <= end:
                observation_times.add(at)
                at += MINUTE
        points = []
        for observed in sorted(observation_times):
            realized = math.fsum(gross for at, gross in exits if at <= observed)
            fees = math.fsum(fee for at, fee in admissions if at <= observed)
            unrealized = 0.0
            for position in positions:
                entry = _clock(position["entry_time"])
                exit_observed = (_clock(position["exit_observed_at"])
                                 if position["status"] == "closed" else None)
                if entry >= observed or (exit_observed is not None and exit_observed <= observed):
                    continue
                mark_at = observed - MINUTE
                try:
                    mark = float(bars.at[mark_at, "close"])
                except (AttributeError, KeyError, TypeError, ValueError):
                    return _mtm_unavailable("missing_required_mark")
                if not math.isfinite(mark) or mark <= 0:
                    return _mtm_unavailable("missing_required_mark")
                unrealized += (mark-float(position["entry_price"]))*float(position["quantity"])
            dollars = realized+unrealized-fees
            if not math.isfinite(dollars):
                return _mtm_unavailable("invalid_mtm_economics")
            points.append({"observed_at": observed.isoformat(), "dollars": dollars})
    except (KeyError, TypeError, ValueError, OverflowError):
        return _mtm_unavailable("invalid_book")
    peak = 0.0; drawdown = 0.0
    for point in points:
        peak = max(peak, point["dollars"])
        drawdown = max(drawdown, peak-point["dollars"])
    closed_net = [float(position["net_pnl"]) for position in positions
                  if position["status"] == "closed"]
    return {"status": "available", "reason": None, "points": points,
            "max_drawdown_dollars": drawdown,
            "worst_closed_loss": min(closed_net) if closed_net else None,
            "ambiguous_bar_count": sum(bool(position.get("ambiguous_bar"))
                                       for position in positions
                                       if position["status"] == "closed")}


def _positive(value):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and value > 0)


def _validate_plan(plan, case, expected_action):
    if not isinstance(plan, dict) or set(plan) != PLAN_KEYS:
        raise ValueError("case plan must use the strict resolver schema")
    if plan["action"] != expected_action or plan["decision_time"] != case["decision_time"]:
        raise ValueError("case plan action or decision clock differs")
    if plan["notional"] != 50000 or plan["routing_seconds"] != 0:
        raise ValueError("campaign uses fixed $50,000 notional and zero routing")
    if plan["cost_bps"] != 12 or plan["processing_seconds"] != 90:
        raise ValueError("frozen menu must use the primary scenario assumptions")
    decision = _clock(plan["decision_time"])
    if (_clock(plan["entry_expiry"]) != decision+pd.Timedelta(minutes=15)
            or _clock(plan["exit_deadline"]) != decision+pd.Timedelta(minutes=1440)):
        raise ValueError("campaign expiry or deadline differs")
    expected_stop = case["source_close"]-2.7*case["atr14"]
    if not math.isclose(float(plan["stop"]), expected_stop, rel_tol=0, abs_tol=1e-9):
        raise ValueError("campaign stop differs from source close minus 2.7 ATR14")
    if expected_action == "wait_close_above":
        if not _positive(plan["level"]):
            raise ValueError("wait plan needs a frozen level")
    elif plan["level"] is not None:
        raise ValueError("non-wait plan cannot carry a level")


def _validate_inputs(manifest, state):
    if (not isinstance(manifest, dict) or set(manifest) != {
            "schema_version", "campaign_manifest", "as_of", "broad_cases", "matched_case_ids"}
            or manifest["schema_version"] != ACCOUNTING_VERSION):
        raise ValueError("exact accounting manifest schema required")
    _clock(manifest["as_of"])
    if not isinstance(manifest["broad_cases"], list) or not isinstance(manifest["matched_case_ids"], list):
        raise ValueError("case collections must be lists")
    if (state["manifest"] != manifest["campaign_manifest"] or state["terminals_locked"] is not True
            or set(state["terminals"]) != set(manifest["matched_case_ids"])
            or set(state["cases"]) != set(manifest["matched_case_ids"])):
        raise ValueError("locked terminal state differs from accounting manifest")
    contract_ids = [item["case_id"] for item in state["manifest"]["cases"]]
    if contract_ids != manifest["matched_case_ids"] or len(set(contract_ids)) != len(contract_ids):
        raise ValueError("matched IDs must preserve the Task-2 manifest order")
    case_keys = {"case_id", "track", "decision_time", "source_close", "atr14", "plans", "facts"}
    cases = {}; prior = None
    for case in manifest["broad_cases"]:
        if not isinstance(case, dict) or set(case) != case_keys:
            raise ValueError("exact accounting case schema required")
        identity = case["case_id"]
        if (not isinstance(identity, str) or not identity or identity in cases
                or case["track"] not in ("hourly", "minute")
                or not _positive(case["source_close"]) or not _positive(case["atr14"])
                or not isinstance(case["facts"], dict)
                or not isinstance(case["plans"], dict)
                or set(case["plans"]) != {"immediate", "wait_5m_high", "reject"}):
            raise ValueError("invalid accounting case")
        decision = _clock(case["decision_time"])
        if prior is not None and decision < prior:
            raise ValueError("broad cases must be chronological")
        prior = decision
        _validate_plan(case["plans"]["immediate"], case, "enter")
        _validate_plan(case["plans"]["wait_5m_high"], case, "wait_close_above")
        _validate_plan(case["plans"]["reject"], case, "reject")
        cases[identity] = case
    if any(identity not in cases for identity in manifest["matched_case_ids"]):
        raise ValueError("matched case missing from broad frozen population")
    return cases


def _candidate(case, plan, variant, unavailable_reason=None):
    identity = {"archetype": "liquidity_compression", "track": case["track"], "variant": variant}
    return dict(identity, candidate_id=case["case_id"], decision_time=case["decision_time"],
                plan=deepcopy(plan), unavailable_reason=unavailable_reason)


def _case_silos(cases):
    grouped = {}
    for case in cases:
        grouped.setdefault(("liquidity_compression", case["track"]), []).append(case)
    return grouped


def _run_books(bars, cases, plans, reasons, variant, as_of):
    silos = []
    for (archetype, track), grouped in sorted(_case_silos(cases).items()):
        identity = {"archetype": archetype, "track": track, "variant": variant}
        rows = [_candidate(case, plans.get(case["case_id"]), variant,
                           reasons.get(case["case_id"])) for case in grouped]
        silos.append(dict(identity, candidates=rows))
    if not silos:
        return []
    return replay_isolated_sleeves(bars, silos, as_of=as_of)["silos"]


def _metrics(bars, book, cases):
    rows = {row["candidate_id"]: row for row in book["ledger"]}
    curve = mtm_curve(bars, book)
    available = book["summary"]["policy_net_pnl"] is not None and curve["status"] == "available"
    positions = [row["position"] for row in rows.values() if row.get("position")]
    closed = [position for position in positions if position["status"] == "closed"]
    net = float(book["summary"]["policy_net_pnl"]) if available else None
    wins = [float(position["net_pnl"]) for position in closed if position["net_pnl"] > 0]
    losses = [float(position["net_pnl"]) for position in closed if position["net_pnl"] < 0]
    breakeven = sum(position["net_pnl"] == 0 for position in closed)
    holds = [(_clock(position["exit_time"])-_clock(position["entry_time"])).total_seconds()/60
             for position in closed]
    occupied_days = set()
    for position in positions:
        start = _clock(position["entry_time"])
        end = (_clock(position["exit_time"]) if position["status"] == "closed"
               else _clock(book["as_of"]))
        for day in pd.date_range(start.floor("D"), end.floor("D"), freq="D"):
            occupied_days.add(day)
    occupied_months = sorted({day.strftime("%Y-%m") for day in occupied_days})
    occupied_weeks = sorted({f"{day.isocalendar().year:04d}-W{day.isocalendar().week:02d}"
                             for day in occupied_days})
    months = {pd.Timestamp(case["decision_time"]).strftime("%Y-%m"): 0.0 for case in cases}
    for case in cases:
        position = rows[case["case_id"]].get("position")
        if position and position["status"] == "closed":
            months[pd.Timestamp(case["decision_time"]).strftime("%Y-%m")] += float(position["net_pnl"])
    lomo = ({month: net-contribution for month, contribution in months.items()}
            if available else {month: None for month in months})
    return {
        "available": available, "reason": None if available else curve["reason"] or "book_incomplete",
        "candidate_count": len(cases), "net_dollars": net,
        "dollars_per_candidate": net/len(cases) if available and cases else (0.0 if available else None),
        "trades": len(positions), "closed_trades": len(closed),
        "fees": math.fsum(float(position["fees"]) for position in positions),
        "average_initial_risk": book["summary"]["average_initial_risk"],
        "wins": len(wins), "losses": len(losses), "breakeven": breakeven,
        "win_rate": len(wins)/len(closed) if closed else None,
        "average_win": math.fsum(wins)/len(wins) if wins else None,
        "average_loss": math.fsum(losses)/len(losses) if losses else None,
        "average_hold_minutes": math.fsum(holds)/len(holds) if holds else None,
        "occupied_months": occupied_months, "occupied_weeks": occupied_weeks,
        "entry_reasons": dict(sorted(Counter(row["status"] for row in rows.values()).items())),
        "admission_order": list(book["admission_order"]), "mtm": curve,
        "decision_month_contributions": months if available else {month: None for month in months},
        "leave_one_month_out": lomo,
    }


def _score_arm(bars, cases, base_plans, reasons, variant, cost, processing, as_of,
               null_tracks=None):
    null_tracks = set() if null_tracks is None else set(null_tracks)
    output = []
    for (archetype, track), grouped in sorted(_case_silos(cases).items()):
        identity = {"archetype": archetype, "track": track, "variant": variant}
        if track in null_tracks:
            output.append({"identity": identity, "denominator": len(grouped),
                           "metrics": None, "reason": "invalid_measured_timing"})
            continue
        plans = {case["case_id"]: (scenario_plan(base_plans[case["case_id"]], cost,
                 processing[case["case_id"]]) if base_plans[case["case_id"]] is not None else None)
                 for case in grouped}
        replay = _run_books(bars, grouped, plans, reasons, variant, as_of)[0]["replay"]
        output.append({"identity": identity, "denominator": len(grouped),
                       "metrics": _metrics(bars, replay, grouped), "reason": None})
    return {"silos": output, "combined_net_dollars": None,
            "combined_dollars_per_candidate": None, "combined_mtm": None}


def _episodes(cases):
    groups = []; current = []; end = None
    for case in cases:
        start = _clock(case["decision_time"]); case_end = start+pd.Timedelta(hours=24)
        if not current:
            current = [case["case_id"]]; end = case_end
        elif start <= end:
            current.append(case["case_id"]); end = max(end, case_end)
        else:
            groups.append(current)
            current = [case["case_id"]]; end = case_end
    return groups + ([current] if current else [])


def _episode_silos(cases):
    return {"silos": [{"archetype": archetype, "track": track,
                        "episodes": _episodes(grouped)}
                       for (archetype, track), grouped in sorted(_case_silos(cases).items())],
            "combined": None}


def _nonentry_attribution(a_book, c_book, c_reasons):
    a_rows = {row["candidate_id"]: row for row in a_book["ledger"]}
    c_rows = {row["candidate_id"]: row for row in c_book["ledger"]}
    output = {}
    for identity, c_row in c_rows.items():
        if c_row.get("position") is not None:
            continue
        reason = c_reasons.get(identity) or c_row["status"]
        bucket = output.setdefault(reason, {"missed_winner": 0, "avoided_loser": 0,
                                            "breakeven": 0, "unknown": 0})
        position = a_rows[identity].get("position")
        if not position or position["status"] != "closed" or position.get("net_pnl") is None:
            bucket["unknown"] += 1
        elif position["net_pnl"] > 0: bucket["missed_winner"] += 1
        elif position["net_pnl"] < 0: bucket["avoided_loser"] += 1
        else: bucket["breakeven"] += 1
    return dict(sorted(output.items()))


def _exploratory_patterns(cases, a_book):
    rows = {row["candidate_id"]: row for row in a_book["ledger"]}
    fields = sorted({key for case in cases for key in case["facts"]})
    output = {}
    for field in fields:
        missing = 0; groups = {}
        for case in cases:
            if field not in case["facts"] or case["facts"][field] is None:
                missing += 1; continue
            label = _canonical(case["facts"][field])
            bucket = groups.setdefault(label, {"candidates": 0, "a_winners": 0,
                                               "a_losers": 0, "a_breakeven": 0,
                                               "a_unknown": 0})
            bucket["candidates"] += 1
            position = rows[case["case_id"]].get("position")
            if not position or position["status"] != "closed" or position.get("net_pnl") is None:
                bucket["a_unknown"] += 1
            elif position["net_pnl"] > 0: bucket["a_winners"] += 1
            elif position["net_pnl"] < 0: bucket["a_losers"] += 1
            else: bucket["a_breakeven"] += 1
        output[field] = {"missing": missing, "groups": dict(sorted(groups.items()))}
    return {"denominator": len(cases), "fitted_thresholds": False, "fields": output}


def _arm_by_track(arm):
    return {item["identity"]["track"]: item for item in arm["silos"]}


def _contrasts(cases, a_arm, b_arm, c_arm):
    a_silos = _arm_by_track(a_arm); b_silos = _arm_by_track(b_arm); c_silos = _arm_by_track(c_arm)
    output = []
    for (archetype, track), grouped in sorted(_case_silos(cases).items()):
        a = a_silos[track]["metrics"]; b = b_silos[track]["metrics"]
        c = c_silos[track]["metrics"]
        cop = c["net_dollars"] if c is not None else None
        output.append({"archetype": archetype, "track": track,
                       "paired_denominator": len(grouped),
                       "C_operational_minus_A": (cop-a["net_dollars"]
                           if cop is not None and a["net_dollars"] is not None else None),
                       "C_operational_minus_B": (cop-b["net_dollars"]
                           if cop is not None and b["net_dollars"] is not None else None)})
    return {"silos": output, "combined_C_operational_minus_A": None,
            "combined_C_operational_minus_B": None, "combined_denominator": None}


def _silo_denominators(cases):
    return [{"archetype": archetype, "track": track, "denominator": len(grouped)}
            for (archetype, track), grouped in sorted(_case_silos(cases).items())]


def score_campaign(bars, manifest, terminals, *, job_loader=None) -> dict:
    """Replay all registered books after a locked Task-2 state is supplied."""
    manifest = deepcopy(manifest)
    state = (validate_state_snapshot(terminals) if job_loader is None
             else validate_state_snapshot(terminals, job_loader=job_loader))
    case_map = _validate_inputs(manifest, state)
    broad = manifest["broad_cases"]
    matched = [case_map[identity] for identity in manifest["matched_case_ids"]]
    as_of = manifest["as_of"]

    broad_plans = {
        "A": {case["case_id"]: case["plans"]["immediate"] for case in broad},
        "B": {case["case_id"]: case["plans"]["wait_5m_high"] for case in broad},
    }
    broad_scenarios = {}
    for scenario, (cost, delay) in SCENARIOS.items():
        per_case = {case["case_id"]: delay for case in broad}
        broad_scenarios[scenario] = {
            arm: _score_arm(bars, broad, plans, {}, "broad_"+arm+"_"+scenario,
                            cost, per_case, as_of)
            for arm, plans in broad_plans.items()
        }

    a_base = {case["case_id"]: case["plans"]["immediate"] for case in matched}
    b_base = {case["case_id"]: case["plans"]["wait_5m_high"] for case in matched}
    op_base = {}; judgment_base = {}; reasons = {}; measured = {}; invalid_timing_tracks = set()
    deliberate_rejects = 0; fallback_reasons = Counter(); status_counts = Counter()
    for case in matched:
        identity = case["case_id"]; terminal = state["terminals"][identity]
        reviewed = None
        if terminal["kind"] == "published_grade":
            status_counts[terminal["status"]] += 1
            if terminal["status"] == "research_ready": reviewed = terminal["research_plan"]
            else: reasons[identity] = terminal["status"]
        else:
            reasons[identity] = terminal["reason"]
        if reviewed is not None and reviewed not in case["plans"].values():
            raise ValueError("published plan is not one frozen source-menu plan")
        if reviewed is not None and reviewed["action"] == "reject": deliberate_rejects += 1
        op = operational_plan(reviewed, case["plans"]["reject"], reasons.get(identity))
        op_base[identity] = op["plan"]; judgment_base[identity] = deepcopy(reviewed)
        if op["origin"] == "controller_fallback": fallback_reasons[op["reason"]] += 1
        path = state["cases"][identity]["decision_path"]
        if path.get("terminal_valid") is not True:
            raise ValueError("case decision path has no valid bound terminal")
        if path.get("timing_valid") is True:
            measured[identity] = ceil_elapsed_seconds(path["elapsed_ns"])
        else:
            invalid_timing_tracks.add(case["track"])

    matched_scenarios = {}
    nonentry = {"silos": [], "combined": None}
    patterns = {"silos": [], "combined": None}
    assumed = dict(SCENARIOS, S4=(12, 90), S5=(24, 90))
    for scenario, (cost, delay) in assumed.items():
        ab_delay = {case["case_id"]: 90 if scenario in ("S4", "S5") else delay for case in matched}
        arms = {
            "A": _score_arm(bars, matched, a_base, {}, "matched_A_"+scenario,
                            cost, ab_delay, as_of),
            "B": _score_arm(bars, matched, b_base, {}, "matched_B_"+scenario,
                            cost, ab_delay, as_of),
        }
        c_delay = measured if scenario in ("S4", "S5") else {
            case["case_id"]: delay for case in matched}
        null_tracks = invalid_timing_tracks if scenario in ("S4", "S5") else set()
        arms["C_operational"] = _score_arm(
            bars, matched, op_base, {}, "matched_C_operational_"+scenario,
            cost, c_delay, as_of, null_tracks)
        judgment_reasons = {identity: reasons.get(identity, "judgment_unavailable")
                            for identity, plan in judgment_base.items() if plan is None}
        arms["C_judgment"] = _score_arm(
            bars, matched, judgment_base, judgment_reasons,
            "matched_C_judgment_"+scenario, cost, c_delay, as_of, null_tracks)
        contrasts = _contrasts(matched, arms["A"], arms["B"], arms["C_operational"])
        if scenario == "S0" and matched:
            a_runs = _run_books(bars, matched, {identity: scenario_plan(plan, cost, delay)
                                for identity, plan in a_base.items()}, {}, "attribution_A", as_of)
            c_runs = _run_books(bars, matched, {identity: scenario_plan(plan, cost, delay)
                                for identity, plan in op_base.items()}, {}, "attribution_C", as_of)
            c_by_track = {item["track"]: item["replay"] for item in c_runs}
            case_silos = _case_silos(matched)
            for item in a_runs:
                track = item["track"]; grouped = case_silos[(item["archetype"], track)]
                identity = {"archetype": item["archetype"], "track": track}
                nonentry["silos"].append({"identity": identity,
                    "attribution": _nonentry_attribution(item["replay"], c_by_track[track], reasons)})
                patterns["silos"].append({"identity": identity,
                    "patterns": _exploratory_patterns(grouped, item["replay"])})
        matched_scenarios[scenario] = dict(arms, contrasts=contrasts)

    role_status = Counter(record["status"] for case in state["cases"].values()
                          for record in case["roles"].values())
    valid_latencies = [ceil_elapsed_seconds(case["decision_path"]["elapsed_ns"])
                       for case in state["cases"].values()
                       if case["decision_path"]["timing_valid"] is True]
    role_records = {role: [case["roles"][role] for case in state["cases"].values()]
                    for role in ("specialist", "reviewer")}
    published_count = sum(terminal["kind"] == "published_grade"
                          for terminal in state["terminals"].values())
    valid_reviewed = status_counts["research_ready"]
    uncertain = status_counts["insufficient_evidence"]
    material = status_counts["review_not_passed"]
    invalid_grade = published_count-valid_reviewed-uncertain-material
    latency_distribution = ({"min": min(valid_latencies),
                             "median": statistics.median(valid_latencies),
                             "max": max(valid_latencies)} if valid_latencies else None)
    measured_s4 = matched_scenarios["S4"]["C_operational"]
    measured_expiries = (None if invalid_timing_tracks else sum(
        item["metrics"]["entry_reasons"].get("expired", 0)
        for item in measured_s4["silos"] if item["metrics"] is not None))
    reliability = {
        "cases": len(matched),
        "specialist_invoked": sum(record["attempt_id"] is not None for record in role_records["specialist"]),
        "specialist_captured": sum(record["status"] == "delivered" for record in role_records["specialist"]),
        "reviewer_invoked": sum(record["attempt_id"] is not None for record in role_records["reviewer"]),
        "reviewer_captured": sum(record["status"] == "delivered" for record in role_records["reviewer"]),
        "published_grades": published_count, "valid_reviewed": valid_reviewed,
        "uncertain": uncertain, "material_or_incomplete_review": material,
        "invalid_grade": invalid_grade,
        "external_failure_terminals": sum(terminal["kind"] == "external_failure"
                                          for terminal in state["terminals"].values()),
        "role_timeouts": role_status["timeout"],
        "role_status_counts": dict(sorted(role_status.items())),
        "grade_status_counts": dict(sorted(status_counts.items())),
        "controller_fallback_reasons": dict(sorted(fallback_reasons.items())),
        "deliberate_reviewed_rejects": deliberate_rejects,
        "timing_valid": len(valid_latencies),
        "timing_invalid": len(matched)-len(valid_latencies),
        "latency_seconds": valid_latencies,
        "latency_distribution_seconds": latency_distribution,
        "measured_expiries_S4": measured_expiries,
    }
    return {
        "schema_version": RESULT_VERSION,
        "assumptions": {"notional": 50000.0, "starting_equity": None,
                        "funded": False, "fixed_notional_is_equal_risk": False},
        "broad": {"silo_denominators": _silo_denominators(broad),
                  "combined_denominator": None, "scenarios": broad_scenarios},
        "matched": {"silo_denominators": _silo_denominators(matched),
                    "combined_denominator": None, "scenarios": matched_scenarios,
                    "episodes": _episode_silos(matched),
                    "nonentry_attribution_S0": nonentry,
                    "exploratory_patterns_S0": patterns},
        "reliability": reliability,
    }
