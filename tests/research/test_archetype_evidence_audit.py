"""Accounting regressions: scale-outs and incomplete histories are not extra trials."""
import importlib.util
from pathlib import Path
import unittest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/research/archetype_evidence_audit.py"


def fill(pid, pnl, *, arch="liquidity_compression", entry="2026-03-10T00:00:00Z",
         exit="2026-03-11T00:00:00Z", reason="stop_loss"):
    return dict(position_id=pid, pnl_usd=pnl, pnl=pnl, archetype=arch,
                direction="long", timestamp_entry=entry, timestamp_exit=exit,
                exit_reason=reason, quantity=1.0, entry_price=100.0,
                stop_loss=95.0, position_size_usd=100.0)


class EvidenceAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("evidence_audit", SCRIPT)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def run_audit(self, trades, opens=()):
        return self.module.audit(
            trades,
            {"heartbeat": {"open_position_details": [{"id": p} for p in opens]}},
            ["liquidity_compression", "whipsaw"], "2026-03-09T00:00:00Z")

    def test_scale_outs_count_as_one_completed_position(self):
        report = self.run_audit([
            fill("a", 40, reason="Scale-out at 0.5R", exit="2026-03-10T12:00:00Z"),
            fill("a", 60), fill("b", -50)])
        row = report["post_cutoff"]["archetypes"]["liquidity_compression"]
        self.assertEqual(row["completed"]["n"], 2)
        self.assertEqual(row["completed"]["win_rate_pct"], 50.0)
        self.assertEqual(row["completed"]["profit_factor"], 2.0)
        self.assertEqual(row["realized_pnl"], 50.0)

    def test_open_partial_is_not_a_completed_winner(self):
        report = self.run_audit([fill("a", 40, reason="Scale-out at 0.5R")], opens=["a"])
        row = report["post_cutoff"]["archetypes"]["liquidity_compression"]
        self.assertEqual(row["completed"]["n"], 0)
        self.assertEqual(row["open_realized_pnl"], 40.0)
        self.assertEqual(row["realized_pnl"], 40.0)

    def test_unknown_ids_are_not_invented_positions(self):
        row = self.run_audit([fill("", 40), fill("", -50)])["all_history"]["totals"]
        self.assertEqual(row["unidentified_exit_rows"], 2)
        self.assertEqual(row["completed"]["n"], 0)
        self.assertEqual(row["realized_pnl"], -10.0)

    def test_entry_cutoff_does_not_use_exit_date(self):
        report = self.run_audit([fill("old", -20, entry="2026-03-08T00:00:00Z")])
        self.assertEqual(report["post_cutoff"]["totals"]["exit_rows"], 0)
        self.assertEqual(report["all_history"]["totals"]["realized_pnl"], -20.0)

    def test_zero_trade_archetypes_remain_visible(self):
        report = self.run_audit([])
        self.assertEqual(set(report["all_history"]["archetypes"]),
                         {"liquidity_compression", "whipsaw"})
        self.assertEqual(report["all_history"]["archetypes"]["whipsaw"]["exit_rows"], 0)

    def test_conflicting_position_identity_raises(self):
        with self.assertRaisesRegex(ValueError, "identity"):
            self.run_audit([fill("a", 40), fill("a", -50, arch="whipsaw")])

    def test_partial_without_matching_open_state_is_unresolved(self):
        row = self.run_audit([fill("a", 10, reason="Scale-out at 0.5R")])["all_history"]["totals"]
        self.assertEqual(row["completed"]["n"], 0)
        self.assertEqual(row["unresolved_positions"], 1)

    def test_position_win_with_losing_final_leg_stays_a_win(self):
        row = self.run_audit([
            fill("a", 100, reason="Scale-out", exit="2026-03-10T12:00:00Z"),
            fill("a", -40)])["all_history"]["totals"]
        self.assertEqual(row["completed"]["wins"], 1)
        self.assertEqual(row["completed"]["pnl"], 60.0)

    def test_nonfinite_pnl_is_rejected(self):
        with self.assertRaises(ValueError):
            self.run_audit([fill("a", float("nan"))])


if __name__ == "__main__":
    unittest.main()
