#!/usr/bin/env python3
"""
Stress Scenario Simulator for Bull Machine

Answers questions like "What if VIX +3 sigma?" or "What if DXY spikes?" by
analyzing historical periods in the feature store where similar conditions
occurred, then computing BTC drawdown, forward returns, trade win rate, and
the system's dynamic threshold behavior during those periods.

Usage:
    # Precompute all scenario stats (run once at startup):
    from bin.live.stress_simulator import StressSimulator
    sim = StressSimulator()
    stats = sim.precompute_all()

    # Check current conditions against scenarios:
    active = sim.check_current(current_features_dict)
    # -> list of active scenario dicts ready for heartbeat.json

Architecture:
    - Loads BTC_1H_FEATURES_V12_ENHANCED.parquet (283 cols, 61,306 bars)
    - Defines 6 stress scenarios using Z-score thresholds on macro indicators
    - For each scenario, identifies all historical periods where the condition held
    - Computes forward BTC returns (24h/72h/168h), max drawdown, trade statistics
    - Results are precomputed at startup, then current values are checked every
      heartbeat cycle (lightweight: just threshold comparisons)

Note on data limitations:
    - USDT.D and BTC.D are hardcoded in the feature store (known issue #14)
    - funding_rate is also hardcoded at 0.0
    - We use VIX_Z, DXY_Z, WTI_Z (oil), GOLD_Z, MOVE_Z, fear_greed_norm,
      YIELD_CURVE, and fusion_total which have real historical variation
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("stress_simulator")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Default feature store path
DEFAULT_FEATURE_STORE = (
    PROJECT_ROOT / "data" / "features_mtf" / "BTC_1H_FEATURES_V12_ENHANCED.parquet"
)

# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------
# Each scenario is a dict with:
#   name: human-readable name
#   description: what this scenario means
#   condition: callable(row) -> bool  (applied to each row of the feature store)
#   condition_desc: human-readable condition string
#   severity: base severity level (yellow/orange/red)
#   check_fields: dict of {feature_name: (operator, threshold)} for live checking
#   recommendation: what the system should do

SCENARIO_DEFINITIONS = [
    {
        "name": "VIX Spike",
        "description": "Equity volatility surges -- risk-off environment. "
                       "Historically correlates with BTC selling pressure as "
                       "institutions de-risk across all assets.",
        "condition_col": "VIX_Z",
        "condition_op": ">",
        "condition_threshold": 2.0,
        "severity": "orange",
        "recommendation": "Defensive posture -- threshold elevated, reduce position sizing",
        "icon": "chart-line",
    },
    {
        "name": "Dollar Surge",
        "description": "US Dollar strengthens sharply. Strong dollar drains "
                       "liquidity from risk assets including crypto. "
                       "Historically one of the strongest headwinds for BTC.",
        "condition_col": "DXY_Z",
        "condition_op": ">",
        "condition_threshold": 2.0,
        "severity": "orange",
        "recommendation": "Liquidity drain active -- threshold elevated, expect lower fusion scores",
    },
    {
        "name": "Oil Shock",
        "description": "Oil price spikes above 2 sigma. Energy cost shocks "
                       "create inflationary pressure and stagflation risk, "
                       "especially toxic when combined with dollar strength.",
        "condition_col": "WTI_Z",
        "condition_op": ">",
        "condition_threshold": 2.0,
        "severity": "yellow",
        "recommendation": "Monitor for stagflation (Oil + DXY both elevated). "
                          "Threshold moderately elevated.",
    },
    {
        "name": "Extreme Fear",
        "description": "Fear & Greed Index falls below 15 (normalized < 0.15). "
                       "Historically a contrarian signal -- extreme fear often "
                       "precedes recoveries, but drawdowns can extend further.",
        "condition_col": "fear_greed_norm",
        "condition_op": "<",
        "condition_threshold": 0.15,
        "severity": "red",
        "recommendation": "Extreme fear -- crisis-level threshold. System nearly "
                          "shut down. Historically contrarian bullish at 1-week horizon.",
    },
    {
        "name": "Correlation Storm",
        "description": "Multiple stress indicators align simultaneously: "
                       "VIX elevated + DXY strong + extreme fear. This is the "
                       "most dangerous macro configuration for BTC.",
        # Special: multi-column condition -- handled separately
        "condition_col": "__multi__",
        "condition_op": "multi",
        "condition_threshold": None,
        "multi_conditions": [
            ("VIX_Z", ">", 1.5),
            ("DXY_Z", ">", 1.5),
            ("fear_greed_norm", "<", 0.25),
        ],
        "severity": "red",
        "recommendation": "Maximum defensive posture. Multiple headwinds active. "
                          "Historical max drawdown during these periods is severe.",
    },
    {
        "name": "Bond Stress",
        "description": "MOVE Index (bond volatility) spikes above 2 sigma AND "
                       "yield curve is inverted. Signals credit market stress "
                       "that historically spills over into crypto.",
        "condition_col": "__multi__",
        "condition_op": "multi",
        "condition_threshold": None,
        "multi_conditions": [
            ("MOVE_Z", ">", 2.0),
            ("YIELD_CURVE", "<", -0.1),
        ],
        "severity": "orange",
        "recommendation": "Credit contagion risk. Bond stress often precedes "
                          "broader risk-off moves. Threshold elevated.",
    },
]

# Forward return horizons (in hours, since data is 1H bars)
FORWARD_HORIZONS = {
    "24h": 24,
    "72h": 72,
    "168h": 168,  # 1 week
}


class StressSimulator:
    """
    Precomputes historical stress scenario statistics from the feature store,
    then provides lightweight current-condition checking for live use.
    """

    def __init__(self, feature_store_path: Optional[str] = None):
        self.feature_store_path = Path(feature_store_path) if feature_store_path else DEFAULT_FEATURE_STORE
        self.df: Optional[pd.DataFrame] = None
        self.scenario_stats: Dict[str, Dict] = {}
        self._loaded = False

    def load_data(self) -> bool:
        """Load the feature store parquet. Returns True on success."""
        if self._loaded:
            return True

        if not self.feature_store_path.exists():
            logger.warning(
                "Feature store not found at %s. Stress simulator disabled.",
                self.feature_store_path,
            )
            return False

        try:
            cols_needed = [
                "close", "VIX_Z", "DXY_Z", "WTI_Z", "GOLD_Z", "MOVE_Z",
                "YIELD_CURVE", "fear_greed_norm", "fusion_total",
            ]
            self.df = pd.read_parquet(self.feature_store_path, columns=cols_needed)
            # Ensure sorted by timestamp
            self.df = self.df.sort_index()
            self._loaded = True
            logger.info(
                "Stress simulator loaded %d bars from %s to %s",
                len(self.df),
                self.df.index.min(),
                self.df.index.max(),
            )
            return True
        except Exception as exc:
            logger.error("Failed to load feature store for stress simulator: %s", exc)
            return False

    def _evaluate_condition(self, df: pd.DataFrame, scenario: dict) -> pd.Series:
        """Evaluate a scenario condition against the dataframe, returning a boolean Series."""
        if scenario["condition_col"] == "__multi__":
            # Multi-column condition: all must be true
            mask = pd.Series(True, index=df.index)
            for col, op, threshold in scenario["multi_conditions"]:
                if col not in df.columns:
                    mask = pd.Series(False, index=df.index)
                    break
                col_data = df[col]
                if op == ">":
                    mask = mask & (col_data > threshold)
                elif op == "<":
                    mask = mask & (col_data < threshold)
                elif op == ">=":
                    mask = mask & (col_data >= threshold)
                elif op == "<=":
                    mask = mask & (col_data <= threshold)
            return mask
        else:
            col = scenario["condition_col"]
            if col not in df.columns:
                return pd.Series(False, index=df.index)

            col_data = df[col]
            op = scenario["condition_op"]
            threshold = scenario["condition_threshold"]

            if op == ">":
                return col_data > threshold
            elif op == "<":
                return col_data < threshold
            elif op == ">=":
                return col_data >= threshold
            elif op == "<=":
                return col_data <= threshold
            else:
                return pd.Series(False, index=df.index)

    def _compute_forward_returns(self, close: pd.Series, mask: pd.Series) -> Dict[str, float]:
        """Compute average forward returns at different horizons for bars where mask is True."""
        results = {}
        scenario_indices = mask[mask].index

        for label, hours in FORWARD_HORIZONS.items():
            forward_returns = []
            for idx in scenario_indices:
                # Find the position in the close series
                pos = close.index.get_loc(idx)
                future_pos = pos + hours
                if future_pos < len(close):
                    current_price = close.iloc[pos]
                    future_price = close.iloc[future_pos]
                    if current_price > 0 and current_price == current_price:  # NaN guard
                        ret = (future_price - current_price) / current_price * 100.0
                        forward_returns.append(ret)

            if forward_returns:
                results[f"avg_return_{label}"] = round(float(np.mean(forward_returns)), 2)
                results[f"median_return_{label}"] = round(float(np.median(forward_returns)), 2)
                results[f"worst_return_{label}"] = round(float(np.min(forward_returns)), 2)
                results[f"best_return_{label}"] = round(float(np.max(forward_returns)), 2)
                results[f"pct_positive_{label}"] = round(
                    float(np.sum(np.array(forward_returns) > 0) / len(forward_returns) * 100), 1
                )
            else:
                results[f"avg_return_{label}"] = None
                results[f"median_return_{label}"] = None
                results[f"worst_return_{label}"] = None
                results[f"best_return_{label}"] = None
                results[f"pct_positive_{label}"] = None

        return results

    def _compute_max_drawdown_during(self, close: pd.Series, mask: pd.Series) -> float:
        """Compute the max drawdown that occurred during scenario periods."""
        if mask.sum() == 0:
            return 0.0

        # Get contiguous blocks of True in mask
        scenario_prices = close[mask]
        if len(scenario_prices) < 2:
            return 0.0

        # Find max drawdown within scenario periods
        peak = scenario_prices.iloc[0]
        max_dd = 0.0
        for price in scenario_prices:
            if price > peak:
                peak = price
            dd = (peak - price) / peak if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd

        return round(max_dd * 100, 2)  # as percentage

    def _compute_trade_stats_during(
        self, df: pd.DataFrame, mask: pd.Series
    ) -> Dict[str, Any]:
        """
        Estimate trade performance during scenario periods.

        Uses fusion_total as proxy: bars where fusion_total > adaptive threshold
        would have been taken. We compute a simplified win rate based on
        forward 24h returns for bars with fusion > 0.45 (approx median threshold).
        """
        if mask.sum() == 0 or "fusion_total" not in df.columns:
            return {
                "trades_eligible": 0,
                "estimated_win_rate": None,
                "avg_fusion_during": None,
            }

        scenario_df = df[mask]
        fusion = scenario_df["fusion_total"].dropna()
        close = df["close"]

        # Estimate how many bars would pass a typical threshold (0.45)
        eligible_mask = (fusion > 0.45)
        trades_eligible = int(eligible_mask.sum())

        # For eligible trades, check if 24h forward return > 0
        wins = 0
        total = 0
        eligible_indices = fusion[eligible_mask].index
        for idx in eligible_indices:
            pos = close.index.get_loc(idx)
            future_pos = pos + 24
            if future_pos < len(close):
                current = close.iloc[pos]
                future = close.iloc[future_pos]
                if current > 0 and current == current:
                    total += 1
                    if future > current:
                        wins += 1

        win_rate = round(wins / total * 100, 1) if total > 0 else None

        return {
            "trades_eligible": trades_eligible,
            "estimated_win_rate": win_rate,
            "avg_fusion_during": round(float(fusion.mean()), 3) if len(fusion) > 0 else None,
        }

    def _compute_avg_threshold_during(
        self, df: pd.DataFrame, mask: pd.Series
    ) -> Optional[float]:
        """
        Estimate the average dynamic threshold during scenario periods.

        Uses the adaptive fusion formula:
          threshold = base + (1 - risk_temp) * temp_range + instability * instab_range

        Since we don't have risk_temp/instability precomputed in the feature store,
        we approximate using available indicators. During stress (high VIX, high DXY),
        the threshold would be elevated. We compute a proxy based on the conditions.
        """
        if mask.sum() == 0:
            return None

        # Use a simple proxy: during stress, threshold is typically 0.55-0.72
        # based on production config (base=0.18, temp_range=0.48, instab_range=0.20)
        # In bull: ~0.34, in bear: ~0.63, in crisis: ~0.72
        scenario_df = df[mask]

        # Estimate risk_temp from available features
        # Higher VIX_Z and DXY_Z -> lower risk_temp -> higher threshold
        vix_z = scenario_df.get("VIX_Z", pd.Series(dtype=float))
        dxy_z = scenario_df.get("DXY_Z", pd.Series(dtype=float))
        fg = scenario_df.get("fear_greed_norm", pd.Series(dtype=float))

        # Simplified risk_temp proxy (lower = more bearish)
        # risk_temp ~ 0.5 neutral, lower in stress
        risk_temp_proxy = pd.Series(0.5, index=scenario_df.index)
        if not vix_z.empty:
            risk_temp_proxy -= np.clip(vix_z.fillna(0) * 0.05, -0.2, 0.2)
        if not dxy_z.empty:
            risk_temp_proxy -= np.clip(dxy_z.fillna(0) * 0.03, -0.15, 0.15)
        if not fg.empty:
            risk_temp_proxy += np.clip((fg.fillna(0.5) - 0.5) * 0.2, -0.15, 0.15)

        risk_temp_proxy = np.clip(risk_temp_proxy, 0.0, 1.0)

        # threshold = base + (1 - risk_temp) * temp_range
        # base=0.18, temp_range=0.48
        threshold_proxy = 0.18 + (1.0 - risk_temp_proxy) * 0.48

        return round(float(threshold_proxy.mean()), 3)

    def precompute_all(self) -> Dict[str, Dict]:
        """
        Precompute statistics for all stress scenarios.
        This is the expensive operation -- run once at startup.

        Returns dict keyed by scenario name with full stats.
        """
        if not self.load_data():
            return {}

        logger.info("Precomputing stress scenario statistics...")
        df = self.df
        close = df["close"]

        for scenario in SCENARIO_DEFINITIONS:
            name = scenario["name"]
            try:
                mask = self._evaluate_condition(df, scenario)
                occurrences = int(mask.sum())

                if occurrences == 0:
                    self.scenario_stats[name] = {
                        "name": name,
                        "description": scenario["description"],
                        "severity": scenario["severity"],
                        "recommendation": scenario["recommendation"],
                        "occurrences": 0,
                        "total_bars": len(df),
                        "pct_of_history": 0.0,
                    }
                    continue

                # Forward returns
                forward_returns = self._compute_forward_returns(close, mask)

                # Max drawdown during scenario
                max_dd = self._compute_max_drawdown_during(close, mask)

                # Trade stats
                trade_stats = self._compute_trade_stats_during(df, mask)

                # Estimated threshold
                avg_threshold = self._compute_avg_threshold_during(df, mask)

                # Time periods: find the date ranges where this scenario was active
                scenario_dates = df.index[mask]
                first_occurrence = str(scenario_dates.min())[:10]
                last_occurrence = str(scenario_dates.max())[:10]

                # Compute average duration of contiguous scenario blocks
                # A "block" is a contiguous run of True values in the mask
                block_lengths = []
                current_block = 0
                for val in mask.values:
                    if val:
                        current_block += 1
                    else:
                        if current_block > 0:
                            block_lengths.append(current_block)
                        current_block = 0
                if current_block > 0:
                    block_lengths.append(current_block)

                avg_duration_hours = (
                    round(float(np.mean(block_lengths)), 1) if block_lengths else 0.0
                )
                num_episodes = len(block_lengths)

                self.scenario_stats[name] = {
                    "name": name,
                    "description": scenario["description"],
                    "severity": scenario["severity"],
                    "recommendation": scenario["recommendation"],
                    "occurrences": occurrences,
                    "total_bars": len(df),
                    "pct_of_history": round(occurrences / len(df) * 100, 1),
                    "num_episodes": num_episodes,
                    "avg_duration_hours": avg_duration_hours,
                    "first_occurrence": first_occurrence,
                    "last_occurrence": last_occurrence,
                    "max_drawdown_pct": max_dd,
                    "avg_threshold_during": avg_threshold,
                    **forward_returns,
                    **trade_stats,
                }

                logger.info(
                    "  %s: %d occurrences (%.1f%%), %d episodes, "
                    "avg 24h return=%.2f%%, max DD=%.1f%%",
                    name,
                    occurrences,
                    occurrences / len(df) * 100,
                    num_episodes,
                    forward_returns.get("avg_return_24h", 0) or 0,
                    max_dd,
                )

            except Exception as exc:
                logger.error("Failed to compute stats for scenario '%s': %s", name, exc)
                self.scenario_stats[name] = {
                    "name": name,
                    "description": scenario["description"],
                    "severity": scenario["severity"],
                    "recommendation": scenario["recommendation"],
                    "error": str(exc),
                }

        logger.info("Stress scenario precomputation complete: %d scenarios", len(self.scenario_stats))
        return self.scenario_stats

    def check_current(self, features: dict) -> List[Dict[str, Any]]:
        """
        Check current feature values against all scenario thresholds.
        This is the lightweight operation -- just threshold comparisons.

        Args:
            features: dict-like with keys matching column names
                      (VIX_Z, DXY_Z, WTI_Z, GOLD_Z, fear_greed_norm, etc.)

        Returns:
            List of active scenario dicts suitable for heartbeat.json
        """
        active_scenarios = []

        for scenario in SCENARIO_DEFINITIONS:
            name = scenario["name"]
            stats = self.scenario_stats.get(name, {})

            try:
                is_active = False
                current_values = {}

                if scenario["condition_col"] == "__multi__":
                    # Multi-condition: all must be true
                    all_met = True
                    for col, op, threshold in scenario["multi_conditions"]:
                        # Map feature store column names to live feature names
                        val = self._get_feature_value(features, col)
                        if val is None:
                            all_met = False
                            break
                        current_values[col] = round(val, 4)
                        if op == ">" and not (val > threshold):
                            all_met = False
                        elif op == "<" and not (val < threshold):
                            all_met = False
                        elif op == ">=" and not (val >= threshold):
                            all_met = False
                        elif op == "<=" and not (val <= threshold):
                            all_met = False
                    is_active = all_met
                else:
                    col = scenario["condition_col"]
                    val = self._get_feature_value(features, col)
                    if val is not None:
                        current_values[col] = round(val, 4)
                        op = scenario["condition_op"]
                        threshold = scenario["condition_threshold"]
                        if op == ">" and val > threshold:
                            is_active = True
                        elif op == "<" and val < threshold:
                            is_active = True
                        elif op == ">=" and val >= threshold:
                            is_active = True
                        elif op == "<=" and val <= threshold:
                            is_active = True

                # Build the output dict
                scenario_result = {
                    "name": name,
                    "active": is_active,
                    "severity": scenario["severity"] if is_active else "inactive",
                    "current_values": current_values,
                    "description": scenario["description"],
                    "recommendation": scenario["recommendation"] if is_active else "",
                }

                # Add historical context from precomputed stats
                if stats and is_active:
                    scenario_result["historical"] = {
                        "occurrences": stats.get("occurrences", 0),
                        "pct_of_history": stats.get("pct_of_history", 0),
                        "num_episodes": stats.get("num_episodes", 0),
                        "avg_duration_hours": stats.get("avg_duration_hours", 0),
                        "avg_return_24h": stats.get("avg_return_24h"),
                        "avg_return_72h": stats.get("avg_return_72h"),
                        "avg_return_168h": stats.get("avg_return_168h"),
                        "median_return_24h": stats.get("median_return_24h"),
                        "worst_return_24h": stats.get("worst_return_24h"),
                        "pct_positive_24h": stats.get("pct_positive_24h"),
                        "pct_positive_168h": stats.get("pct_positive_168h"),
                        "max_drawdown_pct": stats.get("max_drawdown_pct"),
                        "avg_threshold_during": stats.get("avg_threshold_during"),
                        "estimated_win_rate": stats.get("estimated_win_rate"),
                    }

                active_scenarios.append(scenario_result)

            except Exception as exc:
                logger.debug("Error checking scenario '%s': %s", name, exc)

        return active_scenarios

    @staticmethod
    def _get_feature_value(features: dict, col: str) -> Optional[float]:
        """
        Get a feature value from the features dict, handling multiple naming
        conventions (feature store vs live feature computer).
        """
        # Direct lookup
        val = features.get(col)

        # Try alternative names (live feature computer uses different keys)
        if val is None:
            alternatives = {
                "VIX_Z": ["vix_z", "VIX_Z"],
                "DXY_Z": ["dxy_z", "DXY_Z"],
                "WTI_Z": ["wti_z", "WTI_Z", "OIL_Z", "oil_z"],
                "GOLD_Z": ["gold_z", "GOLD_Z"],
                "MOVE_Z": ["move_z", "MOVE_Z"],
                "YIELD_CURVE": ["yield_curve", "YIELD_CURVE"],
                "fear_greed_norm": ["fear_greed_norm", "FEAR_GREED_NORM"],
            }
            for alt in alternatives.get(col, []):
                val = features.get(alt)
                if val is not None:
                    break

        if val is None:
            return None

        try:
            fval = float(val)
            if fval != fval:  # NaN check
                return None
            return fval
        except (TypeError, ValueError):
            return None

    def get_full_report(self) -> Dict[str, Any]:
        """
        Return the full precomputed scenario stats dict.
        Suitable for export or detailed analysis.
        """
        return {
            "scenarios": self.scenario_stats,
            "feature_store": str(self.feature_store_path),
            "total_bars": len(self.df) if self.df is not None else 0,
            "date_range": {
                "start": str(self.df.index.min()) if self.df is not None else None,
                "end": str(self.df.index.max()) if self.df is not None else None,
            },
        }


# ---------------------------------------------------------------------------
# CLI for standalone testing
# ---------------------------------------------------------------------------

def main():
    """Run stress simulator standalone for testing/analysis."""
    import json

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    sim = StressSimulator()
    stats = sim.precompute_all()

    if not stats:
        print("No stats computed. Check that the feature store exists.")
        return

    print("\n" + "=" * 80)
    print("STRESS SCENARIO ANALYSIS -- Historical Statistics (2018-2024)")
    print("=" * 80)

    for name, s in stats.items():
        print(f"\n--- {name} ---")
        print(f"  Severity:         {s.get('severity', '?')}")
        print(f"  Occurrences:      {s.get('occurrences', 0):,} bars "
              f"({s.get('pct_of_history', 0):.1f}% of history)")
        print(f"  Episodes:         {s.get('num_episodes', 0)}")
        print(f"  Avg Duration:     {s.get('avg_duration_hours', 0):.1f} hours")
        if s.get('first_occurrence'):
            print(f"  Date Range:       {s.get('first_occurrence')} to {s.get('last_occurrence')}")
        print(f"  Max Drawdown:     {s.get('max_drawdown_pct', 0):.1f}%")
        print(f"  Avg Threshold:    {s.get('avg_threshold_during', '?')}")
        print(f"  Forward Returns:")
        for horizon in ["24h", "72h", "168h"]:
            avg = s.get(f"avg_return_{horizon}")
            med = s.get(f"median_return_{horizon}")
            worst = s.get(f"worst_return_{horizon}")
            pct_pos = s.get(f"pct_positive_{horizon}")
            if avg is not None:
                print(f"    {horizon}: avg={avg:+.2f}%, median={med:+.2f}%, "
                      f"worst={worst:+.2f}%, {pct_pos:.0f}% positive")
            else:
                print(f"    {horizon}: insufficient data")
        print(f"  Trade Stats:")
        print(f"    Eligible trades:  {s.get('trades_eligible', 0)}")
        print(f"    Est. win rate:    {s.get('estimated_win_rate', '?')}%")
        print(f"    Avg fusion:       {s.get('avg_fusion_during', '?')}")
        print(f"  Recommendation:   {s.get('recommendation', '')}")

    # Test current condition check with sample values
    print("\n" + "=" * 80)
    print("LIVE CHECK TEST -- Simulated extreme conditions")
    print("=" * 80)

    test_features = {
        "VIX_Z": 3.5,
        "DXY_Z": 2.1,
        "WTI_Z": 1.0,
        "GOLD_Z": 1.5,
        "MOVE_Z": 2.5,
        "YIELD_CURVE": -0.2,
        "fear_greed_norm": 0.10,
    }
    active = sim.check_current(test_features)
    print(f"\nTest features: {test_features}")
    print(f"\nActive scenarios ({sum(1 for a in active if a['active'])} of {len(active)}):")
    for a in active:
        status = "ACTIVE" if a["active"] else "inactive"
        print(f"  [{status}] {a['name']} (severity={a['severity']})")
        if a["active"] and "historical" in a:
            h = a["historical"]
            print(f"         24h return: {h.get('avg_return_24h', '?')}%  |  "
                  f"win rate: {h.get('estimated_win_rate', '?')}%  |  "
                  f"max DD: {h.get('max_drawdown_pct', '?')}%")

    # Export full report
    report_path = PROJECT_ROOT / "results" / "stress_scenario_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(sim.get_full_report(), f, indent=2, default=str)
    print(f"\nFull report saved to: {report_path}")


if __name__ == "__main__":
    main()
