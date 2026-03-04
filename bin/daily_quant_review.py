#!/usr/bin/env python3
"""
Daily Quantitative Trade Review for Bull Machine

Reads output files from the Coinbase paper trading engine and produces
a structured analysis report covering 8 sections:
  1. Daily Summary (price range, regime distribution, equity change, drawdown)
  2. Trade Analysis (position-level PnL, WR, PF, best/worst)
  3. Per-Archetype Breakdown (table with WR, PF, avg duration)
  4. Exit Analysis (exit_reason counts, SL%, avg R-multiple)
  5. Threshold Margin Analysis (bypass vs passing WR/PF)
  6. CMI Regime Analysis (per-regime WR/PF, risk_temp/instability vs PnL)
  7. Could Have Done Better (premature SL detection)
  8. Actionable Recommendations (disable/boost/warning rules as JSON)

Outputs:
  - daily_review.json   (full structured report)
  - daily_review.md     (human-readable markdown)
  - daily_reviews/YYYY-MM-DD.json (archived copy)

Usage:
    python3 bin/daily_quant_review.py [options]
      --data-dir PATH    Data directory (default: results/coinbase_paper/)
      --date YYYY-MM-DD  Date to analyze (default: today)
      --days N           Number of days to look back (default: 1)
      --output-dir PATH  Output directory (default: same as data-dir)

Author: Claude Code (System Architect)
Date: 2026-03-03
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("daily_quant_review")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRADES_FILE = "trades.json"
SIGNALS_FILE = "signals.csv"
EQUITY_FILE = "equity_history.csv"
HEARTBEAT_FILE = "heartbeat.json"
STATE_FILE = "state.json"

ARCHIVE_DIR = "daily_reviews"

# Recommendation thresholds
MIN_TRADES_FOR_FLAG = 5
WR_DISABLE_THRESHOLD = 0.30          # WR < 30% over 5+ trades -> recommend disable
WR_BOOST_THRESHOLD = 0.70            # WR > 70% over 10+ trades -> recommend boost
MIN_TRADES_FOR_BOOST = 10
BYPASS_WR_PENALTY_THRESHOLD = 0.20   # bypass WR < passing WR by 20pp -> disable bypass


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_trades(data_dir: Path) -> pd.DataFrame:
    """Load trades.json into a DataFrame. Returns empty DF if missing."""
    path = data_dir / TRADES_FILE
    if not path.exists():
        logger.warning("trades.json not found at %s", path)
        return pd.DataFrame()
    try:
        with open(path) as f:
            raw = json.load(f)
        if not raw:
            logger.info("trades.json is empty")
            return pd.DataFrame()
        df = pd.DataFrame(raw)
        # Parse timestamps
        for col in ("timestamp_entry", "timestamp_exit"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        return df
    except (json.JSONDecodeError, Exception) as exc:
        logger.warning("Failed to load trades.json: %s", exc)
        return pd.DataFrame()


def load_signals(data_dir: Path) -> pd.DataFrame:
    """Load signals.csv into a DataFrame. Returns empty DF if missing."""
    path = data_dir / SIGNALS_FILE
    if not path.exists():
        logger.warning("signals.csv not found at %s", path)
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, parse_dates=["timestamp"])
        return df
    except Exception as exc:
        logger.warning("Failed to load signals.csv: %s", exc)
        return pd.DataFrame()


def load_equity(data_dir: Path) -> pd.DataFrame:
    """Load equity_history.csv into a DataFrame. Returns empty DF if missing."""
    path = data_dir / EQUITY_FILE
    if not path.exists():
        logger.warning("equity_history.csv not found at %s", path)
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, parse_dates=["timestamp"])
        return df
    except Exception as exc:
        logger.warning("Failed to load equity_history.csv: %s", exc)
        return pd.DataFrame()


def load_heartbeat(data_dir: Path) -> dict:
    """Load heartbeat.json. Returns empty dict if missing."""
    path = data_dir / HEARTBEAT_FILE
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, Exception):
        return {}


# ---------------------------------------------------------------------------
# Filtering helpers
# ---------------------------------------------------------------------------

def filter_by_date_range(
    df: pd.DataFrame,
    date_col: str,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    """Filter dataframe rows where date_col falls in [start, end)."""
    if df.empty or date_col not in df.columns:
        return df
    mask = (df[date_col] >= start) & (df[date_col] < end)
    return df.loc[mask].copy()


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    """Safe division, returns default when denominator is zero or NaN."""
    if b == 0 or (b != b):
        return default
    result = a / b
    if result != result:
        return default
    return result


def compute_pf(wins_total: float, losses_total: float) -> float:
    """Compute profit factor = gross_wins / gross_losses."""
    if losses_total == 0:
        return float("inf") if wins_total > 0 else 0.0
    return safe_div(abs(wins_total), abs(losses_total), default=0.0)


# ---------------------------------------------------------------------------
# Section 1: Daily Summary
# ---------------------------------------------------------------------------

def section_daily_summary(
    equity: pd.DataFrame,
    trades: pd.DataFrame,
    start: datetime,
    end: datetime,
) -> dict:
    """Build daily summary: price range, regime distribution, equity change, drawdown."""
    result = {
        "date_range": f"{start.strftime('%Y-%m-%d')} to {(end - timedelta(seconds=1)).strftime('%Y-%m-%d')}",
        "btc_price_high": None,
        "btc_price_low": None,
        "btc_price_open": None,
        "btc_price_close": None,
        "regime_distribution": {},
        "equity_start": None,
        "equity_end": None,
        "equity_change_usd": None,
        "equity_change_pct": None,
        "max_intraday_drawdown_pct": None,
        "total_bars": 0,
    }

    eq = filter_by_date_range(equity, "timestamp", start, end)

    if eq.empty:
        logger.info("No equity data for the selected date range")
        return result

    result["total_bars"] = len(eq)

    # BTC price range
    if "btc_price" in eq.columns:
        result["btc_price_high"] = round(float(eq["btc_price"].max()), 2)
        result["btc_price_low"] = round(float(eq["btc_price"].min()), 2)
        result["btc_price_open"] = round(float(eq["btc_price"].iloc[0]), 2)
        result["btc_price_close"] = round(float(eq["btc_price"].iloc[-1]), 2)

    # Equity change
    if "equity" in eq.columns:
        eq_start = float(eq["equity"].iloc[0])
        eq_end = float(eq["equity"].iloc[-1])
        result["equity_start"] = round(eq_start, 2)
        result["equity_end"] = round(eq_end, 2)
        result["equity_change_usd"] = round(eq_end - eq_start, 2)
        result["equity_change_pct"] = round(
            safe_div(eq_end - eq_start, eq_start) * 100, 4
        )

        # Max intraday drawdown
        running_max = eq["equity"].cummax()
        drawdowns = (eq["equity"] - running_max) / running_max
        result["max_intraday_drawdown_pct"] = round(float(drawdowns.min()) * 100, 4)

    # Regime distribution
    if "regime" in eq.columns:
        counts = eq["regime"].value_counts()
        total = counts.sum()
        result["regime_distribution"] = {
            str(k): round(v / total * 100, 1) for k, v in counts.items()
        }

    return result


# ---------------------------------------------------------------------------
# Section 2: Trade Analysis
# ---------------------------------------------------------------------------

def _group_key(row: dict) -> str:
    """Build a unique position grouping key."""
    pid = row.get("position_id", "")
    if pid:
        return str(pid)
    # Fallback: entry timestamp + archetype
    ts = str(row.get("timestamp_entry", ""))
    arch = str(row.get("archetype", ""))
    return f"{ts}_{arch}"


def section_trade_analysis(trades: pd.DataFrame) -> dict:
    """Trade analysis: position-level grouping, WR, PF, best/worst."""
    result = {
        "unique_positions": 0,
        "total_exit_rows": 0,
        "total_pnl": 0.0,
        "win_rate": 0.0,
        "profit_factor": 0.0,
        "avg_win": 0.0,
        "avg_loss": 0.0,
        "best_trade": None,
        "worst_trade": None,
        "positions": [],
    }

    if trades.empty:
        return result

    result["total_exit_rows"] = len(trades)

    # Group exits by position
    trades = trades.copy()
    trades["_group_key"] = trades.apply(
        lambda r: _group_key(r.to_dict()), axis=1
    )

    grouped = trades.groupby("_group_key")
    positions = []
    for key, group in grouped:
        pnl_sum = float(group["pnl"].sum()) if "pnl" in group.columns else 0.0
        first = group.iloc[0]
        pos = {
            "position_id": key,
            "archetype": str(first.get("archetype", "")),
            "direction": str(first.get("direction", "")),
            "entry_price": float(first.get("entry_price", 0)),
            "pnl": round(pnl_sum, 2),
            "exit_count": len(group),
            "entry_regime": str(first.get("entry_regime", "")),
            "fusion_score": float(first.get("fusion_score", 0)),
            "duration_hours": float(first.get("duration_hours", 0)),
        }
        positions.append(pos)

    result["unique_positions"] = len(positions)
    result["positions"] = positions

    # Win/loss stats
    pnls = [p["pnl"] for p in positions]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    result["total_pnl"] = round(sum(pnls), 2)
    result["win_rate"] = round(safe_div(len(wins), len(pnls)) * 100, 1)
    result["profit_factor"] = round(
        compute_pf(sum(wins) if wins else 0, sum(losses) if losses else 0), 2
    )
    result["avg_win"] = round(safe_div(sum(wins), len(wins)), 2) if wins else 0.0
    result["avg_loss"] = round(safe_div(sum(losses), len(losses)), 2) if losses else 0.0

    # Best / worst
    if positions:
        best = max(positions, key=lambda p: p["pnl"])
        worst = min(positions, key=lambda p: p["pnl"])
        result["best_trade"] = best
        result["worst_trade"] = worst

    return result


# ---------------------------------------------------------------------------
# Section 3: Per-Archetype Breakdown
# ---------------------------------------------------------------------------

def section_archetype_breakdown(trades: pd.DataFrame) -> dict:
    """Per-archetype table: trades, wins, losses, WR, PF, total PnL, avg duration."""
    result = {"archetypes": [], "flagged": []}

    if trades.empty or "archetype" not in trades.columns:
        return result

    # Group by position first, then by archetype
    trades = trades.copy()
    trades["_group_key"] = trades.apply(
        lambda r: _group_key(r.to_dict()), axis=1
    )

    # Build position-level records
    pos_records = []
    for key, group in trades.groupby("_group_key"):
        first = group.iloc[0]
        pos_records.append({
            "archetype": str(first.get("archetype", "")),
            "pnl": float(group["pnl"].sum()),
            "duration_hours": float(first.get("duration_hours", 0)),
        })

    pos_df = pd.DataFrame(pos_records)
    if pos_df.empty:
        return result

    archetypes = []
    flagged = []
    for arch, grp in pos_df.groupby("archetype"):
        n = len(grp)
        wins = int((grp["pnl"] > 0).sum())
        losses = n - wins
        total_pnl = round(float(grp["pnl"].sum()), 2)
        gross_wins = float(grp.loc[grp["pnl"] > 0, "pnl"].sum())
        gross_losses = float(grp.loc[grp["pnl"] <= 0, "pnl"].sum())
        wr = round(safe_div(wins, n) * 100, 1)
        pf = round(compute_pf(gross_wins, gross_losses), 2)
        avg_dur = round(float(grp["duration_hours"].mean()), 1)

        entry = {
            "archetype": str(arch),
            "trades": n,
            "wins": wins,
            "losses": losses,
            "win_rate_pct": wr,
            "profit_factor": pf,
            "total_pnl": total_pnl,
            "avg_duration_hours": avg_dur,
        }
        archetypes.append(entry)

        # Flag poor archetypes
        if n >= MIN_TRADES_FOR_FLAG and wr < WR_DISABLE_THRESHOLD * 100:
            flagged.append({
                "archetype": str(arch),
                "reason": f"WR={wr}% over {n} trades (below {WR_DISABLE_THRESHOLD*100}% threshold)",
            })

    # Sort by total PnL descending
    archetypes.sort(key=lambda x: x["total_pnl"], reverse=True)
    result["archetypes"] = archetypes
    result["flagged"] = flagged
    return result


# ---------------------------------------------------------------------------
# Section 4: Exit Analysis
# ---------------------------------------------------------------------------

def section_exit_analysis(trades: pd.DataFrame) -> dict:
    """Exit reason distribution, SL%, avg R-multiple."""
    result = {
        "exit_reason_counts": {},
        "pct_hit_sl_before_scaleout": None,
        "avg_r_multiple": None,
        "total_exits": 0,
    }

    if trades.empty:
        return result

    result["total_exits"] = len(trades)

    # Count by exit_reason
    if "exit_reason" in trades.columns:
        counts = trades["exit_reason"].value_counts().to_dict()
        result["exit_reason_counts"] = {str(k): int(v) for k, v in counts.items()}

    # % positions that hit SL before any scale-out
    # Group by position; if the first exit row is stop_loss, it hit SL before scale-out
    if "exit_reason" in trades.columns:
        trades_copy = trades.copy()
        trades_copy["_group_key"] = trades_copy.apply(
            lambda r: _group_key(r.to_dict()), axis=1
        )
        sl_first_count = 0
        total_positions = 0
        for _, group in trades_copy.groupby("_group_key"):
            total_positions += 1
            group_sorted = group.sort_values("timestamp_exit")
            first_reason = str(group_sorted.iloc[0].get("exit_reason", ""))
            if "stop_loss" in first_reason.lower():
                sl_first_count += 1
        if total_positions > 0:
            result["pct_hit_sl_before_scaleout"] = round(
                sl_first_count / total_positions * 100, 1
            )

    # Avg R-multiple: pnl / risk_per_trade
    # risk_per_trade = |entry_price - stop_loss| * quantity
    if all(c in trades.columns for c in ("pnl", "entry_price", "stop_loss", "quantity")):
        r_multiples = []
        for _, row in trades.iterrows():
            entry = row.get("entry_price", 0)
            sl = row.get("stop_loss", 0)
            qty = row.get("quantity", 0)
            pnl = row.get("pnl", 0)
            # Guard NaN
            if entry != entry or sl != sl or qty != qty or pnl != pnl:
                continue
            risk = abs(entry - sl) * abs(qty) if sl != 0 else 0
            if risk > 0:
                r_multiples.append(pnl / risk)
        if r_multiples:
            result["avg_r_multiple"] = round(float(np.mean(r_multiples)), 3)

    return result


# ---------------------------------------------------------------------------
# Section 5: Threshold Margin Analysis
# ---------------------------------------------------------------------------

def section_threshold_margin(trades: pd.DataFrame) -> dict:
    """Split trades by would_have_passed; compare WR and PF."""
    result = {
        "passing_trades": {"count": 0, "win_rate_pct": None, "profit_factor": None, "total_pnl": 0.0},
        "bypass_trades": {"count": 0, "win_rate_pct": None, "profit_factor": None, "total_pnl": 0.0},
        "verdict": "Insufficient data",
    }

    if trades.empty or "would_have_passed" not in trades.columns:
        return result

    # Position-level aggregation
    trades_copy = trades.copy()
    trades_copy["_group_key"] = trades_copy.apply(
        lambda r: _group_key(r.to_dict()), axis=1
    )

    # Build position records with would_have_passed from first exit row
    pos_records = []
    for key, group in trades_copy.groupby("_group_key"):
        first = group.iloc[0]
        whp = first.get("would_have_passed")
        # Handle various truthy values
        if isinstance(whp, str):
            whp = whp.lower() in ("true", "1", "yes")
        elif isinstance(whp, (int, float)):
            whp = bool(whp)
        else:
            whp = bool(whp) if whp is not None else True
        pos_records.append({
            "pnl": float(group["pnl"].sum()),
            "would_have_passed": whp,
        })

    if not pos_records:
        return result

    pos_df = pd.DataFrame(pos_records)
    passing = pos_df[pos_df["would_have_passed"] == True]  # noqa: E712
    bypass = pos_df[pos_df["would_have_passed"] == False]  # noqa: E712

    def _stats(df):
        n = len(df)
        if n == 0:
            return {"count": 0, "win_rate_pct": None, "profit_factor": None, "total_pnl": 0.0}
        wins = int((df["pnl"] > 0).sum())
        wr = round(wins / n * 100, 1)
        gw = float(df.loc[df["pnl"] > 0, "pnl"].sum())
        gl = float(df.loc[df["pnl"] <= 0, "pnl"].sum())
        pf = round(compute_pf(gw, gl), 2)
        return {"count": n, "win_rate_pct": wr, "profit_factor": pf, "total_pnl": round(float(df["pnl"].sum()), 2)}

    result["passing_trades"] = _stats(passing)
    result["bypass_trades"] = _stats(bypass)

    # Verdict
    pass_wr = result["passing_trades"]["win_rate_pct"]
    byp_wr = result["bypass_trades"]["win_rate_pct"]
    if pass_wr is not None and byp_wr is not None:
        result["verdict"] = (
            f"Threshold-passing: {pass_wr}% WR, "
            f"Bypass-only: {byp_wr}% WR"
        )
    elif pass_wr is not None:
        result["verdict"] = f"All trades passed threshold ({pass_wr}% WR). No bypass trades."
    else:
        result["verdict"] = "Insufficient data"

    return result


# ---------------------------------------------------------------------------
# Section 6: CMI Regime Analysis
# ---------------------------------------------------------------------------

def section_cmi_regime(trades: pd.DataFrame) -> dict:
    """Per-regime WR/PF + risk_temp/instability vs PnL correlation."""
    result = {
        "per_regime": {},
        "risk_temp_vs_pnl": {},
        "instability_vs_pnl": {},
    }

    if trades.empty:
        return result

    # Position-level aggregation
    trades_copy = trades.copy()
    trades_copy["_group_key"] = trades_copy.apply(
        lambda r: _group_key(r.to_dict()), axis=1
    )

    pos_records = []
    for key, group in trades_copy.groupby("_group_key"):
        first = group.iloc[0]
        pos_records.append({
            "pnl": float(group["pnl"].sum()),
            "entry_regime": str(first.get("entry_regime", "unknown")),
            "risk_temp_at_entry": float(first.get("risk_temp_at_entry", 0.5)),
            "instability_at_entry": float(first.get("instability_at_entry", 0.5)),
        })

    if not pos_records:
        return result

    pos_df = pd.DataFrame(pos_records)

    # Per-regime
    for regime, grp in pos_df.groupby("entry_regime"):
        n = len(grp)
        wins = int((grp["pnl"] > 0).sum())
        wr = round(wins / n * 100, 1) if n > 0 else 0.0
        gw = float(grp.loc[grp["pnl"] > 0, "pnl"].sum())
        gl = float(grp.loc[grp["pnl"] <= 0, "pnl"].sum())
        pf = round(compute_pf(gw, gl), 2)
        result["per_regime"][str(regime)] = {
            "trades": n,
            "wins": wins,
            "losses": n - wins,
            "win_rate_pct": wr,
            "profit_factor": pf,
            "total_pnl": round(float(grp["pnl"].sum()), 2),
        }

    # Quartile analysis for risk_temp
    def _quartile_analysis(df, col):
        if col not in df.columns or df[col].isna().all():
            return {}
        try:
            df = df.dropna(subset=[col, "pnl"])
            if len(df) < 4:
                return {"note": "Too few trades for quartile analysis"}
            quartiles = pd.qcut(df[col], q=4, duplicates="drop")
            buckets = {}
            for label, grp in df.groupby(quartiles, observed=True):
                n = len(grp)
                avg_pnl = round(float(grp["pnl"].mean()), 2)
                wr = round(float((grp["pnl"] > 0).mean()) * 100, 1)
                buckets[str(label)] = {
                    "trades": n,
                    "avg_pnl": avg_pnl,
                    "win_rate_pct": wr,
                }
            return buckets
        except Exception:
            return {"note": "Could not compute quartile analysis"}

    result["risk_temp_vs_pnl"] = _quartile_analysis(pos_df, "risk_temp_at_entry")
    result["instability_vs_pnl"] = _quartile_analysis(pos_df, "instability_at_entry")

    return result


# ---------------------------------------------------------------------------
# Section 7: Could Have Done Better
# ---------------------------------------------------------------------------

def section_could_have_done_better(
    trades: pd.DataFrame, equity: pd.DataFrame
) -> dict:
    """Check SL trades: did BTC recover within 24h after exit?"""
    result = {
        "premature_sl_count": 0,
        "total_sl_trades": 0,
        "premature_sl_trades": [],
    }

    if trades.empty or equity.empty:
        return result

    if "exit_reason" not in trades.columns:
        return result

    # Filter to stop_loss trades
    sl_trades = trades[
        trades["exit_reason"].str.contains("stop_loss", case=False, na=False)
    ].copy()

    if sl_trades.empty:
        return result

    result["total_sl_trades"] = len(sl_trades)

    # Ensure equity is sorted by timestamp
    eq = equity.sort_values("timestamp").copy()

    for _, row in sl_trades.iterrows():
        exit_ts = row.get("timestamp_exit")
        entry_price = row.get("entry_price", 0)
        direction = str(row.get("direction", "long")).lower()

        if exit_ts is None or entry_price != entry_price:
            continue
        if not isinstance(exit_ts, pd.Timestamp):
            exit_ts = pd.to_datetime(exit_ts, errors="coerce")
        if pd.isna(exit_ts):
            continue

        # Look at BTC price in the 24h after exit
        window_start = exit_ts
        window_end = exit_ts + timedelta(hours=24)
        window = eq[
            (eq["timestamp"] >= window_start)
            & (eq["timestamp"] <= window_end)
        ]

        if window.empty or "btc_price" not in window.columns:
            continue

        # Check if price recovered to entry price
        if direction == "long":
            recovered = float(window["btc_price"].max()) >= entry_price
        else:
            recovered = float(window["btc_price"].min()) <= entry_price

        if recovered:
            result["premature_sl_count"] += 1
            result["premature_sl_trades"].append({
                "timestamp_exit": str(exit_ts),
                "archetype": str(row.get("archetype", "")),
                "entry_price": float(entry_price),
                "exit_price": float(row.get("exit_price", 0)),
                "pnl": float(row.get("pnl", 0)),
            })

    return result


# ---------------------------------------------------------------------------
# Section 8: Actionable Recommendations
# ---------------------------------------------------------------------------

def section_recommendations(
    archetype_data: dict,
    threshold_data: dict,
    regime_data: dict,
) -> dict:
    """Generate rule-based recommendations."""
    recommendations = []

    # Rule 1: archetype WR < 30% over 5+ trades -> recommend disable
    for arch in archetype_data.get("archetypes", []):
        if arch["trades"] >= MIN_TRADES_FOR_FLAG and arch["win_rate_pct"] < WR_DISABLE_THRESHOLD * 100:
            recommendations.append({
                "type": "disable_archetype",
                "archetype": arch["archetype"],
                "severity": "high",
                "reason": (
                    f"{arch['archetype']} has {arch['win_rate_pct']}% WR over "
                    f"{arch['trades']} trades (threshold: {WR_DISABLE_THRESHOLD*100}%)"
                ),
                "action": f"Add '{arch['archetype']}' to disabled_archetypes in config",
            })

    # Rule 2: bypass trades WR < passing trades WR by 20pp -> disable bypass
    pass_wr = threshold_data.get("passing_trades", {}).get("win_rate_pct")
    byp_wr = threshold_data.get("bypass_trades", {}).get("win_rate_pct")
    byp_count = threshold_data.get("bypass_trades", {}).get("count", 0)
    if pass_wr is not None and byp_wr is not None and byp_count >= 3:
        if pass_wr - byp_wr >= BYPASS_WR_PENALTY_THRESHOLD * 100:
            recommendations.append({
                "type": "disable_bypass",
                "severity": "medium",
                "reason": (
                    f"Bypass trades WR ({byp_wr}%) is {pass_wr - byp_wr:.1f}pp below "
                    f"passing trades WR ({pass_wr}%)"
                ),
                "action": "Consider disabling bypass mode in adaptive_fusion config",
            })

    # Rule 3: all trades in a regime are losses -> regime warning
    for regime, stats in regime_data.get("per_regime", {}).items():
        if stats["trades"] >= 3 and stats["wins"] == 0:
            recommendations.append({
                "type": "regime_warning",
                "regime": regime,
                "severity": "high",
                "reason": (
                    f"All {stats['trades']} trades in '{regime}' regime are losses "
                    f"(total PnL: ${stats['total_pnl']:.2f})"
                ),
                "action": f"Consider blocking entries during '{regime}' regime",
            })

    # Rule 4: archetype WR > 70% over 10+ trades -> recommend boost
    for arch in archetype_data.get("archetypes", []):
        if arch["trades"] >= MIN_TRADES_FOR_BOOST and arch["win_rate_pct"] > WR_BOOST_THRESHOLD * 100:
            recommendations.append({
                "type": "boost_archetype",
                "archetype": arch["archetype"],
                "severity": "low",
                "reason": (
                    f"{arch['archetype']} has {arch['win_rate_pct']}% WR over "
                    f"{arch['trades']} trades (PF={arch['profit_factor']})"
                ),
                "action": f"Consider lowering base_threshold for '{arch['archetype']}' to generate more signals",
            })

    return {"recommendations": recommendations, "count": len(recommendations)}


# ---------------------------------------------------------------------------
# Report Assembly
# ---------------------------------------------------------------------------

def build_report(
    data_dir: Path,
    target_date: datetime,
    days: int,
) -> dict:
    """Build the full 8-section report."""
    start = target_date.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
    end = start + timedelta(days=days)

    logger.info("Building report for %s to %s (%d day(s))", start.date(), (end - timedelta(days=1)).date(), days)

    # Load all data
    all_trades = load_trades(data_dir)
    all_signals = load_signals(data_dir)
    all_equity = load_equity(data_dir)
    heartbeat = load_heartbeat(data_dir)

    # Filter to date range
    trades = filter_by_date_range(all_trades, "timestamp_entry", start, end)
    signals = filter_by_date_range(all_signals, "timestamp", start, end)
    equity = filter_by_date_range(all_equity, "timestamp", start, end)

    logger.info(
        "Data loaded: %d trades, %d signals, %d equity bars (in range)",
        len(trades), len(signals), len(equity),
    )

    # Build sections
    s1 = section_daily_summary(all_equity, trades, start, end)
    s2 = section_trade_analysis(trades)
    s3 = section_archetype_breakdown(trades)
    s4 = section_exit_analysis(trades)
    s5 = section_threshold_margin(trades)
    s6 = section_cmi_regime(trades)
    s7 = section_could_have_done_better(trades, all_equity)
    s8 = section_recommendations(s3, s5, s6)

    report = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "data_dir": str(data_dir),
        "date_range": {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "days": days,
        },
        "data_stats": {
            "total_trades_in_file": len(all_trades),
            "trades_in_range": len(trades),
            "signals_in_range": len(signals),
            "equity_bars_in_range": len(equity),
        },
        "sections": {
            "1_daily_summary": s1,
            "2_trade_analysis": s2,
            "3_archetype_breakdown": s3,
            "4_exit_analysis": s4,
            "5_threshold_margin": s5,
            "6_cmi_regime": s6,
            "7_could_have_done_better": s7,
            "8_recommendations": s8,
        },
    }

    return report


# ---------------------------------------------------------------------------
# Markdown Rendering
# ---------------------------------------------------------------------------

def render_markdown(report: dict) -> str:
    """Render the report as human-readable markdown."""
    lines = []
    sections = report.get("sections", {})
    dr = report.get("date_range", {})

    lines.append("# Bull Machine Daily Quant Review")
    lines.append("")
    lines.append(f"**Generated**: {report.get('generated_at', 'N/A')}")
    lines.append(f"**Period**: {dr.get('start', '?')[:10]} to {dr.get('end', '?')[:10]} ({dr.get('days', '?')} day(s))")
    lines.append(f"**Data**: {report.get('data_stats', {}).get('trades_in_range', 0)} trades, "
                 f"{report.get('data_stats', {}).get('signals_in_range', 0)} signals, "
                 f"{report.get('data_stats', {}).get('equity_bars_in_range', 0)} equity bars")
    lines.append("")

    # --- Section 1: Daily Summary ---
    s1 = sections.get("1_daily_summary", {})
    lines.append("## 1. Daily Summary")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Date Range | {s1.get('date_range', 'N/A')} |")
    lines.append(f"| BTC Open | ${s1.get('btc_price_open', 'N/A'):,} |" if s1.get('btc_price_open') else "| BTC Open | N/A |")
    lines.append(f"| BTC Close | ${s1.get('btc_price_close', 'N/A'):,} |" if s1.get('btc_price_close') else "| BTC Close | N/A |")
    lines.append(f"| BTC High | ${s1.get('btc_price_high', 'N/A'):,} |" if s1.get('btc_price_high') else "| BTC High | N/A |")
    lines.append(f"| BTC Low | ${s1.get('btc_price_low', 'N/A'):,} |" if s1.get('btc_price_low') else "| BTC Low | N/A |")
    lines.append(f"| Equity Start | ${s1.get('equity_start', 'N/A'):,.2f} |" if s1.get('equity_start') else "| Equity Start | N/A |")
    lines.append(f"| Equity End | ${s1.get('equity_end', 'N/A'):,.2f} |" if s1.get('equity_end') else "| Equity End | N/A |")
    lines.append(f"| Equity Change | ${s1.get('equity_change_usd', 'N/A'):,.2f} ({s1.get('equity_change_pct', 0):.2f}%) |"
                 if s1.get('equity_change_usd') is not None else "| Equity Change | N/A |")
    lines.append(f"| Max Intraday DD | {s1.get('max_intraday_drawdown_pct', 'N/A')}% |"
                 if s1.get('max_intraday_drawdown_pct') is not None else "| Max Intraday DD | N/A |")
    lines.append(f"| Bars Analyzed | {s1.get('total_bars', 0)} |")
    lines.append("")

    # Regime distribution
    regime_dist = s1.get("regime_distribution", {})
    if regime_dist:
        lines.append("**Regime Distribution:**")
        for regime, pct in sorted(regime_dist.items(), key=lambda x: -x[1]):
            lines.append(f"- {regime}: {pct}%")
        lines.append("")

    # --- Section 2: Trade Analysis ---
    s2 = sections.get("2_trade_analysis", {})
    lines.append("## 2. Trade Analysis")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Unique Positions | {s2.get('unique_positions', 0)} |")
    lines.append(f"| Total Exit Rows | {s2.get('total_exit_rows', 0)} |")
    lines.append(f"| Total PnL | ${s2.get('total_pnl', 0):,.2f} |")
    lines.append(f"| Win Rate | {s2.get('win_rate', 0)}% |")
    lines.append(f"| Profit Factor | {s2.get('profit_factor', 0)} |")
    lines.append(f"| Avg Win | ${s2.get('avg_win', 0):,.2f} |")
    lines.append(f"| Avg Loss | ${s2.get('avg_loss', 0):,.2f} |")
    lines.append("")

    best = s2.get("best_trade")
    worst = s2.get("worst_trade")
    if best:
        lines.append(f"**Best Trade**: {best['archetype']} ({best['direction']}) = ${best['pnl']:,.2f}")
    if worst:
        lines.append(f"**Worst Trade**: {worst['archetype']} ({worst['direction']}) = ${worst['pnl']:,.2f}")
    lines.append("")

    # --- Section 3: Per-Archetype Breakdown ---
    s3 = sections.get("3_archetype_breakdown", {})
    archetypes = s3.get("archetypes", [])
    lines.append("## 3. Per-Archetype Breakdown")
    lines.append("")
    if archetypes:
        lines.append("| Archetype | Trades | Wins | Losses | WR% | PF | Total PnL | Avg Dur (h) |")
        lines.append("|-----------|--------|------|--------|-----|-----|-----------|-------------|")
        for a in archetypes:
            lines.append(
                f"| {a['archetype']} | {a['trades']} | {a['wins']} | {a['losses']} "
                f"| {a['win_rate_pct']} | {a['profit_factor']} "
                f"| ${a['total_pnl']:,.2f} | {a['avg_duration_hours']} |"
            )
        lines.append("")
    else:
        lines.append("No trade data available.")
        lines.append("")

    flagged = s3.get("flagged", [])
    if flagged:
        lines.append("**Flagged Archetypes:**")
        for f in flagged:
            lines.append(f"- WARNING: {f['archetype']} -- {f['reason']}")
        lines.append("")

    # --- Section 4: Exit Analysis ---
    s4 = sections.get("4_exit_analysis", {})
    lines.append("## 4. Exit Analysis")
    lines.append("")
    exit_counts = s4.get("exit_reason_counts", {})
    if exit_counts:
        lines.append("| Exit Reason | Count |")
        lines.append("|-------------|-------|")
        for reason, count in sorted(exit_counts.items(), key=lambda x: -x[1]):
            lines.append(f"| {reason} | {count} |")
        lines.append("")
    lines.append(f"- SL before any scale-out: {s4.get('pct_hit_sl_before_scaleout', 'N/A')}%")
    lines.append(f"- Avg R-multiple at exit: {s4.get('avg_r_multiple', 'N/A')}")
    lines.append("")

    # --- Section 5: Threshold Margin Analysis ---
    s5 = sections.get("5_threshold_margin", {})
    lines.append("## 5. Threshold Margin Analysis")
    lines.append("")
    pt = s5.get("passing_trades", {})
    bt = s5.get("bypass_trades", {})
    lines.append("| Group | Count | WR% | PF | Total PnL |")
    lines.append("|-------|-------|-----|-----|-----------|")
    lines.append(
        f"| Threshold-passing | {pt.get('count', 0)} "
        f"| {pt.get('win_rate_pct', 'N/A')} | {pt.get('profit_factor', 'N/A')} "
        f"| ${pt.get('total_pnl', 0):,.2f} |"
    )
    lines.append(
        f"| Bypass-only | {bt.get('count', 0)} "
        f"| {bt.get('win_rate_pct', 'N/A')} | {bt.get('profit_factor', 'N/A')} "
        f"| ${bt.get('total_pnl', 0):,.2f} |"
    )
    lines.append("")
    lines.append(f"**Verdict**: {s5.get('verdict', 'N/A')}")
    lines.append("")

    # --- Section 6: CMI Regime Analysis ---
    s6 = sections.get("6_cmi_regime", {})
    lines.append("## 6. CMI Regime Analysis")
    lines.append("")
    per_regime = s6.get("per_regime", {})
    if per_regime:
        lines.append("| Regime | Trades | Wins | WR% | PF | Total PnL |")
        lines.append("|--------|--------|------|-----|-----|-----------|")
        for regime, stats in sorted(per_regime.items()):
            lines.append(
                f"| {regime} | {stats['trades']} | {stats['wins']} "
                f"| {stats['win_rate_pct']} | {stats['profit_factor']} "
                f"| ${stats['total_pnl']:,.2f} |"
            )
        lines.append("")

    rt_pnl = s6.get("risk_temp_vs_pnl", {})
    if rt_pnl and "note" not in rt_pnl:
        lines.append("**Risk Temperature vs PnL (quartiles):**")
        lines.append("")
        lines.append("| Quartile | Trades | Avg PnL | WR% |")
        lines.append("|----------|--------|---------|-----|")
        for q, stats in rt_pnl.items():
            lines.append(f"| {q} | {stats['trades']} | ${stats['avg_pnl']:,.2f} | {stats['win_rate_pct']} |")
        lines.append("")
    elif "note" in rt_pnl:
        lines.append(f"**Risk Temperature vs PnL**: {rt_pnl['note']}")
        lines.append("")

    inst_pnl = s6.get("instability_vs_pnl", {})
    if inst_pnl and "note" not in inst_pnl:
        lines.append("**Instability vs PnL (quartiles):**")
        lines.append("")
        lines.append("| Quartile | Trades | Avg PnL | WR% |")
        lines.append("|----------|--------|---------|-----|")
        for q, stats in inst_pnl.items():
            lines.append(f"| {q} | {stats['trades']} | ${stats['avg_pnl']:,.2f} | {stats['win_rate_pct']} |")
        lines.append("")
    elif "note" in inst_pnl:
        lines.append(f"**Instability vs PnL**: {inst_pnl['note']}")
        lines.append("")

    # --- Section 7: Could Have Done Better ---
    s7 = sections.get("7_could_have_done_better", {})
    lines.append("## 7. Could Have Done Better")
    lines.append("")
    lines.append(f"- Total SL trades: {s7.get('total_sl_trades', 0)}")
    lines.append(f"- Premature SL (BTC recovered within 24h): {s7.get('premature_sl_count', 0)}")
    premature = s7.get("premature_sl_trades", [])
    if premature:
        lines.append("")
        lines.append("| Exit Time | Archetype | Entry | Exit | PnL |")
        lines.append("|-----------|-----------|-------|------|-----|")
        for t in premature[:10]:  # Limit to 10
            lines.append(
                f"| {t['timestamp_exit']} | {t['archetype']} "
                f"| ${t['entry_price']:,.0f} | ${t['exit_price']:,.0f} "
                f"| ${t['pnl']:,.2f} |"
            )
    lines.append("")

    # --- Section 8: Actionable Recommendations ---
    s8 = sections.get("8_recommendations", {})
    recs = s8.get("recommendations", [])
    lines.append("## 8. Actionable Recommendations")
    lines.append("")
    if recs:
        for i, r in enumerate(recs, 1):
            severity = r.get("severity", "medium").upper()
            lines.append(f"### {i}. [{severity}] {r.get('type', 'unknown')}")
            lines.append(f"- **Reason**: {r.get('reason', '')}")
            lines.append(f"- **Action**: {r.get('action', '')}")
            if "archetype" in r:
                lines.append(f"- **Archetype**: {r['archetype']}")
            if "regime" in r:
                lines.append(f"- **Regime**: {r['regime']}")
            lines.append("")
    else:
        lines.append("No actionable recommendations at this time.")
        lines.append("")

    lines.append("---")
    lines.append(f"*Report generated by `bin/daily_quant_review.py` at {report.get('generated_at', 'N/A')}*")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Output Writers
# ---------------------------------------------------------------------------

def write_outputs(report: dict, output_dir: Path, target_date: datetime) -> None:
    """Write daily_review.json, daily_review.md, and archived copy."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. daily_review.json
    json_path = output_dir / "daily_review.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Wrote %s", json_path)

    # 2. daily_review.md
    md_path = output_dir / "daily_review.md"
    md_content = render_markdown(report)
    with open(md_path, "w") as f:
        f.write(md_content)
    logger.info("Wrote %s", md_path)

    # 3. Archived copy
    archive_dir = output_dir / ARCHIVE_DIR
    archive_dir.mkdir(parents=True, exist_ok=True)
    date_str = target_date.strftime("%Y-%m-%d")
    archive_path = archive_dir / f"{date_str}.json"
    with open(archive_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Wrote archive %s", archive_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Bull Machine Daily Quant Review",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python3 bin/daily_quant_review.py\n"
            "  python3 bin/daily_quant_review.py --date 2026-03-01 --days 7\n"
            "  python3 bin/daily_quant_review.py --data-dir /home/ubuntu/Bull-machine-/results/coinbase_paper/\n"
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="results/coinbase_paper/",
        help="Data directory (default: results/coinbase_paper/)",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Date to analyze as YYYY-MM-DD (default: today)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=1,
        help="Number of days to look back (default: 1)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: same as data-dir)",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Resolve paths
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        # Try relative to script location (project root)
        project_root = Path(__file__).parent.parent
        candidate = project_root / args.data_dir
        if candidate.exists():
            data_dir = candidate
        else:
            data_dir = Path(args.data_dir).resolve()

    output_dir = Path(args.output_dir) if args.output_dir else data_dir

    if not data_dir.exists():
        logger.warning("Data directory does not exist: %s (will produce empty report)", data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)

    # Parse target date
    if args.date:
        try:
            target_date = datetime.strptime(args.date, "%Y-%m-%d")
        except ValueError:
            logger.error("Invalid date format: %s (expected YYYY-MM-DD)", args.date)
            sys.exit(1)
    else:
        target_date = datetime.utcnow()

    # Adjust start date for --days lookback
    start_date = target_date - timedelta(days=args.days - 1)

    logger.info("Bull Machine Daily Quant Review")
    logger.info("  Data dir:   %s", data_dir)
    logger.info("  Output dir: %s", output_dir)
    logger.info("  Date:       %s", target_date.strftime("%Y-%m-%d"))
    logger.info("  Days:       %d (start: %s)", args.days, start_date.strftime("%Y-%m-%d"))

    # Build report
    report = build_report(data_dir, start_date, args.days)

    # Write outputs
    write_outputs(report, output_dir, target_date)

    # Print summary to stdout
    s2 = report["sections"]["2_trade_analysis"]
    s1 = report["sections"]["1_daily_summary"]
    print("\n" + "=" * 60)
    print("DAILY QUANT REVIEW SUMMARY")
    print("=" * 60)
    print(f"  Period:          {s1.get('date_range', 'N/A')}")
    print(f"  Positions:       {s2.get('unique_positions', 0)}")
    print(f"  Total PnL:       ${s2.get('total_pnl', 0):,.2f}")
    print(f"  Win Rate:        {s2.get('win_rate', 0)}%")
    print(f"  Profit Factor:   {s2.get('profit_factor', 0)}")
    eq_change = s1.get('equity_change_usd')
    if eq_change is not None:
        print(f"  Equity Change:   ${eq_change:,.2f} ({s1.get('equity_change_pct', 0):.2f}%)")
    dd = s1.get('max_intraday_drawdown_pct')
    if dd is not None:
        print(f"  Max Intraday DD: {dd:.2f}%")
    recs = report["sections"]["8_recommendations"].get("recommendations", [])
    if recs:
        print(f"  Recommendations: {len(recs)}")
        for r in recs:
            print(f"    [{r['severity'].upper()}] {r['type']}: {r.get('archetype', r.get('regime', ''))}")
    print("=" * 60)
    print(f"  Full report: {output_dir / 'daily_review.md'}")
    print(f"  JSON report: {output_dir / 'daily_review.json'}")
    print("")


if __name__ == "__main__":
    main()
