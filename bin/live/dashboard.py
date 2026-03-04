#!/usr/bin/env python3
"""
Bull Machine — Live Monitoring Dashboard v7.6

Flask API backend serving React SPA from dashboard/dist/.
5 tabs: Dashboard (WhaleIntelligencePanel + CMI + Wyckoff), Strategy, Signals, Backtest, Trades.

Usage:
    python3 bin/live/dashboard.py                        # Default port 8081
    python3 bin/live/dashboard.py --port 9090            # Custom port
"""

import argparse
import csv
import json
import logging
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "coinbase_paper"
BACKTEST_DIR = PROJECT_ROOT / "results" / "web_backtests"
DASHBOARD_DIST = PROJECT_ROOT / "dashboard" / "dist"

app = Flask(__name__, static_folder=str(DASHBOARD_DIST), static_url_path="")

# In-memory backtest job tracker
_backtest_jobs = {}  # job_id -> {status, result, error, started_at}


# ---------------------------------------------------------------------------
# Data readers
# ---------------------------------------------------------------------------

def _read_json(filename: str):
    path = RESULTS_DIR / filename
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def _read_signals_csv(limit: int = 50) -> list:
    path = RESULTS_DIR / "signals.csv"
    if not path.exists():
        return []
    rows = []
    try:
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    except Exception:
        return []
    return rows[-limit:]


def _read_equity_history(limit: int = 500) -> list:
    path = RESULTS_DIR / "equity_history.csv"
    if not path.exists():
        return []
    rows = []
    try:
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Drop None key from rows with more columns than header
                clean = {k: v for k, v in row.items() if k is not None}
                rows.append(clean)
    except Exception:
        return []
    return rows[-limit:]


def _read_candle_history(limit: int = 200) -> list:
    path = RESULTS_DIR / "candle_history.csv"
    if not path.exists():
        return []
    rows = []
    try:
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    except Exception:
        return []
    return rows[-limit:]


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

@app.route("/api/status")
def api_status():
    heartbeat = _read_json("heartbeat.json")
    performance = _read_json("performance_summary.json")
    funding = _read_json("funding_costs.json")
    return jsonify({
        "heartbeat": heartbeat,
        "performance": performance,
        "funding": funding,
        "server_time": datetime.now(timezone.utc).isoformat(),
    })


@app.route("/api/equity-history")
def api_equity_history():
    return jsonify(_read_equity_history(500))


@app.route("/api/signals")
def api_signals():
    return jsonify(_read_signals_csv(50))


@app.route("/api/signal-log")
def api_signal_log():
    data = _read_json("signal_log.json")
    if isinstance(data, list):
        return jsonify(data)
    return jsonify([])


@app.route("/api/trades")
def api_trades():
    data = _read_json("trades.json")
    if isinstance(data, list):
        return jsonify(data)
    return jsonify([])


@app.route("/api/candle-history")
def api_candle_history():
    return jsonify(_read_candle_history(200))


@app.route("/api/price")
def api_price():
    """Return latest BTC price from heartbeat (updated every bar ~1h)."""
    hb = _read_json("heartbeat.json")
    return jsonify({
        "price": hb.get("btc_price", 0),
        "updated_at": hb.get("updated_at", ""),
    })


@app.route("/api/daily-review")
def api_daily_review():
    """Return latest daily quant review report."""
    return jsonify(_read_json("daily_review.json"))


# ---------------------------------------------------------------------------
# Backtest API
# ---------------------------------------------------------------------------

def _find_feature_store() -> str:
    """Find the best feature store parquet file."""
    candidates = [
        PROJECT_ROOT / "data" / "features_mtf" / "BTC_1H_FEATURES_V12_ENHANCED.parquet",
        PROJECT_ROOT / "data" / "features_mtf" / "BTC_1H_CANONICAL_20260202.parquet",
        PROJECT_ROOT / "data" / "features_mtf" / "BTC_1H_LATEST.parquet",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return ""


def _run_backtest_job(job_id: str, params: dict):
    """Run backtest in background thread."""
    try:
        _backtest_jobs[job_id]["status"] = "running"

        output_dir = BACKTEST_DIR / job_id
        output_dir.mkdir(parents=True, exist_ok=True)

        feature_store = _find_feature_store()
        if not feature_store:
            _backtest_jobs[job_id]["status"] = "error"
            _backtest_jobs[job_id]["error"] = "No feature store found on server"
            return

        config_path = params.get("config",
            str(PROJECT_ROOT / "configs" / "bull_machine_isolated_v11_fixed.json"))

        # Build temp config with leverage override if needed
        leverage = params.get("leverage", 1.5)
        with open(config_path) as f:
            config = json.load(f)
        config["leverage"] = leverage
        tmp_config = output_dir / "config_override.json"
        with open(tmp_config, "w") as f:
            json.dump(config, f, indent=2)

        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "bin" / "backtest_v11_standalone.py"),
            "--config", str(tmp_config),
            "--feature-store", feature_store,
            "--initial-cash", str(params.get("capital", 100000)),
            "--commission-rate", str(params.get("commission", 0.0002)),
            "--slippage-bps", str(params.get("slippage", 3.0)),
            "--output-dir", str(output_dir),
        ]

        if params.get("start_date"):
            cmd.extend(["--start-date", params["start_date"]])
        if params.get("end_date"):
            cmd.extend(["--end-date", params["end_date"]])

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            cwd=str(PROJECT_ROOT),
        )

        if result.returncode != 0:
            _backtest_jobs[job_id]["status"] = "error"
            _backtest_jobs[job_id]["error"] = result.stderr[-2000:] if result.stderr else "Unknown error"
            _backtest_jobs[job_id]["stdout"] = result.stdout[-2000:] if result.stdout else ""
            return

        # Read results
        stats_path = output_dir / "performance_stats.json"
        breakdown_path = output_dir / "archetype_breakdown.csv"
        equity_path = output_dir / "equity_curve.csv"

        stats = {}
        if stats_path.exists():
            stats = json.loads(stats_path.read_text())

        breakdown = []
        if breakdown_path.exists():
            with open(breakdown_path) as f:
                reader = csv.DictReader(f)
                breakdown = list(reader)

        equity = []
        if equity_path.exists():
            with open(equity_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    equity.append(row)
            # Downsample if too many points
            if len(equity) > 500:
                step = len(equity) // 500
                equity = equity[::step]

        # Feature store metadata
        feature_meta = {"path": feature_store, "columns": 0, "rows": 0, "column_list": []}
        try:
            import pandas as pd
            fs_df = pd.read_parquet(feature_store, columns=None)
            feature_meta["columns"] = len(fs_df.columns)
            feature_meta["rows"] = len(fs_df)
            feature_meta["column_list"] = sorted(fs_df.columns.tolist())
            feature_meta["date_range"] = f"{fs_df.index.min()} to {fs_df.index.max()}" if hasattr(fs_df.index, 'min') else ""
            del fs_df
        except Exception:
            pass

        _backtest_jobs[job_id]["status"] = "complete"
        _backtest_jobs[job_id]["result"] = {
            "stats": stats,
            "breakdown": breakdown,
            "equity": equity,
            "stdout": result.stdout[-3000:] if result.stdout else "",
            "params": params,
            "feature_store": feature_meta,
        }

    except subprocess.TimeoutExpired:
        _backtest_jobs[job_id]["status"] = "error"
        _backtest_jobs[job_id]["error"] = "Backtest timed out (10 min limit)"
    except Exception as e:
        _backtest_jobs[job_id]["status"] = "error"
        _backtest_jobs[job_id]["error"] = str(e)


@app.route("/api/run-backtest", methods=["POST"])
def api_run_backtest():
    params = request.get_json(force=True, silent=True) or {}

    # Validate
    capital = float(params.get("capital", 10000))
    leverage = float(params.get("leverage", 1.5))
    if capital < 100 or capital > 10_000_000:
        return jsonify({"error": "Capital must be between $100 and $10M"}), 400
    if leverage < 1.0 or leverage > 10.0:
        return jsonify({"error": "Leverage must be between 1x and 10x"}), 400

    job_id = str(uuid.uuid4())[:8]
    _backtest_jobs[job_id] = {
        "status": "queued",
        "result": None,
        "error": None,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }

    thread = threading.Thread(
        target=_run_backtest_job,
        args=(job_id, params),
        daemon=True,
    )
    thread.start()

    return jsonify({"job_id": job_id, "status": "queued"})


@app.route("/api/backtest-status/<job_id>")
def api_backtest_status(job_id):
    job = _backtest_jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


# ---------------------------------------------------------------------------
# React SPA — serve built dashboard from dashboard/dist/
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory(str(DASHBOARD_DIST), "index.html")


# (Old inline DASHBOARD_HTML removed — React SPA served from dashboard/dist/)
# Fallback: serve index.html for any non-API, non-static path (client-side routing)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Bull Machine Live Dashboard")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8081, help="Port (default: 8081)")
    parser.add_argument("--debug", action="store_true", help="Flask debug mode")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Ensure backtest output dir exists
    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Starting Bull Machine Dashboard v7.6 on %s:%d", args.host, args.port)
    logger.info("Reading data from: %s", RESULTS_DIR)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
