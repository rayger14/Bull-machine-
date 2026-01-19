# Paper Trading Dashboard - Implementation Guide

**Companion to**: PAPER_TRADING_METRICS_DASHBOARD_SPEC.md
**Version**: 1.0
**Purpose**: Technical implementation details with working code examples

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Backend Implementation](#backend-implementation)
3. [Dashboard Implementation](#dashboard-implementation)
4. [Metrics Calculators](#metrics-calculators)
5. [Alert System](#alert-system)
6. [Database Queries](#database-queries)
7. [Testing Strategy](#testing-strategy)
8. [Deployment Guide](#deployment-guide)

---

## 1. Quick Start

### Prerequisites

```bash
# Python 3.10+
python --version

# Install dependencies
pip install -r requirements_dashboard.txt
```

**requirements_dashboard.txt**:
```
streamlit==1.29.0
fastapi==0.108.0
uvicorn==0.25.0
redis==5.0.1
psycopg2-binary==2.9.9
sqlalchemy==2.0.23
pandas==2.1.4
numpy==1.26.2
plotly==5.18.0
scipy==1.11.4
prometheus-client==0.19.0
python-multipart==0.0.6
websockets==12.0
pydantic==2.5.3
```

### Local Development Setup

```bash
# 1. Start infrastructure
docker-compose up -d timescale redis

# 2. Initialize database
python scripts/init_dashboard_db.py

# 3. Start backend API
cd backend
uvicorn main:app --reload --port 8000

# 4. Start Streamlit dashboard (separate terminal)
cd dashboard
streamlit run main.py --server.port 8501
```

### Directory Structure

```
Bull-machine-/
├── dashboard/
│   ├── main.py                    # Streamlit entry point
│   ├── pages/
│   │   ├── 1_Archetype_Health.py
│   │   ├── 2_Drift_Detection.py
│   │   ├── 3_Comparison.py
│   │   └── 4_Kill_Switch.py
│   ├── components/
│   │   ├── metrics_cards.py
│   │   ├── charts.py
│   │   └── alerts.py
│   └── utils/
│       ├── api_client.py
│       └── formatters.py
├── backend/
│   ├── main.py                    # FastAPI app
│   ├── api/
│   │   ├── metrics.py
│   │   ├── alerts.py
│   │   └── websocket.py
│   ├── services/
│   │   ├── metrics_calculator.py
│   │   ├── drift_detector.py
│   │   ├── alert_manager.py
│   │   └── kill_switch.py
│   ├── models/
│   │   └── schemas.py
│   └── db/
│       ├── connection.py
│       └── queries.py
├── scripts/
│   ├── init_dashboard_db.py
│   ├── backfill_metrics.py
│   └── test_kill_switch.py
└── docker-compose.yml
```

---

## 2. Backend Implementation

### 2.1 FastAPI Main Application

**backend/main.py**:
```python
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import asyncio
from typing import List

from api import metrics, alerts, websocket
from services.metrics_calculator import MetricsCalculator
from services.alert_manager import AlertManager
from db.connection import init_db, close_db

# Background task for metrics calculation
metrics_calculator = None
alert_manager = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    global metrics_calculator, alert_manager
    await init_db()
    metrics_calculator = MetricsCalculator()
    alert_manager = AlertManager()

    # Start background tasks
    asyncio.create_task(metrics_calculator.run_continuous_updates())
    asyncio.create_task(alert_manager.run_continuous_checks())

    yield

    # Shutdown
    await close_db()

app = FastAPI(
    title="Bull Machine Paper Trading API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS for Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(metrics.router, prefix="/api/metrics", tags=["metrics"])
app.include_router(alerts.router, prefix="/api/alerts", tags=["alerts"])
app.include_router(websocket.router, prefix="/ws", tags=["websocket"])

@app.get("/")
async def root():
    return {"status": "healthy", "service": "Bull Machine Paper Trading API"}

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    from db.connection import check_db_health
    from services.metrics_calculator import check_redis_health

    checks = {
        "database": await check_db_health(),
        "redis": await check_redis_health(),
        "metrics_calculator": metrics_calculator.is_running if metrics_calculator else False,
        "alert_manager": alert_manager.is_running if alert_manager else False
    }

    all_healthy = all(checks.values())
    return {
        "status": "healthy" if all_healthy else "degraded",
        "checks": checks
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 2.2 Metrics API Endpoints

**backend/api/metrics.py**:
```python
from fastapi import APIRouter, HTTPException, Depends
from typing import Optional, List
from datetime import datetime, timedelta

from models.schemas import (
    RealTimeMetrics,
    ArchetypeHealthResponse,
    DriftMetricsResponse,
    ComparisonMetrics
)
from services.metrics_calculator import MetricsCalculator
from db.queries import MetricsQueries

router = APIRouter()

@router.get("/real-time", response_model=RealTimeMetrics)
async def get_real_time_metrics():
    """
    Get current system metrics (cached for 30s).

    Returns:
        - PnL (daily, unrealized)
        - Drawdown (current, max)
        - Sharpe ratios (1d, 7d, 30d)
        - Fill rate
        - Signal counts
    """
    calculator = MetricsCalculator()
    metrics = await calculator.get_cached_real_time_metrics()

    if not metrics:
        raise HTTPException(status_code=503, detail="Metrics service unavailable")

    return metrics

@router.get("/archetype/{archetype_id}", response_model=ArchetypeHealthResponse)
async def get_archetype_metrics(
    archetype_id: str,
    lookback_hours: int = 168  # 7 days default
):
    """
    Get detailed metrics for a single archetype.

    Args:
        archetype_id: Archetype identifier (e.g., 'A', 'C', 'S1')
        lookback_hours: Time window for analysis

    Returns:
        - Health score
        - Performance metrics
        - Domain boost activation
        - Recent signals
    """
    queries = MetricsQueries()
    metrics = await queries.get_archetype_health(archetype_id, lookback_hours)

    if not metrics:
        raise HTTPException(status_code=404, detail=f"Archetype {archetype_id} not found")

    return metrics

@router.get("/drift", response_model=DriftMetricsResponse)
async def get_drift_metrics(
    feature_names: Optional[List[str]] = None,
    severity: Optional[str] = None  # 'minor', 'critical'
):
    """
    Get feature drift analysis.

    Args:
        feature_names: Specific features to check (default: all critical features)
        severity: Filter by drift severity

    Returns:
        - PSI scores
        - KS statistics
        - Distribution comparisons
        - CUSUM trends
    """
    from services.drift_detector import DriftDetector

    detector = DriftDetector()
    drift_report = await detector.get_drift_report(feature_names, severity)

    return drift_report

@router.get("/comparison", response_model=ComparisonMetrics)
async def get_backtest_comparison():
    """
    Compare paper trading vs backtest performance.

    Returns:
        - Side-by-side metrics
        - Statistical significance tests
        - Go-live readiness assessment
    """
    queries = MetricsQueries()
    comparison = await queries.get_backtest_comparison()

    return comparison

@router.get("/ensemble")
async def get_ensemble_metrics():
    """
    Get ensemble-level diagnostics.

    Returns:
        - Signal overlap analysis
        - Archetype correlation matrix
        - Regime coverage
        - Capital allocation
    """
    queries = MetricsQueries()
    ensemble = await queries.get_ensemble_metrics()

    return ensemble
```

### 2.3 WebSocket for Real-Time Updates

**backend/api/websocket.py**:
```python
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import List
import asyncio
import json

router = APIRouter()

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        """Send message to all connected clients"""
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                print(f"Error sending to client: {e}")

manager = ConnectionManager()

@router.websocket("/metrics")
async def websocket_metrics(websocket: WebSocket):
    """
    WebSocket endpoint for real-time metric streaming.

    Sends updated metrics every 5 seconds to all connected clients.
    """
    await manager.connect(websocket)

    try:
        from services.metrics_calculator import MetricsCalculator
        calculator = MetricsCalculator()

        while True:
            # Get latest metrics
            metrics = await calculator.get_cached_real_time_metrics()

            # Send to client
            await websocket.send_json({
                "type": "metrics_update",
                "timestamp": datetime.utcnow().isoformat(),
                "data": metrics.dict()
            })

            # Wait 5 seconds before next update
            await asyncio.sleep(5)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(websocket)

@router.websocket("/alerts")
async def websocket_alerts(websocket: WebSocket):
    """
    WebSocket endpoint for real-time alert streaming.

    Pushes alerts immediately when triggered.
    """
    await manager.connect(websocket)

    try:
        from services.alert_manager import AlertManager
        alert_manager = AlertManager()

        # Subscribe to alert stream
        async for alert in alert_manager.alert_stream():
            await websocket.send_json({
                "type": "alert",
                "timestamp": alert.timestamp.isoformat(),
                "data": alert.dict()
            })

    except WebSocketDisconnect:
        manager.disconnect(websocket)
```

---

## 3. Dashboard Implementation

### 3.1 Streamlit Main Dashboard

**dashboard/main.py**:
```python
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

from utils.api_client import APIClient
from components.metrics_cards import render_metric_cards
from components.charts import render_performance_chart, render_regime_distribution
from components.alerts import render_alerts

# Page config
st.set_page_config(
    page_title="Bull Machine Paper Trading",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize API client
api = APIClient(base_url="http://localhost:8000")

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .status-healthy { color: #4caf50; }
    .status-warning { color: #ff9800; }
    .status-critical { color: #f44336; }
</style>
""", unsafe_allow_html=True)

# Header
st.title("🎯 Bull Machine Paper Trading Dashboard")
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    status = api.get_health_status()
    if status["status"] == "healthy":
        st.success("🟢 ACTIVE | Uptime: 12d 4h 23m | Last Update: 3s ago")
    else:
        st.error("🔴 DEGRADED | Check system health")

with col2:
    st.metric("Paper Start", "Dec 1, 2025")

with col3:
    st.metric("Days to Live", "48 days", delta="-12 days")

# Main content
st.markdown("---")

# Critical metrics cards
st.subheader("Critical Metrics")
metrics = api.get_real_time_metrics()
render_metric_cards(metrics)

st.markdown("---")

# Performance chart
st.subheader("Performance Chart (7-Day Rolling)")
performance_data = api.get_performance_history(days=7)
render_performance_chart(performance_data)

st.markdown("---")

# Two columns: Archetype Health + Regime Status
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Archetype Health")
    archetype_health = api.get_all_archetype_health()

    # Display top archetypes
    for arch in archetype_health[:5]:
        status_icon = "✅" if arch["health_score"] >= 75 else "⚠️"
        st.write(f"{status_icon} {arch['archetype_id']}: {arch['health_score']}/100")

    if st.button("View All 16 Archetypes →"):
        st.switch_page("pages/1_Archetype_Health.py")

with col_right:
    st.subheader("Regime Status")
    regime = api.get_regime_status()

    st.write(f"**Current**: {regime['current_regime']}")
    st.write(f"**Confidence**: {regime['confidence']:.0%}")
    st.write(f"**Duration**: {regime['duration']}")

    # Regime distribution (last 24h)
    render_regime_distribution(regime['distribution_24h'])

st.markdown("---")

# Domain engine status
st.subheader("Domain Engine Status")
domain_status = api.get_domain_engine_status()

for engine, rate in domain_status.items():
    progress_color = "green" if rate >= 0.6 else "orange" if rate >= 0.5 else "red"
    st.progress(rate, text=f"{engine}: {rate:.0%}")

st.markdown("---")

# Alerts section
st.subheader("🚨 Alerts")
alerts = api.get_recent_alerts(limit=10)
render_alerts(alerts)

# Quick actions
st.markdown("---")
st.subheader("Quick Actions")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    if st.button("📊 Detailed Report"):
        st.download_button(
            label="Download Report",
            data=api.export_detailed_report(),
            file_name=f"report_{datetime.now().strftime('%Y%m%d')}.pdf"
        )

with col2:
    if st.button("🔍 Drill into Archetype"):
        st.switch_page("pages/1_Archetype_Health.py")

with col3:
    if st.button("📈 Compare vs Backtest"):
        st.switch_page("pages/3_Comparison.py")

with col4:
    if st.button("🚨 Kill-Switch Panel"):
        st.switch_page("pages/4_Kill_Switch.py")

with col5:
    if st.button("📥 Export Logs"):
        st.download_button(
            label="Download Logs",
            data=api.export_logs(),
            file_name=f"logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

# Auto-refresh
if st.checkbox("Auto-refresh (5s)", value=True):
    import time
    time.sleep(5)
    st.rerun()
```

### 3.2 Metric Cards Component

**dashboard/components/metrics_cards.py**:
```python
import streamlit as st

def render_metric_cards(metrics):
    """Render critical metrics as cards"""
    col1, col2, col3 = st.columns(3)

    with col1:
        # PnL Today
        pnl_status = "✅" if metrics["daily_pnl_pct"] >= 0 else "⚠️"
        st.metric(
            "PnL (Today)",
            f"${metrics['daily_pnl']:,.0f}",
            f"{metrics['daily_pnl_pct']:+.2f}%",
            delta_color="normal"
        )
        st.caption(f"{pnl_status} {'On Track' if metrics['daily_pnl_pct'] >= -3 else 'Watch'}")

    with col2:
        # Drawdown
        dd_status = "✅" if metrics["current_drawdown"] > -15 else "⚠️"
        st.metric(
            "Drawdown",
            f"{metrics['current_drawdown']:.1f}%",
            f"Max: {metrics['max_drawdown']:.1f}%"
        )
        st.caption(f"{dd_status} {'Safe' if metrics['current_drawdown'] > -15 else 'Monitor'}")

    with col3:
        # Sharpe Ratio
        sharpe_deviation = (metrics["sharpe_7d"] / 1.85 - 1) * 100
        sharpe_status = "✅" if sharpe_deviation >= -20 else "⚠️"
        st.metric(
            "Sharpe Ratio (7d)",
            f"{metrics['sharpe_7d']:.2f}",
            f"{sharpe_deviation:+.1f}% vs BT"
        )
        st.caption(f"{sharpe_status} {'Good' if sharpe_deviation >= -20 else 'Below Target'}")

    # Second row
    col4, col5, col6 = st.columns(3)

    with col4:
        fill_status = "✅" if metrics["fill_rate_24h"] >= 0.90 else "⚠️"
        st.metric(
            "Fill Rate (24h)",
            f"{metrics['fill_rate_24h']:.0%}",
            f"{metrics['avg_slippage_bps']:.1f} bps"
        )
        st.caption(f"{fill_status} {'Good' if metrics['fill_rate_24h'] >= 0.90 else 'Low'}")

    with col5:
        st.metric(
            "Signals (Today)",
            f"{metrics['signals_today']:,}",
            f"{metrics['signals_today'] - metrics['signals_expected']:+d} vs expected"
        )
        st.caption("✅ Active")

    with col6:
        overlap_status = "✅" if 0.35 <= metrics["signal_overlap_rate"] <= 0.50 else "⚠️"
        st.metric(
            "Signal Overlap",
            f"{metrics['signal_overlap_rate']:.0%}",
            "Target: 35-45%"
        )
        st.caption(f"{overlap_status} {'OK' if overlap_status == '✅' else 'High'}")
```

### 3.3 Performance Chart Component

**dashboard/components/charts.py**:
```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

def render_performance_chart(data: pd.DataFrame):
    """Render cumulative PnL, Sharpe, and Win Rate"""

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("Cumulative PnL (%)", "Sharpe Ratio & Win Rate"),
        row_heights=[0.6, 0.4]
    )

    # Cumulative PnL
    fig.add_trace(
        go.Scatter(
            x=data["timestamp"],
            y=data["cumulative_pnl_pct"],
            mode="lines",
            name="PnL",
            line=dict(color="#4caf50", width=2),
            fill="tozeroy",
            fillcolor="rgba(76, 175, 80, 0.1)"
        ),
        row=1, col=1
    )

    # Sharpe ratio
    fig.add_trace(
        go.Scatter(
            x=data["timestamp"],
            y=data["sharpe_7d"],
            mode="lines",
            name="Sharpe 7d",
            line=dict(color="#2196f3", width=2)
        ),
        row=2, col=1
    )

    # Win rate
    fig.add_trace(
        go.Scatter(
            x=data["timestamp"],
            y=data["win_rate_50"],
            mode="lines",
            name="Win Rate (50 trades)",
            line=dict(color="#ff9800", width=2),
            yaxis="y3"
        ),
        row=2, col=1
    )

    # Update layout
    fig.update_layout(
        height=600,
        showlegend=True,
        hovermode="x unified",
        template="plotly_white"
    )

    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="PnL (%)", row=1, col=1)
    fig.update_yaxes(title_text="Sharpe", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)

def render_regime_distribution(distribution: dict):
    """Render regime distribution as horizontal bar chart"""

    regimes = list(distribution.keys())
    percentages = [distribution[r] * 100 for r in regimes]

    colors = {
        "risk_on": "#4caf50",
        "risk_off": "#ff9800",
        "crisis": "#f44336",
        "neutral": "#9e9e9e"
    }

    fig = go.Figure(go.Bar(
        y=regimes,
        x=percentages,
        orientation='h',
        marker=dict(color=[colors.get(r, "#000") for r in regimes]),
        text=[f"{p:.0f}%" for p in percentages],
        textposition="inside"
    ))

    fig.update_layout(
        height=200,
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(range=[0, 100], showticklabels=False),
        yaxis=dict(showticklabels=True)
    )

    st.plotly_chart(fig, use_container_width=True)
```

---

## 4. Metrics Calculators

### 4.1 Core Metrics Calculator

**backend/services/metrics_calculator.py**:
```python
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import asyncio
import redis.asyncio as redis
import json

from db.queries import MetricsQueries
from models.schemas import RealTimeMetrics

class MetricsCalculator:
    def __init__(self):
        self.redis = redis.from_url("redis://localhost:6379", decode_responses=True)
        self.queries = MetricsQueries()
        self.is_running = False

    async def run_continuous_updates(self):
        """Background task to update metrics every 30s"""
        self.is_running = True

        while self.is_running:
            try:
                metrics = await self.calculate_real_time_metrics()
                await self.cache_metrics(metrics)
                await asyncio.sleep(30)  # Update every 30s
            except Exception as e:
                print(f"Error in metrics calculator: {e}")
                await asyncio.sleep(5)  # Retry after 5s

    async def calculate_real_time_metrics(self) -> RealTimeMetrics:
        """Calculate all real-time metrics"""

        # Fetch data from database
        trades_24h = await self.queries.get_trades_last_n_hours(24)
        signals_24h = await self.queries.get_signals_last_n_hours(24)
        current_positions = await self.queries.get_active_positions()

        # Calculate PnL
        daily_pnl = sum([t.pnl for t in trades_24h])
        unrealized_pnl = sum([p.unrealized_pnl for p in current_positions])
        session_start_equity = await self.queries.get_session_start_equity()
        current_equity = session_start_equity + daily_pnl + unrealized_pnl
        daily_pnl_pct = (daily_pnl / session_start_equity) * 100 if session_start_equity > 0 else 0

        # Calculate drawdown
        peak_equity = await self.queries.get_peak_equity()
        current_drawdown = ((peak_equity - current_equity) / peak_equity) * 100 if peak_equity > 0 else 0
        max_drawdown = await self.queries.get_max_drawdown()

        # Calculate Sharpe ratios
        returns_24h = await self.queries.get_returns_last_n_hours(24)
        returns_7d = await self.queries.get_returns_last_n_days(7)
        returns_30d = await self.queries.get_returns_last_n_days(30)

        sharpe_1d = self._calculate_sharpe(returns_24h, periods=24)
        sharpe_7d = self._calculate_sharpe(returns_7d, periods=168)
        sharpe_30d = self._calculate_sharpe(returns_30d, periods=720)

        # Fill rate
        executed_count = len([s for s in signals_24h if s.executed])
        fill_rate_24h = executed_count / len(signals_24h) if signals_24h else 0

        # Slippage
        executed_signals = [s for s in signals_24h if s.executed]
        avg_slippage_bps = np.mean([s.slippage_bps for s in executed_signals]) if executed_signals else 0

        # Signal counts
        signals_today = len(signals_24h)
        signals_expected = await self.queries.get_expected_signals_per_day()

        # Overlap rate
        overlap_events = await self.queries.count_overlap_events_24h()
        signal_overlap_rate = overlap_events / len(signals_24h) if signals_24h else 0

        return RealTimeMetrics(
            timestamp=datetime.utcnow(),
            daily_pnl=daily_pnl,
            daily_pnl_pct=daily_pnl_pct,
            unrealized_pnl=unrealized_pnl,
            current_drawdown=current_drawdown,
            max_drawdown=max_drawdown,
            sharpe_1d=sharpe_1d,
            sharpe_7d=sharpe_7d,
            sharpe_30d=sharpe_30d,
            fill_rate_24h=fill_rate_24h,
            avg_slippage_bps=avg_slippage_bps,
            signals_today=signals_today,
            signals_expected=signals_expected,
            signal_overlap_rate=signal_overlap_rate
        )

    def _calculate_sharpe(self, returns: np.ndarray, periods: int, risk_free_rate: float = 0.0) -> float:
        """Calculate annualized Sharpe ratio"""
        if len(returns) == 0:
            return 0.0

        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)

        if std_return == 0:
            return 0.0

        sharpe = np.sqrt(periods) * (mean_return - risk_free_rate) / std_return
        return sharpe

    async def cache_metrics(self, metrics: RealTimeMetrics):
        """Cache metrics in Redis with 30s TTL"""
        cache_key = "metrics:real_time"
        await self.redis.setex(cache_key, 30, metrics.json())

    async def get_cached_real_time_metrics(self) -> RealTimeMetrics:
        """Retrieve cached metrics (or calculate if not cached)"""
        cache_key = "metrics:real_time"
        cached = await self.redis.get(cache_key)

        if cached:
            return RealTimeMetrics.parse_raw(cached)

        # Cache miss, calculate fresh
        metrics = await self.calculate_real_time_metrics()
        await self.cache_metrics(metrics)
        return metrics
```

### 4.2 Drift Detector

**backend/services/drift_detector.py**:
```python
import numpy as np
from scipy import stats
from typing import Dict, List, Optional
import pandas as pd

from db.queries import MetricsQueries

class DriftDetector:
    def __init__(self):
        self.queries = MetricsQueries()

    async def get_drift_report(
        self,
        feature_names: Optional[List[str]] = None,
        severity: Optional[str] = None
    ) -> Dict:
        """Generate comprehensive drift report"""

        if not feature_names:
            feature_names = await self.queries.get_critical_feature_names()

        drift_results = []

        for feature in feature_names:
            # Get backtest and live distributions
            backtest_dist = await self.queries.get_feature_backtest_distribution(feature)
            live_dist = await self.queries.get_feature_live_distribution(feature, days=7)

            # Calculate PSI
            psi = self._calculate_psi(backtest_dist, live_dist)

            # Calculate KS statistic
            ks_stat, ks_pvalue = stats.ks_2samp(backtest_dist, live_dist)

            # Classify drift severity
            drift_severity = self._classify_drift(psi, ks_pvalue)

            # Filter by severity if requested
            if severity and drift_severity.lower() != severity.lower():
                continue

            drift_results.append({
                "feature_name": feature,
                "psi": psi,
                "ks_statistic": ks_stat,
                "ks_pvalue": ks_pvalue,
                "drift_severity": drift_severity,
                "backtest_mean": np.mean(backtest_dist),
                "backtest_std": np.std(backtest_dist),
                "live_mean": np.mean(live_dist),
                "live_std": np.std(live_dist),
                "status": "STABLE" if psi < 0.1 else "MONITOR" if psi < 0.25 else "DRIFTING"
            })

        # Sort by PSI (worst first)
        drift_results.sort(key=lambda x: x["psi"], reverse=True)

        return {
            "total_features_monitored": len(feature_names),
            "features_with_drift": len([r for r in drift_results if r["psi"] > 0.25]),
            "features_monitoring": len([r for r in drift_results if 0.1 < r["psi"] <= 0.25]),
            "features_stable": len([r for r in drift_results if r["psi"] <= 0.1]),
            "drift_details": drift_results
        }

    def _calculate_psi(self, expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
        """
        Calculate Population Stability Index (PSI).

        PSI < 0.1: No drift
        0.1 < PSI < 0.25: Minor drift
        PSI > 0.25: Significant drift
        """
        # Create bins based on expected distribution
        bin_edges = np.percentile(expected, np.linspace(0, 100, bins + 1))

        # Ensure unique bin edges
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) < 2:
            return 0.0  # Cannot calculate PSI with single bin

        # Calculate expected and actual percentages
        expected_percents, _ = np.histogram(expected, bins=bin_edges, density=True)
        actual_percents, _ = np.histogram(actual, bins=bin_edges, density=True)

        # Normalize to sum to 1
        expected_percents = expected_percents / expected_percents.sum()
        actual_percents = actual_percents / actual_percents.sum()

        # Calculate PSI
        psi = 0.0
        for e_pct, a_pct in zip(expected_percents, actual_percents):
            if e_pct > 0 and a_pct > 0:
                psi += (a_pct - e_pct) * np.log(a_pct / e_pct)

        return psi

    def _classify_drift(self, psi: float, ks_pvalue: float) -> str:
        """Classify drift severity"""
        if psi > 0.35 or ks_pvalue < 0.01:
            return "CRITICAL"
        elif psi > 0.25 or ks_pvalue < 0.05:
            return "SIGNIFICANT"
        elif psi > 0.1:
            return "MINOR"
        else:
            return "NONE"

    async def calculate_cusum_win_rate(self, expected_win_rate: float = 0.58, drift_threshold: float = 5.0) -> Dict:
        """
        CUSUM test for win rate degradation.

        Returns alerts when cumulative deviation exceeds threshold.
        """
        trades = await self.queries.get_all_paper_trades()

        cusum_pos = 0
        cusum_neg = 0
        cusum_history = []
        alerts = []

        for i, trade in enumerate(trades):
            win = 1 if trade.pnl > 0 else 0
            deviation = win - expected_win_rate

            cusum_pos = max(0, cusum_pos + deviation - 0.01)
            cusum_neg = min(0, cusum_neg + deviation + 0.01)

            cusum_history.append({
                "trade_index": i,
                "cusum_positive": cusum_pos,
                "cusum_negative": cusum_neg
            })

            if cusum_pos > drift_threshold:
                alerts.append({"trade_idx": i, "type": "POSITIVE_DRIFT", "cusum": cusum_pos})
            if cusum_neg < -drift_threshold:
                alerts.append({"trade_idx": i, "type": "NEGATIVE_DRIFT", "cusum": cusum_neg})

        trend = "IMPROVING" if cusum_pos > 2 else "DEGRADING" if cusum_neg < -2 else "STABLE"

        return {
            "cusum_positive": cusum_pos,
            "cusum_negative": cusum_neg,
            "trend": trend,
            "alerts": alerts,
            "history": cusum_history
        }
```

---

## 5. Alert System

### 5.1 Alert Manager

**backend/services/alert_manager.py**:
```python
import asyncio
from datetime import datetime
from typing import Dict, List
import redis.asyncio as redis

from db.queries import AlertQueries, MetricsQueries
from models.schemas import Alert
from services.notifications import NotificationService

HARD_STOP_CONDITIONS = {
    "daily_loss_limit": {
        "severity": "CRITICAL",
        "condition": lambda m: m.daily_pnl_pct < -5.0,
        "threshold": -5.0,
        "action": "HALT_ALL_TRADING",
        "notification": ["SMS", "EMAIL", "SLACK_URGENT"],
        "message_template": "🚨 CRITICAL: Daily loss limit exceeded at {daily_pnl_pct:.2f}%"
    },
    "drawdown_limit": {
        "severity": "CRITICAL",
        "condition": lambda m: m.current_drawdown > 25.0,
        "threshold": 25.0,
        "action": "HALT_ALL_TRADING",
        "notification": ["SMS", "EMAIL", "SLACK_URGENT"],
        "message_template": "🚨 CRITICAL: Drawdown exceeded at {current_drawdown:.2f}%"
    },
    # ... more conditions
}

SOFT_STOP_CONDITIONS = {
    "win_rate_degradation": {
        "severity": "WARNING",
        "condition": lambda m: m.win_rate_50_trades < 0.40,
        "threshold": 0.40,
        "notification": ["EMAIL", "SLACK"],
        "message_template": "⚠️ WARNING: Win rate dropped to {win_rate_50_trades:.1f}%"
    },
    # ... more conditions
}

class AlertManager:
    def __init__(self):
        self.redis = redis.from_url("redis://localhost:6379", decode_responses=True)
        self.alert_queries = AlertQueries()
        self.metrics_queries = MetricsQueries()
        self.notifications = NotificationService()
        self.is_running = False

    async def run_continuous_checks(self):
        """Background task to check alert conditions every 60s"""
        self.is_running = True

        while self.is_running:
            try:
                await self.check_all_conditions()
                await asyncio.sleep(60)
            except Exception as e:
                print(f"Error in alert manager: {e}")
                await asyncio.sleep(5)

    async def check_all_conditions(self):
        """Evaluate all alert conditions"""

        # Get current metrics
        from services.metrics_calculator import MetricsCalculator
        calculator = MetricsCalculator()
        metrics = await calculator.get_cached_real_time_metrics()

        # Check hard stop conditions
        for condition_name, config in HARD_STOP_CONDITIONS.items():
            if config["condition"](metrics):
                await self.trigger_alert(condition_name, config, metrics)

        # Check soft stop conditions
        for condition_name, config in SOFT_STOP_CONDITIONS.items():
            if config["condition"](metrics):
                await self.trigger_alert(condition_name, config, metrics)

    async def trigger_alert(self, condition_name: str, config: Dict, metrics: any):
        """Trigger an alert if not already sent recently"""

        # Check deduplication (prevent spam)
        if not await self.should_send_alert(condition_name, config["severity"]):
            return

        # Create alert record
        alert = Alert(
            time=datetime.utcnow(),
            severity=config["severity"],
            condition_name=condition_name,
            message=config["message_template"].format(**metrics.dict()),
            current_value=getattr(metrics, condition_name.split("_")[0], None),
            threshold=config.get("threshold"),
            acknowledged=False
        )

        # Save to database
        await self.alert_queries.create_alert(alert)

        # Send notifications
        for channel in config["notification"]:
            await self.notifications.send(channel, alert)

        # Execute action if specified
        if "action" in config:
            await self.execute_action(config["action"], alert)

    async def should_send_alert(self, condition_name: str, severity: str) -> bool:
        """Check if alert should be sent (deduplication)"""
        cache_key = f"alert_sent:{condition_name}"

        if await self.redis.exists(cache_key):
            return False  # Alert recently sent

        # Set throttle window
        ttl = {"INFO": 300, "WARNING": 1800, "CRITICAL": 600}[severity]
        await self.redis.setex(cache_key, ttl, "1")
        return True

    async def execute_action(self, action: str, alert: Alert):
        """Execute automated action (e.g., halt trading)"""

        if action == "HALT_ALL_TRADING":
            # Halt all trading systems
            from services.kill_switch import KillSwitch
            kill_switch = KillSwitch()
            await kill_switch.halt_all_trading(reason=alert.condition_name)

        elif action == "HALT_NEW_POSITIONS":
            from services.kill_switch import KillSwitch
            kill_switch = KillSwitch()
            await kill_switch.halt_new_positions(reason=alert.condition_name)

        elif action == "ALERT_OPERATOR":
            # Just notify, no automated halt
            pass

    async def alert_stream(self):
        """Generator that yields alerts as they occur (for WebSocket)"""
        pubsub = self.redis.pubsub()
        await pubsub.subscribe("alerts")

        async for message in pubsub.listen():
            if message["type"] == "message":
                alert_data = json.loads(message["data"])
                yield Alert.parse_obj(alert_data)
```

---

## 6. Database Queries

### 6.1 Metrics Queries

**backend/db/queries.py**:
```python
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime, timedelta
from typing import List, Optional
import numpy as np

from db.connection import get_db_session
from models.db_models import Trade, Signal, RegimeTransition, FeatureDistribution

class MetricsQueries:
    async def get_trades_last_n_hours(self, hours: int) -> List[Trade]:
        """Get all trades from last N hours"""
        async with get_db_session() as session:
            cutoff = datetime.utcnow() - timedelta(hours=hours)
            result = await session.execute(
                select(Trade).where(Trade.time >= cutoff).order_by(Trade.time.desc())
            )
            return result.scalars().all()

    async def get_signals_last_n_hours(self, hours: int) -> List[Signal]:
        """Get all signals from last N hours"""
        async with get_db_session() as session:
            cutoff = datetime.utcnow() - timedelta(hours=hours)
            result = await session.execute(
                select(Signal).where(Signal.time >= cutoff).order_by(Signal.time.desc())
            )
            return result.scalars().all()

    async def get_returns_last_n_days(self, days: int) -> np.ndarray:
        """Get hourly returns for Sharpe calculation"""
        async with get_db_session() as session:
            cutoff = datetime.utcnow() - timedelta(days=days)

            # Get hourly PnL aggregates
            result = await session.execute("""
                SELECT
                    time_bucket('1 hour', time) AS hour,
                    SUM(pnl_pct) AS hourly_return
                FROM trades
                WHERE time >= :cutoff
                GROUP BY hour
                ORDER BY hour
            """, {"cutoff": cutoff})

            returns = [row.hourly_return for row in result.fetchall()]
            return np.array(returns)

    async def get_session_start_equity(self) -> float:
        """Get equity at start of trading session (today 00:00 UTC)"""
        async with get_db_session() as session:
            # Simplified: return configured starting capital
            # In production, track equity from database
            return 100000.0  # $100K starting capital

    async def get_peak_equity(self) -> float:
        """Get all-time peak equity"""
        async with get_db_session() as session:
            result = await session.execute("""
                SELECT MAX(equity) AS peak
                FROM equity_history
            """)
            row = result.fetchone()
            return row.peak if row else 100000.0

    async def get_max_drawdown(self) -> float:
        """Get historical max drawdown"""
        async with get_db_session() as session:
            # Calculate from equity curve
            result = await session.execute("""
                WITH equity_with_peak AS (
                    SELECT
                        time,
                        equity,
                        MAX(equity) OVER (ORDER BY time) AS peak_equity
                    FROM equity_history
                )
                SELECT MIN((equity - peak_equity) / peak_equity * 100) AS max_dd
                FROM equity_with_peak
            """)
            row = result.fetchone()
            return row.max_dd if row else 0.0

    async def get_feature_backtest_distribution(self, feature_name: str) -> np.ndarray:
        """Get feature distribution from backtest period"""
        async with get_db_session() as session:
            result = await session.execute(
                select(FeatureDistribution.distribution)
                .where(FeatureDistribution.feature_name == feature_name)
                .where(FeatureDistribution.period == "backtest")
            )
            row = result.fetchone()
            return np.array(row.distribution) if row else np.array([])

    async def get_feature_live_distribution(self, feature_name: str, days: int = 7) -> np.ndarray:
        """Get feature distribution from live paper trading"""
        async with get_db_session() as session:
            cutoff = datetime.utcnow() - timedelta(days=days)

            # Fetch feature values from signals metadata
            result = await session.execute("""
                SELECT metadata->>:feature_name AS feature_value
                FROM signals
                WHERE time >= :cutoff
                AND metadata ? :feature_name
            """, {"feature_name": feature_name, "cutoff": cutoff})

            values = [float(row.feature_value) for row in result.fetchall() if row.feature_value]
            return np.array(values)
```

---

## 7. Testing Strategy

### 7.1 Unit Tests

**tests/test_drift_detector.py**:
```python
import pytest
import numpy as np
from backend.services.drift_detector import DriftDetector

def test_psi_no_drift():
    """Test PSI calculation when distributions are identical"""
    detector = DriftDetector()

    expected = np.random.normal(0, 1, 1000)
    actual = np.random.normal(0, 1, 1000)

    psi = detector._calculate_psi(expected, actual)

    # PSI should be very low (<0.1) for identical distributions
    assert psi < 0.1

def test_psi_significant_drift():
    """Test PSI calculation when distributions differ significantly"""
    detector = DriftDetector()

    expected = np.random.normal(0, 1, 1000)
    actual = np.random.normal(2, 1.5, 1000)  # Different mean and std

    psi = detector._calculate_psi(expected, actual)

    # PSI should be high (>0.25) for different distributions
    assert psi > 0.25

def test_cusum_positive_drift():
    """Test CUSUM detects win rate improvement"""
    detector = DriftDetector()

    # Simulate trades with improving win rate
    trades = []
    for i in range(100):
        win_rate = 0.55 if i < 50 else 0.70  # Win rate improves
        trades.append({"pnl": 100 if np.random.random() < win_rate else -100})

    # TODO: Implement test logic
```

### 7.2 Integration Tests

**tests/test_alert_system.py**:
```python
import pytest
from backend.services.alert_manager import AlertManager
from backend.models.schemas import RealTimeMetrics

@pytest.mark.asyncio
async def test_daily_loss_alert():
    """Test that daily loss limit triggers alert"""
    alert_manager = AlertManager()

    # Create metrics that exceed loss limit
    metrics = RealTimeMetrics(
        daily_pnl_pct=-6.0,  # Exceeds -5% threshold
        # ... other metrics
    )

    # Trigger alert
    await alert_manager.check_all_conditions()

    # Verify alert was created
    alerts = await alert_manager.alert_queries.get_recent_alerts(limit=1)
    assert len(alerts) == 1
    assert alerts[0].condition_name == "daily_loss_limit"
    assert alerts[0].severity == "CRITICAL"
```

### 7.3 Load Testing

**tests/load_test.py**:
```python
import asyncio
import aiohttp
from datetime import datetime

async def simulate_signal_load(signals_per_second: int, duration_seconds: int):
    """Simulate high signal volume to test dashboard performance"""

    async with aiohttp.ClientSession() as session:
        start_time = datetime.utcnow()

        while (datetime.utcnow() - start_time).total_seconds() < duration_seconds:
            tasks = []

            for _ in range(signals_per_second):
                # Create signal
                signal_data = {
                    "archetype_id": "A",
                    "direction": "LONG",
                    "price": 43000.0,
                    "confidence": 0.85,
                    # ... other fields
                }

                tasks.append(
                    session.post("http://localhost:8000/api/signals", json=signal_data)
                )

            await asyncio.gather(*tasks)
            await asyncio.sleep(1)

# Run load test: 100 signals/second for 60 seconds
asyncio.run(simulate_signal_load(signals_per_second=100, duration_seconds=60))
```

---

## 8. Deployment Guide

### 8.1 Docker Compose Production

**docker-compose.prod.yml**:
```yaml
version: '3.8'

services:
  dashboard:
    build:
      context: ./dashboard
      dockerfile: Dockerfile.prod
    ports:
      - "8501:8501"
    environment:
      - BACKEND_URL=http://backend:8000
      - REDIS_URL=redis://redis:6379
    depends_on:
      - backend
      - redis
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile.prod
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:${DB_PASSWORD}@timescale:5432/bull_machine
      - REDIS_URL=redis://redis:6379
      - LOG_LEVEL=INFO
    depends_on:
      - timescale
      - redis
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  timescale:
    image: timescale/timescaledb:latest-pg14
    ports:
      - "5432:5432"
    environment:
      - POSTGRES_DB=bull_machine
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - timescale_data:/var/lib/postgresql/data
      - ./scripts/init_db.sql:/docker-entrypoint-initdb.d/init.sql
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
      - GF_USERS_ALLOW_SIGN_UP=false
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./nginx/ssl:/etc/nginx/ssl
    depends_on:
      - dashboard
      - backend
      - grafana
    restart: unless-stopped

volumes:
  timescale_data:
  redis_data:
  prometheus_data:
  grafana_data:
```

### 8.2 Environment Variables

**.env.production**:
```bash
# Database
DB_PASSWORD=<strong_password>

# Grafana
GRAFANA_PASSWORD=<admin_password>

# Notifications
SENDGRID_API_KEY=<sendgrid_key>
TWILIO_ACCOUNT_SID=<twilio_sid>
TWILIO_AUTH_TOKEN=<twilio_token>
SLACK_WEBHOOK_URL=<slack_webhook>

# Trading System
PAPER_TRADING_START_DATE=2025-12-01
INITIAL_CAPITAL=100000

# Security
JWT_SECRET=<random_secret>
DASHBOARD_PASSWORD=<dashboard_password>
```

### 8.3 Deployment Commands

```bash
# 1. Clone repository
git clone https://github.com/yourorg/bull-machine.git
cd bull-machine

# 2. Set environment variables
cp .env.example .env.production
nano .env.production  # Edit with production values

# 3. Build and start services
docker-compose -f docker-compose.prod.yml up -d --build

# 4. Initialize database
docker-compose -f docker-compose.prod.yml exec backend python scripts/init_dashboard_db.py

# 5. Verify health
curl http://localhost:8000/health
curl http://localhost:8501

# 6. View logs
docker-compose -f docker-compose.prod.yml logs -f dashboard
docker-compose -f docker-compose.prod.yml logs -f backend
```

### 8.4 Monitoring & Maintenance

```bash
# Check system status
docker-compose -f docker-compose.prod.yml ps

# Restart services
docker-compose -f docker-compose.prod.yml restart dashboard backend

# Backup database
docker-compose -f docker-compose.prod.yml exec timescale pg_dump -U user bull_machine > backup_$(date +%Y%m%d).sql

# View Prometheus metrics
open http://localhost:9090

# View Grafana dashboards
open http://localhost:3000
```

---

## Summary

This implementation guide provides:

1. **Complete code examples** for FastAPI backend, Streamlit dashboard, metrics calculators
2. **Production-ready** database queries, caching strategies, WebSocket streaming
3. **Comprehensive testing** strategy (unit, integration, load tests)
4. **Docker deployment** with health checks, auto-restart, monitoring

**Next Steps**:
1. Review main specification document: `PAPER_TRADING_METRICS_DASHBOARD_SPEC.md`
2. Set up local development environment following Quick Start
3. Implement Phase 1 (Foundation) over Week 1-2
4. Deploy to staging environment for testing
5. Launch paper trading monitoring dashboard

**Key Files**:
- Spec: `/Users/raymondghandchi/Bull-machine-/Bull-machine-/PAPER_TRADING_METRICS_DASHBOARD_SPEC.md`
- Implementation: `/Users/raymondghandchi/Bull-machine-/Bull-machine-/PAPER_TRADING_DASHBOARD_IMPLEMENTATION_GUIDE.md`
