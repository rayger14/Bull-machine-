#!/usr/bin/env python3
"""
Cointegration Detector for BTC vs Macro Indicators

Computes Engle-Granger cointegration tests between BTC price and macro
indicators (DXY, Gold, VIX, USDT.D, BTC.D, Oil) to identify mean-reversion
opportunities when spreads deviate significantly from equilibrium.

Uses statsmodels ADF test if available; otherwise falls back to a pure-numpy
OLS residual approach with rolling z-score.

Designed to be called every 4 hours from coinbase_runner.py, using the
feature_history ring buffer (list of dicts with macro values).

Author: Claude Code
Date: 2026-02-18
"""

import logging
from datetime import datetime, timezone
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try importing statsmodels for proper ADF test
# ---------------------------------------------------------------------------
STATSMODELS_AVAILABLE = False
try:
    from statsmodels.tsa.stattools import adfuller
    STATSMODELS_AVAILABLE = True
except ImportError:
    logger.info(
        "statsmodels not available; using OLS residual fallback for cointegration."
    )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MIN_BARS_REQUIRED = 100  # Minimum data points for meaningful cointegration test
DEFAULT_WINDOW = 168     # 1 week of hourly bars

# Pair definitions: (feature_key, display_name, expected_relationship, interpretation_templates)
PAIR_DEFINITIONS = [
    {
        "feature_key": "dxy_z",
        "pair_name": "BTC-DXY",
        "expected": "inverse",
        "desc_positive_z": "BTC overvalued vs DXY (dollar weakness not priced in)",
        "desc_negative_z": "BTC undervalued vs DXY (dollar strength overdone)",
        "desc_neutral": "BTC-DXY spread near equilibrium",
    },
    {
        "feature_key": "gold_z",
        "pair_name": "BTC-Gold",
        "expected": "positive",
        "desc_positive_z": "BTC outpacing gold (inflation-hedge premium stretched)",
        "desc_negative_z": "BTC lagging gold (catch-up trade expected)",
        "desc_neutral": "BTC-Gold spread near equilibrium",
    },
    {
        "feature_key": "vix_z",
        "pair_name": "BTC-VIX",
        "expected": "inverse",
        "desc_positive_z": "BTC elevated despite risk (VIX not yet priced in)",
        "desc_negative_z": "BTC depressed relative to calm VIX (recovery expected)",
        "desc_neutral": "BTC-VIX relationship near equilibrium",
    },
    {
        "feature_key": "usdt_d",
        "pair_name": "BTC-USDT.D",
        "expected": "inverse",
        "desc_positive_z": "USDT.D depressed while BTC elevated (frothy, caution)",
        "desc_negative_z": "USDT.D elevated — mean reversion expected (BTC recovery)",
        "desc_neutral": "BTC-USDT.D spread near equilibrium",
    },
    {
        "feature_key": "btc_d",
        "pair_name": "BTC-BTC.D",
        "expected": "positive",
        "desc_positive_z": "BTC price stretched above dominance-implied level",
        "desc_negative_z": "BTC price below dominance-implied level (undervalued)",
        "desc_neutral": "BTC-BTC.D relationship near equilibrium",
    },
    {
        "feature_key": "oil_z",
        "pair_name": "BTC-Oil",
        "expected": "inverse",
        "desc_positive_z": "BTC elevated despite rising oil (inflation headwind ahead)",
        "desc_negative_z": "BTC depressed despite falling oil (deflationary tailwind)",
        "desc_neutral": "BTC-Oil spread near equilibrium",
    },
]


# ---------------------------------------------------------------------------
# Core cointegration functions
# ---------------------------------------------------------------------------

def _safe_series(history: List[Dict], key: str) -> np.ndarray:
    """Extract a time series from feature_history, handling NaN values."""
    values = []
    for h in history:
        v = h.get(key, float("nan"))
        if v is None:
            v = float("nan")
        try:
            f = float(v)
        except (TypeError, ValueError):
            f = float("nan")
        values.append(f)
    return np.array(values)


def _clean_paired_series(
    y: np.ndarray, x: np.ndarray
) -> tuple:
    """Remove rows where either series has NaN. Returns (y_clean, x_clean)."""
    mask = np.isfinite(y) & np.isfinite(x)
    return y[mask], x[mask]


def _ols_residuals(y: np.ndarray, x: np.ndarray) -> tuple:
    """
    Compute OLS regression y = beta * x + alpha and return residuals.
    Returns (residuals, beta, alpha).
    """
    if len(x) < 3:
        return np.array([]), 0.0, 0.0

    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x_var = np.var(x)

    if x_var < 1e-12:
        return np.array([]), 0.0, 0.0

    beta = np.sum((x - x_mean) * (y - y_mean)) / (len(x) * x_var)
    alpha = y_mean - beta * x_mean
    residuals = y - (beta * x + alpha)

    return residuals, beta, alpha


def _adf_test(residuals: np.ndarray) -> tuple:
    """
    Run ADF test on residuals.
    Returns (adf_statistic, p_value).
    Uses statsmodels if available, otherwise a simple autocorrelation proxy.
    """
    if len(residuals) < 10:
        return 0.0, 1.0

    if STATSMODELS_AVAILABLE:
        try:
            # maxlag based on sample size; use Schwarz info criterion
            max_lag = min(int(np.floor(12 * (len(residuals) / 100) ** 0.25)), len(residuals) // 4)
            max_lag = max(1, max_lag)
            result = adfuller(residuals, maxlag=max_lag, autolag="AIC")
            return float(result[0]), float(result[1])
        except Exception as exc:
            logger.debug("ADF test failed: %s. Using fallback.", exc)

    # Fallback: approximate stationarity test via first-order autocorrelation.
    # If |rho| < 1, residuals are mean-reverting. Lower rho = more cointegrated.
    # Map to approximate p-value using rough empirical relationship.
    n = len(residuals)
    r1 = residuals[:-1]
    r2 = residuals[1:]
    if np.std(r1) < 1e-10:
        return 0.0, 1.0

    rho = np.corrcoef(r1, r2)[0, 1]
    if rho != rho:  # NaN
        return 0.0, 1.0

    # Dickey-Fuller t-statistic approximation: t = (rho - 1) / se(rho)
    se_rho = np.sqrt((1 - rho ** 2) / max(n - 2, 1))
    if se_rho < 1e-10:
        t_stat = -10.0 if rho < 0.95 else 0.0
    else:
        t_stat = (rho - 1.0) / se_rho

    # Rough p-value mapping for Dickey-Fuller distribution (n ~ 100-200):
    # t < -3.5 => p < 0.01
    # t < -2.9 => p < 0.05
    # t < -2.6 => p < 0.10
    # t > -1.5 => p > 0.50
    if t_stat < -3.5:
        p_value = 0.005
    elif t_stat < -2.9:
        p_value = 0.03
    elif t_stat < -2.6:
        p_value = 0.08
    elif t_stat < -2.0:
        p_value = 0.20
    elif t_stat < -1.5:
        p_value = 0.40
    else:
        p_value = 0.80

    return float(t_stat), float(p_value)


def _half_life(residuals: np.ndarray) -> float:
    """
    Compute the half-life of mean reversion from the residuals.
    Uses first-order autocorrelation: half_life = -ln(2) / ln(rho).
    Returns half-life in bars (hours). Returns NaN if not mean-reverting.
    """
    if len(residuals) < 5:
        return float("nan")

    r1 = residuals[:-1]
    r2 = residuals[1:]

    if np.std(r1) < 1e-10 or np.std(r2) < 1e-10:
        return float("nan")

    rho = np.corrcoef(r1, r2)[0, 1]
    if rho != rho:  # NaN
        return float("nan")

    # Rho must be between 0 and 1 for valid half-life (mean-reverting)
    if rho <= 0 or rho >= 1.0:
        return float("nan")

    hl = -np.log(2) / np.log(rho)

    # Sanity: half-life should be positive and finite
    if hl <= 0 or not np.isfinite(hl) or hl > 10000:
        return float("nan")

    return float(hl)


def _rolling_stability(
    y: np.ndarray, x: np.ndarray, window: int = 50, step: int = 10
) -> str:
    """
    Check if the cointegration relationship is stable over rolling sub-windows.
    Returns 'stable', 'weakening', or 'unstable'.
    """
    n = len(y)
    if n < window + step:
        return "insufficient_data"

    # Compute beta in rolling windows
    betas = []
    for start in range(0, n - window + 1, step):
        end = start + window
        sub_y = y[start:end]
        sub_x = x[start:end]
        _, beta, _ = _ols_residuals(sub_y, sub_x)
        if beta == 0.0 and np.std(sub_x) < 1e-10:
            continue
        betas.append(beta)

    if len(betas) < 3:
        return "insufficient_data"

    betas = np.array(betas)
    beta_std = np.std(betas)
    beta_mean = np.mean(betas)

    # Coefficient of variation of beta: how much does the relationship drift?
    if abs(beta_mean) < 1e-10:
        cv = beta_std * 100  # Use absolute std if mean is near zero
    else:
        cv = abs(beta_std / beta_mean)

    if cv < 0.3:
        return "stable"
    elif cv < 0.7:
        return "weakening"
    else:
        return "unstable"


def _generate_signal(
    pair_def: dict, z_score: float, cointegrated: bool, stability: str
) -> str:
    """Generate a human-readable signal description."""
    if not cointegrated:
        return "Not cointegrated -- no mean-reversion signal"

    if stability == "unstable":
        return f"Relationship unstable -- spread z={z_score:+.1f} but unreliable"

    if abs(z_score) < 1.0:
        return pair_def["desc_neutral"]

    if z_score > 1.5:
        return f"Spread widening (+{z_score:.1f}s) -- {pair_def['desc_positive_z']}"
    elif z_score < -1.5:
        return f"Spread widening ({z_score:.1f}s) -- {pair_def['desc_negative_z']}"
    elif z_score > 1.0:
        return f"Spread slightly elevated (+{z_score:.1f}s) -- {pair_def['desc_positive_z']}"
    else:
        return f"Spread slightly depressed ({z_score:.1f}s) -- {pair_def['desc_negative_z']}"


# ---------------------------------------------------------------------------
# Main detector class
# ---------------------------------------------------------------------------

class CointegrationDetector:
    """
    Analyzes cointegration between BTC price and macro indicators.

    Usage:
        detector = CointegrationDetector()
        results = detector.analyze(feature_history)
        # results is a dict suitable for heartbeat.json["cointegration"]
    """

    def __init__(self, min_bars: int = MIN_BARS_REQUIRED, window: int = DEFAULT_WINDOW):
        self.min_bars = min_bars
        self.window = window

    def analyze(self, feature_history: List[Dict]) -> Dict:
        """
        Run cointegration analysis on all configured pairs.

        Args:
            feature_history: list of dicts with keys: close, dxy_z, gold_z,
                vix_z, usdt_d, btc_d, oil_z (from _append_feature_snapshot)

        Returns:
            Dict with keys: pairs (list), last_computed (ISO timestamp),
                n_bars_available, has_opportunity (bool)
        """
        n_bars = len(feature_history)

        if n_bars < self.min_bars:
            return {
                "pairs": [],
                "last_computed": datetime.now(timezone.utc).isoformat(),
                "n_bars_available": n_bars,
                "min_bars_required": self.min_bars,
                "status": f"Insufficient data ({n_bars}/{self.min_bars} bars)",
                "has_opportunity": False,
            }

        # Use the most recent 'window' bars (or all available if less)
        history = feature_history[-self.window:]

        # Extract BTC close prices (the dependent variable)
        btc_prices = _safe_series(history, "close")

        pair_results = []
        has_opp = False

        for pair_def in PAIR_DEFINITIONS:
            result = self._analyze_pair(pair_def, btc_prices, history)
            pair_results.append(result)
            if result.get("has_opportunity", False):
                has_opp = True

        return {
            "pairs": pair_results,
            "last_computed": datetime.now(timezone.utc).isoformat(),
            "n_bars_available": n_bars,
            "n_bars_used": len(history),
            "min_bars_required": self.min_bars,
            "status": "active",
            "has_opportunity": has_opp,
            "method": "ADF (statsmodels)" if STATSMODELS_AVAILABLE else "OLS + autocorrelation proxy",
        }

    def _analyze_pair(
        self, pair_def: dict, btc_prices: np.ndarray, history: List[Dict]
    ) -> Dict:
        """Analyze a single BTC vs macro pair."""
        feature_key = pair_def["feature_key"]
        pair_name = pair_def["pair_name"]

        macro_series = _safe_series(history, feature_key)

        # Clean paired data (remove NaN rows)
        btc_clean, macro_clean = _clean_paired_series(btc_prices, macro_series)

        n_valid = len(btc_clean)
        if n_valid < 20:
            return {
                "pair": pair_name,
                "cointegrated": False,
                "p_value": None,
                "half_life_hours": None,
                "current_zscore": None,
                "signal": f"Insufficient valid data ({n_valid} bars)",
                "stability": "insufficient_data",
                "has_opportunity": False,
                "n_valid_bars": n_valid,
            }

        # Step 1: OLS regression (BTC = beta * macro + alpha)
        residuals, beta, alpha = _ols_residuals(btc_clean, macro_clean)

        if len(residuals) < 10:
            return {
                "pair": pair_name,
                "cointegrated": False,
                "p_value": None,
                "half_life_hours": None,
                "current_zscore": None,
                "signal": "Regression failed",
                "stability": "insufficient_data",
                "has_opportunity": False,
                "n_valid_bars": n_valid,
            }

        # Step 2: ADF test on residuals
        adf_stat, p_value = _adf_test(residuals)
        cointegrated = p_value < 0.05

        # Step 3: Half-life of mean reversion
        hl = _half_life(residuals)
        half_life_hours = round(hl, 1) if np.isfinite(hl) else None

        # Step 4: Current spread z-score
        res_mean = np.mean(residuals)
        res_std = np.std(residuals)
        if res_std > 1e-10:
            current_zscore = round(float((residuals[-1] - res_mean) / res_std), 2)
        else:
            current_zscore = 0.0

        # Step 5: Rolling stability
        stability = _rolling_stability(btc_clean, macro_clean)

        # Step 6: Generate signal text
        signal_text = _generate_signal(pair_def, current_zscore, cointegrated, stability)

        # Determine if there is a trading opportunity
        has_opportunity = (
            cointegrated
            and stability in ("stable", "weakening")
            and abs(current_zscore) >= 2.0
        )

        return {
            "pair": pair_name,
            "cointegrated": cointegrated,
            "p_value": round(p_value, 4) if p_value is not None else None,
            "adf_statistic": round(adf_stat, 3),
            "half_life_hours": half_life_hours,
            "current_zscore": current_zscore,
            "signal": signal_text,
            "stability": stability,
            "has_opportunity": has_opportunity,
            "n_valid_bars": n_valid,
            "beta": round(beta, 6),
        }


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

_detector = CointegrationDetector()


def compute_cointegration(feature_history: List[Dict]) -> Dict:
    """
    Module-level convenience function for coinbase_runner integration.

    Args:
        feature_history: list of feature snapshot dicts

    Returns:
        Cointegration results dict for heartbeat.json
    """
    return _detector.analyze(feature_history)
