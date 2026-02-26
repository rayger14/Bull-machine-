"""
Observability Utilities

Provides param echo and gate tracing to diagnose wiring issues
in archetype detection and fusion scoring.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from collections import defaultdict
from datetime import datetime

logger = logging.getLogger(__name__)


class ParamEcho:
    """
    Captures actual parameters used by each archetype during a run.

    Writes artifacts/<run_id>/params_used.json with:
    {
      "trap_within_trend": {
        "slug": "trap_within_trend",
        "class": "TrapWithinTrend",
        "params": {"fusion_threshold": 0.35, "liquidity_threshold": 0.30, ...},
        "source": "config['archetypes']['trap_within_trend']"
      },
      ...
    }
    """

    def __init__(self, output_dir: Optional[str] = None, enabled: bool = True):
        self.enabled = enabled
        self.output_dir = Path(output_dir) if output_dir else Path("artifacts/default_run")
        self.params = {}  # slug -> param snapshot

        if self.enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"ParamEcho enabled: {self.output_dir / 'params_used.json'}")

    def record(self, slug: str, class_name: str, params: Dict[str, Any], source: str):
        """Record parameters actually read by an archetype."""
        if not self.enabled:
            return

        self.params[slug] = {
            "slug": slug,
            "class": class_name,
            "params": params,
            "source": source,
            "recorded_at": datetime.now().isoformat()
        }

    def write(self):
        """Write captured params to disk."""
        if not self.enabled or not self.params:
            return

        output_path = self.output_dir / "params_used.json"
        with open(output_path, 'w') as f:
            json.dump(self.params, f, indent=2)

        logger.info(f"ParamEcho written: {output_path} ({len(self.params)} archetypes)")


class GateTracer:
    """
    Tracks gate pass/fail rates for each archetype check.

    Writes artifacts/<run_id>/gate_stats/<slug>.json with:
    {
      "slug": "trap_within_trend",
      "total_bars": 10000,
      "gates": {
        "liquidity_min": {"pass": 4200, "fail": 5800, "rate": 0.42},
        "htf_quality": {"pass": 3100, "fail": 6900, "rate": 0.31},
        "fusion_threshold": {"pass": 2800, "fail": 7200, "rate": 0.28}
      },
      "final_matches": 180
    }
    """

    def __init__(self, output_dir: Optional[str] = None, enabled: bool = True):
        self.enabled = enabled
        self.output_dir = Path(output_dir) if output_dir else Path("artifacts/default_run")
        self.gate_stats_dir = self.output_dir / "gate_stats"

        # slug -> gate_name -> {"pass": count, "fail": count}
        self.stats = defaultdict(lambda: defaultdict(lambda: {"pass": 0, "fail": 0}))
        self.matches = defaultdict(int)  # slug -> final match count
        self.total_bars = 0

        if self.enabled:
            self.gate_stats_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"GateTracer enabled: {self.gate_stats_dir}")

    def trace(self, slug: str, gate_name: str, passed: bool):
        """Record a single gate check result."""
        if not self.enabled:
            return

        key = "pass" if passed else "fail"
        self.stats[slug][gate_name][key] += 1

    def record_match(self, slug: str):
        """Record a final archetype match."""
        if not self.enabled:
            return

        self.matches[slug] += 1

    def increment_bars(self):
        """Track total bars evaluated."""
        if not self.enabled:
            return

        self.total_bars += 1

    def write(self):
        """Write gate stats to disk."""
        if not self.enabled or not self.stats:
            return

        for slug, gates in self.stats.items():
            output_data = {
                "slug": slug,
                "total_bars": self.total_bars,
                "gates": {},
                "final_matches": self.matches.get(slug, 0)
            }

            # Calculate pass rates
            for gate_name, counts in gates.items():
                total = counts["pass"] + counts["fail"]
                rate = counts["pass"] / total if total > 0 else 0.0
                output_data["gates"][gate_name] = {
                    "pass": counts["pass"],
                    "fail": counts["fail"],
                    "rate": round(rate, 4)
                }

            # Write per-archetype file
            output_path = self.gate_stats_dir / f"{slug}.json"
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2)

        logger.info(f"GateTracer written: {len(self.stats)} archetype stat files")


# Global instances (can be overridden per run)
_param_echo_instance = None
_gate_tracer_instance = None


def init_observability(output_dir: str, enable_echo: bool = True, enable_tracing: bool = True):
    """Initialize global observability instances for a run."""
    global _param_echo_instance, _gate_tracer_instance

    _param_echo_instance = ParamEcho(output_dir, enabled=enable_echo)
    _gate_tracer_instance = GateTracer(output_dir, enabled=enable_tracing)

    return _param_echo_instance, _gate_tracer_instance


def get_param_echo() -> ParamEcho:
    """Get global ParamEcho instance."""
    global _param_echo_instance
    if _param_echo_instance is None:
        _param_echo_instance = ParamEcho(enabled=False)
    return _param_echo_instance


def get_gate_tracer() -> GateTracer:
    """Get global GateTracer instance."""
    global _gate_tracer_instance
    if _gate_tracer_instance is None:
        _gate_tracer_instance = GateTracer(enabled=False)
    return _gate_tracer_instance


def finalize_observability():
    """Write all observability artifacts to disk."""
    if _param_echo_instance:
        _param_echo_instance.write()
    if _gate_tracer_instance:
        _gate_tracer_instance.write()
