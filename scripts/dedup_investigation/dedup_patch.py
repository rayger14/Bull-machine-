"""
Dedup investigation patch — monkey-patches IsolatedArchetypeEngine._deduplicate_signals
to (a) log all dedup events to a CSV and (b) support alternative dedup modes
selected via the env var DEDUP_MODE.

This module is import-only — no command-line interface. It is consumed by
`run_dedup_diag.py` and `run_dedup_ablation.py` which import it BEFORE importing
the backtester module.

Recorded fields per dedup event:
    timestamp, n_signals, mode, winner_id, winner_dir, winner_fusion,
    winner_entry, winner_sl, loser_id, loser_fusion, loser_dir, loser_entry,
    loser_sl, blocked_by

One row per (winner, loser) pair on a given bar. If a bar has 1 signal, no row
is emitted. If a bar has 3 long signals, 2 rows (one per loser) are emitted.

Alternative modes available via DEDUP_MODE env var:
    - status_quo      : pure best_per_direction (baseline, max fusion)
    - normalized      : z-score fusion vs each archetype's 30-day rolling mean+std,
                        rank on percentile
    - unique_sl_zone  : keep both if stop levels don't overlap (delegates to engine
                        built-in `unique_sl_zone`)
    - round_robin     : prefer the archetype with the lowest trade count so far
    - pass_through    : take ALL fired signals (no dedup)
    - hybrid_rr_fusion: round-robin tiebreak only when fusion scores within delta=0.05

The patch must be applied BEFORE the backtester imports IsolatedArchetypeEngine.
"""
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from engine.integrations import isolated_archetype_engine as _iae

ArchetypeSignal = _iae.ArchetypeSignal

# Per-archetype rolling fusion-score buffer (for normalized mode).
# Keyed by archetype_id -> list[float], capped at 720 entries (30 days x 24 H).
_FUSION_BUFFER: Dict[str, List[float]] = defaultdict(list)
_FUSION_BUFFER_CAP = 720

# Per-archetype trade count (for round_robin mode).
# Reset at the start of each backtest run via reset_state().
_TRADE_COUNT: Dict[str, int] = defaultdict(int)

# CSV writer state (lazy init in patched method).
_CSV_PATH: Optional[Path] = None
_CSV_FILE = None
_CSV_WRITER: Optional[csv.writer] = None

_CSV_HEADER = [
    'timestamp', 'n_signals', 'mode',
    'winner_id', 'winner_dir', 'winner_fusion',
    'winner_entry', 'winner_sl',
    'loser_id', 'loser_fusion', 'loser_dir',
    'loser_entry', 'loser_sl',
    'blocked_by',
]


def _ensure_csv():
    """Open CSV at path from env var DEDUP_LOG_PATH if not already open."""
    global _CSV_PATH, _CSV_FILE, _CSV_WRITER
    if _CSV_WRITER is not None:
        return
    path = os.environ.get('DEDUP_LOG_PATH', '').strip()
    if not path:
        return  # not requested → silently skip logging
    _CSV_PATH = Path(path)
    _CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    is_new = not _CSV_PATH.exists()
    _CSV_FILE = open(_CSV_PATH, 'a', newline='')
    _CSV_WRITER = csv.writer(_CSV_FILE)
    if is_new:
        _CSV_WRITER.writerow(_CSV_HEADER)


def reset_state():
    """Reset trade counts and fusion buffers. Call at the start of each run."""
    _FUSION_BUFFER.clear()
    _TRADE_COUNT.clear()


def record_trade(archetype_id: str):
    """Increment per-archetype trade counter (for round_robin mode)."""
    _TRADE_COUNT[archetype_id] += 1


def _normalized_score(sig: ArchetypeSignal) -> float:
    """Z-score the fusion score against the archetype's rolling history."""
    buf = _FUSION_BUFFER.get(sig.archetype_id, [])
    if len(buf) < 20:
        # Insufficient history → return raw fusion (don't penalize new archetypes)
        return sig.fusion_score
    mean = sum(buf) / len(buf)
    var = sum((x - mean) ** 2 for x in buf) / max(len(buf) - 1, 1)
    std = var ** 0.5
    if std < 1e-9:
        return sig.fusion_score
    return (sig.fusion_score - mean) / std


def _push_fusion_buf(sig: ArchetypeSignal):
    """Push the fusion score into the rolling buffer for normalized mode."""
    buf = _FUSION_BUFFER[sig.archetype_id]
    buf.append(sig.fusion_score)
    if len(buf) > _FUSION_BUFFER_CAP:
        buf.pop(0)


def _patched_deduplicate_signals(self, signals, mode='best_per_direction'):
    """
    Drop-in replacement for IsolatedArchetypeEngine._deduplicate_signals.

    Behavior:
    1. Always pushes fusion scores into the per-archetype rolling buffer (so
       'normalized' mode has data even when called with 'status_quo').
    2. If env var DEDUP_MODE is set, overrides the requested mode.
    3. Logs winner/loser pairs to DEDUP_LOG_PATH CSV if set.
    """
    # Push fusion buffer for ALL signals (regardless of mode).
    for s in signals:
        _push_fusion_buf(s)

    # Env-var override of mode.
    override = os.environ.get('DEDUP_MODE', '').strip()
    if override:
        mode = override

    _ensure_csv()

    if mode in ('disabled', 'pass_through') or len(signals) <= 1:
        return signals

    # ============================================================
    # Mode dispatch
    # ============================================================
    if mode in ('best_of_bar',):
        winner = max(signals, key=lambda s: s.fusion_score)
        result = [winner]

    elif mode in ('best_per_direction', 'status_quo'):
        winner = None
        result = []
        longs = [s for s in signals if s.direction == 'long']
        shorts = [s for s in signals if s.direction == 'short']
        if longs:
            best_long = max(longs, key=lambda s: s.fusion_score)
            result.append(best_long)
        if shorts:
            best_short = max(shorts, key=lambda s: s.fusion_score)
            result.append(best_short)

    elif mode == 'normalized':
        # Per-direction; score each by z-score, pick the max z.
        result = []
        for direction in ('long', 'short'):
            cands = [s for s in signals if s.direction == direction]
            if not cands:
                continue
            best = max(cands, key=lambda s: _normalized_score(s))
            result.append(best)

    elif mode == 'round_robin':
        # Per-direction; prefer the archetype with the lowest trade count.
        # Tiebreak: highest fusion score.
        result = []
        for direction in ('long', 'short'):
            cands = [s for s in signals if s.direction == direction]
            if not cands:
                continue
            best = min(
                cands,
                key=lambda s: (_TRADE_COUNT.get(s.archetype_id, 0), -s.fusion_score),
            )
            result.append(best)

    elif mode == 'hybrid_rr_fusion':
        # Round-robin tiebreak when fusion within delta=0.05 of leader.
        DELTA = 0.05
        result = []
        for direction in ('long', 'short'):
            cands = [s for s in signals if s.direction == direction]
            if not cands:
                continue
            top_fusion = max(s.fusion_score for s in cands)
            close = [s for s in cands if (top_fusion - s.fusion_score) <= DELTA]
            best = min(
                close,
                key=lambda s: (_TRADE_COUNT.get(s.archetype_id, 0), -s.fusion_score),
            )
            result.append(best)

    elif mode == 'unique_sl_zone':
        # Use the engine's built-in unique_sl_zone via super-call semantics.
        # We replicate the logic here to avoid recursion.
        sorted_sigs = sorted(signals, key=lambda s: s.fusion_score, reverse=True)
        kept = []
        used_zones = []
        for sig in sorted_sigs:
            is_dup = False
            for (d, sl) in used_zones:
                if d != sig.direction:
                    continue
                if sig.entry_price > 0:
                    sl_diff_pct = abs(sig.stop_loss - sl) / sig.entry_price
                    if sl_diff_pct < 0.02:
                        is_dup = True
                        break
            if not is_dup:
                kept.append(sig)
                used_zones.append((sig.direction, sig.stop_loss))
        result = kept

    else:
        # Unknown mode → fall back to status quo.
        result = []
        for direction in ('long', 'short'):
            cands = [s for s in signals if s.direction == direction]
            if not cands:
                continue
            best = max(cands, key=lambda s: s.fusion_score)
            result.append(best)

    # ============================================================
    # Logging
    # ============================================================
    if _CSV_WRITER is not None:
        kept_ids = {id(s) for s in result}
        for winner in result:
            losers = [
                s for s in signals
                if s.direction == winner.direction and id(s) not in kept_ids
            ]
            for loser in losers:
                _CSV_WRITER.writerow([
                    winner.timestamp.isoformat() if winner.timestamp else '',
                    len(signals),
                    mode,
                    winner.archetype_id,
                    winner.direction,
                    f'{winner.fusion_score:.6f}',
                    f'{winner.entry_price:.4f}',
                    f'{winner.stop_loss:.4f}',
                    loser.archetype_id,
                    f'{loser.fusion_score:.6f}',
                    loser.direction,
                    f'{loser.entry_price:.4f}',
                    f'{loser.stop_loss:.4f}',
                    winner.archetype_id,
                ])
        _CSV_FILE.flush()

    # Stats bookkeeping (matches original engine).
    removed = len(signals) - len(result)
    if removed > 0:
        self.stats.setdefault('dedup_removed', 0)
        self.stats['dedup_removed'] += removed

    return result


def apply_patch():
    """Install the monkey-patch. Idempotent."""
    if getattr(_iae.IsolatedArchetypeEngine, '_dedup_patched', False):
        return
    _iae.IsolatedArchetypeEngine._original_deduplicate_signals = (
        _iae.IsolatedArchetypeEngine._deduplicate_signals
    )
    _iae.IsolatedArchetypeEngine._deduplicate_signals = _patched_deduplicate_signals
    _iae.IsolatedArchetypeEngine._dedup_patched = True
    print(
        f'[dedup_patch] applied — mode override={os.environ.get("DEDUP_MODE", "(none)")} '
        f'log={os.environ.get("DEDUP_LOG_PATH", "(none)")}',
        file=sys.stderr,
    )


# Auto-apply on import.
apply_patch()
