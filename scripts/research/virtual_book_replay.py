"""Bounded synthetic diagnostic of actual V11ShadowRunner.process_bar.

Use a dedicated single-threaded research process. The Python audit/socket/write
guard is process-wide while active, not an OS sandbox for hostile native code or
arbitrary pre-opened file objects. An inert audit hook remains after the scope.
No live/replay loops, saved-state loading, exchange objects, or disk outputs.

Only get_signals and logging/maker endpoints are replaced. Champion configuration
and native downstream branches remain unchanged, including bypass skipping the
allocator/spacing, unlimited paper margin, source fee accounting, and Series
entry-metadata quirks. Maker calculations and persisted logs are NOT certified.
Reconstruct state only by replaying the complete original prehistory into a new
fixture; snapshot is an observable view, never an importable restore blob.
"""
from contextlib import contextmanager, ExitStack
from copy import deepcopy
import hashlib
import importlib
from importlib import metadata
import json
import logging
import math
import os
from pathlib import Path
import platform
import socket
import sys
from unittest.mock import patch

import pandas as pd

from scripts.research.engine_signal_replay import EXPECTED_ARCHETYPES
from scripts.research.live_feature_replay import state_view
from scripts.research.replay_clock import digest, utc

ROOT = Path(__file__).resolve().parents[2]
CHAMPION = ROOT/'configs/champion_paper.json'
OUTPUT = ROOT/'results/live_signals'
_ACTIVE_GUARDS = []
_AUDIT_INSTALLED = False


def _audit(event, args):
    if not _ACTIVE_GUARDS:
        return
    write = False
    if event == 'open':
        _, mode, flags = args
        write = (isinstance(mode, str) and any(c in mode for c in 'wax+')) or (
            isinstance(flags, int) and bool(flags & (
                os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)))
    denied = write or event in {
        'os.mkdir', 'os.remove', 'os.rmdir', 'os.rename', 'os.link', 'os.symlink',
        'os.truncate', 'os.chmod', 'os.chown', 'os.utime', 'os.setxattr',
        'os.removexattr', 'os.chdir', 'os.system', 'os.exec', 'os.posix_spawn',
        'os.fork', 'os.forkpty', 'subprocess.Popen', 'pty.spawn',
    } or event.startswith('socket.')
    if denied:
        message = f'Prohibited side effect: {event}'
        for attempts in _ACTIVE_GUARDS:
            attempts.append(message)
        raise RuntimeError(message)


@contextmanager
def side_effect_guard(records):
    """Sticky denial: a swallowed source exception still fails the whole scope."""
    global _AUDIT_INSTALLED
    if not _AUDIT_INSTALLED:
        sys.addaudithook(_audit)
        _AUDIT_INSTALLED = True
    attempts = []

    def deny_write(*args, **kwargs):
        message = 'Prohibited side effect: write to existing descriptor'
        attempts.append(message)
        raise RuntimeError(message)

    def deny_network(*args, **kwargs):
        message = 'Prohibited side effect: network access'
        attempts.append(message)
        raise RuntimeError(message)

    def record_log(logger, record):
        records.append(dict(kind='python_log', payload=dict(
            logger=record.name, level=record.levelname, message=record.getMessage())))

    with ExitStack() as stack:
        stack.enter_context(patch.object(sys, 'dont_write_bytecode', True))
        stack.enter_context(patch.object(logging.Logger, 'handle', record_log))
        for name in ('write', 'pwrite', 'writev', 'sendfile', 'copy_file_range'):
            if hasattr(os, name):
                stack.enter_context(patch.object(os, name, deny_write))
        for obj, names in ((socket, ('create_connection', 'getaddrinfo')),
                           (socket.socket, ('connect', 'connect_ex', 'send',
                                            'sendall', 'sendto', 'sendmsg'))):
            for name in names:
                if hasattr(obj, name):
                    stack.enter_context(patch.object(obj, name, deny_network))
        _ACTIVE_GUARDS.append(attempts)
        try:
            yield
        finally:
            _ACTIVE_GUARDS.pop()
            if attempts:
                records.append(dict(kind='denied_side_effect', payload=list(attempts)))
                raise RuntimeError('Prohibited side effect inside virtual-book fixture: '
                                   + '; '.join(attempts))


class RecordingMaker:
    """Measurement endpoint only; deliberately excludes maker-fill calculations."""
    def __init__(self, records):
        self.records = records

    def on_bar(self, high, low, close, timestamp):
        self.records.append(dict(kind='maker_on_bar', payload=deepcopy(
            dict(high=high, low=low, close=close, timestamp=timestamp))))

    def record_entry(self, position_id, archetype, direction, price, timestamp):
        self.records.append(dict(kind='maker_entry', payload=deepcopy(dict(
            position_id=position_id, archetype=archetype, direction=direction,
            price=price, timestamp=timestamp))))


class VirtualBookFixture:
    def __init__(self, *, initial_cash, commission_rate, slippage_bps):
        self.records = []
        self.failed = False
        self.last_open = None
        self.maker = RecordingMaker(self.records)
        self.blockers = {
            'synthetic_controlled_signals_not_detector_replay',
            'cold_start_not_live_state', 'no_profitability_certificate',
            'maker_shadow_calculation_and_persistence_excluded',
            'csv_persistence_excluded', 'host_funding_daily_downtrend_and_tape_excluded',
            'python_guard_not_os_sandbox',
        }
        for key, value in dict(initial_cash=initial_cash, commission_rate=commission_rate,
                               slippage_bps=slippage_bps).items():
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f'{key} must be finite and nonnegative')
        if initial_cash <= 0 or slippage_bps >= 10000 or commission_rate >= 1:
            raise ValueError('Invalid initial cash, commission, or slippage')

        with side_effect_guard(self.records):
            # Preserve source-relative model/calibrator resolution, without chdir.
            if Path.cwd().resolve() != ROOT:
                raise ValueError('Run this bounded fixture from the repository root')
            module = importlib.import_module('bin.live.v11_shadow_runner')
            cls = module.V11ShadowRunner
            original_mkdir = Path.mkdir

            def constructor_mkdir(path, mode=0o777, parents=False, exist_ok=False):
                if path == OUTPUT and mode == 0o777 and parents and exist_ok:
                    self.records.append(dict(kind='constructor_mkdir_intercepted',
                                             payload=str(path)))
                    return
                return original_mkdir(path, mode=mode, parents=parents, exist_ok=exist_ok)

            def init_signal(runner):
                if runner.signal_log_path != OUTPUT/'signals.csv':
                    raise RuntimeError('Unexpected signal log path')
                self.records.append(dict(kind='init_signal_log', payload=str(runner.signal_log_path)))

            def init_outcome(runner):
                if runner.outcome_log_path != OUTPUT/'trade_outcomes.csv':
                    raise RuntimeError('Unexpected outcome log path')
                self.records.append(dict(kind='init_outcome_log', payload=str(runner.outcome_log_path)))

            with patch.object(Path, 'mkdir', constructor_mkdir), \
                 patch.object(cls, '_init_signal_log', init_signal), \
                 patch.object(cls, '_init_outcome_log', init_outcome):
                self.runner = cls(config_path=str(CHAMPION), initial_cash=initial_cash,
                                  commission_rate=commission_rate, slippage_bps=slippage_bps)
            if set(self.runner.engine.archetypes) != EXPECTED_ARCHETYPES or self.runner.disabled_archetypes:
                raise ValueError('This contract requires all 17 enabled champion archetypes')
            # Bind instance endpoints; all calculations upstream of them stay native.
            self.runner._maker_shadow = lambda: self.maker
            self.runner._log_signal = self._record_signal
            self.runner._append_outcome_row = self._record_outcome
            self.manifest = self._manifest(initial_cash, commission_rate, slippage_bps)
            self.contract_id = digest(self.manifest)

    def _record_signal(self, value):
        self.records.append(dict(kind='signal_log', payload=deepcopy(value)))

    def _record_outcome(self, pos, pnl, pnl_pct, exit_price, exit_timestamp, exit_reason):
        self.records.append(dict(kind='outcome_log', payload=deepcopy(dict(
            position=pos.to_dict(), pnl=pnl, pnl_pct=pnl_pct, exit_price=exit_price,
            exit_timestamp=exit_timestamp, exit_reason=exit_reason))))

    def _manifest(self, initial_cash, commission_rate, slippage_bps):
        config = self.runner.config
        directory = ROOT/config['archetype_config_dir']
        model = Path(config['regime_classifier']['model_path']).resolve()
        calibrator = ROOT/'models/confidence_calibrator_v1.pkl'
        optimized = ROOT/'configs/optimized/cmi_weights_optimized.json'
        paths = {CHAMPION, Path(__file__), model, calibrator, optimized,
                 ROOT/'scripts/research/live_feature_replay.py',
                 ROOT/'scripts/research/engine_signal_replay.py',
                 ROOT/'scripts/research/replay_clock.py'}
        paths.update((ROOT/'engine').rglob('*.py'))
        paths.update((ROOT/'bin/live').rglob('*.py'))
        paths.update(directory.glob('*.yaml'))
        paths.update(directory.glob('*.yml'))
        regime = self.runner.engine.regime_service
        actual_model = getattr(getattr(regime, 'probabilistic_detector', None), 'crisis_model', None)
        actual_calibrator = getattr(regime, 'confidence_calibrator', None)
        if not model.exists():
            self.blockers.add('regime_model_missing_source_mock_fallback')
        if 'MockCrisisModel' in type(actual_model).__qualname__:
            self.blockers.add('regime_model_mock_fallback')
        if not calibrator.exists():
            self.blockers.add('confidence_calibrator_missing')
        if actual_calibrator is None:
            self.blockers.add('confidence_calibrator_unavailable')
        if optimized.exists() and self.runner.cmi_weights_optimized is None:
            self.blockers.add('optimized_cmi_present_but_not_loaded')
        dependencies = {}
        for package in ('numpy', 'pandas', 'scipy', 'joblib', 'scikit-learn', 'PyYAML'):
            try:
                dependencies[package] = metadata.version(package)
            except metadata.PackageNotFoundError:
                dependencies[package] = None
        return dict(files={str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                           if path.exists() else None for path in sorted(paths)},
                    input_config=json.loads(CHAMPION.read_text()),
                    effective_config=deepcopy(config), archetype_order=list(self.runner.engine.archetypes),
                    constructor_parameters=dict(initial_cash=initial_cash,
                        commission_rate=commission_rate, slippage_bps=slippage_bps),
                    model_class=type(actual_model).__module__+'.'+type(actual_model).__qualname__,
                    calibrator_class=type(actual_calibrator).__module__+'.'+type(actual_calibrator).__qualname__,
                    optimized_cmi=deepcopy(self.runner.cmi_weights_optimized),
                    python=platform.python_version(), dependencies=dependencies,
                    cwd=str(ROOT), signal_source='controlled_deepcopy_once_per_bar',
                    input_clock='source hour open; availability exactly following hour close',
                    blockers=sorted(self.blockers))

    def step(self, features, signals, decision_time):
        if self.failed:
            raise RuntimeError('This fixture failed; use a fresh whole-prehistory replay')
        opened = utc(features['timestamp'])
        available = utc(decision_time)
        if opened != opened.floor('1h') or available != opened + pd.Timedelta('1h'):
            raise ValueError('Requires completed hourly features and exact hourly availability')
        if self.last_open is not None and opened != self.last_open + pd.Timedelta('1h'):
            raise ValueError('Duplicate/out-of-order/gapped virtual-book history')
        bar = pd.Series(deepcopy(features), name=opened)
        batch = deepcopy(list(signals))
        for sig in batch:
            if sig.archetype_id not in EXPECTED_ARCHETYPES or utc(sig.timestamp) != opened:
                raise ValueError('Controlled signals require enabled archetype and source hour timestamp')
        calls = 0

        def controlled_signals(*, bar, bar_index, prev_row, lookback_df):
            nonlocal calls
            calls += 1
            if calls != 1:
                raise RuntimeError('Multiple signal evaluations in one bar are prohibited')
            return deepcopy(batch)

        start = len(self.records)
        try:
            with side_effect_guard(self.records), patch.object(
                    self.runner.engine, 'get_signals', controlled_signals):
                acted = self.runner.process_bar(bar, opened)
                if calls != 1:
                    raise RuntimeError('Native process_bar must request one controlled batch')
        except BaseException:
            self.failed = True
            raise
        self.last_open = opened
        return dict(certified=False, contract_id=self.contract_id, source_hour_open=opened,
                    available_at=available, bar_index=self.runner.bar_index,
                    acted_signals=deepcopy(acted), last_bar_signals=deepcopy(self.runner.last_bar_signals),
                    cash=self.runner.cash, equity=self.runner.equity_curve[-1],
                    records=deepcopy(self.records[start:]), blockers=sorted(self.blockers))

    def snapshot(self):
        unsupported = set()
        # All runner data attributes, plus native engine/exit state; endpoint
        # functions and logging records have their own explicit representation.
        values = {key: value for key, value in vars(self.runner).items()
                  if not callable(value) and key not in {
                      'output_dir', 'signal_log_path', 'outcome_log_path', 'bar_buffer'}}
        values['bar_buffer'] = list(self.runner.bar_buffer)
        # The allocator's Enum contains a reference to its class/mappingproxy.
        # Describe that one known immutable type explicitly; retain all other
        # allocator/engine attributes and surface unknown types as blockers.
        engine = self.runner.engine
        allocator = engine.portfolio_allocator
        mode = allocator.allocation_mode
        allocator_state = dict(vars(allocator), allocation_mode=dict(
            enum=type(mode).__module__+'.'+type(mode).__qualname__,
            name=mode.name, value=mode.value))
        values['engine'] = dict(class_name=type(engine).__module__+'.'+type(engine).__qualname__,
            attributes=dict(vars(engine), portfolio_allocator=dict(
                class_name=type(allocator).__module__+'.'+type(allocator).__qualname__,
                attributes=allocator_state)))
        with side_effect_guard(self.records):
            state = state_view(values, unsupported)
        if unsupported:
            self.blockers.add('unsupported_virtual_book_state_types')
        return dict(state=state, last_open=self.last_open, failed=self.failed,
                    unsupported_state_types=sorted(unsupported), blockers=sorted(self.blockers))
