# Changelog

All notable changes to Bull Machine will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- MIT LICENSE file for legal clarity
- CONTRIBUTING.md with development guidelines
- CHANGELOG.md for version history tracking
- requirements.txt for dependency management

### Changed
- Updated .gitignore to allow documentation diagrams in docs/
- Cleaned root directory: moved 43 logs, 12 DBs, and 5 temp docs to archive

### Fixed
- Synchronized version between README (v1.8.6) and pyproject.toml (was v1.7.2)

## [1.8.6] - 2025-01-XX

### Added - Temporal Intelligence Module
- **Gann Cycles**: 30/60/90 day ACF vibrations for cycle detection
- **Square of 9**: Sacred geometry price level proximity scoring
- **Gann Angles**: 1×1, 2×1 support/resistance analysis
- **Thermo-floor**: Mining cost floor calculation (energy + difficulty)
- **Log Premium**: Difficulty-based time multiplier for value assessment
- **Logistic Bid**: Institutional re-accumulation probability
- **LPPLS Blowoff**: Power law bubble detection with veto capability

### Added - Testing & Validation
- 25 temporal intelligence unit tests (all passing)
- 25 v1.8.5 module unit tests (all passing)
- Full year 2025 validation: 7 trades, +1.95% return, 57% WR, 5.8% max DD

### Changed
- Complete v1.8 stack now integrated: Fusion + Exits + Sizing + Batch + Logic + Temporal
- Production codebase: 12,837 lines across all domain engines
- 3-tier configuration system: Conservative (0.70), Moderate (0.60), Fast (0.60+opts)

## [1.8.5] - 2025-01-XX

### Added - Enhanced Logic
- Negative Fibonacci retracements for extended moves
- Fourier filter for noise reduction in signals
- Event tagging system for narrative detection
- Enhanced confluence logic with multi-domain integration

## [1.8.4] - 2025-01-XX

### Added - Batch Mode
- Window merging for efficient batch processing
- Auto-fallback mechanism for performance optimization
- Batch processing framework for historical analysis

## [1.8.3] - 2025-01-XX

### Added - Dynamic Position Sizing
- Fusion score adaptive sizing (confluence-based allocation)
- ADX trend strength multiplier
- Dynamic position sizing based on market conditions

## [1.8.2] - 2025-01-XX

### Added - Smart Exits
- Regime-adaptive stop losses
- Liquidity trap protection mechanism
- Enhanced exit logic with market context awareness

## [1.8.1] - 2025-01-XX

### Added - True Fusion
- Real Wyckoff domain engine implementation
- Smart Money Concepts (SMC) integration
- Harmonic Order Blocks (HOB) detector
- Momentum domain with divergence detection
- True multi-domain confluence system

## [1.7.3] - 2025-01-XX

### Added - Live Trading Pipeline
- **Mock Feed Runner**: CSV replay with MTF alignment (168 ETH, 97 SOL, 263 BTC signals)
- **Paper Trading**: Realistic execution simulation with P&L tracking
- **Shadow Mode**: Log-only signal tracking for live monitoring
- **Health Monitoring**: Macro veto rate, SMC hit rate tracking
- Live trading presets: ETH_conservative, BTC_vanilla, SOL_tuned

### Added - Macro Context Integration
- VIX hysteresis guards (on=22.0, off=18.0) with state memory
- Macro veto integration with suppression flags
- Fire rate monitoring with 100-bar rolling windows
- Greenlight signals (VIX <18, DXY >100)
- SPY/QQQ support for equity correlation analysis

### Fixed - Critical Bugs
- VIX parameter passing to mtf_confluence() (vix_now + vix_prev)
- VIXHysteresis.previous_value tracking for proper state memory
- OHLCV column case sensitivity (Close vs close) throughout MTF engine
- None/NaN handling for VIX values with safe defaults
- MacroPulse fire_rate_stats initialization TypeError
- CVD dict/float type mismatch in orderflow

### Changed
- Test suite: 318 passed, 0 failed, 0 errors (45 xfailed documented)
- Determinism validated: 2 independent runs identical (48 signals)
- Backtest parity: 8 trades, -0.4% return, 62.5% win rate

## [1.7.2] - 2025-01-XX

### Added - Universal Asset Adapter Architecture
- Multi-asset support: ETH, SOL, XRP, BTC, SPY
- Asset-specific profiling with automated optimization
- Universal backtesting framework
- Adaptive configuration system
- Cross-asset validation suite

### Added - Professional Repository Organization
- Clean directory structure: `/bin/`, `/scripts/research/`, `/tests/`, `/docs/`
- 5 production entry points in `/bin/`
- Comprehensive institutional-grade documentation
- Test consolidation framework
- Professional CLI interfaces

### Changed
- Root directory cleanup: reduced from 45 to 3 Python files
- Organized experimental code into proper structure
- Enhanced error handling and debugging capabilities

### Added - Production Executables
- `bull_machine_cli.py`: Main CLI interface
- `production_backtest.py`: ETH production backtesting
- `run_adaptive_backtest.py`: Multi-asset system
- `run_institutional_testing.py`: Validation framework
- `run_multi_asset_profiler.py`: Asset profiling system

## [1.7.1] - 2025-01-XX

### Added
- Modular configuration system: context, exits, fusion, liquidity, momentum, risk
- Enhanced configuration management
- Improved system modularity

## [1.7.0] - 2025-01-XX

### Added
- Historical calibration and tuning framework
- Asset-specific configuration templates
- Enhanced backtesting capabilities

## [1.6.2] - 2024-XX-XX

### Added
- Production-ready feature set
- Enhanced stability and reliability

## [1.6.1] - 2024-XX-XX

### Added - Fibonacci Clusters
- Price-time confluence zones
- Overlapping Fib bar counts (21, 34, 55, 89, 144)
- Temporal pressure zone detection
- Integration with Wyckoff Phase C/D

## [1.6.0] - 2024-XX-XX

### Added - 9-Layer Enhanced Confluence
1. Wyckoff structure (accumulation/distribution)
2. M1/M2 Wyckoff (spring/shakeout and markup)
3. Liquidity analysis (OB, FVG, sweeps)
4. Structure (support/resistance, trends)
5. Momentum (RSI, MACD, divergences)
6. Volume (profile and confirmation)
7. Context (regime, volatility)
8. Fibonacci clusters (price-time confluence)
9. MTF synchronization (multi-timeframe bias)

### Added - Oracle Whisper System
- Soul-layer wisdom for high-confluence events
- Price-time symmetry detection
- Premium/discount zone awareness
- Temporal pressure insights
- CVD divergence revelation

### Added - Enhanced CVD & Orderflow
- Cumulative Volume Delta with IamZeroIka slope analysis
- Divergence detection (price vs volume intent)
- Break of Structure (BOS) with 1/3 body validation
- Liquidity Capture Analysis (LCA)
- Intent nudging via volume confirmation

## [1.5.x] - 2024-XX-XX

### Added
- Legacy feature set (backward compatibility maintained)
- Risk profile templates: aggressive, balanced, conservative

---

## Version Naming Convention

- **Major (X.0.0)**: Breaking changes, major architecture shifts
- **Minor (1.X.0)**: New features, non-breaking enhancements
- **Patch (1.0.X)**: Bug fixes, minor improvements

## Links

- [Unreleased]: https://github.com/rayger14/Bull-machine-/compare/v1.8.6...HEAD
- [1.8.6]: https://github.com/rayger14/Bull-machine-/releases/tag/v1.8.6
- [Repository]: https://github.com/rayger14/Bull-machine-
- [Bug Reports]: https://github.com/rayger14/Bull-machine-/issues
- [Documentation]: https://github.com/rayger14/Bull-machine-/tree/main/docs

---

**Note**: Dates marked as "2025-01-XX" or "2024-XX-XX" indicate releases without specific public release dates in the current documentation. This changelog will be updated with precise dates as releases are formalized.
