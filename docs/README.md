# Bull Machine Documentation

Comprehensive documentation for the Bull Machine algorithmic trading system.

## Directory Structure

### [Technical Documentation](./technical/)
Architecture, system design, feature pipelines, and implementation details.

**Key Documents:**
- Architecture reviews and design specifications
- Feature pipeline audit and store design
- Funding rates, exits knowledge, and trading mechanics
- Migration guides and institutional structure
- Performance optimization (Numba, vectorization)
- Testing methodology and usage guides

### [Guides](./guides/)
Practical guides for optimization, validation, and backtesting.

**Key Documents:**
- Optimization framework and auto-bounds
- Backtest guide (v1.4)
- Validation checklist

### [Audits](./audits/)
System audits, deployment reviews, and compliance documentation.

**Key Documents:**
- Branch audit
- Production deployment review

### [Reports](./reports/)
Session reports, cleanup summaries, and analysis results organized by date and topic.

### [Backtests](./backtests/)
Backtest results, configurations, and analysis.

### [Releases](./releases/)
Release notes and version history.

### [Archive](./archive/)
Historical documentation organized by date and topic.

### [Analysis](./analysis/)
Analysis scripts, notebooks, and research findings.

## Quick Links

- **Getting Started**: See [USAGE.md](./technical/USAGE.md)
- **System Architecture**: See [BULL_MACHINE_V2_PIPELINE.md](./technical/BULL_MACHINE_V2_PIPELINE.md)
- **Testing**: See [TESTING_METHODOLOGY.md](./technical/TESTING_METHODOLOGY.md)
- **Optimization**: See [OPTIMIZATION_GUIDE.md](./guides/OPTIMIZATION_GUIDE.md)
- **Feature Store**: See [FEATURE_STORE_DESIGN.md](./technical/FEATURE_STORE_DESIGN.md)

## Documentation Standards

All documentation follows these standards:
- Technical specs go in `technical/`
- How-to guides go in `guides/`
- Audit reports go in `audits/`
- Session reports go in `reports/`
- Historical docs go in `archive/`

## Contributing

When adding new documentation:
1. Place in appropriate subdirectory
2. Use descriptive filenames with proper capitalization
3. Update relevant README.md files
4. Keep docs/ root directory clean (only subdirectories and this README)
