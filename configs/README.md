# Configurations

Bull Machine configuration files organized by purpose.

## Structure
- `production/` - Production configurations
  - `frozen/` - Frozen production configs
  - `live/` - Live trading configs
  - `paper_trading/` - Paper trading configs
  - `mvp_*.json` - MVP configurations
- `experimental/` - Experimental configurations
  - `adaptive/` - Adaptive strategy experiments
  - `sweep/` - Parameter sweep configs
  - `knowledge_v2/` - Knowledge v2 experiments
- `archive/` - Historical version configs
- `schema/` - Configuration schemas
- `deltas/` - Configuration deltas
- `stock/` - Stock market configs

## Guidelines
- Production configs in `production/`
- Experiments in `experimental/`
- Archive old versions to `archive/`
- Never modify frozen configs
- Use deltas for variations
