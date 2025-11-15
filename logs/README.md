# Logs Directory

Runtime and debug logs.

## Structure
- `paper_trading/` - Paper trading execution logs
- `archive/` - Historical logs

## Guidelines
- All logs are gitignored by default
- Only critical logs should be committed
- Archive logs after 30 days
- Use log rotation for long-running processes

## Gitignored
All logs in this directory are gitignored except for structure.
See `.gitignore` for details.
