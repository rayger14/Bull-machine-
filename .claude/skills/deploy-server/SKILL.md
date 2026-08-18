---
name: deploy-server
description: Deploy Bull Machine to the Oracle server (165.1.79.19) — SSH access, deploy.sh, service names, log monitoring. Use for any deploy, server restart, or live-service check.
---

# Deploying to the Oracle server

**Standing rule: never run `deploy.sh` or change server state without an explicit "go" from the user in the current exchange. Prior approvals do not carry forward.** Never patch files on the server directly — commit first, then deploy.

- **Server**: `165.1.79.19`
- **SSH**: `ssh -i ~/.ssh/oracle_bullmachine ubuntu@165.1.79.19`
- **Deploy**: `./deploy/deploy.sh` (builds dashboard, syncs code, restarts services)
- **Services**: `coinbase-paper` (800MB memory limit) + `dashboard` (port 8081, 200MB limit)
- **Monitor**: `sudo journalctl -u coinbase-paper -f`

Checklist before deploying:
1. All changes committed and pushed (never deploy uncommitted work).
2. Backtest floors still hold (see Testing Checklist in CLAUDE.md).
3. Explicit user approval obtained in this exchange.
