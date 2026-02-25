#!/usr/bin/env bash
# =============================================================================
# setup_coinbase.sh -- First-time setup for Coinbase paper trading on Oracle server
#
# Run from local machine:
#   ./deploy/setup_coinbase.sh
#
# Prerequisites:
#   - Oracle server already provisioned (deploy/setup_oracle.sh completed)
#   - SSH key at ~/.ssh/oracle_bullmachine
#   - Code synced to server (deploy/deploy.sh run at least once)
# =============================================================================
set -euo pipefail

# ---- Configuration ----
SERVER_USER="ubuntu"
SERVER_IP="165.1.79.19"
SSH_KEY="~/.ssh/oracle_bullmachine"
REMOTE_DIR="/home/ubuntu/Bull-machine-"
SSH_CMD="ssh -i ${SSH_KEY} ${SERVER_USER}@${SERVER_IP}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL_DIR="$(dirname "$SCRIPT_DIR")"

echo "==========================================="
echo " Bull Machine -- Coinbase Paper Trading Setup"
echo "==========================================="
echo ""
echo "Server: ${SERVER_USER}@${SERVER_IP}"
echo "Remote: ${REMOTE_DIR}"
echo ""

# ------------------------------------------------------------------
# Step 1: Install Python dependencies in server venv
# ------------------------------------------------------------------
echo "--- Step 1: Installing Python dependencies ---"
${SSH_CMD} << 'REMOTE_INSTALL'
set -euo pipefail
source /home/ubuntu/Bull-machine-/.venv/bin/activate

echo "  Installing coinbase-advanced-py..."
pip install coinbase-advanced-py 2>&1 | tail -3

echo "  Installing requests (if needed)..."
pip install requests 2>&1 | tail -1

echo "  Verifying installations..."
python3 -c "import coinbase; print(f'  coinbase-advanced-py: OK')" 2>/dev/null || echo "  WARNING: coinbase import failed"
python3 -c "import requests; print(f'  requests: OK')" 2>/dev/null || echo "  WARNING: requests import failed"

echo "  Dependencies installed."
REMOTE_INSTALL

# ------------------------------------------------------------------
# Step 2: Copy systemd service file
# ------------------------------------------------------------------
echo ""
echo "--- Step 2: Installing systemd service ---"
scp -i ${SSH_KEY} \
    "${SCRIPT_DIR}/coinbase-paper.service" \
    "${SERVER_USER}@${SERVER_IP}:/tmp/coinbase-paper.service"

${SSH_CMD} << 'REMOTE_SERVICE'
set -euo pipefail
sudo cp /tmp/coinbase-paper.service /etc/systemd/system/coinbase-paper.service
sudo systemctl daemon-reload
sudo systemctl enable coinbase-paper
echo "  Service installed and enabled (not started yet)."
REMOTE_SERVICE

# ------------------------------------------------------------------
# Step 3: Create .env.coinbase from template
# ------------------------------------------------------------------
echo ""
echo "--- Step 3: Creating environment file ---"
scp -i ${SSH_KEY} \
    "${SCRIPT_DIR}/.env.coinbase.template" \
    "${SERVER_USER}@${SERVER_IP}:/tmp/.env.coinbase.template"

${SSH_CMD} << 'REMOTE_ENV'
set -euo pipefail
ENV_FILE="/home/ubuntu/Bull-machine-/.env.coinbase"
if [ -f "$ENV_FILE" ]; then
    echo "  .env.coinbase already exists. Keeping existing file."
    echo "  (Template saved to /tmp/.env.coinbase.template for reference)"
else
    cp /tmp/.env.coinbase.template "$ENV_FILE"
    chmod 600 "$ENV_FILE"
    echo "  .env.coinbase created with placeholder values."
fi
REMOTE_ENV

# ------------------------------------------------------------------
# Step 4: Create results directory
# ------------------------------------------------------------------
echo ""
echo "--- Step 4: Creating results directory ---"
${SSH_CMD} << 'REMOTE_DIRS'
set -euo pipefail
mkdir -p /home/ubuntu/Bull-machine-/results/coinbase_paper
echo "  results/coinbase_paper/ created."
REMOTE_DIRS

# ------------------------------------------------------------------
# Step 5: Add health check cron for coinbase-paper service
# ------------------------------------------------------------------
echo ""
echo "--- Step 5: Adding health check cron ---"
${SSH_CMD} << 'REMOTE_CRON'
set -euo pipefail

# Create health check script for Coinbase service
cat > /home/ubuntu/check_coinbase_health.sh << 'HEALTH'
#!/usr/bin/env bash
SERVICE="coinbase-paper"
LOG="/home/ubuntu/Bull-machine-/logs/coinbase_health_check.log"
mkdir -p "$(dirname "$LOG")"

if ! systemctl is-active --quiet "$SERVICE"; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') - ${SERVICE} DOWN, restarting..." >> "$LOG"
    sudo systemctl restart "$SERVICE"
else
    LAST=$(journalctl -u "$SERVICE" --since "2 hours ago" --no-pager 2>/dev/null | wc -l)
    if [ "$LAST" -lt 5 ]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') - No activity in 2h, restarting..." >> "$LOG"
        sudo systemctl restart "$SERVICE"
    fi
fi
HEALTH
chmod +x /home/ubuntu/check_coinbase_health.sh

# Add cron if not already present
(crontab -l 2>/dev/null || true) | grep -v "check_coinbase_health" | \
    { cat; echo "*/5 * * * * /home/ubuntu/check_coinbase_health.sh"; } | \
    sort -u | crontab -

echo "  Health check cron installed (every 5 minutes)."
REMOTE_CRON

# ------------------------------------------------------------------
# Done
# ------------------------------------------------------------------
echo ""
echo "==========================================="
echo " Coinbase Setup Complete!"
echo "==========================================="
echo ""
echo " The coinbase-paper service is INSTALLED but NOT STARTED."
echo ""
echo " Next steps:"
echo "   1. SSH into the server:"
echo "      ssh -i ~/.ssh/oracle_bullmachine ubuntu@${SERVER_IP}"
echo ""
echo "   2. Edit the Coinbase API credentials:"
echo "      nano /home/ubuntu/Bull-machine-/.env.coinbase"
echo ""
echo "   3. Update these fields:"
echo "      COINBASE_API_KEY=organizations/YOUR_ORG_ID/apiKeys/YOUR_KEY_ID"
echo '      COINBASE_API_SECRET="-----BEGIN EC PRIVATE KEY-----\nYOUR_KEY\n-----END EC PRIVATE KEY-----"'
echo ""
echo "   4. Start the paper trading service:"
echo "      sudo systemctl start coinbase-paper"
echo ""
echo "   5. Check status:"
echo "      sudo systemctl status coinbase-paper"
echo "      sudo journalctl -u coinbase-paper -f"
echo ""
echo " To switch from Freqtrade to Coinbase (stops Freqtrade):"
echo "   ./deploy/deploy.sh --switch-coinbase"
echo ""
