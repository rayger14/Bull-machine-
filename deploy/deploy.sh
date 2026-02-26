#!/usr/bin/env bash
# =============================================================================
# deploy.sh — Sync Bull Machine code to Oracle Cloud server
#
# Usage:
#   ./deploy/deploy.sh                    # Code-only update + restart Coinbase paper + dashboard
#   ./deploy/deploy.sh --full             # First deploy: code + models
#   ./deploy/deploy.sh --coinbase         # Deploy Coinbase stack (install SDK, copy service, restart)
#   ./deploy/deploy.sh --dashboard        # Deploy Flask dashboard on port 8081
# =============================================================================
set -euo pipefail

# ---- Configuration (edit these) ----
SERVER_USER="ubuntu"
SERVER_IP="165.1.79.19"              # Oracle Cloud VM (VM.Standard.E2.1.Micro)
SSH_KEY="~/.ssh/oracle_bullmachine"
REMOTE_DIR="/home/ubuntu/Bull-machine-"

# ---- Paths ----
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL_DIR="$(dirname "$SCRIPT_DIR")"

# ---- Parse arguments ----
MODE="${1:-default}"

echo "=== Bull Machine Deploy ==="
echo "Local:  ${LOCAL_DIR}"
echo "Remote: ${SERVER_USER}@${SERVER_IP}:${REMOTE_DIR}"
echo "Mode:   ${MODE}"
echo ""

# ---- Build React dashboard ----
build_dashboard() {
    echo "--- Building React dashboard ---"
    local DASH_DIR="${LOCAL_DIR}/dashboard"
    if [ ! -d "${DASH_DIR}/node_modules" ]; then
        echo "  Installing npm dependencies..."
        (cd "${DASH_DIR}" && npm install --silent)
    fi
    (cd "${DASH_DIR}" && npm run build)
    echo "  Dashboard built: ${DASH_DIR}/dist/"
}

# ---- Clean server root (remove old docs/scripts/logs from legacy layout) ----
clean_server_root() {
    echo "--- Cleaning server root (removing legacy files) ---"
    ssh -i ${SSH_KEY} ${SERVER_USER}@${SERVER_IP} bash -s <<'CLEAN_EOF'
cd /home/ubuntu/Bull-machine-
# Remove old root-level docs (*.md except CLAUDE.md, README.md)
find . -maxdepth 1 -name '*.md' ! -name 'CLAUDE.md' ! -name 'README.md' ! -name 'MANIFEST.in' -delete 2>/dev/null
# Remove old root-level data files
rm -f *.log *.csv *.json *.txt *.patch *.sh 2>/dev/null
rm -f *.py.bak 2>/dev/null
# Remove old root-level scripts that should be in bin/
find . -maxdepth 1 -name '*.py' ! -name '__init__.py' ! -name 'conftest.py' ! -name 'setup.py' -delete 2>/dev/null
# Remove old dirs that were archived
rm -rf archive/ IMPROVEMENT_PLANS/ profiles/ reports/ results_reference/ \
       schema/ scripts/ telemetry/ tools/ utils/ 2>/dev/null
# Remove dangling symlink
rm -f chart_logs 2>/dev/null
echo "  Server root cleaned"
CLEAN_EOF
}

# ---- Core code sync (shared by all modes) ----
sync_code() {
    echo "--- Syncing code ---"
    rsync -avz --progress \
        -e "ssh -i ${SSH_KEY}" \
        --include='engine/***' \
        --include='configs/***' \
        --include='bin/__init__.py' \
        --include='bin/live/***' \
        --include='bin/backtest_v11_standalone.py' \
        --include='bin/rebuild_wyckoff_features.py' \
        --include='dashboard/dist/***' \
        --include='requirements.txt' \
        --include='setup.py' \
        --include='pyproject.toml' \
        --include='__init__.py' \
        --include='.gitignore' \
        --include='CLAUDE.md' \
        --include='conftest.py' \
        --include='Makefile' \
        --include='README.md' \
        --include='LICENSE' \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        --exclude='.venv*' \
        --exclude='.git/' \
        --exclude='.claude/' \
        --exclude='.github/' \
        --exclude='data/' \
        --exclude='models/' \
        --exclude='results/' \
        --exclude='logs/' \
        --exclude='docs/' \
        --exclude='tests/' \
        --exclude='examples/' \
        --exclude='artifacts/' \
        --exclude='bull_machine/' \
        --exclude='deploy/' \
        --exclude='*.parquet' \
        --exclude='*.feather' \
        --exclude='*.sqlite*' \
        --exclude='*.log' \
        --exclude='node_modules/' \
        --exclude='package-lock.json' \
        --exclude='.pytest_cache/' \
        "${LOCAL_DIR}/" "${SERVER_USER}@${SERVER_IP}:${REMOTE_DIR}/"
}

# ---- Freqtrade config sync (DEPRECATED — Freqtrade removed 2026-02-17) ----
# Kept for reference only. Service file has been deleted from server.
sync_freqtrade_config() {
    echo "WARNING: Freqtrade was removed 2026-02-17. This function is deprecated."
    return 0
}

# ---- Models sync ----
sync_models() {
    echo ""
    echo "--- Syncing models ---"
    rsync -avz --progress \
        -e "ssh -i ${SSH_KEY}" \
        "${LOCAL_DIR}/models/" "${SERVER_USER}@${SERVER_IP}:${REMOTE_DIR}/models/"
}

# ---- Restart Coinbase paper + dashboard ----
restart_services() {
    echo ""
    echo "--- Restarting Coinbase paper + dashboard ---"
    ssh -i "${SSH_KEY}" "${SERVER_USER}@${SERVER_IP}" \
        "sudo systemctl restart coinbase-paper && sudo systemctl restart dashboard"

    sleep 5
    echo ""
    echo "--- Coinbase paper status ---"
    ssh -i "${SSH_KEY}" "${SERVER_USER}@${SERVER_IP}" "sudo systemctl status coinbase-paper --no-pager | head -10"
    echo ""
    echo "--- Dashboard status ---"
    ssh -i "${SSH_KEY}" "${SERVER_USER}@${SERVER_IP}" "sudo systemctl status dashboard --no-pager | head -10"
}

# ---- Deploy Coinbase stack ----
deploy_coinbase() {
    echo ""
    echo "--- Deploying Coinbase stack ---"

    # Install coinbase-advanced-py in server venv
    echo ""
    echo "  Installing coinbase-advanced-py in server venv..."
    ssh -i "${SSH_KEY}" "${SERVER_USER}@${SERVER_IP}" << 'REMOTE_PIP'
set -euo pipefail
source /home/ubuntu/Bull-machine-/.venv/bin/activate
pip install coinbase-advanced-py requests 2>&1 | tail -5
python3 -c "import coinbase; print('  coinbase-advanced-py: OK')" 2>/dev/null || echo "  WARNING: coinbase import failed"
REMOTE_PIP

    # Copy systemd service file
    echo ""
    echo "  Installing coinbase-paper systemd service..."
    scp -i ${SSH_KEY} \
        "${SCRIPT_DIR}/coinbase-paper.service" \
        "${SERVER_USER}@${SERVER_IP}:/tmp/coinbase-paper.service"

    ssh -i "${SSH_KEY}" "${SERVER_USER}@${SERVER_IP}" << 'REMOTE_SVC'
set -euo pipefail
sudo cp /tmp/coinbase-paper.service /etc/systemd/system/coinbase-paper.service
sudo systemctl daemon-reload
sudo systemctl enable coinbase-paper
REMOTE_SVC

    # Create results directory
    ssh -i "${SSH_KEY}" "${SERVER_USER}@${SERVER_IP}" \
        "mkdir -p /home/ubuntu/Bull-machine-/results/coinbase_paper"

    # Create .env.coinbase if it does not exist
    echo ""
    echo "  Checking .env.coinbase..."
    scp -i ${SSH_KEY} \
        "${SCRIPT_DIR}/.env.coinbase.template" \
        "${SERVER_USER}@${SERVER_IP}:/tmp/.env.coinbase.template"

    ssh -i "${SSH_KEY}" "${SERVER_USER}@${SERVER_IP}" << 'REMOTE_ENV'
ENV_FILE="/home/ubuntu/Bull-machine-/.env.coinbase"
if [ -f "$ENV_FILE" ]; then
    echo "  .env.coinbase already exists. Keeping existing credentials."
else
    cp /tmp/.env.coinbase.template "$ENV_FILE"
    chmod 600 "$ENV_FILE"
    echo "  .env.coinbase created from template (edit with your API keys)."
fi
REMOTE_ENV

    # Restart coinbase-paper service
    echo ""
    echo "  Restarting coinbase-paper service..."
    ssh -i "${SSH_KEY}" "${SERVER_USER}@${SERVER_IP}" "sudo systemctl restart coinbase-paper"

    sleep 5
    echo ""
    echo "--- Coinbase service status ---"
    ssh -i "${SSH_KEY}" "${SERVER_USER}@${SERVER_IP}" "sudo systemctl status coinbase-paper --no-pager | head -15"
}

# ---- Switch to Coinbase (DEPRECATED — Freqtrade removed 2026-02-17) ----
# Coinbase paper is now the only trading service. Freqtrade service file deleted.
switch_to_coinbase() {
    echo "NOTE: Freqtrade was removed 2026-02-17. Coinbase paper is the only active service."
    echo "Use './deploy/deploy.sh' for standard deployment."
    return 0
}

# ---- Deploy Dashboard ----
deploy_dashboard() {
    echo ""
    echo "--- Deploying Flask dashboard ---"

    # Install Flask in server venv
    echo ""
    echo "  Installing Flask in server venv..."
    ssh -i "${SSH_KEY}" "${SERVER_USER}@${SERVER_IP}" << 'REMOTE_FLASK'
set -euo pipefail
source /home/ubuntu/Bull-machine-/.venv/bin/activate
pip install flask 2>&1 | tail -5
python3 -c "import flask; print('  Flask ' + flask.__version__ + ': OK')" 2>/dev/null || echo "  WARNING: flask import failed"
REMOTE_FLASK

    # Copy systemd service file
    echo ""
    echo "  Installing dashboard systemd service..."
    scp -i ${SSH_KEY} \
        "${SCRIPT_DIR}/dashboard.service" \
        "${SERVER_USER}@${SERVER_IP}:/tmp/dashboard.service"

    ssh -i "${SSH_KEY}" "${SERVER_USER}@${SERVER_IP}" << 'REMOTE_SVC'
set -euo pipefail
sudo cp /tmp/dashboard.service /etc/systemd/system/dashboard.service
sudo systemctl daemon-reload
sudo systemctl enable dashboard
sudo systemctl restart dashboard
REMOTE_SVC

    # Open port 8081 in iptables if not already open
    echo ""
    echo "  Ensuring port 8081 is open in iptables..."
    ssh -i "${SSH_KEY}" "${SERVER_USER}@${SERVER_IP}" << 'REMOTE_FW'
set -euo pipefail
if sudo iptables -C INPUT -p tcp --dport 8081 -j ACCEPT 2>/dev/null; then
    echo "  Port 8081 already open."
else
    sudo iptables -I INPUT -p tcp --dport 8081 -j ACCEPT
    echo "  Port 8081 opened in iptables."
fi
REMOTE_FW

    sleep 5
    echo ""
    echo "--- Dashboard service status ---"
    ssh -i "${SSH_KEY}" "${SERVER_USER}@${SERVER_IP}" "sudo systemctl status dashboard --no-pager | head -15"

    echo ""
    echo "  Dashboard URL: http://${SERVER_IP}:8081"
}

# =============================================================================
# Main dispatch
# =============================================================================
case "$MODE" in
    --coinbase)
        sync_code
        deploy_coinbase
        ;;
    --dashboard)
        sync_code
        deploy_dashboard
        ;;
    --full)
        build_dashboard
        sync_code
        sync_models
        restart_services
        ;;
    default|"")
        build_dashboard
        clean_server_root
        sync_code
        restart_services
        ;;
    *)
        echo "Unknown option: ${MODE}"
        echo ""
        echo "Usage:"
        echo "  ./deploy/deploy.sh                    # Code sync + restart Coinbase + dashboard"
        echo "  ./deploy/deploy.sh --full             # Full deploy with models"
        echo "  ./deploy/deploy.sh --coinbase         # Deploy Coinbase stack (first-time setup)"
        echo "  ./deploy/deploy.sh --dashboard        # Deploy Flask dashboard on port 8081"
        exit 1
        ;;
esac

echo ""
echo "=== Deploy complete ==="
