#!/usr/bin/env bash
# =============================================================================
# setup_oracle.sh — One-time setup for Bull Machine on Oracle Cloud
#
# Supports both:
#   - VM.Standard.A1.Flex (ARM, 2+ OCPU, 6+ GB RAM)
#   - VM.Standard.E2.1.Micro (AMD x86, 1 OCPU, 1 GB RAM + 2GB swap)
#
# Usage:
#   ssh -i ~/.ssh/oracle_bullmachine ubuntu@<SERVER_IP>
#   cd /home/ubuntu && git clone https://github.com/rayger14/Bull-machine-.git
#   cd Bull-machine- && bash deploy/setup_oracle.sh
# =============================================================================
set -euo pipefail

REPO_URL="https://github.com/rayger14/Bull-machine-.git"
INSTALL_DIR="/home/ubuntu/Bull-machine-"
VENV_DIR="${INSTALL_DIR}/.venv"

ARCH=$(uname -m)  # x86_64 or aarch64

echo "=========================================="
echo " Bull Machine Server Setup"
echo " Architecture: ${ARCH}"
echo " RAM: $(free -m | awk '/Mem:/{print $2}') MB"
echo "=========================================="

# ------------------------------------------------------------------
# Phase 0: Swap (critical for 1GB RAM VMs)
# ------------------------------------------------------------------
TOTAL_RAM=$(free -m | awk '/Mem:/{print $2}')
if [ "$TOTAL_RAM" -lt 2000 ]; then
    echo ""
    echo ">>> Phase 0: Creating 2GB swap (low RAM detected)..."
    if [ ! -f /swapfile ]; then
        sudo fallocate -l 2G /swapfile
        sudo chmod 600 /swapfile
        sudo mkswap /swapfile
        sudo swapon /swapfile
        echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
        echo "  2GB swap created and enabled."
    else
        echo "  Swap already exists."
    fi
    # Tune swappiness for low-RAM server
    echo 'vm.swappiness=60' | sudo tee -a /etc/sysctl.conf
    sudo sysctl vm.swappiness=60
else
    echo ""
    echo ">>> Phase 0: Swap not needed (${TOTAL_RAM} MB RAM)."
fi

# ------------------------------------------------------------------
# Phase 1: System packages
# ------------------------------------------------------------------
echo ""
echo ">>> Phase 1: Installing system packages..."

sudo apt-get update
sudo apt-get upgrade -y
sudo apt-get install -y \
    software-properties-common

# Python 3.11 (deadsnakes PPA for Ubuntu 22.04)
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt-get update
sudo apt-get install -y \
    python3.11 python3.11-venv python3.11-dev

# Build tools and utilities
sudo apt-get install -y \
    build-essential gcc g++ make cmake autoconf automake \
    libffi-dev libssl-dev \
    git curl wget htop tmux jq \
    fail2ban \
    unattended-upgrades

# Enable automatic security updates
sudo dpkg-reconfigure -plow unattended-upgrades

echo "  System packages installed."

# ------------------------------------------------------------------
# Phase 2: TA-Lib C library
# ------------------------------------------------------------------
echo ""
echo ">>> Phase 2: Installing TA-Lib C library..."

cd /tmp

if [ "$ARCH" = "aarch64" ]; then
    DEB_URL="https://github.com/TA-Lib/ta-lib/releases/download/v0.6.4/ta-lib_0.6.4_arm64.deb"
else
    DEB_URL="https://github.com/TA-Lib/ta-lib/releases/download/v0.6.4/ta-lib_0.6.4_amd64.deb"
fi

if wget -q "$DEB_URL" -O ta-lib.deb 2>/dev/null; then
    sudo dpkg -i ta-lib.deb && echo "  TA-Lib installed from pre-built DEB." || {
        echo "  Pre-built DEB failed. Compiling from source..."
        wget -q https://github.com/TA-Lib/ta-lib/releases/download/v0.6.4/ta-lib-0.6.4-src.tar.gz
        tar xzf ta-lib-0.6.4-src.tar.gz
        cd ta-lib-0.6.4
        ./configure --prefix=/usr/local
        make -j"$(nproc)"
        sudo make install
        sudo ldconfig
        echo "  TA-Lib compiled and installed from source."
    }
else
    echo "  Could not download pre-built DEB. Compiling from source..."
    wget -q https://github.com/TA-Lib/ta-lib/releases/download/v0.6.4/ta-lib-0.6.4-src.tar.gz
    tar xzf ta-lib-0.6.4-src.tar.gz
    cd ta-lib-0.6.4
    ./configure --prefix=/usr/local
    make -j"$(nproc)"
    sudo make install
    sudo ldconfig
    echo "  TA-Lib compiled and installed from source."
fi

# ------------------------------------------------------------------
# Phase 3: Clone repository
# ------------------------------------------------------------------
echo ""
echo ">>> Phase 3: Setting up repository..."

cd /home/ubuntu
if [ -d "${INSTALL_DIR}" ]; then
    echo "  Directory exists, pulling latest..."
    cd "${INSTALL_DIR}" && git pull
else
    git clone "${REPO_URL}" "$(basename "${INSTALL_DIR}")"
fi
cd "${INSTALL_DIR}"

# Create directories the runtime needs
mkdir -p models data/features_mtf logs user_data/data user_data/logs

echo "  Repository ready."

# ------------------------------------------------------------------
# Phase 4: Python virtual environment
# ------------------------------------------------------------------
echo ""
echo ">>> Phase 4: Setting up Python environment..."

python3.11 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

pip install --upgrade pip wheel setuptools

# Install Freqtrade
pip install freqtrade

# Install Bull Machine dependencies (skip numba on 1GB — too much memory)
if [ "$TOTAL_RAM" -lt 2000 ]; then
    pip install \
        numpy \
        pandas \
        scipy \
        scikit-learn \
        structlog \
        PyYAML \
        python-dotenv \
        TA-Lib
    echo "  NOTE: numba skipped (1GB RAM). Pure numpy fallback will be used."
else
    pip install \
        numba \
        numpy \
        pandas \
        scipy \
        scikit-learn \
        structlog \
        PyYAML \
        python-dotenv \
        TA-Lib
fi

echo "  Python environment ready."

# ------------------------------------------------------------------
# Phase 5: Install FreqUI
# ------------------------------------------------------------------
echo ""
echo ">>> Phase 5: Installing FreqUI web dashboard..."

"${VENV_DIR}/bin/freqtrade" install-ui

echo "  FreqUI installed."

# ------------------------------------------------------------------
# Phase 6: Systemd service
# ------------------------------------------------------------------
echo ""
echo ">>> Phase 6: Setting up systemd service..."

# Adjust MemoryMax for small VMs
if [ "$TOTAL_RAM" -lt 2000 ]; then
    sed 's/MemoryMax=8G/MemoryMax=800M/' "${INSTALL_DIR}/deploy/freqtrade.service" | \
        sudo tee /etc/systemd/system/freqtrade.service > /dev/null
else
    sudo cp "${INSTALL_DIR}/deploy/freqtrade.service" /etc/systemd/system/freqtrade.service
fi

sudo systemctl daemon-reload
sudo systemctl enable freqtrade

echo "  Systemd service installed (not started yet — configure API keys first)."

# ------------------------------------------------------------------
# Phase 7: Firewall (open port 8080 for FreqUI)
# ------------------------------------------------------------------
echo ""
echo ">>> Phase 7: Configuring firewall..."

sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 8080 -j ACCEPT
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y iptables-persistent
sudo netfilter-persistent save

echo "  Port 8080 opened."

# ------------------------------------------------------------------
# Phase 8: Log rotation
# ------------------------------------------------------------------
echo ""
echo ">>> Phase 8: Setting up log rotation..."

sudo tee /etc/logrotate.d/freqtrade > /dev/null << 'LOGROTATE'
/home/ubuntu/Bull-machine-/logs/*.log
/home/ubuntu/Bull-machine-/user_data/logs/*.log {
    daily
    missingok
    rotate 14
    compress
    delaycompress
    notifempty
    copytruncate
    maxsize 100M
}
LOGROTATE

# Limit journald size (smaller for low-RAM VMs)
sudo mkdir -p /etc/systemd/journald.conf.d
if [ "$TOTAL_RAM" -lt 2000 ]; then
    JOURNAL_SIZE="200M"
else
    JOURNAL_SIZE="500M"
fi
sudo tee /etc/systemd/journald.conf.d/size.conf > /dev/null << JOURNAL
[Journal]
SystemMaxUse=${JOURNAL_SIZE}
MaxRetentionSec=14day
JOURNAL
sudo systemctl restart systemd-journald

echo "  Log rotation configured."

# ------------------------------------------------------------------
# Phase 9: Oracle keep-alive cron (prevent VM reclamation)
# ------------------------------------------------------------------
echo ""
echo ">>> Phase 9: Setting up keep-alive and health check crons..."

# Keep-alive: CPU burst every 6 hours (lighter for E2.Micro)
if [ "$TOTAL_RAM" -lt 2000 ]; then
    KEEPALIVE="0 */6 * * * /usr/bin/timeout 60 /usr/bin/dd if=/dev/urandom of=/dev/null bs=1M count=256 2>/dev/null"
else
    KEEPALIVE="0 */6 * * * /usr/bin/timeout 180 /usr/bin/dd if=/dev/urandom of=/dev/null bs=1M count=1024 2>/dev/null"
fi
(crontab -l 2>/dev/null || true; echo "$KEEPALIVE") | sort -u | crontab -

# Health check script
cat > /home/ubuntu/check_bot_health.sh << 'HEALTH'
#!/usr/bin/env bash
SERVICE="freqtrade"
LOG="/home/ubuntu/Bull-machine-/logs/health_check.log"
mkdir -p "$(dirname "$LOG")"

if ! systemctl is-active --quiet "$SERVICE"; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') - ${SERVICE} DOWN, restarting..." >> "$LOG"
    sudo systemctl restart "$SERVICE"
else
    LAST=$(journalctl -u freqtrade --since "2 hours ago" --no-pager 2>/dev/null | wc -l)
    if [ "$LAST" -lt 5 ]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') - No activity in 2h, restarting..." >> "$LOG"
        sudo systemctl restart "$SERVICE"
    fi
fi
HEALTH
chmod +x /home/ubuntu/check_bot_health.sh

# Health check every 5 minutes
(crontab -l 2>/dev/null || true; echo "*/5 * * * * /home/ubuntu/check_bot_health.sh") | sort -u | crontab -

echo "  Crons configured."

# ------------------------------------------------------------------
# Phase 10: SSH hardening
# ------------------------------------------------------------------
echo ""
echo ">>> Phase 10: Hardening SSH..."

sudo sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config
sudo sed -i 's/PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config
sudo sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin no/' /etc/ssh/sshd_config
sudo systemctl restart sshd

echo "  SSH hardened (key-only, no root login)."

# ------------------------------------------------------------------
# Done
# ------------------------------------------------------------------
echo ""
echo "=========================================="
echo " Setup Complete!"
echo "=========================================="
echo ""
echo " Architecture: ${ARCH}"
echo " RAM: ${TOTAL_RAM} MB $([ -f /swapfile ] && echo '+ 2GB swap')"
echo ""
echo " Next steps:"
echo "   1. Edit the config with your exchange API keys:"
echo "      nano ${INSTALL_DIR}/user_data/freqtrade_config.json"
echo ""
echo "   2. Update these fields in the config:"
echo '      - "key": "YOUR_KRAKEN_API_KEY"'
echo '      - "secret": "YOUR_KRAKEN_SECRET"'
echo '      - "jwt_secret_key": run: python3 -c "import secrets; print(secrets.token_hex(32))"'
echo '      - "password": "YOUR_STRONG_PASSWORD"'
echo ""
echo "   3. Start the bot:"
echo "      sudo systemctl start freqtrade"
echo ""
echo "   4. Check status:"
echo "      sudo systemctl status freqtrade"
echo "      sudo journalctl -u freqtrade -f"
echo ""
echo "   5. Access FreqUI:"
echo "      http://<YOUR_SERVER_IP>:8080"
echo ""
