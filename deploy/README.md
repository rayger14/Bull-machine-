# Bull Machine — Oracle Cloud Deployment

Deploy the Freqtrade trading bot to Oracle Cloud Always Free tier for 24/7 paper trading.

## What You Get

- Bot runs 24/7 on a free cloud server (no cost, ever)
- FreqUI web dashboard accessible from any browser
- Auto-restarts on crash (systemd watchdog)
- Health monitoring with automatic recovery
- One-command code deployments from your Mac

## Prerequisites

- Oracle Cloud account (free: https://cloud.oracle.com)
- SSH key pair on your Mac

## Step 1: Create Oracle Cloud Account

1. Go to https://cloud.oracle.com and click **Start for Free**
2. Create account (credit card required but never charged for Always Free resources)
3. Choose a Home Region close to you (e.g., US East, Frankfurt, Tokyo)

## Step 2: Generate SSH Key (on your Mac)

```bash
ssh-keygen -t ed25519 -f ~/.ssh/oracle_bullmachine -C "bull-machine"
```

This creates two files:
- `~/.ssh/oracle_bullmachine` (private key — keep secret)
- `~/.ssh/oracle_bullmachine.pub` (public key — upload to Oracle)

## Step 3: Create the VM Instance

1. In OCI Console: **Compute → Instances → Create Instance**
2. Configure:
   - **Name:** `bull-machine-bot`
   - **Image:** Click Change Image → Ubuntu → **22.04 Minimal** → **aarch64**
   - **Shape:** Click Change Shape → Ampere → **VM.Standard.A1.Flex**
     - OCPUs: **2**
     - Memory: **12 GB**
   - **Boot volume:** 50 GB
   - **SSH keys:** Upload `~/.ssh/oracle_bullmachine.pub`
   - **Networking:** Assign public IPv4 address (checked)
3. Click **Create** — wait 2-5 minutes
4. Note the **Public IP Address**

## Step 4: Open Port 8080 for FreqUI

1. In OCI Console: **Networking → Virtual Cloud Networks → your VCN**
2. Click **Security Lists → Default Security List**
3. **Add Ingress Rule:**
   - Source CIDR: `YOUR_HOME_IP/32` (find your IP at https://whatismyip.com)
   - Protocol: TCP
   - Destination Port: `8080`

## Step 5: SSH In and Run Setup

```bash
# SSH into your new server
ssh -i ~/.ssh/oracle_bullmachine ubuntu@YOUR_SERVER_IP

# Download and run the setup script
cd /home/ubuntu
git clone https://github.com/rayger14/Bull-machine-.git
cd Bull-machine-
bash deploy/setup_oracle.sh
```

This takes about 15-20 minutes. It installs everything automatically.

## Step 6: Configure API Keys

```bash
# Edit the config
nano /home/ubuntu/Bull-machine-/user_data/freqtrade_config.json
```

Update these fields:
- `"key"`: Your Kraken API key
- `"secret"`: Your Kraken API secret
- `"listen_ip_address"`: `"0.0.0.0"` (for remote FreqUI)
- `"jwt_secret_key"`: Generate with: `python3 -c "import secrets; print(secrets.token_hex(32))"`
- `"password"`: A strong password for FreqUI login

Or copy the server template:
```bash
cp deploy/freqtrade_config_server.json user_data/freqtrade_config.json
nano user_data/freqtrade_config.json   # fill in your keys
```

## Step 7: Start the Bot

```bash
sudo systemctl start freqtrade
```

## Step 8: Verify

```bash
# Check service status
sudo systemctl status freqtrade

# Watch live logs
sudo journalctl -u freqtrade -f

# Test API
curl http://localhost:8080/api/v1/ping
```

## Step 9: Access FreqUI

Open in your browser: `http://YOUR_SERVER_IP:8080`

Login:
- Username: `freqtrade`
- Password: (whatever you set in step 6)

## Deploying Code Changes

After making changes locally (backtesting, tuning thresholds, etc.):

```bash
# From your Mac — edit deploy.sh first to set YOUR_SERVER_IP
./deploy/deploy.sh          # Sync code + restart bot
./deploy/deploy.sh --full   # First time: also sync models
./deploy/deploy.sh --config # Also overwrite server config (careful!)
```

## Useful Commands (on the server)

```bash
# Bot management
sudo systemctl start freqtrade     # Start
sudo systemctl stop freqtrade      # Stop
sudo systemctl restart freqtrade   # Restart
sudo systemctl status freqtrade    # Status

# Logs
sudo journalctl -u freqtrade -f              # Live logs
sudo journalctl -u freqtrade --since "1h"    # Last hour
sudo journalctl -u freqtrade --since today   # Today

# Health
cat /home/ubuntu/Bull-machine-/logs/health_check.log  # Health check log
htop                                                    # System resources
```

## Architecture

```
Oracle Cloud VM (ARM Ampere A1, 2 OCPU / 12 GB RAM)
├── systemd: freqtrade.service (auto-restart, watchdog)
├── Freqtrade 2026.1 (dry-run mode)
│   ├── Kraken API → live BTC/USDT 1H candles
│   ├── LiveFeatureComputer → 110 features per candle
│   ├── IsolatedArchetypeEngine → 4 active archetypes
│   │   ├── trap_within_trend
│   │   ├── wick_trap (risk_on only)
│   │   ├── liquidity_compression
│   │   └── liquidity_sweep (risk_on only)
│   └── Portfolio allocator → max 3 simultaneous trades
├── FreqUI → web dashboard on port 8080
├── Cron: health check (every 5 min)
└── Cron: keep-alive (every 6 hours, prevents VM reclamation)
```

## Cost

$0. Oracle Cloud Always Free tier. No charges ever for this VM shape.
