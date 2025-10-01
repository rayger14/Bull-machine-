"""
Bull Machine Production Configuration
Handles paths, imports, and environment setup
"""

import os
import sys
from pathlib import Path

# Get project root directory
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
CONFIG_DIR = PROJECT_ROOT / "configs"
RESULTS_DIR = PROJECT_ROOT / "results"

# Add project root to Python path for imports
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Environment-specific paths
DEFAULT_PATHS = {
    "eth_config": CONFIG_DIR / "v170" / "assets" / "ETH_v17_tuned.json",
    "btc_config": CONFIG_DIR / "adaptive" / "COINBASE_BTCUSD_config.json",
    "data_dir": DATA_DIR,
    "results_dir": RESULTS_DIR,
    "logs_dir": PROJECT_ROOT / "logs"
}

def get_config_path(asset="ETH"):
    """Get configuration path for specific asset"""
    asset_configs = {
        "ETH": DEFAULT_PATHS["eth_config"],
        "BTC": DEFAULT_PATHS["btc_config"],
    }

    config_path = asset_configs.get(asset.upper(), DEFAULT_PATHS["eth_config"])

    if not config_path.exists():
        # Fallback to ETH config
        config_path = DEFAULT_PATHS["eth_config"]
        print(f"⚠️  Config for {asset} not found, using ETH config")

    return config_path

def ensure_directories():
    """Create necessary directories if they don't exist"""
    for directory in [DATA_DIR, RESULTS_DIR, PROJECT_ROOT / "logs"]:
        directory.mkdir(parents=True, exist_ok=True)

def get_data_path(asset, timeframe="4h"):
    """Get data file path for specific asset and timeframe"""
    timeframe_mapping = {
        "1h": "60",
        "4h": "240",
        "1d": "1D"
    }

    tf_code = timeframe_mapping.get(timeframe, "240")
    data_file = DATA_DIR / asset.lower() / f"COINBASE_{asset.upper()}USD, {tf_code}_*.csv"

    # Find the actual file (handles different suffixes)
    import glob
    matches = glob.glob(str(data_file))

    if matches:
        return Path(matches[0])
    else:
        raise FileNotFoundError(f"No data file found for {asset} {timeframe}")

# Initialize on import
ensure_directories()