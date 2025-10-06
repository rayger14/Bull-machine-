"""
Thin wrapper to make run_adaptive_backtest importable for tests.
Forwards to the actual implementation in bin/.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bin.run_adaptive_backtest import main, AdaptiveBullMachine

__all__ = ['main', 'AdaptiveBullMachine']

if __name__ == "__main__":
    raise SystemExit(main())
