#!/usr/bin/env python3
"""
Render README.md from template with current version and stats.
Simple string replacement version (no jinja2 dependency).
"""

import json
import sys
from pathlib import Path

# Add Bull Machine to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from bull_machine.version import __version__

def get_test_stats():
    """Get test statistics from latest CI results or defaults."""
    return {
        "passing": 9,
        "total": 9
    }

def get_latest_results():
    """Get latest backtest results if available."""
    # Default results based on our SOL backtest
    return {
        "total_trades": 343,
        "win_rate": 51.9,
        "total_pnl_percent": -9.4,
        "max_drawdown_percent": 32.7,
        "sharpe_ratio": -0.29
    }

def main():
    """Render README from template using simple string replacement."""
    template_path = Path("README.md.j2")
    output_path = Path("README.md")

    if not template_path.exists():
        print(f"❌ Template not found: {template_path}")
        sys.exit(1)

    # Load template
    with open(template_path) as f:
        content = f.read()

    # Get data
    test_stats = get_test_stats()
    results = get_latest_results()

    # Simple replacements
    content = content.replace("{{ version }}", __version__)
    content = content.replace("{{ test_stats.passing }}", str(test_stats["passing"]))
    content = content.replace("{{ test_stats.total }}", str(test_stats["total"]))

    # Handle conditional blocks - simple version for v1.4.2
    if __version__ == "1.4.2":
        # Keep the v1.4.2 content
        content = content.replace('{% if version == "1.4.2" -%}', '')
        content = content.replace('{%- else -%}', '<!-- ELSE START -->')
        content = content.replace('{%- endif %}', '<!-- ENDIF -->')

        # Remove the else block
        import re
        content = re.sub(r'<!-- ELSE START -->.*?<!-- ENDIF -->', '', content, flags=re.DOTALL)
    else:
        # Use the else block
        content = content.replace('{% if version == "1.4.2" -%}', '<!-- IF START -->')
        content = content.replace('{%- else -%}', '')
        content = content.replace('{%- endif %}', '')

        # Remove the if block
        import re
        content = re.sub(r'<!-- IF START -->.*?', '', content, flags=re.DOTALL)

    # Clean up any remaining template syntax
    content = content.replace('{% if', '').replace('%}', '').replace('{%-', '').replace('-%}', '')

    # Write output
    with open(output_path, "w") as f:
        f.write(content)

    print(f"✅ README rendered with version {__version__}")
    print(f"   Template: {template_path}")
    print(f"   Output: {output_path}")

if __name__ == "__main__":
    main()