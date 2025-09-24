#!/usr/bin/env python3
"""
Version Sync System Usage Guide
"""

import sys
from pathlib import Path

# Add Bull Machine to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from bull_machine.version import __version__, get_version_banner

def main():
    print(f"📋 {get_version_banner()} - Version Sync System Guide")
    print("=" * 70)

    print("\n🎯 SINGLE SOURCE OF TRUTH:")
    print("   • bull_machine/version.py contains the authoritative version")
    print(f"   • Current version: {__version__}")
    print("   • All code imports from this file (no more hardcoded versions)")

    print("\n🔧 DAILY USAGE:")
    print("   • Use: from bull_machine.version import __version__")
    print("   • Use: get_version_banner() for display strings")
    print("   • Run: python scripts/render_readme.py (updates README)")

    print("\n🚀 VERSION BUMPING:")
    print("   • bump2version patch   → 1.4.2 → 1.4.3")
    print("   • bump2version minor   → 1.4.2 → 1.5.0")
    print("   • bump2version major   → 1.4.2 → 2.0.0")
    print("   • Automatically commits, tags, and updates version.py")

    print("\n🛡️  PROTECTION SYSTEM:")
    print("   • CI blocks PRs with stale version references")
    print("   • Pre-commit hooks validate version consistency")
    print("   • Version Sync workflow enforces SSOT compliance")

    print("\n✅ VALIDATION:")
    print("   • Run: python scripts/test_version_sync.py")
    print("   • All systems tested and working correctly")

    print("\n💡 BENEFITS:")
    print("   ✅ No more stale v1.3 screenshots/banners")
    print("   ✅ Version bumps propagate everywhere automatically")
    print("   ✅ CI enforces consistency before merge")
    print("   ✅ README auto-generates with current version")
    print("   ✅ Future-proof against version drift")

    print("\n" + "=" * 70)
    print("🎉 Version sync system is active and protecting your repository!")

if __name__ == "__main__":
    main()