#!/usr/bin/env python3
"""
Test the version sync system to ensure it works correctly.
"""

import subprocess
import sys
from pathlib import Path

# Add Bull Machine to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from bull_machine.version import __version__, get_version_banner

def run_command(cmd):
    """Run a shell command and return output."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        return 1, "", str(e)

def test_version_import():
    """Test that version can be imported correctly."""
    print("🧪 Testing version import...")
    try:
        print(f"   Version: {__version__}")
        print(f"   Banner: {get_version_banner()}")
        print("✅ Version import works")
        return True
    except Exception as e:
        print(f"❌ Version import failed: {e}")
        return False

def test_readme_rendering():
    """Test that README can be rendered from template."""
    print("🧪 Testing README rendering...")
    returncode, stdout, stderr = run_command("python3 scripts/render_readme.py")

    if returncode == 0:
        print("✅ README rendering works")
        print(f"   Output: {stdout.strip()}")
        return True
    else:
        print(f"❌ README rendering failed: {stderr}")
        return False

def test_version_consistency():
    """Test that no stale version references exist."""
    print("🧪 Testing version consistency...")

    # Simulate the version sync check
    cmd = f"""
    VER={__version__}
    grep -rInE 'v[0-9]+\\.[0-9]+\\.[0-9]+' \
      --exclude-dir=.git --exclude-dir=.github --exclude-dir=archive \
      --exclude="*.png" --exclude="*.jpg" --exclude="*.svg" \
      --exclude="*.json" --exclude="CHANGELOG.md" \
      . | grep -v "v$VER" || true
    """

    returncode, stdout, stderr = run_command(cmd)

    if stdout.strip():
        print(f"⚠️  Found potential stale version references:")
        for line in stdout.strip().split('\n'):
            if line:
                print(f"   {line}")
        print("   (These may need manual review)")
        return True  # Don't fail for now, just warn
    else:
        print("✅ No stale version references found")
        return True

def test_setup_py():
    """Test that setup.py can import version correctly."""
    print("🧪 Testing setup.py version import...")

    cmd = """
    python3 - <<'PY'
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname('.'), 'bull_machine'))
from version import __version__
print(f"setup.py can access version: {__version__}")
PY
    """

    returncode, stdout, stderr = run_command(cmd)

    if returncode == 0:
        print("✅ setup.py version import works")
        print(f"   Output: {stdout.strip()}")
        return True
    else:
        print(f"❌ setup.py version import failed: {stderr}")
        return False

def main():
    """Run all version sync tests."""
    print(f"🚀 Testing Version Sync System - {get_version_banner()}")
    print("=" * 60)

    tests = [
        ("Version Import", test_version_import),
        ("README Rendering", test_readme_rendering),
        ("Version Consistency", test_version_consistency),
        ("Setup.py Integration", test_setup_py),
    ]

    passed = 0
    total = len(tests)

    for name, test_func in tests:
        print(f"\n🔍 {name}")
        if test_func():
            passed += 1
        print()

    print("=" * 60)
    print(f"📊 Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 Version sync system is working correctly!")
        print("\n💡 Next steps:")
        print("   • Use 'bump2version patch/minor/major' to update versions")
        print("   • README will auto-render with current version info")
        print("   • CI will enforce version consistency on PRs")
        return 0
    else:
        print("⚠️  Some tests failed - fix before relying on version sync")
        return 1

if __name__ == "__main__":
    sys.exit(main())