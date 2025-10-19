#!/usr/bin/env bash
set -euo pipefail

echo "================================================================================"
echo "BULL MACHINE REPOSITORY AUDIT - v2.0 Cleanup"
echo "================================================================================"

echo ""
echo "== Local Changes =="
git status -s

echo ""
echo "== Current Branch =="
git branch --show-current

echo ""
echo "== Unmerged to main =="
git branch --no-merged main || true

echo ""
echo "== Merged to main =="
git branch --merged main || true

echo ""
echo "== Duplicate Domain Symbols (Wyckoff) =="
grep -RIn --include="*.py" -E "class.*Wyckoff|def.*wyckoff|WyckoffEngine|WyckoffSignal" engine bin || true

echo ""
echo "== Duplicate Domain Symbols (SMC/HOB/BOMS) =="
grep -RIn --include="*.py" -E "class.*SMC|def.*detect_boms|BOMSDetector|HOB|OrderBlock" engine bin || true

echo ""
echo "== Duplicate Domain Symbols (PTI/Psychology) =="
grep -RIn --include="*.py" -E "class.*PTI|def.*pti|PsychologyTrap|detect_fakeout" engine bin || true

echo ""
echo "== Duplicate Domain Symbols (FRVP/Volume) =="
grep -RIn --include="*.py" -E "class.*FRVP|def.*frvp|VolumeProfile" engine bin || true

echo ""
echo "== Duplicate Domain Symbols (Macro) =="
grep -RIn --include="*.py" -E "class.*Macro|def.*macro_echo|MacroEngine|MacroRegime" engine bin || true

echo ""
echo "== Import Graph (who calls what) =="
echo "Wyckoff imports:"
grep -RIn --include="*.py" "from.*wyckoff|import.*wyckoff" bin engine | cut -d: -f1 | sort -u || true

echo ""
echo "SMC imports:"
grep -RIn --include="*.py" "from.*smc|import.*smc" bin engine | cut -d: -f1 | sort -u || true

echo ""
echo "Fusion imports:"
grep -RIn --include="*.py" "from.*fusion|import.*fusion|knowledge_hooks" bin engine | cut -d: -f1 | sort -u || true

echo ""
echo "================================================================================"
echo "AUDIT COMPLETE"
echo "================================================================================"
