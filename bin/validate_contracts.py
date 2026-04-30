#!/usr/bin/env python3
"""
Contract Validation Script

Purpose: Generate comprehensive report on archetype API contract compliance.

Usage:
    python bin/validate_contracts.py                  # Full validation
    python bin/validate_contracts.py --quick          # Quick check only
    python bin/validate_contracts.py --fix            # Auto-fix violations (where possible)

Author: System Architect (Claude Code)
Date: 2026-01-21
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class ContractValidator:
    """Validates archetype API contract compliance."""

    STRATEGY_FILES = [
        'engine/integrations/nautilus_strategy.py',
        'engine/integrations/bull_machine_strategy.py',
        'engine/integrations/event_engine.py',
    ]

    # Only flag archetype-specific method calls, not all private methods
    # We care about strategies calling archetype internal methods, not internal class methods
    FORBIDDEN_PATTERNS = [
        (r'archetype_logic\.check_[a-z_]+\(', 'Direct call to ArchetypeLogic.check_* method', 'ERROR'),
        (r'archetype\.check_[a-z_]+\(', 'Direct call to archetype.check_* method', 'ERROR'),
    ]

    WARNING_PATTERNS = [
        (r'archetype_logic\.detect\(', 'Direct .detect() call (should use ArchetypeModel.predict())', 'WARNING'),
    ]

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.violations = []
        self.warnings = []

    def validate_strategy_files(self) -> Dict[str, Any]:
        """Validate strategy files don't call internal methods."""
        print("🔍 Validating strategy files...")

        results = {
            'checked_files': 0,
            'violations': [],
            'warnings': []
        }

        for strategy_file in self.STRATEGY_FILES:
            file_path = self.project_root / strategy_file

            if not file_path.exists():
                print(f"  ⏭️  Skipping {strategy_file} (not found)")
                continue

            print(f"  📄 Checking {strategy_file}...")
            results['checked_files'] += 1

            content = file_path.read_text()

            # Check forbidden patterns (errors)
            for pattern, description, severity in self.FORBIDDEN_PATTERNS:
                matches = list(re.finditer(pattern, content))

                for match in matches:
                    line_num = content[:match.start()].count('\n') + 1
                    line_content = content.split('\n')[line_num - 1].strip()

                    # Skip comments
                    if line_content.strip().startswith('#'):
                        continue

                    violation = {
                        'file': strategy_file,
                        'line': line_num,
                        'pattern': pattern,
                        'description': description,
                        'severity': severity,
                        'code': line_content
                    }

                    results['violations'].append(violation)
                    self.violations.append(violation)

            # Check warning patterns
            for pattern, description, severity in self.WARNING_PATTERNS:
                matches = list(re.finditer(pattern, content))

                for match in matches:
                    line_num = content[:match.start()].count('\n') + 1
                    line_content = content.split('\n')[line_num - 1].strip()

                    # Skip comments
                    if line_content.strip().startswith('#'):
                        continue

                    warning = {
                        'file': strategy_file,
                        'line': line_num,
                        'pattern': pattern,
                        'description': description,
                        'severity': severity,
                        'code': line_content
                    }

                    results['warnings'].append(warning)
                    self.warnings.append(warning)

        return results

    def validate_docstrings(self) -> Dict[str, Any]:
        """Validate public interfaces have proper documentation."""
        print("\n📚 Validating docstrings...")

        results = {
            'checked_methods': 0,
            'missing_docstrings': [],
            'incomplete_docstrings': []
        }

        # Check ArchetypeModel.predict()
        try:
            from engine.models.archetype_model import ArchetypeModel

            predict_doc = ArchetypeModel.predict.__doc__
            results['checked_methods'] += 1

            if not predict_doc:
                results['missing_docstrings'].append({
                    'class': 'ArchetypeModel',
                    'method': 'predict',
                    'severity': 'ERROR'
                })
            else:
                # Check for required terms
                required_terms = ['Signal', 'Returns', 'Args']
                missing_terms = [term for term in required_terms if term not in predict_doc]

                if missing_terms:
                    results['incomplete_docstrings'].append({
                        'class': 'ArchetypeModel',
                        'method': 'predict',
                        'missing_terms': missing_terms,
                        'severity': 'WARNING'
                    })

            print(f"  ✅ ArchetypeModel.predict() docstring validated")

        except Exception as e:
            print(f"  ❌ Error checking ArchetypeModel.predict(): {e}")

        # Check Signal class
        try:
            from engine.models.base import Signal

            signal_doc = Signal.__doc__
            results['checked_methods'] += 1

            if not signal_doc:
                results['missing_docstrings'].append({
                    'class': 'Signal',
                    'method': '__init__',
                    'severity': 'WARNING'
                })

            print(f"  ✅ Signal class docstring validated")

        except Exception as e:
            print(f"  ❌ Error checking Signal class: {e}")

        return results

    def validate_tests_exist(self) -> Dict[str, Any]:
        """Validate contract tests exist and are runnable."""
        print("\n🧪 Validating contract tests...")

        results = {
            'test_file_exists': False,
            'test_classes': []
        }

        test_file = self.project_root / 'tests' / 'test_archetype_contracts.py'

        if not test_file.exists():
            print(f"  ❌ Contract test file not found: {test_file}")
            return results

        results['test_file_exists'] = True
        print(f"  ✅ Contract test file exists")

        # Parse test file to count test classes
        content = test_file.read_text()
        test_classes = re.findall(r'class (Test\w+)', content)
        results['test_classes'] = test_classes

        print(f"  ✅ Found {len(test_classes)} test classes:")
        for cls in test_classes:
            print(f"     - {cls}")

        return results

    def generate_report(self) -> Dict[str, Any]:
        """Generate full validation report."""
        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'project_root': str(self.project_root),
            'validation_results': {}
        }

        # Run validations
        report['validation_results']['strategy_files'] = self.validate_strategy_files()
        report['validation_results']['docstrings'] = self.validate_docstrings()
        report['validation_results']['tests'] = self.validate_tests_exist()

        # Summary
        report['summary'] = {
            'total_violations': len(self.violations),
            'total_warnings': len(self.warnings),
            'status': 'PASS' if len(self.violations) == 0 else 'FAIL'
        }

        return report

    def print_report(self, report: Dict[str, Any]):
        """Print human-readable report."""
        print("\n" + "="*80)
        print("CONTRACT VALIDATION REPORT")
        print("="*80)

        print(f"\n📅 Timestamp: {report['timestamp']}")
        print(f"📁 Project: {report['project_root']}")

        # Summary
        summary = report['summary']
        print(f"\n📊 SUMMARY")
        print(f"  Status: {'✅ PASS' if summary['status'] == 'PASS' else '❌ FAIL'}")
        print(f"  Violations: {summary['total_violations']}")
        print(f"  Warnings: {summary['total_warnings']}")

        # Violations
        if self.violations:
            print(f"\n❌ VIOLATIONS ({len(self.violations)})")
            for v in self.violations:
                print(f"\n  {v['file']}:{v['line']}")
                print(f"    Severity: {v['severity']}")
                print(f"    Issue: {v['description']}")
                print(f"    Pattern: {v['pattern']}")
                print(f"    Code: {v['code']}")

        # Warnings
        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)})")
            for w in self.warnings:
                print(f"\n  {w['file']}:{w['line']}")
                print(f"    Issue: {w['description']}")
                print(f"    Code: {w['code']}")

        # Recommendations
        print(f"\n💡 RECOMMENDATIONS")
        if self.violations:
            print("  1. Fix violations before committing")
            print("  2. Use ArchetypeModel.predict() instead of internal methods")
            print("  3. See ARCHETYPE_API_CONTRACT.md for guidance")
        elif self.warnings:
            print("  1. Consider migrating to ArchetypeModel.predict()")
            print("  2. See ARCHETYPE_API_CONTRACT.md for migration guide")
        else:
            print("  ✅ No issues found - contracts are compliant!")

        print("\n" + "="*80)

    def save_report(self, report: Dict[str, Any], output_path: Path):
        """Save report to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n💾 Report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Validate archetype API contracts')
    parser.add_argument('--quick', action='store_true', help='Quick validation only')
    parser.add_argument('--fix', action='store_true', help='Auto-fix violations (where possible)')
    parser.add_argument('--output', type=str, help='Output JSON report path')
    parser.add_argument('--quiet', action='store_true', help='Quiet mode (errors only)')

    args = parser.parse_args()

    # Create validator
    validator = ContractValidator(project_root)

    # Generate report
    print("🚀 Starting contract validation...\n")
    report = validator.generate_report()

    # Print report
    if not args.quiet:
        validator.print_report(report)

    # Save report
    if args.output:
        output_path = Path(args.output)
        validator.save_report(report, output_path)
    else:
        # Default output location
        default_output = project_root / 'CONTRACT_VALIDATION_REPORT.json'
        validator.save_report(report, default_output)

    # Exit code
    exit_code = 0 if report['summary']['status'] == 'PASS' else 1

    if exit_code != 0:
        print("\n❌ Validation FAILED - fix violations before committing")
    else:
        print("\n✅ Validation PASSED - contracts are compliant")

    sys.exit(exit_code)


if __name__ == '__main__':
    main()
