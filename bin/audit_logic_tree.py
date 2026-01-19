#!/usr/bin/env python3
"""
Logic Tree Audit - Map Wired vs Unwired vs Ghost Components

Traces the complete dependency graph from:
Config → LogicAdapter → Domain Engines → Features

Categorizes components as:
- GREEN (Wired & Used): Actually connected and used in logic
- YELLOW (Unwired): Exists but not connected to archetype logic
- RED (Ghost): Referenced but doesn't exist

Generates visual dependency graphs and actionable reports.
"""

import ast
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import graphviz
    HAS_GRAPHVIZ = True
except ImportError:
    HAS_GRAPHVIZ = False
    print("Warning: graphviz not installed. Install with: pip install graphviz")
    print("Continuing with text-only analysis...")


class LogicTreeAuditor:
    """Audit the logic tree to identify wired vs unwired vs ghost components."""

    def __init__(self, project_root: Path):
        self.project_root = project_root

        # Results storage
        self.green_features = defaultdict(set)  # archetype -> features
        self.yellow_features = set()  # exists but unwired
        self.red_features = set()  # ghost/idea-only

        self.archetype_checks = {}  # archetype -> check function code
        self.domain_engines = {}  # engine -> methods
        self.feature_store_columns = set()
        self.config_references = defaultdict(set)  # archetype -> referenced features

        # Dependency edges for graph
        self.edges = []  # (from, to, status, archetype)

    def run_full_audit(self):
        """Run complete audit pipeline."""
        print("=" * 80)
        print("LOGIC TREE AUDIT - Bull Machine Wiring Analysis")
        print("=" * 80)
        print()

        print("Phase 1: Scanning feature store schema...")
        self.scan_feature_store()

        print("Phase 2: Analyzing domain engines...")
        self.scan_domain_engines()

        print("Phase 3: Parsing archetype check functions...")
        self.scan_archetype_logic()

        print("Phase 4: Analyzing configs...")
        self.scan_configs()

        print("Phase 5: Categorizing features...")
        self.categorize_features()

        print("Phase 6: Generating reports...")
        self.generate_text_report()

        if HAS_GRAPHVIZ:
            print("Phase 7: Generating visual diagrams...")
            self.generate_visual_diagrams()

        print()
        print("=" * 80)
        print("AUDIT COMPLETE")
        print("=" * 80)

    def scan_feature_store(self):
        """Extract all columns from feature store schema."""
        schema_file = self.project_root / "docs" / "FEATURE_STORE_SCHEMA_v2.md"

        if not schema_file.exists():
            print(f"  Warning: {schema_file} not found")
            # Fallback: scan actual feature files
            self._scan_feature_modules()
            return

        content = schema_file.read_text()

        # Extract feature names from schema markdown
        # Look for patterns like: | feature_name | type | description |
        pattern = r'\|\s+([a-z_][a-z0-9_]+)\s+\|'
        matches = re.findall(pattern, content, re.IGNORECASE)

        for match in matches:
            # Filter out table headers
            if match.lower() not in ['column', 'name', 'feature', 'type', 'description']:
                self.feature_store_columns.add(match)

        print(f"  Found {len(self.feature_store_columns)} features in schema")

        # Also scan actual feature implementations
        self._scan_feature_modules()

    def _scan_feature_modules(self):
        """Scan feature implementation files for feature names."""
        features_dir = self.project_root / "engine" / "features"

        if not features_dir.exists():
            return

        # Scan all Python files in features directory
        for py_file in features_dir.glob("*.py"):
            if py_file.name == "__init__.py":
                continue

            try:
                content = py_file.read_text()

                # Look for feature registration patterns
                # e.g., df['feature_name'] = ...
                pattern = r"df\[['\"]([-a-z0-9_]+)['\"]\]\s*="
                matches = re.findall(pattern, content)
                self.feature_store_columns.update(matches)

                # Look for feature computation patterns
                # e.g., compute_feature_name(...)
                pattern = r"def\s+compute_([a-z_][a-z0-9_]+)\s*\("
                matches = re.findall(pattern, content)
                self.feature_store_columns.update(matches)

            except Exception as e:
                print(f"  Warning: Error scanning {py_file}: {e}")

        print(f"  Total features after scanning modules: {len(self.feature_store_columns)}")

    def scan_domain_engines(self):
        """Scan domain engine implementations."""
        engines = {
            'wyckoff': self.project_root / "engine" / "wyckoff" / "wyckoff_engine.py",
            'smc': self.project_root / "engine" / "smc" / "smc_engine.py",
            'temporal': self.project_root / "engine" / "temporal" / "temporal_fusion.py",
            'hob': self.project_root / "engine" / "hob" / "hob_engine.py",
        }

        for engine_name, engine_file in engines.items():
            if not engine_file.exists():
                print(f"  Warning: {engine_file} not found")
                continue

            try:
                content = engine_file.read_text()
                tree = ast.parse(content)

                methods = set()
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        methods.add(node.name)

                self.domain_engines[engine_name] = methods
                print(f"  {engine_name}: {len(methods)} methods")

            except Exception as e:
                print(f"  Warning: Error parsing {engine_file}: {e}")

    def scan_archetype_logic(self):
        """Parse archetype check functions to find feature usage."""
        logic_file = self.project_root / "engine" / "archetypes" / "logic_v2_adapter.py"

        if not logic_file.exists():
            print(f"  Error: {logic_file} not found")
            return

        content = logic_file.read_text()

        # Extract check functions - use uppercase for actual function names
        archetypes = [
            ('S1', 's1'),
            ('S2', 's2'),
            ('S4', 's4'),
            ('S5', 's5'),
            ('A', 'bos_choch'),
            ('B', 'order_block_retest'),
            ('C', 'trap_within_trend'),
            ('D', 'long_squeeze'),
            ('E', 'failed_rally'),
            ('F', 'funding_divergence'),
            ('G', 'liquidity_vacuum'),
        ]

        for func_name, archetype_id in archetypes:
            # Find the check function - must be actual function definition, not dict reference
            # Pattern: "def _check_XXX(self, context:" through next method definition
            pattern = rf"def _check_{func_name}\s*\(\s*self\s*,\s*context:.*?(?=\n\s+def\s+_check_|\nclass\s+|\Z)"
            match = re.search(pattern, content, re.DOTALL)

            if match:
                func_code = match.group(0)
                self.archetype_checks[archetype_id] = func_code

                # Extract feature accesses
                features_used = self._extract_features_from_code(func_code)
                self.green_features[archetype_id] = features_used

                print(f"  {archetype_id.upper()}: {len(features_used)} features used")
            else:
                print(f"  Warning: _check_{func_name} not found")

    def _extract_features_from_code(self, code: str) -> Set[str]:
        """Extract feature names accessed in code."""
        features = set()

        # Pattern 1: df['feature_name'] or df["feature_name"]
        pattern1 = r"df\[['\"]([-a-z0-9_]+)['\"]\]"
        features.update(re.findall(pattern1, code, re.IGNORECASE))

        # Pattern 2: df.get('feature_name') or df.get("feature_name")
        pattern2 = r"df\.get\(['\"]([a-z0-9_]+)['\"]\)"
        features.update(re.findall(pattern2, code, re.IGNORECASE))

        # Pattern 3: row['feature_name'] or row["feature_name"]
        pattern3 = r"row\[['\"]([-a-z0-9_]+)['\"]\]"
        features.update(re.findall(pattern3, code, re.IGNORECASE))

        # Pattern 4: row.get('feature_name')
        pattern4 = r"row\.get\(['\"]([a-z0-9_]+)['\"]\)"
        features.update(re.findall(pattern4, code, re.IGNORECASE))

        # Pattern 5: feature_name in row or feature_name in df
        pattern5 = r"['\"]([a-z0-9_]+)['\"]\s+in\s+(?:row|df)"
        features.update(re.findall(pattern5, code, re.IGNORECASE))

        # Pattern 6: self.g(context.row, 'feature_name', ...)
        pattern6 = r"self\.g\s*\(\s*context\.row\s*,\s*['\"]([a-z0-9_]+)['\"]\s*,"
        features.update(re.findall(pattern6, code, re.IGNORECASE))

        # Pattern 7: self.get_feature(context, 'feature_name')
        pattern7 = r"self\.get_feature\s*\(\s*context\s*,\s*['\"]([a-z0-9_]+)['\"]\s*\)"
        features.update(re.findall(pattern7, code, re.IGNORECASE))

        # Pattern 8: context.row['feature_name']
        pattern8 = r"context\.row\[['\"]([-a-z0-9_]+)['\"]\]"
        features.update(re.findall(pattern8, code, re.IGNORECASE))

        # Pattern 9: context.row.get('feature_name')
        pattern9 = r"context\.row\.get\(['\"]([a-z0-9_]+)['\"]\)"
        features.update(re.findall(pattern9, code, re.IGNORECASE))

        return features

    def scan_configs(self):
        """Scan config files for feature references."""
        config_dir = self.project_root / "configs"

        # Focus on production configs
        config_files = [
            config_dir / "mvp" / "mvp_bull_market_v1.json",
            config_dir / "mvp" / "mvp_bear_market_v1.json",
            config_dir / "system_b0_production.json",
            config_dir / "s1_v2_production.json",
        ]

        for config_file in config_files:
            if not config_file.exists():
                continue

            try:
                with open(config_file) as f:
                    config = json.load(f)

                # Extract archetype from config
                archetype_name = self._extract_archetype_from_config(config, config_file)

                # Find all feature references in config
                features = self._extract_features_from_dict(config)
                self.config_references[archetype_name].update(features)

                print(f"  {config_file.name}: {len(features)} feature references")

            except Exception as e:
                print(f"  Warning: Error parsing {config_file}: {e}")

    def _extract_archetype_from_config(self, config: dict, config_file: Path) -> str:
        """Extract archetype identifier from config."""
        # Try to get from config structure
        if 'archetype' in config:
            return config['archetype']

        # Try from filename
        name = config_file.stem.lower()
        if 's1' in name:
            return 's1'
        elif 's2' in name:
            return 's2'
        elif 's4' in name:
            return 's4'
        elif 's5' in name:
            return 's5'
        elif 'b0' in name:
            return 'b0'
        elif 'bull' in name:
            return 'bull_market'
        elif 'bear' in name:
            return 'bear_market'

        return 'unknown'

    def _extract_features_from_dict(self, d: dict, features: Set[str] = None) -> Set[str]:
        """Recursively extract feature references from config dict."""
        if features is None:
            features = set()

        # Filter out config metadata fields (these are not features)
        metadata_prefixes = [
            '_comment', '_note', '_group', '_philosophy', '_metadata',
            '_expected', '_known', '_mode', '_fix', '_historical',
            '_domain_engine', '_calibration', '_production', '_gate',
            '_exhaustion', '_confluence', '_regime'
        ]

        config_keys = [
            'archetype_weight', 'feature_flags', 'risk_management',
            'deployment', 'validation', 'performance', 'safety',
            'enabled', 'require', 'allow', 'auto_stop', 'max_', 'min_',
            'use_', 'enable_', 'cooldown', 'kill_switch', 'emergency',
            'webhook', 'log_', 'cache', 'check_interval', 'commission',
            'slippage', 'backtest', 'walk_forward', 'study_name',
            'trial_', 'optuna', 'test_', 'train_', 'step_', 'time_limit',
            'operator_guide', 'tuning_guide', 'created_at', 'last_',
            'applied_date', 'update_reason', 'backup', 'recovery',
            'model_path', 'model_class', 'ml_filter', 'pnl_tracker',
            'portfolio_size', 'position_sizing', 'base_risk', 'scale_out',
            'trail_', 'stop_', 'profit_target', 'data_source', 'feature_store_path',
            'feature_order', 'zero_fill'
        ]

        for key, value in d.items():
            # Skip metadata fields
            if isinstance(key, str):
                # Skip if starts with metadata prefix
                if any(key.startswith(prefix) for prefix in metadata_prefixes):
                    continue

                # Skip if matches config key patterns
                if any(pattern in key for pattern in config_keys):
                    continue

                # Keys that might be feature names (lowercase with underscore)
                if '_' in key and key.islower() and not key.startswith('_'):
                    features.add(key)

            # Recurse into nested dicts
            if isinstance(value, dict):
                self._extract_features_from_dict(value, features)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        self._extract_features_from_dict(item, features)

        return features

    def categorize_features(self):
        """Categorize features into GREEN/YELLOW/RED."""
        # GREEN: Features actually used in archetype logic
        all_green = set()
        for features in self.green_features.values():
            all_green.update(features)

        # YELLOW: Exists in feature store but not used in any archetype
        self.yellow_features = self.feature_store_columns - all_green

        # RED: Referenced in configs but doesn't exist
        all_config_refs = set()
        for features in self.config_references.values():
            all_config_refs.update(features)

        self.red_features = all_config_refs - self.feature_store_columns - all_green

        # Build dependency edges for graph
        for archetype, features in self.green_features.items():
            for feature in features:
                self.edges.append((archetype, feature, 'green', archetype))

        for feature in self.yellow_features:
            self.edges.append(('feature_store', feature, 'yellow', 'unused'))

        for archetype, features in self.config_references.items():
            for feature in features:
                if feature in self.red_features:
                    self.edges.append((f"config_{archetype}", feature, 'red', archetype))

    def generate_text_report(self):
        """Generate comprehensive text report."""
        report = []

        report.append("=" * 80)
        report.append("LOGIC TREE AUDIT REPORT")
        report.append("=" * 80)
        report.append("")

        report.append("EXECUTIVE SUMMARY")
        report.append("-" * 80)
        report.append(f"GREEN (Wired & Used):     {len(set().union(*self.green_features.values()))} features")
        report.append(f"YELLOW (Unwired):         {len(self.yellow_features)} features")
        report.append(f"RED (Ghost/Idea-Only):    {len(self.red_features)} features")
        report.append("")

        # GREEN features by archetype
        report.append("")
        report.append("GREEN - WIRED & USED FEATURES")
        report.append("=" * 80)
        for archetype in sorted(self.green_features.keys()):
            features = sorted(self.green_features[archetype])
            report.append(f"\n{archetype.upper()} ({len(features)} features):")
            report.append("-" * 40)
            for feat in features:
                report.append(f"  ✓ {feat}")

        # YELLOW features
        report.append("")
        report.append("")
        report.append("YELLOW - UNWIRED FEATURES (Exist but Not Used)")
        report.append("=" * 80)
        report.append(f"\nFound {len(self.yellow_features)} unwired features:")
        report.append("")
        for feat in sorted(self.yellow_features):
            report.append(f"  ⚠ {feat}")

        # RED features
        report.append("")
        report.append("")
        report.append("RED - GHOST FEATURES (Referenced but Don't Exist)")
        report.append("=" * 80)
        report.append(f"\nFound {len(self.red_features)} ghost features:")
        report.append("")
        for feat in sorted(self.red_features):
            report.append(f"  ✗ {feat}")

        # Domain engines analysis
        report.append("")
        report.append("")
        report.append("DOMAIN ENGINES")
        report.append("=" * 80)
        for engine, methods in self.domain_engines.items():
            report.append(f"\n{engine.upper()} ({len(methods)} methods):")
            report.append("-" * 40)
            for method in sorted(methods)[:10]:  # Show first 10
                report.append(f"  • {method}")
            if len(methods) > 10:
                report.append(f"  ... and {len(methods) - 10} more")

        # Save report
        report_file = self.project_root / "LOGIC_TREE_AUDIT_REPORT.md"
        report_file.write_text("\n".join(report))
        print(f"\n  ✓ Saved: {report_file}")

        # Generate priority list
        self._generate_priority_report()

    def _generate_priority_report(self):
        """Generate prioritized action list for unwired features."""
        report = []

        report.append("# UNWIRED FEATURES - PRIORITY ACTION LIST")
        report.append("")
        report.append("Features that exist in the system but are not wired into archetype logic.")
        report.append("Priority is based on potential impact and domain engine coverage.")
        report.append("")

        # Categorize yellow features by domain
        wyckoff_features = [f for f in self.yellow_features if 'wyckoff' in f.lower()]
        smc_features = [f for f in self.yellow_features if 'smc' in f.lower() or 'bos' in f.lower() or 'choch' in f.lower()]
        temporal_features = [f for f in self.yellow_features if 'temporal' in f.lower() or 'confluence' in f.lower()]
        hob_features = [f for f in self.yellow_features if 'hob' in f.lower() or 'imbalance' in f.lower()]
        volume_features = [f for f in self.yellow_features if 'volume' in f.lower() or 'vol_' in f.lower()]
        funding_features = [f for f in self.yellow_features if 'funding' in f.lower()]
        other_features = [f for f in self.yellow_features if f not in wyckoff_features + smc_features + temporal_features + hob_features + volume_features + funding_features]

        # High Priority
        report.append("## HIGH PRIORITY (Wire These First)")
        report.append("")

        if wyckoff_features:
            report.append("### Wyckoff Features")
            report.append("**Impact**: High - Structural market phases")
            report.append("**Suggested Archetypes**: S1 (Liquidity Vacuum), S5 (Trap Within Trend)")
            report.append("")
            for feat in sorted(wyckoff_features):
                report.append(f"- [ ] `{feat}`")
            report.append("")

        if smc_features:
            report.append("### Smart Money Concepts (SMC)")
            report.append("**Impact**: High - Institutional order flow")
            report.append("**Suggested Archetypes**: S1, S4, S5")
            report.append("")
            for feat in sorted(smc_features):
                report.append(f"- [ ] `{feat}`")
            report.append("")

        # Medium Priority
        report.append("## MEDIUM PRIORITY")
        report.append("")

        if temporal_features:
            report.append("### Temporal/Confluence Features")
            report.append("**Impact**: Medium - Multi-timeframe confirmation")
            report.append("**Suggested Archetypes**: All archetypes")
            report.append("")
            for feat in sorted(temporal_features):
                report.append(f"- [ ] `{feat}`")
            report.append("")

        if hob_features:
            report.append("### HOB (High-Order Book) Features")
            report.append("**Impact**: Medium - Liquidity analysis")
            report.append("**Suggested Archetypes**: S1, S4")
            report.append("")
            for feat in sorted(hob_features):
                report.append(f"- [ ] `{feat}`")
            report.append("")

        # Low Priority
        report.append("## LOW PRIORITY")
        report.append("")

        if volume_features:
            report.append("### Volume Features")
            for feat in sorted(volume_features):
                report.append(f"- [ ] `{feat}`")
            report.append("")

        if funding_features:
            report.append("### Funding Features")
            for feat in sorted(funding_features):
                report.append(f"- [ ] `{feat}`")
            report.append("")

        if other_features:
            report.append("### Other Features")
            for feat in sorted(other_features):
                report.append(f"- [ ] `{feat}`")
            report.append("")

        # Ghost features section
        report.append("")
        report.append("## GHOST FEATURES (Remove or Implement)")
        report.append("")
        report.append("These are referenced in configs but have no implementation:")
        report.append("")
        for feat in sorted(self.red_features):
            report.append(f"- [ ] `{feat}` - Remove from configs or implement")

        # Save report
        priority_file = self.project_root / "UNWIRED_FEATURES_PRIORITY.md"
        priority_file.write_text("\n".join(report))
        print(f"  ✓ Saved: {priority_file}")

    def generate_visual_diagrams(self):
        """Generate visual dependency graphs using graphviz."""
        if not HAS_GRAPHVIZ:
            return

        results_dir = self.project_root / "results"
        results_dir.mkdir(exist_ok=True)

        # Generate full logic tree
        self._generate_full_tree(results_dir)

        # Generate per-archetype trees
        for archetype in self.green_features.keys():
            self._generate_archetype_tree(archetype, results_dir)

    def _generate_full_tree(self, results_dir: Path):
        """Generate complete logic tree diagram."""
        dot = graphviz.Digraph(comment='Bull Machine Logic Tree', format='png')
        dot.attr(rankdir='LR', size='16,12')
        dot.attr('node', shape='box', style='rounded,filled')

        # Add nodes
        for archetype in self.green_features.keys():
            dot.node(archetype, archetype.upper(), fillcolor='lightblue')

        # Add feature nodes with color coding
        all_features = set().union(*self.green_features.values())
        for feat in all_features:
            dot.node(feat, feat, fillcolor='lightgreen')

        for feat in self.yellow_features:
            dot.node(feat, feat, fillcolor='yellow')

        for feat in self.red_features:
            dot.node(feat, feat, fillcolor='lightcoral')

        # Add edges
        for archetype, features in self.green_features.items():
            for feat in features:
                dot.edge(archetype, feat, color='green')

        # Save
        output_file = results_dir / "logic_tree"
        try:
            dot.render(output_file, cleanup=True)
            print(f"  ✓ Saved: {output_file}.png")
        except Exception as e:
            print(f"  Warning: Error generating full tree: {e}")

    def _generate_archetype_tree(self, archetype: str, results_dir: Path):
        """Generate logic tree for specific archetype."""
        dot = graphviz.Digraph(comment=f'{archetype.upper()} Logic Tree', format='png')
        dot.attr(rankdir='LR', size='12,8')
        dot.attr('node', shape='box', style='rounded,filled')

        # Add archetype node
        dot.node(archetype, archetype.upper(), fillcolor='lightblue')

        # Add green features
        if archetype in self.green_features:
            for feat in sorted(self.green_features[archetype]):
                dot.node(feat, feat, fillcolor='lightgreen')
                dot.edge(archetype, feat, color='green', label='USED')

        # Add yellow features (relevant to this archetype)
        # Show domain-related unwired features
        relevant_yellow = set()
        if 's1' in archetype.lower():
            relevant_yellow = {f for f in self.yellow_features if 'wyckoff' in f or 'smc' in f}
        elif 's4' in archetype.lower():
            relevant_yellow = {f for f in self.yellow_features if 'funding' in f or 'hob' in f}
        elif 's5' in archetype.lower():
            relevant_yellow = {f for f in self.yellow_features if 'wyckoff' in f or 'temporal' in f}

        for feat in sorted(relevant_yellow):
            dot.node(feat, feat, fillcolor='yellow')
            dot.edge(archetype, feat, color='orange', style='dashed', label='UNWIRED')

        # Save
        output_file = results_dir / f"logic_tree_{archetype}"
        try:
            dot.render(output_file, cleanup=True)
            print(f"  ✓ Saved: {output_file}.png")
        except Exception as e:
            print(f"  Warning: Error generating {archetype} tree: {e}")


def main():
    """Main entry point."""
    project_root = Path(__file__).parent.parent

    auditor = LogicTreeAuditor(project_root)
    auditor.run_full_audit()

    print()
    print("NEXT STEPS:")
    print("-" * 80)
    print("1. Review LOGIC_TREE_AUDIT_REPORT.md for full details")
    print("2. Review UNWIRED_FEATURES_PRIORITY.md for action items")
    if HAS_GRAPHVIZ:
        print("3. Check results/ directory for visual diagrams")
        print("4. Wire high-priority yellow features into archetype logic")
    else:
        print("3. Install graphviz for visual diagrams: pip install graphviz")
        print("4. Wire high-priority yellow features into archetype logic")
    print("5. Remove or implement red ghost features")
    print()


if __name__ == "__main__":
    main()
