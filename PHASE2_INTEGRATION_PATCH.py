"""
Phase 2 Regime Classifier - Integration Patch for hybrid_runner.py

This file contains the exact code to add to hybrid_runner.py for Phase 2 integration.
Apply these changes in order for seamless regime adaptation.

SAFETY: Start with shadow_mode=true to log without applying adjustments.
"""

# ============================================================================
# STEP 1: Add imports at top of hybrid_runner.py (after line 45)
# ============================================================================

IMPORTS_TO_ADD = """
# Phase 2: Regime adaptation
from engine.context.regime_classifier import RegimeClassifier
from engine.context.regime_policy import RegimePolicy
"""

# ============================================================================
# STEP 2: Initialize regime components in __init__ (after line 193)
# ============================================================================

INIT_CODE = """
        # Phase 2: Regime adaptation (optional)
        self.regime_enabled = self.config.get('regime', {}).get('enabled', False)
        self.regime_shadow_mode = self.config.get('regime', {}).get('shadow_mode', True)
        self.regime_classifier = None
        self.regime_policy = None
        self.regime_stats = {'total_bars': 0, 'adjustments_applied': 0, 'regime_counts': {}}

        if self.regime_enabled:
            print("🧠 Loading Phase 2 regime classifier...")
            feature_order = [
                "VIX", "DXY", "MOVE", "YIELD_2Y", "YIELD_10Y",
                "USDT.D", "BTC.D", "TOTAL", "TOTAL2",
                "funding", "oi", "rv_20d", "rv_60d"
            ]
            try:
                self.regime_classifier = RegimeClassifier.load(
                    "models/regime_classifier_gmm.pkl",
                    feature_order
                )
                self.regime_policy = RegimePolicy.load("configs/v19/regime_policy.json")
                mode_str = "SHADOW MODE (logging only)" if self.regime_shadow_mode else "ACTIVE"
                print(f"✅ Regime adaptation enabled ({mode_str})")
            except Exception as e:
                print(f"⚠️  Regime classifier load failed: {e}")
                print(f"   Falling back to baseline (regime_enabled=False)")
                self.regime_enabled = False
"""

# ============================================================================
# STEP 3: Apply regime adjustments in bar loop
# Find: macro_row = fetch_macro_snapshot(...)
# Add: AFTER the fetch_macro_snapshot call
# ============================================================================

BAR_LOOP_CODE = """
        # Phase 2: Regime adaptation
        regime_info = None
        regime_adjustment = None
        adjusted_threshold = self.config['fusion']['entry_threshold_confidence']
        adjusted_weights = self.config['fusion']['weights'].copy()
        risk_multiplier = 1.0

        if self.regime_enabled and self.regime_classifier and self.regime_policy:
            try:
                # Classify current regime
                regime_info = self.regime_classifier.classify(macro_row)

                # Apply policy adjustments
                regime_adjustment = self.regime_policy.apply(self.config, regime_info)

                # Track stats
                self.regime_stats['total_bars'] += 1
                regime_label = regime_info['regime']
                self.regime_stats['regime_counts'][regime_label] = \\
                    self.regime_stats['regime_counts'].get(regime_label, 0) + 1

                if regime_adjustment['applied']:
                    self.regime_stats['adjustments_applied'] += 1

                    # Calculate adjustments
                    base_threshold = self.config['fusion']['entry_threshold_confidence']
                    adjusted_threshold = base_threshold + regime_adjustment['enter_threshold_delta']

                    # Apply weight nudges
                    for domain, nudge in regime_adjustment['weight_nudges'].items():
                        if domain in adjusted_weights:
                            adjusted_weights[domain] += nudge

                    # Renormalize weights to preserve sum
                    weight_sum = sum(adjusted_weights.values())
                    original_sum = sum(self.config['fusion']['weights'].values())
                    if weight_sum > 0:
                        adjusted_weights = {k: v * original_sum / weight_sum
                                          for k, v in adjusted_weights.items()}

                    # Store risk multiplier for position sizing
                    risk_multiplier = regime_adjustment['risk_multiplier']

                    # Log regime state (every adjustment)
                    conf = regime_info['proba'][regime_label]
                    print(f"  [REGIME] {regime_label} (conf={conf:.2f})")
                    print(f"           threshold: {base_threshold:.2f} → {adjusted_threshold:.2f} (Δ{regime_adjustment['enter_threshold_delta']:+.2f})")
                    print(f"           risk_mult: {risk_multiplier:.2f}x")
                    if regime_adjustment['weight_nudges']:
                        print(f"           weight_nudges: {regime_adjustment['weight_nudges']}")

                    # Apply adjustments (unless shadow mode)
                    if not self.regime_shadow_mode:
                        # Actually modify config for this bar
                        self.config['fusion']['entry_threshold_confidence'] = adjusted_threshold
                        self.config['fusion']['weights'] = adjusted_weights
                    else:
                        print(f"           [SHADOW MODE - NOT APPLIED]")

            except Exception as e:
                print(f"⚠️  Regime classification error: {e}")
                # Fall through to baseline on error
"""

# ============================================================================
# STEP 4: Apply risk multiplier to position sizing
# Find: size_usd = balance * (config['risk']['risk_per_trade_pct'] / 100.0)
# Replace with:
# ============================================================================

POSITION_SIZING_CODE = """
        # Calculate base position size
        size_usd = balance * (config['risk']['risk_per_trade_pct'] / 100.0)

        # Phase 2: Apply regime risk multiplier (unless shadow mode)
        if self.regime_enabled and regime_adjustment and not self.regime_shadow_mode:
            size_usd *= risk_multiplier
"""

# ============================================================================
# STEP 5: Add regime stats to final report
# Find: print("💰 Final Results:")
# Add: BEFORE the final results print
# ============================================================================

FINAL_REPORT_CODE = """
        # Phase 2: Regime adaptation summary
        if self.regime_enabled and self.regime_stats['total_bars'] > 0:
            print("\\n" + "="*70)
            print("🧠 PHASE 2 REGIME ADAPTATION SUMMARY")
            print("="*70)
            total = self.regime_stats['total_bars']
            applied = self.regime_stats['adjustments_applied']
            print(f"Total bars processed: {total}")
            print(f"Adjustments applied: {applied} ({applied/total*100:.1f}%)")
            print(f"Mode: {'SHADOW (logged only)' if self.regime_shadow_mode else 'ACTIVE'}")
            print(f"\\nRegime distribution:")
            for regime, count in sorted(self.regime_stats['regime_counts'].items()):
                pct = count / total * 100
                print(f"  {regime:12s}: {count:5d} bars ({pct:5.1f}%)")
            print("="*70 + "\\n")
"""

# ============================================================================
# STEP 6: Config section to add to your JSON configs
# ============================================================================

CONFIG_SECTION = {
    "regime": {
        "enabled": False,  # Set to True to enable Phase 2
        "shadow_mode": True,  # Set to False to actually apply adjustments
        "min_confidence": 0.60,  # Min confidence to apply adjustments
        "max_threshold_delta": 0.05,  # Cap threshold adjustments (safer than 0.10)
        "max_risk_multiplier": 1.15,  # Cap risk multiplier (safer than 1.25)
        "description": "Phase 2 regime adaptation - start with shadow_mode=true"
    }
}

# ============================================================================
# INTEGRATION INSTRUCTIONS
# ============================================================================

INSTRUCTIONS = """
HOW TO INTEGRATE:

1. BACKUP FIRST:
   cp bin/live/hybrid_runner.py bin/live/hybrid_runner.py.backup

2. ADD IMPORTS (top of file, after line 45):
   - Add the IMPORTS_TO_ADD code block

3. ADD INIT CODE (in __init__, after line 193):
   - Add the INIT_CODE block

4. ADD BAR LOOP CODE (after fetch_macro_snapshot call):
   - Search for: macro_row = fetch_macro_snapshot(
   - Add BAR_LOOP_CODE after that line

5. MODIFY POSITION SIZING:
   - Search for: size_usd = balance * (config['risk']
   - Replace with POSITION_SIZING_CODE

6. ADD FINAL REPORT:
   - Search for: print("💰 Final Results:")
   - Add FINAL_REPORT_CODE before it

7. UPDATE YOUR CONFIG:
   - Add CONFIG_SECTION to configs/v18/BTC_conservative.json

8. TEST IN SHADOW MODE:
   python3 bin/live/hybrid_runner.py --asset BTC --start 2024-07-01 --end 2024-09-30 \\
     --config configs/v18/BTC_conservative.json

   - Verify regime logging appears
   - Check "[SHADOW MODE - NOT APPLIED]" messages
   - Confirm no actual config modifications

9. ENABLE THRESHOLD-ONLY MODE:
   - Set: regime.enabled=true, shadow_mode=false
   - Set: max_risk_multiplier=1.0 (no risk scaling yet)
   - Run and compare vs baseline

10. ENABLE FULL REGIME MODE:
    - Set: max_risk_multiplier=1.15
    - Validate acceptance gates
    - Monitor for regime whipsaw

SAFETY CHECKLIST:
☐ Backup hybrid_runner.py before editing
☐ Test with shadow_mode=true first
☐ Start with conservative caps (0.05 threshold, 1.15 risk)
☐ Monitor regime switch frequency
☐ Validate trade count retention ≥80%
☐ Check for regime whipsaw (rapid switching)
☐ Verify confidence threshold enforcement
☐ Compare PF/Sharpe vs baseline

ROLLBACK:
If issues occur:
1. Set regime.enabled=false in config
2. OR: cp bin/live/hybrid_runner.py.backup bin/live/hybrid_runner.py
3. Re-run with baseline config
"""

if __name__ == '__main__':
    print(__doc__)
    print(INSTRUCTIONS)
    print("\\n" + "="*70)
    print("All code blocks are available as variables in this module:")
    print("  - IMPORTS_TO_ADD")
    print("  - INIT_CODE")
    print("  - BAR_LOOP_CODE")
    print("  - POSITION_SIZING_CODE")
    print("  - FINAL_REPORT_CODE")
    print("  - CONFIG_SECTION")
    print("="*70)
