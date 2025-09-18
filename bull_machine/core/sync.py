"""Bull Machine v1.3 - Multi-Timeframe Sync Decision Logic"""

from typing import List
from bull_machine.core.types import BiasCtx, SyncReport

def decide_mtf_entry(
    htf: BiasCtx,
    mtf: BiasCtx,
    ltf_bias: str,
    nested_ok: bool,
    eq_magnet: bool,
    policy: dict
) -> SyncReport:
    """
    Return ALLOW / RAISE / VETO with optional threshold_bump and notes.

    HTF dominance principle: Higher timeframe bias takes priority
    Alignment scoring: Bonus for all TFs pointing same direction
    EQ magnet: Veto or raise based on policy
    """

    notes = []
    alignment_score = 0.0
    threshold_bump = 0.0
    decision = 'allow'

    # Check for desync (HTF vs LTF conflict)
    desync = False
    if htf.bias != 'neutral' and ltf_bias != 'neutral':
        if htf.bias != ltf_bias:
            desync = True
            notes.append(f"HTF-LTF desync: {htf.tf}={htf.bias} vs LTF={ltf_bias}")

    # Calculate alignment score
    aligned_count = 0
    total_count = 0

    if htf.bias != 'neutral':
        total_count += 1
        if htf.bias == ltf_bias:
            aligned_count += 1

    if mtf.bias != 'neutral':
        total_count += 1
        if mtf.bias == ltf_bias:
            aligned_count += 1
        if mtf.bias == htf.bias:
            aligned_count += 0.5  # Partial credit for HTF-MTF alignment

    if total_count > 0:
        alignment_score = aligned_count / (total_count + 1)  # +1 for LTF itself
    else:
        alignment_score = 0.5  # Neutral case

    # Decision logic based on policy
    desync_behavior = policy.get('desync_behavior', 'raise')
    eq_magnet_gate = policy.get('eq_magnet_gate', True)

    # 1. Hard veto on strong desync
    if desync and htf.confirmed:
        if desync_behavior == 'veto':
            decision = 'veto'
            notes.append("VETO: Strong HTF-LTF desync with confirmed HTF")
        else:
            decision = 'raise'
            threshold_bump = policy.get('desync_bump', 0.10)
            notes.append(f"RAISE: HTF-LTF desync, threshold +{threshold_bump:.2f}")

    # 2. EQ magnet handling
    elif eq_magnet:
        if eq_magnet_gate:
            decision = 'veto'
            notes.append("VETO: Price in equilibrium zone (chop risk)")
        else:
            decision = 'raise'
            threshold_bump = max(threshold_bump, policy.get('eq_bump', 0.05))
            notes.append(f"RAISE: EQ magnet active, threshold +{threshold_bump:.2f}")

    # 3. Poor nested confluence
    elif not nested_ok:
        decision = 'raise'
        threshold_bump = max(threshold_bump, policy.get('nested_bump', 0.03))
        notes.append(f"RAISE: Poor nested confluence, threshold +{threshold_bump:.2f}")

    # 4. Perfect alignment bonus
    elif alignment_score >= 0.9:
        decision = 'allow'
        threshold_bump = -policy.get('alignment_discount', 0.05)  # Lower threshold
        notes.append(f"BONUS: Strong alignment ({alignment_score:.1%}), threshold -{abs(threshold_bump):.2f}")

    # 5. HTF strength bonus
    elif htf.confirmed and htf.strength > 0.7:
        decision = 'allow'
        notes.append(f"ALLOW: Strong HTF bias ({htf.bias}, strength={htf.strength:.2f})")

    # Default allow
    else:
        decision = 'allow'
        notes.append("ALLOW: Standard conditions met")

    return SyncReport(
        htf=htf,
        mtf=mtf,
        ltf_bias=ltf_bias,
        nested_ok=nested_ok,
        eq_magnet=eq_magnet,
        desync=desync,
        decision=decision,
        threshold_bump=threshold_bump,
        alignment_score=alignment_score,
        notes=notes
    )