# The All-Seeing Eye — Architecture Spec v1 (2026-08-05, DRAFT for user sign-off)

One HTF authority that reads the market's story top-down and scales conviction across
the entire book. Design is constrained by BOTH evidence bases: Wyckoff_Insider's
actual mechanics (wyckoff_audit.md addenda 6-11) and our own validation history
(boosts 6/6, filters 0/9, full-story standalone mechanization 0/4).

## Core principle (source + evidence agree)
The eye is a **SIZE DIAL with tiers, never a hard gate**. WI's own system: against-HTF
= tactical size only; full size only after model confirmation. Our history: every
veto-style filter failed; every sizing dial passed. The eye SCALES; it does not veto.

## The state machine (per timeframe: 1D primary, 1W secondary; body-close semantics)
States (WI's operative set):
1. IN_RANGE          — default; HTF range boundaries from confirmed body-close extremes
                       ("HTF range until proven otherwise"); track premium/discount
2. MANIPULATION      — wick(s) beyond a boundary WITHOUT body close (sweep). Explicitly
                       NOT a bias flip; arms reversal attention (feeds M1/M2 watch)
3. MODEL_FORMING     — M1/M2 accumulation or distribution developing at an extreme
                       (we already detect: phase machine + M2 path + spring/LPS events)
4. CONFIRMED_BREAK   — real candle-BODY close beyond the range + acceptance (holds,
                       does not immediately re-enter). Direction-tagged.
5. TRENDING          — post-break with successful retest/LPS. Direction-tagged.
Transitions: wick≠flip; body close + acceptance = flip; completed M2 (LPS+MSS) at an
extreme may shift LOCAL bias pre-break (WI rule). Early invalidation: absence of the
expected model at the extreme within a time window is itself a state signal.

## Output surface (features on the 1H grid, causal, closed-HTF-bar discipline)
- eye_state_1d / eye_state_1w  (enum above)
- eye_dir (accum/bull vs dist/bear vs neutral), eye_location (premium/discount pct)
- eye_conviction_tier: ALIGNED_CONFIRMED > ALIGNED_FORMING > NEUTRAL > COUNTER
  (entry direction vs eye_dir, with MODEL_FORMING/CONFIRMED distinctions)

## Sizing policy (the dial — pre-registered mapping, each rung validated separately)
- ALIGNED_CONFIRMED: 1.25x (extends the validated C_accum boost pattern; capex-scoped)
- ALIGNED_FORMING:   1.10x (small, must earn its own co-move pass)
- NEUTRAL:           1.00x (base risk — today's behavior)
- COUNTER:           0.75x (the "scalp tier"; DOWN-dial — this is the risky rung given
                     filters-0/9; validated LAST, shipped only on its own co-move pass)
Junk-book scope: DECISION PENDING (user) — whether the 13 unvalidated archetypes fall
under the eye's authority or stay exempt at full size for data collection.

## Validation ladder (no big-bang; every step gated like the C_accum boost was)
P1. FEATURES: build the state machine as a registry-style causal feature pass
    (patch-vNN pattern); truncation no-repaint tests; sanity distributions;
    spot-check against WI's labeled charts (the ground-truth batch).
P2. STAGE-1 DESCRIPTIVE (the gate): split existing champion entries by
    eye_conviction_tier — tier ordering must hold in BOTH eras (train + holdout)
    for the tiers to earn a build. If ordering fails → stop, report, redesign.
P3. UP-DIALS: wire ALIGNED_* boosts (mirror Boost 6 pattern), bit-identical OFF
    control, co-move battery per rung. Ship rungs that pass; drop rungs that fail.
P4. DOWN-DIAL: COUNTER 0.75x tested alone, same protocol, explicit awareness that
    this is filter-adjacent (history 0/9) — expected to fail honestly unless the
    eye's read is genuinely better than the old blanket filters.
P5. LIVE: mirror to v11_shadow_runner (parity tests — the step that catches
    silent-never-fires), deploy ONLY on explicit user go, watch composition.

## What the eye replaces / absorbs (coherence, not addition)
Absorbs over time: wyckoff_phase_boost (becomes ALIGNED_CONFIRMED rung), breadth/
rotation context (candidate CONFIRMING inputs to eye_dir), the parked HTF ideas
(Bojan∩accumulation as a P3+ rung once live n suffices). The CMI regime system stays
orthogonal (risk plumbing), untouched — no fusion changes anywhere (Lesson #54).

## Honest priors
- Each rung individually is boost-shaped (good prior); the COUNTER rung is not.
- The state machine's transition rules are mechanizations of discretionary reads —
  the ground-truth charts constrain them, but "acceptance" and range-boundary choice
  carry residual judgment we may mechanize wrong; P2's both-era gate is the guard.
- Junk-book authority is a user decision with a real trade-off (data vs coherence).
