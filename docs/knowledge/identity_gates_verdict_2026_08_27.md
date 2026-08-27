# Identity-Gate Restoration — verdicts (2026-08-27)

Premise (rally forensic): archetypes with binding structural gates win to-spec; soft/inert/bypassed gates produce "beta in costume". Three pre-registered A/Bs vs the package baseline ($304,487.82):

**1. OBR gate_mode soft→hard: ACCEPT.** Archetype −$3,770/WR25%/n44 → **+$1,131/WR38%/n91**. Book +$1,610, DD −15.8→−15.5. MECHANISM (key insight): soft mode's fusion penalty made OBR LOSE dedup slots even on decent bars; hard mode blocks fakes outright and lets genuine setups compete at full fusion → MORE trades, all to-spec. Hardening an identity gate can INCREASE interpretable data — no junk-book tension.

**2. OBR boms_strength floor 0.0→0.3: INERT (dead config).** Byte-identical results. The thresholds-key is not consumed by the gating path. Do not treat it as a knob (audit-#12 family).

**3. CB enforce_gates_under_bypass true: REJECT.** CB 221→26 trades, +$9,807→−$960; book −$7,466 (2023 −$3.5K, 2024 −$7.4K). Confirms repair-package §5 at scale: CB's written gates do not describe its winners (which are bypass-mode flush-buys at range lows, e.g. its best live entry: RSI 28 at range bottom). Not a gate-strength problem — a WRONG-IDENTITY problem. Future study (separate, not queued): re-spec CB's gates from its actual winner profile.

**Meta-lesson**: "restore identity gates" is not one lever. OBR = right gates, no teeth → harden (works). CB = teeth available, wrong gates → enforcement kills it. Judge per archetype: do the gates describe the archetype's actual edge?

Shipped change (pending user approval): OBR gate_mode hard in both config dirs.
