# Pre-outcome scoring clarification — September 11, 2026

Independent scorer review identified an ambiguity in the original protocol's
unqualified sentence about a bar touching both barriers. This clarification is
recorded while the single assessor is in flight and before any outcome reveal.
The original frozen protocol, scorer, probe, packets and baseline masks remain
unchanged, including their hashes.

Known bar-opening prices take precedence over unknowable later intrabar order.
An open at/below the stop fills at that open. An open at/above the target fills
conservatively at the target, even if the same bar's subsequent low crosses the
stop. Only when the open lies between the barriers does a both-hit OHLC bar
receive the ambiguous flag and conservative stop-first treatment.

This describes the already-frozen scorer and its pre-assessment passing test
`test_open_target_gap_precedes_later_same_bar_stop`. It is not an outcome-driven
change to the scoring code or to the agent's fixed management contract. The
report must disclose this clarification and whether any scored case used it.
