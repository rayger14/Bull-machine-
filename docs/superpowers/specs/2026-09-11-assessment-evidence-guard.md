# Future assessment evidence guard

Quant guide approved this no-inference follow-up to the four-case teaching
transfer probe. It fixes research plumbing, not a trading strategy.

- Standard-library-only pure functions in a new research module; no model calls,
  file/network I/O, trading integration or changes to existing frozen artifacts.
- Input is exactly one JSON object with a nonempty string `case_id` and a `plan`
  containing finite positive `indicative_close`, `stop`, `notional`, and finite
  nonnegative `roundtrip_cost`; stop must be strictly below indicative close.
  Reject bools as numbers and reject nonfinite/unrepresentable derived values.
- Canonical JSON: sorted keys, compact separators, ensure_ascii=True, allow_nan=False.
  Split its ASCII text into chunks of at most4096bytes by default, with caller
  configurable positive integer bound. Envelope records case ID, canonical
  packet SHA256/byte count, sorted top-level section inventory, ordered chunks
  with index/text/byte count/SHA256, and deterministic indicative economics.
- Economics: quantity=notional/indicative_close; risk=(indicative_close-stop)*quantity;
  target=indicative_close+2*(indicative_close-stop); cost_R=roundtrip_cost/risk;
  breakeven=indicative_close*(1+roundtrip_cost/notional); ordinary net stop=-risk-cost;
  net target=2*risk-cost. Label indicative, actual fill unknown, no execution authority.
- Delivery validation consumes the trusted original packet/envelope and ordered
  caller-supplied records with exactly index,text,truncated. Rebuild/verify the
  expected envelope; reject missing/extra/reordered/duplicated/changed chunks,
  nonboolean or true truncation status, mismatched packet/envelope/sections and
  cross-case receipt. Require explicit expected case_id. Return valid/errors,
  execution_authorized=False and authority=caller_attested_transport_only.
  Exact supplied text agreement does NOT prove actual tool rendering, model
  attention/comprehension, source authenticity or future-data cleanliness.
- Exact evidence locator is a typed path list (string dictionary keys or integer
  list indices, excluding bool/negative indices) starting at one top-level section.
  Resolve against the original packet and return a detached value; wrong-source,
  missing-key, invalid-index, empty-path and wrong-container lookups raise ValueError.
  Existence proves neither numerical claim equality nor natural-language entailment.
- Future factual reviewers each receive one case only, without other case histories
  or outcomes, until all reviews lock. This is a workflow requirement, not something
  the pure helper can certify. Current eight responses remain immutable; no retries.

Transport capture/integration and actual model-delivery smoke tests are explicitly
not implemented in this step. Existing packets may be used only as local no-model
roundtrip fixtures; they are not new independent assessment evidence.
