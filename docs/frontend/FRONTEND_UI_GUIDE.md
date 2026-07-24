# Bull Machine Frontend UI Guide

This UI is a strategy workbench for understanding how the trading system makes entry decisions.

## What is improved in v2

- Reads config JSON directly from this repository (via the static server) so thresholds/weights can mirror real settings.
- Adds scenario presets (balanced, macro veto, low confluence, risk-on).
- Shows a decision trace so users can see **which gate passed/failed** in order.

## What this frontend models

- Domain aggregation and consensus from the fusion flow in `engine/fusion.py`.
- Minimum domain requirements and confidence/strength gate checks.
- Macro suppression as a hard veto.
- Blocking veto severity checks (`>= 0.8`) to prevent trade entry.

## Run locally

```bash
python -m http.server 8080
```

Then open: `http://localhost:8080/ui/trader_dashboard.html`

## Notes

This remains intentionally lightweight (HTML/CSS/JS) so it runs in any environment with no additional dependencies.
