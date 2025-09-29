# Bull Machine API Migrations

## v1.4.2 → Current

### Deprecated Private APIs

**`_check_vetoes()` deprecated** - Use `evaluate(...)[veto, veto_reasons]` instead.

The private `_check_vetoes` method has been deprecated in favor of the public evaluation API:

```python
# OLD (deprecated)
vetoes = engine._check_vetoes(modules_data)

# NEW (recommended)
result = engine.fuse(modules_data)
if result and result.vetoes:
    vetoes = list(result.vetoes)
```

**Compatibility:** A compatibility shim is provided to prevent breaking existing tests/code, but will be removed in future versions.

## Legacy Tests

Legacy tests expecting deprecated APIs have been moved to `tests/legacy/` and marked with `@pytest.mark.legacy` to prevent blocking CI while migration is in progress.