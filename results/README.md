# Results Directory

Production benchmark results and experiment outputs.

## Structure
- `bench_v2/` - Production v2 benchmarks (keep)
- `bear_patterns/` - Bear market pattern analysis (keep)
- `archive/` - Historical results (keep)

## Guidelines
- Experimental results are gitignored by default
- Only commit production-validated benchmarks
- Use timestamped subdirectories: `YYYYMMDD_experiment_name/`
- Archive old results to `archive/YYYY-QQ/` after 90 days

## Gitignored Patterns
All results are gitignored except for specific production benchmarks.
See `.gitignore` for details.
