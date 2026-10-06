"""Bounded local LC experiment; stages never submit orders or call market models."""
import argparse
import json

from scripts.research.lc_context_source import load_inputs, prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['preflight', 'prepare', 'score'])
    parser.add_argument('--output')
    parser.add_argument('--source')
    parser.add_argument('--review')
    args = parser.parse_args()
    if args.stage == 'preflight':
        files, sources, parents, expected, raw = load_inputs()
        result = {'case_count': len(raw), 'source_months': len(sources), 'bound_files': len(files),
                  'parent_timeframes': sorted(parents), 'execution_authorized': False}
    else:
        if not args.output:
            parser.error('--output is required')
        if args.stage == 'score':
            if not args.source or not args.review:
                parser.error('--source and --review are required for score')
            from scripts.research.lc_context_study import score
            result = score(args.source, args.output, args.review)
        else:
            result = prepare(args.output)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
