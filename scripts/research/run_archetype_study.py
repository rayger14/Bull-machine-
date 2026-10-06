"""Explicit staged entry point; expensive/economic stages default locked."""
import argparse
import json
import sys

from scripts.research.study_source import build_pilot, preflight


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='stage', required=True)
    commands.add_parser('preflight')
    pilot = commands.add_parser('pilot')
    pilot.add_argument('--output-dir', required=True)
    commands.add_parser('census')
    commands.add_parser('score')
    args = parser.parse_args(argv)
    if args.stage in ('census', 'score'):
        print('Locked: continuous-source, resource and integrated quant review receipts required.', file=sys.stderr)
        return 2
    if args.stage == 'preflight':
        result = preflight()
        print(json.dumps(result, sort_keys=True, indent=2))
        return 0 if result['source_ready'] else 2
    result = build_pilot(args.output_dir)
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
