"""Locked January engineering launcher. Full-development economics is unavailable."""
import argparse
import json
import signal

from scripts.research.thesis_contract import protocol
from scripts.research.thesis_study import run_engineering, run_source


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['source', 'engineering'])
    parser.add_argument('--output', required=True)
    parser.add_argument('--source')
    parser.add_argument('--review')
    args = parser.parse_args(argv)
    if args.stage == 'engineering' and (not args.source or not args.review):
        parser.error('engineering requires a bound source and quant review')
    if args.stage == 'source' and (args.source or args.review):
        parser.error('source does not consume an economics review')
    def timeout(signum, frame):
        raise TimeoutError('bounded thesis study runtime exceeded')
    previous = signal.signal(signal.SIGALRM, timeout)
    signal.alarm(protocol()['maximum_seconds'])
    try:
        receipt = run_source(args.output) if args.stage == 'source' else run_engineering(args.source, args.output, args.review)
        print(json.dumps(receipt, sort_keys=True))
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


if __name__ == '__main__':
    main()
