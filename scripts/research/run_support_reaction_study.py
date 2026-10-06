"""One bounded support-reaction source/semantic export; no economic launch."""
import argparse
import json

from scripts.research.support_reaction_study import run_source


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['source'])
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run_source(args.output), sort_keys=True))


if __name__ == '__main__':
    main()
