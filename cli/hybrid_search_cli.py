import argparse

from lib.hybrid_search import normalize_cmd, weighted_search_cmd
from lib.constants import (
    DEFAULT_SEARCH_LIMIT, DEFAULT_ALPHA_BLEND
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="Normalizes a list of numbers.")
    normalize_parser.add_argument("numbers", type=float, nargs="+", help="list of numbers to normalize.")

    weighted_search_parser = subparsers.add_parser("weighted-search", help="Performs a weighted search on a query. Hopefully.")
    weighted_search_parser.add_argument('query', type=str, help="Query on which to search.")
    weighted_search_parser.add_argument('--alpha', type=float, nargs="?", default=DEFAULT_ALPHA_BLEND, help=f"Alpha blending value. Default {DEFAULT_ALPHA_BLEND}.")
    weighted_search_parser.add_argument('--limit', type=int, nargs="?", default=DEFAULT_SEARCH_LIMIT, help=f"Limit of search results. Default {DEFAULT_SEARCH_LIMIT}.")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalize_cmd(args.numbers)
        case "weighted-search":
            weighted_search_cmd(args.query, args.alpha, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()