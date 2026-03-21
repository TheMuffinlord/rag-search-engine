import argparse

from lib.hybrid_search import normalize_cmd, weighted_search_cmd, rrf_search_cmd
from lib.constants import (
    DEFAULT_SEARCH_LIMIT, DEFAULT_ALPHA_BLEND, DEFAULT_RRF_K, DEFAULT_RRF_SEARCH_LIMIT
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

    rrf_search_parser = subparsers.add_parser("rrf-search", help="Performes a search using reciprocal rank fusion.")
    rrf_search_parser.add_argument('query', type=str, help="Query to search.")
    rrf_search_parser.add_argument('--k', type=int, default=DEFAULT_RRF_K, help=f"K-value to use when searching. Default {DEFAULT_RRF_K}.")
    rrf_search_parser.add_argument('--limit', type=int, default=DEFAULT_RRF_SEARCH_LIMIT, help=f"Same limit argument as always. Default {DEFAULT_RRF_SEARCH_LIMIT}.")
    rrf_search_parser.add_argument('--enhance', type=str, choices=['spell', 'rewrite', 'expand'], required=False, help="Processes your search input using an LLM. Available methods: spell")
    rrf_search_parser.add_argument('--rerank-method', type=str, choices=['individual', 'batch', 'cross_encoder'], required=False, help="Defines an optional reranking method.")
    rrf_search_parser.add_argument('--debug', type=bool, default=False, help="Enables debug output.")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalize_cmd(args.numbers)
        case "weighted-search":
            weighted_search_cmd(args.query, args.alpha, args.limit)
        case "rrf-search":
            rrf_search_cmd(args.query, args.k, args.limit, args.enhance, args.rerank_method, args.debug)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()