import argparse

from lib.augmented_generation import rag_cmd, summarize_cmd, citations_cmd
from lib.constants import DEFAULT_SEARCH_LIMIT

def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    sum_parser = subparsers.add_parser('summarize', help="make the ai summarize the search results bc you're a mush brain")
    sum_parser.add_argument('query', type=str, help='query to needful')
    sum_parser.add_argument('--limit', type=int, default=DEFAULT_SEARCH_LIMIT, help='optional limit parameter')

    cite_parser = subparsers.add_parser('citations', help="make the ai summarize the search results WITH CITATIONS bc you're a mush brain deluxe")
    cite_parser.add_argument('query', type=str, help='query to needful')
    cite_parser.add_argument('--limit', type=int, default=DEFAULT_SEARCH_LIMIT, help='optional limit parameter')


    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            rag_cmd(query)
        case "summarize":
            query = args.query
            limit = args.limit
            summarize_cmd(query, limit)
        case "citations":
            query = args.query
            limit = args.limit
            citations_cmd(query, limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()