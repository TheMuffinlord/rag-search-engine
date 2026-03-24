import argparse

from lib.augmented_generation import rag_cmd, summarize_cmd

def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    sum_parser = subparsers.add_parser('summarize', help="make the ai summarize the search results bc you're a mush brain")
    sum_parser.add_argument('query', type=str, help='query to needful')
    sum_parser.add_argument('--limit', type=int, default=5, help='optional limit parameter')

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            rag_cmd(query)
        case "summarize":
            query = args.query
            limit = args.limit
            summarize_cmd(query, limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()