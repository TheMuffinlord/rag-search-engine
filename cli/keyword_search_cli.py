#!/usr/bin/env python3

import argparse, json

def search(terms) -> None:
    with open('data/movies.json', 'r') as f:
        movieList = json.load(f)
    matchCount = 0
    for movie in movieList["movies"]:
        if terms in movie["title"]:
            matchCount += 1
            print(f"{matchCount}. {movie['title']}")
        if matchCount >= 5:
            return
    if matchCount == 0:
        print(f"No results found.\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}\n")
            search(args.query)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()