#!/usr/bin/env python3

import argparse, json, string

DEFAULT_SEARCH_LIMIT = 5

def stripper(term: str):
    return term.translate(str.maketrans('','', string.punctuation)).lower()

def separator(term: str):
    term = stripper(term)
    terms = term.split()
    valid_terms = []
    for t in terms:
        if t:
            valid_terms.append(t)
    return valid_terms

def load_movies():
    with open('data/movies.json', 'r') as f:
        movieList = json.load(f) 
    return movieList["movies"]

def search(terms: str, limit: int = DEFAULT_SEARCH_LIMIT):
    movieList = load_movies()  
    matchList = []
    for movie in movieList:
        query_tokens = separator(terms)
        title_tokens = separator(movie["title"])
        if match_tokens(query_tokens, title_tokens):
            matchList.append(movie)
            if len(matchList) >= limit:
                break
    return matchList

def match_tokens(query_tokens, title_tokens):
    for query in query_tokens:
        for title in title_tokens:
            if query in title:
                return True
    return False

# the solution has the results being sent to its own function to display. probably smart. do that.

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}\n")
            results = search(args.query)
            for i, result in enumerate(results, 1):
                print(f"{i}. {result['title']}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()