#!/usr/bin/env python3

# TO DO: SPLIT ALL THIS SHIT INTO REASONABLE FILES.

import argparse

from word_actions import *
from inverted_index import InvertedIndex

DEFAULT_SEARCH_LIMIT = 5



def search(movieDB: InvertedIndex, terms: str, limit: int = DEFAULT_SEARCH_LIMIT):
    try:
        movieDB.load()  
        '''matchList = []
        for movie in movieList:
            query_tokens = separator(terms)
            title_tokens = separator(movie["title"])
            if match_tokens(query_tokens, title_tokens):
                matchList.append(movie)
                if len(matchList) >= limit:
                    break
        return matchList'''
    except Exception as e:
        print(f'whoops! try running the build command first. Returned error {e}')
    else:
        matchList = []
        queryTokens = separator(terms)
        for token in queryTokens:
            print(f'searching: {token}')
            matchDocs = movieDB.get_documents(token)
            for match in matchDocs:
                matchList.append(movieDB.docmap[match])
                if len(matchList) >= limit:
                    break
        return matchList



# the solution has the results being sent to its own function to display. probably smart. do that.

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Builds an index of available movies")
    
    movieDB = InvertedIndex()

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}\n")
            results = search(movieDB, args.query)
            for i, result in enumerate(results, 1):
                print(f"{i}. {result['title']}")
        case "build":
            print("Attempting to build a movie database...")
            movieDB.build()
            print("Movie database built. Attempting to save...")
            movieDB.save()
            print("Index saved!")
            #print(f"First document for token 'merida' = {movieDB.get_documents('merida')}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()