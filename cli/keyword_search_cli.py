#!/usr/bin/env python3

# TO DO: SPLIT ALL THIS SHIT INTO REASONABLE FILES.

import argparse, math

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

def idf(movieDB: InvertedIndex, term: str):
    
    doc_count = len(movieDB.docmap)
    match_count = len(movieDB.index[term])
    return math.log((doc_count + 1) / (match_count + 1))
    


# the solution has the results being sent to its own function to display. probably smart. do that.

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Builds an index of available movies")
    
    tf_parser = subparsers.add_parser("tf", help="Returns the frequency of the requested term in a specific document ID.")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Search term. One word only please.")

    idf_parser = subparsers.add_parser("idf", help="Calculates the inverse document frequency(IDF) of a given term across all documents.")
    idf_parser.add_argument("term", help="The term to calculate. One word only please.")

    movieDB = InvertedIndex()

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}\n")
            results = search(movieDB, args.query)
            for i, result in enumerate(results, 1):
                print(f"{i}. {result['title']}")
        case "build":
            print("Attempting to build a movie database. This may take some time.")
            movieDB.build()
            print("Movie database built. Attempting to save...")
            movieDB.save()
            print("Index saved!")
            #print(f"First document for token 'merida' = {movieDB.get_documents('merida')}")
        case "tf":
            try:
                movieDB.load()
            except Exception as e:
                print(f"Whoops! Got an error: {e}")
            else:
                print(f"Term frequency for {args.term} in document {args.doc_id}\n")
                result = movieDB.get_tf(args.doc_id, args.term)
                print(f"The term '{args.term}' appears {result} time(s).\n")
        case "idf":
            try:
                movieDB.load()
            except Exception as e:
                print(f"Whoops! Got an error: {e}")
            else:
                term = separator(args.term)
                if len(term) > 1:
                    print(f"multiple terms; only the first one will be counted.")
                idf_count = idf(movieDB, term[0])
                print(f"Inverse document frequency of {term[0]}: {idf_count:.2f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()