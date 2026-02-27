#!/usr/bin/env python3

# TO DO: SPLIT ALL THIS SHIT INTO REASONABLE FILES.

import argparse, math

from cli.lib.word_actions import *
from cli.lib.inverted_index import InvertedIndex
from cli.lib.constants import BM25_K1, BM25_B

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
    term = separator(term)
    if len(term) > 1:
        print(f"multiple terms; only the first one will be counted.")
    doc_count = len(movieDB.docmap)
    match_count = len(movieDB.index[term[0]])
    return math.log((doc_count + 1) / (match_count + 1))
    
def movieDB_loader(movieDB: InvertedIndex): # this solves a problem but i don't actually care to clean up this code right now or maybe ever?
    try:
        movieDB.load()
    except Exception as e:
        print(f"Whoops! Got an error: {e}")
    else:
        return movieDB

def bm25_tf_command(movieDB: InvertedIndex, doc_id: int, term: str, k1=BM25_K1, b=BM25_B):
    movieDB = movieDB_loader(movieDB)
    return movieDB.get_bm25_tf(doc_id, term, k1, b)
    

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

    tfidf_parser = subparsers.add_parser("tfidf", help="Calculates the TF-IDF (look it up) of a specific term in a specific document.")
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID to calculate for.")
    tfidf_parser.add_argument("term", type=str, help="The term to calculate. One word only please.")

    bm25idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")  

    bm25_tf_parser = subparsers.add_parser("bm25tf", help="Get BM25 TF score for a given document ID and term")
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs='?', default=BM25_K1, help="Tunable BM25 K1 parameter")
    bm25_tf_parser.add_argument("b", type=float, nargs="?", default=BM25_B, help="Tunable BM25 b parameter")

    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument("limit", type=int, nargs='?', default=5, help="Limit of results to return")

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
            movieDB = movieDB_loader(movieDB)
            print(f"Term frequency for {args.term} in document {args.doc_id}\n")
            result = movieDB.get_tf(args.doc_id, args.term)
            print(f"The term '{args.term}' appears {result} time(s).\n")
        case "idf":
            movieDB = movieDB_loader(movieDB)
            idf_count = idf(movieDB, args.term)
            print(f"Inverse document frequency of {args.term}: {idf_count:.2f}")
        case "tfidf":
            try:
                movieDB.load()
            except Exception as e:
                print(f"Whoops! Got an error: {e}")
            else:
                tf_count = movieDB.get_tf(args.doc_id, args.term)
                idf_count = idf(movieDB, args.term)
                tf_idf = tf_count * idf_count
                print(f"TF-IDF score of '{args.term}' in '{args.doc_id}': {tf_idf:.2f}")
        case "bm25idf":
            try:
                movieDB.load()
            except Exception as e:
                print(f"Whoops! Got an error: {e}")
            else:
                bm25idf = movieDB.get_bm25_idf(args.term)
                print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
        case "bm25tf":
            bm25tf = bm25_tf_command(movieDB, args.doc_id, args.term, args.k1)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")
        case "bm25search":
            movieDB = movieDB_loader(movieDB)
            results = movieDB.bm25_search(args.query, args.limit)
            #print(results)
            print(f"Searching for terms '{args.query}':")
            i = 0
            for result, score in results.items():
                i+=1
                movie = movieDB.docmap[result]
                print(f"{i}. ({movie['id']}) {movie['title']} - Score: {score:.2f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()