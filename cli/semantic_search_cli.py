#!/usr/bin/env python3

import argparse

from lib.semantic_search import verify_model, embed_text, verify_embeddings, embed_query_text, semantic_search


def chunk_command(text:str, size=200):
    words = text.split()
    all_chunks = [words[i:i + size] for i in range(0, len(words), size)]
    r_chunks = []
    for chunk in all_chunks:
        chunked_line = " ".join(chunk)
        r_chunks.append(chunked_line)
    print(f"Chunking {len(text)} characters")
    for n, chunk in enumerate(r_chunks):
        print(f"{n+1}. {chunk}")




    


#be wary when running the boot.dev cli on these; results may not show success. Run tests by hand to verify.

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser("verify", help="Verifies that the LLM has been loaded.")

    embed_text_parser = subparsers.add_parser("embed_text", help="Embeds text or something idk")
    embed_text_parser.add_argument("text", type=str, help="Text to be embedded.")

    verify_embeddings_parser = subparsers.add_parser("verify_embeddings", help="Verifies the cached list of embeddings.")

    ebq_parser = subparsers.add_parser("embedquery", help="Converts a search query into embedding vectors.")
    ebq_parser.add_argument("query", type=str, help="Query to embed.")

    search_parser = subparsers.add_parser("search", help="Runs a semantic search on a query.")
    search_parser.add_argument("query", type=str, help="Query to run a search against.")
    search_parser.add_argument("--limit", type=int, nargs='?', default=5, help="Limit of results to return")

    chunk_parser = subparsers.add_parser("chunk", help="Chunk text")
    chunk_parser.add_argument("text", type=str, help="Text to chunk.")
    chunk_parser.add_argument("--chunk-size", type=int, nargs="?", default=200, help="Limit of text to chunk.")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embedquery":
            embed_query_text(args.query)
        case "search":
            semantic_search(args.query, args.limit)
        case "chunk":
            chunk_command(args.text, args.chunk_size)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()