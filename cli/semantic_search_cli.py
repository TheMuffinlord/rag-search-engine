#!/usr/bin/env python3

import argparse

from lib.semantic_search import (verify_model, embed_text, verify_embeddings, embed_query_text, 
                                 semantic_search, chunk_command, semantic_chunk_cmd, embed_chunks, search_chunked_cmd)
from cli.lib.constants import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, DEFAULT_SEMANTIC_CHUNK_SIZE

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
    chunk_parser.add_argument("--chunk-size", type=int, nargs="?", default=DEFAULT_CHUNK_SIZE, help=f"Limit of text to chunk, default {DEFAULT_CHUNK_SIZE}")
    chunk_parser.add_argument("--overlap", type=int, nargs="?", default=DEFAULT_CHUNK_OVERLAP, help=f"Amount of chunk overlap, default {DEFAULT_CHUNK_OVERLAP}")

    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="Semantically chunks text into sentences.")
    semantic_chunk_parser.add_argument("text", type=str, help="Text to chunk.")
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, nargs="?", default=DEFAULT_SEMANTIC_CHUNK_SIZE, help=f"Limit of sentences to chunk, default {DEFAULT_SEMANTIC_CHUNK_SIZE}")
    semantic_chunk_parser.add_argument("--overlap", type=int, nargs="?", default=DEFAULT_CHUNK_OVERLAP, help=f"Amount of chunk overlap, default {DEFAULT_CHUNK_OVERLAP}")

    embed_chunk_parser = subparsers.add_parser("embed_chunks", help="Builds chunked embeddings.")

    search_chunk_parser = subparsers.add_parser("search_chunked", help="Searches the chunked whatever i'm tired of writing these")
    search_chunk_parser.add_argument('text', type=str, help='text to search.')
    search_chunk_parser.add_argument('--limit', type=int, nargs='?', default=5, help='limit of results. default 5')

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
            chunk_command(args.text, args.chunk_size, args.overlap)
        case "semantic_chunk":
            semantic_chunk_cmd(args.text, args.max_chunk_size, args.overlap)
        case "embed_chunks":
            embed_chunks()
        case "search_chunked":
            search_chunked_cmd(args.text, args.limit)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()