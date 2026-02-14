#!/usr/bin/env python3

import argparse

from lib.semantic_search import verify_model, embed_text, verify_embeddings


#be wary when running the boot.dev cli on these; results may not show success. Run tests by hand to verify.

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser("verify", help="Verifies that the LLM has been loaded.")

    embed_text_parser = subparsers.add_parser("embed_text", help="Embeds text or something idk")
    embed_text_parser.add_argument("text", type=str, help="Text to be embedded.")

    verify_embeddings_parser = subparsers.add_parser("verify_embeddings", help="Verifies the cached list of embeddings.")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()