#!/usr/bin/env python3

import argparse

from lib.semantic_search import SemanticSearch

def verify_model():
    SemanticSearchModel = SemanticSearch()
    print(f"Model loaded: {SemanticSearchModel.model}")
    print(f"Max sequence length: {SemanticSearchModel.model.max_seq_length}")


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser("verify", help="Verifies that the LLM has been loaded.")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()