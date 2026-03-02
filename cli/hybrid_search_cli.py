import argparse

from lib.hybrid_search import normalize


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="Normalizes a list of numbers.")
    normalize_parser.add_argument("numbers", type=float, nargs="+", help="list of numbers to normalize.")


    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalize(args.numbers)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()