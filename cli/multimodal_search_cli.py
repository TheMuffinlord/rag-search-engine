import argparse

from lib.multimodal_search import verify_image_embedding, image_search_cmd

def main():
    parser = argparse.ArgumentParser(description="Multimodal Search parser")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    VIE_parser = subparsers.add_parser("verify_image_embedding", help="verifies an image embedding.")
    VIE_parser.add_argument("image_path", type=str, help="Path to image file to verify.")

    IS_parser = subparsers.add_parser("image_search", help="searches movie database for a result based on a given image.")
    IS_parser.add_argument("image_path", type=str, help="path for the image.")

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            image_path = args.image_path
            verify_image_embedding(image_path)
        case "image_search":
            image_path = args.image_path
            image_search_cmd(image_path)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()