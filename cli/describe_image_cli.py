import argparse, mimetypes, os, types

from dotenv import load_dotenv
from google import genai


def main():

    parser = argparse.ArgumentParser(description="CLI that rewrites a query based on an image and a prompt.")
    parser.add_argument("--image", type=str, help="Path to the image to describe.")
    parser.add_argument("--query", type=str, help="Query to search on.")

    args = parser.parse_args()
    image = args.image
    query = args.query
    mime, _ = mimetypes.guess_type(image)
    mime = mime or "image/jpeg"

    with open(image, 'rb') as f:
        image_data = f.read()

    prompt = """Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
        - Synthesize visual and textual information
        - Focus on movie-specific details (actors, scenes, style, etc.)
        - Return only the rewritten query, without any additional commentary"""
    
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")
    client = genai.Client(api_key=api_key)

    parts = [
        prompt, 
        genai.types.Part.from_bytes(data=image_data, mime_type=mime),
        query.strip(),
    ]

    response = client.models.generate_content(model="gemma-3-27b-it", contents=parts)

    print(f"Rewritten query: {response.text.strip()}")
    if response.usage_metadata is not None:
        print(f"Total tokens:    {response.usage_metadata.total_token_count}")

if __name__ == "__main__":
    main()