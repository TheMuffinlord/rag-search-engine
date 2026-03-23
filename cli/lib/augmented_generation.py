from dotenv import load_dotenv
from google import genai

from .word_actions import format_search_result, load_movies

from .hybrid_search import HybridSearch

from .constants import DEFAULT_SEARCH_LIMIT

import os

def rag_cmd(query):
    search_docs = load_movies()
    hs = HybridSearch(search_docs)
    rrf_results = hs.rrf_search(query, limit=DEFAULT_SEARCH_LIMIT)
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")
    client = genai.Client(api_key=api_key)
    docs_list = []
    for result in rrf_results[:DEFAULT_SEARCH_LIMIT]:
        docs_list.append(f"{result['title']} - {result['document'][:200]}\n")
    rag_string = "\n".join(docs_list)

    prompt = f"""Answer the question or provide information based on the provided documents. This should be tailored to Hoopla users. Hoopla is a movie streaming service.

        Query: {query}

        Documents:
        {rag_string}

        Provide a comprehensive answer that addresses the query:"""
    response = client.models.generate_content(model="gemma-3-27b-it", contents=prompt)
    rag_answer = (response.text or "").strip()
    
    print('Search results:')
    for result in rrf_results[:DEFAULT_SEARCH_LIMIT]:
        print(f'  - {result["title"]}')
    print()
    print('RAG Response:')
    print(f'"{rag_answer}"')