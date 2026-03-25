from dotenv import load_dotenv
from google import genai

from .word_actions import format_search_result, load_movies

from .hybrid_search import HybridSearch

from .constants import DEFAULT_SEARCH_LIMIT

import os

def llm_action(prompt):
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")
    client = genai.Client(api_key=api_key)
    return client.models.generate_content(model="gemma-3-27b-it", contents=prompt)

def initial_rrf(query, limit=DEFAULT_SEARCH_LIMIT):
    search_docs = load_movies()
    hs = HybridSearch(search_docs)
    rrf_results = hs.rrf_search(query, limit=limit)
    #print(f"DEBUG: RRF Results: {rrf_results[:limit]}")
    return rrf_results

def rrf_joiner(rrf_results, limit=DEFAULT_SEARCH_LIMIT):
    docs_list = []
    for result in rrf_results[:limit]:
        docs_list.append(f"{result['title']} - {result['document'][:200]}\n")
    return "\n".join(docs_list)

def rag_cmd(query):
    rrf_results = initial_rrf(query)
    rag_string = rrf_joiner(rrf_results)
    
    prompt = f"""Answer the question or provide information based on the provided documents. This should be tailored to Hoopla users. Hoopla is a movie streaming service.

        Query: {query}

        Documents:
        {rag_string}

        Provide a comprehensive answer that addresses the query:"""
    response = llm_action(prompt)
    rag_answer = (response.text or "").strip()
    
    print('Search results:')
    for result in rrf_results[:DEFAULT_SEARCH_LIMIT]:
        print(f'  - {result["title"]}')
    print()
    print('RAG Response:')
    print(f'"{rag_answer}"')

def summarize_cmd(query, limit=DEFAULT_SEARCH_LIMIT):
    rrf_results = initial_rrf(query, limit)
    sum_string = rrf_joiner(rrf_results, limit)

    prompt = f"""Provide information useful to the query below by synthesizing data from multiple search results in detail.

        The goal is to provide comprehensive information so that users know what their options are.
        Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.

        This should be tailored to Hoopla users. Hoopla is a movie streaming service.

        Query: {query}

        Search results:
        {sum_string}

        Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:"""
    response = llm_action(prompt)
    sum_answer = (response.text or "").strip()
    
    print('Search results:')
    for result in rrf_results[:DEFAULT_SEARCH_LIMIT]:
        print(f'  - {result["title"]}')
    print()
    print('LLM Summary:')
    print(f'"{sum_answer}"')

def citations_cmd(query, limit=DEFAULT_SEARCH_LIMIT):
    rrf_results = initial_rrf(query, limit)
    cite_string = rrf_joiner(rrf_results, limit)

    prompt = f"""Answer the query below and give information based on the provided documents.

    The answer should be tailored to users of Hoopla, a movie streaming service.
    If not enough information is available to provide a good answer, say so, but give the best answer possible while citing the sources available.

    Query: {query}

    Documents:
    {cite_string}

    Instructions:
    - Provide a comprehensive answer that addresses the query
    - Cite sources in the format [1], [2], etc. when referencing information
    - If sources disagree, mention the different viewpoints
    - If the answer isn't in the provided documents, say "I don't have enough information"
    - Be direct and informative

    Answer:"""
    #print(f"DEBUG: PROMPT: \n\n{prompt}\n\n")
    response = llm_action(prompt)
    cite_answer = (response.text or "").strip()

    print('Search results:')
    for result in rrf_results[:DEFAULT_SEARCH_LIMIT]:
        print(f'  - {result["title"]}')
    print()
    print('LLM Answer:')
    print(f'"{cite_answer}"')