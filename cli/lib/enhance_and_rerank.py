from dotenv import load_dotenv
from google import genai
from google.genai import types
from sentence_transformers import CrossEncoder

import os, time, json

from .word_actions import load_movies, format_search_result
from .constants import (
    DEFAULT_SEARCH_LIMIT, DEFAULT_RRF_SEARCH_LIMIT, SCORE_PRECISION, DATA_PATH, RETURN_DOCUMENT_LIMIT,
    DEFAULT_LARGE_SEARCH_MULTIPLIER, DEFAULT_ALPHA_BLEND, DEFAULT_RRF_K
)


def spellcheck_module(query):
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model="gemma-3-27b-it", 
        contents=f"""Fix any spelling errors in the user-provided movie search query below.
        Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
        Preserve punctuation and capitalization unless a change is required for a typo fix.
        If there are no spelling errors, or if you're unsure, output the original query unchanged.
        Output only the final query text, nothing else.
        User query: "{query}"
        """)
    output = response.text
    print(f"Enhanced query (spell): '{query}' -> '{output}'")
    return output

def rewrite_module(query):
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model="gemma-3-27b-it", contents=f"""Rewrite the user-provided movie search query below to be more specific and searchable.

        Consider:
        - Common movie knowledge (famous actors, popular films)
        - Genre conventions (horror = scary, animation = cartoon)
        - Keep the rewritten query concise (under 10 words)
        - It should be a Google-style search query, specific enough to yield relevant results
        - Don't use boolean logic

        Examples:
        - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
        - "movie about bear in london with marmalade" -> "Paddington London marmalade"
        - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

        If you cannot improve the query, output the original unchanged.
        Output only the rewritten query text, nothing else.

        User query: "{query}"
        """)
    output = response.text
    print(f"Enhanced query (rewrite): '{query}' -> '{output}'")
    return output


def expand_module(query):
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model="gemma-3-27b-it", contents=f"""Expand the user-provided movie search query below with related terms.

        Add synonyms and related concepts that might appear in movie descriptions.
        Keep expansions relevant and focused.
        Output only the additional terms; they will be appended to the original query.

        Examples:
        - "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
        - "action movie with bear" -> "action thriller bear chase fight adventure"
        - "comedy with bear" -> "comedy funny bear humor lighthearted"

        User query: "{query}"
        """)
    output = response.text
    print(f"Enhanced query (expand): '{query}' -> '{output}'")
    return output


def individual_rerank(query, results):
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")
    client = genai.Client(api_key=api_key)
    llm_scores = []
    print("Reranking results. This may take some time.")
    for result in results:
        response = client.models.generate_content(model="gemma-3-27b-it", contents=f"""Rate how well this movie matches the search query.

            Query: "{query}"
            Movie: {result["title"]} - {result['document'][:RETURN_DOCUMENT_LIMIT]}

            Consider:
            - Direct relevance to query
            - User intent (what they're looking for)
            - Content appropriateness

            Rate 0-10 (10 = perfect match).
            Output ONLY the number in your response, no other text or explanation.

            Score:""")
        llm_scores.append((result, response.text))
        time.sleep(3)
    llm_scores_sorted = sorted(llm_scores, key=lambda item: item[1], reverse=True)
    return llm_scores_sorted
        
def batch_rerank(query, results):
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")
    client = genai.Client(api_key=api_key)
    
    print("Reranking results. This may take some time.")
    docs_list = []
    for result in results:
        docs_list.append(f"{result['id']}: {result['title']} - {result['document'][:200]}")
    rerank_string = "\n".join(docs_list)
    #print(f"DEBUG: rerank string: {rerank_string}")
    response = client.models.generate_content(model="gemma-3-27b-it", contents=f"""Rank the movies listed below by relevance to the following search query.

        Query: "{query}"

        Movies:
        {rerank_string}

        Return ONLY the movie IDs in order of relevance (best match first). Return a valid JSON list, nothing else.

        For example:
        [75, 12, 34, 2, 1]

        Ranking:""", 
        )
    #print(f"DEBUG: response text: {response.text}")
    rank_text = (response.text or "").strip()
    llm_rank_list = json.loads(rank_text)
    llm_response = []
    for rank, position in enumerate(llm_rank_list):
        cor_movie = list(filter(lambda item: item['id'] == position, results))[0]
        #print(f"DEBUG: {cor_movie}")
        
        response = format_search_result(
            doc_id=cor_movie['id'],
            title=cor_movie['title'],
            document=cor_movie['document'],
            score=(cor_movie['score'] or None),
            rrf_score=(cor_movie['metadata']['rrf_score'] or None),
            bm25_rank=(cor_movie['metadata']['bm25_rank'] or None),
            semantic_rank=(cor_movie['metadata']['semantic_rank'] or None),
            rerank_score=rank
        )
        llm_response.append(response)
    return sorted(llm_response, key=lambda item: item['metadata']['rerank_score'])

def cross_encoder_rerank(query, results):
    pairs = []
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
    for result in results:
        pairs.append([query, f"{result['title']} - {result['document']}"])
    scores = cross_encoder.predict(pairs)
    cross_response = []
    for rank, score in enumerate(scores):
        cor_movie = results[rank]
        #print(f"DEBUG: {cor_movie}")
        
        response = format_search_result(
            doc_id=cor_movie['id'],
            title=cor_movie['title'],
            document=cor_movie['document'],
            score=(cor_movie['score'] or None),
            rrf_score=(cor_movie['metadata']['rrf_score'] or None),
            bm25_rank=(cor_movie['metadata']['bm25_rank'] or None),
            semantic_rank=(cor_movie['metadata']['semantic_rank'] or None),
            rerank_score=score
        )
        cross_response.append(response)
    return sorted(cross_response, key=lambda item: item['metadata']['rerank_score'], reverse=True)