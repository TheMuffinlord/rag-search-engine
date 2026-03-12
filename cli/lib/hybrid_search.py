import os, time, json

from dotenv import load_dotenv
from google import genai
from google.genai import types


from .semantic_search import ChunkedSemanticSearch
from .inverted_index import InvertedIndex
from .word_actions import load_movies, format_search_result
from .constants import (
    DEFAULT_SEARCH_LIMIT, DEFAULT_RRF_SEARCH_LIMIT, SCORE_PRECISION, DATA_PATH, RETURN_DOCUMENT_LIMIT,
    DEFAULT_LARGE_SEARCH_MULTIPLIER, DEFAULT_ALPHA_BLEND, DEFAULT_RRF_K
)

def normalize(scores: list):
    if len(scores) == 0:
        return []
    normalized_scores = []
    min_score = min(scores)
    max_score = max(scores)
    for score in scores:
        if min_score != max_score:
            normalized_scores.append((score - min_score)/(max_score - min_score))
        else:
            normalized_scores.append(1.0)
    return normalized_scores

def normalize_search_results(results: list):
    scores = []
    for result in results:
        scores.append(result['score'])
    
    normalized_scores = normalize(scores)
    for i, result in enumerate(results):
        result['normalized_score'] = normalized_scores[i]
    
    return results


def normalize_cmd(scores: list):
    normalized_scores = normalize(scores)
    for n_score in normalized_scores:
        print(f"* {round(n_score, SCORE_PRECISION)}")

def hybrid_score(bm25_score, semantic_score, alpha=DEFAULT_ALPHA_BLEND):
    return alpha * bm25_score + (1 - alpha) * semantic_score

def rrf_score(rank, k=DEFAULT_RRF_K):
    return 1 / (k + rank)

def weighted_search_cmd(query, alpha = DEFAULT_ALPHA_BLEND, limit = DEFAULT_SEARCH_LIMIT):
    documents = load_movies()
    hybrid_search = HybridSearch(documents)
    weighted_results = hybrid_search.weighted_search(query, alpha, limit)
    print(f"Displaying {limit} results:")
    for i, result in enumerate(weighted_results):
        #print(result)
        print(f"{i+1}. {result['title']}, ID: {result['id']}")
        print(f"   Hybrid Score: {result['score']:.4f}")
        print(f"   BM25: {result['metadata']['bm25_score']:.4f}, Semantic: {result['metadata']['semantic_score']:.4f}")
        print(f"   {result['document'][:RETURN_DOCUMENT_LIMIT]}")

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
    


def rrf_search_cmd(query, k = DEFAULT_RRF_K, limit = DEFAULT_RRF_SEARCH_LIMIT, enhance=None, rerank=None):
    documents = load_movies()
    hybrid_search = HybridSearch(documents)
    match enhance:
        case "spell":
            query = spellcheck_module(query)
        case "rewrite":
            query = rewrite_module(query)
        case "expand":
            query = expand_module(query)
    
    match rerank:
        case "individual":
            rrf_results = hybrid_search.rrf_search(query, k, limit * 5)
            rrf_results = individual_rerank(query, rrf_results)
            print(f"Reranking top {limit} results using individual method:")
            for i, result in enumerate(rrf_results[:limit]):
                print(f"{i+1}. {result[0]['title']}, ID: {result[0]['id']}")
                print(f"   Re-rank score: {result[1]}/{limit}")
                print(f"   RRF Score: {result[0]['score']:.4f}")
                print(f"   BM25 Rank: {result[0]['metadata']['bm25_rank']}, Semantic Rank: {result[0]['metadata']['semantic_rank']}")
                print(f"   {result[0]['document'][:RETURN_DOCUMENT_LIMIT]}")
        case "batch":
            rrf_results = hybrid_search.rrf_search(query, k, limit * 5)
            batch_results = batch_rerank(query, rrf_results)
            print(f"Reranking top {limit} results using batch method:")
            for i, result in enumerate(batch_results[:limit]):
            #for i, result in enumerate(batch_results):
                print(f"{i+1}. {result['title']}, ID: {result['id']}")
                print(f"   Re-rank score: {result['metadata']['rerank_score']+1}/{limit}")
                print(f"   RRF Score: {result['score']:.4f}")
                print(f"   BM25 Rank: {result['metadata']['bm25_rank']}, Semantic Rank: {result['metadata']['semantic_rank']}")
                print(f"   {result['document'][:RETURN_DOCUMENT_LIMIT]}\n")
        case None:
            print(f"Displaying {limit} results:")
            for i, result in enumerate(rrf_results[:limit]):
                #print(result)
                print(f"{i+1}. {result['title']}, ID: {result['id']}")
                print(f"   RRF Score: {result['score']:.4f}")
                print(f"   BM25 Rank: {result['metadata']['bm25_rank']}, Semantic Rank: {result['metadata']['semantic_rank']}")
                print(f"   {result['document'][:RETURN_DOCUMENT_LIMIT]}")


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.inverted_index = InvertedIndex()
        if not os.path.exists(self.inverted_index.index_path):
            self.inverted_index.build()
            self.inverted_index.save()

    def _bm25_search(self, query, limit):
        self.inverted_index.load()
        return self.inverted_index.bm25_search(query, limit)
           
    
    def weighted_search(self, query, alpha, limit=DEFAULT_SEARCH_LIMIT):
        bm25_results = self._bm25_search(query, limit * DEFAULT_LARGE_SEARCH_MULTIPLIER)
        n_bm25_scorelist = normalize_search_results(bm25_results)
        css_results = self.semantic_search.search_chunk(query, limit * DEFAULT_LARGE_SEARCH_MULTIPLIER)
        n_css_scorelist = normalize_search_results(css_results)
        #fucking shit that didn't work get it outta here
        combined_scores = {}

        for result in n_bm25_scorelist:
            doc_id = result['id']
            if doc_id not in combined_scores:
                combined_scores[doc_id] = {
                    'title': result['title'],
                    'document': result['document'],
                    'bm25_score': 0.0,
                    'semantic_score': 0.0,
                }
            if result['normalized_score'] > combined_scores[doc_id]['bm25_score']:
                combined_scores[doc_id]['bm25_score'] = result['normalized_score']
        
        for result in n_css_scorelist:
            doc_id = result['id']
            #gosh you can really see the exact moment i just lifted the goddamn code huh
            if doc_id not in combined_scores:
                combined_scores[doc_id] = {
                    "title": result["title"],
                    "document": result["document"],
                    "bm25_score": 0.0,
                    "semantic_score": 0.0,
                }
            if result["normalized_score"] > combined_scores[doc_id]["semantic_score"]:
                combined_scores[doc_id]["semantic_score"] = result["normalized_score"]

        hybrid_results = []
        for doc_id, data in combined_scores.items():
            score_value = hybrid_score(data["bm25_score"], data["semantic_score"], alpha)
            result = format_search_result(
                doc_id=doc_id,
                title=data["title"],
                document=data["document"],
                score=score_value,
                bm25_score=data["bm25_score"],
                semantic_score=data["semantic_score"],
            )
            hybrid_results.append(result)

        return sorted(hybrid_results, key=lambda x: x["score"], reverse=True)

        
    
    def rrf_search(self, query, k, limit=DEFAULT_RRF_SEARCH_LIMIT):
        bm25_results = self._bm25_search(query, limit * DEFAULT_LARGE_SEARCH_MULTIPLIER)
        css_results = self.semantic_search.search_chunk(query, limit * DEFAULT_LARGE_SEARCH_MULTIPLIER)
        combined_scores = {}

        for rank, result in enumerate(bm25_results):
            doc_id = result['id']
            if doc_id not in combined_scores:
                combined_scores[doc_id] = {
                    'title': result['title'],
                    'document': result['document'],
                    'rrf_score': 0.0,
                    'bm25_rank': None,
                    'semantic_rank': None,
                }
            if combined_scores[doc_id]['bm25_rank'] is None:
                combined_scores[doc_id]['bm25_rank'] = rank
                combined_scores[doc_id]['rrf_score'] += rrf_score(rank, k)
            
        for rank, result in enumerate(css_results):
            doc_id = result['id']
            if doc_id not in combined_scores:
                combined_scores[doc_id] = {
                    'title': result['title'],
                    'document': result['document'],
                    'rrf_score': 0.0,
                    'bm25_rank': None,
                    'semantic_rank': None,
                }
            if combined_scores[doc_id]['semantic_rank'] is None:
                combined_scores[doc_id]['semantic_rank'] = rank
                combined_scores[doc_id]['rrf_score'] += rrf_score(rank, k)
        
        rrf_results = []
        for doc_id, data in combined_scores.items():
            result = format_search_result(doc_id, data['title'], data['document'], 
                                          data['rrf_score'], rrf_score=data['rrf_score'], 
                                          bm25_rank=data['bm25_rank'], semantic_rank=data['semantic_rank'])
            rrf_results.append(result)
        
        return sorted(rrf_results, key=lambda x: x["score"], reverse=True)
    
