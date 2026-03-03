import os

from .semantic_search import ChunkedSemanticSearch
from .inverted_index import InvertedIndex
from .word_actions import load_movies, format_search_result
from .constants import (
    DEFAULT_SEARCH_LIMIT, DEFAULT_RRF_SEARCH_LIMIT, SCORE_PRECISION, DATA_PATH, RETURN_DOCUMENT_LIMIT,
    DEFAULT_LARGE_SEARCH_MULTIPLIER
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

def hybrid_score(bm25_score, semantic_score, alpha=0.5):
    return alpha * bm25_score + (1 - alpha) * semantic_score

def weighted_search_cmd(query, alpha = 0.5, limit = DEFAULT_SEARCH_LIMIT):
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
        raise NotImplementedError("Haven't done RRF search yet. What could that be?")