import os

from .semantic_search import ChunkedSemanticSearch
from .inverted_index import InvertedIndex
from .constants import (
    DEFAULT_SEARCH_LIMIT, DEFAULT_RRF_SEARCH_LIMIT, SCORE_PRECISION
)

def normalize(scores: list):
    if len(scores) == 0:
        return
    normalized_scores = []
    min_score = min(scores)
    max_score = max(scores)
    for score in scores:
        if min_score != max_score:
            normalized_scores.append((score - min_score)/(max_score - min_score))
        else:
            normalized_scores.append(1.0)
    for n_score in normalized_scores:
        print(f"* {round(n_score, SCORE_PRECISION)}")

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
        raise NotImplementedError("Haven't done weighted search yet!")
    
    def rrf_search(self, query, k, limit=DEFAULT_RRF_SEARCH_LIMIT):
        raise NotImplementedError("Haven't done RRF search yet. What could that be?")