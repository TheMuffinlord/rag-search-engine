import os



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
from .enhance_and_rerank import (
    spellcheck_module, rewrite_module, expand_module,
    batch_rerank, individual_rerank, cross_encoder_rerank,
    evaluate_results
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



    


def rrf_search_cmd(query, k = DEFAULT_RRF_K, limit = DEFAULT_RRF_SEARCH_LIMIT, enhance=None, rerank=None, evaluate=False, debug=False):
    documents = load_movies()
    hybrid_search = HybridSearch(documents)
    match enhance:
        case "spell":
            query = spellcheck_module(query)
        case "rewrite":
            query = rewrite_module(query)
        case "expand":
            query = expand_module(query)
        case None:
            pass
    
    match rerank:
        case "individual":
            rrf_results = hybrid_search.rrf_search(query, k, limit * 5, debug)
            if debug == True:
                print(f"DEBUG: Displaying {limit * 5} pre-enhancement results:")
                for i, result in enumerate(rrf_results[:limit * 5]):
                    print(f"{i+1}. {result['title']}, ID: {result['id']}")
                    print(f"   RRF Score: {result['score']:.4f}")
                    print(f"   BM25 Rank: {result['metadata']['bm25_rank']}, Semantic Rank: {result['metadata']['semantic_rank']}")
                    print(f"   {result['document'][:RETURN_DOCUMENT_LIMIT]}")
            ir_results = individual_rerank(query, rrf_results)
            print(f"Reranking top {limit} results using individual method:")
            for i, result in enumerate(ir_results[:limit]):
                print(f"{i+1}. {result[0]['title']}, ID: {result[0]['id']}")
                print(f"   Re-rank score: {result[1]}/{limit}")
                print(f"   RRF Score: {result[0]['score']:.4f}")
                print(f"   BM25 Rank: {result[0]['metadata']['bm25_rank']}, Semantic Rank: {result[0]['metadata']['semantic_rank']}")
                print(f"   {result[0]['document'][:RETURN_DOCUMENT_LIMIT]}")
            passthrough_results = ir_results
        case "batch":
            rrf_results = hybrid_search.rrf_search(query, k, limit * 5, debug)
            if debug == True:
                print(f"DEBUG: Displaying {limit * 5} pre-enhancement results:")
                for i, result in enumerate(rrf_results[:limit * 5]):
                    print(f"{i+1}. {result['title']}, ID: {result['id']}")
                    print(f"   RRF Score: {result['score']:.4f}")
                    print(f"   BM25 Rank: {result['metadata']['bm25_rank']}, Semantic Rank: {result['metadata']['semantic_rank']}")
                    print(f"   {result['document'][:RETURN_DOCUMENT_LIMIT]}")
            batch_results = batch_rerank(query, rrf_results)
            print(f"Reranking top {limit} results using batch method:")
            for i, result in enumerate(batch_results[:limit]):
            #for i, result in enumerate(batch_results):
                print(f"{i+1}. {result['title']}, ID: {result['id']}")
                print(f"   Re-rank score: {result['metadata']['rerank_score']+1}/{limit}")
                print(f"   RRF Score: {result['score']:.4f}")
                print(f"   BM25 Rank: {result['metadata']['bm25_rank']}, Semantic Rank: {result['metadata']['semantic_rank']}")
                print(f"   {result['document'][:RETURN_DOCUMENT_LIMIT]}\n")
            passthrough_results = batch_results
        case "cross_encoder":
            rrf_results = hybrid_search.rrf_search(query, k, limit * 5, debug)
            if debug == True:
                print(f"DEBUG: Displaying {limit * 5} pre-enhancement results:")
                for i, result in enumerate(rrf_results[:limit * 5]):
                    print(f"{i+1}. {result['title']}, ID: {result['id']}")
                    print(f"   RRF Score: {result['score']:.4f}")
                    print(f"   BM25 Rank: {result['metadata']['bm25_rank']}, Semantic Rank: {result['metadata']['semantic_rank']}")
                    print(f"   {result['document'][:RETURN_DOCUMENT_LIMIT]}")
            print(f"Reranking top {limit} results using cross encoding method:")
            cross_results = cross_encoder_rerank(query, rrf_results)
            for i, result in enumerate(cross_results[:limit]):
                print(f"{i+1}. {result['title']}, ID: {result['id']}")
                print(f"   Re-rank score: {result['metadata']['rerank_score']:.4f}")
                print(f"   RRF Score: {result['score']:.4f}")
                print(f"   BM25 Rank: {result['metadata']['bm25_rank']}, Semantic Rank: {result['metadata']['semantic_rank']}")
                print(f"   {result['document'][:RETURN_DOCUMENT_LIMIT]}\n")
            passthrough_results = cross_results
        case None:
            rrf_results = hybrid_search.rrf_search(query, k, limit, debug)
            print(f"Displaying {limit} results:")
            for i, result in enumerate(rrf_results[:limit]):
                #print(result)
                print(f"{i+1}. {result['title']}, ID: {result['id']}")
                print(f"   RRF Score: {result['score']:.4f}")
                print(f"   BM25 Rank: {result['metadata']['bm25_rank']}, Semantic Rank: {result['metadata']['semantic_rank']}")
                print(f"   {result['document'][:RETURN_DOCUMENT_LIMIT]}")
            passthrough_results = rrf_results
    if evaluate == True:
        eval_results = evaluate_results(query, passthrough_results[:limit])
        print("Evaluation report:")
        for i, result in enumerate(eval_results[:limit], 1):
            print(f"{i}. {result['title']}: {result['metadata']['evaluation_rank']}/3")


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.inverted_index = InvertedIndex()
        if not os.path.exists(self.inverted_index.index_path):
            self.inverted_index.build()
            self.inverted_index.save()

    def _bm25_search(self, query, limit, debug = False):
        self.inverted_index.load()
        return self.inverted_index.bm25_search(query, limit, debug)
           
    
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

        
    
    def rrf_search(self, query, k=DEFAULT_RRF_K, limit=DEFAULT_RRF_SEARCH_LIMIT, debug=False):
        true_debug = debug
        if debug == True:
            print(f"DEBUG: Original query: {query}")
        bm25_results = self._bm25_search(query, limit * DEFAULT_LARGE_SEARCH_MULTIPLIER, debug)
        css_results = self.semantic_search.search_chunk(query, limit * DEFAULT_LARGE_SEARCH_MULTIPLIER, debug)
        combined_scores = {}

        for bm25_rank, result in enumerate(bm25_results, 1):
            doc_id = result['id']
            if doc_id not in combined_scores:
                if debug:
                    print(f"DEBUG: BM25: Adding title {result['title']} to combined scores with a bm25 score of {bm25_rank}")
                combined_scores[doc_id] = {
                    'title': result['title'],
                    'document': result['document'],
                    'rrf_score': 0.0,
                    'bm25_rank': None,
                    'semantic_rank': None,
                }
            if combined_scores[doc_id]['bm25_rank'] is None:
                combined_scores[doc_id]['bm25_rank'] = bm25_rank
                combined_scores[doc_id]['rrf_score'] += rrf_score(bm25_rank, k)
                if debug:
                    print(f"DEBUG: BM25: title {combined_scores[doc_id]['title']} with a bm25 rank of {bm25_rank} and an rrf score of {rrf_score(bm25_rank, k)}")
                    print(f"DEBUG: BM25: semantic rank for title is currently {combined_scores[doc_id]['semantic_rank']}")
                    kg = input("Press Enter to continue...")
                    if kg:
                        debug = False
        debug = true_debug
        for css_rank, result in enumerate(css_results, 1):
            doc_id = result['id']
            if debug:
                print(f"DEBUG: CSS: result {result['title']}, rank {css_rank}")
                print(f"DEBUG: CSS: document id is {doc_id}")
            if doc_id not in combined_scores:
                if debug: 
                    print(f"DEBUG: Doc ID not in combined scores.")
                combined_scores[doc_id] = {
                    'title': result['title'],
                    'document': result['document'],
                    'rrf_score': 0.0,
                    'bm25_rank': None,
                    'semantic_rank': None,
                }
            else:
                if debug:
                    print(f"DEBUG: CSS: document {doc_id} in combined scores. Should correspond to: {combined_scores[doc_id]}")
            if combined_scores[doc_id]['semantic_rank'] is None:
                combined_scores[doc_id]['semantic_rank'] = css_rank
                if debug:
                    print(f"DEBUG: CSS: title {combined_scores[doc_id]['title']} with a CSS rank of {css_rank} and a current rrf of {combined_scores[doc_id]['rrf_score']}, adding {rrf_score(css_rank, k)}")
                    kg = input("Press Enter to continue...")
                    if kg:
                        debug = False
                combined_scores[doc_id]['rrf_score'] += rrf_score(css_rank, k)

        sorted_items = sorted(combined_scores.items(), key=lambda x: x[1]['rrf_score'], reverse=True)
        debug = true_debug
        rrf_results = []
        for doc_id, data in sorted_items:
            result = format_search_result(doc_id, data['title'], data['document'][:RETURN_DOCUMENT_LIMIT], 
                                          data['rrf_score'], rrf_score=data['rrf_score'], 
                                          bm25_rank=data['bm25_rank'], semantic_rank=data['semantic_rank'])
            if debug:
                print(f"DEBUG: rrf search result: {result}")
                kg = input("Press Enter to continue...")
                if kg:
                    debug = False
            rrf_results.append(result)
        
            
        return rrf_results
        #return sorted(rrf_results, key=lambda x: x["score"], reverse=True)
    
