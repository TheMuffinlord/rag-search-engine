from sentence_transformers import SentenceTransformer
from .word_actions import format_search_result, load_movies
from .constants import (CACHE_DIR, DATA_PATH, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, 
                       DEFAULT_SEMANTIC_CHUNK_SIZE, DEFAULT_CHUNK_SEARCH_LIMIT, SCORE_PRECISION, 
                       RETURN_DOCUMENT_LIMIT)
import numpy as np
import os, json, re

def semantic_chunk_cmd(text:str, size=DEFAULT_SEMANTIC_CHUNK_SIZE, overlap=DEFAULT_CHUNK_OVERLAP):
    r_chunks = sentence_chunk_doer(text, size, overlap)
    print(f"Semantically chunking {len(text)} characters")
    for n, chunk in enumerate(r_chunks):
        print(f"{n+1}. {chunk}")


def sentence_chunk_doer(text:str, size, overlap):
    text = text.strip()
    if text == "" or not text:
        return []
    sentences = re.split(pattern=r"(?<=[.!?])\s+", string=text)
    #more CH9.4 fuckery, switched these around
    if len(sentences) == 1 and not text.endswith((".", "!", "?")):
        #print(f"DEBUG: non-punctuated edge case. Confirmation: '{sentences}'")
        sentences = [text]
    chunks = []
    n_sent = len(sentences)
    i = 0
    while i < n_sent:
        chunk_sents = sentences[i: i + size]
        if chunks and len(chunk_sents) <= overlap:
            break
        clean_sent = []
        for chunk_sent in chunk_sents:
            clean_sent.append(chunk_sent.strip())
        if not clean_sent:
            continue
        chunk = ' '.join(clean_sent)

        if chunk != "":
            chunks.append(chunk)
        i += size - overlap
    return chunks

def embed_text(text):
    newSearch = SemanticSearch()
    embedding = newSearch.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def embed_query_text(query):
    ssm = SemanticSearch()
    embedding = ssm.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")

def verify_model():
    SemanticSearchModel = SemanticSearch()
    print(f"Model loaded: {SemanticSearchModel.model}")
    print(f"Max sequence length: {SemanticSearchModel.model.max_seq_length}")

def verify_embeddings():
    ssm = SemanticSearch()
    documents = load_movies()
    embeddings = ssm.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(dot_product / (norm1 * norm2))

def semantic_search(query, limit=5):
    ssm = SemanticSearch()
    documents = load_movies()
    embeddings = ssm.load_or_create_embeddings(documents)
    result = ssm.search(query, limit)
    for r in range(len(result)):
        print(f"{r+1}. {result[r]['title']} (score: {result[r]['score']:.4f})")
        print(f"   {result[r]['description']}")
        print()

def chunk_doer(text, size, overlap):
    words = text.split()
    chunks = []

    n_words = len(words)
    i = 0
    while i < n_words:
        chunk_words = words[i: i + size]
        if chunks and len(chunk_words) <= overlap:
            break
        chunks.append(' '.join(chunk_words))
        i += size - overlap
    return chunks

def chunk_command(text:str, size=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_CHUNK_OVERLAP):
    r_chunks = chunk_doer(text, size, overlap)
    print(f"Chunking {len(text)} characters")
    for n, chunk in enumerate(r_chunks):
        print(f"{n+1}. {chunk}")

def embed_chunks():
    csm = ChunkedSemanticSearch()
    documents = load_movies()
    chunk_embeddings = csm.load_or_create_chunk_embeddings(documents)
    print(f"Generated {len(chunk_embeddings)} chunked embeddings")

def search_chunked_cmd(query: str, limit=DEFAULT_CHUNK_SEARCH_LIMIT): 
    with open(DATA_PATH, 'r') as f:
        movieList = json.load(f)
    csm = ChunkedSemanticSearch()
    chunk_embeddings = csm.load_or_create_chunk_embeddings(movieList['movies'])
    results = csm.search_chunk(query, limit)
    print(f"DEBUG: RESULTS DUMP: {results}")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['title']} (score: {result['score']:.4f})")
        print(f"    {result['document']}...")

class SemanticSearch:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map = {}
        self.embeddings_path = os.path.join(CACHE_DIR, 'movie_embeddings.npy')
        pass

    def generate_embedding(self, text):
        if not text or not text.strip():
            raise ValueError("generate embedding whitespace error")
        return self.model.encode([text])[0]

    def build_embeddings(self, documents):
        self.documents = documents
        doclist = []
        for doc in documents:
            self.document_map[doc['id']] = doc
            doclist.append(f"{doc['title']}: {doc['description']}")
        self.embeddings = self.model.encode(doclist, show_progress_bar=True)
        os.makedirs(os.path.dirname(self.embeddings_path), exist_ok=True)
        np.save(self.embeddings_path, self.embeddings)
        return self.embeddings
    
    def load_or_create_embeddings(self, documents):
        self.documents = documents
        self.document_map = {}
        for doc in documents:
            self.document_map[doc['id']] = doc
        if os.path.exists(self.embeddings_path):
            self.embeddings = np.load(self.embeddings_path)
            if len(self.embeddings) == len(documents):
                return self.embeddings
        return self.build_embeddings(documents)
    
    def search(self, query, limit):
        if self.embeddings is None or self.embeddings.size == 0:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        if self.documents is None or len(self.documents) == 0:
            raise ValueError("No documents loaded. Call 'load_or_create_embeddings' first.")
        query_embedding = self.generate_embedding(query)
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            cs = cosine_similarity(query_embedding, doc_embedding)
            similarities.append((cs, self.documents[i]))
        similarities.sort(key=lambda item: item[0], reverse=True)
        results = []
        for score, doc in similarities[:limit]:
            entry = {'score': score,
                     'title': doc['title'],
                     'description': doc['description']}
            results.append(entry)
        return results
    

class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_embeddings_path = os.path.join(CACHE_DIR, 'chunk_embeddings.npy')
        self.chunk_metadata = None
        self.chunk_metadata_path = os.path.join(CACHE_DIR, 'chunk_metadata.json')

    def build_chunk_embeddings(self, documents: list[dict]):
        self.documents = documents
        self.document_map = {}
        #ch9.4 rearing its head again
        for doc in self.documents:
            self.document_map[doc['id']] = doc
        all_chunks = []
        all_chunk_meta = []
        for i, doc in enumerate(documents):
            text = doc.get('description', '')
            if not text.strip():
                continue
            #if len(doc['description']) != 0:
            chunks = sentence_chunk_doer(text, DEFAULT_SEMANTIC_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP)
            for j, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_chunk_meta.append({'movie_idx': i, 'chunk_idx': j, 'total_chunks': len(chunks)})
        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = all_chunk_meta
        os.makedirs(os.path.dirname(self.chunk_embeddings_path), exist_ok=True)
        np.save(self.chunk_embeddings_path, self.chunk_embeddings)
        with open(self.chunk_metadata_path, 'w') as f:
            json.dump({"chunks": self.chunk_metadata, "total_chunks": len(all_chunks)}, f, indent=2)
        return self.chunk_embeddings
    
    def load_or_create_chunk_embeddings(self, documents):
        self.documents = documents
        for doc in documents:
            self.document_map[doc['id']] = doc
        if os.path.exists(self.chunk_embeddings_path) and os.path.exists(self.chunk_metadata_path):
            self.chunk_embeddings = np.load(self.chunk_embeddings_path)
            with open(self.chunk_metadata_path, 'r') as f:
                chunk_meta = json.load(f)
                self.chunk_metadata = chunk_meta['chunks']
            #if len(self.chunk_embeddings) == chunk_meta['total_chunks']:
                #print("DEBUG: chunk load full success.")
            return self.chunk_embeddings
            #print("DEBUG: chunks are of different size; not sure this matters?")

        print("DEBUG: chunk load irregularity. Rebuilding chunk embedding.")
        return self.build_chunk_embeddings(documents)
        

    def search_chunk(self, query:str, limit: int = DEFAULT_CHUNK_SEARCH_LIMIT):
        query_embedding = self.generate_embedding(query)
        chunk_scores = []
        for i, chunk_embedding in enumerate(self.chunk_embeddings):
            c_score = cosine_similarity(query_embedding, chunk_embedding)
            chunk_scores.append({'chunk_idx':i, 'movie_idx':self.chunk_metadata[i]['movie_idx'], 'score':c_score})
        movie_score_map = {}
        for chunk_score in chunk_scores:
            movie_index = chunk_score['movie_idx']
            #print(f"DEBUG: movie index {movie_index}, chunk_score {chunk_score}")
            if (movie_index not in movie_score_map or movie_score_map[movie_index] < chunk_score['score']):
                movie_score_map[movie_index] = chunk_score['score']
        sorted_movies = sorted(movie_score_map.items(), key=lambda item: item[1], reverse=True)
        sorted_scores = []
        #CH9.4 more cleanup. 
        for d, score in sorted_movies[:limit]:
            #print(d)
            #print(score)
            
            doc = self.documents[d]
            #print(doc)
            sorted_scores.append(format_search_result(d, doc['title'], doc['description'][:RETURN_DOCUMENT_LIMIT], score))
            
        return sorted_scores