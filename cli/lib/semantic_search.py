from sentence_transformers import SentenceTransformer
from constants import CACHE_DIR, DATA_PATH, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
import numpy as np
import os, json



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
    with open(DATA_PATH, 'r') as f:
        movieList = json.load(f)
    documents = movieList['movies']
    embeddings = ssm.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def semantic_search(query, limit=5):
    ssm = SemanticSearch()
    with open(DATA_PATH, 'r') as f:
        movieList = json.load(f)
    documents = movieList['movies']
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
        with open(self.embeddings_path, 'wb') as f:
            np.save(f, self.embeddings)
        return self.embeddings
    
    def load_or_create_embeddings(self, documents):
        self.documents = documents
        for doc in documents:
            self.document_map[doc['id']] = doc
        if os.path.exists(self.embeddings_path):
            with open(self.embeddings_path, 'rb') as f:
                self.embeddings = np.load(f)
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