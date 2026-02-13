from sentence_transformers import SentenceTransformer

def embed_text(text):
    newSearch = SemanticSearch()
    embedding = newSearch.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_model():
    SemanticSearchModel = SemanticSearch()
    print(f"Model loaded: {SemanticSearchModel.model}")
    print(f"Max sequence length: {SemanticSearchModel.model.max_seq_length}")

class SemanticSearch:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        pass

    def generate_embedding(self, text):
        if not text or not text.strip():
            raise ValueError("generate embedding whitespace error")
        return self.model.encode([text])[0]
        