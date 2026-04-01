from PIL import Image
from sentence_transformers import SentenceTransformer
from .semantic_search import cosine_similarity
from .word_actions import load_movies

def verify_image_embedding(image_path):
    mms = MultiModalSearch()
    embedding = mms.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")

def image_search_cmd(image_path):
    documents = load_movies()
    mms = MultiModalSearch(documents)
    results = mms.search_with_image(image_path)
    for i, result in enumerate(results[:5], 1):
        print(f"{i}. {result['title']} (similarity: {result['score']:.3f})")
        print(f"   {result['description'][:200]}")
        print()

class MultiModalSearch:
    def __init__(self, doc_list: list, model_name="clip-ViT-B-32",):
        self.model = SentenceTransformer(model_name)
        self.doc_list = doc_list
        self.texts = []
        for doc in doc_list:
            self.texts.append(f"{doc['title']}: {doc['description']}")
        print("Generating text embeddings. This may take some time.")
        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)
        print("Embeddings are generated. Moving on.")

    def embed_image(self, image_path):
        image_data = Image.open(image_path)
        image_embedding = self.model.encode([image_data])
        return image_embedding[0]
    
    def search_with_image(self, image_path):
        image_embedding = self.embed_image(image_path)
        embedding_results = []
        for i, text_embed in enumerate(self.text_embeddings):
            cosine_sim = cosine_similarity(image_embedding, text_embed)
            doc = self.doc_list[i]
            embedding_results.append({
                "id": doc['id'],
                "title": doc['title'],
                "description": doc['description'],
                "score": cosine_sim
            })
        sorted_results = sorted(embedding_results, key=lambda x: x['score'], reverse=True)
        return sorted_results[:5] #this is the last lesson, i guess we've earned a few hard-coded variables as a treat
    