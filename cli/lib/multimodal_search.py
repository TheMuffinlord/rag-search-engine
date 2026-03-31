from PIL import Image
from sentence_transformers import SentenceTransformer

def verify_image_embedding(image_path):
    mms = MultiModalSearch()
    embedding = mms.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")

class MultiModalSearch:
    def __init__(self, model_name="clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)

    def embed_image(self, image_path):
        image_data = Image.open(image_path)
        image_embedding = self.model.encode([image_data])
        return image_embedding[0]