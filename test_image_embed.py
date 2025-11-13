#test_image_embed.py
from dataset import Dataset
from facenet_model import TinyFaceNetModel
from embedding import Embedding

dataset = Dataset("olivetti")
model = TinyFaceNetModel()
embedder = Embedding(model, dataset, cache_path="cache/facenet_embeddings.npy")

embeddings = embedder.generate()
print("Embeddings shape:", embeddings.shape)
