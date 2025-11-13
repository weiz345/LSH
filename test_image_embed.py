#test_image_embed.py
from dataset import Dataset
from facenet_model import TinyFaceNetModel
from embedding import Embedding
from random_model import RandomModel

dataset = Dataset("olivetti")
model = RandomModel(dim=64)
embedder = Embedding(model, dataset, cache_path="cache/random_embeddings.npy")

embeddings = embedder.generate()
print("Embeddings shape:", embeddings.shape)
