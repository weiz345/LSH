# test_pipeline.py
import numpy as np
from dataset import Dataset
from toy_model import ToyModel
from embedding import Embedding
from similarity import Similarity

if __name__ == "__main__":
    # 1️⃣ Load dataset (cached automatically)
    dataset = Dataset(name="olivetti")
    print("Loaded dataset")

    # 2️⃣ Initialize model + embedding generator
    model = ToyModel(embedding_dim=64)
    embedder = Embedding(model, dataset, cache_path="cache/olivetti_embeddings.npy")

    # 3️⃣ Generate or load embeddings
    embeddings = embedder.generate(use_cache=True)
    print("Embeddings shape:", embeddings.shape)

    # 4️⃣ Compute similarity and query
    sim = Similarity(embeddings, metric="cosine")

    idx = 0
    k = 5
    top_k = sim.query(idx=idx, k=k)
    print(f"\nTop {k} most similar to index {idx}: {top_k}")
