# test_pipeline.py
import numpy as np
from toy_model import ToyModel
from embedding import Embedding
from similarity import Similarity

if __name__ == "__main__":
    dataset = [np.random.rand(5, 5, 3) for _ in range(5)]

    model = ToyModel(embedding_dim=4)
    embedder = Embedding(model, dataset, cache_path="toy_embeddings.npy")
    embeddings = embedder.generate()
    print("Embeddings:\n", embeddings)

    sim = Similarity(embeddings, metric="cosine")
    top_k = sim.query(idx=0, k=2)
    print("\nTop 2 most similar to index 0:", top_k)
