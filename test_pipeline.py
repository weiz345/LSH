import numpy as np
from toy_model import ToyModel
from embedding import Embedding
from similarity import Similarity

if __name__ == "__main__":
    # Fake dataset: 5 random RGB images (5x5x3)
    dataset = [np.random.rand(5, 5, 3) for _ in range(5)]

    # Step 1: Generate embeddings
    model = ToyModel(embedding_dim=4)
    embedder = Embedding(model, dataset)
    embeddings = embedder.generate()
    print("Embeddings:\n", embeddings)

    # Step 2: Run similarity search
    sim = Similarity(embeddings, metric="cosine")
    top_k = sim.query(idx=0, k=2)
    print("\nTop 2 most similar to index 0:", top_k)
