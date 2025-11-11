# test_similarity.py
import numpy as np
from similarity import Similarity

if __name__ == "__main__":
    try:
        embeddings = np.load("toy_embeddings.npy")
        print("Loaded cached embeddings from toy_embeddings.npy")
    except FileNotFoundError:
        print("No cached embeddings found, creating random ones instead.")
        embeddings = np.random.rand(5, 4)
        np.save("toy_embeddings.npy", embeddings)

    sim = Similarity(embeddings, metric="cosine")
    top_k = sim.query(idx=0, k=2)
    print("\nQuery index:", 0)
    print("Top 2 most similar indices:", top_k)
