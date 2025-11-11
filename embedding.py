#embedding.py
import numpy as np
import os

class Embedding:
    def __init__(self, model, dataset, cache_path="embeddings.npy"):
        self.model = model
        self.dataset = dataset
        self.cache_path = cache_path

    def generate(self, use_cache=True):
        if use_cache and os.path.exists(self.cache_path):
            print(f"Loading cached embeddings from {self.cache_path}")
            return np.load(self.cache_path)

        print("Generating new embeddings...")
        embeddings = np.array([self.model(img) for img in self.dataset])
        np.save(self.cache_path, embeddings)
        print(f"Saved embeddings to {self.cache_path}")
        return embeddings
