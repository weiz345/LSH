# embedding.py
import numpy as np
import os
from tqdm import tqdm

class Embedding:
    def __init__(self, model, dataset, cache_path="embeddings.npy"):
        self.model = model
        self.dataset = dataset
        self.cache_path = cache_path

    def generate(self, use_cache=True):
        if use_cache and os.path.exists(self.cache_path):
            print(f"Loaded cached embeddings from {self.cache_path}")
            return np.load(self.cache_path)

        items = []
        print("Generating embeddings...")

        for img in tqdm(self.dataset, desc="Embedding images", ncols=80):
            vec = self.model(img)
            items.append(vec)

        embeddings = np.array(items)
        np.save(self.cache_path, embeddings)
        print(f"Saved embeddings to {self.cache_path}")

        return embeddings
