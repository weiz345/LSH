import numpy as np
from profiling import profile   # <-- NEW

class Similarity:
    def __init__(self, embeddings, metric="cosine"):
        self.embeddings = embeddings
        self.metric = metric

    def _cosine(self, a, b):
        a, b = np.array(a), np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

    def _euclidean(self, a, b):
        return -np.linalg.norm(a - b)

    @profile                      # <-- ADD THIS
    def query(self, idx, k=3):
        q = self.embeddings[idx]
        sims = []
        for i, emb in enumerate(self.embeddings):
            if i == idx:
                continue
            if self.metric == "cosine":
                s = self._cosine(q, emb)
            else:
                s = self._euclidean(q, emb)
            sims.append((i, s))

        sims.sort(key=lambda x: x[1], reverse=True)
        return [i for i, _ in sims[:k]]
