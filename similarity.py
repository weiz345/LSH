import numpy as np
from profiling import profile


class Similarity:
    def __init__(self, embeddings, metric="cosine", backend="bruteforce"):
        self.embeddings = embeddings
        self.metric = metric
        self.backend = backend

        # ---- Metric dispatch table ----
        self.metric_fns = {
            "cosine": self._cosine,
            "euclidean": self._euclidean,
        }

        if metric not in self.metric_fns:
            raise ValueError(f"Unknown metric: {metric}")

        self.metric_fn = self.metric_fns[metric]

        # ---- Backend dispatch table ----
        self.backend_fns = {
            "bruteforce": self._bruteforce_query,
            # "lsh": self._lsh_query,         # (later)
            # "faiss": self._faiss_query,     # (later)
            # "hnsw": self._hnsw_query,       # (later)
        }

        if backend not in self.backend_fns:
            raise ValueError(f"Unknown backend: {backend}")

        self.query_fn = self.backend_fns[backend]

    # ----------------- Metrics -----------------

    def _cosine(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

    def _euclidean(self, a, b):
        # Return NEGATIVE distance so that "higher = better"
        return -np.linalg.norm(a - b)

    def compute_similarity(self, a, b):
        """Unified similarity call."""
        return self.metric_fn(a, b)

    # ----------------- Query Backends -----------------

    @profile
    def query(self, idx, k=3):
        """Top-level query — calls whichever backend the config selected."""
        return self.query_fn(idx, k)

    def _bruteforce_query(self, idx, k=3):
        """Current brute-force implementation."""
        q = self.embeddings[idx]
        sims = []

        for i, emb in enumerate(self.embeddings):
            if i == idx:
                continue
            s = self.compute_similarity(q, emb)
            sims.append((i, s))

        sims.sort(key=lambda x: x[1], reverse=True)
        return [i for i, _ in sims[:k]]

    # ----------------- Future backends -----------------
    # def _lsh_query(self, idx, k):
    #     ...
    # def _faiss_query(self, idx, k):
    #     ...
    # def _hnsw_query(self, idx, k):
    #     ...
