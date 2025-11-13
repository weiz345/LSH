import numpy as np
from collections import defaultdict
from profiling import profile


class Similarity:
    """
    Generic similarity interface that supports:
    - Metrics: cosine, euclidean
    - Backends: bruteforce, lsh
    """

    def __init__(self, embeddings, metric="cosine", backend="bruteforce"):
        self.embeddings = embeddings
        self.metric = metric
        self.backend = backend

        # ---------------- Metric dispatch table ----------------
        self.metric_fns = {
            "cosine": self._cosine,
            "euclidean": self._euclidean,
        }

        if metric not in self.metric_fns:
            raise ValueError(f"Unknown metric: {metric}")

        self.metric_fn = self.metric_fns[metric]

        # ---------------- Backend dispatch table ----------------
        self.backend_fns = {
            "bruteforce": self._bruteforce_query,
            "lsh": self._lsh_query,
        }

        if backend not in self.backend_fns:
            raise ValueError(f"Unknown backend: {backend}")

        self.query_fn = self.backend_fns[backend]

        # ---------------- Backend Initialization ----------------
        if backend == "lsh":
            self._init_lsh()     # build LSH tables


    # ============================================================
    #                        METRICS
    # ============================================================

    def _cosine(self, a, b):
        """Cosine similarity."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

    def _euclidean(self, a, b):
        """Negative Euclidean distance so 'higher = better'."""
        return -np.linalg.norm(a - b)

    def compute_similarity(self, a, b):
        """Unified similarity call."""
        return self.metric_fn(a, b)


    # ============================================================
    #                  PUBLIC QUERY ENTRYPOINT
    # ============================================================

    @profile
    def query(self, idx, k=3):
        """
        Dispatch to chosen backend.
        Automatically profiled for time + memory.
        """
        return self.query_fn(idx, k)


    # ============================================================
    #                     BRUTE-FORCE BACKEND
    # ============================================================

    def _bruteforce_query(self, idx, k=3):
        q = self.embeddings[idx]
        sims = []

        for i, emb in enumerate(self.embeddings):
            if i == idx:
                continue
            sim = self.compute_similarity(q, emb)
            sims.append((i, sim))

        sims.sort(key=lambda x: x[1], reverse=True)
        return [i for i, _ in sims[:k]]


    # ============================================================
    #                          LSH BACKEND
    # ============================================================

    def _init_lsh(self, num_planes=10, num_tables=5):
        """
        Build LSH tables using Random Hyperplane Hashing (SimHash).
        num_planes = K  (planes per table)
        num_tables = L  (number of hash tables)
        """
        dim = self.embeddings.shape[1]

        self.num_planes = num_planes
        self.num_tables = num_tables

        # Create random hyperplanes
        self.lsh_planes = [
            np.random.randn(num_planes, dim)     # shape (K, D)
            for _ in range(num_tables)
        ]

        # Initialize hash buckets
        self.lsh_tables = [defaultdict(list) for _ in range(num_tables)]

        # Populate hash tables
        for idx, vec in enumerate(self.embeddings):
            for t, planes in enumerate(self.lsh_planes):
                sig = self._hash_signature(vec, planes)
                self.lsh_tables[t][sig].append(idx)

        print(f"[LSH] Initialized {num_tables} tables × {num_planes} planes")


    def _hash_signature(self, vec, planes):
        """
        Convert vector to bit signature using random hyperplanes.
        """
        dots = np.dot(planes, vec)       # (K,)
        bits = (dots > 0).astype(int)

        sig = 0
        for b in bits:
            sig = (sig << 1) | b

        return sig


    def _lsh_query(self, idx, k=3):
        """
        Approximate search using LSH buckets.
        Falls back to brute-force if no candidates found.
        """
        q = self.embeddings[idx]
        candidates = set()

        # Search for items in same buckets across all tables
        for t, planes in enumerate(self.lsh_planes):
            sig = self._hash_signature(q, planes)
            bucket = self.lsh_tables[t].get(sig, [])
            for item in bucket:
                if item != idx:
                    candidates.add(item)

        # Fallback if no candidates
        if len(candidates) == 0:
            print("[LSH] No candidates — fallback to bruteforce")
            return self._bruteforce_query(idx, k)

        # Compute similarity only on candidate set
        scored = []
        for j in candidates:
            s = self.compute_similarity(q, self.embeddings[j])
            scored.append((j, s))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [i for i, _ in scored[:k]]
