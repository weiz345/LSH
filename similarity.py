import numpy as np
from collections import defaultdict
from profiling import profile


class Similarity:
    """
    Generic similarity engine that supports:
    - Metrics: cosine, euclidean
    - Backends: bruteforce, lsh
    """

    def __init__(self, embeddings, metric="cosine", backend="bruteforce"):
        # Always use float32 for speed
        self.embeddings = embeddings.astype(np.float32)

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

        # Backend-specific initialization
        if backend == "lsh":
            self._init_lsh()

        self.query_fn = self.backend_fns[backend]

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
        return self.metric_fn(a, b)

    # ============================================================
    #                  PUBLIC QUERY ENTRYPOINT
    # ============================================================

    @profile
    def query(self, idx, k=3):
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

            sim = self.metric_fn(q, emb)
            sims.append((i, sim))

        sims.sort(key=lambda x: x[1], reverse=True)
        return [i for i, _ in sims[:k]]

    # ============================================================
    #                          LSH BACKEND
    # ============================================================

    def _init_lsh(self, num_planes=10, num_tables=5):
        """
        Build LSH tables using Random Hyperplane Hashing.
        K = num_planes  (planes per table)
        L = num_tables  (number of hash tables)
        """
        dim = self.embeddings.shape[1]

        self.num_planes = num_planes
        self.num_tables = num_tables

        # Random projection hyperplanes
        self.lsh_planes = [
            np.random.randn(num_planes, dim)  # (K, D)
            for _ in range(num_tables)
        ]

        # Hash buckets
        self.lsh_tables = [defaultdict(list) for _ in range(num_tables)]

        # Populate LSH tables
        for idx, vec in enumerate(self.embeddings):
            for t, planes in enumerate(self.lsh_planes):
                sig = self._hash_signature(vec, planes)
                self.lsh_tables[t][sig].append(idx)

        print(f"[LSH] Initialized {num_tables} tables × {num_planes} planes")

    def _hash_signature(self, vec, planes):
        dots = np.dot(planes, vec)
        bits = (dots > 0).astype(int)

        sig = 0
        for b in bits:
            sig = (sig << 1) | b
        return sig

    def _lsh_query(self, idx, k=3):
        q = self.embeddings[idx]
        candidates = set()

        # Search buckets from each table
        for t, planes in enumerate(self.lsh_planes):
            sig = self._hash_signature(q, planes)
            bucket = self.lsh_tables[t].get(sig, [])
            for item in bucket:
                if item != idx:
                    candidates.add(item)

        # Fallback if empty
        if not candidates:
            print("[LSH] No candidates — fallback to brute-force")
            return self._bruteforce_query(idx, k)

        # Score only candidates
        scored = [(j, self.metric_fn(q, self.embeddings[j])) for j in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)

        return [i for i, _ in scored[:k]]
