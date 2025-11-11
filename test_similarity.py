import numpy as np
from similarity import Similarity

# pretend we already have 5 embeddings of dimension 4
embeddings = np.random.rand(5, 4)

sim = Similarity(embeddings, metric="cosine")
top_k = sim.query(idx=0, k=2)

print("Query index:", 0)
print("Top 2 most similar indices:", top_k)
