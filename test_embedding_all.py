# test_embedding_all.py
import numpy as np
from embedding import Embedding
import os

models = []
for name, module in [("ToyModel", "toy_model"),
                     ("TensorFlow", "real_model_tf")]:
    try:
        m = __import__(module)
        models.append((name, getattr(m, [k for k in dir(m) if k.lower().startswith("realmodel") or k == "ToyModel"][0])))
    except ImportError:
        pass

if not models:
    print("No models found.")
    exit()

dataset = [np.random.rand(224, 224, 3) for _ in range(3)]

for name, Model in models:
    cache_path = f"{name.lower()}_embeddings.npy"
    print(f"\n--- {name} ---")
    embedder = Embedding(Model(), dataset, cache_path=cache_path)
    embeddings = embedder.generate()
    print(f"{name} embeddings shape:", embeddings.shape)

    if os.path.exists(cache_path):
        print(f"(Cached at {cache_path})")
