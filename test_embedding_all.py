import numpy as np
from embedding import Embedding

models = []
for name, module in [("ToyModel", "toy_model"),
                     ("TensorFlow", "real_model_tf")]:
    try:
        m = __import__(module)
        models.append((name, getattr(m, [k for k in dir(m) if k.lower().startswith("realmodel") or k=="ToyModel"][0])))
    except ImportError:
        pass

if not models:
    print("No models found."); exit()

dataset = [np.random.rand(224, 224, 3) for _ in range(3)]

for name, Model in models:
    print(f"\n--- {name} ---")
    emb = Embedding(Model(), dataset).generate()
    print("shape:", emb.shape)
    print(emb)
