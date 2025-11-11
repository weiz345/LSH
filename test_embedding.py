# Dummy example
import numpy as np
from embedding import Embedding

# fake model that outputs 3D embedding
def fake_model(img):
    return np.mean(img, axis=(0, 1))[:3]

# fake dataset (3 images of 5x5x3)
dataset = [np.random.rand(5,5,3) for _ in range(3)]

embed = Embedding(fake_model, dataset)
embeddings = embed.generate()
print(embeddings)
