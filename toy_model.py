#toy_model.py
import numpy as np

class ToyModel:
    def __init__(self, embedding_dim=4):
        self.embedding_dim = embedding_dim

    def __call__(self, img):
        flat = img.flatten()
        np.random.seed(42)
        W = np.random.randn(self.embedding_dim, flat.size)
        return np.dot(W, flat)
