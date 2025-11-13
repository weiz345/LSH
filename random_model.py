# random_model.py
import numpy as np

class RandomModel:
    def __init__(self, dim=128):
        self.dim = dim
        self.name = f"random_{dim}"

    def __call__(self, img):
        # Return completely random embedding unrelated to the image
        return np.random.randn(self.dim).astype("float32")
