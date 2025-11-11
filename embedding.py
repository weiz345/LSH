import numpy as np

class Embedding:
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset

    def generate(self):
        return np.array([self.model(img) for img in self.dataset])
