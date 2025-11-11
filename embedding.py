import numpy as np

class Embedding:
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset

    def generate(self):
        embeddings = []
        for img in self.dataset:
            emb = self.model(img)
            embeddings.append(emb)
        return np.array(embeddings)
