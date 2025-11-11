# dataset.py
import os
import numpy as np
import matplotlib.pyplot as plt

class Dataset:
    def __init__(self, name="olivetti", cache_dir="cache"):
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"{name}.npy")

        if os.path.exists(cache_path):
            self.images = np.load(cache_path)
            print(f"Loaded cached dataset from {cache_path}")
            return

        if name == "olivetti":
            from sklearn.datasets import fetch_olivetti_faces
            data = fetch_olivetti_faces(shuffle=True)
            imgs = data.images
            self.images = np.stack([imgs]*3, axis=-1)
        elif name == "mnist":
            from tensorflow.keras.datasets import mnist
            (x, _), _ = mnist.load_data()
            x = x / 255.0
            self.images = np.stack([x]*3, axis=-1)
        else:
            raise ValueError(f"Unknown dataset name: {name}")

        np.save(cache_path, self.images)
        print(f"Saved dataset cache to {cache_path}")

    def show(self, n=9, save_path="grid.png"):
        plt.figure(figsize=(6, 6))
        for i in range(n):
            plt.subplot(3, 3, i + 1)
            plt.imshow(self.images[i])
            plt.axis("off")
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Saved grid to {save_path}")

    def get(self):
        return self.images
