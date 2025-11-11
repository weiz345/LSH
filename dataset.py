#dataset.py
import numpy as np
import matplotlib.pyplot as plt

class Dataset:
    def __init__(self, name="olivetti"):
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
