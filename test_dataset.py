import matplotlib
matplotlib.use("Agg")  # Safe for WSL/headless
from dataset import Dataset

datasets = ["olivetti", "mnist"]

for name in datasets:
    print(f"\n=== Testing dataset: {name} ===")
    ds = Dataset(name=name)
    ds.show(save_path=f"{name}_grid.png")
    imgs = ds.get()
    print("Shape:", imgs.shape)
