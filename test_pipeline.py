# test_pipeline.py
import yaml
from dataset import Dataset
from toy_model import ToyModel
from embedding import Embedding
from similarity import Similarity
import matplotlib.pyplot as plt
import numpy as np


def load_model(cfg):
    """Factory for model selection"""
    model_type = cfg["model"]["type"]
    if model_type == "toy":
        return ToyModel(cfg["model"]["embedding_dim"])
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def main(config_path="config.yaml"):
    # 1️⃣ Load config
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    print("Loaded config:", cfg)

    # 2️⃣ Load dataset
    dataset = Dataset(name=cfg["dataset"]["name"])
    imgs = dataset.get()

    # --- prepare random faces ---
    rand_indices = np.random.choice(len(imgs), size=5, replace=False)

    # 3️⃣ Initialize model + embedding generator
    model = load_model(cfg)
    embedder = Embedding(
        model,
        dataset,
        cache_path=cfg["embedding"]["cache_path"]
    )

    # 4️⃣ Generate or load embeddings
    embeddings = embedder.generate(use_cache=cfg["embedding"]["use_cache"])
    print("Embeddings shape:", embeddings.shape)

    # 5️⃣ Similarity query
    sim = Similarity(embeddings, metric=cfg["similarity"]["metric"])
    idx = cfg["similarity"]["query_index"]
    k = cfg["similarity"]["k"]
    top_k = sim.query(idx=idx, k=k)
    print(f"\nTop {k} most similar to index {idx}: {top_k}")

    # 6️⃣ Combined PNG: random + query + top-k
    # ------------------------------------------------
    total_cols = max(5, k + 1)
    total_rows = 2

    plt.figure(figsize=(3 * total_cols, 6))

    # Row 1: random faces
    for i, r in enumerate(rand_indices):
        plt.subplot(total_rows, total_cols, i + 1)
        plt.imshow(imgs[r])
        plt.title(f"Rand {r}")
        plt.axis("off")

    # Row 2: query image
    plt.subplot(total_rows, total_cols, total_cols + 1)
    plt.imshow(imgs[idx])
    plt.title("Query")
    plt.axis("off")

    # Row 2: top-k similar images
    for i, j in enumerate(top_k):
        plt.subplot(total_rows, total_cols, total_cols + 2 + i)
        plt.imshow(imgs[j])
        plt.title(f"Rank {i+1}")
        plt.axis("off")

    plt.tight_layout()
    out_path = "results.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved combined results → {out_path}")


if __name__ == "__main__":
    main()
