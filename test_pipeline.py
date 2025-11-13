# test_pipeline.py
import yaml
import numpy as np
import matplotlib.pyplot as plt

from dataset import Dataset
from toy_model import ToyModel
from embedding import Embedding
from similarity import Similarity


def load_model(cfg):
    """Factory for model selection."""
    model_type = cfg["model"]["type"]
    if model_type == "toy":
        return ToyModel(cfg["model"]["embedding_dim"])
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def main(config_path="config.yaml"):
    # ----------------------------------------------------
    # 1️⃣ Load config
    # ----------------------------------------------------
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    print("Loaded config:", cfg)

    # ----------------------------------------------------
    # 2️⃣ Load dataset + images
    # ----------------------------------------------------
    dataset = Dataset(name=cfg["dataset"]["name"])
    imgs = dataset.get()
    num_imgs = len(imgs)

    # random sample for visualization
    rand_indices = np.random.choice(num_imgs, size=5, replace=False)

    # ----------------------------------------------------
    # 3️⃣ Select similarity source: "embedding" OR "raw"
    # ----------------------------------------------------
    sim_source = cfg["similarity"].get("source", "embedding")
    backend = cfg["similarity"].get("backend", "bruteforce")

    if sim_source == "embedding":
        print("🔹 Using EMBEDDING similarity")

        # model + embedding generator
        model = load_model(cfg)
        embedder = Embedding(
            model,
            dataset,
            cache_path=cfg["embedding"]["cache_path"]
        )

        vectors = embedder.generate(use_cache=cfg["embedding"]["use_cache"])

    elif sim_source == "raw":
        print("🔹 Using RAW IMAGE similarity (flattened pixels)")

        # flatten all images → vectors
        flat = [img.flatten().astype("float32") for img in imgs]
        vectors = np.array(flat)

        # normalize for cosine similarity
        if cfg["similarity"]["metric"] == "cosine":
            norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9
            vectors = vectors / norms

    else:
        raise ValueError(f"Unknown similarity source: {sim_source}")

    print("Vectors shape:", vectors.shape)

    # ----------------------------------------------------
    # 4️⃣ Similarity search (backend: bruteforce, lsh, etc.)
    # ----------------------------------------------------
    sim = Similarity(
        vectors,
        metric=cfg["similarity"]["metric"],
        backend=backend
    )

    idx = cfg["similarity"]["query_index"]
    k = cfg["similarity"]["k"]

    print(f"\nRunning similarity search → backend={backend}")
    top_k = sim.query(idx=idx, k=k)

    print(f"\nTop {k} most similar to index {idx}: {top_k}")

    # ----------------------------------------------------
    # 5️⃣ Visualization: random → query → top-k neighbors
    # ----------------------------------------------------
    total_cols = max(5, k + 1)
    total_rows = 2

    plt.figure(figsize=(3 * total_cols, 6))

    # row 1 — random samples
    for i, r in enumerate(rand_indices):
        plt.subplot(total_rows, total_cols, i + 1)
        plt.imshow(imgs[r], cmap="gray")
        plt.title(f"Rand {r}")
        plt.axis("off")

    # row 2 — query face
    plt.subplot(total_rows, total_cols, total_cols + 1)
    plt.imshow(imgs[idx], cmap="gray")
    plt.title("Query")
    plt.axis("off")

    # row 2 — top-k neighbors
    for i, j in enumerate(top_k):
        plt.subplot(total_rows, total_cols, total_cols + 2 + i)
        plt.imshow(imgs[j], cmap="gray")
        plt.title(f"Rank {i+1}")
        plt.axis("off")

    plt.tight_layout()
    out_path = "results.png"
    plt.savefig(out_path)
    plt.close()

    print(f"\nSaved combined results → {out_path}")


if __name__ == "__main__":
    main()
