# test_pipeline.py
import yaml
import numpy as np
import matplotlib.pyplot as plt

from dataset import Dataset
from toy_model import ToyModel
from embedding import Embedding
from similarity import Similarity


def load_model(cfg):
    model_type = cfg["model"]["type"]
    if model_type == "toy":
        return ToyModel(cfg["model"]["embedding_dim"])
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def main(config_path="config.yaml"):
    # 1. Load config
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    print("Loaded config:", cfg)

    # 2. Load dataset
    dataset = Dataset(cfg["dataset"]["name"])
    imgs = dataset.get()
    num_imgs = len(imgs)

    # Random images for visualization
    rand_indices = np.random.choice(num_imgs, size=5, replace=False)

    # 3. Choose similarity source
    sim_source = cfg["similarity"].get("source", "embedding")
    backend = cfg["similarity"].get("backend", "bruteforce")

    if sim_source == "embedding":
        print("🔹 Using EMBEDDING similarity")
        model = load_model(cfg)
        embedder = Embedding(model, dataset, cache_path=cfg["embedding"]["cache_path"])
        vectors = embedder.generate(use_cache=cfg["embedding"]["use_cache"])

    elif sim_source == "raw":
        print("🔹 Using RAW IMAGE similarity")
        flat = [img.flatten().astype("float32") for img in imgs]
        vectors = np.array(flat)

        if cfg["similarity"]["metric"] == "cosine":
            norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9
            vectors = vectors / norms
    else:
        raise ValueError(f"Unknown similarity source: {sim_source}")

    print("Vectors shape:", vectors.shape)

    # 4. Create similarity engine
    sim = Similarity(
        vectors,
        metric=cfg["similarity"]["metric"],
        backend=backend
    )

    # --------------------------------------------------------------
    # OPTIONAL: Evaluate recall vs brute force
    # --------------------------------------------------------------
    if cfg["similarity"].get("compare_to_bruteforce", False):
        print("\n🔍 Evaluating backend accuracy vs brute force…")

        from similarity_eval import compare_to_bruteforce

        recall = compare_to_bruteforce(
            sim,
            k=cfg["similarity"]["k"],
            num_queries=cfg["similarity"].get("eval_queries", 20)
        )

        print(f"📊 Recall@{cfg['similarity']['k']} = {recall:.4f}\n")

    # --------------------------------------------------------------
    # MASS RUN MODE (benchmark all queries)
    # --------------------------------------------------------------
    if cfg["similarity"].get("mass_run", False):
        print("\n🚀 MASS RUN MODE ENABLED — Running all queries…")

        times = []
        peaks = []

        from profiling import capture_profile

        for i in range(num_imgs):
            top_k, t, mem = capture_profile(sim, i, cfg["similarity"]["k"])
            times.append(t)
            peaks.append(mem)

        print("\n===== MASS RUN SUMMARY =====")
        print(f"Queries run: {num_imgs}")
        print(f"Backend     : {backend}")
        print(f"Source      : {sim_source}")
        print(f"Metric      : {cfg['similarity']['metric']}\n")

        print(f"Avg time    : {np.mean(times)*1000:.3f} ms")
        print(f"Min time    : {np.min(times)*1000:.3f} ms")
        print(f"Max time    : {np.max(times)*1000:.3f} ms\n")

        print(f"Avg memory  : {np.mean(peaks)/1024:.2f} KB")
        print(f"Max memory  : {np.max(peaks)/1024:.2f} KB")
        print("================================\n")
        return

    # --------------------------------------------------------------
    # Normal mode: single query + visualization
    # --------------------------------------------------------------
    idx = cfg["similarity"]["query_index"]
    k = cfg["similarity"]["k"]

    print(f"\nRunning similarity search → backend={backend}")
    top_k = sim.query(idx=idx, k=k)
    print(f"\nTop {k} results for index {idx}: {top_k}")

    total_cols = max(5, k + 1)
    total_rows = 2

    plt.figure(figsize=(3 * total_cols, 6))

    for i, r in enumerate(rand_indices):
        plt.subplot(total_rows, total_cols, i + 1)
        plt.imshow(imgs[r], cmap="gray")
        plt.title(f"Rand {r}")
        plt.axis("off")

    plt.subplot(total_rows, total_cols, total_cols + 1)
    plt.imshow(imgs[idx], cmap="gray")
    plt.title("Query")
    plt.axis("off")

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
