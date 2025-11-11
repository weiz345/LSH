# test_pipeline.py
import yaml
from dataset import Dataset
from toy_model import ToyModel
from embedding import Embedding
from similarity import Similarity

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

if __name__ == "__main__":
    main()
