import numpy as np

def compare_to_bruteforce(sim, k=5, num_queries=20):
    """
    Compare the current backend (LSH or brute-force)
    against brute-force ground truth.

    Returns the average Recall@k.
    """
    n = len(sim.embeddings)
    recalls = []

    for _ in range(num_queries):
        idx = np.random.randint(0, n)

        # Ground truth
        gt = sim._bruteforce_query(idx, k)

        # Current backend
        approx = sim.query(idx, k)

        recall = len(set(gt) & set(approx)) / k
        recalls.append(recall)

        print(f"[Query {idx}] recall={recall:.3f} | gt={gt} | approx={approx}")

    avg = np.mean(recalls)
    print(f"\n[Recall@{k}] Average = {avg:.3f}")
    return avg
