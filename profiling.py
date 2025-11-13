import time
import tracemalloc

def profile(func):
    """Decorator: print time + memory."""
    def wrapper(*args, **kwargs):
        tracemalloc.start()
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print(f"[PROFILE] Time: {elapsed*1000:.3f} ms | Peak: {peak/1024:.1f} KB")
        return result
    return wrapper


def capture_profile(sim, idx, k):
    """
    Run similarity.query(idx) but return:
    - results
    - time in seconds
    - peak memory in bytes
    """
    tracemalloc.start()
    t0 = time.perf_counter()
    out = sim.query_fn(idx, k)   # ❗ bypass decorator
    elapsed = time.perf_counter() - t0
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return out, elapsed, peak
