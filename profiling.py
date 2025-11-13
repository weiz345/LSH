# profiling.py
import time
import tracemalloc
from functools import wraps

def profile(func):
    """Decorator for profiling time + peak memory of similarity lookups."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        tracemalloc.start()
        t0 = time.perf_counter()

        result = func(*args, **kwargs)

        elapsed = time.perf_counter() - t0
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        print(f"[PROFILE] Time: {elapsed*1000:.3f} ms")
        print(f"[PROFILE] Memory peak: {peak/1024:.1f} KB")

        return result
    return wrapper
