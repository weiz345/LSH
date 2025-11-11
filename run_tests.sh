#!/bin/bash
# run_tests.sh
# Runs all embedding/similarity pipeline tests in order.

set -e  # stop immediately if any test fails

echo "=== Running test_pipeline.py ==="
python3 test_pipeline.py
echo

echo "=== Running test_similarity.py ==="
python3 test_similarity.py
echo

echo "=== Running test_embedding_all.py ==="
python3 test_embedding_all.py
echo

echo "✅ All tests completed successfully!"
