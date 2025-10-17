#!/bin/bash
# Clean up temporary files and caches

echo "🧹 Cleaning up temporary files..."

# Remove Python cache files
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true

# Remove test artifacts
rm -rf .pytest_cache/
rm -rf htmlcov/
rm -rf .coverage
rm -rf .mypy_cache/

# Remove temporary files
rm -rf /tmp/transformers_test_*

# Clean up logs (keep last 10)
find logs/ -name "*.log" -type f | head -n -10 | xargs rm -f 2>/dev/null || true

echo "✅ Cleanup completed"
