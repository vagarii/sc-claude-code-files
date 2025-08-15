#!/bin/bash

# Code quality check script for the RAG chatbot project
# Runs black formatter in check mode to verify code formatting

echo "🔍 Checking code formatting with black..."

# Check if code needs formatting (without making changes)
if uv run black backend/ main.py --check; then
    echo "✅ All Python files are properly formatted!"
else
    echo "❌ Some files need formatting. Run './scripts/format.sh' to fix."
    exit 1
fi