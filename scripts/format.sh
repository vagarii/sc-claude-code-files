#!/bin/bash

# Code formatting script for the RAG chatbot project
# Runs black formatter on all Python files

echo "🎨 Formatting Python code with black..."

# Run black on all Python files
uv run black backend/ main.py

echo "✅ Code formatting complete!"