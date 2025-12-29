#!/bin/bash

set -e

echo "🚀 Setting up Data Analyst Forecasting Project..."

if ! command -v uv &> /dev/null
then
    echo "📦 Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

echo "📚 Installing dependencies..."
uv sync --group dev

echo "✅ Setup complete!"
echo ""
echo "To run the project:"
echo "  uv run python main.py"
echo "  or"
echo "  make run"

