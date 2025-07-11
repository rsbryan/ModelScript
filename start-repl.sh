#!/bin/bash
# ModelScript REPL Launcher
# This script ensures the virtual environment is activated before starting the REPL

echo "🚀 Starting ModelScript REPL..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run setup first:"
    echo "  python3 -m venv venv"
    echo "  source venv/bin/activate"
    echo "  pip install -r modelscript/requirements.txt"
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Check if TensorFlow is installed
if ! python3 -c "import tensorflow" 2>/dev/null; then
    echo "⚠️  TensorFlow not found in virtual environment!"
    echo "Installing requirements..."
    pip install -r modelscript/requirements.txt
fi

# Start the REPL
echo "✅ Starting ModelScript REPL..."
python3 repl.py