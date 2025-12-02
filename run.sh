#!/bin/bash

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run the FastAPI app with Uvicorn
echo "Starting Local Voice Bot (Web Interface)..."
echo "Open http://localhost:8000 in your browser."
echo "Preloading AI models - this may take 1-2 minutes on first start..."
uvicorn bot:app --host 0.0.0.0 --port 8000
