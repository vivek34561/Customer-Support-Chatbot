#!/bin/bash
# Launch script for Customer Support Chatbot with Streamlit UI

echo "============================================"
echo "  Customer Support Chatbot Launcher"
echo "============================================"
echo

# Activate virtual environment if not already activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Activating virtual environment..."
    source myenv/bin/activate
    echo
fi

# Start FastAPI backend in background
echo "Starting FastAPI backend on port 8000..."
python api.py &
FASTAPI_PID=$!
sleep 5

# Start Streamlit frontend
echo "Starting Streamlit UI on port 8501..."
streamlit run streamlit_app.py

# Cleanup on exit
kill $FASTAPI_PID
