#!/usr/bin/env bash
set -e

SERVICE_MODE="${1:-all}"

if [ "$SERVICE_MODE" = "streamlit" ] || [ "$SERVICE_MODE" = "web" ]; then
    echo "Starting Streamlit Loan Default Platform on port 8501..."
    exec streamlit run app/streamlit_app.py --server.port=8501 --server.address=0.0.0.0
elif [ "$SERVICE_MODE" = "all" ]; then
    echo "Starting FastAPI Backend and Streamlit Portal concurrently..."
    uvicorn app.main:app --host 0.0.0.0 --port 8000 &
    exec streamlit run app/streamlit_app.py --server.port=8501 --server.address=0.0.0.0
elif [ "$SERVICE_MODE" = "pipeline" ]; then
    echo "Running Retraining Pipeline..."
    exec python run_pipeline.py
else
    echo "Starting FastAPI Loan Underwriting Service on port 8000..."
    exec uvicorn app.main:app --host 0.0.0.0 --port 8000
fi
