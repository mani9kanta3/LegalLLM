#!/bin/bash
# Start both FastAPI and Streamlit

# Start FastAPI in background
uvicorn app.api:app --host 0.0.0.0 --port 8000 &

# Start Streamlit in foreground
streamlit run app/frontend.py --server.port 8501 --server.address 0.0.0.0