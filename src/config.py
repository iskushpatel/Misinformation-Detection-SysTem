import streamlit as st

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "fact_check_ledger"
MODEL_NAME = "all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD = 0.55

try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except FileNotFoundError:
    GEMINI_API_KEY = "MISSING_KEY"
    print("⚠️ Warning: No secrets found. Check .streamlit/secrets.toml")