import streamlit as st
import pandas as pd
import json
from search import EvidenceRetriever
from explain import generate_explanation
if "cache_cleared" not in st.session_state:
    st.cache_resource.clear()
    st.session_state["cache_cleared"] = True
    print("🧹 System Cache Cleared!")
st.set_page_config(page_title="FactChk:: Disinformation Shield", page_icon="🛡️")

# ---  Load Data & Sidebar ---
try:
    with open('data/knowledge_base.json', 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    # Sidebar Stats
    st.sidebar.title("📊 Live Disaster Monitor")
    st.sidebar.metric("Verified Facts", len(df))
    if 'verdict' in df.columns:
        st.sidebar.metric("Active Hoaxes", len(df[df['verdict'] == 'False']))
    
    # Chart
    st.sidebar.subheader("Misinformation by Category")
    if 'category' in df.columns:
        category_counts = df['category'].value_counts()
        st.sidebar.bar_chart(category_counts)
except Exception as e:
    st.sidebar.error(f"Error loading data: {e}")

# ---  Main UI ---
st.title("🛡️ FactChk:: Disinformation Shield: Disinformation Shield")
st.markdown("Enter a claim below to cross-reference it with our verified knowledge base.")

# Initialize Retrieval Engine (Cached)
@st.cache_resource
def load_engine():
    return EvidenceRetriever()

try:
    retriever = load_engine()
except Exception as e:
    st.error(f"Failed to load search engine. Is Qdrant running? Error: {e}")

# User Input
user_claim = st.text_input("Enter a claim (e.g., 'The dam has collapsed!')", "")

if st.button("Verify Claim"):
    if user_claim:
        with st.spinner('Scanning verified database...'):
            results = retriever.search(user_claim)
            report = generate_explanation(user_claim, results)
            st.subheader("Analysis Report")

            # check for keywords 
            if "UNCERTAIN" in report:
                st.warning(report)
                with st.expander("🚩 Report this as a new rumor"):
                    st.write("Send this to our fact-checking team for urgent review.")
                    if st.button("Submit Report"):
                        st.success("Report submitted! ID: #99281. We are investigating.")
                        st.balloons()
                        
            elif "False" in report or "Misleading" in report or "fake" in report.lower() or "hoax" in report.lower():
                st.error(report) # Red Box
            else:
                st.success(report) # Green Box
    else:
        st.warning("Please enter some text first.")