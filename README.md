# 🛡️ FactChk: AI-Powered Disinformation Shield

> **"Rumors spread faster than disasters. FactChk stops them cold."**

**FactChk** is a real-time misinformation detection system designed for high-stakes environments like natural disasters. It uses **Retrieval-Augmented Generation (RAG)** to cross-reference user claims against a verified knowledge base instantly.

By combining the semantic memory of **Qdrant** with the reasoning power of **Google Gemini**, FactChk delivers accurate, context-aware verdicts to stop the spread of hoaxes.

---

## 🚀 Key Features

* **Context-Aware Verification:** Uses semantic search (not just keywords) to understand the *meaning* behind a claim.
* **Real-Time Verdicts:** Delivers "True," "False," or "Uncertain" analysis in <2 seconds.
* **Hybrid Intelligence:** Combines a Vector Database (Qdrant) for facts with Generative AI (Gemini) for reasoning.
* **Cloud-Ready:** Runs entirely in memory (RAM), making it deployable on free tiers like Streamlit Cloud.

---

## 💻  Setup Instructions

Follow these commands in your terminal to initialize the system.

### 1. System Initialization
```bash
# Clone the repository
git clone [https://github.com/YOUR_USERNAME/FactChk.git](https://github.com/YOUR_USERNAME/FactChk.git)
cd FactChk
```

Initialize Virtual Environment (Windows)
```bash
python -m venv venv
.\venv\Scripts\activate
```

Initialize Virtual Environment (Mac/Linux)
```bash
python3 -m venv venv
source venv/bin/activate
```
### 2.Install dependencies
```bash
# Clone the repository
pip install -r requirements.txt
```
### 3.Configure Security Protocols
```bash
# Create the secret configuration directory
mkdir .streamlit

# Create the secrets file (Replace YOUR_KEY_HERE with actual key)
echo 'GEMINI_API_KEY = "AIzaSyD_YOUR_KEY_HERE"' > .streamlit/secrets.toml

# Verify the file is ignored by Git (Crucial for security)
cat .gitignore
# Output should include: .streamlit/secrets.toml
```
### 4.Launch the System
```bash
streamlit run src/app.py
```
### ☁️ Deployment Guide (Streamlit Cloud)
FactChk is designed to be deployed instantly on Streamlit Cloud.

1.Push Code: git push origin main .

2.New App: Go to share.streamlit.io -> "New App".

3.Settings: Point to src/app.py.

4.Secrets: Go to "Advanced Settings" -> "Secrets" and paste your key:
```Ini,TOML
GEMINI_API_KEY = "AIzaSyD_YOUR_ACTUAL_KEY_HERE"
```
5.Deploy: The system will auto-build and go live in 60 seconds.

### Live Demo:
https://misinformation-detection-system-jhbggi7mwwpxgrkiynnhta.streamlit.app/
