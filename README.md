# AI Data Analyst using RAG

A Streamlit application that lets you upload documents (CSV, Excel, PDF, Word, images, or plain text) and ask natural language questions about them. It uses a **Retrieval-Augmented Generation (RAG)** pipeline with FAISS vector search and Meta's LLaMA model via the Together AI API.

## Features

- 📂 **Multi-format uploads** — CSV, XLSX, PDF, DOCX, TXT, PNG/JPG
- 🔍 **RAG pipeline** — Sentence-transformer embeddings + FAISS vector index
- 🤖 **LLaMA 4 Maverick** — Powered by Together AI
- 📊 **Evaluation dashboard** — Faithfulness, answer relevance, retrieval metrics
- 📈 **Data visualisation** — Histogram, bar chart, scatter plot for tabular data
- 🔐 **Secure secrets** — API keys loaded from environment, never hardcoded

## Local Setup

```bash
# 1. Clone the repo
git clone https://github.com/echetan-max/Data-Analyst-Agent.git
cd Data-Analyst-Agent

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Set your Together AI API key
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit secrets.toml and replace the placeholder with your real key

# 4. Run the app
streamlit run ml.py
```

## Streamlit Cloud Deployment

1. Push this repo to GitHub (already done ✅)
2. Go to [share.streamlit.io](https://share.streamlit.io) and create a new app
3. Set **Main file path** to `ml.py`
4. Under **Advanced settings → Secrets**, add:
   ```toml
   TOGETHER_API_KEY = "your-key-here"
   ```
5. Click **Deploy**

## Project Structure

```
├── ml.py                          # Main Streamlit app
├── requirements.txt               # Python dependencies
├── packages.txt                   # System-level packages (easyocr/OpenCV)
├── .gitignore                     # Prevents secrets & cache from being committed
└── .streamlit/
    └── secrets.toml.example       # Template — copy to secrets.toml for local dev
```

## Security Notes

- **Never commit** `.streamlit/secrets.toml` — it is gitignored
- The Together AI API key must be provided via Streamlit secrets or the `TOGETHER_API_KEY` environment variable
