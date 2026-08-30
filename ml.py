import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from docx import Document
import fitz
import easyocr
import together
import tempfile
import os
import time
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Data Analyst using RAG", layout="wide")

# ── API & Model Configuration ────────────────────────────────────────────────
# Automatically loads from Streamlit Cloud secrets or environment variables
secrets_key = ""
try:
    if "TOGETHER_API_KEY" in st.secrets:
        secrets_key = str(st.secrets["TOGETHER_API_KEY"]).strip()
except Exception:
    pass

env_key = os.environ.get("TOGETHER_API_KEY", "").strip() or secrets_key

with st.sidebar:
    st.header("⚙️ Configuration")
    
    if secrets_key:
        st.success("🔒 API Key loaded from `st.secrets`")
        api_key_input = st.text_input(
            "Override API Key (Optional)",
            value="",
            type="password",
            placeholder="Leave blank to use st.secrets key",
            help="Your API key is already configured in Streamlit Cloud Secrets."
        )
    else:
        api_key_input = st.text_input(
            "Together AI API Key",
            value=env_key,
            type="password",
            placeholder="Paste your Together AI key here...",
            help="You can also add TOGETHER_API_KEY in Streamlit Advanced Settings -> Secrets."
        )
        if not api_key_input:
            st.info("ℹ️ Enter an API key above or configure `st.secrets`.")
    
    model_options = [
        "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
    ]
    selected_model = st.selectbox(
        "LLM Model",
        options=model_options,
        index=0
    )
    
    st.markdown("---")
    st.markdown("💡 *Get your free API key at [together.ai](https://api.together.ai/)*")

TOGETHER_API_KEY = api_key_input.strip() if api_key_input else (secrets_key or env_key)
ACTIVE_MODEL = selected_model

# ── Session-state ────────────────────────────────────────────────────────────
if "eval_log" not in st.session_state:
    st.session_state.eval_log = []
if "chat_history" not in st.session_state:
    # Each entry: {"role": "user"|"assistant", "content": str, "metrics": dict|None}
    st.session_state.chat_history = []
    
# ── Cached resources ──────────────────────────────────────────────────────────
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

# NOTE: EasyOCR is loaded lazily inside read_image() to save RAM on startup.
# Streamlit Cloud free tier has only 1GB RAM — loading both OCR + sentence
# transformer at boot causes memory crashes when no image is being processed.

# ══════════════════════════════════════════════════════════════════════════════
# FILE READERS
# ══════════════════════════════════════════════════════════════════════════════
def read_txt(f):
    return f.read().decode("utf-8", errors="ignore")

def read_csv(f):
    return pd.read_csv(f)

def read_xlsx(f):
    return pd.read_excel(f)

def read_docx(f):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        tmp.write(f.read()); path = tmp.name
    doc  = Document(path)
    text = "\n".join(p.text for p in doc.paragraphs)
    os.unlink(path)
    return text

def read_pdf(f):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(f.read()); path = tmp.name
    pdf  = fitz.open(path)
    text = "".join(page.get_text() for page in pdf)
    pdf.close(); os.unlink(path)
    return text

def read_image(f):
    @st.cache_resource
    def load_ocr():
        return easyocr.Reader(['en'])
    img_bytes = f.read()
    results   = load_ocr().readtext(img_bytes)
    return "\n".join(r[1] for r in results) or "[No text detected]"

def process_upload(uploaded_file):
    ext = uploaded_file.name.lower().split('.')[-1]
    if ext == "txt":                           return read_txt(uploaded_file),  None
    if ext == "csv":                           return None, read_csv(uploaded_file)
    if ext == "xlsx":                          return None, read_xlsx(uploaded_file)
    if ext == "docx":                          return read_docx(uploaded_file), None
    if ext == "pdf":                           return read_pdf(uploaded_file),  None
    if ext in ["png", "jpg", "jpeg"]:          return read_image(uploaded_file), None
    st.error("Unsupported file type!"); st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# CHUNKING
# ══════════════════════════════════════════════════════════════════════════════
def chunk_text(text, chunk_size=800, overlap=100):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i: i + chunk_size])
    return chunks

# ── Chunking quality metrics ───────────────────────────────────────────────────
def chunking_metrics(chunks, chunk_size=800, overlap=100):
    lengths     = [len(c) for c in chunks]
    avg_len     = float(np.mean(lengths)) if lengths else 0.0
    overlap_r   = overlap / chunk_size                       # theoretical overlap ratio
    # Sentence-boundary respect: does each chunk end with punctuation?
    boundary_ok = sum(1 for c in chunks if c.rstrip()[-1:] in ".!?") / max(len(chunks), 1)
    return {
        "num_chunks":          len(chunks),
        "avg_chars_per_chunk": round(avg_len, 1),
        "overlap_ratio":       round(overlap_r, 3),
        "boundary_respect":    round(boundary_ok, 3),
    }

# ══════════════════════════════════════════════════════════════════════════════
# FAISS INDEX
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def create_faiss_index(chunks):
    embeddings = embedder.encode(chunks, convert_to_numpy=True, batch_size=32)
    dim        = embeddings.shape[1]
    index      = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, embeddings

def retrieve_top_k(query, index, chunks, embeddings, k=3):
    t0              = time.perf_counter()
    query_emb       = embedder.encode([query], convert_to_numpy=True)
    D, I            = index.search(query_emb, k)
    faiss_latency   = time.perf_counter() - t0              # seconds

    retrieved_chunks = [chunks[i] for i in I[0]]
    retrieved_embs   = embeddings[I[0]]                     # (k, dim)

    # ── Retrieval metrics ─────────────────────────────────────────────────────
    # Cosine similarity: query vs each retrieved chunk
    cos_sims = cosine_similarity(query_emb, retrieved_embs)[0]  # shape (k,)

    # Hit rate: fraction of chunks with cosine-sim > threshold (0.3)
    threshold  = 0.3
    hit_rate   = float(np.mean(cos_sims > threshold))

    # Chunk diversity: mean pairwise cosine distance among retrieved chunks
    if k > 1:
        pairwise     = cosine_similarity(retrieved_embs)
        upper_tri    = pairwise[np.triu_indices(k, k=1)]
        diversity    = float(1 - np.mean(upper_tri))
    else:
        diversity = 1.0

    retrieval_metrics = {
        "faiss_latency_ms":        round(faiss_latency * 1000, 2),
        "cos_sim_scores":          [round(float(s), 4) for s in cos_sims],
        "avg_cos_sim":             round(float(np.mean(cos_sims)), 4),
        "hit_rate":                round(hit_rate, 3),
        "chunk_diversity":         round(diversity, 4),
        "mrr":                     round(1 / (np.argmax(cos_sims) + 1), 4),
    }

    context = "\n".join(retrieved_chunks)
    return context, retrieval_metrics, retrieved_chunks

# ══════════════════════════════════════════════════════════════════════════════
# LLaMA QUERY
# ══════════════════════════════════════════════════════════════════════════════
def llama_query(prompt, context):
    full_prompt = f"""Use the following context to answer accurately.
Context:
{context}
Question:
{prompt}
"""
    t0 = time.perf_counter()
    if not TOGETHER_API_KEY:
        return "⚠️ Error: Together AI API Key is missing. Please enter your API key in the sidebar.", 0.0

    try:
        client = together.Together(api_key=TOGETHER_API_KEY)
        resp = client.chat.completions.create(
            model=ACTIVE_MODEL,
            messages=[{"role": "user", "content": full_prompt}]
        )
        llm_latency = time.perf_counter() - t0
        answer = resp.choices[0].message.content.strip()
        return answer, llm_latency
    except together.APIStatusError as e:
        llm_latency = time.perf_counter() - t0
        if "402" in str(e) or "credit_limit" in str(e).lower():
            err_msg = (
                "⚠️ **Together AI Credit Limit Exceeded (Error 402)**\n\n"
                "The current API key has run out of free credits.\n\n"
                "**How to fix:**\n"
                "1. Sign up for a free account at [together.ai](https://api.together.ai/).\n"
                "2. Copy your new API key from Settings → API Keys.\n"
                "3. Paste it in the **Sidebar ⚙️ Configuration** on the left."
            )
        elif "401" in str(e) or "auth" in str(e).lower():
            err_msg = "⚠️ **Invalid API Key (Error 401)**. Please verify your Together AI API Key in the sidebar."
        else:
            err_msg = f"⚠️ **Together AI API Error**: {str(e)}"
        return err_msg, llm_latency
    except Exception as e:
        llm_latency = time.perf_counter() - t0
        return f"⚠️ **Generation Error**: {str(e)}", llm_latency

# ══════════════════════════════════════════════════════════════════════════════
# GENERATION METRICS (no reference needed)
# ══════════════════════════════════════════════════════════════════════════════
def generation_metrics(query, answer, retrieved_chunks, llm_latency_s):
    """
    Faithfulness  — keyword overlap between answer and context (proxy).
    Answer relevance — cosine similarity between query embedding and answer embedding.
    """
    # Faithfulness: what fraction of answer tokens appear in the retrieved context?
    context_text = " ".join(retrieved_chunks).lower()
    answer_tokens = set(answer.lower().split())
    context_tokens = set(context_text.split())
    overlap = answer_tokens & context_tokens
    faithfulness = len(overlap) / max(len(answer_tokens), 1)

    # Answer relevance: embedding cosine similarity (query ↔ answer)
    q_emb = embedder.encode([query],  convert_to_numpy=True)
    a_emb = embedder.encode([answer], convert_to_numpy=True)
    answer_relevance = float(cosine_similarity(q_emb, a_emb)[0][0])

    # Answer length
    word_count = len(answer.split())

    return {
        "faithfulness":       round(faithfulness,      4),
        "answer_relevance":   round(answer_relevance,  4),
        "llm_latency_ms":     round(llm_latency_s * 1000, 2),
        "answer_word_count":  word_count,
    }

# ══════════════════════════════════════════════════════════════════════════════
# DATA QUALITY METRICS (CSV / XLSX path)
# ══════════════════════════════════════════════════════════════════════════════
def data_quality_metrics(df_raw, df_clean):
    total_cells      = df_raw.shape[0] * df_raw.shape[1]
    null_cells_raw   = int(df_raw.isnull().sum().sum())
    null_rate        = null_cells_raw / max(total_cells, 1)

    rows_dropped     = df_raw.shape[0] - df_clean.shape[0]
    cols_dropped     = df_raw.shape[1] - df_clean.shape[1]

    numeric_cols     = df_clean.select_dtypes(include="number").shape[1]
    total_cols_clean = max(df_clean.shape[1], 1)
    type_infer_rate  = numeric_cols / total_cols_clean

    return {
        "rows_raw":          df_raw.shape[0],
        "cols_raw":          df_raw.shape[1],
        "null_rate":         round(null_rate,       4),
        "rows_dropped":      rows_dropped,
        "cols_dropped":      cols_dropped,
        "numeric_col_ratio": round(type_infer_rate, 4),
    }

# ══════════════════════════════════════════════════════════════════════════════
# RENDER EVALUATION DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
def render_eval_dashboard():
    if not st.session_state.eval_log:
        st.info("No evaluation data yet — ask a question first.")
        return

    last = st.session_state.eval_log[-1]

    st.subheader("📊 Evaluation Dashboard")

    # ── Row 1: generation ────────────────────────────────────────────────────
    gen = last.get("generation", {})
    ret = last.get("retrieval",  {})
    sys_m = {
        "total_latency_ms": round(
            ret.get("faiss_latency_ms", 0) + gen.get("llm_latency_ms", 0), 1
        )
    }

    cols = st.columns(4)
    cols[0].metric("Faithfulness",       f"{gen.get('faithfulness',      0):.2%}",
                   help="Keyword overlap between answer and retrieved context")
    cols[1].metric("Answer Relevance",   f"{gen.get('answer_relevance',  0):.2%}",
                   help="Cosine similarity between query and answer embeddings")
    cols[2].metric("Avg Retrieval Sim",  f"{ret.get('avg_cos_sim',       0):.2%}",
                   help="Mean cosine similarity of retrieved chunks to query")
    cols[3].metric("Total Latency",      f"{sys_m['total_latency_ms']} ms",
                   help="FAISS + LLM combined latency")

    # ── Row 2: retrieval detail ───────────────────────────────────────────────
    cols2 = st.columns(4)
    cols2[0].metric("Hit Rate",         f"{ret.get('hit_rate',         0):.2%}",
                    help="Fraction of chunks with cosine sim > 0.3")
    cols2[1].metric("MRR",              f"{ret.get('mrr',              0):.3f}",
                    help="Mean Reciprocal Rank (position of best chunk)")
    cols2[2].metric("Chunk Diversity",  f"{ret.get('chunk_diversity',  0):.3f}",
                    help="1 − mean pairwise cosine sim among retrieved chunks")
    cols2[3].metric("FAISS Latency",    f"{ret.get('faiss_latency_ms', 0)} ms")

    # ── Per-chunk similarity bar chart ────────────────────────────────────────
    cos_sims = ret.get("cos_sim_scores", [])
    if cos_sims:
        fig, ax = plt.subplots(figsize=(5, 2.5))
        bars = ax.barh(
            [f"Chunk {i+1}" for i in range(len(cos_sims))],
            cos_sims, color="#5DCAA5", edgecolor="white"
        )
        ax.set_xlim(0, 1)
        ax.set_xlabel("Cosine Similarity")
        ax.set_title("Retrieved chunk relevance to query", fontsize=10)
        for bar, val in zip(bars, cos_sims):
            ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ── Historical trend (if multiple queries) ────────────────────────────────
    if len(st.session_state.eval_log) > 1:
        st.markdown("**Metric trend across queries**")
        hist_df = pd.DataFrame([
            {
                "Query #":           i + 1,
                "Faithfulness":      e["generation"]["faithfulness"],
                "Answer Relevance":  e["generation"]["answer_relevance"],
                "Avg Cos Sim":       e["retrieval"]["avg_cos_sim"],
            }
            for i, e in enumerate(st.session_state.eval_log)
            if "generation" in e and "retrieval" in e
        ])
        if not hist_df.empty:
            fig2, ax2 = plt.subplots(figsize=(6, 3))
            for col, color in zip(
                ["Faithfulness", "Answer Relevance", "Avg Cos Sim"],
                ["#5DCAA5", "#7F77DD", "#EF9F27"]
            ):
                ax2.plot(hist_df["Query #"], hist_df[col], marker="o",
                         label=col, color=color)
            ax2.set_xlabel("Query #")
            ax2.set_ylim(0, 1)
            ax2.legend(fontsize=8)
            ax2.set_title("Evaluation metrics over time", fontsize=10)
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)

    # ── Raw log export ────────────────────────────────────────────────────────
    with st.expander("Raw evaluation log (JSON)"):
        import json
        st.json(st.session_state.eval_log[-1])

# ══════════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════════
st.title("AI Data Analyst using RAG")

uploaded = st.file_uploader(
    "Upload file",
    type=["txt", "csv", "xlsx", "docx", "pdf", "png", "jpg", "jpeg"]
)

data_text, data_df, data_df_raw = None, None, None
index, chunks, embeddings       = None, None, None
chunk_stats, dq_stats           = {}, {}

# ── File Processing ───────────────────────────────────────────────────────────
if uploaded:
    data_text, data_df = process_upload(uploaded)

    # ── Tabular path ──────────────────────────────────────────────────────────
    if data_df is not None:
        data_df_raw = data_df.copy()

        data_df = data_df.dropna(how="all").dropna(axis=1, how="all")
        for c in data_df.columns:
            try:
                data_df[c] = pd.to_numeric(data_df[c])
            except Exception:
                pass

        dq_stats = data_quality_metrics(data_df_raw, data_df)

        st.subheader("Data Preview")
        st.dataframe(data_df.head())

        # Data quality card
        with st.expander("🔍 Data Quality Metrics"):
            dq_cols = st.columns(3)
            dq_cols[0].metric("Null Rate",          f"{dq_stats['null_rate']:.2%}")
            dq_cols[1].metric("Rows Dropped",       dq_stats["rows_dropped"])
            dq_cols[2].metric("Cols Dropped",       dq_stats["cols_dropped"])
            dq_cols2 = st.columns(3)
            dq_cols2[0].metric("Numeric Col Ratio", f"{dq_stats['numeric_col_ratio']:.2%}")
            dq_cols2[1].metric("Rows (raw)",        dq_stats["rows_raw"])
            dq_cols2[2].metric("Cols (raw)",        dq_stats["cols_raw"])

        context = data_df.to_string(index=False)

        st.divider()
        st.header("Data Visualization")

        numeric_cols     = data_df.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = data_df.select_dtypes(include=["object", "category"]).columns.tolist()

        tab1, tab2, tab3 = st.tabs(["Histogram", "Bar Chart", "Scatter Plot"])

        # Histogram ────────────────────────────────────────────────────────────
        with tab1:
            if not numeric_cols:
                st.info("No numeric columns available")
            else:
                hist_col = st.selectbox("Numeric Column", numeric_cols)
                bins     = st.slider("Bins", 5, 100, 30)
                if st.button("Plot Histogram"):
                    fig, ax = plt.subplots()
                    ax.hist(data_df[hist_col].dropna(), bins=bins, edgecolor="black")
                    ax.set_title(f"Histogram of {hist_col}")
                    ax.set_xlabel(hist_col); ax.set_ylabel("Frequency")
                    st.pyplot(fig)

        # Bar Chart ────────────────────────────────────────────────────────────
        with tab2:
            if not categorical_cols or not numeric_cols:
                st.info("Need categorical + numeric column")
            else:
                cat_col = st.selectbox("Category", categorical_cols)
                num_col = st.selectbox("Value", numeric_cols)
                agg     = st.selectbox("Aggregation", ["mean", "sum", "count"])
                if st.button("Plot Bar Chart"):
                    grouped = data_df.groupby(cat_col)[num_col]
                    plot_data = {"mean": grouped.mean, "sum": grouped.sum,
                                 "count": grouped.count}[agg]()
                    fig, ax = plt.subplots()
                    plot_data.plot(kind="bar", ax=ax)
                    ax.set_title(f"{agg} of {num_col} by {cat_col}")
                    st.pyplot(fig)

        # Scatter Plot ─────────────────────────────────────────────────────────
        with tab3:
            if len(numeric_cols) < 2:
                st.info("Need at least two numeric columns")
            else:
                x = st.selectbox("X Axis", numeric_cols)
                y = st.selectbox("Y Axis", [c for c in numeric_cols if c != x])
                if st.button("Plot Scatter"):
                    fig, ax = plt.subplots()
                    ax.scatter(data_df[x], data_df[y], alpha=0.7)
                    ax.set_xlabel(x); ax.set_ylabel(y)
                    ax.set_title(f"{y} vs {x}")
                    st.pyplot(fig)

    # ── Text path ─────────────────────────────────────────────────────────────
    else:
        st.subheader("Text Preview")
        st.code(data_text[:1000])

        chunks       = chunk_text(data_text)
        chunk_stats  = chunking_metrics(chunks)
        index, embeddings = create_faiss_index(tuple(chunks))

        # Chunking quality card
        with st.expander("🔍 Chunking Quality Metrics"):
            ck_cols = st.columns(4)
            ck_cols[0].metric("Chunks",              chunk_stats["num_chunks"])
            ck_cols[1].metric("Avg Chars / Chunk",   chunk_stats["avg_chars_per_chunk"])
            ck_cols[2].metric("Overlap Ratio",        f"{chunk_stats['overlap_ratio']:.2%}")
            ck_cols[3].metric("Boundary Respect",     f"{chunk_stats['boundary_respect']:.2%}",
                              help="Fraction of chunks ending with sentence-terminal punctuation")

# ══════════════════════════════════════════════════════════════════════════════
# CHAT HISTORY STYLES
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
.chat-wrapper { display: flex; flex-direction: column; gap: 12px; margin-bottom: 8px; }
.chat-bubble {
    padding: 12px 16px;
    border-radius: 16px;
    max-width: 80%;
    font-size: 0.95rem;
    line-height: 1.5;
    word-wrap: break-word;
}
.user-row   { display: flex; justify-content: flex-end; }
.ai-row     { display: flex; justify-content: flex-start; }
.user-bubble {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    border-bottom-right-radius: 4px;
}
.ai-bubble {
    background: #f0f2f6;
    color: #1a1a2e;
    border-bottom-left-radius: 4px;
    border: 1px solid #e0e0e0;
}
.bubble-label {
    font-size: 0.75rem;
    color: #888;
    margin-bottom: 3px;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# QUESTION & ANSWER
# ══════════════════════════════════════════════════════════════════════════════
st.divider()

chat_header_col, clear_col = st.columns([5, 1])
chat_header_col.header("💬 Chat with your Data")
if clear_col.button("🗑️ Clear Chat", use_container_width=True):
    st.session_state.chat_history = []
    st.session_state.eval_log = []
    st.rerun()

# ── Render full chat history ───────────────────────────────────────────────────
if st.session_state.chat_history:
    st.markdown('<div class="chat-wrapper">', unsafe_allow_html=True)
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="user-row">'
                f'<div>'
                f'<div class="bubble-label" style="text-align:right">You</div>'
                f'<div class="chat-bubble user-bubble">{msg["content"]}</div>'
                f'</div></div>',
                unsafe_allow_html=True
            )
        else:
            metrics = msg.get("metrics")
            metrics_html = ""
            if metrics:
                metrics_html = (
                    f'<div style="font-size:0.75rem;color:#888;margin-top:8px;">'
                    f'Faithfulness: <b>{metrics["faithfulness"]:.0%}</b> · '
                    f'Relevance: <b>{metrics["answer_relevance"]:.0%}</b> · '
                    f'Latency: <b>{metrics["llm_latency_ms"]} ms</b>'
                    f'</div>'
                )
            st.markdown(
                f'<div class="ai-row">'
                f'<div>'
                f'<div class="bubble-label">🤖 AI Analyst</div>'
                f'<div class="chat-bubble ai-bubble">{msg["content"]}{metrics_html}</div>'
                f'</div></div>',
                unsafe_allow_html=True
            )
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("Upload a file above, then ask your first question below!")

# ── Input row ─────────────────────────────────────────────────────────────────
input_col, btn_col = st.columns([5, 1])
query = input_col.text_input(
    "Ask a question about your data…",
    placeholder="e.g. What is the average sales by region?",
    label_visibility="collapsed"
)
ask_clicked = btn_col.button("Ask ➤", use_container_width=True)

if ask_clicked and query:
    if data_df is None and not data_text:
        st.warning("⚠️ Upload a file first.")
        st.stop()

    # ── Add user message to history ───────────────────────────────────────────
    st.session_state.chat_history.append({"role": "user", "content": query})

    if data_df is not None:
        # ── Tabular path ──────────────────────────────────────────────────────
        context = data_df.to_string(index=False)
        with st.spinner("Generating answer…"):
            answer, llm_lat = llama_query(query, context)
        fake_chunks = [context[:800]]
        gen_m = generation_metrics(query, answer, fake_chunks, llm_lat)
        ret_m = {
            "faiss_latency_ms": 0,
            "cos_sim_scores":   [],
            "avg_cos_sim":      0,
            "hit_rate":         0,
            "chunk_diversity":  0,
            "mrr":              0,
        }

    else:
        # ── RAG path ──────────────────────────────────────────────────────────
        with st.spinner("Retrieving & generating…"):
            context, ret_m, retrieved = retrieve_top_k(
                query, index, chunks, embeddings, k=3
            )
            answer, llm_lat = llama_query(query, context)
            gen_m = generation_metrics(query, answer, retrieved, llm_lat)

    # ── Add AI message to history ─────────────────────────────────────────────
    st.session_state.chat_history.append({
        "role":    "assistant",
        "content": answer,
        "metrics": gen_m,
    })

    # ── Store in eval log ─────────────────────────────────────────────────────
    st.session_state.eval_log.append({
        "query":        query,
        "answer":       answer,
        "retrieval":    ret_m,
        "generation":   gen_m,
        "data_quality": dq_stats if dq_stats else {},
        "chunking":     chunk_stats if chunk_stats else {},
    })

    st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION DASHBOARD (always visible once a query has been made)
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.header("Evaluation Dashboard")
render_eval_dashboard()