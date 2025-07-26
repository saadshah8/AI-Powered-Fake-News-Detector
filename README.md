# AI‑Powered Fake News Detector

An **AI‑driven web application** for detecting misinformation by comparing news statements against **trusted and flagged sources** using **semantic search and LLM reasoning**.

---

## Features
- **News Ingestion**: Upload news articles as `trusted` or `flagged` for reference.
- **Vector Storage**: Store articles in **ChromaDB** with **HuggingFace embeddings** for semantic search.
- **Fact‑Checking**: Input news statements to get:
  - A **credibility score** (0–10)
  - A **verdict**: `TRUE`, `FALSE`, or `INCONCLUSIVE`
  - A **concise explanation**
- **Web Interface**: FastAPI‑based, simple and intuitive.

---

## Tech Stack
- **Backend**: FastAPI  
- **Vector DB**: ChromaDB  
- **Embeddings**: HuggingFace (`all-MiniLM-L6-v2`)  
- **LLM**: Groq LLaMA‑3 via LangChain  
- **Other Libraries**: LangChain, Newspaper3k, Requests, Jinja2  

---

## How It Works
- **Upload Articles**: News content is extracted, chunked, and stored with embeddings.
- **Fact‑Check** Statements: Retrieves relevant context from trusted & flagged sources and evaluates with LLM reasoning.
- **Output**: Provides a score, verdict, and short explanation.
