# UM-Dearborn AI Assistant

A Retrieval-Augmented Generation (RAG) chatbot for the University of Michigan-Dearborn. Scrapes the official university website, embeds the content locally, and answers questions using a free Groq-hosted LLM — no OpenAI required.

## Architecture

```
frontend/index.html   ←  single-page chat UI
backend/server.py     ←  FastAPI REST API
backend/rag_engine.py ←  chunker · ChromaDB vector store · Groq LLM
backend/scraper_selenium.py  ←  curl_cffi concurrent scraper
```

**Stack:** Python · FastAPI · ChromaDB · sentence-transformers (`all-MiniLM-L6-v2`) · Groq (Llama 3) · vanilla HTML/CSS/JS

## Quickstart

### 1. Clone & install

```bash
git clone <repo-url>
cd Final_Project
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your free Groq API key (https://console.groq.com)
```

### 3. Run the backend

```bash
cd backend
uvicorn server:app --reload --port 8000
```

### 4. Open the frontend

Open `frontend/index.html` in your browser (no build step needed).

### 5. Scrape & index

In the UI (or via curl):

```bash
# Scrape the UM-Dearborn website
curl -X POST http://localhost:8000/scrape -H "Content-Type: application/json" \
     -d '{"max_pages": 200}'

# Wait for scraping to finish, then index
curl -X POST http://localhost:8000/index
```

Once indexing is done, the chat is ready.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Health / version |
| GET | `/health` | Status + chunk count |
| POST | `/chat` | Ask a question |
| POST | `/scrape` | Scrape website (background) |
| GET | `/scrape-status` | Scrape progress |
| POST | `/index` | Embed & index pages (background) |
| GET | `/index-status` | Index progress |
| GET | `/search` | Raw vector search |
| POST | `/clear-memory` | Reset conversation history |

## Evaluation

```bash
python scripts/run_pipeline.py   # end-to-end scrape → index → sample queries
python scripts/evaluate.py       # RAGAS metrics on test_questions.json
```

Results are saved to `data/eval_report.json`.

## Project Structure

```
├── backend/
│   ├── server.py           # FastAPI app
│   ├── rag_engine.py       # Core RAG logic
│   └── scraper_selenium.py # Web scraper
├── frontend/
│   └── index.html          # Chat UI
├── scripts/
│   ├── run_pipeline.py     # End-to-end pipeline runner
│   └── evaluate.py         # RAGAS evaluation
├── data/                   # Gitignored — created at runtime
│   ├── scraped_pages.json
│   └── chroma_db/
├── doc_assets/             # Charts & diagrams
├── .env.example            # Template — copy to .env
└── requirements.txt
```
