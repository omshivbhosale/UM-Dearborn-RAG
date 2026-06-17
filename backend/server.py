"""
UM-Dearborn RAG - FastAPI Backend Server
"""

import os
import json
import logging
import time
from typing import Optional, List, Dict
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from rag_engine import TextChunker, VectorStore, RAGEngine, Document
from scraper_selenium import scrape_with_selenium

PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Global state
# ─────────────────────────────────────────────

vector_store: Optional[VectorStore] = None
# Per-session RAG engines so each user has their own conversation memory
_sessions: Dict[str, RAGEngine] = {}
scrape_status = {"status": "idle", "pages_scraped": 0, "message": ""}
index_status = {"status": "idle", "chunks_indexed": 0, "message": ""}


def get_engine(session_id: str) -> RAGEngine:
    if session_id not in _sessions:
        _sessions[session_id] = RAGEngine(vector_store=vector_store)
    return _sessions[session_id]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global vector_store
    logger.info("Starting UM-Dearborn RAG API...")

    chroma_dir = os.getenv("CHROMA_DIR", "data/chroma_db")
    if not os.path.isabs(chroma_dir):
        chroma_dir = str(PROJECT_ROOT / chroma_dir)

    vector_store = VectorStore(chroma_dir=chroma_dir, collection_name="umdearborn")

    if not os.getenv("GROQ_API_KEY"):
        logger.warning("GROQ_API_KEY not set — chat endpoint will not work.")

    yield
    logger.info("Shutting down RAG API.")


app = FastAPI(
    title="UM-Dearborn RAG API",
    description="Retrieval-Augmented Generation for University of Michigan-Dearborn",
    version="2.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Pydantic models
# ─────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=20)
    session_id: str = "default"


class ChatResponse(BaseModel):
    answer: str
    sources: List[dict]
    query: str
    model: str
    response_time: float


class ScrapeRequest(BaseModel):
    max_pages: int = Field(default=50, ge=1, le=5000)
    start_url: str = "https://umdearborn.edu"


class IndexRequest(BaseModel):
    scraped_file: str = ""
    chunk_size: int = Field(default=500, ge=100, le=2000)
    overlap: int = Field(default=100, ge=0, le=500)


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "name": "UM-Dearborn RAG API",
        "version": "2.1.0",
        "status": "running",
        "llm": os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"),
    }


@app.get("/health")
async def health():
    count = vector_store.count() if vector_store else 0
    return {
        "status": "healthy",
        "chunks_indexed": count,
        "rag_ready": bool(os.getenv("GROQ_API_KEY")),
        "scrape_status": scrape_status,
        "index_status": index_status,
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not os.getenv("GROQ_API_KEY"):
        raise HTTPException(status_code=503, detail="GROQ_API_KEY not set. Add it to your .env file.")
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    if len(query) > 2000:
        raise HTTPException(status_code=400, detail="Query too long (max 2000 characters).")
    if vector_store.count() == 0:
        raise HTTPException(status_code=503, detail="No documents indexed yet. Run /index first.")

    try:
        start = time.time()
        engine = get_engine(request.session_id)
        answer, chunks = engine.chat(query, top_k=request.top_k)
        elapsed = round(time.time() - start, 3)

        # Deduplicate sources by URL, keeping the highest-scoring occurrence
        seen_urls: Dict[str, dict] = {}
        for c in chunks:
            if c.doc_url not in seen_urls or c.score > seen_urls[c.doc_url]["score"]:
                seen_urls[c.doc_url] = {
                    "title": c.doc_title,
                    "url": c.doc_url,
                    "content_preview": c.text[:200] + "..." if len(c.text) > 200 else c.text,
                    "score": round(c.score, 4),
                }
        unique_sources = sorted(seen_urls.values(), key=lambda x: x["score"], reverse=True)

        return ChatResponse(
            answer=answer,
            sources=unique_sources,
            query=query,
            model=os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"),
            response_time=elapsed,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/clear-memory")
async def clear_memory(session_id: str = "default"):
    if session_id in _sessions:
        _sessions[session_id].reset_memory()
    return {"message": "Conversation memory cleared.", "session_id": session_id}


@app.post("/scrape")
async def scrape(request: ScrapeRequest, background_tasks: BackgroundTasks):
    global scrape_status
    if scrape_status["status"] == "running":
        return {"message": "Scraping already in progress.", "status": scrape_status}

    def run_scrape():
        global scrape_status
        try:
            scrape_status = {"status": "running", "pages_scraped": 0, "message": "Scraping started..."}
            data_dir = PROJECT_ROOT / "data"
            os.makedirs(data_dir, exist_ok=True)
            save_path = str(data_dir / "scraped_pages.json")
            pages = scrape_with_selenium(max_pages=request.max_pages, save_path=save_path)
            scrape_status = {
                "status": "done",
                "pages_scraped": len(pages),
                "message": f"Done. {len(pages)} pages saved.",
            }
        except Exception as e:
            scrape_status = {"status": "error", "pages_scraped": 0, "message": str(e)}

    background_tasks.add_task(run_scrape)
    return {"message": "Scraping started in background.", "status": scrape_status}


@app.get("/scrape-status")
async def get_scrape_status():
    return scrape_status


@app.post("/index")
async def index(request: IndexRequest, background_tasks: BackgroundTasks):
    global index_status
    scraped_file = request.scraped_file or str(PROJECT_ROOT / "data" / "scraped_pages.json")
    if not os.path.exists(scraped_file):
        raise HTTPException(status_code=404, detail=f"File not found: {scraped_file}. Run /scrape first.")

    if index_status["status"] == "running":
        return {"message": "Indexing already in progress.", "status": index_status}

    def run_index():
        global index_status
        try:
            index_status = {"status": "running", "chunks_indexed": 0, "message": "Loading pages..."}
            with open(scraped_file) as f:
                pages_data = json.load(f)
            index_status["message"] = f"Chunking {len(pages_data)} pages..."

            documents = [Document(url=p["url"], title=p["title"], content=p["content"]) for p in pages_data]
            chunker = TextChunker(chunk_size=request.chunk_size, overlap=request.overlap)
            chunks = chunker.chunk(documents)
            index_status["message"] = f"Indexing {len(chunks)} chunks..."

            vector_store.clear()
            vector_store.add_chunks(chunks)
            count = vector_store.count()
            index_status = {"status": "done", "chunks_indexed": count, "message": f"Done. {count} chunks indexed."}
            logger.info(f"Indexed {count} chunks.")
        except Exception as e:
            index_status = {"status": "error", "chunks_indexed": 0, "message": str(e)}
            logger.error(f"Indexing error: {e}", exc_info=True)

    background_tasks.add_task(run_index)
    return {"message": "Indexing started in background.", "status": index_status}


@app.get("/index-status")
async def get_index_status():
    return index_status


@app.get("/search")
async def search(query: str, top_k: int = 5):
    if not vector_store:
        raise HTTPException(status_code=503, detail="Vector store not ready.")
    if vector_store.count() == 0:
        raise HTTPException(status_code=503, detail="No documents indexed yet. Run /index first.")
    chunks = vector_store.query(query, top_k=max(1, min(top_k, 20)))
    return {
        "query": query,
        "results": [
            {
                "title": c.doc_title,
                "url": c.doc_url,
                "content": c.text[:300],
                "score": round(c.score, 4),
            }
            for c in chunks
        ],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000)
