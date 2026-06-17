"""
UM-Dearborn RAG Engine — Groq-powered retrieval-augmented generation
"""

import os
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────

@dataclass
class Document:
    url: str
    title: str
    content: str
    metadata: Dict = field(default_factory=dict)


@dataclass
class Chunk:
    text: str
    doc_url: str
    doc_title: str
    chunk_index: int
    metadata: Dict = field(default_factory=dict)


@dataclass
class RetrievedChunk:
    text: str
    doc_url: str
    doc_title: str
    score: float
    chunk_index: int


# ─────────────────────────────────────────────
# Text Chunker
# ─────────────────────────────────────────────

class TextChunker:
    def __init__(self, chunk_size: int = 500, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, documents: List[Document]) -> List[Chunk]:
        chunks = []
        for doc in documents:
            words = doc.content.split()
            start = 0
            idx = 0
            while start < len(words):
                end = min(start + self.chunk_size, len(words))
                text = " ".join(words[start:end])
                # Prepend the page title to the first chunk so proper nouns
                # in titles (e.g. "Khalid Kattan") are found by keyword search.
                if idx == 0 and doc.title:
                    text = f"{doc.title}\n{text}"
                chunks.append(Chunk(
                    text=text,
                    doc_url=doc.url,
                    doc_title=doc.title,
                    chunk_index=idx,
                    metadata=doc.metadata
                ))
                idx += 1
                start += self.chunk_size - self.overlap
        return chunks


# ─────────────────────────────────────────────
# Vector Store
# ─────────────────────────────────────────────

class VectorStore:
    def __init__(self, chroma_dir: str = "data/chroma_db", collection_name: str = "umdearborn"):
        self.chroma_dir = chroma_dir
        self.collection_name = collection_name
        self.model = SentenceTransformer(os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
        logger.info(f"Load pretrained SentenceTransformer: {os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')}")

        Path(chroma_dir).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=chroma_dir)

        try:
            self.collection = self.client.get_collection(collection_name)
            logger.info(f"Loaded existing collection '{collection_name}'")
        except Exception:
            logger.info(f"Collection {collection_name} is not created.")
            self.collection = self.client.create_collection(
                collection_name,
                metadata={"hnsw:space": "cosine"}
            )

        count = self.collection.count()
        logger.info(f"Vector store ready. Collection: '{collection_name}' | Documents: {count}")

    def clear(self):
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"Collection '{self.collection_name}' cleared.")

    def add_chunks(self, chunks: List[Chunk], batch_size: int = 5000):
        if not chunks:
            return
        texts = [c.text for c in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=True).tolist()
        ids = [f"{c.doc_url}__chunk_{c.chunk_index}" for c in chunks]
        metadatas = [{"url": c.doc_url, "title": c.doc_title, "chunk_index": c.chunk_index} for c in chunks]
        for i in range(0, len(chunks), batch_size):
            self.collection.add(
                documents=texts[i:i+batch_size],
                embeddings=embeddings[i:i+batch_size],
                ids=ids[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size]
            )
            logger.info(f"  Indexed batch {i//batch_size + 1}: {min(i+batch_size, len(chunks))}/{len(chunks)} chunks")
        logger.info(f"Added {len(chunks)} chunks to vector store.")

    def query(self, query_text: str, top_k: int = 5, keyword_text: Optional[str] = None) -> List[RetrievedChunk]:
        embedding = self.model.encode([query_text]).tolist()

        # Hybrid: keyword search uses original user text so proper nouns (professor
        # names, etc.) are not lost when the query is rewritten for semantic search.
        # We extract: (a) any word > 9 chars, OR (b) mid-sentence capitalized words
        # > 3 chars that aren't common honorifics — these are almost always names.
        _SKIP_CAPS = {"Professor", "Doctor", "University", "College", "Department",
                      "Campus", "Michigan", "Dearborn", "What", "Who", "Where",
                      "When", "Why", "How", "Tell", "Please", "Give"}
        _SKIP_LOWER = {"professor", "doctor", "university", "college", "department",
                       "campus", "michigan", "dearborn", "research", "programs",
                       "undergraduate", "graduate", "admission", "financial",
                       "tuition", "scholarship", "everything", "about", "have"}
        keyword_chunks: List[RetrievedChunk] = []
        raw_words = (keyword_text or query_text).split()
        word_candidates = []
        for idx, w in enumerate(raw_words):
            clean = w.strip('?,."\'!:;')
            if not clean:
                continue
            is_long     = len(clean) > 9 and clean.lower() not in _SKIP_LOWER
            is_cap_name = (idx > 0 and clean[0].isupper() and len(clean) > 3
                           and clean not in _SKIP_CAPS)
            # Lowercase-typed names: mid-sentence, 5–9 chars, not a common word
            is_lower_name = (idx > 0 and clean[0].islower() and 4 < len(clean) <= 9
                             and clean.lower() not in _SKIP_LOWER)
            if is_long or is_cap_name or is_lower_name:
                # Try both as-typed AND title-cased so "kattan" → also "Kattan"
                for candidate in {clean, clean.capitalize()}:
                    word_candidates.append((len(clean), candidate))

        # Process rarest (longest) keywords first so they claim the top slots
        # before common shorter words; limit each to 2 hits to avoid flooding.
        word_candidates.sort(key=lambda x: x[0], reverse=True)
        seen_words: set = set()
        for _, word in word_candidates:
            if word in seen_words:
                continue
            seen_words.add(word)
            try:
                kw_results = self.collection.query(
                    query_embeddings=embedding,
                    n_results=2,          # 2 per keyword — specific beats generic
                    where_document={"$contains": word}
                )
                for i in range(len(kw_results["documents"][0])):
                    keyword_chunks.append(RetrievedChunk(
                        text=kw_results["documents"][0][i],
                        doc_url=kw_results["metadatas"][0][i]["url"],
                        doc_title=kw_results["metadatas"][0][i]["title"],
                        score=1 - kw_results["distances"][0][i],
                        chunk_index=kw_results["metadatas"][0][i]["chunk_index"]
                    ))
            except Exception:
                pass

        # Standard semantic search
        results = self.collection.query(query_embeddings=embedding, n_results=top_k)
        semantic_chunks: List[RetrievedChunk] = []
        for i in range(len(results["documents"][0])):
            semantic_chunks.append(RetrievedChunk(
                text=results["documents"][0][i],
                doc_url=results["metadatas"][0][i]["url"],
                doc_title=results["metadatas"][0][i]["title"],
                score=1 - results["distances"][0][i],
                chunk_index=results["metadatas"][0][i]["chunk_index"]
            ))

        # Merge: keyword hits first (guaranteed inclusion for rare proper nouns),
        # then fill remaining slots with semantic results.
        seen = set()
        merged = []
        for chunk in keyword_chunks + semantic_chunks:
            if chunk.text not in seen:
                seen.add(chunk.text)
                merged.append(chunk)

        return merged[:top_k]

    def count(self) -> int:
        return self.collection.count()


# ─────────────────────────────────────────────
# RAG Engine (Groq-powered)
# ─────────────────────────────────────────────

class RAGEngine:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.conversation_history = []

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            logger.warning("GROQ_API_KEY not set. Chat endpoint will not work.")
        self.client = Groq(api_key=api_key)
        self.model = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
        logger.info(f"RAG Engine initialized with Groq | model: {self.model}")

    def rewrite_query(self, query: str) -> str:
        """Rewrite user query for better retrieval."""
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a query rewriter for a university information system. "
                            "Rewrite the user's question to be more specific and search-friendly. "
                            "Return ONLY the rewritten query, nothing else."
                        )
                    },
                    {"role": "user", "content": query}
                ],
                max_tokens=100,
                temperature=0.0
            )
            rewritten = resp.choices[0].message.content.strip()
            logger.info(f"Query rewritten: '{query}' → '{rewritten}'")
            return rewritten
        except Exception as e:
            logger.warning(f"Query rewrite failed: {e}")
            return query

    def chat(self, user_message: str, top_k: int = 5) -> Tuple[str, List[RetrievedChunk]]:
        """Main RAG chat method."""
        # Step 1: Rewrite query
        rewritten = self.rewrite_query(user_message)

        # Step 2: Retrieve relevant chunks (keyword search uses original message to
        # preserve proper nouns that the rewriter may drop or alter)
        chunks = self.vector_store.query(rewritten, top_k=top_k, keyword_text=user_message)

        # Step 3: Build context
        context = "\n\n".join([
            f"[Source: {c.doc_title}]\n{c.text}"
            for c in chunks
        ])

        # Step 4: Build messages with memory
        system_prompt = (
            "You are a helpful AI assistant for the University of Michigan-Dearborn (UM-Dearborn). "
            "Answer questions accurately using ONLY the provided context. "
            "If the answer is not in the context, say so honestly.\n\n"
            "Formatting rules:\n"
            "- When listing people (professors, staff, etc.), list each person ONCE — never repeat the same name.\n"
            "- Use bullet points or numbered lists for clarity when listing multiple items.\n"
            "- Keep answers concise but complete.\n"
            "- Always be friendly and helpful.\n\n"
            f"CONTEXT:\n{context}"
        )

        messages = [{"role": "system", "content": system_prompt}]
        messages += self.conversation_history[-6:]  # last 3 turns
        messages.append({"role": "user", "content": user_message})

        # Step 5: Generate answer
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=1000,
            temperature=0.2
        )
        answer = resp.choices[0].message.content.strip()

        # Step 6: Update memory (cap at 40 messages = 20 turns to prevent unbounded growth)
        self.conversation_history.append({"role": "user", "content": user_message})
        self.conversation_history.append({"role": "assistant", "content": answer})
        if len(self.conversation_history) > 40:
            self.conversation_history = self.conversation_history[-40:]

        return answer, chunks

    def reset_memory(self):
        self.conversation_history = []
        logger.info("Conversation memory cleared.")
