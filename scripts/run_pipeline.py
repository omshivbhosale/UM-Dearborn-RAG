"""
UM-Dearborn RAG - CLI Pipeline
Run: python scripts/run_pipeline.py [scrape|index|chat|full]
"""

import sys
import os
import json
import argparse

# add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from rag_engine import TextChunker, VectorStore, RAGEngine, Document
from scraper_selenium import scrape_with_selenium
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))


def get_chroma_path():
    return os.path.join(os.path.dirname(__file__), '..', 'data', 'chroma_db')


def run_scrape(args):
    print(f"\n🌐 Starting crawl of {args.url} (max {args.max_pages} pages)...")
    save_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'scraped_pages.json')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    pages = scrape_with_selenium(max_pages=args.max_pages, save_path=save_path)
    print(f"✅ Scraped {len(pages)} pages → saved to {save_path}")
    return pages


def run_index(args):
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'scraped_pages.json')
    if not os.path.exists(data_path):
        print("❌ No scraped data found. Run 'scrape' first.")
        return

    print(f"\n📚 Loading pages from {data_path}...")
    with open(data_path) as f:
        pages_data = json.load(f)
    print(f"   Loaded {len(pages_data)} pages.")

    # Convert to Document objects
    documents = [Document(url=p["url"], title=p["title"], content=p["content"]) for p in pages_data]

    print(f"\n✂️  Chunking documents (size={args.chunk_size}, overlap={args.overlap})...")
    chunker = TextChunker(chunk_size=args.chunk_size, overlap=args.overlap)
    chunks = chunker.chunk(documents)
    print(f"   Created {len(chunks)} chunks.")

    chroma_path = get_chroma_path()
    print(f"\n🔢 Building vector index at {chroma_path}...")
    store = VectorStore(chroma_dir=chroma_path, collection_name="umdearborn")
    print("   Clearing old data...")
    store.clear()
    store.add_chunks(chunks)
    print(f"✅ Vector store ready. Total chunks: {store.count()}")
    return store


def run_chat(args):
    chroma_path = get_chroma_path()
    print(f"\n🤖 Loading RAG Engine...")
    store = VectorStore(chroma_dir=chroma_path, collection_name="umdearborn")
    print(f"   Vector store: {store.count()} chunks loaded.")

    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        print("❌ GROQ_API_KEY not set. Add it to your .env file.")
        return

    engine = RAGEngine(vector_store=store)
    print(f"   LLM: {os.getenv('LLM_MODEL', 'llama-3.3-70b-versatile')} (Groq)")
    print("\n" + "="*60)
    print("  UM-Dearborn AI Assistant")
    print("  Type 'quit' to exit | 'clear' to reset memory")
    print("="*60 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() == 'quit':
                print("Goodbye!")
                break
            if user_input.lower() == 'clear':
                engine.reset_memory()
                print("Memory cleared.\n")
                continue

            print("\nAssistant: ", end="", flush=True)
            answer, chunks = engine.chat(user_input)
            print(answer)

            if args.show_sources:
                print(f"\n📎 Sources ({len(chunks)}):")
                for i, c in enumerate(chunks, 1):
                    print(f"  {i}. [{c.score:.2f}] {c.doc_title}")
                    print(f"     {c.doc_url}")

            print()

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


def run_full(args):
    print("\n🚀 Running FULL pipeline: Scrape → Index → Chat\n")
    run_scrape(args)
    run_index(args)
    run_chat(args)


def main():
    parser = argparse.ArgumentParser(description="UM-Dearborn RAG Pipeline CLI")

    parser.add_argument('command', choices=['scrape', 'index', 'chat', 'full'])
    parser.add_argument('--url', default='https://umdearborn.edu')
    parser.add_argument('--max-pages', type=int, default=50)
    parser.add_argument('--chunk-size', type=int, default=500)
    parser.add_argument('--overlap', type=int, default=100)
    parser.add_argument('--show-sources', action='store_true')

    args = parser.parse_args()

    commands = {
        'scrape': run_scrape,
        'index': run_index,
        'chat': run_chat,
        'full': run_full
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
