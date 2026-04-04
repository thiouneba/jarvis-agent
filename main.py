"""
Main entrypoint — run as CLI chatbot or start the API server.

Usage:
    python main.py            # Interactive CLI
    python main.py --serve    # Start FastAPI server
    python main.py --ingest   # Ingest documents and exit
"""

import argparse
import uvicorn
import sys

from agent.core import JarvisAgent
from agent.rag import RAGPipeline
from config import settings
import logging

logging.basicConfig(
    level=settings.LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

BANNER = """
╔══════════════════════════════════════════════════════╗
║         🤖  J A R V I S  A G E N T  v1.0            ║
║   Autonomous AI · RAG · Web Search · Memory          ║
╚══════════════════════════════════════════════════════╝
  Type your message and press Enter.
  Commands: /reset · /notes · /quit
"""


def run_cli():
    """Interactive terminal chatbot."""
    print(BANNER)
    agent = JarvisAgent(session_id="cli_session", verbose=True)

    while True:
        try:
            user_input = input("\n🧑 You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye! 👋")
            sys.exit(0)

        if not user_input:
            continue

        if user_input == "/quit":
            print("Goodbye! 👋")
            break

        if user_input == "/reset":
            agent.reset_memory()
            print("🗑️  Memory cleared.")
            continue

        if user_input == "/notes":
            # Quick hack to peek at notes via tool
            result = agent.chat("list all notes")
            print(f"\n🤖 JARVIS: {result['output']}")
            continue

        result = agent.chat(user_input)

        print(f"\n🤖 JARVIS: {result['output']}")

        if result["steps"] and settings.DEBUG:
            print("\n[Debug] Tools used:")
            for step in result["steps"]:
                print(f"  → [{step['tool']}] {str(step['input'])[:80]}")


def run_server():
    """Start the FastAPI server."""
    from api.routes import app
    print(f"\n🚀 Starting JARVIS API on http://localhost:8000")
    print(f"📖 Docs: http://localhost:8000/docs\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=settings.DEBUG)


def run_ingest():
    """Ingest all documents from DOCS_DIR and exit."""
    print(f"📂 Ingesting documents from '{settings.DOCS_DIR}'...")
    rag = RAGPipeline()
    count = rag.ingest_directory(settings.DOCS_DIR)
    print(f"✅ Done! {count} chunks added to the vector store.")
    print(f"📦 Index saved at: {settings.FAISS_INDEX_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JARVIS Agent Launcher")
    parser.add_argument("--serve", action="store_true", help="Start FastAPI server")
    parser.add_argument("--ingest", action="store_true", help="Ingest docs and exit")
    args = parser.parse_args()

    if args.serve:
        run_server()
    elif args.ingest:
        run_ingest()
    else:
        run_cli()
