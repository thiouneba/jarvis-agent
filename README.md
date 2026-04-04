# 🤖 JARVIS — Autonomous AI Agent

> A production-ready autonomous agent built with **LangChain + Python** that combines RAG, real-time web search, and persistent memory to answer complex questions intelligently.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/LangChain-0.3-1C3C3C?style=for-the-badge&logo=chainlink&logoColor=white"/>
  <img src="https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white"/>
  <img src="https://img.shields.io/badge/FAISS-Vector_Store-blue?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/OpenAI-GPT--4o--mini-412991?style=for-the-badge&logo=openai&logoColor=white"/>
</p>

---

## 🧠 What is this?

I built JARVIS as a real-world demonstration of what an **AI engineer** actually does day-to-day: designing intelligent systems that go beyond simple chatbots.

This agent can:
- Autonomously **decide which tool to use** given a user query
- **Search the web** for real-time information (DuckDuckGo)
- **Query a local knowledge base** (RAG over PDFs, docs, notes)
- **Remember context** across a conversation (in-memory or Redis)
- **Perform calculations**, **take notes**, **check weather**, **summarize URLs**
- Expose everything via a clean **REST API**

The architecture is designed to be **extended** — adding a new tool is a single function with a decorator.

---

## 🏗️ Architecture

```
jarvis-agent/
│
├── main.py                  # Entrypoint — CLI or API server
├── config.py                # Pydantic settings (env-based)
│
├── agent/
│   ├── core.py              # JarvisAgent — orchestrates everything
│   ├── memory.py            # Sliding-window chat history (in-memory / Redis)
│   ├── rag.py               # RAG pipeline — FAISS + document loaders
│   └── tools.py             # Tool registry (6 tools)
│
├── api/
│   └── routes.py            # FastAPI REST endpoints
│
├── docs/                    # Drop your PDFs/TXTs here for RAG
├── data/                    # FAISS index is stored here
├── tests/
│   └── test_agent.py        # Unit + integration tests (pytest)
│
├── .env.example
└── requirements.txt
```

---

## 🛠️ Tools Available to the Agent

| Tool | Description |
|------|-------------|
| 🔍 `web_search` | Real-time web search via DuckDuckGo |
| 📚 `rag_search` | Semantic search over your local documents |
| 🧮 `calculator` | Safe math evaluation (AST-based, no `eval`) |
| 📝 `save_note` / `get_note` | In-session key-value memory |
| 🌦️ `get_weather` | Real-time weather via OpenWeatherMap |
| 🔗 `summarize_url` | Fetch and read any public webpage |

The agent selects tools **automatically** using OpenAI's tool-calling API — no hardcoded routing.

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone https://github.com/bassirou/jarvis-agent.git
cd jarvis-agent

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Fill in your OPENAI_API_KEY (required)
# Optionally add OPENWEATHERMAP_API_KEY
```

### 3. (Optional) Add documents for RAG

Drop any `.pdf`, `.txt`, `.md`, or `.docx` files into the `docs/` folder, then:

```bash
python main.py --ingest
```

### 4. Run the CLI agent

```bash
python main.py
```

```
╔══════════════════════════════════════════════════════╗
║         🤖  J A R V I S  A G E N T  v1.0            ║
║   Autonomous AI · RAG · Web Search · Memory          ║
╚══════════════════════════════════════════════════════╝

🧑 You: What is the capital of Senegal and what's the weather like there?

🤖 JARVIS: The capital of Senegal is Dakar. 
  Currently in Dakar: 28°C, partly cloudy, humidity 74%, wind 15 km/h.
```

### 5. Run the API server

```bash
python main.py --serve
# API: http://localhost:8000
# Docs: http://localhost:8000/docs
```

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Search the web for LangChain news today", "session_id": "demo"}'
```

---

## 🔌 API Reference

### `POST /chat`
Send a message to the agent.

```json
{
  "message": "What is the GDP of France?",
  "session_id": "user_42"
}
```

Response:
```json
{
  "output": "According to recent data, France's GDP is approximately $2.78 trillion...",
  "steps": [
    {"tool": "web_search", "input": "France GDP 2024", "output": "..."}
  ],
  "session_id": "user_42"
}
```

### `POST /ingest`
Upload a document to the knowledge base (multipart/form-data).

### `DELETE /memory/{session_id}`
Clear conversation history for a session.

### `GET /health`
Health check + model info.

---

## 🧪 Tests

```bash
pytest tests/ -v
```

Tests cover:
- Memory sliding window & persistence
- Calculator safety (no `eval`, division by zero, injection)
- Note taker CRUD
- RAG pipeline (mocked embeddings)
- Full agent chat flow (mocked LLM)

---

## ⚙️ Configuration

All settings are managed via Pydantic BaseSettings and can be overridden via `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | Required |
| `OPENAI_MODEL` | `gpt-4o-mini` | LLM model |
| `MAX_AGENT_ITERATIONS` | `8` | Max tool calls per query |
| `MAX_MEMORY_MESSAGES` | `10` | Sliding window size |
| `CHUNK_SIZE` | `1000` | RAG chunk size |
| `USE_REDIS_MEMORY` | `false` | Enable Redis-backed memory |

---

## 🔧 Extending the Agent

Adding a new tool is as simple as:

```python
# agent/tools.py

@tool
def my_new_tool(query: str) -> str:
    """
    Describe what this tool does — the agent reads this description
    to decide when to use it.
    Input: a natural language query.
    """
    # your logic here
    return result
```

Then register it in `get_all_tools()`:

```python
def get_all_tools() -> list:
    return [
        ...,
        my_new_tool,  # ← just add it here
    ]
```

That's it. No changes to the agent core needed.

---

## 🗺️ Roadmap

- [ ] LangGraph multi-agent orchestration (planner + executor)
- [ ] Streaming responses via WebSocket
- [ ] Support for HuggingFace embeddings (no-cost RAG)
- [ ] Evaluation framework (RAGAS for RAG quality)
- [ ] Docker + docker-compose setup
- [ ] Long-term memory with summarization

---

## 💡 Design Decisions

**Why FAISS instead of Chroma/Pinecone?**  
FAISS runs fully locally with no server needed, making this repo zero-infrastructure to run. For production, swapping to Chroma or Pinecone requires changing a single class.

**Why `create_openai_tools_agent` instead of ReAct?**  
OpenAI's native tool-calling is more reliable, structured, and less prone to parsing errors than ReAct's text-based approach. The agent doesn't need to format thoughts as text — it calls tools natively.

**Why Pydantic settings?**  
Type-safe, validated config with IDE autocompletion. Every setting has a clear type, default, and can be overridden via environment variables without touching code.

---

## 👤 Author

**Bassirou** — AI Engineer  
Specializing in LangChain, RAG pipelines, and production AI systems.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=flat&logo=linkedin)](https://www.linkedin.com/in/bassirou-thioune-01b9131b6/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat&logo=github)](https://github.com/thiouneba)

---

## 📄 License

MIT — feel free to use, fork, and build on this.
