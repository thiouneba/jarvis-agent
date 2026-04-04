"""
API Layer — FastAPI REST interface for JARVIS Agent.

Endpoints:
    POST /chat             → Send a message, get a response
    POST /ingest           → Add documents to the knowledge base
    DELETE /memory/{id}    → Clear session memory
    GET  /health           → Health check
    GET  /sessions/{id}    → Session info
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import logging
import tempfile
import os
import shutil

from agent.core import JarvisAgent
from agent.rag import RAGPipeline
from config import settings

# ── Logging ──────────────────────────────────────
logging.basicConfig(
    level=settings.LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────
app = FastAPI(
    title=settings.APP_TITLE,
    description="Autonomous AI Agent with RAG, Web Search, and Memory",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Agent registry (one per session) ─────────────
_agents: dict[str, JarvisAgent] = {}

def get_or_create_agent(session_id: str) -> JarvisAgent:
    if session_id not in _agents:
        _agents[session_id] = JarvisAgent(
            session_id=session_id,
            verbose=settings.DEBUG,
        )
    return _agents[session_id]


# ── Schemas ───────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000, example="What is LangChain?")
    session_id: str = Field(default="default", example="user_42")

class ChatResponse(BaseModel):
    output: str
    steps: list[dict]
    session_id: str

class IngestResponse(BaseModel):
    status: str
    chunks_added: int
    filename: str

class HealthResponse(BaseModel):
    status: str
    model: str
    sessions_active: int


# ── Routes ────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse, tags=["Agent"])
async def chat(request: ChatRequest):
    """
    Send a message to the JARVIS agent and get an intelligent response.
    The agent automatically selects and uses the best tools for your query.
    """
    agent = get_or_create_agent(request.session_id)
    try:
        result = agent.chat(request.message)
        return ChatResponse(**result)
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest", response_model=IngestResponse, tags=["Knowledge Base"])
async def ingest_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Upload a document (PDF, TXT, MD, DOCX) to the agent's knowledge base.
    The document will be chunked and embedded for RAG retrieval.
    """
    allowed_types = {".pdf", ".txt", ".md", ".docx"}
    ext = os.path.splitext(file.filename)[-1].lower()

    if ext not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {allowed_types}"
        )

    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        rag = RAGPipeline()
        chunks = rag.ingest_directory(os.path.dirname(tmp_path))
        return IngestResponse(
            status="success",
            chunks_added=chunks,
            filename=file.filename,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)


@app.delete("/memory/{session_id}", tags=["Memory"])
async def clear_memory(session_id: str):
    """Clear the conversation history for a given session."""
    if session_id in _agents:
        _agents[session_id].reset_memory()
        return {"status": "cleared", "session_id": session_id}
    raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")


@app.get("/sessions/{session_id}", tags=["Memory"])
async def get_session(session_id: str):
    """Get memory summary for a session."""
    if session_id not in _agents:
        raise HTTPException(status_code=404, detail="Session not found.")
    agent = _agents[session_id]
    return {"summary": agent.memory.get_summary()}


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Returns API health status."""
    return HealthResponse(
        status="ok",
        model=settings.OPENAI_MODEL,
        sessions_active=len(_agents),
    )
