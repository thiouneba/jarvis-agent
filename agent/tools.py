"""
Tool Registry — All tools available to the JARVIS agent.

Tools:
    1. web_search        → DuckDuckGo real-time search
    2. rag_search        → Vector store knowledge base lookup
    3. calculator        → Safe mathematical evaluation
    4. note_taker        → Save and retrieve session notes
    5. get_weather       → OpenWeatherMap real-time weather
    6. summarize_url     → Fetch + summarize any webpage
"""

from langchain_core.tools import tool, StructuredTool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import OpenWeatherMapAPIWrapper

from agent.rag import RAGPipeline
from config import settings
import requests
import ast
import operator
import logging

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# 1. WEB SEARCH
# ─────────────────────────────────────────────

def get_web_search_tool():
    """Returns a DuckDuckGo search tool."""
    return DuckDuckGoSearchRun(
        name="web_search",
        description=(
            "Search the internet for current, real-time information. "
            "Use this for recent events, facts you are unsure about, or anything requiring up-to-date data. "
            "Input: a natural language search query."
        ),
    )


# ─────────────────────────────────────────────
# 2. RAG — VECTOR STORE SEARCH
# ─────────────────────────────────────────────

_rag_pipeline: RAGPipeline | None = None

def _get_rag_pipeline() -> RAGPipeline:
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = RAGPipeline()
    return _rag_pipeline


@tool
def rag_search(query: str) -> str:
    """
    Search the internal knowledge base for relevant documents.
    Use this tool when the user asks about topics that may be covered
    in the loaded documents (PDFs, notes, reports).
    Input: a natural language question or keyword query.
    """
    try:
        pipeline = _get_rag_pipeline()
        results = pipeline.search(query, k=4)
        if not results:
            return "No relevant documents found in the knowledge base."
        
        formatted = "\n\n---\n\n".join([
            f"📄 **Source**: {doc.metadata.get('source', 'unknown')}\n{doc.page_content}"
            for doc in results
        ])
        return f"Knowledge base results:\n\n{formatted}"
    except Exception as e:
        logger.error(f"RAG search error: {e}")
        return f"Error querying knowledge base: {str(e)}"


# ─────────────────────────────────────────────
# 3. SAFE CALCULATOR
# ─────────────────────────────────────────────

SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.Mod: operator.mod,
}

def _safe_eval(node):
    if isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.BinOp):
        op = SAFE_OPERATORS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported operator: {type(node.op)}")
        return op(_safe_eval(node.left), _safe_eval(node.right))
    elif isinstance(node, ast.UnaryOp):
        op = SAFE_OPERATORS.get(type(node.op))
        return op(_safe_eval(node.operand))
    else:
        raise ValueError(f"Unsupported expression: {type(node)}")


@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression safely.
    Supports: +, -, *, /, **, %, parentheses.
    Examples: '2 ** 10', '(3 + 5) * 12 / 4', '100 % 7'
    Input: a mathematical expression as a string.
    """
    try:
        tree = ast.parse(expression, mode='eval')
        result = _safe_eval(tree.body)
        return f"Result: {result}"
    except Exception as e:
        return f"Calculation error: {str(e)}"


# ─────────────────────────────────────────────
# 4. SESSION NOTE TAKER
# ─────────────────────────────────────────────

_notes: dict[str, str] = {}


@tool
def save_note(key: str, content: str) -> str:
    """
    Save a note with a given key for later retrieval during the conversation.
    Useful for remembering facts, user preferences, or important info.
    Input: key (short identifier) and content (what to remember).
    Example: save_note('user_name', 'Bassirou')
    """
    _notes[key] = content
    return f"✅ Note saved under key '{key}'."


@tool
def get_note(key: str) -> str:
    """
    Retrieve a previously saved note by its key.
    Returns the stored content or a message if not found.
    Input: the key string used when saving.
    """
    if key in _notes:
        return f"📝 Note '{key}': {_notes[key]}"
    return f"No note found for key '{key}'. Available keys: {list(_notes.keys())}"


@tool
def list_notes(_: str = "") -> str:
    """
    List all saved notes and their keys.
    No input needed.
    """
    if not _notes:
        return "No notes saved yet."
    return "\n".join([f"• {k}: {v}" for k, v in _notes.items()])


# ─────────────────────────────────────────────
# 5. WEATHER
# ─────────────────────────────────────────────

@tool
def get_weather(city: str) -> str:
    """
    Get the current weather for any city in the world.
    Input: a city name (e.g. 'Paris', 'Dakar', 'New York').
    Returns temperature, conditions, humidity, and wind speed.
    """
    try:
        weather = OpenWeatherMapAPIWrapper(
            openweathermap_api_key=settings.OPENWEATHERMAP_API_KEY
        )
        return weather.run(city)
    except Exception as e:
        logger.error(f"Weather tool error: {e}")
        return f"Could not retrieve weather for '{city}': {str(e)}"


# ─────────────────────────────────────────────
# 6. URL SUMMARIZER
# ─────────────────────────────────────────────

@tool
def summarize_url(url: str) -> str:
    """
    Fetch the content of a URL and return a cleaned summary of the page text.
    Useful for reading articles, documentation, or any public webpage.
    Input: a valid URL string (must start with http:// or https://).
    """
    try:
        from langchain_community.document_loaders import WebBaseLoader
        loader = WebBaseLoader(url)
        docs = loader.load()
        if not docs:
            return "Could not load content from this URL."
        
        content = docs[0].page_content[:3000]  # Limit to avoid token explosion
        return f"Content from {url}:\n\n{content}"
    except Exception as e:
        logger.error(f"URL summarizer error: {e}")
        return f"Error fetching URL: {str(e)}"


# ─────────────────────────────────────────────
# TOOL REGISTRY
# ─────────────────────────────────────────────

def get_all_tools() -> list:
    """Return all tools to be registered with the agent."""
    return [
        get_web_search_tool(),
        rag_search,
        calculator,
        save_note,
        get_note,
        list_notes,
        get_weather,
        summarize_url,
    ]
