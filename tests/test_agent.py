"""
Tests — Unit and integration tests for JARVIS Agent components.
Run: pytest tests/ -v
"""

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage


# ─────────────────────────────────────────────
# MEMORY TESTS
# ─────────────────────────────────────────────

class TestMemoryManager:
    """Tests for the MemoryManager class."""

    def setup_method(self):
        from agent.memory import MemoryManager
        # Use unique session IDs to avoid cross-test contamination
        self.mem = MemoryManager(session_id="test_memory_01")
        self.mem.clear()

    def test_initial_history_is_empty(self):
        assert self.mem.get_history() == []

    def test_add_and_retrieve_exchange(self):
        self.mem.add_exchange("Hello", "Hi there!")
        history = self.mem.get_history()
        assert len(history) == 2
        assert isinstance(history[0], HumanMessage)
        assert isinstance(history[1], AIMessage)
        assert history[0].content == "Hello"
        assert history[1].content == "Hi there!"

    def test_clear_memory(self):
        self.mem.add_exchange("test", "response")
        self.mem.clear()
        assert self.mem.get_history() == []

    def test_sliding_window(self):
        """Memory should return only the last N exchanges."""
        from unittest.mock import patch
        with patch("agent.memory.settings") as mock_settings:
            mock_settings.USE_REDIS_MEMORY = False
            mock_settings.MAX_MEMORY_MESSAGES = 2  # keep 2 exchanges = 4 messages
            from agent.memory import MemoryManager
            mem = MemoryManager(session_id="test_window")
            mem.clear()

            for i in range(5):
                mem.add_exchange(f"msg {i}", f"reply {i}")

            history = mem.get_history()
            assert len(history) <= 4  # 2 pairs max

    def test_summary_format(self):
        self.mem.add_exchange("q", "a")
        summary = self.mem.get_summary()
        assert "test_memory_01" in summary
        assert "2" in summary


# ─────────────────────────────────────────────
# CALCULATOR TESTS
# ─────────────────────────────────────────────

class TestCalculator:
    """Tests for the safe calculator tool."""

    def setup_method(self):
        from agent.tools import calculator
        self.calc = calculator

    def test_basic_addition(self):
        result = self.calc.invoke({"expression": "2 + 3"})
        assert "5" in result

    def test_multiplication(self):
        result = self.calc.invoke({"expression": "6 * 7"})
        assert "42" in result

    def test_power(self):
        result = self.calc.invoke({"expression": "2 ** 10"})
        assert "1024" in result

    def test_complex_expression(self):
        result = self.calc.invoke({"expression": "(10 + 5) * 4 / 3"})
        assert "20" in result

    def test_modulo(self):
        result = self.calc.invoke({"expression": "17 % 5"})
        assert "2" in result

    def test_invalid_expression(self):
        result = self.calc.invoke({"expression": "import os"})
        assert "error" in result.lower()

    def test_division_by_zero(self):
        result = self.calc.invoke({"expression": "10 / 0"})
        assert "error" in result.lower()


# ─────────────────────────────────────────────
# NOTE TAKER TESTS
# ─────────────────────────────────────────────

class TestNoteTaker:
    """Tests for save/get/list note tools."""

    def setup_method(self):
        # Reset notes between tests
        import agent.tools as tools_module
        tools_module._notes.clear()
        from agent.tools import save_note, get_note, list_notes
        self.save = save_note
        self.get = get_note
        self.list = list_notes

    def test_save_and_retrieve(self):
        self.save.invoke({"key": "name", "content": "Bassirou"})
        result = self.get.invoke({"key": "name"})
        assert "Bassirou" in result

    def test_missing_key(self):
        result = self.get.invoke({"key": "nonexistent"})
        assert "No note found" in result

    def test_list_notes(self):
        self.save.invoke({"key": "city", "content": "Dakar"})
        self.save.invoke({"key": "role", "content": "AI Engineer"})
        result = self.list.invoke({"_": ""})
        assert "city" in result
        assert "role" in result

    def test_overwrite_note(self):
        self.save.invoke({"key": "status", "content": "junior"})
        self.save.invoke({"key": "status", "content": "senior"})
        result = self.get.invoke({"key": "status"})
        assert "senior" in result


# ─────────────────────────────────────────────
# RAG PIPELINE TESTS
# ─────────────────────────────────────────────

class TestRAGPipeline:
    """Tests for the RAG ingestion and search pipeline."""

    @patch("agent.rag.OpenAIEmbeddings")
    @patch("agent.rag.FAISS")
    def test_ingest_text(self, mock_faiss, mock_embeddings):
        """Test that raw text can be ingested and chunked."""
        from agent.rag import RAGPipeline

        mock_faiss.from_documents.return_value = MagicMock()
        mock_faiss.from_documents.return_value.save_local = MagicMock()
        mock_faiss.from_documents.return_value.add_documents = MagicMock()

        rag = RAGPipeline.__new__(RAGPipeline)
        rag.embeddings = MagicMock()
        rag.vectorstore = None

        from langchain_text_splitters import RecursiveCharacterTextSplitter
        rag.splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)

        with patch("agent.rag.FAISS") as mock_faiss_cls:
            mock_vs = MagicMock()
            mock_faiss_cls.from_documents.return_value = mock_vs
            rag.vectorstore = None

            chunks = rag.ingest_text("LangChain is a framework for building AI applications.", source="test")
            assert chunks >= 1

    def test_search_empty_store(self):
        """Search on empty store should return []."""
        from agent.rag import RAGPipeline
        rag = RAGPipeline.__new__(RAGPipeline)
        rag.vectorstore = None
        rag.embeddings = MagicMock()
        results = rag.search("anything")
        assert results == []


# ─────────────────────────────────────────────
# INTEGRATION TEST
# ─────────────────────────────────────────────

class TestJarvisAgentIntegration:
    """Light integration tests (mock LLM to avoid API calls)."""

    @patch("agent.core.ChatOpenAI")
    @patch("agent.core.get_all_tools")
    def test_agent_chat_flow(self, mock_tools, mock_llm):
        """Test the full chat flow with mocked LLM."""
        from agent.core import JarvisAgent

        mock_llm_instance = MagicMock()
        mock_llm.return_value = mock_llm_instance
        mock_tools.return_value = []

        with patch("agent.core.AgentExecutor") as mock_executor_cls:
            mock_executor = MagicMock()
            mock_executor.invoke.return_value = {
                "output": "Hello, I am JARVIS!",
                "intermediate_steps": [],
            }
            mock_executor_cls.return_value = mock_executor

            agent = JarvisAgent(session_id="test_integration")
            result = agent.chat("Hello!")

            assert result["output"] == "Hello, I am JARVIS!"
            assert isinstance(result["steps"], list)
            assert result["session_id"] == "test_integration"

    @patch("agent.core.ChatOpenAI")
    @patch("agent.core.get_all_tools")
    def test_memory_persists_across_turns(self, mock_tools, mock_llm):
        """Memory should accumulate messages across multiple turns."""
        from agent.core import JarvisAgent

        mock_tools.return_value = []

        with patch("agent.core.AgentExecutor") as mock_executor_cls:
            mock_executor = MagicMock()
            mock_executor.invoke.return_value = {
                "output": "Got it.",
                "intermediate_steps": [],
            }
            mock_executor_cls.return_value = mock_executor

            agent = JarvisAgent(session_id="test_memory_persist")
            agent.chat("My name is Bassirou")
            agent.chat("What is my name?")

            history = agent.memory.get_history()
            assert len(history) == 4  # 2 turns × 2 messages
