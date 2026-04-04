"""
Memory Manager — Persistent conversation memory using LangChain's message history.
Supports in-memory (default) and Redis backend (optional).
"""

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from config import settings
import logging

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Manages per-session chat history.
    
    - Default: In-memory (fast, ephemeral)
    - Optional: Redis-backed (persistent across restarts)
    
    Usage:
        mem = MemoryManager(session_id="user_42")
        mem.add_exchange("Hello!", "Hi there!")
        history = mem.get_history()  # returns list of BaseMessage
    """

    _store: dict[str, ChatMessageHistory] = {}  # class-level store for in-memory sessions

    def __init__(self, session_id: str = "default"):
        self.session_id = session_id

        if settings.USE_REDIS_MEMORY:
            self._init_redis()
        else:
            self._init_in_memory()

    def _init_in_memory(self):
        if self.session_id not in MemoryManager._store:
            MemoryManager._store[self.session_id] = ChatMessageHistory()
        self._history = MemoryManager._store[self.session_id]
        logger.debug(f"In-memory history initialized for session={self.session_id}")

    def _init_redis(self):
        try:
            from langchain_community.chat_message_histories import RedisChatMessageHistory
            self._history = RedisChatMessageHistory(
                session_id=self.session_id,
                url=settings.REDIS_URL,
                ttl=settings.REDIS_TTL_SECONDS,
            )
            logger.debug(f"Redis history initialized for session={self.session_id}")
        except ImportError:
            logger.warning("Redis not available, falling back to in-memory.")
            self._init_in_memory()

    def add_exchange(self, human: str, ai: str):
        """Append a human/AI message pair to history."""
        self._history.add_message(HumanMessage(content=human))
        self._history.add_message(AIMessage(content=ai))

    def get_history(self) -> list[BaseMessage]:
        """Return the last N messages (sliding window)."""
        all_messages = self._history.messages
        max_msgs = settings.MAX_MEMORY_MESSAGES * 2  # pairs
        return all_messages[-max_msgs:] if len(all_messages) > max_msgs else all_messages

    def clear(self):
        """Clear all messages for this session."""
        self._history.clear()
        logger.info(f"Cleared memory for session={self.session_id}")

    def get_summary(self) -> str:
        """Return a readable summary of message count."""
        count = len(self._history.messages)
        return f"Session '{self.session_id}': {count} messages in history."
