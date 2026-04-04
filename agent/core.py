"""
JARVIS Agent Core — Autonomous multi-tool AI agent
Author: Bassirou
Stack: LangChain + Python
"""

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage

from agent.memory import MemoryManager
from agent.tools import get_all_tools
from config import settings
import logging

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are JARVIS, an intelligent autonomous AI assistant.
You have access to a set of powerful tools:

- 🔍 **Web Search**: Search the internet for up-to-date information
- 📚 **RAG Knowledge Base**: Query an internal vector store loaded with documents
- 🧮 **Calculator**: Perform precise mathematical computations
- 📝 **Note Taker**: Save and retrieve important information across conversations
- 🌦️ **Weather**: Get real-time weather data for any city
- 🔗 **URL Summarizer**: Fetch and summarize content from any URL

Guidelines:
- Always think step-by-step before choosing a tool
- Combine multiple tools when needed to give the best answer
- Be concise but thorough
- If unsure, search before answering
- Always cite your sources when using web search or RAG

Current conversation context is provided below.
"""


def build_agent(verbose: bool = False) -> AgentExecutor:
    """Build and return the JARVIS agent executor."""
    
    llm = ChatOpenAI(
        model=settings.OPENAI_MODEL,
        temperature=settings.TEMPERATURE,
        openai_api_key=settings.OPENAI_API_KEY,
    )

    tools = get_all_tools()

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_openai_tools_agent(llm=llm, tools=tools, prompt=prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=verbose,
        handle_parsing_errors=True,
        max_iterations=settings.MAX_AGENT_ITERATIONS,
        return_intermediate_steps=True,
    )


class JarvisAgent:
    """High-level wrapper around the AgentExecutor with memory management."""

    def __init__(self, session_id: str = "default", verbose: bool = False):
        self.session_id = session_id
        self.memory = MemoryManager(session_id=session_id)
        self.executor = build_agent(verbose=verbose)
        logger.info(f"JARVIS agent initialized | session={session_id}")

    def chat(self, user_input: str) -> dict:
        """
        Send a message to the agent and get a response.
        Returns a dict with 'output', 'steps', and 'history'.
        """
        chat_history = self.memory.get_history()

        logger.info(f"[{self.session_id}] User: {user_input}")

        result = self.executor.invoke({
            "input": user_input,
            "chat_history": chat_history,
        })

        output = result["output"]
        steps = result.get("intermediate_steps", [])

        # Persist to memory
        self.memory.add_exchange(
            human=user_input,
            ai=output,
        )

        logger.info(f"[{self.session_id}] Agent: {output[:100]}...")

        return {
            "output": output,
            "steps": self._format_steps(steps),
            "session_id": self.session_id,
        }

    def _format_steps(self, steps: list) -> list[dict]:
        """Format intermediate steps for readability."""
        formatted = []
        for action, observation in steps:
            formatted.append({
                "tool": action.tool,
                "input": action.tool_input,
                "output": str(observation)[:300],
            })
        return formatted

    def reset_memory(self):
        """Clear conversation history for this session."""
        self.memory.clear()
        logger.info(f"Memory cleared for session={self.session_id}")
