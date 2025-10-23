# agent_demo.py  (llama-index==0.10.55 style)
import os
from llama_index.core.agent import ReActAgentWorker, AgentRunner
from llama_index.core.tools import FunctionTool
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings


# optional: set your LLM via env (OPENAI_API_KEY) or use default
# Settings.llm stays default if you don't set one.
Settings.llm = OpenAI(model="gpt-4o")  # model name can be anything your endpoint understands

# Try to use your Weaviate-backed RAG query engine if available
rag_tool = None
try:
    from rag_setup import make_rag_index
    idx = make_rag_index()
    qe = idx.as_query_engine(similarity_top_k=3)

    def ask_docs(question: str) -> str:
        """Query the vector store for relevant context."""
        return qe.query(question).response

    rag_tool = FunctionTool.from_defaults(
        fn=ask_docs,
        name="ask_docs",
        description="Answer questions using the RAG index (Weaviate + LlamaIndex).",
    )
except Exception as e:
    print("[agent_demo] RAG tool not available, falling back to math tool:", e)

# Simple fallback tool so the agent can still act even if RAG isn’t up
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b

math_tool = FunctionTool.from_defaults(
    fn=multiply,
    name="multiply",
    description="Multiply two numbers (a*b).",
)

tools = [math_tool] if rag_tool is None else [rag_tool, math_tool]

# Build the ReAct agent
worker = ReActAgentWorker.from_tools(
    tools=tools,
    verbose=True,
)
agent = AgentRunner(worker)

# Demo queries
print("---- Demo 1: math tool ----")
print(agent.query("What is 7 * 6?"))

print("\n---- Demo 2: RAG tool (if available) ----")
print(agent.query("What role does Weaviate play in this system?"))
