# agent_demo.py
import os
import httpx

from llama_index.core import VectorStoreIndex
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI
from llama_index.core.agent.react import ReActAgentWorker
from llama_index.core.agent.runner import AgentRunner
from llama_index.core.tools import FunctionTool



from rag_setup import make_rag_index

# --- LLM (OpenAI-compatible; works with Friendli if you set OPENAI_BASE_URL + OPENAI_API_KEY) ---
llm = OpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    api_base=os.getenv("OPENAI_BASE_URL"),   # e.g. https://inference.friendli.ai/v1
    api_key=os.getenv("OPENAI_API_KEY"),
)

# --- Tool: call your local FastAPI policy evaluator ---
def evaluate_text(text: str, policy: str | None = None) -> dict:
    """Call the /evaluate endpoint of the local policy service."""
    url = os.getenv("POLICY_API", "http://127.0.0.1:8000/evaluate")
    with httpx.Client(timeout=10.0) as client:
        r = client.post(url, json={"text": text, "policy": policy})
        r.raise_for_status()
        return r.json()

evaluate_tool = FunctionTool.from_defaults(
    fn=evaluate_text,
    name="evaluate_text",
    description="Evaluate a string against a policy. Args: text (str), policy (str or null).",
)

# --- RAG retriever from your Weaviate-backed index ---
idx: VectorStoreIndex = make_rag_index()
retriever = idx.as_retriever(similarity_top_k=3)

# --- Agent (new API in LlamaIndex 0.14.x) ---
worker = ReActAgentWorker.from_tools(
    tools=[evaluate_tool],
    llm=llm,
    verbose=True,
    context=(
        "You are a helpful assistant. Use the retriever for context and call "
        "'evaluate_text' to check policy compliance when asked."
    ),
    retriever=retriever,
)
agent = AgentRunner(worker)

if __name__ == "__main__":
    q = "Draft a one-liner about Weaviate's role, then check it against the 'forbidden' policy."
    resp = agent.query(q)
    print("\n--- AGENT ANSWER ---\n", resp.response)
