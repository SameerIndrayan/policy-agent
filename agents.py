# agents.py
from llama_index.core.agent import ReActAgent
from llama_index.core.query_engine import RetrieverQueryEngine
from rag_setup import make_rag_index
from tools import web_fetch_tool, policy_tool
from llm_setup import make_llm

# RAG Agent (Weaviate-backed)
def make_rag_agent():
    index = make_rag_index()
    qe = RetrieverQueryEngine.from_args(index.as_retriever(similarity_top_k=5))
    # Wrap the retriever as a tool
    def rag_query(q: str) -> str:
        return qe.query(q).response

    from llama_index.core.tools import FunctionTool
    rag_tool = FunctionTool.from_defaults(
        fn=rag_query, name="rag_search", description="Search memory (Weaviate) for relevant context."
    )
    return ReActAgent.from_tools([rag_tool], llm=make_llm(), system_prompt=(
        "You are the RAG agent. Use rag_search to answer with retrieved context."
    ))

# Web Agent
def make_web_agent():
    return ReActAgent.from_tools([web_fetch_tool], llm=make_llm(), system_prompt=(
        "You are the Web agent. Use web_fetch to retrieve pages and summarize."
    ))

# Policy Agent (wraps your API as a tool caller)
def make_policy_agent():
    return ReActAgent.from_tools([policy_tool], llm=make_llm(), system_prompt=(
        "You are the Policy agent. Use policy_check to assess compliance."
    ))
