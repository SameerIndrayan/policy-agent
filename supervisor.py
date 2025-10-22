# supervisor.py
from typing import Literal
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llm_setup import make_llm
from agents import make_rag_agent, make_web_agent, make_policy_agent

rag_agent   = make_rag_agent()
web_agent   = make_web_agent()
policy_agent= make_policy_agent()

def route_task(task: str) -> Literal["rag","web","policy"]:
    """Very simple routing heuristic via LLM."""
    llm = make_llm()
    prompt = f"""Decide which agent to use for this task: "{task}"
Choices: rag, web, policy
Return only one word."""
    return llm.complete(prompt).text.strip().lower()[:10]  # crude but works

def solve(task: str) -> str:
    dest = route_task(task)
    if dest == "web":
        return web_agent.chat(task).response
    if dest == "policy":
        return policy_agent.chat(task).response
    return rag_agent.chat(task).response  # default to RAG
