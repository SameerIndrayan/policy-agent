# tools.py
import os, httpx
from llama_index.core.tools import FunctionTool

# Web fetch (super simple)
def web_fetch(url: str) -> str:
    r = httpx.get(url, timeout=20)
    r.raise_for_status()
    return r.text[:4000]  # keep it short for tokens

web_fetch_tool = FunctionTool.from_defaults(
    fn=web_fetch, name="web_fetch", description="Fetch raw HTML/text from a URL."
)

# Policy Agent (your running FastAPI)
def policy_check(text: str, policy: str|None=None) -> str:
    base = os.getenv("POLICY_API_BASE", "http://127.0.0.1:8000")
    r = httpx.post(f"{base}/evaluate", json={"text": text, "policy": policy}, timeout=30)
    r.raise_for_status()
    data = r.json()
    allowed = "Allowed" if data.get("allowed") else "Blocked"
    return f"{allowed}: {data.get('reason')}"

policy_tool = FunctionTool.from_defaults(
    fn=policy_check, name="policy_check",
    description="Evaluate text against a policy using the Policy Agent API."
)
