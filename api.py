# api.py
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

class EvalRequest(BaseModel):
    text: str
    policy: str | None = None

class EvalResponse(BaseModel):
    allowed: bool
    reason: str

app = FastAPI(title="Policy Agent API")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def root():
    return RedirectResponse(url="/docs")

@app.post("/evaluate", response_model=EvalResponse)
def evaluate(req: EvalRequest):
    if not (req.text or "").strip():
        return EvalResponse(allowed=False, reason="No text provided.")
    if "forbidden" in req.text.lower():
        return EvalResponse(allowed=False, reason="Contains a forbidden keyword.")
    return EvalResponse(allowed=True, reason="No issues found.")
