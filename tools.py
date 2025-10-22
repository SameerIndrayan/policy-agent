# tools.py
from fastapi import FastAPI, Body
import requests, os

# 1) Spin-up a dummy Jira-like microservice so the demo is 100% local
app = FastAPI()
TICKETS = []

@app.post("/tickets")
def create_ticket(summary: str = Body(...), description: str = Body(...)):
    ticket = {"id": len(TICKETS)+1, "summary": summary, "description": description}
    TICKETS.append(ticket)
    return ticket

# 2) LlamaIndex Tool wrapper
from llama_index.tools import FunctionTool

def raise_ticket(summary: str, description: str) -> str:
    """Create an HR policy update ticket and return ticket id"""
    resp = requests.post("http://localhost:8001/tickets", json={"summary": summary, "description": description})
    resp.raise_for_status()
    return f"Ticket #{resp.json()['id']} created"

RaiseTicketTool = FunctionTool.from_defaults(fn=raise_ticket, name="RaiseTicketTool")
