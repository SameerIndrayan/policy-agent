# llm_setup.py
import os
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

def make_llm():
    """
    Uses OpenAI-style env:
      OPENAI_API_KEY=sk-...
      (optional) OPENAI_BASE_URL=http://localhost:8080/v1  # Friendli proxy or self-hosted
      (optional) OPENAI_MODEL, default gpt-4o-mini
    """
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    base_url = os.getenv("OPENAI_BASE_URL")  # leave None for real OpenAI
    return OpenAI(model=model, api_base=base_url)

def make_embed():
    emb_model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    base_url = os.getenv("OPENAI_BASE_URL")
    return OpenAIEmbedding(model=emb_model, api_base=base_url)
