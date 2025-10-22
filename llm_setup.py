# llm_setup.py
import os
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

def make_llm():
    return LlamaOpenAI(
        model=os.getenv("LLM_MODEL", "llama-3.1-70b-instruct"),
        api_key=os.environ["FRIENDLI_API_KEY"],
        base_url=os.getenv("FRIENDLI_BASE_URL"),   # Friendli base URL
        temperature=0.2,
        timeout=60,
    )

def make_embed():
    return OpenAIEmbedding(
        model=os.getenv("EMBED_MODEL", "text-embedding-3-small"),
        api_key=os.environ["FRIENDLI_API_KEY"],
        base_url=os.getenv("FRIENDLI_BASE_URL"),
    )
