# rag_setup.py
import os
from urllib.parse import urlparse

import weaviate
from llama_index.core import Document, StorageContext, VectorStoreIndex, Settings
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI


WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
WEAVIATE_INDEX = os.getenv("WEAVIATE_INDEX", "HackathonDocs")


def make_weaviate_client():
    """
    Weaviate **v3** client (you have 3.26.7 in this venv).
    """
    u = urlparse(WEAVIATE_URL)
    # v3 uses weaviate.Client(url=...) — no grpc args
    return weaviate.Client(url=f"{u.scheme}://{u.hostname}:{u.port or 8080}")


def _vector_store():
    client = make_weaviate_client()
    return WeaviateVectorStore(
        weaviate_client=client,
        index_name=WEAVIATE_INDEX,
    )


def make_rag_index():
    """
    Reconnects to the existing Weaviate collection and builds a VectorStoreIndex on top.
    """
    # LlamaIndex 0.10.x needs explicit defaults
    Settings.llm = OpenAI(model="gpt-4o")  # or gpt-4o-mini if your 0.10 build supports it
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

    vs = _vector_store()
    storage_context = StorageContext.from_defaults(vector_store=vs)

    # empty docs here: we’re just attaching to the existing store
    return VectorStoreIndex.from_documents([], storage_context=storage_context)


def ingest_texts(texts):
    """
    Create/attach the Weaviate collection and ingest raw strings.
    """
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

    vs = _vector_store()
    storage_context = StorageContext.from_defaults(vector_store=vs)

    docs = [Document(text=t) for t in texts]  # <-- correct ctor is Document(text=...)
    # In 0.10.x, insertion happens during index construction:
    VectorStoreIndex.from_documents(docs, storage_context=storage_context)
