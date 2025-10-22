# rag_setup.py
import os
from dotenv import load_dotenv
import weaviate
from llama_index.core import VectorStoreIndex, StorageContext, Settings, Document
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llm_setup import make_llm, make_embed
from urllib.parse import urlparse
from llama_index.core import VectorStoreIndex, StorageContext, Document
from llama_index.vector_stores.weaviate import WeaviateVectorStore
import weaviate

load_dotenv()

def make_weaviate_client() -> weaviate.WeaviateClient:
    url = os.getenv("WEAVIATE_URL", "http://localhost:8080")
    u = urlparse(url)
    http_secure = (u.scheme == "https")
    http_host = u.hostname or "localhost"
    http_port = u.port or (443 if http_secure else 8080)

    # If you didn’t expose gRPC, we’ll still pass defaults; most basic ops work over HTTP.
    grpc_host = http_host
    grpc_port = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))
    grpc_secure = http_secure

    return weaviate.connect_to_custom(
        http_host=http_host,
        http_port=http_port,
        http_secure=http_secure,
        grpc_host=grpc_host,
        grpc_port=grpc_port,
        grpc_secure=grpc_secure,
    )


def make_rag_index() -> VectorStoreIndex:
    client = make_weaviate_client()
    vs = WeaviateVectorStore(weaviate_client=client, index_name="HackathonDocs")
    # ✅ create a handle backed by the existing Weaviate class
    return VectorStoreIndex.from_vector_store(vs)

def ingest_texts(texts: list[str]) -> None:
    client = make_weaviate_client()
    try:
        vs = WeaviateVectorStore(weaviate_client=client, index_name="HackathonDocs")
        storage_context = StorageContext.from_defaults(vector_store=vs)
        docs = [Document(text=t) for t in texts]
        VectorStoreIndex.from_documents(docs, storage_context=storage_context, show_progress=False)
    finally:
        client.close()  # avoids the “connection not closed” warning
