# rag_setup.py
import os, weaviate
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llm_setup import make_llm, make_embed

def make_rag_index():
    Settings.llm = make_llm()
    Settings.embed_model = make_embed()

    client = weaviate.Client(os.getenv("WEAVIATE_URL", "http://localhost:8080"))
    vector_store = WeaviateVectorStore(weaviate_client=client, index_name="AgentMemory")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Create an empty index (or ingest docs first)
    index = VectorStoreIndex.from_documents([], storage_context=storage_context)
    return index
