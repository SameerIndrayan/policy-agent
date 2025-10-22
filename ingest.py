# ingest.py
from llama_index.core import Document
from rag_setup import make_rag_index

def ingest_texts(texts: list[str]):
    index = make_rag_index()
    docs = [Document(t) for t in texts]
    index.insert_nodes(index.service_context.node_parser.get_nodes_from_documents(docs))

if __name__ == "__main__":
    ingest_texts([
        "Friendli serves open-source models behind an OpenAI-compatible API.",
        "Weaviate stores vectors; you can query with LlamaIndex retrievers."
    ])
