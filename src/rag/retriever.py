"""Vector store setup and retrieval logic using ChromaDB."""

from pathlib import Path

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from src.rag.ingest import ingest


CHROMA_DIR = Path(__file__).resolve().parents[2] / "data" / "chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def get_embeddings(model_name: str = EMBEDDING_MODEL):
    """Initialize the embedding model."""
    return HuggingFaceEmbeddings(model_name=model_name)


def build_vectorstore(chunks, persist_dir: str | Path = CHROMA_DIR, embedding_model: str = EMBEDDING_MODEL):
    """Create a ChromaDB vector store from document chunks."""
    persist_dir = Path(persist_dir)
    embeddings = get_embeddings(embedding_model)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(persist_dir),
    )
    print(f"Vector store built with {len(chunks)} chunks at {persist_dir}")
    return vectorstore


def load_vectorstore(persist_dir: str | Path = CHROMA_DIR, embedding_model: str = EMBEDDING_MODEL):
    """Load an existing ChromaDB vector store from disk."""
    persist_dir = Path(persist_dir)
    if not persist_dir.exists():
        raise FileNotFoundError(
            f"No vector store found at {persist_dir}. Run build_vectorstore first."
        )

    embeddings = get_embeddings(embedding_model)
    vectorstore = Chroma(
        persist_directory=str(persist_dir),
        embedding_function=embeddings,
    )
    print(f"Loaded vector store from {persist_dir}")
    return vectorstore


def retrieve(query: str, vectorstore=None, k: int = 4):
    """Retrieve the top-k most relevant chunks for a query."""
    if vectorstore is None:
        vectorstore = load_vectorstore()

    results = vectorstore.similarity_search(query, k=k)
    return results


def ingest_and_build(data_dir=None, persist_dir: str | Path = CHROMA_DIR):
    """End-to-end: ingest documents and build the vector store."""
    kwargs = {}
    if data_dir is not None:
        kwargs["data_dir"] = data_dir
    chunks = ingest(**kwargs)
    vectorstore = build_vectorstore(chunks, persist_dir=persist_dir)
    return vectorstore


if __name__ == "__main__":
    # Build the vector store from raw documents
    vs = ingest_and_build()

    # Test a query
    query = "What is this document about?"
    results = retrieve(query, vectorstore=vs)
    print(f"\nQuery: {query}")
    for i, doc in enumerate(results):
        print(f"\n--- Result {i} ---")
        print(doc.page_content[:200])
