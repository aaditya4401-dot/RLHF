"""Document ingestion and chunking for the RAG pipeline."""

import os
from pathlib import Path

from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter


RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def load_documents(data_dir: str | Path = RAW_DATA_DIR):
    """Load .txt and .pdf files from the data directory."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    docs = []

    # Load text files
    txt_loader = DirectoryLoader(
        str(data_dir), glob="**/*.txt", loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    docs.extend(txt_loader.load())

    # Load PDFs if any exist
    pdf_files = list(data_dir.glob("**/*.pdf"))
    if pdf_files:
        pdf_loader = DirectoryLoader(
            str(data_dir), glob="**/*.pdf", loader_cls=PyPDFLoader,
        )
        docs.extend(pdf_loader.load())

    print(f"Loaded {len(docs)} document(s) from {data_dir}")
    return docs


def chunk_documents(docs, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
    """Split documents into chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunk(s)")
    return chunks


def ingest(data_dir: str | Path = RAW_DATA_DIR, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
    """Full ingestion pipeline: load docs then chunk them."""
    docs = load_documents(data_dir)
    chunks = chunk_documents(docs, chunk_size, chunk_overlap)
    return chunks


if __name__ == "__main__":
    chunks = ingest()
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n--- Chunk {i} ---")
        print(chunk.page_content[:200])
