"""OpenAI API response generation with retrieved context."""

import os

from dotenv import load_dotenv
from openai import OpenAI

from src.rag.retriever import load_vectorstore, retrieve


load_dotenv()

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def get_client():
    """Initialize the OpenAI client."""
    return OpenAI()  # uses OPENAI_API_KEY from env


def format_context(docs) -> str:
    """Format retrieved documents into a context string."""
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[Document {i} | source: {source}]\n{doc.page_content}")
    return "\n\n".join(parts)


SYSTEM_PROMPT = """You are a helpful assistant. Answer the user's question based on the provided context documents.
If the context doesn't contain enough information to answer, say so clearly.
Always ground your answers in the provided context."""


def generate(query: str, vectorstore=None, model: str = MODEL, k: int = 4) -> dict:
    """Retrieve context and generate a response using OpenAI."""
    # Retrieve relevant chunks
    if vectorstore is None:
        vectorstore = load_vectorstore()
    docs = retrieve(query, vectorstore=vectorstore, k=k)
    context = format_context(docs)

    # Build messages
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {query}",
        },
    ]

    # Call OpenAI
    client = get_client()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
    )

    answer = response.choices[0].message.content

    return {
        "query": query,
        "answer": answer,
        "context_docs": docs,
        "model": model,
    }


if __name__ == "__main__":
    query = "What is this document about?"
    result = generate(query)
    print(f"Query: {result['query']}")
    print(f"Model: {result['model']}")
    print(f"\nAnswer:\n{result['answer']}")
