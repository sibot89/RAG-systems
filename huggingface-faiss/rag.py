import os

import faiss
import fitz
import numpy as np

from sentence_transformers import SentenceTransformer
from transformers import pipeline


# =========================================================
# CONFIGURATION
# =========================================================

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GENERATION_MODEL = "gpt2"

TOP_K = 3
CHUNK_SIZE = 500


# =========================================================
# DOCUMENT LOADING
# =========================================================

def load_pdf(file_path: str) -> str:
    """Extract text from a PDF file."""

    document = fitz.open(file_path)

    text = ""

    for page in document:
        text += page.get_text()

    document.close()

    return text


# =========================================================
# TEXT CHUNKING
# =========================================================

def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE
) -> list[str]:
    """Split text into fixed-size chunks."""

    chunks = []

    for start in range(
        0,
        len(text),
        chunk_size
    ):
        chunk = text[start:start + chunk_size]

        if chunk.strip():
            chunks.append(chunk)

    return chunks


# =========================================================
# VECTOR INDEX
# =========================================================

def build_faiss_index(
    chunks: list[str],
    embedding_model: SentenceTransformer
):
    """Create embeddings and build a FAISS index."""

    embeddings = embedding_model.encode(
        chunks
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    index = faiss.IndexFlatL2(
        embeddings.shape[1]
    )

    index.add(embeddings)

    return index


# =========================================================
# RETRIEVAL
# =========================================================

def retrieve(
    query: str,
    chunks: list[str],
    index,
    embedding_model: SentenceTransformer,
    top_k: int = TOP_K
) -> list[str]:
    """Retrieve the most relevant document chunks."""

    query_embedding = embedding_model.encode(
        [query]
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    _, indices = index.search(
        query_embedding,
        top_k
    )

    return [
        chunks[i]
        for i in indices[0]
        if i < len(chunks)
    ]


# =========================================================
# RAG PIPELINE
# =========================================================

def retrieve_and_generate(
    query: str,
    chunks: list[str],
    index,
    embedding_model,
    generator
) -> str:
    """Retrieve relevant context and generate an answer."""

    relevant_chunks = retrieve(
        query,
        chunks,
        index,
        embedding_model
    )

    context = "\n\n".join(
        relevant_chunks
    )

    prompt = (
        "Answer the question using the following context.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )

    response = generator(
        prompt,
        max_new_tokens=100,
        num_return_sequences=1
    )

    return response[0]["generated_text"]


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    pdf_path = input(
        "Enter the path to a PDF file: "
    ).strip()

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(
            f"PDF file not found: {pdf_path}"
        )

    print("\nLoading document...")

    pdf_text = load_pdf(pdf_path)

    document_chunks = chunk_text(
        pdf_text
    )

    print(
        f"Created {len(document_chunks)} chunks."
    )

    print("\nLoading embedding model...")

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    print("\nBuilding FAISS index...")

    index = build_faiss_index(
        document_chunks,
        embedding_model
    )

    print("\nLoading generation model...")

    generator = pipeline(
        "text-generation",
        model=GENERATION_MODEL
    )

    query = input(
        "\nEnter your question: "
    ).strip()

    response = retrieve_and_generate(
        query,
        document_chunks,
        index,
        embedding_model,
        generator
    )

    print("\n########## Generated Answer ##########\n")

    print(response)
