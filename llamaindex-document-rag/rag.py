import os

from dotenv import load_dotenv

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
)

from llama_index.llms.gemini import Gemini
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


# =========================================================
# CONFIGURATION
# =========================================================

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError(
        "GOOGLE_API_KEY is not set. "
        "Add it to your .env file."
    )


# =========================================================
# RAG PIPELINE
# =========================================================

def run_document_rag(
    document_dir: str,
    query_text: str,
    generative_model: str = "gemini-2.0-flash",
) -> str:
    """
    Run a simple document-based RAG pipeline.

    The pipeline:
    1. Loads documents from a directory.
    2. Creates embeddings using Hugging Face.
    3. Builds an in-memory vector index.
    4. Retrieves relevant document content.
    5. Generates an answer using Gemini.
    """

    llm = Gemini(
        model=generative_model,
        api_key=api_key,
    )

    embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )

    Settings.llm = llm
    Settings.embed_model = embed_model

    documents = SimpleDirectoryReader(
        document_dir
    ).load_data()

    index = VectorStoreIndex.from_documents(
        documents
    )

    query_engine = index.as_query_engine()

    response = query_engine.query(
        query_text
    )

    return str(response)


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    document_dir = "./data"

    query_text = (
        "What information do these documents provide?"
    )

    print(
        "\n########## Starting Document RAG ##########\n"
    )

    response = run_document_rag(
        document_dir=document_dir,
        query_text=query_text,
    )

    print(
        "\n########## Response ##########\n"
    )

    print(response)
