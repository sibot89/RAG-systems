# Document RAG with LlamaIndex

A simple Retrieval-Augmented Generation (RAG) pipeline built with **LlamaIndex**, **Google Gemini**, and **Hugging Face embeddings**.

## Overview

This project demonstrates a basic document-based RAG workflow.

Documents are loaded from a local directory, converted into embeddings using a Hugging Face embedding model, indexed with LlamaIndex, and queried using a Gemini language model.

### Workflow

    Documents
        │
        ▼
    SimpleDirectoryReader
        │
        ▼
    Hugging Face Embeddings
        │
        ▼
    VectorStoreIndex
        │
        ▼
    Query Engine
        │
        ▼
    Gemini
        │
        ▼
    Generated Answer

## How It Works

The pipeline consists of several steps:

1. Documents are loaded from the `data/` directory.
2. LlamaIndex creates an index from the documents.
3. Hugging Face's `BAAI/bge-small-en-v1.5` model is used for embeddings.
4. The query engine retrieves relevant information from the indexed documents.
5. Google Gemini generates the final response based on the retrieved context.

## Tech Stack

- Python
- LlamaIndex
- Google Gemini
- Hugging Face
- `BAAI/bge-small-en-v1.5`
- python-dotenv

## Project Structure

    llamaindex-document-rag/
    ├── rag.py
    ├── requirements.txt
    ├── .env.example
    ├── data/
    │   └── sample_document.txt
    └── README.md

## Setup

### 1. Install dependencies

    pip install -r requirements.txt

### 2. Configure the API key

Create a `.env` file based on `.env.example`:

    GOOGLE_API_KEY=your_google_api_key_here

The `.env` file is excluded from version control.

### 3. Add documents

Place text or other supported documents inside the `data/` directory.

A small example document is included:

    data/sample_document.txt

You can replace it or add additional documents for testing.

### 4. Run the application

    python rag.py

The application loads the documents from `data/`, builds the index, and runs the example query.

## Example

The included sample document contains information about the role of AI in education.

The default query is:

    What information do these documents provide?

The system retrieves relevant content and generates an answer using Gemini.

## Key Concepts Demonstrated

This project was created to explore the fundamental components of a RAG pipeline:

- Document loading
- Text embeddings
- Vector indexing
- Semantic retrieval
- LLM-based answer generation
- Integration of LlamaIndex with Gemini
- Use of Hugging Face embedding models

## Limitations

This is a simple educational implementation rather than a production-ready RAG system.

The vector index is created in memory each time the application runs. It does not use a persistent vector database.

The system also does not currently include:

- Source citation
- Retrieval evaluation
- Persistent vector storage
- Document metadata filtering
- Advanced chunking configuration
- Reranking
- Conversational memory

## Future Improvements

Potential extensions include:

- Persistent vector storage
- ChromaDB integration
- Source-aware responses
- Retrieval evaluation
- Reranking
- Better document chunking
- Conversational question answering
- A Streamlit interface

---

Built as an exploration of **Retrieval-Augmented Generation with LlamaIndex, Gemini, and Hugging Face embeddings**.
