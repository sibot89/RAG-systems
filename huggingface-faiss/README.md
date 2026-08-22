# RAG with Hugging Face & FAISS

A simple Retrieval-Augmented Generation (RAG) pipeline built from scratch using Hugging Face models, Sentence Transformers, FAISS, and PyMuPDF.

This project demonstrates the core components of a document-based RAG system without relying on a high-level RAG framework.

## Overview

The pipeline takes a PDF document, extracts and chunks its text, converts the chunks into vector embeddings, stores them in a FAISS index, retrieves the most relevant chunks for a user query, and uses a local language model to generate an answer from the retrieved context.

## Architecture

PDF Document → PyMuPDF → Text Chunking → Sentence Transformers → Vector Embeddings → FAISS Index → Top-K Retrieval → GPT-2 → Generated Answer

## Technologies

- Python
- PyMuPDF
- Sentence Transformers
- Hugging Face Transformers
- FAISS
- NumPy
- PyTorch

## Models

### Embedding Model

`all-MiniLM-L6-v2`

Used to convert document chunks and user queries into vector embeddings.

### Generation Model

`gpt2`

Used as a lightweight local text-generation model to generate an answer from the retrieved context.

## How It Works

### 1. Document Loading

A PDF file is loaded using PyMuPDF and its text is extracted page by page.

### 2. Text Chunking

The extracted text is divided into fixed-size chunks of approximately 500 characters.

### 3. Embedding Generation

Each document chunk is converted into a vector representation using the Sentence Transformers model `all-MiniLM-L6-v2`.

### 4. Vector Indexing

The embeddings are stored in a FAISS `IndexFlatL2` index.

### 5. Retrieval

When a user submits a question, the question is embedded using the same embedding model.

FAISS retrieves the three most similar document chunks.

### 6. Generation

The retrieved chunks are combined into a context and passed to GPT-2 together with the user's question.

The model generates an answer based on the retrieved context.

## Installation

Navigate to this directory:

`cd huggingface-faiss`

Install the dependencies:

`pip install -r requirements.txt`

## Usage

Run the application:

`python rag.py`

The program will ask for the path to a PDF file and then ask for a question about its contents.

Example:

`Enter the path to a PDF file: ./example.pdf`

`Enter your question: What is weight regularization?`

The system retrieves the most relevant document chunks and generates an answer.

## Project Structure

- `rag.py` — Main RAG pipeline
- `RAG_system_HuggingFace_FAISS.ipynb` — Original Colab implementation
- `requirements.txt` — Python dependencies
- `README.md` — Project documentation

## Limitations

This project is intentionally designed as a simple demonstration of the core RAG pipeline.

It has several limitations:

- Uses fixed-size character-based chunking.
- Uses FAISS `IndexFlatL2` without advanced indexing strategies.
- Retrieves only the top 3 chunks.
- Uses GPT-2 as a lightweight local generator.
- Does not implement conversation memory.
- Does not include reranking.
- Does not provide source citations in generated answers.
- It is not intended as a production-ready RAG system.

The purpose of the project is to demonstrate the fundamental relationship between document processing, embeddings, vector search, retrieval, and generation.

## Learning Goals

This project explores the main building blocks of a RAG system:

- Document ingestion
- Text chunking
- Embedding generation
- Vector similarity search
- FAISS indexing
- Context retrieval
- Local language-model generation

It provides a lower-level implementation to complement the framework-based RAG implementation in the parent repository.

## Related Project

This repository also contains a separate document RAG implementation using LlamaIndex, Hugging Face embeddings, and Gemini.

See the `llamaindex-document-rag` directory in the parent repository.

## License

This project is intended for educational and portfolio purposes.
