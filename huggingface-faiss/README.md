# RAG with Hugging Face & FAISS

A simple Retrieval-Augmented Generation (RAG) pipeline built from scratch using Hugging Face models, Sentence Transformers, FAISS, and PyMuPDF.

This project demonstrates the core components of a document-based RAG system without relying on a high-level RAG framework.

## Overview

The pipeline takes a PDF document, extracts and chunks its text, converts the chunks into vector embeddings, stores them in a FAISS index, retrieves the most relevant chunks for a user query, and uses a local language model to generate an answer from the retrieved context.

## Architecture

PDF Document  
↓  
PyMuPDF  
↓  
Text Chunking  
↓  
Sentence Transformers  
↓  
Vector Embeddings  
↓  
FAISS Index  
↓  
Top-K Retrieval  
↓  
GPT-2  
↓  
Generated Answer

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

Each document chunk is converted into a vector representation using the Sentence Transformers model:

`all-MiniLM-L6-v2`

### 4. Vector Indexing

The embeddings are stored in a FAISS `IndexFlatL2` index.

### 5. Retrieval

When a user submits a question, the question is embedded using the same embedding model.

FAISS then retrieves the three most similar document chunks.

### 6. Generation

The retrieved chunks are combined into a context and passed to GPT-2 together with the user's question.

The model generates an answer based on the retrieved context.

## Installation

Clone the repository and navigate to this directory:

```bash
cd huggingface-faiss
