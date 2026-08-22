RAG Systems

A collection of small Retrieval-Augmented Generation (RAG) implementations exploring document retrieval, vector search, embeddings, and LLM-based generation.

The repository contains both a low-level RAG implementation and a framework-based document RAG implementation.

Projects
1. Hugging Face + FAISS

A lightweight RAG pipeline implemented using the core building blocks of a RAG system rather than a high-level framework.

Pipeline:
PDF → PyMuPDF → Chunking → Sentence Transformers → FAISS → Top-K Retrieval → GPT-2

Technologies:

Python
PyMuPDF
Sentence Transformers
FAISS
Hugging Face Transformers
PyTorch

View project →

2. LlamaIndex Document RAG

A document-based RAG implementation using LlamaIndex, Hugging Face embeddings, and Google's Gemini model.

The project demonstrates how a higher-level RAG framework can simplify document ingestion, indexing, retrieval, and query processing.

Technologies:

Python
LlamaIndex
Google Gemini
Hugging Face Embeddings
Vector Store Index
python-dotenv

View project →

Comparison
Project	Approach	Embeddings	Retrieval	Generation
Hugging Face + FAISS	Low-level	Sentence Transformers	FAISS	GPT-2
LlamaIndex Document RAG	Framework-based	Hugging Face	LlamaIndex	Gemini
What This Repository Demonstrates

These projects explore the main components of modern RAG systems:

Document ingestion
Text extraction
Text chunking
Embedding generation
Vector indexing
Similarity search
Context retrieval
LLM-based generation
Framework-based RAG development

The implementations range from a lower-level pipeline built directly with FAISS to a higher-level implementation using LlamaIndex.

Repository Structure

RAG-systems/

├── huggingface-faiss/
│ ├── rag.py
│ ├── RAG_system_HuggingFace_FAISS.ipynb
│ ├── requirements.txt
│ └── README.md
│
└── llamaindex-document-rag/
├── rag.py
├── requirements.txt
├── .env.example
├── data/
│ └── sample_document.txt
└── README.md

Notes

These projects are primarily educational and portfolio implementations rather than production-ready RAG systems.

The goal is to demonstrate understanding of the underlying RAG pipeline as well as experience working with established frameworks and tools.

Author

Sima Baynaqi

AI & Generative AI Engineer

GitHub: @sibot89
