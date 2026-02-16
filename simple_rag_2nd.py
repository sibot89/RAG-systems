import os
import asyncio
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import google.generativeai as genai

load_dotenv()

def run_rag_competition(
    document_dir: str,
    query_text: str,
    generative_model: str = "gemini-1.5-flash"
) -> str:
    
    api_key = os.getenv("AIzaSyBC3vJv7TZMP4eaGKU_g5gFMQ08J8Kc5SU")
    if not api_key:
        raise ValueError("API KEY ERROR")
    
    genai.configure(api_key=api_key)
    
    llm = Gemini(
        model=generative_model,
        api_key=api_key
    )
    
    embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )
    
    Settings.llm = llm
    Settings.embed_model = embed_model
    
    documents = SimpleDirectoryReader(document_dir).load_data()
    index = VectorStoreIndex.from_documents(documents)
    response = index.as_query_engine().aquery(query_text)
    
    return str(response)

document_dir = "./data"
query_text = "how is the weather"
print("////////////////")
responce = run_rag_competition(document_dir, query_text)
print("////////////////")
print(responce, "************")
print("////////////////")

