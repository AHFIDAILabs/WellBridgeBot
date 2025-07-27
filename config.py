# Configuration settings for Lighthouse HealthConnect: config.py
import os
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_REGION = os.getenv("PINECONE_REGION")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD")  
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

PINECONE_INDEX_NAME = "lighthouse-healthconnect"

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

LLM_MODEL = "deepseek/deepseek-chat-v3-0324:free"