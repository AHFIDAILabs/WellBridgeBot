# Vector store management utilities for Lighthouse HealthConnect: modules/vector_store_manager.py
import logging
from typing import List
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import Pinecone as LangchainPinecone
from pinecone import Pinecone, ServerlessSpec
from langchain_core.documents import Document

from modules.knowledge_base_manager import chunk_documents
from config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_REGION,
    PINECONE_CLOUD,
    EMBEDDING_MODEL
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

EMBEDDING_DIM = 384
TEXT_KEY = "text"

_pinecone_client = None # Global Pinecone client instance

def init_pinecone():
    """
    Initializes the Pinecone client and ensures the index exists.
    Returns the Pinecone index object.
    """
    global _pinecone_client
    if _pinecone_client is None:
        try:
            _pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
            logger.info("Pinecone client initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone client: {e}")
            raise

    try:
        indexes = _pinecone_client.list_indexes().names()

        if PINECONE_INDEX_NAME not in indexes:
            logger.info(f"Creating new Pinecone index: {PINECONE_INDEX_NAME}")
            _pinecone_client.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=EMBEDDING_DIM,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=PINECONE_CLOUD,
                    region=PINECONE_REGION,
                )
            )
            logger.info(f"Pinecone index '{PINECONE_INDEX_NAME}' created.")
        else:
            logger.info(f"Using existing Pinecone index: {PINECONE_INDEX_NAME}")

        return _pinecone_client.Index(PINECONE_INDEX_NAME)

    except Exception as e:
        logger.error(f"Failed to connect to Pinecone or manage index '{PINECONE_INDEX_NAME}': {e}")
        raise

def get_vector_store():
    """
    Returns an initialized LangchainPinecone vector store instance.
    This function can be used for querying the vector store.
    """
    try:
        index = init_pinecone()
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        vector_store = LangchainPinecone(index=index, embedding=embeddings, text_key=TEXT_KEY)
        logger.info(f"Successfully connected to LangchainPinecone vector store for index: {PINECONE_INDEX_NAME}")
        return vector_store
    except Exception as e:
        logger.error(f"Could not get LangchainPinecone vector store: {e}")
        raise

def save_vector_store(documents: List[Document]):
    """
    Saves a list of Langchain Document objects to the Pinecone vector store using batching.
    """
    if not documents:
        logger.info("No documents provided to save to Pinecone. Skipping.")
        return

    try:
        index = init_pinecone() # Get the Pinecone index object

        # Chunk all documents first
        chunks = chunk_documents(documents)
        
        if not chunks:
            logger.warning("Document chunking resulted in no chunks. No documents to save.")
            return

        logger.info(f"Preparing to save {len(chunks)} chunks to Pinecone.")

        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        
        # Initialize the LangchainPinecone vector store wrapper once
        vector_store = LangchainPinecone(index=index, embedding=embeddings, text_key=TEXT_KEY)

        # Define a batch size for upserting
        # Pinecone recommends 100 for optimal performance, though up to 1000 is allowed per call
        # For memory, 100-200 is a good starting point if you have many large chunks or metadata
        BATCH_SIZE = 100

        # Batch the documents and add them
        total_chunks_saved = 0
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i:i + BATCH_SIZE]
            # The add_documents method of LangchainPinecone handles embedding and upserting the batch
            vector_store.add_documents(batch)
            total_chunks_saved += len(batch)
            logger.info(f"Saved batch {i//BATCH_SIZE + 1} of {len(chunks) // BATCH_SIZE + 1} ({total_chunks_saved}/{len(chunks)} chunks saved)...")

        logger.info(f"Successfully saved all {total_chunks_saved} document chunks to Pinecone index '{PINECONE_INDEX_NAME}'.")

    except Exception as e:
        logger.error(f"Failed to save documents to Pinecone index '{PINECONE_INDEX_NAME}': {e}")
        raise