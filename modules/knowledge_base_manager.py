# Knowledge base management utilities for lightHouse Connect: modules/knowledge_base_manager.py
import os
import tempfile
import zipfile
import logging
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader
)

logger = logging.getLogger(__name__)

def load_documents_from_zip(zip_file_path: str) -> List[Document]:
    """
    Extracts and loads documents from a ZIP archive using a temp dir.
    Supports .txt, .pdf, and .md files.
    """
    documents = []
    
    if not os.path.exists(zip_file_path):
        logger.error(f"ZIP file not found: {zip_file_path}")
        return documents
    
    try:
        with zipfile.ZipFile(zip_file_path, 'r') as zf:
            with tempfile.TemporaryDirectory() as temp_dir:
                for filename in zf.namelist():
                    # Skip directories and non-supported files
                    if filename.endswith('/') or not filename.lower().endswith((".pdf", ".txt", ".md")):
                        continue
                    
                    try:
                        # Extract file to temp directory
                        temp_file_path = os.path.join(temp_dir, os.path.basename(filename))
                        with open(temp_file_path, "wb") as f:
                            f.write(zf.read(filename))
                        
                        # Load document based on file type
                        if filename.lower().endswith(".pdf"):
                            loader = PyPDFLoader(temp_file_path)
                        elif filename.lower().endswith(".txt"):
                            loader = TextLoader(temp_file_path, encoding='utf-8')
                        elif filename.lower().endswith(".md"):
                            loader = UnstructuredMarkdownLoader(temp_file_path)
                        else:
                            continue
                        
                        loaded_docs = loader.load()
                        # Add source metadata
                        for doc in loaded_docs:
                            doc.metadata['source'] = filename
                        
                        documents.extend(loaded_docs)
                        logger.info(f"Loaded {len(loaded_docs)} documents from {filename}")
                        
                    except Exception as e:
                        logger.error(f"Failed to process file {filename}: {e}")
                        continue
                        
    except zipfile.BadZipFile:
        logger.error(f"Invalid ZIP file: {zip_file_path}")
    except Exception as e:
        logger.error(f"Error reading ZIP file {zip_file_path}: {e}")
    
    logger.info(f"Total documents loaded: {len(documents)}")
    return documents

def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents into smaller chunks for better retrieval.
    """
    if not documents:
        logger.warning("No documents provided for chunking")
        return []
    
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunked_docs = splitter.split_documents(documents)
        logger.info(f"Split {len(documents)} documents into {len(chunked_docs)} chunks")
        return chunked_docs
    except Exception as e:
        logger.error(f"Error chunking documents: {e}")
        raise