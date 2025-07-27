# Knowledge base management utilities for lightHouse Connect: modules/knowledge_base_manager.py
import os
import tempfile
import zipfile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader
)

def load_documents_from_zip(zip_file_path):
    """
    Extracts and loads documents from a ZIP archive using a temp dir.
    Supports .txt, .pdf, and .md files.
    """
    documents = []
    with zipfile.ZipFile(zip_file_path, 'r') as zf:
        with tempfile.TemporaryDirectory() as temp_dir:
            for filename in zf.namelist():
                if not filename.lower().endswith((".pdf", ".txt", ".md")):
                    continue
                try:
                    temp_file_path = os.path.join(temp_dir, os.path.basename(filename))
                    with open(temp_file_path, "wb") as f:
                        f.write(zf.read(filename))
                    if filename.endswith(".pdf"):
                        loader = PyPDFLoader(temp_file_path)
                    elif filename.endswith(".txt"):
                        loader = TextLoader(temp_file_path)
                    elif filename.endswith(".md"):
                        loader = UnstructuredMarkdownLoader(temp_file_path)
                    else:
                        continue
                    documents.extend(loader.load())
                except Exception as e:
                    print(f"Failed to process file {filename}: {e}")
    return documents

def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(documents)