# populate_kb.py
import sys
import logging
from modules.knowledge_base_manager import load_documents_from_zip, chunk_documents
from modules.vector_store_manager import save_vector_store
from modules.utils import get_file_hash, save_last_kb_hash

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main(zip_path):
    try:
        logger.info(f"Loading documents from: {zip_path}")
        documents = load_documents_from_zip(zip_path)

        if not documents:
            logger.error("No documents found in the ZIP file.")
            sys.exit(1)

        logger.info(f"{len(documents)} documents loaded. Splitting into chunks...")
        chunked_docs = chunk_documents(documents)
        logger.info(f"{len(chunked_docs)} chunks created.")

        logger.info("Saving vector store...")
        save_vector_store(chunked_docs)

        # Save new hash
        new_hash = get_file_hash(zip_path)
        save_last_kb_hash(new_hash)

        logger.info("Vector store successfully updated.")
    except Exception as e:
        logger.exception("Error populating knowledge base")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        logger.error("Usage: python populate_kb.py <path_to_zip_file>")
        sys.exit(1)

    zip_path = sys.argv[1]
    main(zip_path)