# populate_kb.py: populate knowledge base from ZIP file
# populate_kb.py
import sys
import logging
import os
from pathlib import Path

# Add the current directory to Python path to ensure modules can be imported
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

from modules.knowledge_base_manager import load_documents_from_zip, chunk_documents
from modules.vector_store_manager import save_vector_store
from modules.utils import get_file_hash, save_last_kb_hash

# Set up logging with UTF-8 encoding to handle special characters
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Fix Windows console encoding issues
if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
logger = logging.getLogger(__name__)

def main(zip_path):
    """Main function to populate the knowledge base from a ZIP file."""
    try:
        # Validate input
        if not os.path.exists(zip_path):
            logger.error(f"ZIP file not found: {zip_path}")
            sys.exit(1)
        
        if not zip_path.lower().endswith('.zip'):
            logger.error(f"File must be a ZIP file: {zip_path}")
            sys.exit(1)
        
        logger.info(f"Starting knowledge base population from: {zip_path}")
        
        # Step 1: Load documents from ZIP
        logger.info("Step 1: Loading documents from ZIP file...")
        documents = load_documents_from_zip(zip_path)

        if not documents:
            logger.error("No documents found in the ZIP file or all documents failed to load.")
            logger.info("Supported file types: .pdf, .txt, .md")
            sys.exit(1)

        logger.info(f"Successfully loaded {len(documents)} documents")

        # Step 2: Chunk documents
        logger.info("Step 2: Splitting documents into chunks...")
        chunked_docs = chunk_documents(documents)
        
        if not chunked_docs:
            logger.error("Document chunking failed or resulted in no chunks.")
            sys.exit(1)
            
        logger.info(f"Created {len(chunked_docs)} chunks from documents")

        # Step 3: Save to vector store
        logger.info("Step 3: Saving chunks to vector store...")
        save_vector_store(chunked_docs)
        logger.info("Vector store update completed successfully")

        # Step 4: Save hash for future comparison
        logger.info("Step 4: Updating hash record...")
        new_hash = get_file_hash(zip_path)
        save_last_kb_hash(new_hash)
        logger.info(f"Hash record updated: {new_hash}")

        logger.info("Knowledge base population completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Error populating knowledge base: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        logger.error("Usage: python populate_kb.py <path_to_zip_file>")
        logger.info("Example: python populate_kb.py data/health_documents.zip")
        sys.exit(1)

    zip_path = sys.argv[1]
    
    # Convert to absolute path
    zip_path = os.path.abspath(zip_path)
    
    logger.info(f"Python executable: {sys.executable}")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Script location: {__file__}")
    logger.info(f"Target ZIP file: {zip_path}")
    
    main(zip_path)