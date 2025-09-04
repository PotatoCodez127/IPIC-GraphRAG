#
# -------------------- create_vector_index.py --------------------
#
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()

SOURCE_DIRECTORY_PATH = "data/"
VECTOR_INDEX_PATH = "faiss_index"

def create_vector_index():
    """
    Loads documents, splits them into chunks, creates embeddings,
    and saves them to a FAISS vector store.
    """
    print("Starting vector index creation process...")
    
    # 1. Load all documents from the data directory
    print(f"Loading documents from: {SOURCE_DIRECTORY_PATH}")
    all_docs = []
    for filename in os.listdir(SOURCE_DIRECTORY_PATH):
        file_path = os.path.join(SOURCE_DIRECTORY_PATH, filename)
        if os.path.isfile(file_path):
            print(f"  - Loading {filename}")
            loader = TextLoader(file_path, encoding='utf-8')
            documents = loader.load()
            all_docs.extend(documents)

    if not all_docs:
        print("No documents found to process. Exiting.")
        return

    # 2. Split the documents into smaller chunks
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)
    chunks = text_splitter.split_documents(all_docs)
    print(f"Created {len(chunks)} document chunks.")

    # 3. Initialize the embedding model
    print("Initializing embedding model...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # 4. Create the FAISS vector store from the chunks
    print("Creating FAISS vector store from document chunks... (This may take a moment)")
    # FAISS.from_documents will calculate embeddings for each chunk and build the index
    vector_store = FAISS.from_documents(chunks, embeddings)

    # 5. Save the vector store locally
    print(f"Saving vector store to: {VECTOR_INDEX_PATH}")
    vector_store.save_local(VECTOR_INDEX_PATH)
    
    print("\n✅ Vector index creation complete!")

if __name__ == "__main__":
    create_vector_index()