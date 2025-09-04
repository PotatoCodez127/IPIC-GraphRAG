#
# -------------------- create_vector_index.py --------------------
#
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from supabase.client import Client, create_client

# Load environment variables
load_dotenv()

SOURCE_DIRECTORY_PATH = "data/"
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

def create_vector_index():
    """
    Loads documents, splits them into chunks, creates embeddings,
    and uploads them to a Supabase vector store.
    """
    print("Starting vector index creation process...")

    # 1. Initialize Supabase client
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        print("Error: Supabase URL or Service Key is missing from .env file.")
        return
        
    print("Connecting to Supabase...")
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

    # 2. Load all documents from the data directory
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

    # 3. Split the documents into smaller chunks
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)
    chunks = text_splitter.split_documents(all_docs)
    print(f"Created {len(chunks)} document chunks.")

    # 4. Initialize the embedding model
    print("Initializing embedding model...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # 5. Create the Supabase vector store from the chunks
    print("Uploading document chunks and embeddings to Supabase... (This may take a moment)")
    
    # This will create a table named 'documents' in your Supabase DB if it doesn't exist.
    vector_store = SupabaseVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        client=supabase,
        table_name="documents", # You can name this table whatever you like
        query_name="match_documents" # Function name for similarity search
    )
    
    print("\n✅ Supabase vector store creation and upload complete!")

if __name__ == "__main__":
    create_vector_index()