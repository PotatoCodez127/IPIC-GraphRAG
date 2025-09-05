#
# -------------------- ingest_data.py (v4) --------------------
#
import os
import re
import json
import hashlib
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import Neo4jGraph
from langchain_community.vectorstores import SupabaseVectorStore
from supabase.client import Client, create_client
from langchain_core.documents import Document

# Load environment variables
load_dotenv()

# --- Configuration ---
SOURCE_DIRECTORY_PATH = "data/"
TRACKING_FILE_PATH = "processed_files_log.json"
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

# --- Helper Functions for Tracking and Hashing ---
def calculate_checksum(file_path):
    """Calculates a SHA256 checksum for a file."""
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def get_processed_files_log():
    """Loads the tracking file, or returns an empty dict if it doesn't exist."""
    if os.path.exists(TRACKING_FILE_PATH):
        with open(TRACKING_FILE_PATH, 'r') as f:
            return json.load(f)
    return {}

def save_processed_files_log(log):
    """Saves the tracking file."""
    with open(TRACKING_FILE_PATH, 'w') as f:
        json.dump(log, f, indent=2)

# --- Pre-processing Functions (Unchanged) ---
def normalize_text(text):
    return re.sub(r'\s+', ' ', text).strip()

def standardize_terms(text):
    term_map = {
        r'\bplay park\b': 'IPIC Play', r'\bplay area\b': 'IPIC Play',
        r'\bgym\b': 'IPIC Active',
    }
    for old, new in term_map.items():
        text = re.sub(old, new, text, flags=re.IGNORECASE)
    return text

def advanced_ingestion_pipeline():
    """
    An advanced pipeline that tracks file changes to perform intelligent
    'upsert' and 'delete' operations on the knowledge base.
    """
    print("Starting advanced ingestion pipeline...")
    # --- 1. Initialize Connections ---
    graph = Neo4jGraph()
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    graph_generation_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # --- 2. Check for File Changes ---
    print("\nStep 2: Checking for file changes...")
    processed_log = get_processed_files_log()
    current_files = {os.path.join(SOURCE_DIRECTORY_PATH, f): calculate_checksum(os.path.join(SOURCE_DIRECTORY_PATH, f))
                     for f in os.listdir(SOURCE_DIRECTORY_PATH) if os.path.isfile(os.path.join(SOURCE_DIRECTORY_PATH, f))}

    files_to_add = {f for f in current_files if f not in processed_log}
    files_to_delete = {f for f in processed_log if f not in current_files}
    files_to_update = {f for f in current_files if f in processed_log and current_files[f] != processed_log[f]}

    if not files_to_add and not files_to_delete and not files_to_update:
        print("✅ Knowledge base is already up-to-date. No changes needed.")
        return

    # --- 3. Handle Deletions and Updates (by deleting old data first) ---
    files_requiring_deletion = files_to_delete.union(files_to_update)
    if files_requiring_deletion:
        print(f"\nStep 3: Deleting data for {len(files_requiring_deletion)} removed/updated file(s)...")
        for file_path in files_requiring_deletion:
            print(f"  - Deleting data from: {os.path.basename(file_path)}")
            # Delete from Neo4j
            graph.query("""
                MATCH (doc:Document {source: $source_path})
                OPTIONAL MATCH (doc)-[r]-(n)
                DETACH DELETE doc, n
            """, params={"source_path": file_path})
            # Delete from Supabase
            supabase.table("documents").delete().eq("metadata->>source", file_path).execute()

    # --- 4. Handle Additions and Updates (by processing files) ---
    files_to_process = files_to_add.union(files_to_update)
    if files_to_process:
        print(f"\nStep 4: Processing {len(files_to_process)} new/updated file(s)...")

        # Load, pre-process, and chunk only the necessary files
        docs_to_process = []
        for file_path in files_to_process:
            print(f"  - Loading: {os.path.basename(file_path)}")
            loader = TextLoader(file_path, encoding='utf-8')
            doc = loader.load()
            # Add file_path to metadata for later use in deletion
            for d in doc:
                d.metadata["source"] = file_path
            docs_to_process.extend(doc)
        
        preprocessed_docs = [Document(page_content=normalize_text(standardize_terms(doc.page_content)), metadata=doc.metadata) for doc in docs_to_process]
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = text_splitter.split_documents(preprocessed_docs)
        print(f"Created {len(chunks)} document chunks from changed files.")

        # Initialize Graph Transformer
        llm_transformer = LLMGraphTransformer(
            llm=graph_generation_llm,
            allowed_nodes=["Policy", "Rule", "Membership", "Party", "Guest", "Item", "Payment", "Action", "Condition", "Location"],
            allowed_relationships=["APPLIES_TO", "CONCERNS", "PROHIBITS", "REQUIRES", "INCLUDES", "HAS_CONDITION", "MUST_PERFORM", "HAS_FEE"],
            strict_mode=True
        )
        
        # Process chunks to create graph data and enriched vectors
        all_graph_documents = []
        enriched_chunks = []
        for chunk in chunks:
            graph_document = llm_transformer.convert_to_graph_documents([chunk])
            if graph_document:
                for node in graph_document[0].nodes:
                    node_text = f"A node representing a '{node.type}' named '{node.id}'."
                    node.properties["embedding"] = embeddings.embed_query(node_text)
                all_graph_documents.extend(graph_document)
            
                extracted_entities = [f"{node.type}:{node.id}" for node in graph_document[0].nodes]
                enriched_metadata = chunk.metadata.copy()
                enriched_metadata['graph_entities'] = extracted_entities
                enriched_chunks.append(Document(page_content=chunk.page_content, metadata=enriched_metadata))

        # Populate databases
        if all_graph_documents:
            graph.add_graph_documents(all_graph_documents, baseEntityLabel=True, include_source=True)
            print("Successfully populated Neo4j.")
        if enriched_chunks:
            SupabaseVectorStore.from_documents(
                documents=enriched_chunks, embedding=embeddings, client=supabase,
                table_name="documents", query_name="match_documents"
            )
            print("Successfully populated Supabase.")

    # --- 5. Update the Tracking Log ---
    print("\nStep 5: Updating file processing log...")
    new_log = {f: current_files.get(f) for f in current_files}
    save_processed_files_log(new_log)
    print("\n✅ Advanced data ingestion complete!")

if __name__ == "__main__":
    advanced_ingestion_pipeline()