#
# -------------------- ingest_data.py (v2) --------------------
#
import os
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
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

def unified_ingestion_pipeline():
    """
    A unified pipeline that generates a knowledge graph and an enriched vector store.
    - Extracts structured entities for the graph.
    - Adds vector embeddings to each graph node.
    - Adds extracted graph entities as metadata to the document chunks.
    """
    # --- 1. Initialize all connections (Neo4j, Supabase, LLMs) ---
    print("Step 1: Initializing connections...")
    graph = Neo4jGraph()
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    graph_generation_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # --- 2. Clear existing data for a clean slate ---
    print("Step 2: Clearing existing data...")
    graph.query("MATCH (n) DETACH DELETE n")
    supabase.table("documents").delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()

    # --- 3. Load and Chunk Documents ---
    print("Step 3: Loading and chunking documents...")
    all_docs = []
    for filename in os.listdir(SOURCE_DIRECTORY_PATH):
        file_path = os.path.join(SOURCE_DIRECTORY_PATH, filename)
        if os.path.isfile(file_path):
            loader = TextLoader(file_path, encoding='utf-8')
            documents = loader.load()
            all_docs.extend(documents)
            
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_documents(all_docs)
    print(f"Created {len(chunks)} document chunks.")

    # --- 4. Generate Graph, Enrich Metadata, and Add Node Embeddings ---
    print("Step 4: Generating graph, enriching metadata, and adding node embeddings...")
    llm_transformer = LLMGraphTransformer(
        llm=graph_generation_llm,
        allowed_nodes=[
            "Policy", "Rule", "Membership", "Party", "Guest",
            "Item", "Payment", "Action", "Condition", "Location"
        ],
        allowed_relationships=[
            "APPLIES_TO", "CONCERNS", "PROHIBITS", "REQUIRES",
            "INCLUDES", "HAS_CONDITION", "MUST_PERFORM", "HAS_FEE"
        ],
        strict_mode=True
    )
    
    all_graph_documents = []
    enriched_chunks = []

    for chunk in chunks:
        graph_document = llm_transformer.convert_to_graph_documents([chunk])
        
        if graph_document:
            # --- START: NEW LOGIC TO ADD NODE EMBEDDINGS ---
            for node in graph_document[0].nodes:
                # Create a textual representation of the node for embedding
                node_text = f"A node representing a '{node.type}' named '{node.id}'."
                # Generate and add the embedding to the node's properties
                node.properties["embedding"] = embeddings.embed_query(node_text)
            # --- END: NEW LOGIC TO ADD NODE EMBEDDINGS ---

            all_graph_documents.extend(graph_document)
        
            extracted_entities = []
            for node in graph_document[0].nodes:
                extracted_entities.append(f"{node.type}:{node.id}")
        
            enriched_metadata = chunk.metadata.copy()
            enriched_metadata['graph_entities'] = extracted_entities
        
            enriched_chunk = Document(page_content=chunk.page_content, metadata=enriched_metadata)
            enriched_chunks.append(enriched_chunk)

    # --- 5. Populate the Knowledge Stores sequentially ---
    print("Step 5: Populating Neo4j and Supabase...")
    if all_graph_documents:
        graph.add_graph_documents(all_graph_documents, baseEntityLabel=True, include_source=True)
        print("Successfully populated Neo4j Knowledge Graph with node embeddings.")

    if enriched_chunks:
        SupabaseVectorStore.from_documents(
            documents=enriched_chunks,
            embedding=embeddings,
            client=supabase,
            table_name="documents",
            query_name="match_documents"
        )
        print("Successfully populated Supabase Vector Store with enriched data.")

    print("\n✅ Unified data ingestion complete!")

if __name__ == "__main__":
    unified_ingestion_pipeline()