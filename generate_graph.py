#
# -------------------- generate_graph.py --------------------
#
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import TokenTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import Neo4jGraph

# Load environment variables from .env file
load_dotenv()

# The path now points to the directory containing the files.
SOURCE_DIRECTORY_PATH = "data/"

def generate_knowledge_graph():
    """
    Connects to Neo4j, loads all documents from a directory, and uses a guided LLM to transform them
    into a structured knowledge graph, storing it in the database.
    """
    # 1. Connect to Neo4j
    graph = Neo4jGraph()
    print("Connected to Neo4j database.")

    # Optional: Clear the database for a fresh start
    graph.query("MATCH (n) DETACH DELETE n")
    print("Cleared existing graph data.")

    # 2. Initialize the LLM and the Graph Transformer
    print("Initializing LLM and Graph Transformer...")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0)

    # --- START OF IMPROVEMENTS ---p

    # Define a specific schema to guide the LLM's extraction process.
    # This makes the graph more consistent and predictable.
    allowed_nodes = [
        "Policy", "Rule", "Membership", "Party", "Guest",
        "Item", "Payment", "Action", "Condition", "Location"
    ]

    allowed_relationships = [
        "APPLIES_TO", "CONCERNS", "PROHIBITS", "REQUIRES",
        "INCLUDES", "HAS_CONDITION", "MUST_PERFORM", "HAS_FEE"
    ]
    
    # Custom instructions to guide the LLM
    instructions = (
        "You are an expert at creating knowledge graphs.\n"
        "Your task is to extract entities and relationships from the provided text according to the defined schema.\n"
        "- Only extract entities and relationships that fit the provided schema.\n"
        "- Focus on identifying specific rules, policies, and the items or actions they relate to.\n"
        "- For example, if a text says 'Outside food is not allowed', you should extract a 'Rule' node connected to an 'Item' node ('Outside food') via a 'PROHIBITS' relationship.\n"
        "- If a policy requires a payment, create a 'REQUIRES' relationship between the 'Policy' and a 'Payment' node.\n"
        "- Be precise and do not add any information that is not explicitly mentioned in the text."
    )

    llm_transformer = LLMGraphTransformer(
        llm=llm,
        allowed_nodes=allowed_nodes,
        allowed_relationships=allowed_relationships,
        additional_instructions=instructions,
        strict_mode=True # Enforces that only the defined schema is used
    )

    # --- END OF IMPROVEMENTS ---

    # 3. Load and process documents from the source directory
    print(f"Loading documents from: {SOURCE_DIRECTORY_PATH}")
    
    all_graph_documents = []

    for filename in os.listdir(SOURCE_DIRECTORY_PATH):
        file_path = os.path.join(SOURCE_DIRECTORY_PATH, filename)

        if os.path.isfile(file_path):
            print(f"Processing file: {filename}...")
            loader = TextLoader(file_path)
            documents = loader.load()
            # Experiment with chunking strategy for better context
            text_splitter = TokenTextSplitter(chunk_size=250, chunk_overlap=50) #chuncks
            docs = text_splitter.split_documents(documents)

            print(f"Transforming documents from {filename} into graph data...")
            graph_documents = llm_transformer.convert_to_graph_documents(docs)
            
            all_graph_documents.extend(graph_documents)

    # 4. Add all collected graph documents to Neo4j
    if all_graph_documents:
        print(f"Generated a total of {len(all_graph_documents)} graph documents from all files.")
        graph.add_graph_documents(all_graph_documents, baseEntityLabel=True, include_source=True)
        print("\n✅ Knowledge Graph generation complete!")
    else:
        print("No files were found in the directory to process.")

if __name__ == "__main__":
    generate_knowledge_graph()