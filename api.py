#
# -------------------- api.py --------------------
#
from fastapi import FastAPI
from pydantic import BaseModel
from supabase.client import Client, create_client
import os

from core_logic import initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import PostgreChatMessageHistory

# Initialize FastAPI app
app = FastAPI(
    title="Sparky AI Agent API",
    description="An API to interact with the Sparky AI agent for IPIC.",
    version="1.0.0"
)

# --- Supabase Connection ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# Define the request body for the API endpoint
class ChatRequest(BaseModel):
    conversation_id: str # e.g., the user's WhatsApp number
    query: str

@app.post("/chat")
def chat_with_agent(request: ChatRequest):
    """
    Main endpoint to chat with the agent.
    Handles loading and saving conversation history for each user.
    """
    try:
        # 1. Create a PostgreChatMessageHistory object for the specific conversation
        # This object knows how to talk to our Supabase table.
        message_history = PostgreChatMessageHistory(
            connection_string=f"postgresql://postgres:{os.getenv('SUPABASE_DB_PASSWORD')}@{os.getenv('SUPABASE_DB_HOST')}:{os.getenv('SUPABASE_DB_PORT')}/postgres", # You'll need to add these to your .env
            session_id=request.conversation_id,
            table_name="conversation_history",
        )

        # 2. Create the memory buffer
        memory = ConversationBufferMemory(
            memory_key="history",
            chat_memory=message_history,
            return_messages=True
        )

        # 3. Initialize the agent with the user's specific memory
        agent_executor = initialize_agent(memory)

        # 4. Invoke the agent with the new query
        response = agent_executor.invoke({"input": request.query})

        return {"response": response["output"]}

    except Exception as e:
        # It's good practice to handle errors and return a proper HTTP status
        return {"error": f"An error occurred: {str(e)}"}, 500

# To run this API, save it as api.py and run from your terminal:
# uvicorn api:app --reload