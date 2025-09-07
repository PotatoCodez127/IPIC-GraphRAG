#
# -------------------- api.py (Final - Using Supabase Client) --------------------
#
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from supabase.client import Client, create_client
import os
import json

from core_logic import initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseChatMessageHistory
from langchain.schema.messages import BaseMessage, messages_from_dict, messages_to_dict

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

# --- START: NEW Custom Supabase Chat History Class ---
class SupabaseChatMessageHistory(BaseChatMessageHistory):
    """
    A custom chat message history class that uses the Supabase Python client
    to store and retrieve conversation history.
    """
    def __init__(self, session_id: str, table_name: str):
        self.session_id = session_id
        self.table_name = table_name

    @property
    def messages(self):
        """Retrieve messages from Supabase."""
        response = supabase.table(self.table_name).select("history").eq("conversation_id", self.session_id).execute()
        if response.data:
            return messages_from_dict(response.data[0]['history'])
        return []

    def add_messages(self, messages: list[BaseMessage]) -> None:
        """Save messages to Supabase."""
        current_history = messages_to_dict(self.messages)
        new_history = messages_to_dict(messages)
        updated_history = current_history + new_history
        
        # Upsert the entire conversation history for the session_id
        supabase.table(self.table_name).upsert({
            "conversation_id": self.session_id,
            "history": updated_history
        }).execute()

    def clear(self) -> None:
        """Clear session history from Supabase."""
        supabase.table(self.table_name).delete().eq("conversation_id", self.session_id).execute()
# --- END: NEW Custom Supabase Chat History Class ---

# Define the request body for the API endpoint
class ChatRequest(BaseModel):
    conversation_id: str
    query: str

@app.post("/chat")
def chat_with_agent(request: ChatRequest):
    """
    Main endpoint to chat with the agent.
    Handles loading and saving conversation history for each user.
    """
    try:
        # 1. Use our new, reliable Supabase chat history class
        message_history = SupabaseChatMessageHistory(
            session_id=request.conversation_id,
            table_name="conversation_history"
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
        
        # 5. The memory object automatically calls our add_messages method, saving the history.
        
        return {"response": response["output"]}

    except Exception as e:
        # Return a proper HTTP exception for better error handling
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# To run this API, save it as api.py and run from your terminal:
# uvicorn api:app --reload