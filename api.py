#
# -------------------- api.py (Asynchronous Version) --------------------
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

# --- Custom Supabase Chat History Class (Unchanged) ---
class SupabaseChatMessageHistory(BaseChatMessageHistory):
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
        
        supabase.table(self.table_name).upsert({
            "conversation_id": self.session_id,
            "history": updated_history
        }).execute()

    def clear(self) -> None:
        """Clear session history from Supabase."""
        supabase.table(self.table_name).delete().eq("conversation_id", self.session_id).execute()

# Define the request body for the API endpoint
class ChatRequest(BaseModel):
    conversation_id: str
    query: str

# --- THIS FUNCTION IS NOW ASYNCHRONOUS ---
@app.post("/chat")
async def chat_with_agent(request: ChatRequest):
    """
    Main endpoint to chat with the agent.
    Handles loading and saving conversation history for each user.
    """
    try:
        message_history = SupabaseChatMessageHistory(
            session_id=request.conversation_id,
            table_name="conversation_history"
        )
        
        memory = ConversationBufferMemory(
            memory_key="history",
            chat_memory=message_history,
            return_messages=True
        )

        agent_executor = initialize_agent(memory)

        # Use the asynchronous 'ainvoke' method and 'await' the result
        response = await agent_executor.ainvoke({"input": request.query})
        
        return {"response": response["output"]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# To run this API, save it as api.py and run from your terminal:
# uvicorn api:app --reload