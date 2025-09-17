#
# -------------------- api.py (Optimized, Secure & Concurrency-Safe) --------------------
#
from fastapi import FastAPI, HTTPException, Depends, Header, status
from pydantic import BaseModel
from supabase.client import Client, create_client
import os
import json
import asyncio

# --- Core Logic Imports for Pre-loading ---
from core_logic import (
    initialize_agent,
    create_graph_qa_tool,
    create_vector_search_tool,
    book_gym_trial,
    gather_party_details,
    escalate_to_human,
    BookGymTrialArgs,
    GatherPartyDetailsArgs,
    EscalateToHumanArgs
)
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import StructuredTool
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
# --- End Core Logic Imports ---

from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseChatMessageHistory
from langchain.schema.messages import BaseMessage, messages_from_dict, messages_to_dict

# Initialize FastAPI app
app = FastAPI(
    title="Sparky AI Agent API",
    description="An API to interact with the Sparky AI agent for IPIC.",
    version="1.4.0" # Version bump for routing logic
)

# --- API Security ---
API_SECRET_KEY = os.getenv("API_SECRET_KEY")

async def verify_api_key(x_api_key: str = Header()):
    """Dependency to verify the API key in the request header."""
    if not API_SECRET_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API Secret Key not configured on the server."
        )
    if x_api_key != API_SECRET_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key."
        )
# --- End API Security ---


# --- Create a semaphore to limit concurrent agent executions ---
CONCURRENCY_LIMIT = 2
agent_semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)


# --- Supabase Connection ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# --- Pre-load Heavy Components on Application Startup ---
print("Pre-loading AI components...")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1, convert_system_message_to_human=True)
tools = [create_graph_qa_tool(), create_vector_search_tool(), # ... (rest of tools are the same)
    StructuredTool.from_function(
        name="Book 7-Day Gym Trial", func=book_gym_trial, args_schema=BookGymTrialArgs,
        description="Use this when a user wants to book, schedule, or start a 7-day free trial for the IPIC Active gym. You must ask for their full name, email, and phone number first before using this tool."
    ),
    StructuredTool.from_function(
        name="Gather Party Inquiry Details", func=gather_party_details, args_schema=GatherPartyDetailsArgs,
        description="Use this tool when a user wants to inquire about booking a kids' party and needs a price estimate. You must ask for the number of kids, their age range, and the desired date first."
    ),
    StructuredTool.from_function(
        name="Escalate to a Human", func=escalate_to_human, args_schema=EscalateToHumanArgs,
        description="Use this tool when the user explicitly asks to speak to a person, staff member, or human. You must ask for their name, phone number, and a brief reason for their request first."
    )
]

agent_executor = initialize_agent(llm, tools)
print("✅ AI components pre-loaded successfully!")
# --- End Pre-loading Section ---

# --- Intent Classification ---
async def classify_intent(query: str) -> str:
    """Classifies the user's intent."""
    prompt = f"""
    Please classify the user's intent into one of the following categories:
    - "GREETING" (e.g., "hi", "hello")
    - "THANKS" (e.g., "thanks", "thank you")
    - "OUT_OF_SCOPE" (e.g., "what's the weather like?", "tell me a joke")
    - "TOOL_REQUIRED" (e.g., "what are your hours?", "how do I book a party?")

    User query: "{query}"
    """
    response = await llm.ainvoke(prompt)
    return response.content.strip()

# --- Custom Supabase Chat History Class (Unchanged) ---
class SupabaseChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, session_id: str, table_name: str):
        self.session_id = session_id
        self.table_name = table_name
    @property
    def messages(self):
        response = supabase.table(self.table_name).select("history").eq("conversation_id", self.session_id).execute()
        if response.data:
            return messages_from_dict(response.data[0]['history'])
        return []
    def add_messages(self, messages: list[BaseMessage]) -> None:
        current_history = messages_to_dict(self.messages)
        new_history = messages_to_dict(messages)
        updated_history = current_history + new_history
        supabase.table(self.table_name).upsert({"conversation_id": self.session_id, "history": updated_history}).execute()
    def clear(self) -> None:
        supabase.table(self.table_name).delete().eq("conversation_id", self.session_id).execute()

# Define the request body (Unchanged)
class ChatRequest(BaseModel):
    conversation_id: str
    query: str

@app.post("/chat", dependencies=[Depends(verify_api_key)])
async def chat_with_agent(request: ChatRequest):
    """
    Main endpoint to chat with the agent.
    This endpoint is now SECURED with an API key.
    """
    async with agent_semaphore:
        print(f"Semaphore acquired by conversation_id: {request.conversation_id}")
        try:
            intent = await classify_intent(request.query)

            if intent == "GREETING":
                return {"response": "Hi there! I'm Sparky, your friendly guide to all things IPIC. How can I help you today? ✨"}
            elif intent == "THANKS":
                return {"response": "You're welcome! If you need anything else, just ask. 😊"}
            elif intent == "OUT_OF_SCOPE":
                return {"response": "I'm sorry, I can only answer questions about IPIC. If you'd like to speak to a human, just let me know! 🤖"}
            else:
                message_history = SupabaseChatMessageHistory(
                    session_id=request.conversation_id,
                    table_name="conversation_history"
                )
                memory = ConversationBufferMemory(
                    memory_key="history",
                    chat_memory=message_history,
                    return_messages=True
                )
                agent_executor.memory = memory
                response = await agent_executor.ainvoke({"input": request.query})
                
                print(f"Semaphore released by conversation_id: {request.conversation_id}")
                return {"response": response["output"]}

        except Exception as e:
            print(f"An error occurred in /chat for conversation_id {request.conversation_id}: {e}")
            raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# To run this API, save it as api.py and run from your terminal:
# uvicorn api:app --reload