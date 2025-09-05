#
# -------------------- core_logic.py (v2 with Tool Schemas) --------------------
#
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool, StructuredTool
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
from langchain.prompts import PromptTemplate
from supabase.client import Client, create_client
from pydantic import BaseModel, Field # <-- IMPORT THIS

load_dotenv()

# --- Tool Creation Functions (Unchanged) ---
def create_graph_qa_tool():
    graph = Neo4jGraph()
    chain = GraphCypherQAChain.from_llm(
        ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0),
        graph=graph, verbose=False, allow_dangerous_requests=True
    )
    return Tool(
        name="Knowledge Graph Search", func=chain.invoke,
        description="Use for specific questions about rules, policies, costs, fees. e.g., 'What is the cakeage fee?'"
    )

def create_vector_search_tool():
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        raise ValueError("Supabase URL or Service Key is missing. Please check your .env file.")
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = SupabaseVectorStore(
        client=supabase, embedding=embeddings, table_name="documents", query_name="match_documents"
    )
    retriever = vector_store.as_retriever()
    return Tool(
        name="General Information Search", func=retriever.invoke,
        description="Use for general, conceptual, or 'how-to' questions, and for information like operating hours and location. e.g., 'How do I prepare for a party?' or 'What are your hours?'"
    )

# --- START: MODIFIED TOOL FUNCTIONS AND SCHEMAS ---

# 1. Schema for the Gym Trial Tool
class BookGymTrialArgs(BaseModel):
    name: str = Field(description="The user's full name.")
    email: str = Field(description="The user's email address.")
    phone: str = Field(description="The user's phone number.")

def book_gym_trial(args: BookGymTrialArgs) -> str:
    """
    Books a 7-day free gym trial. Gathers user's name, email, and phone,
    then sends this information to the sales team and confirms with the user.
    """
    print(f"--- ACTION: Sending lead to sales team ---")
    print(f"Name: {args.name}, Email: {args.email}, Phone: {args.phone}")
    print(f"--- END ACTION ---")
    return f"Great news, {args.name}! 🎉 I've scheduled your 7-day free trial. A sales representative will contact you shortly at {args.phone} or {args.email} to confirm the details. Get ready to have a great workout! 💪"

# 2. Schema for the Party Details Tool
class GatherPartyDetailsArgs(BaseModel):
    num_kids: int = Field(description="The number of children attending the party.")
    age_range: str = Field(description="The approximate age range of the children.")
    desired_date: str = Field(description="The user's desired date for the party.")

def gather_party_details(args: GatherPartyDetailsArgs) -> str:
    """
    Gathers initial details for a kids' party inquiry to provide a price estimate
    and pass the lead to the party planning team.
    """
    print(f"--- ACTION: Party lead gathered ---")
    print(f"Kids: {args.num_kids}, Ages: {args.age_range}, Date: {args.desired_date}")
    print(f"--- END ACTION ---")
    cost_per_child = 350
    estimated_cost = args.num_kids * cost_per_child
    return f"Awesome! For a party of {args.num_kids} kids around the age of {args.age_range} on {args.desired_date}, you're looking at an estimated cost of R{estimated_cost}. I've sent these details to our party coordinators, and they'll be in touch soon to help plan the perfect celebration! 🎈"

# 3. Schema for the Escalation Tool
class EscalateToHumanArgs(BaseModel):
    name: str = Field(description="The user's full name.")
    phone: str = Field(description="The user's phone number.")
    reason: str = Field(description="A brief reason for the user's request to speak to a human.")

def escalate_to_human(args: EscalateToHumanArgs) -> str:
    """
    Handles a user's request to speak to a person. Collects their contact info
    and the reason, then informs the support team.
    """
    print(f"--- ACTION: Escalating to human support ---")
    print(f"Name: {args.name}, Phone: {args.phone}, Reason: {args.reason}")
    print(f"--- END ACTION ---")
    return f"Thank you, {args.name}. I've passed your request on to our team. Someone will call you back at {args.phone} as soon as possible to help with: '{args.reason}'. 😊"

# --- END: MODIFIED TOOL FUNCTIONS AND SCHEMAS ---


# --- Main Agent Initialization Function ---
def initialize_agent():
    """Creates and returns the main agent executor."""
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1, convert_system_message_to_human=True)
    
    tools = [
        create_graph_qa_tool(),
        create_vector_search_tool(),
        # --- UPDATE TOOL CREATION TO USE THE NEW SCHEMAS ---
        StructuredTool.from_function(
            name="Book 7-Day Gym Trial",
            func=book_gym_trial,
            args_schema=BookGymTrialArgs, # <-- Add this
            description="Use this when a user wants to book, schedule, or start a 7-day free trial for the IPIC Active gym. You must ask for their full name, email, and phone number first before using this tool."
        ),
        StructuredTool.from_function(
            name="Gather Party Inquiry Details",
            func=gather_party_details,
            args_schema=GatherPartyDetailsArgs, # <-- Add this
            description="Use this tool when a user wants to inquire about booking a kids' party and needs a price estimate. You must ask for the number of kids, their age range, and the desired date first."
        ),
        StructuredTool.from_function(
            name="Escalate to a Human",
            func=escalate_to_human,
            args_schema=EscalateToHumanArgs, # <-- Add this
            description="Use this tool when the user explicitly asks to speak to a person, staff member, or human. You must ask for their name, phone number, and a brief reason for their request first."
        )
    ]
    
    persona_template = """
    You are a helpful assistant for IPIC Active (a gym) and IPIC Play (a kids' play park).
    Your name is "Sparky," the friendly and energetic guide for our family hub.

    **Your Persona:**
    - **Friendly & Professional:** Be warm, welcoming, and clear in your answers.
    - **Playful Energy:** Use emojis where appropriate (like ✨, 💪, 🎉, 😊).
    - **Always use the tools provided to answer questions.** Do not make up information.

    **You have access to the following tools:**
    {tools}

    **Use the following format:**

    Question: the input question you must answer
    Thought: You must think about what to do. Your goal is to answer the user's question or guide them to the next step. Choose the best tool from [{tool_names}].
    Action: the action to take, should be one of the tool names [{tool_names}].
    Action Input: the input to the action. This should be a JSON object that strictly adheres to the tool's argument schema.
    Observation: the result of the action.
    Thought: I now have enough information to answer the user in my Sparky persona.
    Final Answer: Your final, customer-facing answer.

    Begin!

    Question: {input}
    Thought:{agent_scratchpad}
    """
    
    agent_prompt = PromptTemplate.from_template(persona_template)
    agent = create_react_agent(llm, tools, agent_prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True, max_iterations=7)