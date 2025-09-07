#
# -------------------- core_logic.py (v3 with Memory) --------------------
#
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool, StructuredTool
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
from langchain.prompts import PromptTemplate, MessagesPlaceholder
from supabase.client import Client, create_client
from pydantic import BaseModel, Field
from langchain.memory import ConversationBufferMemory # <-- IMPORT THIS

load_dotenv()

# --- All Tool Creation Functions and Schemas remain the same ---
# (create_graph_qa_tool, create_vector_search_tool, BookGymTrialArgs, book_gym_trial, etc.)
# ... [Keeping the rest of the file the same for brevity, only showing the changed function] ...

# --- Main Agent Initialization Function (MODIFIED) ---
def initialize_agent(memory): # <-- ADD MEMORY PARAMETER
    """Creates and returns the main agent executor, now with conversation memory."""
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1, convert_system_message_to_human=True)
    
    # Tool list remains the same
    tools = [
        create_graph_qa_tool(),
        create_vector_search_tool(),
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
    
    # --- PROMPT TEMPLATE IS MODIFIED ---
    persona_template = """
    You are a helpful assistant for IPIC Active (a gym) and IPIC Play (a kids' play park).
    Your name is "Sparky," the friendly and energetic guide for our family hub.

    **Your Persona:**
    - **Friendly & Professional:** Be warm, welcoming, and clear in your answers.
    - **Playful Energy:** Use emojis where appropriate (like ✨, 💪, 🎉, 😊).
    - **Always use the tools provided to answer questions.** Do not make up information.
    - **Remember the conversation history to provide context-aware responses.**

    **You have access to the following tools:**
    {tools}

    **Use the following format:**

    Question: the input question you must answer
    Thought: You must think about what to do, considering the conversation history. Your goal is to answer the user's question or guide them to the next step. Choose the best tool from [{tool_names}].
    Action: the action to take, should be one of the tool names [{tool_names}].
    Action Input: the input to the action. This should be a JSON object that strictly adheres to the tool's argument schema.
    Observation: the result of the action.
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now have enough information to answer the user in my Sparky persona.
    Final Answer: Your final, customer-facing answer.

    Begin!

    Previous conversation history:
    {history}

    New question: {input}
    Thought:{agent_scratchpad}
    """
    
    agent_prompt = PromptTemplate.from_template(persona_template)
    # This also needs the 'history' variable from the memory object.
    # We will pass this when we create the agent.
    
    agent = create_react_agent(llm, tools, agent_prompt)

    # AgentExecutor now gets the memory object
    return AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True, handle_parsing_errors=True, max_iterations=7)