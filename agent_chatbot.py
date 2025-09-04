#
# -------------------- agent_chatbot.py --------------------
#
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool, StructuredTool
from langchain_neo4j import GraphCypherQAChain
from langchain_neo4j import Neo4jGraph
from langchain.prompts import PromptTemplate
from supabase.client import Client, create_client

load_dotenv()

# --- Tool Creation Functions ---
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
    # --- START OF CHANGES ---
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        raise ValueError("Supabase URL or Service Key is missing. Please check your .env file.")

    supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    vector_store = SupabaseVectorStore(
        client=supabase,
        embedding=embeddings,
        table_name="documents",
        query_name="match_documents"
    )
    # --- END OF CHANGES ---
    
    retriever = vector_store.as_retriever()
    return Tool(
        name="General Information Search", func=retriever.invoke,
        description="Use for general, conceptual, or 'how-to' questions, and for information like operating hours and location. e.g., 'How do I prepare for a party?' or 'What are your hours?'"
    )

# --- Tool Functions for Sales and Support (Unchanged) ---
def book_gym_trial(name: str, email: str, phone: str) -> str:
    """
    Books a 7-day free gym trial. Gathers user's name, email, and phone,
    then sends this information to the sales team and confirms with the user.
    """
    print(f"--- ACTION: Sending lead to sales team ---")
    print(f"Name: {name}, Email: {email}, Phone: {phone}")
    print(f"--- END ACTION ---")
    return f"Great news, {name}! 🎉 I've scheduled your 7-day free trial. A sales representative will contact you shortly at {phone} or {email} to confirm the details. Get ready to have a great workout! 💪"

def gather_party_details(num_kids: int, age_range: str, desired_date: str) -> str:
    """
    Gathers initial details for a kids' party inquiry to provide a price estimate
    and pass the lead to the party planning team.
    """
    print(f"--- ACTION: Party lead gathered ---")
    print(f"Kids: {num_kids}, Ages: {age_range}, Date: {desired_date}")
    print(f"--- END ACTION ---")
    cost_per_child = 350
    estimated_cost = num_kids * cost_per_child
    return f"Awesome! For a party of {num_kids} kids around the age of {age_range} on {desired_date}, you're looking at an estimated cost of R{estimated_cost}. I've sent these details to our party coordinators, and they'll be in touch soon to help plan the perfect celebration! 🎈"

def escalate_to_human(name: str, phone: str, reason: str) -> str:
    """
    Handles a user's request to speak to a person. Collects their contact info
    and the reason, then informs the support team.
    """
    print(f"--- ACTION: Escalating to human support ---")
    print(f"Name: {name}, Phone: {phone}, Reason: {reason}")
    print(f"--- END ACTION ---")
    return f"Thank you, {name}. I've passed your request on to our team. Someone will call you back at {phone} as soon as possible to help with: '{reason}'. 😊"

# --- Main Agent Initialization Function (Unchanged) ---
def initialize_agent():
    """Creates and returns the main agent executor."""
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1, convert_system_message_to_human=True)
    
    tools = [
        create_graph_qa_tool(),
        create_vector_search_tool(),
        StructuredTool.from_function(
            name="Book 7-Day Gym Trial",
            func=book_gym_trial,
            description="Use this when a user wants to book, schedule, or start a 7-day free trial for the IPIC Active gym. You must ask for their full name, email, and phone number first before using this tool."
        ),
        StructuredTool.from_function(
            name="Gather Party Inquiry Details",
            func=gather_party_details,
            description="Use this tool when a user wants to inquire about booking a kids' party and needs a price estimate. You must ask for the number of kids, their age range, and the desired date first."
        ),
        StructuredTool.from_function(
            name="Escalate to a Human",
            func=escalate_to_human,
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
    Action Input: the input to the action.
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

def get_agent_response(agent_executor, query):
    """Invokes the agent and cleans the output."""
    try:
        response = agent_executor.invoke({"input": query})
        raw_output = response["output"]
        if "Final Answer:" in raw_output:
            clean_response = raw_output.split("Final Answer:")[-1].strip()
        else:
            clean_response = raw_output.strip()
        return clean_response if clean_response else "I've processed the information, but I don't have a specific answer to provide."
    except Exception as e:
        print(f"Agent invocation error: {e}")
        return "I'm sorry, but I encountered an error while trying to process your request."

# --- Main function for command-line testing (Unchanged) ---
def main_cli():
    print("⚙️  Setting up Sparky, the Sophisticated Agent for CLI...")
    agent_executor = initialize_agent()
    print("\n🤖 Hi, I'm Sparky! How can I help you today? ✨")
    print("=" * 50)
    
    while True:
        query = input("\nYou: ").strip()
        if query.lower() in ["exit", "quit", "q"]:
            break
        response = get_agent_response(agent_executor, query)
        print("\n🤖 Sparky:", response)

if __name__ == "__main__":
    main_cli()