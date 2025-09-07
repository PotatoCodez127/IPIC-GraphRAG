#
# -------------------- whatsapp_bot/utils.py --------------------
#
import httpx
import os

# The backend API is expected to be running on this URL
BACKEND_API_URL = os.getenv("BACKEND_API_URL", "http://127.0.0.1:8000/chat")

async def get_agent_response(conversation_id: str, query: str) -> str:
    """
    Calls the backend AI agent API to get a response for a given query.
    """
    payload = {
        "conversation_id": conversation_id,
        "query": query
    }
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(BACKEND_API_URL, json=payload)
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
            
            data = response.json()
            return data.get("response", "I'm sorry, I received an unexpected response from the AI agent.")

    except httpx.HTTPStatusError as e:
        print(f"Error calling backend API: {e.response.status_code} - {e.response.text}")
        return "I'm sorry, I'm having trouble connecting to my brain right now. Please try again in a moment."
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return "I'm sorry, a critical error occurred. Please try again later."
