#
# -------------------- whatsapp_bot/bot.py --------------------
#
import os
from fastapi import FastAPI, Form, Response
from twilio.twiml.messaging_response import MessagingResponse
from dotenv import load_dotenv

from utils import get_agent_response

# Load .env from the parent directory
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path)

app = FastAPI(
    title="WhatsApp Bot for Sparky AI",
    description="A webhook server to connect the Sparky AI agent to the Twilio WhatsApp API.",
    version="1.0.0"
)

@app.post("/whatsapp")
async def handle_whatsapp_message(From: str = Form(...), Body: str = Form(...)):
    """
    This endpoint is a webhook for Twilio. It receives incoming WhatsApp messages,
    forwards them to the backend AI agent, and sends the agent's response back to the user.
    """
    print(f"Received message from {From}: {Body}")

    # The user's phone number is the unique conversation ID
    conversation_id = From

    # Get the response from our AI agent API
    agent_response = await get_agent_response(conversation_id, Body)

    # Create a TwiML response to send back to the user
    twiml_response = MessagingResponse()
    twiml_response.message(agent_response)

    # Return the TwiML response as XML
    return Response(content=str(twiml_response), media_type="application/xml")
