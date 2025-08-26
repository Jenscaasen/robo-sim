#!/usr/bin/env python3
"""
Main application entry point for the AI-UI chat interface.
This connects the Mistral AI with the URDF simulator tools.
"""

import os
import uvicorn
from dotenv import load_dotenv
from chat_ui import app, ChatUI
from mistral_client import MistralClient
from simulator_client import SimulatorClient

# Load environment variables
load_dotenv()

# Initialize components
simulator = SimulatorClient(
    host=os.getenv("SIMULATOR_HOST", "127.0.0.1"),
    port=int(os.getenv("SIMULATOR_PORT", "5000"))
)

mistral = MistralClient(api_key=os.getenv("MISTRAL_API_KEY"), simulator=simulator)

# Create and initialize chat UI
chat_ui = ChatUI(mistral)
app.state.mistral = mistral
app.state.chat_ui = chat_ui

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )