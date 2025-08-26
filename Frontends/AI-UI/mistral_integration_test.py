import asyncio
import os
from dotenv import load_dotenv
from mistral_client import MistralClient

# Load environment variables
load_dotenv()

async def test_mistral_connection():
    # Get the Mistral API key
    api_key = os.getenv("MISTRAL_API_KEY")

    # Initialize the Mistral client
    mistral_client = MistralClient(api_key=api_key, simulator=None)

    # Test message
    messages = [{"role": "user", "content": "hi"}]

    try:
        # Send the message to Mistral API
        response = await mistral_client.chat_completion(messages, tools=None)

        # Check if the response is valid
        if response and "choices" in response and response["choices"]:
            print("Test passed: Received response from Mistral API.")
            print("Response:", response["choices"][0]["message"]["content"])
        else:
            print("Test failed: Invalid response format.")
            print("Response:", response)
    except Exception as e:
        print("Test failed: Exception occurred.")
        print("Exception:", e)

async def test_tool_functionality():
    """Test the tool functionality with the new client"""
    api_key = os.getenv("MISTRAL_API_KEY")

    # Initialize the Mistral client
    mistral_client = MistralClient(api_key=api_key, simulator=None)

    try:
        # Test with tools enabled
        messages = [{"role": "user", "content": "What can you do with the simulator?"}]

        # Get the tools from the client
        tools = mistral_client.tools

        response = await mistral_client.chat_completion(messages, tools=tools)

        if response and "choices" in response and response["choices"]:
            print("\nTool test passed: Received response with tools.")
            print("Response:", response["choices"][0]["message"]["content"])
        else:
            print("\nTool test failed: Invalid response format.")
            print("Response:", response)
    except Exception as e:
        print("\nTool test failed: Exception occurred.")
        print("Exception:", e)

if __name__ == "__main__":
    asyncio.run(test_mistral_connection())
    asyncio.run(test_tool_functionality())