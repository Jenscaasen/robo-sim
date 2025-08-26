import asyncio
import os
import json
from dotenv import load_dotenv
from mistral_client import MistralClient

# Load environment variables
load_dotenv()

async def test_tool_call_loop():
    """Test a complete tool call loop: AI calls tool, gets response, then says goodbye"""
    # Get the Mistral API key
    api_key = os.getenv("MISTRAL_API_KEY")

    # Initialize the Mistral client with no simulator
    mistral_client = MistralClient(api_key=api_key, simulator=None)

    # Start with a user message that will trigger a tool call
    messages = [{"role": "user", "content": "Please get the current joint states from the simulator"}]

    try:
        # First call - AI should request to use the get_joints tool
        print("First call: Requesting joint states...")
        response = await mistral_client.chat_completion(messages, tools=mistral_client.tools)

        if response and "choices" in response and response["choices"]:
            print("AI Response:", response["choices"][0]["message"]["content"])

            # Check if there are tool calls
            tool_calls = response["choices"][0]["message"].get("tool_calls")
            if tool_calls:
                print(f"\nTool calls detected: {len(tool_calls)}")

                # Process each tool call
                for tool_call in tool_calls:
                    # Check if tool_call is a dictionary or an object
                    if isinstance(tool_call, dict):
                        tool_name = tool_call["function"]["name"]
                        tool_args = json.loads(tool_call["function"]["arguments"])
                        tool_call_id = tool_call["id"]
                    else:
                        # Handle the case where tool_call is a ToolCall object
                        tool_name = tool_call.function.name
                        tool_args = json.loads(tool_call.function.arguments)
                        tool_call_id = tool_call.id

                    print(f"Calling tool: {tool_name} with args: {tool_args}")

                    # Call the tool function
                    tool_result = await mistral_client.call_tool(tool_name, tool_args)
                    print(f"Tool result: {tool_result}")

                    # Add the tool response to messages
                    messages.append({
                        "role": "assistant",
                        "content": response["choices"][0]["message"]["content"],
                        "tool_calls": [tool_call]
                    })
                    messages.append({
                        "role": "tool",
                        "name": tool_name,
                        "content": json.dumps(tool_result),
                        "tool_call_id": tool_call_id
                    })

                # Second call - AI should respond with the tool result and say goodbye
                print("\nSecond call: AI responding with tool result...")
                response = await mistral_client.chat_completion(messages, tools=mistral_client.tools)

                if response and "choices" in response and response["choices"]:
                    print("Final Response:", response["choices"][0]["message"]["content"])
                else:
                    print("Invalid response format in second call")
            else:
                print("No tool calls detected in the response")
        else:
            print("Invalid response format in first call")

    except Exception as e:
        print("Test failed: Exception occurred.")
        print("Exception:", e)

if __name__ == "__main__":
    asyncio.run(test_tool_call_loop())