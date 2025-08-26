#!/usr/bin/env python3
"""
Chat UI implementation using FastAPI web framework.
"""

import asyncio
from typing import Dict, List, Optional
import base64
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, APIRouter
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import json
from mistral_client import MistralClient
import httpx

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

router = APIRouter()

class ChatUI:
    def __init__(self, mistral_client: MistralClient):
        self.mistral = mistral_client
        self.messages = []
        self.websocket = None  # Store the current websocket connection

    async def handle_tool_use(self, tool_calls: List[Dict]) -> List[Dict]:
        """Handle tool calls from the AI"""
        tool_results = []
        if tool_calls:  # Only process if tool_calls is not None or empty
            for tool_call in tool_calls:
                try:
                    name = tool_call["function"]["name"]
                    args = json.loads(tool_call["function"]["arguments"]) if tool_call["function"].get("arguments") else {}

                    # Special handling for read_cam_image - display image to user only
                    if name == "read_cam_image":
                        camera_id = args["camera_id"]
                        image_data = await self.mistral.simulator.read_cam_image(camera_id)

                        # Send the image directly to the user via WebSocket
                        if self.websocket:
                            image_message = {
                                "type": "image",
                                "camera_id": camera_id,
                                "image_data": image_data["image_data"],
                                "message": f"Camera {camera_id} image"
                            }
                            await self.websocket.send_text(json.dumps(image_message))

                        # Return only the confirmation message string to the LLM (no base64 image)
                        tool_results.append({
                            "tool_call_id": tool_call["id"],
                            "role": "tool",
                            "content": image_data["message"]
                        })

                    else:
                        # Normal tool handling (including self_read_cam_image and others)
                        result = await self.mistral.call_tool(name, args)
                        content = result if isinstance(result, str) else json.dumps(result)
                        tool_results.append({
                            "tool_call_id": tool_call["id"],
                            "role": "tool",
                            "content": content
                        })

                except Exception as e:
                    tool_results.append({
                        "tool_call_id": tool_call["id"],
                        "role": "tool",
                        "content": json.dumps({"error": str(e)})
                    })
        return tool_results


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    chat_ui = app.state.chat_ui
    chat_ui.websocket = websocket  # Store the websocket connection
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            # Add user message
            user_message = {"role": "user", "content": message["content"]}
            chat_ui.messages.append(user_message)

            # Debug: Print current message history
            print("\n=== DEBUG: Current message history ===")
            for i, msg in enumerate(chat_ui.messages):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                if isinstance(content, list):
                    content = "MULTIPART MESSAGE: " + str([c.get("type", "") for c in content])
                print(f"Message {i}: {role} - {content[:100]}...")

            # Debug: Print the exact message being sent to LLM
            print("\n=== DEBUG: Sending to LLM ===")
            print(f"User message: {message['content']}")
            print(f"Total messages in history: {len(chat_ui.messages)}")

            # First AI response
            response = await chat_ui.mistral.chat_completion(chat_ui.messages)

            max_tool_loops = 5
            loops = 0

            while True:
                assistant_msg = response["choices"][0]["message"]
                assistant_content = assistant_msg.get("content")
                tool_calls = assistant_msg.get("tool_calls") or []

                # Debug: Print the LLM response for this iteration
                print("\n=== DEBUG: LLM Iteration Response ===")
                print(f"Assistant content: {assistant_content}")
                if tool_calls:
                    print(f"Tool calls: {tool_calls}")

                # Append assistant message if it has content or tool calls
                if assistant_content or tool_calls:
                    assistant_message = {
                        "role": "assistant",
                        "content": assistant_content or ""
                    }
                    if tool_calls:
                        assistant_message["tool_calls"] = tool_calls
                    chat_ui.messages.append(assistant_message)

                # Handle tool calls loop
                if tool_calls:
                    # Process the tool calls
                    tool_results = await chat_ui.handle_tool_use(tool_calls)

                    # Add tool results to message history with proper format
                    for result in tool_results:
                        # Ensure the tool result has the required 'name' field
                        tool_call = next((tc for tc in tool_calls if tc["id"] == result["tool_call_id"]), None)
                        if tool_call:
                            result["name"] = tool_call["function"]["name"]
                        chat_ui.messages.append(result)

                    # Debug: Print tool results
                    print("\n=== DEBUG: Tool Results ===")
                    for result in tool_results:
                        print(f"Tool result: {result}")

                    # Get a new response with the updated message history (assistant acknowledges tool results)
                    response = await chat_ui.mistral.chat_completion(chat_ui.messages)

                    # ALWAYS add the assistant acknowledgment to message history to maintain proper conversation flow
                    # This prevents "Unexpected role 'user' after role 'tool'" errors
                    ack_content = response["choices"][0]["message"]["content"]
                    chat_ui.messages.append({
                        "role": "assistant",
                        "content": ack_content or ""  # Use empty string if no content
                    })

                    # Check if there are self_read_cam_image calls AFTER the assistant acknowledgment
                    has_self_read = any(tc["function"]["name"] == "self_read_cam_image" for tc in tool_calls)

                    # Handle self_read_cam_image by adding image as user message AFTER assistant acknowledgment
                    if has_self_read:
                        for tc in tool_calls:
                            if tc["function"]["name"] == "self_read_cam_image":
                                # Get the image data from the simulator
                                camera_id = json.loads(tc["function"]["arguments"])["camera_id"]
                                image_data = await chat_ui.mistral.simulator.read_cam_image(camera_id)

                                # Add the image as a user message (multipart content)
                                chat_ui.messages.append({
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "image_url",
                                            "image_url": {
                                                "url": f"data:image/jpeg;base64,{image_data['image_data']}"
                                            }
                                        },
                                        {
                                            "type": "text",
                                            "text": f"Camera {camera_id} image for AI observation"
                                        }
                                    ]
                                })

                        # Get another response now that the image has been added
                        response = await chat_ui.mistral.chat_completion(chat_ui.messages)

                    loops += 1
                    if loops >= max_tool_loops:
                        print("Max tool loop iterations reached; breaking to avoid infinite loop.")
                        break

                    # Continue to evaluate next response which may contain more tool calls or final answer
                    continue

                # No tool calls; break the loop
                break

            # Send AI response - use the most recent non-empty content
            final_response_content = response["choices"][0]["message"]["content"]

            # If the final response is empty, look for the last meaningful assistant response in history
            if not final_response_content:
                for msg in reversed(chat_ui.messages):
                    if msg.get("role") == "assistant" and msg.get("content"):
                        final_response_content = msg["content"]
                        break

            # If still no content, send a default acknowledgment
            if not final_response_content:
                final_response_content = "Tool executed successfully."

            await websocket.send_text(json.dumps(final_response_content))

    except WebSocketDisconnect:
        print("Client disconnected")
        chat_ui.websocket = None  # Clear the websocket connection

@router.get("/api/camera/{camera_id}")
async def get_camera_image(camera_id: int):
    """Endpoint to get camera images for the UI"""
    try:
        result = await app.state.chat_ui.mistral.simulator.read_cam_image(camera_id)
        # Return the base64 encoded image directly
        return {"image_data": result["image_data"]}
    except Exception as e:
        return {"error": str(e)}

@router.get("/", response_class=HTMLResponse)
async def get_chat(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})


app.include_router(router)