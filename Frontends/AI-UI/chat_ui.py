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
                    # Special handling for read_cam_image - display image to user
                    if tool_call["function"]["name"] == "read_cam_image":
                        # Fetch the image data separately for display
                        camera_id = json.loads(tool_call["function"]["arguments"])["camera_id"]
                        image_data = await app.state.chat_ui.mistral.simulator.read_cam_image(camera_id)
                        # Send the image directly to the user via WebSocket
                        if self.websocket:
                            image_message = {
                                "type": "image",
                                "camera_id": camera_id,
                                "image_data": image_data['image_data'],
                                "message": f"Camera {camera_id} image"
                            }
                            await self.websocket.send_text(json.dumps(image_message))
                        
                        # Return only the confirmation message to the LLM
                        tool_results.append({
                            "tool_call_id": tool_call["id"],
                            "role": "tool",
                            "content": json.dumps(image_data['message'])
                        })
                    else:
                        # Normal tool handling
                        result = await self.mistral.call_tool(
                            tool_call["function"]["name"],
                            json.loads(tool_call["function"]["arguments"])
                        )
                        tool_results.append({
                            "tool_call_id": tool_call["id"],
                            "role": "tool",
                            "content": json.dumps(result)
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

            # Get AI response
            response = await chat_ui.mistral.chat_completion(chat_ui.messages)

            # Debug: Print the LLM response
            print("\n=== DEBUG: LLM Response ===")
            print(f"Response content: {response['choices'][0]['message']['content']}")
            if "tool_calls" in response["choices"][0]["message"]:
                print(f"Tool calls: {response['choices'][0]['message']['tool_calls']}")

            # Always add the assistant's response to message history
            assistant_message = {
                "role": "assistant",
                "content": response["choices"][0]["message"]["content"]
            }

            # Include tool calls if they exist
            if "tool_calls" in response["choices"][0]["message"] and response["choices"][0]["message"]["tool_calls"]:
                assistant_message["tool_calls"] = response["choices"][0]["message"]["tool_calls"]

            chat_ui.messages.append(assistant_message)

            # Handle tool calls if any
            if "tool_calls" in response["choices"][0]["message"] and response["choices"][0]["message"]["tool_calls"]:
                tool_calls = response["choices"][0]["message"]["tool_calls"]

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

                # Add the assistant acknowledgment to message history
                assistant_ack_message = {
                    "role": "assistant",
                    "content": response["choices"][0]["message"]["content"]
                }
                chat_ui.messages.append(assistant_ack_message)

                # Check if there are self_read_cam_image calls AFTER the assistant acknowledgment
                has_self_read = any(tool_call["function"]["name"] == "self_read_cam_image" for tool_call in tool_calls)

                # Handle self_read_cam_image by adding image as user message AFTER assistant acknowledgment
                if has_self_read:
                    for tool_call in tool_calls:
                        if tool_call["function"]["name"] == "self_read_cam_image":
                            # Get the image data from the simulator
                            camera_id = json.loads(tool_call["function"]["arguments"])["camera_id"]
                            image_data = await app.state.chat_ui.mistral.simulator.read_cam_image(camera_id)

                            # Add the image as a user message
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

                # Add the final assistant response to message history
                new_assistant_message = {
                    "role": "assistant",
                    "content": response["choices"][0]["message"]["content"]
                }
                chat_ui.messages.append(new_assistant_message)

                # Debug: Print the final LLM response
                print("\n=== DEBUG: Final LLM Response ===")
                print(f"Response content: {response['choices'][0]['message']['content']}")

            # Send AI response
            await websocket.send_text(json.dumps(response["choices"][0]["message"]["content"]))

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