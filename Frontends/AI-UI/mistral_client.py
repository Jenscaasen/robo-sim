#!/usr/bin/env python3
"""
Mistral API client with tools integration for the URDF simulator using the official Mistral Python client.
"""

import os
import json
import functools
from typing import Dict, List, Optional, Any
from simulator_client import SimulatorClient
from mistralai import Mistral
from mistralai.models.function import Function
from mistralai.models.toolmessage import ToolMessage
from mistralai.models.usermessage import UserMessage
from mistralai.models.assistantmessage import AssistantMessage

class MistralClient:
    def __init__(self, simulator: SimulatorClient, api_key: str = None):
        self.simulator = simulator
        self.api_key = api_key or os.environ.get("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("Mistral API key is required")

        # Initialize the official Mistral client
        self.client = Mistral(api_key=self.api_key)

        # Define tools for function calling
        self.tools = [
            {
                "type": "function",
                "function": Function(
                    name="get_joints",
                    description="Get current joint states from the simulator",
                    parameters={
                        "type": "object",
                        "properties": {}
                    },
                ),
            },
            {
                "type": "function",
                "function": Function(
                    name="set_joint",
                    description="Set a specific joint position",
                    parameters={
                        "type": "object",
                        "properties": {
                            "joint_id": {"type": "integer", "description": "ID of the joint to set"},
                            "value": {"type": "number", "description": "Target position value"}
                        },
                        "required": ["joint_id", "value"]
                    },
                ),
            },
            {
                "type": "function",
                "function": Function(
                    name="set_multiple_joints",
                    description="Set multiple joint positions simultaneously",
                    parameters={
                        "type": "object",
                        "properties": {
                            "joints": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "id": {"type": "integer", "description": "Joint ID"},
                                        "pos": {"type": "number", "description": "Target position"}
                                    },
                                    "required": ["id", "pos"]
                                }
                            },
                            "fast": {"type": "boolean", "description": "Use fast mode if true"}
                        },
                        "required": ["joints"]
                    },
                ),
            },
            {
                "type": "function",
                "function": Function(
                    name="reset_joints",
                    description="Reset all joints to neutral position",
                    parameters={
                        "type": "object",
                        "properties": {}
                    },
                ),
            },
            {
                "type": "function",
                "function": Function(
                    name="read_cam_image",
                    description="Read image from a specific camera",
                    parameters={
                        "type": "object",
                        "properties": {
                            "camera_id": {"type": "integer", "description": "ID of the camera to read"}
                        },
                        "required": ["camera_id"]
                    },
                ),
            },
            {
                "type": "function",
                "function": Function(
                    name="self_read_cam_image",
                    description="Internal tool for AI to read camera images automatically",
                    parameters={
                        "type": "object",
                        "properties": {
                            "camera_id": {"type": "integer", "description": "ID of the camera to read"}
                        },
                        "required": ["camera_id"]
                    },
                ),
            },
            {
                "type": "function",
                "function": Function(
                    name="detect_color_position",
                    description="Detect the position of a specific color object across all cameras",
                    parameters={
                        "type": "object",
                        "properties": {
                            "color_name": {
                                "type": "string",
                                "description": "Name of the color to detect (red, blue, green, yellow, orange, purple)",
                                "enum": ["red", "blue", "green", "yellow", "orange", "purple"]
                            }
                        },
                        "required": ["color_name"]
                    },
                ),
            }
        ]

        # Map tool names to simulator functions
        self.names_to_functions = {}

        # Only add simulator functions if simulator is provided
        if self.simulator:
            self.names_to_functions = {
                "get_joints": self.simulator.get_joints,
                "set_joint": functools.partial(self.simulator.set_joint),
                "set_multiple_joints": functools.partial(self.simulator.set_multiple_joints),
                "reset_joints": self.simulator.reset_joints,
                "read_cam_image": functools.partial(self.simulator.read_cam_image),
                "self_read_cam_image": functools.partial(self.simulator.read_cam_image),
                "detect_color_position": functools.partial(self.simulator.detect_color_position)
            }
        else:
            # Create dummy functions for when simulator is None
            async def dummy_function(*args, **kwargs):
                return {"error": "Simulator not available"}

            self.names_to_functions = {
                "get_joints": dummy_function,
                "set_joint": dummy_function,
                "set_multiple_joints": dummy_function,
                "reset_joints": dummy_function,
                "read_cam_image": dummy_function,
                "self_read_cam_image": dummy_function,
                "detect_color_position": dummy_function
            }

    async def call_tool(self, tool_name: str, parameters: Dict) -> Any:
        """Execute a tool function based on the tool name"""
        if tool_name in self.names_to_functions:
            if tool_name == "set_joint":
                return await self.names_to_functions[tool_name](parameters["joint_id"], parameters["value"])
            elif tool_name == "set_multiple_joints":
                return await self.names_to_functions[tool_name](parameters["joints"], parameters.get("fast", False))
            elif tool_name == "read_cam_image" or tool_name == "self_read_cam_image":
                result = await self.names_to_functions[tool_name](parameters["camera_id"])
                # For read_cam_image, return only the message to the LLM
                if tool_name == "read_cam_image":
                    return result["message"]
                # For self_read_cam_image, return only the text message to the LLM
                # The image data will be sent as a separate user message
                else:
                    return "The image is being sent to you by the user now"
            elif tool_name == "detect_color_position":
                result = await self.names_to_functions[tool_name](parameters["color_name"])
                # Return the message to the LLM
                return result["message"]
            else:
                return await self.names_to_functions[tool_name]()
        else:
            raise ValueError(f"Unknown tool: {tool_name}")

    async def chat_completion(self, messages: List[Dict], tools: Optional[List[Dict]] = None) -> Dict:
        """Call Mistral API with chat completion using the official client"""
        # Convert messages to Mistral message objects
        mistral_messages = []
        for message in messages:
            if isinstance(message, dict):
                if message.get("role") == "user":
                    mistral_messages.append(UserMessage(content=message["content"]))
                elif message.get("role") == "assistant":
                    # Handle tool calls if present
                    tool_calls = message.get("tool_calls")
                    if tool_calls:
                        mistral_messages.append(AssistantMessage(
                            content=message.get("content", ""),
                            tool_calls=tool_calls
                        ))
                    else:
                        mistral_messages.append(AssistantMessage(content=message["content"]))
                elif message.get("role") == "tool":
                    mistral_messages.append(ToolMessage(
                        name=message["name"],
                        content=message["content"],
                        tool_call_id=message.get("tool_call_id")
                    ))
            else:
                # Default to user message if not properly formatted
                mistral_messages.append(UserMessage(content=str(message)))

        # Use provided tools or default to our tools
        tools_to_use = tools if tools is not None else self.tools

        # Call the Mistral API - note: chat.complete is not async
        response = self.client.chat.complete(
            model="mistral-medium-latest",
            temperature=0.7,
            messages=mistral_messages,
            tools=tools_to_use
        )

        # Convert the response to a format similar to the original implementation
        tool_calls = None
        if hasattr(response.choices[0].message, "tool_calls") and response.choices[0].message.tool_calls:
            # Convert ToolCall objects to dictionaries
            tool_calls = []
            for tool_call in response.choices[0].message.tool_calls:
                tool_calls.append({
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments
                    }
                })

        result = {
            "id": response.id,
            "object": "chat.completion",
            "created": response.created,
            "model": response.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response.choices[0].message.content,
                        "tool_calls": tool_calls
                    },
                    "finish_reason": response.choices[0].finish_reason
                }
            ],
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        }

        return result

    async def close(self):
        """Close any resources if needed"""
        # The official client doesn't need explicit closing
        pass