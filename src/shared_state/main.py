#!/usr/bin/env python
from dotenv import load_dotenv
load_dotenv(override=True)

import json
import logging
from datetime import datetime
from enum import Enum
from typing import Optional, List, Any
from collections import defaultdict

from litellm import completion
from crewai.flow import start
from pydantic import BaseModel, Field
from copilotkit.crewai import (
    CopilotKitFlow,
    FlowInputState,
)
from crewai.flow import persist
from crewai.utilities.events import crewai_event_bus

# ==================== CUSTOM EVENTS WITH PROPER TIMESTAMPS ====================

class OrderedStreamChunkEvent(BaseModel):
    """Custom event that sets timestamp at emission time"""
    type: str = "ordered_stream_chunk"
    timestamp: datetime = Field(default_factory=datetime.now)
    sequence: int = 0
    chunk: str = ""
    context: Optional[str] = None  # "generating_recipe", "thinking", etc.
    tool_call: Optional[dict] = None

class OrderedToolCallEvent(BaseModel):
    """Custom event for tool calls"""
    type: str = "ordered_tool_call"
    timestamp: datetime = Field(default_factory=datetime.now)
    sequence: int = 0
    tool_name: str
    status: str  # "started" or "completed"
    context: str = ""
    result: Optional[Any] = None

# Global sequence counter for ordering
_sequence_counter = 0

def emit_ordered_event(event):
    """Emit event with proper timestamp and sequence"""
    global _sequence_counter

    # Set timestamp at emission time, not creation time
    event.timestamp = datetime.now()
    event.sequence = _sequence_counter
    _sequence_counter += 1

    # Use existing CrewAI event bus
    crewai_event_bus.emit(source="recipe_flow", event=event)

# ==================== EVENT HANDLER FOR DEBUGGING ====================

@crewai_event_bus.on(OrderedStreamChunkEvent)
def handle_stream_chunk(source, event: OrderedStreamChunkEvent):
    """Handle ordered stream chunk events"""
    context_info = f" [{event.context}]" if event.context else ""
    print(f"🔥 CHUNK #{event.sequence} ({event.timestamp.strftime('%H:%M:%S.%f')[:-3]}){context_info}: '{event.chunk}'")

@crewai_event_bus.on(OrderedToolCallEvent)
def handle_tool_call(source, event: OrderedToolCallEvent):
    """Handle ordered tool call events"""
    if event.status == "started":
        print(f"🛠️ TOOL START #{event.sequence}: {event.tool_name} - {event.context}")
    elif event.status == "completed":
        print(f"✅ TOOL DONE #{event.sequence}: {event.tool_name} - {event.context}")

# ==================== RECIPE MODELS ====================

class SkillLevel(str, Enum):
    """
    The level of skill required for the recipe.
    """
    BEGINNER = "Beginner"
    INTERMEDIATE = "Intermediate"
    ADVANCED = "Advanced"

class CookingTime(str, Enum):
    """
    The cooking time of the recipe.
    """
    FIVE_MIN = "5 min"
    FIFTEEN_MIN = "15 min"
    THIRTY_MIN = "30 min"
    FORTY_FIVE_MIN = "45 min"
    SIXTY_PLUS_MIN = "60+ min"

class Ingredient(BaseModel):
    """
    An ingredient with its details.
    """
    icon: str = Field(..., description="Emoji icon representing the ingredient.")
    name: str = Field(..., description="Name of the ingredient.")
    amount: str = Field(..., description="Amount or quantity of the ingredient.")

GENERATE_RECIPE_TOOL = {
    "type": "function",
    "function": {
        "name": "generate_recipe",
        "description": (
            "Generate or modify an existing recipe. "
            "When creating a new recipe, specify all fields. "
            "When modifying, only fill optional fields if they need changes; "
            "otherwise, leave them empty."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "recipe": {
                    "description": "The recipe object containing all details.",
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "The title of the recipe.",
                        },
                        "skill_level": {
                            "type": "string",
                            "enum": [level.value for level in SkillLevel],
                            "description": "The skill level required for the recipe.",
                        },
                        "dietary_preferences": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "A list of dietary preferences (e.g., Vegetarian, Gluten-free).",
                        },
                        "cooking_time": {
                            "type": "string",
                            "enum": [time.value for time in CookingTime],
                            "description": "The estimated cooking time for the recipe.",
                        },
                        "ingredients": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "icon": {
                                        "type": "string",
                                        "description": "Emoji icon for the ingredient.",
                                    },
                                    "name": {
                                        "type": "string",
                                        "description": "Name of the ingredient.",
                                    },
                                    "amount": {
                                        "type": "string",
                                        "description": "Amount/quantity of the ingredient.",
                                    },
                                },
                                "required": ["icon", "name", "amount"],
                            },
                            "description": "A list of ingredients required for the recipe.",
                        },
                        "instructions": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Step-by-step instructions for preparing the recipe.",
                        },
                    },
                    "required": [
                        "title",
                        "skill_level",
                        "cooking_time",
                        "dietary_preferences",
                        "ingredients",
                        "instructions",
                    ],
                }
            },
            "required": ["recipe"],
        },
    },
}

logging.basicConfig(level=logging.INFO)

class Recipe(BaseModel):
    """
    A recipe.
    """
    title: str
    skill_level: SkillLevel
    dietary_preferences: List[str] = Field(default_factory=list)
    cooking_time: CookingTime
    ingredients: List[Ingredient] = Field(default_factory=list)
    instructions: List[str] = Field(default_factory=list)


class AgentState(FlowInputState):
    """
    The state of the recipe.
    """
    recipe: Optional[dict] = None

    def get_recipe(self) -> Optional[Recipe]:
        """Get the recipe as a Recipe object"""
        if self.recipe is None:
            return None
        return Recipe(**self.recipe)

    def set_recipe(self, recipe: Recipe):
        """Set the recipe from a Recipe object"""
        self.recipe = recipe.model_dump()


@persist()
class SharedStateFlow(CopilotKitFlow[AgentState]):
    @start()
    def chat(self):
        """
        Standard chat node with proper event ordering.
        """
        print("DEBUG: ENTERED CHAT METHOD")
        print(f"DEBUG: Current recipe state: {self.state.recipe}")
        print(f"DEBUG: Current messages: {self.state.messages}")

        system_prompt = f"""
        You are a helpful assistant for creating recipes.
        To generate or modify a recipe, you MUST use the generate_recipe tool.
        When you generated or modified the recipe, DO NOT repeat it as a message.
        Just briefly summarize the changes you made. 2 sentences max.
        This is the current state of the recipe: ----\n {json.dumps(self.state.recipe, indent=2) if self.state.recipe else "No recipe created yet"}\n-----
        """

        # Get message history using the base class method
        messages = self.get_message_history(system_prompt=system_prompt)

        try:
            print("DEBUG: Starting litellm streaming call...")
            final_response = ""

            # Use litellm with streaming
            response_stream = completion(
                model="gpt-4o",
                messages=messages,
                tools=[GENERATE_RECIPE_TOOL],
                parallel_tool_calls=False,
                stream=True
            )

            full_response = ""
            tool_calls = []
            accumulated_tool_args = defaultdict(lambda: {'function': {'name': None, 'arguments': ''}})
            current_context = "thinking"  # Start with thinking context

            # Process stream in order and emit events immediately
            for chunk in response_stream:
                if hasattr(chunk, 'choices') and chunk.choices:
                    delta = chunk.choices[0].delta

                    # Handle regular content
                    if hasattr(delta, 'content') and delta.content:
                        full_response += delta.content

                        # Emit chunk event immediately with current context
                        emit_ordered_event(OrderedStreamChunkEvent(
                            chunk=delta.content,
                            context=current_context
                        ))

                    # Handle tool calls
                    if hasattr(delta, 'tool_calls') and delta.tool_calls:
                        # Switch context to recipe generation
                        if current_context != "generating_recipe":
                            current_context = "generating_recipe"
                            emit_ordered_event(OrderedToolCallEvent(
                                tool_name="generate_recipe",
                                status="started",
                                context="Starting recipe generation"
                            ))

                        for tool_call in delta.tool_calls:
                            index = tool_call.index
                            current_tool = accumulated_tool_args[index]

                            if hasattr(tool_call, 'id') and tool_call.id:
                                current_tool['id'] = tool_call.id

                            if hasattr(tool_call, 'function'):
                                if hasattr(tool_call.function, 'name') and tool_call.function.name:
                                    current_tool['function']['name'] = tool_call.function.name

                                if hasattr(tool_call.function, 'arguments') and tool_call.function.arguments:
                                    current_tool['function']['arguments'] += tool_call.function.arguments

                                    # Emit tool call chunk event immediately
                                    emit_ordered_event(OrderedStreamChunkEvent(
                                        chunk=tool_call.function.arguments,
                                        context="generating_recipe",
                                        tool_call={
                                            "name": current_tool['function']['name'],
                                            "arguments_chunk": tool_call.function.arguments
                                        }
                                    ))

            # Convert accumulated tool args to list
            tool_calls = [tool_data for tool_data in accumulated_tool_args.values() if tool_data['function']['name']]

            # Handle tool calls
            if tool_calls:
                for tool_call in tool_calls:
                    if tool_call['function']['name'] == 'generate_recipe':
                        try:
                            args = json.loads(tool_call['function']['arguments'])
                            print(f"DEBUG: Calling generate_recipe with args: {args}")
                            result = self.generate_recipe_handler(args['recipe'])
                            print(f"DEBUG: Tool result: {result}")

                            # Emit tool completion event
                            emit_ordered_event(OrderedToolCallEvent(
                                tool_name="generate_recipe",
                                status="completed",
                                context="Recipe generation completed",
                                result=result
                            ))

                        except Exception as e:
                            print(f"DEBUG: Error in tool call: {e}")

            final_response = full_response if full_response else "Recipe created successfully"

            # ---- Maintain conversation history ----
            # 1. Add the current user message(s) to conversation history
            for msg in self.state.messages:
                if msg.get('role') == 'user' and msg not in self.state.conversation_history:
                    self.state.conversation_history.append(msg)

            # 2. Add the assistant's response to conversation history
            assistant_message = {"role": "assistant", "content": final_response}
            self.state.conversation_history.append(assistant_message)

            return final_response

        except Exception as e:
            print(f"DEBUG: Exception occurred: {e}")
            return f"\n\nAn error occurred: {str(e)}\n\n"

    def generate_recipe_handler(self, recipe):
        """Handler for the generate_recipe tool"""

        # Convert the recipe dict to a Recipe object for validation
        recipe_obj = Recipe(**recipe)
        # Store using the helper method
        self.state.set_recipe(recipe_obj)

        return "Recipe created successfully"


def kickoff():
    print("🚀 Starting Recipe Flow with Ordered Events")
    print("📋 Events will show in proper chronological order:")
    print()

    shared_state_flow = SharedStateFlow()
    result = shared_state_flow.kickoff({
        "state": {
            "recipe": None
        },
        "messages": [
            {
                "role": "user",
                "content": "Create a simple pasta recipe for beginners"
            }
        ]
    })
    print()
    print("✨ Final Result:")
    print(result)

def plot():
    shared_state_flow = SharedStateFlow()
    shared_state_flow.plot()


if __name__ == "__main__":
    kickoff()