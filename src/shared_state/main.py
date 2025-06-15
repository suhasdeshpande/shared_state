#!/usr/bin/env python
from dotenv import load_dotenv
load_dotenv(override=True)

import json
import logging
from datetime import datetime
from enum import Enum
from typing import Optional, List
from collections import defaultdict

import threading

from litellm import completion
from crewai.flow import start
from pydantic import BaseModel, Field
from copilotkit.crewai import (
    CopilotKitFlow,
    FlowInputState,
)
from crewai.flow import persist
from crewai.utilities.events import crewai_event_bus
from crewai.utilities.events.base_events import BaseEvent

# ==================== SIMPLE DEBOUNCED CHUNK EVENT ====================

class DebouncedChunkEvent(BaseEvent):
    """Simple debounced chunk event"""
    type: str = "debounced_chunk"
    chunk: str = ""
    context: str = "thinking"
    sequence: int = 0
    timestamp: Optional[datetime] = None

# ==================== SIMPLE DEBOUNCER ====================

class SimpleDebouncer:
    """Dead simple debouncer - accumulate chunks for 50ms then emit"""

    def __init__(self, delay_ms=50):
        self.delay = delay_ms / 1000.0  # Convert to seconds
        self.accumulated = ""
        self.context = "thinking"
        self.timer = None
        self.sequence = 0

    def add_chunk(self, content: str, context: str = "thinking"):
        """Add content and start/reset timer"""
        self.accumulated += content
        self.context = context

        # Cancel existing timer
        if self.timer:
            self.timer.cancel()

        # Start new timer
        self.timer = threading.Timer(self.delay, self._emit)
        self.timer.start()

        # Also emit if we've accumulated a decent amount
        if len(self.accumulated) >= 30:
            self._emit_now()

    def _emit_now(self):
        """Emit immediately"""
        if self.timer:
            self.timer.cancel()
        self._emit()

    def _emit(self):
        """Emit the accumulated content"""
        if not self.accumulated:
            return

        event = DebouncedChunkEvent(
            chunk=self.accumulated,
            context=self.context
        )
        event.timestamp = datetime.now()
        event.sequence = self.sequence
        self.sequence += 1

        crewai_event_bus.emit(source="recipe_flow", event=event)

        # Reset
        self.accumulated = ""
        self.timer = None

# Global debouncer
debouncer = SimpleDebouncer(delay_ms=500)

# ==================== EVENT HANDLER ====================

@crewai_event_bus.on(DebouncedChunkEvent)
def handle_debounced_chunk(source, event: DebouncedChunkEvent):
    """Handle debounced chunk events"""
    chunk_preview = event.chunk[:80] + "..." if len(event.chunk) > 80 else event.chunk
    print(f"🔥 CHUNK #{event.sequence} [{event.context}]: '{chunk_preview}' (len: {len(event.chunk)})")

# ==================== RECIPE MODELS ====================

class SkillLevel(str, Enum):
    BEGINNER = "Beginner"
    INTERMEDIATE = "Intermediate"
    ADVANCED = "Advanced"

class CookingTime(str, Enum):
    FIVE_MIN = "5 min"
    FIFTEEN_MIN = "15 min"
    THIRTY_MIN = "30 min"
    FORTY_FIVE_MIN = "45 min"
    SIXTY_PLUS_MIN = "60+ min"

class Ingredient(BaseModel):
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
    title: str
    skill_level: SkillLevel
    dietary_preferences: List[str] = Field(default_factory=list)
    cooking_time: CookingTime
    ingredients: List[Ingredient] = Field(default_factory=list)
    instructions: List[str] = Field(default_factory=list)

class AgentState(FlowInputState):
    recipe: Optional[dict] = None

    def get_recipe(self) -> Optional[Recipe]:
        if self.recipe is None:
            return None
        return Recipe(**self.recipe)

    def set_recipe(self, recipe: Recipe):
        self.recipe = recipe.model_dump()

@persist()
class SharedStateFlow(CopilotKitFlow[AgentState]):
    @start()
    def chat(self):
        """Standard chat with simple debounced chunks"""

        system_prompt = f"""
        You are a helpful assistant for creating recipes.
        To generate or modify a recipe, you MUST use the generate_recipe tool.
        When you generated or modified the recipe, DO NOT repeat it as a message.
        Just briefly summarize the changes you made. 2 sentences max.
        This is the current state of the recipe: ----\n {json.dumps(self.state.recipe, indent=2) if self.state.recipe else "No recipe created yet"}\n-----
        """

        messages = self.get_message_history(system_prompt=system_prompt)

        try:
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
            current_context = "thinking"

            for chunk in response_stream:
                if hasattr(chunk, 'choices') and chunk.choices:
                    delta = chunk.choices[0].delta

                    # Handle regular content
                    if hasattr(delta, 'content') and delta.content:
                        full_response += delta.content
                        debouncer.add_chunk(delta.content, current_context)

                    # Handle tool calls
                    if hasattr(delta, 'tool_calls') and delta.tool_calls:
                        current_context = "generating_recipe"

                        for tool_call in delta.tool_calls:
                            index = tool_call.index
                            current_tool = accumulated_tool_args[index]

                            if hasattr(tool_call, 'function'):
                                if hasattr(tool_call.function, 'name') and tool_call.function.name:
                                    current_tool['function']['name'] = tool_call.function.name

                                if hasattr(tool_call.function, 'arguments') and tool_call.function.arguments:
                                    current_tool['function']['arguments'] += tool_call.function.arguments
                                    debouncer.add_chunk(tool_call.function.arguments, "generating_recipe")

            # Flush any remaining content
            debouncer._emit_now()

            # Handle tool calls
            tool_calls = [tool_data for tool_data in accumulated_tool_args.values() if tool_data['function']['name']]
            if tool_calls:
                for tool_call in tool_calls:
                    if tool_call['function']['name'] == 'generate_recipe':
                        try:
                            args = json.loads(tool_call['function']['arguments'])
                            result = self.generate_recipe_handler(args['recipe'])
                        except Exception as e:
                            print(f"Error in tool call: {e}")

            final_response = full_response if full_response else "Recipe created successfully"

            # Maintain conversation history
            for msg in self.state.messages:
                if msg.get('role') == 'user' and msg not in self.state.conversation_history:
                    self.state.conversation_history.append(msg)

            assistant_message = {"role": "assistant", "content": final_response}
            self.state.conversation_history.append(assistant_message)

            return final_response

        except Exception as e:
            debouncer._emit_now()  # Flush on error
            return f"An error occurred: {str(e)}"

    def generate_recipe_handler(self, recipe):
        recipe_obj = Recipe(**recipe)
        self.state.set_recipe(recipe_obj)
        return "Recipe created successfully"

def kickoff():
    print("🚀 Starting Recipe Flow with Simple Debounced Chunks")
    print("⏱️  500ms debounce - accumulate then emit quality chunks")
    print()

    shared_state_flow = SharedStateFlow()
    result = shared_state_flow.kickoff({
        "state": {"recipe": None},
        "messages": [{"role": "user", "content": "Create a simple pasta recipe for beginners"}]
    })

    print("\n✨ Final Result:")
    print(result)

if __name__ == "__main__":
    kickoff()