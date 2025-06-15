#!/usr/bin/env python
from dotenv import load_dotenv
load_dotenv(override=True)

import json
import logging
from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field
from copilotkit.crewai import (
    CopilotKitFlow,
    FlowInputState,
)
from crewai.flow import start, persist

# Import our clean abstraction
from shared_state.copilotkit_streaming import copilotkit_stream_completion
from shared_state.copilotkit_predict_state import copilotkit_predict_state

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
        """Clean chat with abstracted streaming and tool handling"""

        system_prompt = f"""
        You are a helpful assistant for creating recipes.
        To generate or modify a recipe, you MUST use the generate_recipe tool.
        When you generated or modified the recipe, DO NOT repeat it as a message.
        Just briefly summarize the changes you made. 2 sentences max.
        This is the current state of the recipe: ----\n {json.dumps(self.state.recipe, indent=2) if self.state.recipe else "No recipe created yet"}\n-----
        """

        print(f"🔥 SYSTEM PROMPT: {system_prompt}")

        messages = self.get_message_history(system_prompt=system_prompt)

        try:
            copilotkit_predict_state([
                {
                    "state_key": "recipe",
                    "tool": "generate_recipe",
                    "tool_argument": "recipe"
                }
            ])
            # Clean, simple streaming with automatic chunking and tool handling
            response = copilotkit_stream_completion(
                model="gpt-4o",
                messages=messages,
                tools=[GENERATE_RECIPE_TOOL]
            )

            final_response = response.content

            # Clean, simple tool call handling - exactly what you wanted!
            for tool_call in response.tool_calls:
                if tool_call['function']['name'] == 'generate_recipe':
                    try:
                        recipe_data = tool_call['function']['arguments']['recipe']
                        final_response = self.generate_recipe_handler(recipe_data)
                    except Exception as e:
                        print(f"Error in tool call: {e}")

            # Maintain conversation history
            for msg in self.state.messages:
                if msg.get('role') == 'user' and msg not in self.state.conversation_history:
                    self.state.conversation_history.append(msg)

            assistant_message = {"role": "assistant", "content": final_response}
            self.state.conversation_history.append(assistant_message)

            return final_response

        except Exception as e:
            return f"An error occurred: {str(e)}"

    def generate_recipe_handler(self, recipe):
        recipe_obj = Recipe(**recipe)
        self.state.set_recipe(recipe_obj)
        return "Recipe created successfully"

def kickoff():
    shared_state_flow = SharedStateFlow()
    result = shared_state_flow.kickoff({
        "state": {"recipe": None},
        "messages": [{"role": "user", "content": "Create a simple pasta recipe for beginners"}]
    })
    print("\n✨ Final Result:")
    print(result)

if __name__ == "__main__":
    kickoff()