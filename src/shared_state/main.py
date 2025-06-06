#!/usr/bin/env python
from dotenv import load_dotenv
load_dotenv(override=True)

import json
import logging
from enum import Enum
from pprint import pprint
from typing import Optional, List
from crewai import LLM
from crewai.flow import start
from pydantic import BaseModel, Field
# Import from copilotkit_integration
from copilotkit.crewai import (
    CopilotKitFlow,
    tool_calls_log,
    FlowInputState,
)
from crewai.flow import persist

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
    recipe: Optional[Recipe] = None


@persist()
class SharedStateFlow(CopilotKitFlow[AgentState]):
    @start()
    def chat(self):
        """
        Standard chat node.
        """
        system_prompt = f"""
        You are a helpful assistant for creating recipes.
        To generate or modify a recipe, you MUST use the generate_recipe tool.
        When you generated or modified the recipe, DO NOT repeat it as a message.
        Just briefly summarize the changes you made. 2 sentences max.
        This is the current state of the recipe: ----\n {json.dumps(self.state.recipe, indent=2) if self.state.recipe else "No recipe created yet"}\n-----
        """

        # Initialize CrewAI LLM with streaming enabled
        llm = LLM(model="gpt-4o", stream=True)

        # Get message history using the base class method
        messages = self.get_message_history(system_prompt=system_prompt)


        try:
            # Track tool calls
            initial_tool_calls_count = len(tool_calls_log)

            response_content = llm.call(
                messages=messages,
                tools=[GENERATE_RECIPE_TOOL],
                available_functions={"generate_recipe": self.generate_recipe_handler}
            )

            # Handle tool responses using the base class method
            final_response = self.handle_tool_responses(
                llm=llm,
                response_text=response_content,
                messages=messages,
                tools_called_count_before_llm_call=initial_tool_calls_count
            )

            # ---- Maintain conversation history ----
            # 1. Add the current user message(s) to conversation history
            for msg in self.state.messages:
                if msg.get('role') == 'user' and msg not in self.state.conversation_history:
                    self.state.conversation_history.append(msg)

            # 2. Add the assistant's response to conversation history
            assistant_message = {"role": "assistant", "content": final_response}
            self.state.conversation_history.append(assistant_message)

            print("Final response: ", final_response)

            return json.dumps({
                "response": final_response,
                "id": self.state.id
            })

        except Exception as e:
            return f"\n\nAn error occurred: {str(e)}\n\n"

    def generate_recipe_handler(self, recipe):
        """Handler for the generate_recipe tool"""
        # Convert the recipe dict to a Recipe object for validation
        recipe_obj = Recipe(**recipe)
        # Store as dict for JSON serialization, but validate first
        self.state.recipe = recipe_obj.model_dump()
        return recipe_obj.model_dump_json(indent=2)


def kickoff():
    shared_state_flow = SharedStateFlow()
    result = shared_state_flow.kickoff({
        "inputs": {
            "messages": [
                {
                    "role": "user",
                    "content": "Create a simple pasta recipe for beginners"
                }
            ],
            "recipe": None  # Initialize recipe field
        }
    })
    print(result)

def plot():
    shared_state_flow = SharedStateFlow()
    shared_state_flow.plot()


if __name__ == "__main__":
    kickoff()
