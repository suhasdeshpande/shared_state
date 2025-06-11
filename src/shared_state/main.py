#!/usr/bin/env python
from dotenv import load_dotenv
load_dotenv(override=True)

import json
import logging
from enum import Enum
from typing import Optional, List, Any
import litellm
from crewai.flow import start
from pydantic import BaseModel, Field
from copilotkit.crewai import (
    CopilotKitFlow,
    tool_calls_log,
    FlowInputState,
    emit_copilotkit_state_update_event
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

# Tool 1: Generate basic recipe info
GENERATE_RECIPE_BASICS_TOOL = {
    "type": "function",
    "function": {
        "name": "generate_recipe_basics",
        "description": "Generate basic recipe information like title, skill level, dietary preferences, and cooking time.",
        "parameters": {
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
            },
            "required": ["title", "skill_level", "cooking_time", "dietary_preferences"],
        },
    },
}

# Tool 2: Generate ingredients
GENERATE_INGREDIENTS_TOOL = {
    "type": "function",
    "function": {
        "name": "generate_ingredients",
        "description": "Generate the list of ingredients for the recipe.",
        "parameters": {
            "type": "object",
            "properties": {
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
            },
            "required": ["ingredients"],
        },
    },
}

# Tool 3: Generate instructions
GENERATE_INSTRUCTIONS_TOOL = {
    "type": "function",
    "function": {
        "name": "generate_instructions",
        "description": "Generate step-by-step cooking instructions for the recipe.",
        "parameters": {
            "type": "object",
            "properties": {
                "instructions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Step-by-step instructions for preparing the recipe.",
                },
            },
            "required": ["instructions"],
        },
    },
}

logging.basicConfig(level=logging.INFO)

class Recipe(BaseModel):
    """
    A recipe.
    """
    title: Optional[str] = None
    skill_level: Optional[SkillLevel] = None
    dietary_preferences: List[str] = Field(default_factory=list)
    cooking_time: Optional[CookingTime] = None
    ingredients: List[Ingredient] = Field(default_factory=list)
    instructions: List[str] = Field(default_factory=list)


class AgentState(FlowInputState):
    """
    The state of the recipe.
    """
    data: dict[str, Any] = Field(default_factory=lambda: {"recipe": None}, description="Data containing recipe")


@persist()
class SharedStateFlow(CopilotKitFlow[AgentState]):
    @start()
    def chat(self):
        """
        Sequential recipe generation with streaming-like updates.
        """
        print(f"DEBUG: Current recipe state: {self.state.data['recipe']}")
        print(f"DEBUG: Current messages: {self.state.messages}")

        # Initialize empty recipe if none exists
        if not self.state.data['recipe']:
            self.state.data['recipe'] = Recipe().model_dump()

        # Get user request
        user_request = self.state.messages[-1].get('content', '') if self.state.messages else ''

        # Execute all steps sequentially for streaming experience
        try:
            # Step 1: Generate basics
            self._generate_basics_step(user_request)

            # Step 2: Generate ingredients
            self._generate_ingredients_step()

            # Step 3: Generate instructions
            self._generate_instructions_step()

            # Add to conversation history
            for msg in self.state.messages:
                if msg.get('role') == 'user' and msg not in self.state.conversation_history:
                    self.state.conversation_history.append(msg)

            assistant_message = {"role": "assistant", "content": "🎉 Recipe completed! I've created it step by step for you."}
            self.state.conversation_history.append(assistant_message)

            return json.dumps({
                "response": "🎉 Recipe completed! I've created it step by step for you.",
                "id": self.state.id
            })

        except Exception as e:
            return json.dumps({
                "response": f"An error occurred: {str(e)}",
                "id": self.state.id
            })

    def _generate_basics_step(self, user_request: str):
        """Step 1: Generate basic recipe information"""
        print("DEBUG: *** STEP 1: GENERATING BASICS ***")

        system_prompt = f"""
        You are creating a recipe. Generate the basic information (title, skill level, dietary preferences, cooking time).
        User request: {user_request}
        Use the generate_recipe_basics tool to create this information.
        """

        messages = self.get_message_history(system_prompt=system_prompt)

        response = litellm.completion(
            model="gpt-4o",
            messages=messages,
            tools=[GENERATE_RECIPE_BASICS_TOOL],
            stream=False,
            temperature=0.7
        )

        # Process tool call
        if hasattr(response, 'choices') and response.choices[0].message.tool_calls:
            tool_call = response.choices[0].message.tool_calls[0]
            function_args = json.loads(tool_call.function.arguments)
            self.generate_recipe_basics_handler(function_args)

    def _generate_ingredients_step(self):
        """Step 2: Generate ingredients"""
        print("DEBUG: *** STEP 2: GENERATING INGREDIENTS ***")

        current_recipe = self.state.data['recipe']
        system_prompt = f"""
        You are adding ingredients to a recipe. Here's the current recipe basics:
        Title: {current_recipe.get('title', 'Unknown')}
        Skill Level: {current_recipe.get('skill_level', 'Unknown')}
        Cooking Time: {current_recipe.get('cooking_time', 'Unknown')}
        Dietary Preferences: {current_recipe.get('dietary_preferences', [])}

        Generate appropriate ingredients for this recipe using the generate_ingredients tool.
        """

        messages = [{"role": "system", "content": system_prompt}]

        response = litellm.completion(
            model="gpt-4o",
            messages=messages,
            tools=[GENERATE_INGREDIENTS_TOOL],
            stream=False,
            temperature=0.7
        )

        # Process tool call
        if hasattr(response, 'choices') and response.choices[0].message.tool_calls:
            tool_call = response.choices[0].message.tool_calls[0]
            function_args = json.loads(tool_call.function.arguments)
            self.generate_ingredients_handler(function_args)

    def _generate_instructions_step(self):
        """Step 3: Generate cooking instructions"""
        print("DEBUG: *** STEP 3: GENERATING INSTRUCTIONS ***")

        current_recipe = self.state.data['recipe']
        system_prompt = f"""
        You are completing a recipe by adding cooking instructions. Here's the current recipe:
        Title: {current_recipe.get('title', 'Unknown')}
        Ingredients: {len(current_recipe.get('ingredients', []))} ingredients

        Generate step-by-step cooking instructions using the generate_instructions tool.
        """

        messages = [{"role": "system", "content": system_prompt}]

        response = litellm.completion(
            model="gpt-4o",
            messages=messages,
            tools=[GENERATE_INSTRUCTIONS_TOOL],
            stream=False,
            temperature=0.7
        )

        # Process tool call
        if hasattr(response, 'choices') and response.choices[0].message.tool_calls:
            tool_call = response.choices[0].message.tool_calls[0]
            function_args = json.loads(tool_call.function.arguments)
            self.generate_instructions_handler(function_args)

    def generate_recipe_basics_handler(self, data):
        """Handler for recipe basics generation"""
        print(f"DEBUG: *** GENERATING RECIPE BASICS ***")

        # Update recipe with basics
        if not self.state.data['recipe']:
            self.state.data['recipe'] = Recipe().model_dump()

        current_recipe = self.state.data['recipe']
        current_recipe.update({
            'title': data['title'],
            'skill_level': data['skill_level'],
            'dietary_preferences': data['dietary_preferences'],
            'cooking_time': data['cooking_time']
        })

        self.state.data['recipe'] = current_recipe

        # Emit state update
        emit_copilotkit_state_update_event(
            tool_name="generate_recipe_basics",
            args=current_recipe
        )

        print(f"DEBUG: *** EMITTED BASICS STATE UPDATE ***")
        return "Recipe basics created"

    def generate_ingredients_handler(self, data):
        """Handler for ingredients generation"""
        print(f"DEBUG: *** GENERATING INGREDIENTS ***")

        current_recipe = self.state.data['recipe']
        current_recipe['ingredients'] = data['ingredients']
        self.state.data['recipe'] = current_recipe

        # Emit state update
        emit_copilotkit_state_update_event(
            tool_name="generate_ingredients",
            args=current_recipe
        )

        print(f"DEBUG: *** EMITTED INGREDIENTS STATE UPDATE ***")
        return "Ingredients added"

    def generate_instructions_handler(self, data):
        """Handler for instructions generation"""
        print(f"DEBUG: *** GENERATING INSTRUCTIONS ***")

        current_recipe = self.state.data['recipe']
        current_recipe['instructions'] = data['instructions']
        self.state.data['recipe'] = current_recipe

        # Validate final recipe
        recipe_obj = Recipe(**current_recipe)
        self.state.data['recipe'] = recipe_obj.model_dump()

        # Emit final state update
        emit_copilotkit_state_update_event(
            tool_name="generate_instructions",
            args=self.state.data['recipe']  # Use the validated final recipe
        )

        print(f"DEBUG: *** EMITTED FINAL STATE UPDATE ***")
        return "Recipe completed"


def kickoff():
    shared_state_flow = SharedStateFlow()
    result = shared_state_flow.kickoff({
        "state": {
            "data": {
                "recipe": None
            }
        },
        "messages": [
            {
                "role": "user",
                "content": "Create a simple pasta recipe for beginners"
            }
        ]
    })
    print(result)

def plot():
    shared_state_flow = SharedStateFlow()
    shared_state_flow.plot()


if __name__ == "__main__":
    kickoff()
