#!/usr/bin/env python
from dotenv import load_dotenv
load_dotenv(override=True)

from pydantic import BaseModel
from crewai.flow import Flow, start
from crewai import LLM
from pprint import pprint
import logging
import json

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)

class SharedState(BaseModel):
    messages: list[dict[str, str]] = []

llm = LLM(
    model="gpt-4o"
)
class SharedStateFlow(Flow[SharedState]):

    @start()
    def chat(self):
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            }
        ]

        logger.info(f"Initial input state: {self.state}")
        logger.info(f"Raw state dump: {json.dumps(self.state.model_dump(), indent=2)}")
        logger.info(f"Messages in state: {self.state.messages}")

        logger.info(f"####STATE####\n{json.dumps(self.state.model_dump(), indent=2)}")

        if self.state.messages:
            messages.append(self.state.messages[-1])

        response = llm.call(messages)

        self.state.messages.append({
            "role": "assistant",
            "content": response
        })
        return response

    def __repr__(self):
        pprint(vars(self), width=120, depth=3)

def kickoff():
    shared_state_flow = SharedStateFlow()
    result = shared_state_flow.kickoff(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Hello, how are you?"
                }
            ]
        }
    )
    print(result)

def plot():
    shared_state_flow = SharedStateFlow()
    shared_state_flow.plot()


if __name__ == "__main__":
    kickoff()
