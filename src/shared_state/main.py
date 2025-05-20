#!/usr/bin/env python
from dotenv import load_dotenv
load_dotenv(override=True)

from pydantic import BaseModel
from crewai.flow import Flow, start
from crewai import LLM

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

        if self.state.messages:
            messages.append(self.state.messages[-1])
        
        response = llm.call(messages)

        self.state.messages.append({
            "role": "assistant", 
            "content": response
        })
        return response


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
