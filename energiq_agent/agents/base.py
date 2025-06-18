from typing import Any
from energiq_agent.agents.client import get_openai_client
from abc import ABC, abstractmethod


class BaseAgent(ABC):
    client: Any
    history: list[dict[str, str]]
    model: str
    reasoning_effort: str

    def __init__(
        self,
        client=None,
        model: str = "qwen3:32b",
        reasoning_effort: str = "low",
    ):
        self.client = client or get_openai_client()
        self.history = [{"role": "system", "content": self.prompt()}]
        self.model = model
        self.reasoning_effort = reasoning_effort

    def reset_history(self):
        self.history = [{"role": "system", "content": self.prompt()}]

    def run(self, input_message: str, reset_history: bool = False) -> str:
        msg_input = [
            {
                "role": "user",
                "content": f"{input_message}",
            },
        ]

        self.history = self.history + msg_input

        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.history,
            reasoning_effort=self.reasoning_effort,
        )

        rsp = response.choices[0].message.content
        rsp = rsp.replace("\n", " ")

        if "</think>" in rsp:
            rsp = rsp.split("</think>")[1]

        if reset_history:
            self.reset_history()
        else:
            self.history.append({"role": "assistant", "content": rsp})

        return rsp

    @abstractmethod
    def prompt(self) -> str: ...
