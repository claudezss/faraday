from qwen_agent.agents import Assistant
from qwen_agent.utils.output_beautify import typewriter_print
import json
import re


class Executor:
    bot: Assistant
    history: list[dict[str, str]]

    def __init__(
        self,
        top_p: float = 0.8,
    ):
        llm_cfg = {
            "model": "qwen3:32b",
            "model_server": "http://localhost:11434/v1/",
            "api_key": "",
            "generate_cfg": {"top_p": top_p},
        }

        self.bot = Assistant(
            llm=llm_cfg,
            system_message=self.prompt,
            function_list=self.tools,
            files=self.files,
        )
        self.history = [{"role": "system", "content": self.prompt}]

    def run(self, input_message: str) -> str:
        response_plain_text = ""
        messages = self.history + [{"role": "user", "content": input_message}]
        for response in self.bot.run(messages=messages):
            # Streaming output.
            response_plain_text = typewriter_print(response, response_plain_text)

        self.history = messages + [
            {"role": "assistant", "content": response_plain_text}
        ]

        return self.extract_json_from_llm_response(response_plain_text)["path"]

    @staticmethod
    def extract_json_from_llm_response(response: str) -> dict:
        # Use regex to find the first JSON code block
        match = re.search(r"```json\s*(\{.*?})\s*```", response, re.DOTALL)
        if match:
            json_str = match.group(1)
            return json.loads(json_str)
        else:
            raise ValueError("No JSON block found in the response")

    @property
    def files(self) -> list[str]:
        return []

    @property
    def tools(self) -> list:
        return [
            {
                "mcpServers": {
                    "filesystem": {
                        "command": "npx",
                        "args": [
                            "-y",
                            "@modelcontextprotocol/server-filesystem",
                            "D:\\Onedrive\\Yan\\study\\articles",
                            "D:\\Dev\\repo\\EnergiQ-Agent\\data",
                        ],
                    },
                    "simulator": {
                        "command": "uv",
                        "args": [
                            "--directory",
                            "D:\\Dev\\repo\\EnergiQ-Agent",
                            "run",
                            "energiq_agent\\simulator\\api.py",
                        ],
                    },
                },
            },
            "code_interpreter",
        ]

    @property
    def prompt(self) -> str:
        return """
        /no_think
        You will receive the action plan from the power grid operator.
        You need to follow the action plan to resolve the violations in the network.
        You should use the network in editing to resolve the violations.
        You can ignore the actions that hasn't support in tools.
        After all actions are executed, get the network status and return status file path to user.
        
        Output Format:

        Return ONLY the absolute path to the saved network status file and follow this format. 
        
        ```json
        {
            "path": ${state_path}
        }
        ```
        Do not include any other text or explanation.
        """
