from qwen_agent.agents import Assistant
from qwen_agent.gui import WebUI

llm_cfg = {
    "model": "qwen3:32b",
    "model_server": "http://localhost:11434/v1/",
    "api_key": "",
    "generate_cfg": {"top_p": 0.8},
}

system_instruction = """
/no_think You will receive the action plan from the power grid operator.
You need to follow the action plan to resolve the violations in the network.
You should use the network in editing to resolve the violations.
You can ignore the actions that hasn't support in tools.
After all actions are executed, get the network status and return status file path to user.
"""

tools = [
    {
        "mcpServers": {
            "filesystem": {
                "command": "npx",
                "args": [
                    "-y",
                    "@modelcontextprotocol/server-filesystem",
                    "D:\\Onedrive\\Yan\\study\\articles",
                    "D:\\Dev\\repo\\faraday\\data",
                ],
            },
            "simulator": {
                "command": "uv",
                "args": [
                    "--directory",
                    "D:\\Dev\\repo\\faraday",
                    "run",
                    "faraday\\simulator\\api.py",
                ],
            },
        },
    },
    "code_interpreter",
]


def get_bot():
    files = []
    bot = Assistant(
        llm=llm_cfg, system_message=system_instruction, function_list=tools, files=files
    )
    return bot


if __name__ == "__main__":
    bot = get_bot()
    WebUI(bot).run()
