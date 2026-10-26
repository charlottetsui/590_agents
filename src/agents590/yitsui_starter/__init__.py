import json
from typing import cast

from openai.types.responses import FunctionToolParam, ResponseInputParam

from agents590.contracts import RunContext
from agents590.provider import UNCOpenAISettings
from agents590.yitsui_starter.tools import run_command

TOOLS: list[FunctionToolParam] = [
    {
        "type": "function",
        "name": "run_command",
        "description": "Run a shell command in the task directory.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]


class YiTsuiStarterAgent:
    max_iterations = 25

    def provider_settings(self, context: RunContext) -> UNCOpenAISettings:
        """Prefer the per-run model to the environment's configured deployment."""
        return UNCOpenAISettings.from_env(model=context.model)

    async def run(self, instruction: str, context: RunContext) -> str:
        settings = self.provider_settings(context)
        input_list: ResponseInputParam = [{"role": "user", "content": instruction}]
        async with settings.create_async_client() as client:
            for _ in range(self.max_iterations):
                response = await client.responses.create(
                    model=settings.model,
                    input=input_list,
                    tools=TOOLS,
                    include=["reasoning.encrypted_content"],
                    store=False,
                    background=False,
                )
                if response.status != "completed":
                    break
                input_list += cast(ResponseInputParam, response.output)
                calls = [item for item in response.output if item.type == "function_call"]

                if not calls:
                    return response.output_text

                for call in calls:
                    arguments = json.loads(call.arguments)
                    output = await run_command(arguments["command"], context.cwd)
                    input_list.append(
                        {
                            "type": "function_call_output",
                            "call_id": call.call_id,
                            "output": output,
                        }
                    )
        raise ValueError("Agent could not complete the task.")
