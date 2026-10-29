import asyncio
import os
from dotenv import load_dotenv
load_dotenv()
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

load_dotenv()

async def main():
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("MODEL", "gpt-4o-mini")

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is missing. Check your .env file."
        )

    model_client = OpenAIChatCompletionClient(
        model=model,
        api_key=api_key,
    )

    try:
        agent = AssistantAgent(
            name="hr_assistant",
            model_client=model_client,
            system_message="You are an HR assistant.",
        )

        await Console(
            agent.run_stream(
                task=(
                    "Write a job description for a Junior Python Developer "
                    "in exactly five bullet points. Each bullet must be "
                    "one line. Do not add a title or introductory text."
                )
            )
        )

    finally:
        await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())

    