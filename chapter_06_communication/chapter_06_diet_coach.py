"""
Mastering Agentic AI — Chapter 6
Communication Protocols

Demonstrates cross-framework agent communication using the A2A protocol.
A Google ADK Workout Agent is published as an A2A server, and a CrewAI
Diet Coach discovers and calls it over the wire.
"""

import asyncio
import threading
import time

import uvicorn

from crewai import Agent as CrewAgent, Crew, Task
from a2a.client import ClientConfig, ClientFactory, create_text_message_object
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill, TransportProtocol
from a2a.utils.constants import AGENT_CARD_WELL_KNOWN_PATH
from google.adk.a2a.executor.a2a_agent_executor import (
    A2aAgentExecutor,
    A2aAgentExecutorConfig,
)
from google.adk.agents import Agent as AdkAgent
from google.adk.artifacts import InMemoryArtifactService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService


# ADK side: define one agent and publish it through A2A.
workout_agent = AdkAgent(
    model="gemini-2.5-pro",
    name="workout_agent",
    instruction=(
        "Create short, beginner-friendly running plans. "
        "Reply with only the plan."
    ),
)

workout_agent_card = AgentCard(
    name="Workout Agent",
    url="http://127.0.0.1:10020",
    description="Creates running plans for beginners.",
    version="1.0",
    capabilities=AgentCapabilities(streaming=False),
    default_input_modes=["text/plain"],
    default_output_modes=["text/plain"],
    preferred_transport=TransportProtocol.jsonrpc,
    skills=[
        AgentSkill(
            id="build_running_plan",
            name="Build Running Plan",
            description="Creates a beginner-friendly 5k plan.",
            examples=["Create a 4-week 5k plan for a beginner."],
        )
    ],
)


def create_agent_a2a_server(agent, agent_card):
    runner = Runner(
        app_name=agent.name,
        agent=agent,
        artifact_service=InMemoryArtifactService(),
        session_service=InMemorySessionService(),
        memory_service=InMemoryMemoryService(),
    )

    executor = A2aAgentExecutor(
        runner=runner,
        config=A2aAgentExecutorConfig(),
    )

    handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
    )

    return A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=handler,
    )


async def run_workout_server():
    app = create_agent_a2a_server(workout_agent, workout_agent_card)
    config = uvicorn.Config(
        app.build(),
        host="127.0.0.1",
        port=10020,
        log_level="warning",
        loop="none",
    )
    server = uvicorn.Server(config)
    await server.serve()


def start_server_in_background():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(run_workout_server())


class CrewA2ATool:
    async def ask(self, prompt: str) -> str:
        client = ClientFactory(ClientConfig()).create(workout_agent_card)
        message = create_text_message_object(content=prompt)

        async for response in client.send_message(message):
            task = response[0]
            return task.artifacts[0].parts[0].root.text

        return "No response received."


async def crewai_agent_calls_adk() -> None:
    a2a_tool = CrewA2ATool()

    diet_coach = CrewAgent(
        role="Diet Coach",
        goal="Help the user combine food advice with a simple workout plan.",
        backstory="You are a friendly coach who can call a remote workout agent.",
        verbose=False,
    )

    request = (
        "Ask the remote Workout Agent for a beginner-friendly 4-week 5k plan "
        "for someone who has not been running recently."
    )
    remote_reply = await a2a_tool.ask(request)

    task = Task(
        description=(
            "Summarize this workout plan for the user in a warm, simple tone:\n\n"
            f"{remote_reply}"
        ),
        expected_output="A short, friendly summary of the workout plan.",
        agent=diet_coach,
    )

    crew = Crew(agents=[diet_coach], tasks=[task], verbose=False)
    result = crew.kickoff()

    print("CrewAI discovered the ADK agent at:")
    print(f"  http://127.0.0.1:10020{AGENT_CARD_WELL_KNOWN_PATH}")
    print("\nReply from the ADK Workout Agent:\n")
    print(remote_reply)
    print("\nFinal message from the CrewAI Diet Coach:\n")
    print(result)


if __name__ == "__main__":
    threading.Thread(target=start_server_in_background, daemon=True).start()
    time.sleep(2)
    asyncio.run(crewai_agent_calls_adk())
