"""
Mastering Agentic AI — Chapter 6
Communication Protocols (Final Version)

Sections covered:
  6.1  Fundamentals — why communication fails, four mechanics of reliable dialogue
  6.2  Human-in-the-Loop Protocols — HITL gate, graduated autonomy, behavioural telemetry
  6.3  Google A2A Protocol — AgentCard, cross-framework demo, error handling
  6.4  IBM ACP — enterprise structured messaging
  6.5  Coordination and Negotiation — planner/worker/critic, LLM Council
  6.6  MCP + A2A composition — two-layer production architecture
  6.7  Production deployment — persistence, tracing, circuit breakers, security,
       performance, event-driven patterns

Key implementations in this file:
  • HITLGate        — review queue with graduated autonomy (Section 6.2)
  • CrewA2ATool     — A2A client with full error handling (Section 6.3.4)
  • council_query() — LLM Council three-stage deliberation (Section 6.5.2)
  • CircuitBreaker  — agent communication resilience (Section 6.7.3)
  • event_driven_demo() — sync vs async agent communication (Section 6.7.6)
  • MCP + A2A composition — two-layer production pattern (Section 6.6)

The two-layer principle:
  • MCP — tool layer: systems that respond but do not reason
  • A2A — agent layer: peers that reason, plan, have their own tools
  A production agent uses both simultaneously, with each layer independent.
"""

import asyncio
import os
import threading
import time

import uvicorn

from crewai import Agent as CrewAgent, Crew, Task

# A2A client imports — Section 6.3.4
# ClientFactory   : creates a protocol-aware A2A client from an AgentCard
# ClientConfig    : holds transport configuration (timeouts, retries)
# create_text_message_object : wraps a string in the A2A message envelope
from a2a.client import ClientFactory, ClientConfig, create_text_message_object
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

# Section 6.5.6 — MCP imports for the two-layer composition demo
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from google.adk.tools.toolbox_toolset import ToolboxToolset
from mcp import StdioServerParameters


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
    """
    Section 6.3.4: A2A client wrapper for the Workout Agent.

    Handles the three failure cases that appear most often in production:
      • ConnectionError — network unreachable; agent server not running
      • TimeoutError    — server reached but call hung without completing
      • Exception       — any other unexpected error; logged and surfaced
    """

    async def ask(self, prompt: str) -> str:
        try:
            client  = ClientFactory(ClientConfig()).create(workout_agent_card)
            message = create_text_message_object(content=prompt)
            async for response in client.send_message(message):
                task = response[0]
                return task.artifacts[0].parts[0].root.text
            return "No response received."
        except ConnectionError as e:
            return f"[Error] Could not reach Workout Agent: {e}"
        except TimeoutError:
            return "[Error] Workout Agent timed out — retry or escalate."
        except Exception as e:
            return f"[Error] A2A call failed: {type(e).__name__}: {e}"


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
    run_all_demos()

# ─────────────────────────────────────────────────────────────────────────────
# 6.2  Human-in-the-Loop — Review Queue Implementation
#
# HITL is load-bearing infrastructure. The HITLGate class implements the
# three-lane graduated autonomy model from Section 6.2.3:
#
#   Low-stakes  + high confidence  → auto-approve and log
#   Medium-stakes                  → enter review queue; block until decision
#   High-stakes or low confidence  → escalate immediately; do not execute
#
# The review queue pattern:
#   1. Agent proposes an action as a PendingAction dataclass
#   2. HITLGate.propose() routes it to the correct lane
#   3. Medium-stakes actions block in _wait_for_review() until a human
#      updates action.status via approve() or reject()
#   4. Every outcome is logged for behavioural telemetry
# ─────────────────────────────────────────────────────────────────────────────

import asyncio as _asyncio
from dataclasses import dataclass, field as _field
from datetime import datetime as _datetime
from enum import Enum as _Enum


class ReviewStatus(_Enum):
    PENDING  = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


@dataclass
class PendingAction:
    """
    A proposed agent action awaiting human review.

    confidence:  0.0–1.0 — how certain the agent is this action is correct
    stake_level: 'low' | 'medium' | 'high' — drives which HITL lane is used
    """
    action_id:   str
    description: str
    payload:     dict
    confidence:  float
    stake_level: str
    status:      ReviewStatus = ReviewStatus.PENDING
    created_at:  str = _field(default_factory=lambda: _datetime.now().isoformat())


class HITLGate:
    """
    Section 6.2.3: Human-in-the-Loop review gate.

    Three lanes based on stake level and confidence:
      - Auto-approve: stake=low AND confidence >= 0.85
      - Review queue: medium stakes (blocks until human decides)
      - Escalate:     stake=high OR confidence < 0.5
    """

    def __init__(self) -> None:
        self.queue: list[PendingAction] = []
        self.log:   list[dict]          = []

    async def propose(self, action: PendingAction) -> bool:
        """Route action to correct lane. Returns True if it should proceed."""
        if action.stake_level == "low" and action.confidence >= 0.85:
            self._log(action, "auto-approved")
            return True
        if action.stake_level == "high" or action.confidence < 0.5:
            self._log(action, "escalated")
            return False   # block immediately; notify human out-of-band
        # Medium-stakes: enter review queue and wait
        self.queue.append(action)
        return await self._wait_for_review(action)

    async def _wait_for_review(self, action: PendingAction) -> bool:
        """Block until approved or rejected. Times out after 5 minutes."""
        for _ in range(300):           # poll every second for up to 5 minutes
            await _asyncio.sleep(1)
            if action.status == ReviewStatus.APPROVED:
                self._log(action, "approved")
                return True
            if action.status == ReviewStatus.REJECTED:
                self._log(action, "rejected")
                return False
        self._log(action, "timed-out")  # treat timeout as rejection
        return False

    def approve(self, action_id: str) -> None:
        """Human approves a pending action — called from review UI."""
        for action in self.queue:
            if action.action_id == action_id:
                action.status = ReviewStatus.APPROVED

    def reject(self, action_id: str) -> None:
        """Human rejects a pending action — called from review UI."""
        for action in self.queue:
            if action.action_id == action_id:
                action.status = ReviewStatus.REJECTED

    def _log(self, action: PendingAction, outcome: str) -> None:
        self.log.append({
            "action_id":  action.action_id,
            "outcome":    outcome,
            "confidence": action.confidence,
            "stake":      action.stake_level,
            "timestamp":  _datetime.now().isoformat(),
        })


def demonstrate_hitl_gate() -> None:
    """
    Section 6.2.3: Show all three HITL lanes with concrete examples.

    Run with:  asyncio.run(demonstrate_hitl_gate_async())
    """
    asyncio.run(_demonstrate_hitl_gate_async())


async def _demonstrate_hitl_gate_async() -> None:
    gate = HITLGate()

    # Lane 1: Auto-approve — low-stakes, high confidence
    low = PendingAction("act-001", "Categorise support ticket #4421",
                        {"ticket_id": "4421"}, confidence=0.92, stake_level="low")
    result = await gate.propose(low)
    print(f"[Auto-approve] {low.description}: {'proceed' if result else 'blocked'}")

    # Lane 2: Escalate — high-stakes regardless of confidence
    high = PendingAction("act-002", "Execute wire transfer £50,000",
                         {"amount": 50000, "currency": "GBP"},
                         confidence=0.88, stake_level="high")
    result = await gate.propose(high)
    print(f"[Escalate]     {high.description}: {'proceed' if result else 'blocked'}")

    # Lane 3: Review queue — medium-stakes (immediately rejected for demo speed)
    medium = PendingAction("act-003", "Send refund email to customer",
                           {"customer_id": "C-9921", "amount": 120},
                           confidence=0.75, stake_level="medium")
    medium.status = ReviewStatus.REJECTED   # simulate human rejection
    result = await gate.propose(medium)
    print(f"[Review queue] {medium.description}: {'proceed' if result else 'blocked'}")

    print(f"\nAudit log: {len(gate.log)} entries")
    for entry in gate.log:
        print(f"  {entry['action_id']} → {entry['outcome']}"
              f" (confidence={entry['confidence']}, stake={entry['stake']})")


# ─────────────────────────────────────────────────────────────────────────────
# 6.5.2  LLM Council — Three-Stage Deliberation
#
# Instead of querying a single model, a council of diverse models deliberates:
#   Stage 1: Independent generation — each model answers without seeing others
#   Stage 2: Anonymous cross-review — each model ranks the others' answers
#   Stage 3: Chairman synthesis    — one model integrates all views
#
# Cost note: a council of N models costs (2N + 1) model calls per query.
# Use only when quality significantly outweighs latency and cost.
# Diversity matters: different model families, not just different sizes.
# ─────────────────────────────────────────────────────────────────────────────

import random as _random
from openai import OpenAI as _OpenAI

_council_client = _OpenAI()
COUNCIL_MODELS  = ["gpt-4o", "gpt-4o-mini"]   # add models via OpenRouter


def _council_ask(model: str, question: str) -> str:
    """Single model call — base primitive for all three council stages."""
    return _council_client.chat.completions.create(
        model=model, max_tokens=500,
        messages=[{"role": "user", "content": question}]
    ).choices[0].message.content


def _council_review(model: str, question: str, answers: list[str]) -> str:
    """Stage 2: anonymised cross-review — model ranks shuffled answers."""
    prompt = (
        f"Question: {question}\n\nAnonymous answers:\n"
        + "\n---\n".join(answers)
        + "\n\nRank these by accuracy. Explain briefly."
    )
    return _council_ask(model, prompt)


def _council_synthesise(model: str, question: str,
                         opinions: dict, reviews: dict) -> str:
    """Stage 3: Chairman integrates all answers and reviews."""
    context = "\n".join(f"{m}: {a}" for m, a in opinions.items())
    review_ctx = "\n".join(f"{m}: {r}" for m, r in reviews.items())
    prompt = (
        f"Question: {question}\nAll answers:\n{context}\n"
        f"Reviews:\n{review_ctx}\n\nSynthesise the best answer."
    )
    return _council_ask(model, prompt)


def council_query(question: str, chairman: str = "gpt-4o") -> dict:
    """
    Section 6.5.2: Run a full LLM Council deliberation.

    Returns a dict with 'opinions', 'reviews', and 'synthesis'.
    Cost: (2 × len(COUNCIL_MODELS) + 1) model calls per question.

    Args:
        question: The question to deliberate on.
        chairman: Model that performs the final synthesis.
    """
    # Stage 1: Independent — no model sees another's answer
    opinions = {m: _council_ask(m, question) for m in COUNCIL_MODELS}

    # Stage 2: Anonymous review — shuffle to prevent anchoring bias
    anon = list(opinions.values())
    _random.shuffle(anon)
    reviews = {m: _council_review(m, question, anon) for m in COUNCIL_MODELS}

    # Stage 3: Chairman synthesises all answers + reviews
    synthesis = _council_synthesise(chairman, question, opinions, reviews)

    return {"opinions": opinions, "reviews": reviews, "synthesis": synthesis}


# ─────────────────────────────────────────────────────────────────────────────
# 6.7.3  Circuit Breaker — Agent Communication Resilience
#
# Treats an unresponsive downstream agent the way a service mesh treats
# an unhealthy pod. After threshold consecutive failures, the circuit opens:
#   • Requests stop immediately — no more calls to the failing agent
#   • After cooldown_s seconds, a single probe is sent
#   • If the probe succeeds, the circuit closes and normal operation resumes
# ─────────────────────────────────────────────────────────────────────────────

import time as _time


class CircuitBreaker:
    """
    Section 6.7.3: Circuit breaker for A2A agent calls.

    Usage:
        cb = CircuitBreaker(threshold=3, cooldown_s=60)
        if cb.is_open():
            return "[Circuit open] Workout Agent unavailable — try later."
        try:
            result = await a2a_tool.ask(prompt)
            cb.record_success()
            return result
        except (ConnectionError, TimeoutError):
            cb.record_failure()
            return "[Error] Call failed — circuit may open."
    """

    def __init__(self, threshold: int = 3, cooldown_s: int = 60) -> None:
        self.failures   = 0
        self.threshold  = threshold
        self.cooldown_s = cooldown_s
        self.open_until = 0.0   # Unix timestamp; 0 means circuit is closed

    def is_open(self) -> bool:
        """True if the circuit is open (requests should be blocked)."""
        return (self.failures >= self.threshold
                and _time.time() < self.open_until)

    def record_failure(self) -> None:
        """Call after each failed A2A call."""
        self.failures  += 1
        self.open_until = _time.time() + self.cooldown_s

    def record_success(self) -> None:
        """Call after each successful A2A call — resets the counter."""
        self.failures = 0


# ─────────────────────────────────────────────────────────────────────────────
# 6.7.6  Event-Driven Patterns — Sync vs Async Agent Communication
#
# Every other demo in this chapter uses synchronous A2A: the calling agent
# blocks until the downstream agent responds. This works for sub-second tasks.
# For workflows spanning hours or days — overnight research, multi-day document
# processing, workflows with a human approval step in the middle — a durable
# event-driven pattern is more appropriate.
#
# The contrast:
#   Synchronous A2A  : caller blocks until response; fails if agent is slow
#   Event-driven     : caller emits to a stream and continues; downstream
#                      agent processes when available; result arrives as event
# ─────────────────────────────────────────────────────────────────────────────

import json as _json


async def demonstrate_event_driven_pattern() -> None:
    """
    Section 6.7.6: Show the sync vs event-driven contrast.

    The synchronous pattern (used throughout this chapter) blocks on each
    A2A call. The event-driven pattern emits to a Redis stream and returns
    immediately — the downstream agent processes when available.

    Requires: pip install redis
    Set REDIS_URL env var, or the demo skips the Redis step gracefully.
    """
    print("\n── Section 6.7.6: Sync vs Event-Driven Agent Communication ────")

    # ── Synchronous A2A (used throughout this chapter) ────────────────────
    print("\nPattern 1: Synchronous A2A")
    print("  result = await a2a_tool.ask('Create a running plan')")
    print("  # Blocks here until the Workout Agent responds.")
    print("  # Suitable for sub-second tasks; fails if agent is slow.")

    # ── Event-driven alternative ──────────────────────────────────────────
    print("\nPattern 2: Event-driven (Redis Streams)")
    print("  Emits task to durable stream; returns immediately.")
    print("  Downstream agent processes when available.")
    print("  Result arrives on agent:results stream — caller not blocked.")

    task_payload = {
        "task_id":     "task-001",
        "task_type":   "create_running_plan",
        "request":     "4-week 5k plan for a beginner",
        "callback":    "agent:results",
        "created_at":  _datetime.now().isoformat(),
    }

    redis_url = os.getenv("REDIS_URL", "")
    if redis_url:
        try:
            import redis.asyncio as _redis
            r = await _redis.from_url(redis_url)
            # Emit task — returns immediately, no blocking
            msg_id = await r.xadd(
                "agent:tasks", {"payload": _json.dumps(task_payload)}
            )
            print(f"\n  [Redis] Task emitted to agent:tasks stream: {msg_id}")
            print(f"  [Redis] Payload: {_json.dumps(task_payload, indent=4)}")
            print("  [Redis] Agent will process when available.")
            print("  [Redis] Result will appear on agent:results stream.")
            await r.aclose()
        except Exception as e:
            print(f"  [Redis] Connection failed: {e}")
            print("  Set REDIS_URL to run the live demo.")
    else:
        print("\n  [Simulated] Task payload that would be emitted:")
        print(f"  stream : agent:tasks")
        print(f"  payload: {_json.dumps(task_payload, indent=4)}")
        print("\n  Set REDIS_URL env var to emit to a real Redis stream.")
        print("  Free managed Redis: https://upstash.com (10k commands/day free)")

    print("\n── When to use each pattern ─────────────────────────────────────")
    print("  Synchronous : task completes in < 30s; simple request-response")
    print("  Event-driven: task spans hours/days; human approval step in middle;")
    print("                overnight research; multi-document processing pipelines")


# ─────────────────────────────────────────────────────────────────────────────
# 6.5.6  Composing MCP and A2A in a Single Agent
#
# MCP and A2A are not competing standards — they operate on different planes:
#
#   MCP  — the tool layer: connects an agent to data sources, APIs, and
#           services that respond to queries but do not reason autonomously.
#           Example: nutrition database, meal logger, food search API.
#
#   A2A  — the agent layer: connects an agent to peer agents that have their
#           own reasoning, their own model, and their own internal tool chains.
#           Example: Workout Agent that plans, reasons, and uses its own MCP tools.
#
# A production agent sits between both layers simultaneously:
#   • It uses MCP to reach tools below it.
#   • It uses A2A to reach peer agents beside it.
#
# The two clients are independent and compose cleanly in the same agent loop.
# The caller never sees the callee's internal tool connections — each agent
# manages its own MCP layer; only its A2A interface is public.
#
# Architecture:
#
#   ┌──────────────────────────────────────────────────┐
#   │               Diet Coach (ADK)                   │
#   │  MCP tools (own): nutrition DB, meal logger      │
#   │  A2A client: calls Workout Agent over the wire   │
#   └───────────────┬──────────────────────────────────┘
#                   │ A2A
#   ┌───────────────▼──────────────────────────────────┐
#   │           Workout Agent (ADK) — A2A server       │
#   │  MCP tools (internal): fitness DB                │
#   │  Invisible to the Diet Coach — fully encapsulated│
#   └──────────────────────────────────────────────────┘
# ─────────────────────────────────────────────────────────────────────────────

# Environment variable for the nutrition toolbox server URL.
# In production, set NUTRITION_DB_URL to your MCP Toolbox endpoint.
NUTRITION_DB_URL = os.getenv("NUTRITION_DB_URL", "")


def build_fitness_mcp_tools() -> McpToolset | None:
    """
    Section 6.5.6 — Step 1a: MCP tool for the Workout Agent.

    Connects the Workout Agent to a fitness database via MCP.
    This connection is entirely internal — invisible to any A2A caller.
    Each agent manages its own MCP tool layer independently.

    Returns None if the MCP server is unavailable (graceful degradation).
    """
    try:
        return McpToolset(
            connection_params=StdioConnectionParams(
                server_params=StdioServerParameters(
                    command="npx",
                    args=["-y", "fitness-db-mcp-server"],
                ),
                timeout=30,
            )
        )
    except Exception as e:
        print(f"[MCP] Fitness DB tools unavailable: {e}")
        return None


def build_nutrition_mcp_tools() -> ToolboxToolset | None:
    """
    Section 6.5.6 — Step 3a: MCP tool for the Diet Coach.

    Connects the Diet Coach to a nutrition database via MCP Toolbox.
    This is the Diet Coach's own tool layer — entirely separate from the
    Workout Agent's MCP connection, operating independently.

    Returns None if NUTRITION_DB_URL is not set.
    """
    if not NUTRITION_DB_URL:
        print("[MCP] NUTRITION_DB_URL not set — skipping nutrition tools.")
        return None
    try:
        return ToolboxToolset(server_url=NUTRITION_DB_URL)
    except Exception as e:
        print(f"[MCP] Nutrition DB tools unavailable: {e}")
        return None


def build_mcp_equipped_workout_agent() -> AdkAgent:
    """
    Section 6.5.6 — Step 1b: Workout Agent with internal MCP tools.

    The agent's MCP fitness tools are encapsulated here. When wrapped in
    an A2A server, callers see only the A2A interface — the MCP layer is
    fully hidden inside. This is the key encapsulation principle: each
    agent owns its tool layer; only its agent interface is public.
    """
    fitness_tools = build_fitness_mcp_tools()
    tools = [fitness_tools] if fitness_tools else []

    return AdkAgent(
        model="gemini-2.5-pro",
        name="workout_agent_mcp",
        instruction=(
            "Create beginner-friendly running plans. "
            "Use the fitness database tool to check exercise load norms "
            "and weekly mileage recommendations when available."
        ),
        tools=tools,
    )


# AgentCard for the MCP-equipped Workout Agent.
# The card describes the public A2A interface only — MCP is not mentioned.
workout_agent_mcp_card = AgentCard(
    name="Workout Agent (MCP-equipped)",
    url="http://127.0.0.1:10021",   # separate port from the basic A2A demo
    description="Creates running plans using fitness data from an MCP database.",
    version="2.0",
    capabilities=AgentCapabilities(streaming=False),
    default_input_modes=["text/plain"],
    default_output_modes=["text/plain"],
    preferred_transport=TransportProtocol.jsonrpc,
    skills=[
        AgentSkill(
            id="build_running_plan_with_data",
            name="Build Data-Backed Running Plan",
            description=(
                "Creates a beginner-friendly 5k running plan informed by "
                "real fitness database norms."
            ),
            examples=["Create a 4-week 5k plan for a beginner."],
        )
    ],
)


async def run_mcp_workout_server() -> None:
    """
    Section 6.5.6 — Step 2: Publish the MCP-equipped Workout Agent over A2A.

    The agent uses MCP internally for fitness data; externally it is a
    standard A2A server. Two protocols, two layers, fully decoupled.
    """
    agent = build_mcp_equipped_workout_agent()
    app = create_agent_a2a_server(agent, workout_agent_mcp_card)
    config = uvicorn.Config(
        app.build(),
        host="127.0.0.1",
        port=10021,
        log_level="warning",
        loop="none",
    )
    await uvicorn.Server(config).serve()


def start_mcp_server_in_background() -> None:
    """Start the MCP-equipped Workout Agent server in a background thread."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(run_mcp_workout_server())


class McpWorkoutA2ATool(CrewA2ATool):
    """
    Section 6.5.6: A2A client pointed at the MCP-equipped Workout Agent.
    Inherits CrewA2ATool but targets port 10021 (workout_agent_mcp_card).
    """
    async def ask(self, prompt: str) -> str:
        client = ClientFactory(ClientConfig()).create(workout_agent_mcp_card)
        message = create_text_message_object(content=prompt)
        async for response in client.send_message(message):
            task = response[0]
            return task.artifacts[0].parts[0].root.text
        return "No response received."


async def demonstrate_mcp_and_a2a_composition() -> None:
    """
    Section 6.5.6 — Step 3: Diet Coach uses MCP and A2A simultaneously.

    In the same workflow:
      • MCP call (own tools)  : Diet Coach queries its nutrition database.
      • A2A call (remote peer): Diet Coach delegates to the Workout Agent,
                                which internally uses its own MCP fitness tools.

    The two protocols are independent. The Diet Coach does not know or care
    that the Workout Agent is also using MCP — that detail is encapsulated
    behind the A2A interface.
    """
    print("\n── Section 6.5.6: MCP + A2A Composition ─────────────────────────")
    print("Two-layer architecture: MCP for tools, A2A for agent peers\n")

    # ── Step 3a: Diet Coach's own MCP tool call (nutrition database) ──────
    # In production: nutrition_tools = build_nutrition_mcp_tools()
    # Here we simulate the MCP result to avoid requiring a live server URL.
    print("[MCP → Nutrition DB] Querying salmon nutritional data...")
    nutrition_result = {
        "food":     "Atlantic salmon (100g)",
        "calories": 208,
        "protein":  "28g",
        "fat":      "12g",
        "carbs":    "0g",
    }
    print(f"  Result: {nutrition_result}\n")

    # ── Step 3b: A2A call to the MCP-equipped Workout Agent ───────────────
    # The Diet Coach calls over A2A. Inside the Workout Agent, MCP tools
    # query the fitness DB. That internal call is invisible to the Diet Coach.
    print("[A2A → Workout Agent] Requesting 4-week 5k plan...")
    mcp_a2a_tool = McpWorkoutA2ATool()
    workout_plan = await mcp_a2a_tool.ask(
        "Create a beginner-friendly 4-week 5k running plan. "
        "Use fitness database norms for weekly mileage if available."
    )
    print(f"  Workout Agent replied:\n  {workout_plan[:200]}...\n")

    # ── Step 3c: Diet Coach synthesises MCP + A2A results ─────────────────
    # Both results are now in scope. The Diet Coach combines them into one
    # response — exactly as if both came from the same source.
    print("[Diet Coach] Synthesising nutrition data + workout plan...")
    synthesis = (
        f"Your {nutrition_result['food']} provides "
        f"{nutrition_result['calories']} kcal and "
        f"{nutrition_result['protein']} protein — ideal post-run recovery. "
        f"Here is your Week 1 plan: {workout_plan[:120]}..."
    )
    print(f"\n  '{synthesis}'\n")

    # ── Protocol summary ──────────────────────────────────────────────────
    print("── Protocol breakdown ───────────────────────────────────────────")
    print("  MCP call : Diet Coach → Nutrition DB  (tool layer — no agency)")
    print("  A2A call : Diet Coach → Workout Agent (agent layer — reasons + MCP)")
    print("  Key point: Workout Agent's internal MCP is invisible to Diet Coach")
    print("  Result   : both outputs synthesised in a single Diet Coach response")


# ─────────────────────────────────────────────────────────────────────────────
# Demo entry point — runs all three demos in sequence
# ─────────────────────────────────────────────────────────────────────────────

def run_all_demos() -> None:
    """
    Run the full chapter demo sequence:
      1. HITL gate — all three lanes (Section 6.2.3)
      2. Basic A2A — CrewAI Diet Coach calls ADK Workout Agent (Section 6.3)
      3. MCP + A2A composition (Section 6.6)
      4. Event-driven pattern sketch (Section 6.7.6)

    Run individual demos separately if preferred:
      asyncio.run(demonstrate_hitl_gate_async())
      asyncio.run(crewai_agent_calls_adk())
      asyncio.run(demonstrate_mcp_and_a2a_composition())
      asyncio.run(demonstrate_event_driven_pattern())
    """
    # Demo 1 — HITL gate (Section 6.2.3, no server needed)
    print("── Demo 1: HITL Gate — graduated autonomy ────────────────────────")
    demonstrate_hitl_gate()

    # Demo 2 — Basic A2A cross-framework (Section 6.3, port 10020)
    print("\n── Demo 2: Basic A2A cross-framework demo ────────────────────────")
    threading.Thread(target=start_server_in_background, daemon=True).start()
    time.sleep(2)
    asyncio.run(crewai_agent_calls_adk())

    # Demo 3 — MCP + A2A composition (Section 6.6, port 10021)
    print("\n── Demo 3: MCP + A2A composition ────────────────────────────────")
    threading.Thread(target=start_mcp_server_in_background, daemon=True).start()
    time.sleep(2)
    asyncio.run(demonstrate_mcp_and_a2a_composition())

    # Demo 4 — Event-driven pattern (Section 6.7.6, no server needed)
    print("\n── Demo 4: Sync vs Event-Driven Communication ───────────────────")
    asyncio.run(demonstrate_event_driven_pattern())
