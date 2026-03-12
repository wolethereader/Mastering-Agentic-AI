"""
Mastering Agentic AI — Chapter 6
Communication Protocols

Sections covered:
  6.1  Fundamentals of Agent-to-Agent (A2A) Protocols
  6.2  Human-to-Agent (H2A) and Agent-to-Human (A2H) Protocols
  6.3  Agent Communication Protocol (ACP)
  6.4  Coordination and Negotiation Between Agents
  6.5  Interoperability Standards Across Frameworks

As soon as you have more than one agent, you need
a protocol. Without it, agents talk past each other. With it, they compose.

Running example: the AI Diet Coach becomes a MULTI-AGENT SYSTEM.
  • OrchestratorAgent   — routes user requests, synthesises outputs
  • NutritionAnalyst    — deep nutrition analysis
  • MealPlannerAgent    — practical meal planning
  • BehaviourCoach      — habit formation and accountability
"""

import json
import asyncio
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any
import anthropic

# ─────────────────────────────────────────────────────────────────────────────
# 6.1  A2A Protocol foundations — message envelope
# ─────────────────────────────────────────────────────────────────────────────

class MessageType(str, Enum):
    REQUEST    = "request"
    RESPONSE   = "response"
    DELEGATE   = "delegate"
    RESULT     = "result"
    ERROR      = "error"


@dataclass
class AgentMessage:
    """
    Standard message envelope for agent-to-agent communication.
    Inspired by FIPA ACL and the Google A2A draft spec.
    """
    sender:      str
    recipient:   str
    msg_type:    MessageType
    content:     Any
    message_id:  str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    reply_to:    str | None = None
    metadata:    dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["msg_type"] = self.msg_type.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "AgentMessage":
        d["msg_type"] = MessageType(d["msg_type"])
        return cls(**d)


class MessageBus:
    """
    Section 6.1: Simple in-process message bus.
    Production replacement: Redis pub/sub, RabbitMQ, or Google Cloud Pub/Sub.
    """

    def __init__(self):
        self._queues: dict[str, list[AgentMessage]] = {}

    def register(self, agent_id: str) -> None:
        self._queues.setdefault(agent_id, [])

    def send(self, message: AgentMessage) -> None:
        self._queues.setdefault(message.recipient, []).append(message)

    def receive(self, agent_id: str) -> list[AgentMessage]:
        msgs = self._queues.get(agent_id, [])
        self._queues[agent_id] = []
        return msgs


# ─────────────────────────────────────────────────────────────────────────────
# 6.2  H2A / A2H Protocols — structured handshake
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class H2ARequest:
    """Human-to-Agent request with explicit capability declaration."""
    user_id:        str
    message:        str
    capabilities_needed: list[str]  # e.g. ["nutrition_analysis", "meal_planning"]
    context:        dict = field(default_factory=dict)


@dataclass
class A2HResponse:
    """Agent-to-Human response with confidence and source attribution."""
    message:    str
    confidence: float           # 0.0 – 1.0
    sources:    list[str]       # which sub-agents contributed
    follow_up:  str | None = None


# ─────────────────────────────────────────────────────────────────────────────
# 6.3  Agent Communication Protocol (ACP) — capability registry
# ─────────────────────────────────────────────────────────────────────────────

class AgentRegistry:
    """
    Section 6.3: A capability registry so the orchestrator can discover
    which agents handle which tasks at runtime.
    """

    def __init__(self):
        self._registry: dict[str, dict] = {}

    def register(self, agent_id: str, capabilities: list[str], description: str) -> None:
        self._registry[agent_id] = {"capabilities": capabilities, "description": description}

    def find(self, capability: str) -> list[str]:
        return [
            agent_id
            for agent_id, info in self._registry.items()
            if capability in info["capabilities"]
        ]

    def all_capabilities(self) -> list[str]:
        caps: set[str] = set()
        for info in self._registry.values():
            caps.update(info["capabilities"])
        return sorted(caps)


# ─────────────────────────────────────────────────────────────────────────────
# 6.4  Sub-Agents — each with a focused role and prompt
# ─────────────────────────────────────────────────────────────────────────────

class BaseSubAgent:
    def __init__(self, agent_id: str, system_prompt: str, model: str = "claude-opus-4-5"):
        self.agent_id      = agent_id
        self.system_prompt = system_prompt
        self.client        = anthropic.Anthropic()
        self.model         = model

    def process(self, query: str, context: dict | None = None) -> str:
        ctx_str = json.dumps(context, indent=2) if context else ""
        user_msg = f"{query}\n\n[Context]\n{ctx_str}" if ctx_str else query

        response = self.client.messages.create(
            model=self.model,
            max_tokens=512,
            system=self.system_prompt,
            messages=[{"role": "user", "content": user_msg}],
        )
        return response.content[0].text


class NutritionAnalystAgent(BaseSubAgent):
    def __init__(self):
        super().__init__(
            agent_id="nutrition_analyst",
            system_prompt=(
                "You are NutritionAnalyst. Provide precise, evidence-based "
                "macro- and micro-nutrient analysis. Cite NHS/RDA reference values. "
                "Output: Overview, Key Nutrients, Recommendation. Be concise."
            ),
        )


class MealPlannerAgent(BaseSubAgent):
    def __init__(self):
        super().__init__(
            agent_id="meal_planner",
            system_prompt=(
                "You are MealPlanner. Create practical, time-efficient meal plans "
                "tailored to the user's goals and restrictions. "
                "Output: a markdown table with Day, Meal, Key Ingredients, Protein (g)."
            ),
        )


class BehaviourCoachAgent(BaseSubAgent):
    def __init__(self):
        super().__init__(
            agent_id="behaviour_coach",
            system_prompt=(
                "You are BehaviourCoach. Apply behaviour change science (habit loops, "
                "implementation intentions, motivational interviewing) to help users "
                "stick to their nutrition plans. Focus on identity, not willpower."
            ),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator Agent — routes, coordinates, synthesises
# ─────────────────────────────────────────────────────────────────────────────

class OrchestratorAgent:
    """
    Section 6.4: Coordinates the sub-agents using explicit delegation messages.

    Routing strategy:
      1. Ask the LLM to classify the intent → select agents
      2. Send AgentMessages to selected agents
      3. Collect results
      4. Synthesise into a single A2HResponse
    """

    def __init__(self):
        self.client   = anthropic.Anthropic()
        self.bus      = MessageBus()
        self.registry = AgentRegistry()

        self.agents: dict[str, BaseSubAgent] = {
            "nutrition_analyst": NutritionAnalystAgent(),
            "meal_planner":      MealPlannerAgent(),
            "behaviour_coach":   BehaviourCoachAgent(),
        }

        for agent_id, agent in self.agents.items():
            self.bus.register(agent_id)
            self.registry.register(
                agent_id,
                capabilities={"nutrition_analyst": ["nutrition_analysis", "macro_tracking"],
                               "meal_planner":      ["meal_planning", "recipe_suggestion"],
                               "behaviour_coach":   ["habit_formation", "motivation", "accountability"]}[agent_id],
                description=agent.system_prompt[:60],
            )

    def _route(self, user_message: str) -> list[str]:
        """Ask the LLM which sub-agents to invoke."""
        available = ", ".join(self.agents.keys())
        routing_prompt = (
            f"Available agents: {available}\n"
            f"User message: {user_message}\n\n"
            "Which agents should handle this? Return JSON: "
            '{"agents": ["agent_id1", ...], "sub_queries": {"agent_id": "specific question"}}'
        )

        response = self.client.messages.create(
            model="claude-opus-4-5",
            max_tokens=256,
            system="You are a routing orchestrator. Output only valid JSON.",
            messages=[{"role": "user", "content": routing_prompt}],
        )

        raw = response.content[0].text.strip().lstrip("```json").rstrip("```").strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"agents": ["nutrition_analyst"], "sub_queries": {"nutrition_analyst": user_message}}

    def handle(self, request: H2ARequest) -> A2HResponse:
        routing = self._route(request.message)
        selected_agents = routing.get("agents", ["nutrition_analyst"])
        sub_queries     = routing.get("sub_queries", {})

        # Delegate to each selected agent via message bus
        results: dict[str, str] = {}
        for agent_id in selected_agents:
            if agent_id not in self.agents:
                continue
            query = sub_queries.get(agent_id, request.message)
            msg = AgentMessage(
                sender="orchestrator",
                recipient=agent_id,
                msg_type=MessageType.DELEGATE,
                content={"query": query, "context": request.context},
            )
            self.bus.send(msg)

            # Process (synchronous for clarity — async in section 6.5)
            result_text = self.agents[agent_id].process(query, request.context)
            results[agent_id] = result_text

        # Synthesise results
        synthesis_prompt = (
            f"User asked: {request.message}\n\n"
            "Sub-agent results:\n"
            + "\n\n".join(f"[{k}]\n{v}" for k, v in results.items())
            + "\n\nSynthesize into one clear, friendly response for the user."
        )

        synth_response = self.client.messages.create(
            model="claude-opus-4-5",
            max_tokens=512,
            system="You are a synthesising AI Diet Coach. Integrate sub-agent outputs coherently.",
            messages=[{"role": "user", "content": synthesis_prompt}],
        )

        return A2HResponse(
            message=synth_response.content[0].text,
            confidence=0.85,
            sources=list(results.keys()),
            follow_up="Would you like me to elaborate on any of these points?",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    orchestrator = OrchestratorAgent()

    request = H2ARequest(
        user_id="jordan",
        message=(
            "I keep skipping lunch because I'm too busy. "
            "I want high protein meals I can prep in under 10 minutes. "
            "Also I've been snacking on crisps out of stress — help!"
        ),
        capabilities_needed=["nutrition_analysis", "meal_planning", "habit_formation"],
        context={"weight_kg": 78, "goal": "Lose 5 kg", "restriction": "Lactose intolerant"},
    )

    print("── Multi-Agent Diet Coach (Chapter 6) ──────────────────────")
    response = orchestrator.handle(request)
    print(f"\nCoach (synthesised from {response.sources}):\n")
    print(response.message)
    print(f"\nConfidence: {response.confidence:.0%}")
    print(f"Follow-up:  {response.follow_up}")
