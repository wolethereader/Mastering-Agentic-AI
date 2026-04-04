"""
Mastering Agentic AI — Chapter 5
Memory Systems for Agents

Sections covered:
  5.1  What is Memory and Why It Matters
  5.2  Types of Memory
  5.3  Introduction to RAG and GraphRAG
  5.4  State-of-the-Art Memory Architectures
  5.5  Giving Our AI Diet Coach a Memory
  5.6  Context Engineering

Memory is what turns a stateless model into a
personalised agent. Without memory, the coach asks the same questions
every session. With memory, it tracks progress over weeks.

Context engineering — Section 5.6 — is the complementary discipline:
memory stores answer WHAT the agent knows; context engineering answers
WHAT the model sees right now, in what order, and at what level of detail.

Memory taxonomy used throughout this chapter:
  • In-context     — conversation history in the current context window
  • Episodic       — past session summaries retrieved from a store
  • Semantic       — facts about the user (profile, preferences)
  • Procedural     — skills and workflows (SKILL.md from Chapter 3)
"""

import json
import datetime
import hashlib
from pathlib import Path
from typing import Any, List, Dict
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# 5.2  Types of Memory — concrete implementations
# ─────────────────────────────────────────────────────────────────────────────

class SemanticMemory:
    """
    Stores structured facts about a user that persist across sessions.
    In production: use a key-value store (Redis) or document DB.
    For this chapter: JSON file on disk.
    """

    def __init__(self, user_id: str, store_path: Path = Path(".memory")):
        self.user_id = user_id
        self.path = store_path / f"{user_id}_semantic.json"
        self.path.parent.mkdir(exist_ok=True)
        self._data: dict = json.loads(self.path.read_text()) if self.path.exists() else {}

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value
        self.path.write_text(json.dumps(self._data, indent=2))

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def all(self) -> dict:
        return dict(self._data)

    def as_context_string(self) -> str:
        if not self._data:
            return "No user profile stored yet."
        lines = [f"  {k}: {v}" for k, v in self._data.items()]
        return "User profile (from semantic memory):\n" + "\n".join(lines)


class EpisodicMemory:
    """
    Stores summaries of past coaching sessions.
    Each episode: {date, summary, key_insights, goal_set}.
    In production: embed and index with FAISS or Pinecone for RAG retrieval.
    """

    def __init__(self, user_id: str, store_path: Path = Path(".memory")):
        self.user_id = user_id
        self.path = store_path / f"{user_id}_episodes.json"
        self.path.parent.mkdir(exist_ok=True)
        self._episodes: list[dict] = json.loads(self.path.read_text()) if self.path.exists() else []

    def add_episode(self, summary: str, key_insights: list[str], goal_set: str | None = None) -> None:
        episode = {
            "date": datetime.date.today().isoformat(),
            "summary": summary,
            "key_insights": key_insights,
            "goal_set": goal_set,
            "id": hashlib.md5(f"{self.user_id}{datetime.datetime.now()}".encode()).hexdigest()[:8],
        }
        self._episodes.append(episode)
        self.path.write_text(json.dumps(self._episodes, indent=2))

    def get_recent(self, n: int = 3) -> list[dict]:
        return self._episodes[-n:]

    def as_context_string(self, n: int = 3) -> str:
        recent = self.get_recent(n)
        if not recent:
            return "No past sessions recorded."
        lines = []
        for ep in recent:
            lines.append(f"  [{ep['date']}] {ep['summary']}")
            if ep.get("goal_set"):
                lines.append(f"    Goal set: {ep['goal_set']}")
        return "Recent session history:\n" + "\n".join(lines)


class InContextMemory:
    """
    Manages the current conversation history with sliding window pruning.
    Ensures we never exceed the model's context window.
    """

    def __init__(self, max_turns: int = 20):
        self.max_turns = max_turns
        self._history: list[dict] = []

    def add(self, role: str, content: Any) -> None:
        self._history.append({"role": role, "content": content})

    def get_window(self) -> list[dict]:
        return self._history[-self.max_turns * 2:]  # 2 entries per turn

    def __len__(self) -> int:
        return len(self._history)


# ─────────────────────────────────────────────────────────────────────────────
# 5.3  RAG — retrieve relevant episodes using simple cosine similarity
#          (production: use embeddings + vector DB)
# ─────────────────────────────────────────────────────────────────────────────

def naive_keyword_retrieval(query: str, episodes: list[dict], top_k: int = 2) -> list[dict]:
    """
    Section 5.3: Simplified RAG retrieval using keyword overlap.
    Replace with embedding-based retrieval (OpenAI/Cohere embeddings + FAISS)
    in a production system.
    """
    query_words = set(query.lower().split())
    scored = []
    for ep in episodes:
        text = f"{ep['summary']} {' '.join(ep.get('key_insights', []))}"
        doc_words = set(text.lower().split())
        overlap = len(query_words & doc_words)
        scored.append((overlap, ep))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [ep for _, ep in scored[:top_k] if _ > 0]


# ─────────────────────────────────────────────────────────────────────────────
# 5.5  Memory for our Diet Coach (LangChain + Mem0)
# ─────────────────────────────────────────────────────────────────────────────

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from mem0 import MemoryClient

# Step 1: Initialise the model and memory client
llm = ChatOpenAI(model="gpt-4.1-nano")
memory_client = MemoryClient()

# Step 2: Prompt with a reserved slot for memory-derived context
prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="""
You are a helpful AI Diet Coach.
Use the provided context to remember user preferences, dietary restrictions,
and long-term goals. Base your recommendations on this information.
"""),
    MessagesPlaceholder(variable_name="context"),
    HumanMessage(content="{input}")
])


def retrieve_context(query: str, user_id: str) -> List[Dict]:
    """Step 3: Retrieve relevant memories for the current query."""
    memories = memory_client.search(query, user_id=user_id)
    if not memories.get("results"):
        return [{"role": "user", "content": query}]
    serialized = " ".join(memory["memory"] for memory in memories["results"])
    return [
        {"role": "system", "content": f"Relevant user information: {serialized}"},
        {"role": "user", "content": query}
    ]


def generate_response(user_input: str, context: List[Dict]) -> str:
    """Step 4: Pass assembled context to the LLM."""
    chain = prompt | llm
    response = chain.invoke({
        "context": context,
        "input": user_input
    })
    return response.content


def save_interaction(user_id: str, user_input: str, assistant_response: str):
    """Step 5: Persist the interaction back into Mem0."""
    interaction = [
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": assistant_response}
    ]
    memory_client.add(interaction, user_id=user_id)


def chat_turn(user_input: str, user_id: str) -> str:
    """Step 6: A single conversation turn — retrieve, generate, persist."""
    context = retrieve_context(user_input, user_id)
    response = generate_response(user_input, context)
    save_interaction(user_id, user_input, response)
    return response


# ─────────────────────────────────────────────────────────────────────────────
# 5.6  Context Engineering
#
# Memory stores — semantic, episodic, in-context — answer WHAT the agent
# knows. Context engineering answers a different question: WHAT does the
# model see right now, in what order, and at what level of detail?
#
# Every token in the context window is a decision. The function below
# assembles that window deliberately:
#
#   Layer 0: Skill protocol    — injected via system prompt (not here)
#   Layer 1: User profile      — stable facts; injected first as background
#   Layer 2: Recent meals      — dynamic, recency-weighted; last 3 only
#   Layer 3: Current message   — always last; highest attention weight
#
# The fake assistant acknowledgement turns ("Understood — I have your
# profile loaded.") are a priming technique: they condition the model to
# treat the injected context as already processed rather than summarising
# it back to the user.
# ─────────────────────────────────────────────────────────────────────────────

# Load the Skill text once at module level so build_context_window can
# reference it as a default argument — consistent with how MemoryEnabledDietCoach
# loads it via SKILL_PATH.
SKILL_PATH = Path(__file__).parent.parent / "chapter_03_prompting" / "SKILL.md"

NUTRITION_ASSESSMENT_SKILL: str = (
    SKILL_PATH.read_text() if SKILL_PATH.exists() else ""
)


def build_context_window(
    user_message: str,
    meal_history: list[dict] | None = None,
    user_profile: dict | None = None,
    skill_text: str = NUTRITION_ASSESSMENT_SKILL,
) -> list[dict]:
    """
    Section 5.6: Assemble the message list the model will see.

    Treats every token as a deliberate decision rather than passing
    everything into the context and hoping the model pays attention
    to the right things.

    Args:
        user_message:  The user's current input — always the final message.
        meal_history:  Full meal log; only the last 3 entries are included.
        user_profile:  Dict of user facts (weight, goal, restrictions, etc.).
        skill_text:    The Nutrition Assessment Skill protocol from SKILL.md.
                       Defaults to the module-level constant; override for
                       testing without hitting disk.

    Returns:
        A list of message dicts in OpenAI API format, ordered so the
        model's attention naturally falls on the most relevant context.
    """
    messages: list[dict] = []

    # Layer 1 — User profile (stable background context)
    # Injected first because it changes rarely and provides the frame
    # for everything that follows. The synthetic assistant reply primes
    # the model to treat this as already-acknowledged context rather
    # than something it needs to summarise or repeat back.
    if user_profile:
        profile_note = f"[Context] User profile:\n{json.dumps(user_profile, indent=2)}"
        messages.append({"role": "user",      "content": profile_note})
        messages.append({"role": "assistant", "content": "Understood — I have your profile loaded."})

    # Layer 2 — Recent meals (dynamic, recency-weighted)
    # Limit to the last 3 entries: older entries are noise, not signal.
    # Sending the full meal log would consume tokens without improving
    # the model's reasoning — the "lost in the middle" problem means
    # information buried deep in long contexts receives less attention.
    if meal_history:
        recent = meal_history[-3:]
        history_note = (
            "Recent meals logged:\n"
            + "\n".join(
                f"  - {m.get('food', '?')} ({m.get('meal_type', '?')})"
                for m in recent
            )
        )
        messages.append({"role": "user",      "content": history_note})
        messages.append({"role": "assistant", "content": "Got it — I'll factor in your recent meals."})

    # Layer 3 — Current user message (always last, always highest weight)
    # Recency bias in attention means this placement matters. The model
    # weighs recent tokens more heavily; the user's actual question
    # should be the last thing it sees before generating a response.
    messages.append({"role": "user", "content": user_message})
    return messages


def demonstrate_context_engineering() -> None:
    """
    Section 5.6: Show build_context_window in action,
    demonstrating how memory stores feed into context assembly.
    """
    user_profile = {
        "name":           "Jordan",
        "age":            32,
        "weight_kg":      78,
        "goal":           "Lose 5 kg over 3 months",
        "restrictions":   "Lactose intolerant",
        "activity_level": "Runs 3x per week",
    }

    meal_history = [
        {"food": "oats with banana",       "meal_type": "breakfast"},
        {"food": "chicken salad",          "meal_type": "lunch"},
        {"food": "salmon with brown rice", "meal_type": "dinner"},
        {"food": "apple",                  "meal_type": "snack"},   # dropped: > last 3
        {"food": "eggs and smoked salmon", "meal_type": "breakfast"},
    ]

    messages = build_context_window(
        user_message="How am I doing on protein today?",
        meal_history=meal_history,
        user_profile=user_profile,
    )

    print("── Context window assembled (" + str(len(messages)) + " messages) ──")
    for i, msg in enumerate(messages):
        content_preview = (
            msg["content"][:80] + "..."
            if len(msg["content"]) > 80
            else msg["content"]
        )
        print(f"  [{i}] {msg['role']:10s} | {content_preview}")
    print()
    print("Layer breakdown:")
    print("  Messages 0–1 : Layer 1 — user profile (stable background)")
    print("  Messages 2–3 : Layer 2 — last 3 meals (oats, chicken salad, eggs)")
    print("  Message  4   : Layer 3 — current question (highest attention weight)")
    print()
    print("Note: the apple snack was excluded — it fell outside the last-3 window.")




# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    USER_ID = "jordan_demo"

    # ── Section 5.5: Diet Coach with Mem0 memory ────────────────────────────
    print("── Memory-Enabled Diet Coach (Mem0) ────────────────────────")
    reply1 = chat_turn("I'm on a cut. High protein, low carbs.", USER_ID)
    print(f"Coach: {reply1}\n")

    reply2 = chat_turn("What should I have for dinner?", USER_ID)
    print(f"Coach: {reply2}\n")

    # ── Section 5.6: Context Engineering ────────────────────────────────────
    print("\n── Section 5.6: Context Engineering ───────────────────────")
    demonstrate_context_engineering()
