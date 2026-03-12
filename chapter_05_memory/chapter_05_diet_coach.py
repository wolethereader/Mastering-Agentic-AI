"""
Mastering Agentic AI — Chapter 5
Memory Systems for Agents

Sections covered:
  5.1  What is Memory and Why It Matters
  5.2  Types of Memory
  5.3  Introduction to RAG and GraphRAG
  5.4  State-of-the-Art Memory Architectures
  5.5  Giving Our AI Diet Coach a Memory

Memory is what turns a stateless model into a
personalised agent. Without memory, the coach asks the same questions
every session. With memory, it tracks progress over weeks.

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
from typing import Any
import anthropic

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
# 5.5  AI Diet Coach with Full Memory
# ─────────────────────────────────────────────────────────────────────────────

SKILL_PATH = Path(__file__).parent.parent / "chapter_03_prompting" / "SKILL.md"


class MemoryEnabledDietCoach:
    """
    Section 5.5: The Diet Coach now remembers you.

    Memory layers active:
      ✓ Semantic   — your profile (weight, goals, restrictions)
      ✓ Episodic   — past session summaries (RAG-retrieved)
      ✓ In-context — this conversation's history
      ✓ Procedural — SKILL.md assessment protocol (Chapter 3)
    """

    def __init__(self, user_id: str, model: str = "claude-opus-4-5"):
        self.client      = anthropic.Anthropic()
        self.model       = model
        self.user_id     = user_id
        self.semantic    = SemanticMemory(user_id)
        self.episodic    = EpisodicMemory(user_id)
        self.in_context  = InContextMemory(max_turns=20)
        self.skill_text  = SKILL_PATH.read_text() if SKILL_PATH.exists() else ""

    def _build_system_prompt(self, query: str) -> str:
        # RAG: retrieve relevant past episodes
        all_episodes   = self.episodic._episodes
        relevant_eps   = naive_keyword_retrieval(query, all_episodes, top_k=2)
        retrieved_text = ""
        if relevant_eps:
            retrieved_text = "\nRelevant past sessions (retrieved):\n"
            for ep in relevant_eps:
                retrieved_text += f"  [{ep['date']}] {ep['summary']}\n"

        return f"""You are an AI Diet Coach with persistent memory.

[Nutrition Assessment Skill]
{self.skill_text}

[User Profile — Semantic Memory]
{self.semantic.as_context_string()}

[Session History — Episodic Memory]
{self.episodic.as_context_string(n=3)}
{retrieved_text}
[Instructions]
- Greet returning users by name and reference their last goal if known.
- When the user mentions a new food or meal, note it.
- At the end of the conversation, summarise key insights for episode storage.
- Never ask for information already in the semantic memory.
"""

    def chat(self, user_message: str) -> str:
        self.in_context.add("user", user_message)

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=self._build_system_prompt(user_message),
            messages=self.in_context.get_window(),
        )

        reply = response.content[0].text
        self.in_context.add("assistant", reply)
        return reply

    def update_profile(self, **kwargs: Any) -> None:
        """Store user facts in semantic memory."""
        for key, value in kwargs.items():
            self.semantic.set(key, value)
        print(f"[memory] Profile updated: {list(kwargs.keys())}")

    def save_session(self, summary: str, insights: list[str], goal: str | None = None) -> None:
        """Persist the session to episodic memory."""
        self.episodic.add_episode(summary, insights, goal)
        print(f"[memory] Session saved for {self.user_id}")


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    coach = MemoryEnabledDietCoach(user_id="jordan_demo")

    # Populate semantic memory (normally gathered over first session)
    coach.update_profile(
        name="Jordan",
        age=32,
        weight_kg=78,
        goal="Lose 5 kg over 3 months",
        restrictions="Lactose intolerant",
        activity_level="Runs 3x per week",
    )

    # Seed an episodic memory entry (simulating a past session)
    coach.episodic.add_episode(
        summary="Jordan discussed breakfast habits and protein intake.",
        key_insights=["Skips breakfast most days", "Protein at lunch only ~20 g"],
        goal_set="Eat 30 g of protein at breakfast every day this week.",
    )

    print("── Memory-Enabled Diet Coach ───────────────────────────────")
    reply1 = coach.chat("Hi, I'm back. I've been struggling to hit my protein goal.")
    print(f"Coach: {reply1}\n")

    reply2 = coach.chat("I had eggs and smoked salmon this morning — felt great.")
    print(f"Coach: {reply2}\n")

    # Save session episode
    coach.save_session(
        summary="Jordan successfully added a high-protein breakfast with eggs and salmon.",
        insights=["Responded well to concrete meal suggestions", "Morning routine is the highest-leverage window"],
        goal="Continue 30 g protein at breakfast and add a mid-afternoon snack.",
    )
