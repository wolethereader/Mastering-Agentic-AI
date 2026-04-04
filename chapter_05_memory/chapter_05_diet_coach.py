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

New additions (Sections 5.2a, 5.4a, 5.6a):
  • Compaction     — compress history rather than just truncating it
  • Writable skills — procedural memory the agent can update from experience
  • Ephemeral ctx  — context assembled fresh each turn, never stored
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
# 5.2a  Compaction — When the Window Fills
#
# Sliding the window drops old turns but loses their information entirely.
# Compaction is an alternative: compress the history before it overflows,
# preserving meaning while reducing token count.
#
# Three patterns cover most cases:
#
#   1. Result-only  — keep outcomes, discard working details.
#      Coding agent: drop the compiler log, keep "compile successful".
#      Research agent: drop training curves, keep the validation loss.
#
#   2. LLM summarisation — when results cannot be reduced to a single value,
#      use a separate model call to compress the history into a few sentences.
#
#   3. Ephemeral context — some information is critical now but useless
#      next turn. Append it to the context for this iteration only and
#      never write it into history. See Section 5.6a for the full pattern.
#
# The Diet Coach already uses compaction implicitly:
#   • EpisodicMemory stores distilled summaries, not raw transcripts.
#   • build_context_window includes only the last 3 meals, not the full log.
# These are result-only compaction decisions — this section names the pattern.
# ─────────────────────────────────────────────────────────────────────────────

def compact_history_result_only(
    history: list[dict],
    result_roles: tuple[str, ...] = ("assistant",),
) -> list[dict]:
    """
    Section 5.2a: Result-only compaction — keep the outcome of each
    exchange and strip intermediate working detail.

    In this simplified version, every assistant turn is kept (it is the
    "result") while any tool/function intermediate turns are discarded.
    In a real coding or research agent, you would add logic to strip
    verbose logs from within assistant content too.

    Args:
        history:      Full turn history as a list of role/content dicts.
        result_roles: Roles considered "results" worth keeping.

    Returns:
        Compacted history containing only result-role turns.
    """
    return [msg for msg in history if msg.get("role") in result_roles]


def compact_history_llm_summary(
    history: list[dict],
    client: OpenAI,
    model: str = "gpt-4o-mini",
) -> list[dict]:
    """
    Section 5.2a: LLM-summarisation compaction — compress the full history
    into a single system message when the context is too long to truncate
    cleanly.

    Use this when result-only compaction would lose important reasoning
    steps, or when the history spans many heterogeneous topic shifts.

    Args:
        history: Full turn history to compress.
        client:  OpenAI client (already initialised).
        model:   Model to use for summarisation.

    Returns:
        A single-element list containing a system message with the summary.
    """
    if not history:
        return []

    transcript = "\n".join(
        f"{m['role'].upper()}: {str(m['content'])[:300]}"
        for m in history
    )
    response = client.chat.completions.create(
        model=model,
        max_tokens=200,
        messages=[{
            "role": "user",
            "content": (
                "Summarise this conversation in 2–3 sentences. "
                "Retain only facts needed to continue the coaching session:\n\n"
                + transcript
            ),
        }],
    )
    summary = response.choices[0].message.content.strip()
    return [{"role": "system", "content": f"[Compacted history] {summary}"}]


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
# 5.4a  Writable Skills — Procedural Memory the Agent Can Update
#
# Procedural memory (SKILL.md) is the one memory type the agent can rewrite.
# Semantic, episodic, and in-context memory are all written by the surrounding
# system — they record what happened. Procedural memory records HOW the agent
# should think — and that is something the agent can update from experience.
#
# A skill is just a text file. If the agent observes that a different approach
# consistently produces better outcomes, it can write a revised instruction
# into SKILL.md. The change propagates to every future session with no
# fine-tuning, no gradient updates, and no redeployment. Just a file write.
#
# This is the foundation for self-evolving agents (see Chapter 9).
# ─────────────────────────────────────────────────────────────────────────────

def read_skill(skill_path: Path) -> str:
    """
    Section 5.4a: Load the current procedural skill from disk.
    Returns an empty string if the file does not exist yet.
    """
    return skill_path.read_text(encoding="utf-8") if skill_path.exists() else ""


def update_skill(
    skill_path: Path,
    new_content: str,
    backup: bool = True,
) -> None:
    """
    Section 5.4a: Overwrite the skill file with updated content.

    The agent calls this after a session to refine its own reasoning
    protocol based on what worked. If backup=True, the previous version
    is saved to SKILL.md.bak before overwriting, enabling rollback.

    Args:
        skill_path:  Path to the SKILL.md file.
        new_content: The full updated skill text.
        backup:      Whether to save the previous version first.
    """
    skill_path.parent.mkdir(parents=True, exist_ok=True)
    if backup and skill_path.exists():
        bak = skill_path.with_suffix(".md.bak")
        bak.write_text(skill_path.read_text(encoding="utf-8"), encoding="utf-8")
    skill_path.write_text(new_content, encoding="utf-8")
    print(f"[skill] Updated: {skill_path}")


def append_skill_note(skill_path: Path, note: str) -> None:
    """
    Section 5.4a: Append a short observation to the existing skill rather
    than replacing it. Useful for adding a learned heuristic without
    rewriting the whole protocol.

    Example:
        append_skill_note(SKILL_PATH,
            "Observation: users respond better when activity level is "
            "established before discussing dietary deficits.")
    """
    current = read_skill(skill_path)
    updated = current.rstrip() + f"\n\n# Agent observation\n{note.strip()}\n"
    update_skill(skill_path, updated, backup=True)
    print(f"[skill] Note appended to {skill_path.name}")


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


# ─────────────────────────────────────────────────────────────────────────────
# 5.6a  Ephemeral Context — Use It Once, Discard It
#
# build_context_window already uses ephemeral context implicitly: meal
# history and the user profile are assembled fresh on every call and are
# never written into the in-context window. This section names and
# generalises the pattern.
#
# Ephemeral context is information that is:
#   • Critical for the current generation turn
#   • Not useful in subsequent turns (stale, changes rapidly, or one-off)
#
# It appears in the context window for one iteration, influences generation,
# and is NOT carried forward into token history.
#
# Contrast with:
#   • In-context memory  — persists turn-to-turn within a session
#   • Semantic memory    — persists across sessions (user profile)
#   • Episodic memory    — persists across sessions (session summaries)
#
# Decision rule: if the data changes faster than the conversation does,
# make it ephemeral. Meal history today ≠ meal history last Tuesday.
# The user's name does not change — it belongs in semantic memory.
# ─────────────────────────────────────────────────────────────────────────────

def build_context_window_with_ephemeral(
    user_message: str,
    in_context_window: list[dict],
    meal_history: list[dict] | None = None,
    user_profile: dict | None = None,
    situational_data: dict | None = None,
    skill_text: str = NUTRITION_ASSESSMENT_SKILL,
) -> list[dict]:
    """
    Section 5.6a: Extended version of build_context_window that makes
    the ephemeral/stored distinction explicit.

    Ephemeral layers (assembled fresh, never stored in in_context_window):
        • user_profile     — re-fetched from SemanticMemory each call
        • meal_history[-3] — recency-weighted, stale by next turn
        • situational_data — any real-time signal (location, time, device)

    Stored layer (passed in from InContextMemory.get_window()):
        • in_context_window — the actual conversation turns

    The caller is responsible for appending only the assistant reply to
    in_context_window after generation — NOT the ephemeral layers.

    Args:
        user_message:      Current user input — always the final message.
        in_context_window: Recent conversation turns from InContextMemory.
        meal_history:      Full meal log; last 3 entries used (ephemeral).
        user_profile:      User facts from SemanticMemory.all() (ephemeral).
        situational_data:  Any real-time context dict, e.g. current time,
                           location, or device state (ephemeral).
        skill_text:        Procedural skill from SKILL.md.

    Returns:
        Full message list for the API call — ephemeral layers + stored window
        + current query, in attention-optimised order.
    """
    messages: list[dict] = []

    # ── Ephemeral Layer 1: stable user facts (re-fetched each call) ──────────
    if user_profile:
        profile_note = f"[Context] User profile:\n{json.dumps(user_profile, indent=2)}"
        messages.append({"role": "user",      "content": profile_note})
        messages.append({"role": "assistant", "content": "Understood — I have your profile loaded."})

    # ── Ephemeral Layer 2: recent meals (last 3, re-assembled each call) ─────
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

    # ── Ephemeral Layer 3: situational data (real-time signal, one-off) ──────
    if situational_data:
        situation_note = (
            "[Situational context — valid for this turn only]\n"
            + json.dumps(situational_data, indent=2)
        )
        messages.append({"role": "user",      "content": situation_note})
        messages.append({"role": "assistant", "content": "Noted — I'll use this for my response."})

    # ── Stored layer: actual conversation turns from InContextMemory ──────────
    # These ARE stored in in_context_window between turns — unlike the above.
    messages.extend(in_context_window)

    # ── Current query: always last, highest attention weight ──────────────────
    messages.append({"role": "user", "content": user_message})
    return messages


def demonstrate_ephemeral_context() -> None:
    """
    Section 5.6a: Show the ephemeral/stored distinction in practice.

    Runs two turns of a conversation. After each turn, only the assistant
    reply is added to the in-context window — the ephemeral layers (profile,
    meals, situational data) are re-assembled fresh on every call.
    """
    from openai import OpenAI

    client = OpenAI()
    in_context: list[dict] = []   # stored — grows turn by turn

    user_profile = {"name": "Jordan", "goal": "Lose 5 kg", "restrictions": "Lactose intolerant"}
    meal_history = [
        {"food": "oats",   "meal_type": "breakfast"},
        {"food": "salmon", "meal_type": "dinner"},
    ]

    for turn, user_message in enumerate([
        "How am I doing on protein today?",
        "What should I eat tomorrow morning?",
    ], start=1):
        situational_data = {"current_time": "19:30", "day_of_week": "Tuesday"}

        # Build context — ephemeral layers assembled fresh every call
        messages = build_context_window_with_ephemeral(
            user_message=user_message,
            in_context_window=in_context,
            meal_history=meal_history,
            user_profile=user_profile,
            situational_data=situational_data,
        )

        system_msg = {"role": "system", "content": "You are an AI Diet Coach."}
        response = client.chat.completions.create(
            model="gpt-4o-mini", max_tokens=150,
            messages=[system_msg, *messages],
        )
        reply = response.choices[0].message.content.strip()

        # Only the user message and reply enter in_context — NOT ephemeral layers
        in_context.append({"role": "user",      "content": user_message})
        in_context.append({"role": "assistant", "content": reply})

        print(f"Turn {turn}")
        print(f"  User:  {user_message}")
        print(f"  Coach: {reply[:120]}...")
        print(f"  in_context turns stored: {len(in_context) // 2}")
        print(f"  Ephemeral layers used this turn: profile + meals + situational")
        print()



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

    # ── Section 5.2a: Compaction ─────────────────────────────────────────────
    print("\n── Section 5.2a: Compaction ────────────────────────────────")
    sample_history = [
        {"role": "user",      "content": "What should I eat for breakfast?"},
        {"role": "assistant", "content": "Try eggs and smoked salmon — high protein."},
        {"role": "user",      "content": "What about lunch?"},
        {"role": "assistant", "content": "A chicken salad would keep you on track."},
    ]
    compacted = compact_history_result_only(sample_history)
    print(f"Original turns: {len(sample_history)}, After result-only compaction: {len(compacted)}")
    for msg in compacted:
        print(f"  [{msg['role']}] {msg['content'][:60]}")

    # ── Section 5.4a: Writable Skills ────────────────────────────────────────
    print("\n── Section 5.4a: Writable Skills ───────────────────────────")
    skill_path = Path(".memory/demo_SKILL.md")
    skill_path.parent.mkdir(exist_ok=True)
    skill_path.write_text("# Demo Skill\nStep 1: Establish baseline.\nStep 2: Identify deficits.\n")
    print(f"Before: {skill_path.read_text().strip()}")
    append_skill_note(skill_path, "Users respond better when activity level is asked first.")
    print(f"After:  {skill_path.read_text().strip()}")

    # ── Section 5.6: Context Engineering ────────────────────────────────────
    print("\n── Section 5.6: Context Engineering ───────────────────────")
    demonstrate_context_engineering()

    # ── Section 5.6a: Ephemeral Context ─────────────────────────────────────
    print("\n── Section 5.6a: Ephemeral Context ─────────────────────────")
    demonstrate_ephemeral_context()
