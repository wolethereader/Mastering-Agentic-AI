"""
Mastering Agentic AI — Chapter 1
The AI Diet Coach: A Simple Conversational Assistant

We start simple, understand the loop,
then layer complexity in every chapter that follows.

At this stage the coach has NO tools, NO memory, NO multi-agent
orchestration. It is a plain perceive→reason→respond loop powered
by an LLM. That is intentional. You cannot understand what agents
add until you feel what is missing without them.
"""

import os
from openai import OpenAI

# ── Constants ────────────────────────────────────────────────────────────────
MODEL = "gpt-4.1-nano"
MAX_TOKENS = 1024
HISTORY_LIMIT = 20                  # keep last N turns in context window

SYSTEM_PROMPT = """You are an AI Diet Coach — a knowledgeable, encouraging
nutrition assistant. You help users understand their dietary habits, set
realistic goals, and make sustainable food choices.

You follow the Perceive → Reason → Respond loop described in Chapter 1:
  • Perceive  : read the user's message plus conversation history
  • Reason    : think about nutritional principles, the user's stated goals,
                and any constraints they have mentioned
  • Respond   : give one clear, actionable, friendly reply

Rules for this chapter (Chapter 1 — no tools yet):
  - You work from your training knowledge only; you cannot look up live data.
  - You cannot remember anything across separate sessions.
  - Never fabricate specific calorie counts for branded products.
  - Always recommend consulting a registered dietitian for medical nutrition therapy.
"""

# ── Core agent loop ───────────────────────────────────────────────────────────

def run_diet_coach() -> None:
    """
    The simplest possible agentic loop.

    Chapter 1 insight: even a single-turn LLM call is a degenerate agent —
    perceive (user prompt) → reason (LLM forward pass) → act (print reply).
    Multi-turn conversation is the first step toward real agency.
    """
    client = OpenAI()
    history: list[dict] = []

    print("=" * 60)
    print("  AI Diet Coach  ·  Chapter 1  ·  Mastering Agentic AI")
    print("=" * 60)
    print("Hi! I'm your AI Diet Coach. Tell me about your goals,")
    print("what you ate today, or any nutrition question you have.")
    print("Type 'quit' to exit.\n")

    while True:
        # ── Perceive ──────────────────────────────────────────────────
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "bye"):
            print("Coach: Great talking with you — stay consistent! 👋")
            break

        history.append({"role": "user", "content": user_input})

        # Trim history to avoid blowing the context window
        trimmed = history[-HISTORY_LIMIT:]

        # ── Reason + Respond ──────────────────────────────────────────
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + trimmed,
        )

        reply = response.choices[0].message.content
        history.append({"role": "assistant", "content": reply})

        print(f"\nCoach: {reply}\n")

        # ── Observe (chapter 1 version: just print stop reason) ───────
        if response.choices[0].finish_reason != "stop":
            print(f"[debug] finish_reason={response.choices[0].finish_reason}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_diet_coach()
