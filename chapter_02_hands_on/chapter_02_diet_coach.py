"""
Mastering Agentic AI — Chapter 2
Hands-On Introduction to Building Agents

Sections covered:
  2.1  Agentic Workflows and CrewAI
  2.2  Building Our First Agentic Workflow
  2.3  Rebuilding the Same Use Case Without a Framework
  2.4  Boilerplate Builders: Reducing Repetition


Running example: AI Diet Coach gains its FIRST TOOLS in this chapter.
We give it access to a (mock) Nutrition Database and a Meal Logger.
"""

# ── pip install crewai anthropic ──────────────────────────────────────────────

import json
import datetime
from typing import Any

# ─────────────────────────────────────────────────────────────────────────────
# 2.1 & 2.2  CrewAI version
# ─────────────────────────────────────────────────────────────────────────────
try:
    from crewai import Agent, Task, Crew, Process
    from crewai.tools import tool
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False
    print("[warning] crewai not installed — skipping section 2.2")


# ── Mock Nutrition Database ───────────────────────────────────────────────────
NUTRITION_DB: dict[str, dict] = {
    "apple":          {"calories": 95,  "protein_g": 0.5,  "carbs_g": 25, "fat_g": 0.3, "fibre_g": 4.4},
    "chicken breast": {"calories": 165, "protein_g": 31.0, "carbs_g": 0,  "fat_g": 3.6, "fibre_g": 0},
    "brown rice":     {"calories": 216, "protein_g": 5.0,  "carbs_g": 45, "fat_g": 1.8, "fibre_g": 3.5},
    "broccoli":       {"calories": 55,  "protein_g": 3.7,  "carbs_g": 11, "fat_g": 0.6, "fibre_g": 5.1},
    "greek yoghurt":  {"calories": 100, "protein_g": 17.0, "carbs_g": 6,  "fat_g": 0.7, "fibre_g": 0},
    "oats":           {"calories": 150, "protein_g": 5.0,  "carbs_g": 27, "fat_g": 2.5, "fibre_g": 4.0},
    "banana":         {"calories": 105, "protein_g": 1.3,  "carbs_g": 27, "fat_g": 0.4, "fibre_g": 3.1},
    "salmon":         {"calories": 208, "protein_g": 28.0, "carbs_g": 0,  "fat_g": 10,  "fibre_g": 0},
    "almonds":        {"calories": 164, "protein_g": 6.0,  "carbs_g": 6,  "fat_g": 14,  "fibre_g": 3.5},
    "spinach":        {"calories": 23,  "protein_g": 2.9,  "carbs_g": 3.6,"fat_g": 0.4, "fibre_g": 2.2},
}

# Simple in-memory meal log (keyed by date string)
MEAL_LOG: dict[str, list[dict]] = {}


# ─────────────────────────────────────────────────────────────────────────────
# TOOL DEFINITIONS
# These are what Chapter 4 will contrast against Skill files.
# A tool gives the agent *access*; a Skill gives it *judgment*.
# ─────────────────────────────────────────────────────────────────────────────

def lookup_nutrition(food_item: str) -> str:
    """
    Tool: Nutrition Database Lookup
    Returns macro-nutrient data for a given food (per standard serving).
    """
    key = food_item.strip().lower()
    if key in NUTRITION_DB:
        data = NUTRITION_DB[key]
        return json.dumps({"food": key, **data})
    # Fuzzy fallback: check partial matches
    matches = [k for k in NUTRITION_DB if key in k or k in key]
    if matches:
        data = NUTRITION_DB[matches[0]]
        return json.dumps({"food": matches[0], "note": f"closest match for '{food_item}'", **data})
    return json.dumps({"error": f"'{food_item}' not found in nutrition database"})


def log_meal(food_item: str, quantity_g: float = 100.0, meal_type: str = "snack") -> str:
    """
    Tool: Meal Logger
    Records a food entry for today's date.
    """
    today = datetime.date.today().isoformat()
    entry = {
        "food": food_item,
        "quantity_g": quantity_g,
        "meal_type": meal_type,
        "timestamp": datetime.datetime.now().isoformat(),
    }
    MEAL_LOG.setdefault(today, []).append(entry)
    return json.dumps({"logged": True, "entry": entry, "daily_entries": len(MEAL_LOG[today])})


def get_daily_summary(date_str: str | None = None) -> str:
    """
    Tool: Daily Nutrition Summary
    Aggregates all logged meals for a given date (defaults to today).
    """
    target = date_str or datetime.date.today().isoformat()
    entries = MEAL_LOG.get(target, [])
    if not entries:
        return json.dumps({"date": target, "message": "No meals logged yet."})

    totals: dict[str, float] = {"calories": 0, "protein_g": 0, "carbs_g": 0, "fat_g": 0, "fibre_g": 0}
    for entry in entries:
        food_data_str = lookup_nutrition(entry["food"])
        food_data = json.loads(food_data_str)
        if "error" not in food_data:
            scale = entry["quantity_g"] / 100.0
            for macro in totals:
                totals[macro] += food_data.get(macro, 0) * scale

    return json.dumps({"date": target, "entries": len(entries), "totals": {k: round(v, 1) for k, v in totals.items()}})


# ─────────────────────────────────────────────────────────────────────────────
# 2.2  CrewAI Workflow
# ─────────────────────────────────────────────────────────────────────────────

def build_crewai_diet_coach():
    """Section 2.2: AI Diet Coach as a CrewAI single-agent crew."""
    if not CREWAI_AVAILABLE:
        print("crewai not installed; run: pip install crewai")
        return

    @tool("NutritionLookup")
    def crewai_lookup(food_item: str) -> str:
        """Look up nutritional information for a food item."""
        return lookup_nutrition(food_item)

    @tool("MealLogger")
    def crewai_log(food_item: str, quantity_g: float = 100.0, meal_type: str = "lunch") -> str:
        """Log a meal entry for today."""
        return log_meal(food_item, quantity_g, meal_type)

    @tool("DailySummary")
    def crewai_summary(date_str: str = "") -> str:
        """Get today's nutrition summary."""
        return get_daily_summary(date_str or None)

    coach_agent = Agent(
        role="AI Diet Coach",
        goal="Help users track nutrition and make healthier food choices",
        backstory=(
            "You are a certified nutrition coach with deep knowledge of macro- "
            "and micro-nutrients. You always log meals your user mentions and "
            "proactively surface their daily totals."
        ),
        tools=[crewai_lookup, crewai_log, crewai_summary],
        verbose=True,
    )

    task = Task(
        description=(
            "The user says: 'I had oats for breakfast and chicken breast with "
            "broccoli for lunch. How am I doing on protein today?'\n"
            "1. Log both meals.\n"
            "2. Look up nutritional data for each food.\n"
            "3. Generate a friendly summary focusing on protein intake."
        ),
        expected_output="A friendly, data-backed nutrition summary with protein focus.",
        agent=coach_agent,
    )

    crew = Crew(agents=[coach_agent], tasks=[task], process=Process.sequential, verbose=True)
    result = crew.kickoff()
    print("\n── CrewAI Result ──────────────────────────────────────────")
    print(result)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 2.3  Same Use Case — No Framework (raw Anthropic tool-use API)
# ─────────────────────────────────────────────────────────────────────────────

import anthropic

TOOLS_SPEC = [
    {
        "name": "lookup_nutrition",
        "description": "Return macro-nutrient data for a food item (per 100 g serving).",
        "input_schema": {
            "type": "object",
            "properties": {"food_item": {"type": "string", "description": "Name of the food"}},
            "required": ["food_item"],
        },
    },
    {
        "name": "log_meal",
        "description": "Record a food entry in the daily meal log.",
        "input_schema": {
            "type": "object",
            "properties": {
                "food_item":   {"type": "string"},
                "quantity_g":  {"type": "number", "description": "Quantity in grams"},
                "meal_type":   {"type": "string", "enum": ["breakfast", "lunch", "dinner", "snack"]},
            },
            "required": ["food_item"],
        },
    },
    {
        "name": "get_daily_summary",
        "description": "Return today's aggregated nutrition totals from logged meals.",
        "input_schema": {
            "type": "object",
            "properties": {"date_str": {"type": "string", "description": "ISO date (omit for today)"}},
        },
    },
]

TOOL_REGISTRY = {
    "lookup_nutrition":  lookup_nutrition,
    "log_meal":          log_meal,
    "get_daily_summary": get_daily_summary,
}


def run_tool(tool_name: str, tool_input: dict) -> str:
    """Dispatch a tool call to the correct Python function."""
    fn = TOOL_REGISTRY.get(tool_name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})
    return fn(**tool_input)


def run_diet_coach_no_framework(user_message: str) -> str:
    """
    Section 2.3: The same diet coach behaviour — but using the raw
    Anthropic Messages API with tool_use content blocks.

    Agentic loop:
        while model returns tool_use blocks:
            execute each tool
            append tool_results to messages
        return final text response
    """
    client = anthropic.Anthropic()
    messages = [{"role": "user", "content": user_message}]

    system = (
        "You are an AI Diet Coach. When a user mentions food they have eaten, "
        "always log it with log_meal, look up its nutrition with lookup_nutrition, "
        "and close with get_daily_summary. Be concise and encouraging."
    )

    print(f"\n[no-framework] User: {user_message}\n")

    while True:
        response = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=1024,
            system=system,
            tools=TOOLS_SPEC,
            messages=messages,
        )

        # Collect text blocks for display
        for block in response.content:
            if block.type == "text" and block.text:
                print(f"[assistant text] {block.text}")

        if response.stop_reason == "end_turn":
            # Final answer — extract and return text
            final_text = " ".join(b.text for b in response.content if b.type == "text")
            return final_text

        if response.stop_reason != "tool_use":
            break  # unexpected stop reason

        # Append the assistant's full content (including tool_use blocks)
        messages.append({"role": "assistant", "content": response.content})

        # Execute each tool and build the tool_result turn
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"  → tool call: {block.name}({block.input})")
                result_str = run_tool(block.name, block.input)
                print(f"  ← result: {result_str}")
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_str,
                })

        messages.append({"role": "user", "content": tool_results})

    return "[loop ended unexpectedly]"


# ─────────────────────────────────────────────────────────────────────────────
# 2.4  Boilerplate Builder: reduce repetition with a thin wrapper
# ─────────────────────────────────────────────────────────────────────────────

class DietCoachAgent:
    """
    Section 2.4: A minimal reusable wrapper that packages the agentic loop,
    system prompt, and tool registry so you don't repeat the while-loop
    in every script.
    """

    def __init__(self, model: str = "claude-opus-4-5", max_tokens: int = 1024):
        self.client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens
        self.system = (
            "You are an AI Diet Coach with access to a nutrition database "
            "and a meal logger. Use your tools proactively."
        )
        self.tools = TOOLS_SPEC
        self.history: list[dict] = []

    def chat(self, user_message: str) -> str:
        """Send a message and run the tool loop until a final answer."""
        self.history.append({"role": "user", "content": user_message})

        while True:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=self.system,
                tools=self.tools,
                messages=self.history,
            )

            self.history.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "end_turn":
                return " ".join(b.text for b in response.content if b.type == "text")

            if response.stop_reason != "tool_use":
                return f"[unexpected stop: {response.stop_reason}]"

            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result_str = run_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_str,
                    })

            self.history.append({"role": "user", "content": tool_results})


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n── Section 2.3: No-Framework Agent ───────────────────────")
    answer = run_diet_coach_no_framework(
        "I had oats for breakfast and chicken breast with broccoli for lunch. "
        "How am I doing on protein today?"
    )
    print(f"\nFinal answer:\n{answer}")

    print("\n── Section 2.4: Boilerplate Builder ──────────────────────")
    coach = DietCoachAgent()
    reply = coach.chat("Log my dinner: salmon, 150 g. Then give me today's summary.")
    print(f"\nCoach: {reply}")
