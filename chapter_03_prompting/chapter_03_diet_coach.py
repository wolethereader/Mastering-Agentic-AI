"""
Mastering Agentic AI — Chapter 3
Prompting and Context Engineering

Sections covered:
  3.1  Prompt Engineering for Agents
  3.2  Principles for Multi-Agent Prompting
  3.3  Building a Prompting Framework
  3.4  Prompt Optimisation with DSPy
  3.5  Structured Outputs
  3.6  Context Engineering: Designing Effective Inputs and Information Flow

The quality of your agent's reasoning is bounded
by the quality of its prompt. Time spent here compounds across every
downstream chapter.

New in this chapter: the AI Diet Coach reads from SKILL.md — a
Nutrition Assessment Protocol. The Skill is *procedural* (it describes
HOW to think). The tools from Chapter 2 are *declarative* (they describe
WHAT the agent can access). Together they form the basis of a capable agent.
"""

import os
import json
import re
from pathlib import Path
from typing import Any
import anthropic

# ─────────────────────────────────────────────────────────────────────────────
# Skill loader — reads the procedural SKILL.md from this chapter's directory
# ─────────────────────────────────────────────────────────────────────────────

SKILL_PATH = Path(__file__).parent / "SKILL.md"


def load_skill(path: Path = SKILL_PATH) -> str:
    """
    Load the Nutrition Assessment Protocol from SKILL.md.

    In production agents (Chapter 5+) this would be retrieved from a
    vector store via semantic search. For now we inject it in full.
    """
    if path.exists():
        return path.read_text()
    return ""  # graceful degradation


NUTRITION_ASSESSMENT_SKILL = load_skill()


# ─────────────────────────────────────────────────────────────────────────────
# 3.1  Prompt Engineering for Agents
# ─────────────────────────────────────────────────────────────────────────────

def build_system_prompt(skill: str = NUTRITION_ASSESSMENT_SKILL) -> str:
    """
    Section 3.1: Compose a layered system prompt.

    Structure:
        [ROLE]       Who the agent is and what it cares about
        [SKILL]      Procedural protocol injected from SKILL.md
        [CONSTRAINTS] Hard guardrails the agent must never violate
        [FORMAT]     Expected output structure
    """
    return f"""[ROLE]
You are an AI Diet Coach with formal training in nutrition science.
Your mission: help users build sustainable, evidence-based eating habits.

[SKILL — Nutrition Assessment Protocol]
{skill}

[CONSTRAINTS]
- Never diagnose medical conditions. Refer users to a registered dietitian
  for therapeutic diets (e.g. renal diet, eating disorder recovery).
- Do not fabricate specific calorie counts for branded or restaurant foods.
- Do not recommend supplements above safe upper intake levels.
- If a user reports symptoms (dizziness, nausea, rapid weight loss), advise
  them to seek medical attention immediately.

[FORMAT]
When conducting a nutrition assessment, follow the four steps in the Skill
exactly (Baseline → Deficits → Priorities → Single Goal).
For casual questions, reply conversationally in 2–4 sentences.
Always end assessments with a clearly labelled "Your Goal This Week:" section.
"""


# ─────────────────────────────────────────────────────────────────────────────
# 3.2  Principles for Multi-Agent Prompting
# ─────────────────────────────────────────────────────────────────────────────

ORCHESTRATOR_PROMPT = """You are the Diet Coach Orchestrator.
Your job is to route user requests to the correct specialist sub-agent:

  • NutritionAnalyst  — detailed macro/micro-nutrient breakdowns
  • MealPlanner       — weekly meal plans and recipe suggestions
  • BehaviourCoach    — habit formation, motivation, accountability

Decide which agent to call, pass only the relevant context, and
synthesise their outputs into a single coherent reply for the user.
Return your routing decision as JSON: {"agent": "<name>", "query": "<sub-query>"}
"""

NUTRITION_ANALYST_PROMPT = """You are the NutritionAnalyst sub-agent.
Receive a specific nutrition question and answer it with precision.
Cite nutrient reference values (e.g. NHS/RDA) where applicable.
Output: a structured analysis with headings: Overview, Key Nutrients, Recommendation.
"""

MEAL_PLANNER_PROMPT = """You are the MealPlanner sub-agent.
Create practical, time-efficient meal plans that match the user's goals and preferences.
Output: a markdown table with Day, Meal, Key Ingredients, Approx. Protein (g).
"""


# ─────────────────────────────────────────────────────────────────────────────
# 3.3  Building a Prompting Framework
# ─────────────────────────────────────────────────────────────────────────────

class PromptTemplate:
    """
    Section 3.3: A lightweight prompt template with variable substitution.
    Avoids f-string spaghetti as prompts grow across chapters.
    """

    def __init__(self, template: str):
        self._template = template
        self._variables = re.findall(r"\{\{(\w+)\}\}", template)

    @property
    def variables(self) -> list[str]:
        return self._variables

    def render(self, **kwargs: Any) -> str:
        result = self._template
        for key, value in kwargs.items():
            result = result.replace(f"{{{{{key}}}}}", str(value))
        missing = [v for v in self._variables if f"{{{{{v}}}}}" in result]
        if missing:
            raise ValueError(f"Missing template variables: {missing}")
        return result


ASSESSMENT_TEMPLATE = PromptTemplate("""
Conduct a nutrition assessment for the following user profile:
  Name:           {{name}}
  Age:            {{age}}
  Weight (kg):    {{weight_kg}}
  Goals:          {{goals}}
  Typical Day:    {{typical_day}}
  Restrictions:   {{restrictions}}

Follow the four-step Nutrition Assessment Protocol from your Skill file.
""")


# ─────────────────────────────────────────────────────────────────────────────
# 3.4  Prompt Optimisation with DSPy  (conceptual scaffold)
# ─────────────────────────────────────────────────────────────────────────────

try:
    import dspy  # type: ignore

    class NutritionAssessmentSignature(dspy.Signature):
        """Assess a user's diet and return a structured recommendation."""
        user_profile: str = dspy.InputField(desc="JSON string of user profile data")
        assessment:   str = dspy.OutputField(desc="Four-step nutrition assessment per SKILL.md")

    class DietCoachDSPy(dspy.Module):
        """Section 3.4: DSPy module wrapping the diet assessment."""

        def __init__(self):
            super().__init__()
            self.assess = dspy.ChainOfThought(NutritionAssessmentSignature)

        def forward(self, user_profile: str) -> dspy.Prediction:
            return self.assess(user_profile=user_profile)

    DSPy_AVAILABLE = True

except ImportError:
    DSPy_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# 3.5  Structured Outputs
# ─────────────────────────────────────────────────────────────────────────────

STRUCTURED_ASSESSMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "baseline_summary":   {"type": "string"},
        "deficits": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "nutrient":  {"type": "string"},
                    "severity":  {"type": "string", "enum": ["low", "medium", "high"]},
                    "note":      {"type": "string"},
                },
                "required": ["nutrient", "severity", "note"],
            },
        },
        "priority_actions":   {"type": "array", "items": {"type": "string"}},
        "goal_this_week":     {"type": "string"},
    },
    "required": ["baseline_summary", "deficits", "priority_actions", "goal_this_week"],
}


def run_structured_assessment(user_profile: dict) -> dict:
    """
    Section 3.5: Ask the model to return a JSON assessment that conforms
    to our schema — avoiding free-text parsing entirely.
    """
    client = anthropic.Anthropic()

    prompt = (
        f"User profile: {json.dumps(user_profile, indent=2)}\n\n"
        "Conduct a nutrition assessment following your SKILL protocol. "
        "Return ONLY valid JSON matching this schema:\n"
        f"{json.dumps(STRUCTURED_ASSESSMENT_SCHEMA, indent=2)}\n"
        "No markdown fences. No extra keys. Pure JSON."
    )

    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        system=build_system_prompt(),
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip()
    # Strip any accidental fences
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\n?", "", raw).rstrip("`").strip()

    return json.loads(raw)


# ─────────────────────────────────────────────────────────────────────────────
# 3.6  Context Engineering: Designing Effective Inputs
# ─────────────────────────────────────────────────────────────────────────────

def build_context_window(
    user_message: str,
    meal_history: list[dict] | None = None,
    user_profile: dict | None = None,
    skill_text: str = NUTRITION_ASSESSMENT_SKILL,
) -> list[dict]:
    """
    Section 3.6: Deliberately compose the context window.

    Rule of thumb (Andrew Ng): treat every token in the context window
    as a decision. Ask 'does this token help the model reason better?'
    If not, remove it.

    Context layers (most → least important for this agent):
        1. User's current message
        2. User profile (goals, weight, restrictions)
        3. Recent meal history (last 3 meals)
        4. Skill protocol (injected via system prompt, not here)
    """
    messages: list[dict] = []

    # Layer: user profile context (prepended as a structured note)
    if user_profile:
        profile_note = (
            f"[Context] User profile:\n{json.dumps(user_profile, indent=2)}"
        )
        messages.append({"role": "user", "content": profile_note})
        messages.append({"role": "assistant", "content": "Understood — I have your profile loaded."})

    # Layer: recent meal history (last 3 meals only — token budget)
    if meal_history:
        recent = meal_history[-3:]
        history_note = (
            "Recent meals logged:\n"
            + "\n".join(f"  - {m.get('food', '?')} ({m.get('meal_type', '?')})" for m in recent)
        )
        messages.append({"role": "user", "content": history_note})
        messages.append({"role": "assistant", "content": "Got it — I'll factor in your recent meals."})

    # Layer: actual user message
    messages.append({"role": "user", "content": user_message})
    return messages


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("── Chapter 3 Demo: Structured Nutrition Assessment ─────────")

    profile = {
        "name": "Jordan",
        "age": 32,
        "weight_kg": 78,
        "goals": "Lose 5 kg over 3 months, maintain energy for morning runs",
        "typical_day": "Coffee for breakfast, sandwich at lunch, pasta for dinner",
        "restrictions": "Lactose intolerant",
    }

    result = run_structured_assessment(profile)
    print(json.dumps(result, indent=2))

    print("\n── Prompt Template Demo ────────────────────────────────────")
    rendered = ASSESSMENT_TEMPLATE.render(
        name="Jordan",
        age=32,
        weight_kg=78,
        goals="Lose 5 kg, maintain running energy",
        typical_day="Coffee, sandwich, pasta",
        restrictions="Lactose intolerant",
    )
    print(rendered)
