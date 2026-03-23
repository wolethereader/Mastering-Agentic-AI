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
import dspy
from dspy.teleprompt import MIPROv2
from dotenv import load_dotenv


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
        text = path.read_text()
        # Skill files can include YAML front matter for metadata. Strip it
        # before injecting the procedural instructions into the prompt.
        if text.startswith("---"):
            parts = text.split("---", 2)
            if len(parts) == 3:
                return parts[2].strip()
        return text
    return ""


NUTRITION_ASSESSMENT_SKILL = load_skill()


# ─────────────────────────────────────────────────────────────────────────────
# 3.1  Prompt Engineering for Agents
# ─────────────────────────────────────────────────────────────────────────────

def build_system_prompt(skill: str = NUTRITION_ASSESSMENT_SKILL) -> str:
    """
    Section 3.1: Compose a layered system prompt.

    Structure:
        [ROLE]        Who the agent is and what it cares about
        [SKILL]       Procedural protocol injected from SKILL.md
        [CONSTRAINTS] Hard guardrails the agent must never violate
        [FORMAT]      Expected output structure
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
# 3.4  Prompt Optimisation with DSPy
# ─────────────────────────────────────────────────────────────────────────────

load_dotenv()

# LM Setup
main_lm = dspy.LM(model="openai/gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
main_lm("Hello", max_tokens=10)
dspy.settings.configure(lm=main_lm)

judge_lm = dspy.LM(model="openai/gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))


# Food Database Tool
@dspy.Tool
def foodDBtool(food_item: str) -> str:
    with open(Path(__file__).parent / "foodDB.json", "r") as f:
        nutrition_db = json.load(f)

    query = food_item.lower().strip()

    if query in nutrition_db:
        info = nutrition_db[query]
        return (
            f"Nutrition for {info['serving_size']} of {query.capitalize()}: "
            f"{info['calories']} calories, "
            f"{info['protein_g']}g protein, "
            f"{info['carbs_g']}g carbs, "
            f"{info['fat_g']}g fat, "
            f"{info['fiber_g']}g fiber."
        )
    else:
        return f"Sorry, nutrition information for '{food_item}' was not found in the database."


class DietAnalysis(dspy.Signature):
    """Analyze a meal and provide nutritional breakdown"""
    meal: str = dspy.InputField(desc="Description of the meal eaten")
    analysis: str = dspy.OutputField(
        desc="Nutritional breakdown with calories, protein, carbs, fat and health assessment"
    )


diet_agent = dspy.ReAct(
    DietAnalysis,
    tools=[foodDBtool],
)
diet_agent.set_lm(main_lm)


class NutritionJudge(dspy.Signature):
    """Judge the quality of nutritional analysis"""
    meal: str = dspy.InputField()
    analysis: str = dspy.InputField()
    quality_score: float = dspy.OutputField(
        desc="Score 0-1 based on accuracy, completeness, and helpfulness of nutrition analysis"
    )


judge = dspy.ChainOfThought(NutritionJudge)
judge.set_lm(judge_lm)


def nutrition_metric(gold, pred, trace=None):
    result = judge(meal=gold.meal, analysis=pred.analysis)
    return result.quality_score


trainset = [
    dspy.Example(
        meal="Grilled chicken breast with steamed broccoli and brown rice",
        analysis="Meal Breakdown:\n- Chicken breast (100g): 165 calories, 31g protein, 0g carbs, 3.6g fat\n- Broccoli (1 cup): 25 calories, 3g protein, 5g carbs, 0.3g fat\n- Rice (1 cup cooked): 205 calories, 4.3g protein, 45g carbs, 0.4g fat\n\nTotal: This meal contains approximately 395 calories, 38.3g protein, 50g carbs, and 4.3g fat. A very healthy, balanced meal."
    ).with_inputs("meal"),
    dspy.Example(
        meal="Two scrambled eggs with two slices of toast",
        analysis="Meal Breakdown:\n- Egg (2 large): 140 calories, 12g protein, 1.2g carbs, 10g fat\n- Bread (2 slices): 160 calories, 8g protein, 30g carbs, 2g fat\n\nTotal: This breakfast provides about 300 calories, 20g protein, 31.2g carbs, and 12g fat. A solid, protein-rich breakfast."
    ).with_inputs("meal"),
    dspy.Example(
        meal="Apple and banana for snack",
        analysis="Meal Breakdown:\n- Apple (1 medium): 95 calories, 0.5g protein, 25g carbs, 0.3g fat\n- Banana (1 medium): 105 calories, 1.3g protein, 27g carbs, 0.4g fat\n\nTotal: This fruit snack contains 200 calories, 1.8g protein, 52g carbs, and 0.7g fat. A healthy snack for quick energy."
    ).with_inputs("meal"),
    dspy.Example(
        meal="100g salmon with 1 cup of pasta",
        analysis="Meal Breakdown:\n- Salmon (100g): 208 calories, 20g protein, 0g carbs, 13g fat\n- Pasta (1 cup cooked): 220 calories, 8g protein, 44g carbs, 1g fat\n\nTotal: This meal has 428 calories, 28g protein, 44g carbs, and 14g fat. Rich in omega-3 fatty acids."
    ).with_inputs("meal"),
    dspy.Example(
        meal="A salad with 100g chicken breast, 1 cup spinach, and one tablespoon of olive oil",
        analysis="Meal Breakdown:\n- Chicken breast (100g): 165 calories, 31g protein, 0g carbs, 3.6g fat\n- Spinach (1 cup raw): 7 calories, 0.9g protein, 1g carbs, 0.1g fat\n- Olive oil (1 tablespoon): 120 calories, 0g protein, 0g carbs, 14g fat\n\nTotal: This salad has approximately 292 calories, 31.9g protein, 1g carbs, and 17.7g fat. A very low-carb, high-protein meal."
    ).with_inputs("meal"),
    dspy.Example(
        meal="A bowl of cooked oats with a sliced banana and a quarter cup of almonds",
        analysis="Meal Breakdown:\n- Oats (1 cup cooked): 150 calories, 6g protein, 27g carbs, 3g fat\n- Banana (1 medium): 105 calories, 1.3g protein, 27g carbs, 0.4g fat\n- Nuts (almonds, 1/4 cup): 164 calories, 6g protein, 6g carbs, 14g fat\n\nTotal: This breakfast contains approximately 419 calories, 13.3g protein, 60g carbs, and 17.4g fat. A high-fiber, energy-rich meal."
    ).with_inputs("meal"),
    dspy.Example(
        meal="A bowl of Greek yogurt with a tablespoon of honey",
        analysis="Meal Breakdown:\n- Yogurt (greek, 100g): 97 calories, 17g protein, 3.5g carbs, 2.5g fat\n- honey: Not found in database.\n\nTotal: Based on available data, this snack has 97 calories, 17g protein, and 3.5g carbs. This is a high-protein snack."
    ).with_inputs("meal"),
]


optimizer = MIPROv2(metric=nutrition_metric, auto="light")
optimized_agent = optimizer.compile(
    diet_agent,
    trainset=trainset,
    requires_permission_to_run=False,
)


# ─────────────────────────────────────────────────────────────────────────────
# 3.5  Structured Outputs
# ─────────────────────────────────────────────────────────────────────────────

STRUCTURED_ASSESSMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "baseline_summary": {"type": "string"},
        "deficits": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "nutrient": {"type": "string"},
                    "severity": {"type": "string", "enum": ["low", "medium", "high"]},
                    "note": {"type": "string"}
                },
                "required": ["nutrient", "severity", "note"]
            }
        },
        "priority_actions": {"type": "array", "items": {"type": "string"}},
        "goal_this_week": {"type": "string"}
    },
    "required": ["baseline_summary", "deficits", "priority_actions", "goal_this_week"]
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

    if user_profile:
        profile_note = f"[Context] User profile:\n{json.dumps(user_profile, indent=2)}"
        messages.append({"role": "user", "content": profile_note})
        messages.append({"role": "assistant", "content": "Understood — I have your profile loaded."})

    if meal_history:
        recent = meal_history[-3:]
        history_note = (
            "Recent meals logged:\n"
            + "\n".join(f"  - {m.get('food', '?')} ({m.get('meal_type', '?')})" for m in recent)
        )
        messages.append({"role": "user", "content": history_note})
        messages.append({"role": "assistant", "content": "Got it — I'll factor in your recent meals."})

    messages.append({"role": "user", "content": user_message})
    return messages


if __name__ == "__main__":
    result = optimized_agent(meal="pasta, 4 eggs and a banana")
    print(result.analysis)

    optimized_agent.inspect_history()
