"""
Mastering Agentic AI — Chapter 3
Prompt Engineering for Agents

Sections covered:
  3.1  Prompt Engineering for Agents
  3.2  Principles for Multi-Agent Prompting
  3.3  Building a Prompting Framework (incl. Fact-Checker worked example)
  3.4  Agent Skills: Reusable Procedural Knowledge
  3.5  Prompt Optimisation with DSPy
  3.6  Structured Outputs
  3.7  The Full Pattern: Skills + Structured Outputs

Context Engineering moved to Chapter 5 — it belongs alongside the
memory systems that make dynamic context assembly possible.

The key insight: agent behaviour is a product of instruction design,
not model capability. The same model, given better instructions,
behaves like a different system entirely.
"""

import os
import json
import re
from pathlib import Path
from typing import Any

from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError
import dspy
from dspy.teleprompt import MIPROv2
from dotenv import load_dotenv

load_dotenv()


# ─────────────────────────────────────────────────────────────────────────────
# 3.1  Prompt Engineering for Agents
# ─────────────────────────────────────────────────────────────────────────────

SKILL_PATH = Path(__file__).parent / "SKILL.md"


def load_skill(path: Path = SKILL_PATH) -> str:
    """
    Load the Nutrition Assessment Protocol from SKILL.md.

    The Skill is *procedural* — it defines HOW to reason, not what facts
    to mention. In Chapter 5+ this would be retrieved dynamically from a
    vector store. For now we inject the full text into the system prompt.

    Returns an empty string if the file is not found so the agent
    continues to operate (degraded but not crashed).
    """
    try:
        text = path.read_text()
        if text.startswith("---"):
            parts = text.split("---", 2)
            if len(parts) == 3:
                return parts[2].strip()
        return text
    except FileNotFoundError:
        return ""   # agent degrades gracefully; [SKILL] section will be empty


NUTRITION_ASSESSMENT_SKILL = load_skill()


def build_system_prompt(skill: str = NUTRITION_ASSESSMENT_SKILL) -> str:
    """
    Section 3.1: Compose a layered system prompt from four sections.

    [ROLE]        Who the agent is and what it cares about
    [SKILL]       Procedural protocol injected from SKILL.md
    [CONSTRAINTS] Hard guardrails the agent must never violate
    [FORMAT]      Expected output structure for downstream parsing
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
Create practical, time-efficient meal plans that match the user's goals.
Output: a markdown table with Day, Meal, Key Ingredients, Approx. Protein (g).
"""


# ─────────────────────────────────────────────────────────────────────────────
# 3.3  Building a Prompting Framework
# ─────────────────────────────────────────────────────────────────────────────

class PromptTemplate:
    """
    Section 3.3: A lightweight prompt template with variable substitution.

    Uses {{variable}} syntax (double braces) to avoid conflicts with
    Python f-strings where single braces are evaluated immediately.
    The render() method uses string concatenation — not f-strings —
    to construct the placeholder, avoiding brace-escaping confusion.
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
            # String concatenation — not f-string — to avoid escaping errors
            placeholder = "{{" + key + "}}"
            result = result.replace(placeholder, str(value))
        missing = [v for v in self._variables if "{{" + v + "}}" in result]
        if missing:
            raise ValueError(f"Missing template variables: {missing}")
        return result


# Verify correct substitution — catches escaping bugs immediately
assert PromptTemplate("Hello {{name}}!").render(name="Alice") == "Hello Alice!"


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


# ── 3.3.3  Worked Example: Fact-Checker Agent ────────────────────────────────
#
# The Fact-Checker illustrates how all four prompt dimensions — identity,
# strategy, capabilities, collaboration — are encoded in a single profile.
# Each section is annotated to show which dimension it implements.

def make_fact_checker_prompt() -> str:
    """
    Build the Fact-Checker agent backstory with explicit dimension labels.
    In a framework like CrewAI this goes in the 'backstory' field.
    """
    return (
        # IDENTITY: who the agent is and the scope of its role
        "You are a precision nutrition scientist. Your purpose is to verify "
        "factual accuracy of dietary claims, not to generate content. "
        # STRATEGY: how it reasons and when to stop
        "Extract each factual claim. Verify or refute it using the database. "
        "Stop once all claims are checked or a high-confidence verdict is reached. "
        # CAPABILITIES: which tools and how to choose between them
        "Use lookup_nutrition for all numerical values. "
        "Never estimate nutrient values from memory. "
        # COLLABORATION: handoff format and explicit prohibitions
        "Return results to Coach Advisor as JSON. "
        "Escalate conflicting data to the Orchestrator. "
        "You are never responsible for publishing. Verify only."
    )


FACT_CHECKER_BACKSTORY = make_fact_checker_prompt()


# ─────────────────────────────────────────────────────────────────────────────
# 3.4  Agent Skills: Reusable Procedural Knowledge
# ─────────────────────────────────────────────────────────────────────────────
#
# Skills are stored in SKILL.md — a separate file from agent code.
# This keeps procedural knowledge (HOW to think) separate from
# tool access (WHAT to call) and control logic (HOW to structure reasoning).
#
# Tools without Skills → raw outputs with no structure.
# Skills without Tools → ungrounded reasoning with no verified data.
# Together they enable grounded, structured decision-making.
#
# load_skill() is defined in Section 3.1 where it is first used.
# In Chapter 5 it will be upgraded to dynamic retrieval from a vector store.


def display_skill_summary() -> None:
    """Print a summary of the loaded Skill for inspection."""
    skill = load_skill()
    if skill:
        lines = skill.splitlines()
        print(f"Skill loaded: {len(lines)} lines")
        print("First section:", next((l for l in lines if l.startswith("##")), "—"))
    else:
        print("No Skill file found — agent will use only system prompt constraints.")


# ─────────────────────────────────────────────────────────────────────────────
# 3.5  Prompt Optimisation with DSPy
# ─────────────────────────────────────────────────────────────────────────────
#
# IMPORTANT: DSPy setup and optimizer.compile() are wrapped in a function.
# Running optimizer.compile() at module import time triggers an expensive
# multi-step LLM optimisation on every `import chapter_03_diet_coach`.
# Always wrap demo code in run_dspy_demo() and call it from __main__.

FOOD_DB_PATH = Path(__file__).parent / "foodDB.json"


@dspy.Tool
def foodDBtool(food_item: str) -> str:
    """Look up nutrition data for a food item from the local database.

    Returns a formatted string with calories, protein, carbs, fat, and
    fibre. Returns a structured 'not found' message — never raises —
    so the agent can recognise and handle missing data explicitly rather
    than defaulting to hallucinated values.
    """
    with open(FOOD_DB_PATH, "r") as f:
        nutrition_db = json.load(f)

    query = food_item.lower().strip()   # normalise: 'Yoghurt' → 'yoghurt'

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
    return f"Sorry, nutrition information for '{food_item}' was not found in the database."


class DietAnalysis(dspy.Signature):
    """Analyze a meal and provide nutritional breakdown."""

    meal: str = dspy.InputField(desc="Description of the meal eaten")
    analysis: str = dspy.OutputField(
        desc=(
            "Nutritional breakdown with calories, protein, carbs, fat "
            "and health assessment. Acknowledge missing items explicitly."
        )
    )


class NutritionJudge(dspy.Signature):
    """Judge the quality of a nutritional analysis.

    The judge runs on a stronger model (gpt-4o) than the agent (gpt-4o-mini).
    This evaluator-must-be-stronger-than-agent rule is a production discipline:
    a judge that is weaker than the system it evaluates cannot reliably
    distinguish good outputs from bad ones.
    """

    meal: str = dspy.InputField()
    analysis: str = dspy.InputField()
    quality_score: float = dspy.OutputField(
        desc="Score 0–1 based on accuracy, completeness, and helpfulness"
    )


# Training data: seven examples covering different meal types.
# The final example (Greek yogurt + honey) deliberately includes a missing
# food item — this teaches the agent to acknowledge gaps, not fabricate values.
TRAINSET = [
    dspy.Example(
        meal="Grilled chicken breast with steamed broccoli and brown rice",
        analysis=(
            "Meal Breakdown:\n"
            "- Chicken breast (100g): 165 cal, 31g protein, 0g carbs, 3.6g fat\n"
            "- Broccoli (1 cup): 25 cal, 3g protein, 5g carbs, 0.3g fat\n"
            "- Brown rice (1 cup cooked): 205 cal, 4.3g protein, 45g carbs, 0.4g fat\n\n"
            "Total: ~395 cal, 38.3g protein, 50g carbs, 4.3g fat. "
            "A well-balanced, high-protein meal."
        ),
    ).with_inputs("meal"),
    dspy.Example(
        meal="Two scrambled eggs with two slices of toast",
        analysis=(
            "Meal Breakdown:\n"
            "- Eggs (2 large): 140 cal, 12g protein, 1.2g carbs, 10g fat\n"
            "- Bread (2 slices): 160 cal, 8g protein, 30g carbs, 2g fat\n\n"
            "Total: ~300 cal, 20g protein, 31g carbs, 12g fat. "
            "A solid, protein-rich breakfast."
        ),
    ).with_inputs("meal"),
    dspy.Example(
        meal="Apple and banana for snack",
        analysis=(
            "Meal Breakdown:\n"
            "- Apple (1 medium): 95 cal, 0.5g protein, 25g carbs, 0.3g fat\n"
            "- Banana (1 medium): 105 cal, 1.3g protein, 27g carbs, 0.4g fat\n\n"
            "Total: 200 cal, 1.8g protein, 52g carbs, 0.7g fat. "
            "A high-fibre snack for quick energy."
        ),
    ).with_inputs("meal"),
    dspy.Example(
        meal="100g salmon with 1 cup of pasta",
        analysis=(
            "Meal Breakdown:\n"
            "- Salmon (100g): 208 cal, 20g protein, 0g carbs, 13g fat\n"
            "- Pasta (1 cup cooked): 220 cal, 8g protein, 44g carbs, 1g fat\n\n"
            "Total: 428 cal, 28g protein, 44g carbs, 14g fat. "
            "Rich in omega-3 fatty acids."
        ),
    ).with_inputs("meal"),
    dspy.Example(
        meal="Salad with 100g chicken breast, 1 cup spinach, 1 tbsp olive oil",
        analysis=(
            "Meal Breakdown:\n"
            "- Chicken breast (100g): 165 cal, 31g protein, 0g carbs, 3.6g fat\n"
            "- Spinach (1 cup raw): 7 cal, 0.9g protein, 1g carbs, 0.1g fat\n"
            "- Olive oil (1 tbsp): 120 cal, 0g protein, 0g carbs, 14g fat\n\n"
            "Total: ~292 cal, 31.9g protein, 1g carbs, 17.7g fat. "
            "A very low-carb, high-protein meal."
        ),
    ).with_inputs("meal"),
    dspy.Example(
        meal="Bowl of cooked oats with a sliced banana and a quarter cup of almonds",
        analysis=(
            "Meal Breakdown:\n"
            "- Oats (1 cup cooked): 150 cal, 6g protein, 27g carbs, 3g fat\n"
            "- Banana (1 medium): 105 cal, 1.3g protein, 27g carbs, 0.4g fat\n"
            "- Almonds (1/4 cup): 164 cal, 6g protein, 6g carbs, 14g fat\n\n"
            "Total: ~419 cal, 13.3g protein, 60g carbs, 17.4g fat. "
            "A high-fibre, energy-rich breakfast."
        ),
    ).with_inputs("meal"),
    dspy.Example(
        meal="Bowl of Greek yogurt with a tablespoon of honey",
        analysis=(
            "Meal Breakdown:\n"
            "- Greek yogurt (100g): 97 cal, 17g protein, 3.5g carbs, 2.5g fat\n"
            "- Honey: not found in database — unable to verify exact values.\n\n"
            "Based on available data: 97 cal, 17g protein, 3.5g carbs. "
            "A high-protein snack; total may be higher with honey added."
        ),
    ).with_inputs("meal"),
]


def run_dspy_demo() -> None:
    """
    Section 3.5: Run the full DSPy optimisation demo.

    This function is intentionally NOT called at module level.
    optimizer.compile() triggers multiple LLM calls and takes 1–5 minutes.
    Call it explicitly from __main__ or a notebook cell.
    """
    main_lm = dspy.LM(
        model="openai/gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    main_lm("Hello", max_tokens=10)   # verify connection before proceeding
    dspy.settings.configure(lm=main_lm)

    judge_lm = dspy.LM(
        model="openai/gpt-4o",        # stronger model for evaluation
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    diet_agent = dspy.ReAct(DietAnalysis, tools=[foodDBtool])
    diet_agent.set_lm(main_lm)

    judge = dspy.ChainOfThought(NutritionJudge)
    judge.set_lm(judge_lm)

    def nutrition_metric(gold, pred, trace=None) -> float:
        result = judge(meal=gold.meal, analysis=pred.analysis)
        return result.quality_score

    optimizer = MIPROv2(metric=nutrition_metric, auto="light")
    optimized_agent = optimizer.compile(
        diet_agent,
        trainset=TRAINSET,
        requires_permission_to_run=False,
    )

    result = optimized_agent(meal="pasta, 4 eggs and a banana")
    print(result.analysis)
    optimized_agent.inspect_history()


# ─────────────────────────────────────────────────────────────────────────────
# 3.6  Structured Outputs
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
                    "nutrient":  {"type": "string"},
                    "severity":  {"type": "string", "enum": ["low", "medium", "high"]},
                    "note":      {"type": "string"},
                },
                "required": ["nutrient", "severity", "note"],
            },
        },
        "priority_actions": {"type": "array", "items": {"type": "string"}},
        "goal_this_week":   {"type": "string"},   # ← Step 4 of the Skill
    },
    "required": ["baseline_summary", "deficits", "priority_actions", "goal_this_week"],
}


# Pydantic model mirrors the schema — used for runtime validation
class NutritionDeficit(BaseModel):
    nutrient:  str
    severity:  str = Field(..., pattern="^(low|medium|high)$")
    note:      str


class NutritionAssessmentResult(BaseModel):
    baseline_summary:  str
    deficits:          list[NutritionDeficit]
    priority_actions:  list[str]
    goal_this_week:    str


# Separate client for repair calls — avoids polluting the main client state
_repair_client = OpenAI()


def repair_output(output: dict, schema: dict, max_retries: int = 3) -> dict:
    """
    Use a separate LLM call to correct a malformed JSON output.

    Raises ValueError after max_retries — never loops indefinitely.
    A hard failure here is preferable to silently propagating bad data.
    """
    for attempt in range(max_retries):
        prompt = (
            f"Fix this JSON to match the schema exactly.\n"
            f"Schema:\n{json.dumps(schema, indent=2)}\n\n"
            f"Malformed JSON:\n{json.dumps(output, indent=2)}\n\n"
            "Return ONLY the corrected JSON. No explanation. No markdown fences."
        )
        resp = _repair_client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\n?", "", raw).rstrip("`").strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            continue   # retry
    raise ValueError(f"repair_output failed after {max_retries} attempts")


def run_structured_assessment(user_profile: dict) -> dict:
    """
    Section 3.6: Request a schema-conforming JSON assessment.

    Uses the model specified by OPENAI_MODEL env var (default: gpt-4o-mini).
    Falls back to the repair loop on malformed JSON before raising.
    """
    client = OpenAI()

    prompt = (
        f"User profile: {json.dumps(user_profile, indent=2)}\n\n"
        "Conduct a nutrition assessment following your SKILL protocol. "
        "Return ONLY valid JSON matching this schema:\n"
        f"{json.dumps(STRUCTURED_ASSESSMENT_SCHEMA, indent=2)}\n"
        "No markdown fences. No extra keys. Pure JSON."
    )

    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0.2,
        messages=[
            {"role": "system", "content": build_system_prompt()},
            {"role": "user",   "content": prompt},
        ],
    )

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\n?", "", raw).rstrip("`").strip()

    result = json.loads(raw)   # hard fail on malformed JSON — do not propagate

    # Validate with Pydantic; repair and re-validate on failure
    try:
        validated = NutritionAssessmentResult.model_validate(result)
    except ValidationError:
        repaired = repair_output(result, STRUCTURED_ASSESSMENT_SCHEMA)
        validated = NutritionAssessmentResult.model_validate(repaired)

    return validated.model_dump()


# ─────────────────────────────────────────────────────────────────────────────
# 3.7  The Full Pattern: Skills + Structured Outputs
# ─────────────────────────────────────────────────────────────────────────────
#
# Skills govern HOW the agent reasons.
# Schemas govern WHAT the agent returns.
# Context engineering — governing WHAT the agent sees — is in Chapter 5,
# alongside the memory systems that make dynamic context assembly possible.
#
# This function is the composition payoff: both layers working together
# in a single production-ready call.

def run_skill_guided_assessment(user_message: str, user_profile: dict) -> dict:
    """
    Section 3.7: Full pattern — Skills + Structured Outputs.

    The Skill defines HOW to reason (four-step nutrition protocol).
    The schema defines WHAT to return (validated JSON with fixed fields).
    Neither is sufficient alone:
      - Skill without schema → right reasoning, unparseable output.
      - Schema without Skill → right format, inconsistent reasoning.
    """
    skill  = load_skill()               # HOW to think — procedural protocol
    system = build_system_prompt(skill) # Skill + constraints + format directive

    prompt = (
        f"User profile: {json.dumps(user_profile, indent=2)}\n\n"
        f"User message: {user_message}\n\n"
        "Follow the four-step Nutrition Assessment Protocol in your Skill. "
        "Return your response as valid JSON matching this schema:\n"
        f"{json.dumps(STRUCTURED_ASSESSMENT_SCHEMA, indent=2)}\n"
        "No markdown fences. Pure JSON only."
    )

    client   = OpenAI()
    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0.2,
        messages=[
            {"role": "system", "content": system},   # Skill via system prompt
            {"role": "user",   "content": prompt},   # schema in user turn
        ],
    )

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\n?", "", raw).rstrip("`").strip()

    result = json.loads(raw)   # hard fail — never propagate malformed data

    try:
        validated = NutritionAssessmentResult.model_validate(result)
    except ValidationError:
        repaired  = repair_output(result, STRUCTURED_ASSESSMENT_SCHEMA)
        validated = NutritionAssessmentResult.model_validate(repaired)

    return validated.model_dump()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # --- 3.5 DSPy demo (runs optimiser — takes 1–5 minutes) ---
    run_dspy_demo()

    # --- 3.7 Full pattern demo ---
    sample_profile = {
        "name": "Alex",
        "age": 34,
        "weight_kg": 78,
        "goals": "lose 5kg over 3 months",
        "typical_day": "skip breakfast, large lunch, snack at 3pm, dinner at 8pm",
        "restrictions": "no dairy",
    }
    assessment = run_skill_guided_assessment(
        user_message="Can you assess my current diet and give me one thing to change this week?",
        user_profile=sample_profile,
    )
    print(json.dumps(assessment, indent=2))
