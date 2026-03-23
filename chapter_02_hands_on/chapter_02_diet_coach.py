"""
Mastering Agentic AI — Chapter 2
Building Agents

Minimal teaching example:
1. A simple CrewAI diet coach team.
2. The same workflow rebuilt with plain OpenAI API calls.
"""

import json
import os

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


# ─────────────────────────────────────────────────────────────────────────────
# Simple mock data
# ─────────────────────────────────────────────────────────────────────────────

FOOD_DB = {
    "greek yogurt": {"calories": 100, "protein_g": 17, "carbs_g": 6, "fat_g": 0.7},
    "oats": {"calories": 150, "protein_g": 5, "carbs_g": 27, "fat_g": 2.5},
    "banana": {"calories": 105, "protein_g": 1.3, "carbs_g": 27, "fat_g": 0.4},
    "chicken breast": {"calories": 165, "protein_g": 31, "carbs_g": 0, "fat_g": 3.6},
    "brown rice": {"calories": 216, "protein_g": 5, "carbs_g": 45, "fat_g": 1.8},
    "broccoli": {"calories": 55, "protein_g": 3.7, "carbs_g": 11, "fat_g": 0.6},
    "salmon": {"calories": 208, "protein_g": 28, "carbs_g": 0, "fat_g": 10},
    "spinach": {"calories": 23, "protein_g": 2.9, "carbs_g": 3.6, "fat_g": 0.4},
}

GOAL_TEMPLATES = {
    "fat loss": {
        "priority": "high protein, high fibre, controlled calories",
        "protein_target_g": 120,
    },
    "muscle gain": {
        "priority": "higher calories, high protein, balanced carbs",
        "protein_target_g": 150,
    },
}

MEAL_PLAN_LOG = []

CLIENT_PROFILE = {
    "name": "Maya",
    "goal": "fat loss",
    "dietary_preferences": "high-protein, mostly whole foods",
    "dislikes": "mushrooms",
    "calories_target": 1800,
    "meals_per_day": 3,
    "candidate_foods": [
        "greek yogurt",
        "oats",
        "banana",
        "chicken breast",
        "broccoli",
        "brown rice",
        "salmon",
        "spinach",
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# Tools
# ─────────────────────────────────────────────────────────────────────────────

def lookup_food(food_name: str) -> str:
    key = food_name.strip().lower()
    if key in FOOD_DB:
        return json.dumps({"food": key, **FOOD_DB[key]})
    return json.dumps({"error": f"'{food_name}' is not in the food database"})


def get_goal_template(goal: str) -> str:
    key = goal.strip().lower()
    if key in GOAL_TEMPLATES:
        return json.dumps({"goal": key, **GOAL_TEMPLATES[key]})
    return json.dumps({"error": f"Unknown goal '{goal}'"})


def save_meal_plan(user_name: str, plan_markdown: str) -> str:
    MEAL_PLAN_LOG.append({"user_name": user_name, "plan": plan_markdown})
    return json.dumps({"saved": True, "plans_saved": len(MEAL_PLAN_LOG)})


# ─────────────────────────────────────────────────────────────────────────────
# 2.2 CrewAI version
# ─────────────────────────────────────────────────────────────────────────────

crew = None

try:
    from crewai import Agent, Crew, LLM, Process, Task
    from crewai_tools import tool

    if OPENAI_API_KEY:
        llm = LLM(model=OPENAI_MODEL, temperature=0.2)

        @tool("goal_template_lookup")
        def goal_template_lookup(goal: str) -> str:
            """Look up the nutrition goal template."""
            return get_goal_template(goal)

        @tool("food_lookup")
        def food_lookup(food_name: str) -> str:
            """Look up nutrition facts for one food."""
            return lookup_food(food_name)

        @tool("plan_saver")
        def plan_saver(user_name: str, plan_markdown: str) -> str:
            """Save the final meal plan."""
            return save_meal_plan(user_name, plan_markdown)

        intake_agent = Agent(
            role="Intake Coach",
            goal="Turn the client brief into clear diet requirements.",
            backstory="You clarify the user's goal, calorie target, and food constraints.",
            tools=[goal_template_lookup],
            llm=llm,
            verbose=True,
        )

        analyst_agent = Agent(
            role="Nutrition Analyst",
            goal="Choose foods that fit the client's goal.",
            backstory="You use calories and protein data to select sensible foods.",
            tools=[food_lookup],
            llm=llm,
            verbose=True,
        )

        planner_agent = Agent(
            role="Meal Planner",
            goal="Create a simple one-day meal plan.",
            backstory="You turn the brief and food analysis into practical meals.",
            tools=[plan_saver],
            llm=llm,
            verbose=True,
        )

        intake_task = Task(
            description=(
                "The client is {name}. Goal: {goal}. Preferences: {dietary_preferences}. "
                "Dislikes: {dislikes}. Calories target: {calories_target}. Meals per day: {meals_per_day}. "
                "Use the goal template tool and write a short intake brief."
            ),
            expected_output="A short intake brief with priorities and protein target.",
            agent=intake_agent,
        )

        analyst_task = Task(
            description=(
                "Using the intake brief and these candidate foods: {candidate_foods}, choose the best foods. "
                "Use the food lookup tool and return 5-7 foods with a short reason for each."
            ),
            expected_output="A shortlist of foods with short reasons.",
            agent=analyst_agent,
        )

        planner_task = Task(
            description=(
                "Create a one-day meal plan for {name}. Write breakfast, lunch, dinner, and one coaching note. "
                "Keep it close to {calories_target} calories. Then save the plan."
            ),
            expected_output="A saved one-day meal plan.",
            agent=planner_agent,
        )

        crew = Crew(
            agents=[intake_agent, analyst_agent, planner_agent],
            tasks=[intake_task, analyst_task, planner_task],
            process=Process.sequential,
            memory=True,
            verbose=True,
        )
        
except ImportError:
    pass


# ─────────────────────────────────────────────────────────────────────────────
# 2.3 Same workflow without a framework
# ─────────────────────────────────────────────────────────────────────────────

client = None

try:
    from openai import OpenAI

    if OPENAI_API_KEY:
        client = OpenAI(api_key=OPENAI_API_KEY)
except ImportError:
    pass


if __name__ == "__main__":
    print("Client profile:")
    print(json.dumps(CLIENT_PROFILE, indent=2))

    if crew and OPENAI_API_KEY:
        print("\n--- CrewAI workflow ---")
        result = crew.kickoff(
            inputs={
                "name": CLIENT_PROFILE["name"],
                "goal": CLIENT_PROFILE["goal"],
                "dietary_preferences": CLIENT_PROFILE["dietary_preferences"],
                "dislikes": CLIENT_PROFILE["dislikes"],
                "calories_target": CLIENT_PROFILE["calories_target"],
                "meals_per_day": CLIENT_PROFILE["meals_per_day"],
                "candidate_foods": ", ".join(CLIENT_PROFILE["candidate_foods"]),
            }
        )
        print(result)
    else:
        print("\n[CrewAI example skipped] Install crewai/crewai-tools and set OPENAI_API_KEY.")

    if client and OPENAI_API_KEY:
        print("\n--- No-framework workflow ---")

        goal_template = get_goal_template(CLIENT_PROFILE["goal"])
        food_facts = [json.loads(lookup_food(food)) for food in CLIENT_PROFILE["candidate_foods"]]

        planning_brief = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0.2,
            messages=[
                {"role": "system", "content": "You are a concise diet coach."},
                {
                    "role": "user",
                    "content": f"""
Interpret this client brief and write a short planning brief.

Client:
{json.dumps(CLIENT_PROFILE, indent=2)}

Goal template:
{goal_template}
""",
                },
            ],
        ).choices[0].message.content

        food_shortlist = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0.2,
            messages=[
                {"role": "system", "content": "You are a nutrition analyst. Use only the provided food facts."},
                {
                    "role": "user",
                    "content": f"""
Planning brief:
{planning_brief}

Food facts:
{json.dumps(food_facts, indent=2)}

Choose the best 5-7 foods and explain each choice in one short sentence.
""",
                },
            ],
        ).choices[0].message.content

        meal_plan = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0.2,
            messages=[
                {"role": "system", "content": "You are a practical diet coach writing simple meal plans."},
                {
                    "role": "user",
                    "content": f"""
Create a one-day meal plan for this client.

Client:
{json.dumps(CLIENT_PROFILE, indent=2)}

Planning brief:
{planning_brief}

Food shortlist:
{food_shortlist}

Write:
- Breakfast
- Lunch
- Dinner
- One snack
- One coaching note
""",
                },
            ],
        ).choices[0].message.content

        save_meal_plan(CLIENT_PROFILE["name"], meal_plan)
        print(meal_plan)
    else:
        print("\n[No-framework example skipped] Install openai and set OPENAI_API_KEY.")
