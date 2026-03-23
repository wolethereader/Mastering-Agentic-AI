---
name: nutrition-assessment-protocol
description: Use this skill when the task is to assess a person's diet, identify likely nutritional gaps, prioritise improvements, and end with exactly one measurable next step. Do not use it for medical diagnosis, therapeutic diets, or unrelated general nutrition trivia.
---

# Nutrition Assessment Protocol

This is a procedural skill for the Diet Coach. It defines how to run a
structured dietary assessment, not just what facts to mention.

## Use This Skill When

- The user asks for a diet review or nutrition assessment
- The user wants feedback on their current eating pattern
- The user wants practical next steps to improve their diet
- The task requires a repeatable assessment workflow rather than a casual answer

## Do Not Use This Skill When

- The user is asking for diagnosis or treatment of a medical condition
- The user needs a therapeutic diet for a disease state
- The request is a simple factual nutrition lookup
- The user is asking for a full meal plan rather than an assessment

## Inputs To Collect

Before assessing, gather these inputs if they are missing:

- Typical eating pattern across a normal day
- Main goal, such as fat loss, muscle gain, or maintenance
- Dietary restrictions or strong dislikes
- Activity level
- Any relevant context such as hydration, meal timing, or recurring skipped meals

Do not invent missing inputs. If the user has not given enough detail,
state the assumption or ask for the missing information.

## Workflow

### Step 1 — Establish Baseline

Summarise the user’s current eating pattern in plain language.
Focus on meal structure, protein distribution, fruit and vegetable intake,
hydration, and any obvious routine constraints.

### Step 2 — Identify Deficits

Check for likely gaps such as:

- Protein shortfall
- Fibre shortfall
- Low fruit and vegetable variety
- Poor hydration
- Large timing gaps between meals

Flag deficits clearly. Do not soften them into vague language.

### Step 3 — Prioritise Changes

Rank changes by impact and ease:

1. High impact, low effort
2. High impact, higher effort
3. Lower impact improvements

Lead with the smallest useful change that meaningfully improves the diet.

### Step 4 — Set One Measurable Goal

End with exactly one concrete, time-bound goal.

Good example:
"Eat 30 grams of protein at breakfast every day this week."

Bad example:
"Try to eat healthier and drink more water."

## Output Format

Use this order:

1. Baseline
2. Deficits
3. Priority Action
4. Your Goal This Week

Keep the answer practical and direct.

## Guardrails

- Never diagnose medical conditions
- Never prescribe therapeutic diets
- Refer the user to a registered dietitian or clinician for medical nutrition therapy
- Prefer clear, simple actions over exhaustive advice
