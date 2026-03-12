# SKILL: Nutrition Assessment Protocol
Version: 1.0  |  Chapter: 3  |  Mastering Agentic AI

## Purpose
This is a **procedural** skill — it tells the AI Diet Coach *how* to
think during a dietary evaluation, not just *what facts* to know.

## Protocol Steps

### Step 1 — Establish Baseline
Ask the user for: current eating pattern (typical day), dietary
restrictions, health goals, and activity level. Do not skip this step
even if some answers seem implied. A baseline without data is a guess.

### Step 2 — Identify Deficits
Against established nutritional reference values, check for:
- Protein shortfall (< 0.8 g/kg body weight)
- Fibre shortfall (< 25 g/day)
- Missing micronutrient groups (dairy/calcium, leafy greens/iron)
- Hydration (< 2 L water daily)
Flag deficits explicitly. Do not soften findings — clarity helps users.

### Step 3 — Prioritise Changes
Rank interventions by impact-to-effort ratio:
1. **High impact, low effort** — fix these first (e.g. add a serving of veg)
2. **High impact, high effort** — plan these for week 2+
3. **Low impact** — mention but do not dwell

### Step 4 — Set One Measurable Goal
End every assessment with exactly one concrete, time-bound goal.
Example: "Eat 30 g of protein at breakfast every day this week."
One goal. Not five. Behaviour change requires focus.

## Output Format
Assessment → Deficit List → Priority Action → Single Goal
