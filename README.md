# Mastering Agentic AI — Code Repository
### AI Diet Coach: Prototype → Production

---

## Overview

This repository contains all code files for **Mastering Agentic AI** (Manning Publications)

The AI Diet Coach evolves across 10 chapters — same system, increasing sophistication.

---

## Chapter Map

| Chapter | File | What the Coach Gains |
|---------|------|---------------------|
| 1 | `chapter_01_introduction/diet_coach_v1.py` | Conversational loop (no tools) |
| 2 | `chapter_02_hands_on/diet_coach_v2.py` | First tools: Nutrition DB + Meal Logger |
| 3 | `chapter_03_prompting/diet_coach_v3.py` + `SKILL.md` | Prompt framework + Nutrition Assessment Skill |
| 4 | `chapter_04_tools/diet_coach_v4.py` | MCP client/server + tool schema auto-generation |
| 5 | `chapter_05_memory/diet_coach_v5.py` | Semantic, episodic + in-context memory |
| 6 | `chapter_06_communication/diet_coach_v6.py` | Multi-agent orchestration (A2A/H2A protocols) |
| 7 | `chapter_07_evaluation_fundamentals/diet_coach_v7.py` | Eval suite, LLM-as-Judge, design pattern notes |
| 8 | `chapter_08_evaluation_practice/diet_coach_v8.py` | Non-determinism, RAG eval, adversarial testing |
| 9 | `chapter_09_reinforcement_learning/diet_coach_v9.py` | RLHF preference loop, bandit prompt optimisation |
| 10 | `chapter_10_security_governance/diet_coach_v10.py` | Guardrails, audit log, governance, ethics |

---

## The SKILL.md Concept (Chapter 3 & 4)

`chapter_03_prompting/SKILL.md` contains the **Nutrition Assessment Protocol** —
a 200-word procedural skill that tells the coach *how to think*, not just *what to access*.

**Tools vs Skills — the core distinction:**

| | Tool (`lookup_nutrition`) | Skill (`SKILL.md`) |
|--|--|--|
| Nature | Declarative — *what* to call | Procedural — *how* to think |
| Gives the agent | Access to data | Judgment about data |
| Without the other | Raw numbers, no strategy | Advice with no grounding |

---

## Setup

```bash
pip install anthropic crewai dspy-ai mcp
export ANTHROPIC_API_KEY=your_key_here
```

Run any chapter independently:
```bash
python chapter_01_introduction/diet_coach_v1.py
python chapter_10_security_governance/diet_coach_v10.py
```

---

*"You cannot improve what you cannot measure. And you cannot build what you
do not understand from first principles."* — Chapter 7
