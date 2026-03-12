# Repository Structure
## Mastering Agentic AI — Manning Publications

*Andrew Ng principle: a well-structured codebase is a form of documentation.
A reader who can navigate the repository confidently can learn from it independently.*

---

## Complete Directory Tree

```
mastering-agentic-ai/
│
├── README.md                          ← Start here: chapter map, quick-start
├── STRUCTURE.md                       ← This file: full repo navigation guide
├── requirements.txt                   ← Core dependencies (all chapters)
├── requirements-optional.txt          ← Chapter-specific + production extras
├── .env.example                       ← Template for your .env file
├── .gitignore                         ← Excludes .env, .memory/, __pycache__/
│
├── chapter_01_introduction/
│   ├── diet_coach_v1.py               ← Runnable Python script
│   ├── chapter_01_diet_coach.ipynb    ← Jupyter notebook (same content)
│   └── appendix_a_api_keys.md         ← Credential setup for this chapter
│
├── chapter_02_hands_on/
│   ├── diet_coach_v2.py
│   ├── chapter_02_diet_coach.ipynb
│   └── appendix_a_api_keys.md
│
├── chapter_03_prompting/
│   ├── diet_coach_v3.py
│   ├── chapter_03_diet_coach.ipynb
│   ├── SKILL.md                       ← Nutrition Assessment Protocol (loaded at runtime)
│   └── appendix_a_api_keys.md
│
├── chapter_04_tools/
│   ├── diet_coach_v4.py
│   ├── chapter_04_diet_coach.ipynb
│   ├── diet_coach_mcp_server.py       ← Standalone MCP server (run separately)
│   └── appendix_a_api_keys.md
│
├── chapter_05_memory/
│   ├── diet_coach_v5.py
│   ├── chapter_05_diet_coach.ipynb
│   └── appendix_a_api_keys.md
│
├── chapter_06_communication/
│   ├── diet_coach_v6.py
│   ├── chapter_06_diet_coach.ipynb
│   └── appendix_a_api_keys.md
│
├── chapter_07_evaluation_fundamentals/
│   ├── diet_coach_v7.py
│   ├── chapter_07_diet_coach.ipynb
│   └── appendix_a_api_keys.md
│
├── chapter_08_evaluation_practice/
│   ├── diet_coach_v8.py
│   ├── chapter_08_diet_coach.ipynb
│   └── appendix_a_api_keys.md
│
├── chapter_09_reinforcement_learning/
│   ├── diet_coach_v9.py
│   ├── chapter_09_diet_coach.ipynb
│   └── appendix_a_api_keys.md
│
└── chapter_10_security_governance/
    ├── diet_coach_v10.py
    ├── chapter_10_diet_coach.ipynb
    └── appendix_a_api_keys.md
```

---

## The Two-File Pattern: `.py` + `.ipynb`

Every chapter ships both a Python script and a Jupyter notebook covering identical concepts. They are not duplicates — they serve different purposes.

### Python Script (`diet_coach_vN.py`)

```
chapter_04_tools/diet_coach_v4.py
```

- Self-contained, runnable from the terminal: `python diet_coach_v4.py`
- Designed for engineers who prefer scripts over notebooks
- Contains the full, unabridged implementation including helper classes
- Imports between chapters work cleanly (e.g., v4 loads `SKILL.md` from `chapter_03_prompting/`)
- Safe to integrate into CI pipelines and automated tests
- Each file has a `if __name__ == "__main__":` block that runs a demo

**Run any chapter:**
```bash
cd mastering-agentic-ai/
python chapter_01_introduction/diet_coach_v1.py
python chapter_10_security_governance/diet_coach_v10.py
```

### Jupyter Notebook (`chapter_NN_diet_coach.ipynb`)

```
chapter_04_tools/chapter_04_diet_coach.ipynb
```

- Cell-by-cell pedagogical walkthrough — concept explained before code shown
- Inline markdown cells provide Andrew Ng-style commentary and design rationale
- Each cell builds on the previous: run them in order top to bottom
- Output cells show expected results so readers know what to expect
- Designed for learners, educators, and anyone working interactively
- The `# !pip install ...` comment at the top of the setup cell can be uncommented to install deps in Colab or a fresh environment

**Run any notebook:**
```bash
cd mastering-agentic-ai/
jupyter lab chapter_04_tools/chapter_04_diet_coach.ipynb
# or
jupyter notebook chapter_04_tools/chapter_04_diet_coach.ipynb
```

### When to Use Which

| Situation | Use |
|---|---|
| Following along with the book chapter | `.ipynb` — see the story unfold cell by cell |
| Building on top of the code | `.py` — cleaner imports and module structure |
| Running in CI/CD | `.py` — scriptable and testable |
| Teaching or presenting | `.ipynb` — slides-friendly, runnable live |
| Google Colab | `.ipynb` — upload directly |
| Debugging a specific function | `.py` — easier with a debugger |
| First time reading the chapter | `.ipynb` — richer context |

---

## File-by-File Reference

### `README.md`
The entry point. Chapter map table showing what each version of the coach gains. Quick-start setup commands. The Tools vs Skills conceptual table.

### `STRUCTURE.md` (this file)
Full directory tree, file-by-file descriptions, the two-file pattern explanation, environment setup instructions, and the .env.example template.

### `requirements.txt`
All packages needed to run every script and notebook. Pinned to compatible version ranges. Includes the Jupyter runtime. Install once for the whole book:
```bash
pip install -r requirements.txt
```

### `requirements-optional.txt`
Chapter-specific production upgrades (Pinecone, Redis, Composio, TRL, Sentry, W&B) and development tools (pytest, ruff, mypy). Each section is labelled with the chapter and appendix reference it corresponds to.

### `chapter_03_prompting/SKILL.md`
The Nutrition Assessment Protocol — a procedural skill file that tells the Diet Coach *how* to think, not just *what* data to access. Loaded at runtime by `load_skill()` in both `diet_coach_v3.py` and `diet_coach_v4.py`. The central concept of Chapter 3. Do not delete this file — it is a required runtime dependency for Chapters 3 and 4.

### `chapter_04_tools/diet_coach_mcp_server.py`
A standalone MCP server that exposes the `lookup_nutrition` tool over stdio transport. Run in a separate terminal:
```bash
pip install mcp
python chapter_04_tools/diet_coach_mcp_server.py
```
Then connect from the notebook using `SimpleMCPClient`. This file is generated by `diet_coach_v4.py` if it does not exist, but the pre-generated version is included for convenience.

### `appendix_a_api_keys.md` (one per chapter)
Step-by-step credential setup for every API key needed in that chapter. Includes: where to get the key, how to add it to `.env`, a verification snippet, a token budget table, and a common issues section. Read this *before* running the chapter for the first time.

---

## Environment Setup

### Step 1 — Clone and enter the repo

```bash
git clone https://github.com/your-org/mastering-agentic-ai.git
cd mastering-agentic-ai
```

### Step 2 — Create a virtual environment

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Always use a virtual environment. Dependencies across chapters differ; a shared environment avoids version conflicts.

### Step 3 — Install core dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Create your `.env` file

Copy the template:
```bash
cp .env.example .env
```

Open `.env` and fill in your API key:
```
ANTHROPIC_API_KEY=sk-ant-api03-YOUR-KEY-HERE
```

See each chapter's `appendix_a_api_keys.md` for keys required by that chapter's optional features.

### Step 5 — Verify setup

```bash
python -c "
import os
from anthropic import Anthropic
from dotenv import load_dotenv
load_dotenv()
c = Anthropic()
r = c.messages.create(model='claude-opus-4-5', max_tokens=20,
    messages=[{'role':'user','content':'Reply: ready'}])
print(r.content[0].text)
"
```

Expected output: `ready` (or similar). If this works, every chapter will work.

### Step 6 — Launch Jupyter

```bash
jupyter lab
```

Navigate to any chapter directory and open the `.ipynb` file.

---

## Chapter Dependencies

Each chapter builds on the previous in terms of concepts, but all Python scripts are standalone and runnable independently. The only runtime file dependency across chapters is:

```
chapter_04_tools/diet_coach_v4.py
  └── loads: chapter_03_prompting/SKILL.md   (relative path)
```

If you run Chapter 4 from a different working directory, update the `SKILL_PATH` constant at the top of `diet_coach_v4.py`.

### Dependency Graph

```
Ch1 (conversational loop)
 └── Ch2 (+ tools: lookup, log, summarise)
      └── Ch3 (+ SKILL.md + prompting framework)
           └── Ch4 (+ MCP + production tool design)
                └── Ch5 (+ 3-layer memory)
                     └── Ch6 (+ multi-agent communication)
                          ├── Ch7 (+ evaluation fundamentals)
                          │    └── Ch8 (+ evaluation practice)
                          └── Ch9 (+ RL + preference data)
                               └── Ch10 (+ security + governance)
```

This is a conceptual dependency graph, not a Python import graph. Every `.py` file imports only from the standard library and installed packages — never from other chapter directories (except the SKILL.md file path noted above).

---

## Running Tests

A minimal test for each chapter script:

```bash
# Test that each chapter script imports and runs without errors
for chapter in chapter_0*/diet_coach_v*.py; do
    echo "Testing $chapter..."
    python -c "import importlib.util, sys
spec = importlib.util.spec_from_file_location('m', '$chapter')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
print('  OK')
"
done
```

With pytest (install from `requirements-optional.txt`):

```bash
# Run all tests (if tests/ directory is added in future)
pytest tests/ -v

# Convert notebooks to scripts and run them
jupyter nbconvert --to script chapter_01_introduction/chapter_01_diet_coach.ipynb
python chapter_01_introduction/chapter_01_diet_coach.py
```

---

## The `.env.example` Template

A `.env.example` file should exist at the root of the repository. It lists every environment variable used across all 10 chapters, with placeholder values and chapter references. Copy it to `.env` and fill in your actual keys.

```bash
# =========================================================
# Mastering Agentic AI — .env.example
# Copy to .env and fill in your actual keys.
# NEVER commit .env to version control.
# =========================================================

# REQUIRED — All chapters (1–10)
ANTHROPIC_API_KEY=sk-ant-api03-YOUR-KEY-HERE

# OPTIONAL — Chapter 2 (CrewAI with OpenAI backend)
OPENAI_API_KEY=sk-YOUR-OPENAI-KEY-HERE

# OPTIONAL — Chapter 3 (DSPy prompt optimisation)
# Same ANTHROPIC_API_KEY works via litellm. No extra key needed.

# OPTIONAL — Chapter 4 (NutritionIX production tool)
NUTRITIONIX_APP_ID=your_nutritionix_app_id
NUTRITIONIX_API_KEY=your_nutritionix_api_key

# OPTIONAL — Chapter 4 (Composio tooling ecosystem)
COMPOSIO_API_KEY=your_composio_api_key

# OPTIONAL — Chapter 5 (Pinecone vector store for episodic memory)
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=us-east-1-aws

# OPTIONAL — Chapters 5 & 6 (Redis for semantic memory + message bus)
REDIS_URL=redis://default:password@your-endpoint.upstash.io:port

# REQUIRED for production audit log — Chapter 8 & 10
# Generate: python3 -c "import secrets; print(secrets.token_hex(32))"
AUDIT_HMAC_KEY=your-64-char-hex-key-here

# OPTIONAL — Chapter 9 (Weights & Biases experiment tracking)
WANDB_API_KEY=your_wandb_api_key

# OPTIONAL — Chapter 9 (HuggingFace dataset export)
HF_TOKEN=hf_your-token-here

# OPTIONAL — Chapter 10 (Sentry error monitoring)
SENTRY_DSN=https://your-dsn@sentry.io/project-id

# OPTIONAL — Chapter 10 (Open Policy Agent)
OPA_URL=http://localhost:8181
```

---

## `.gitignore` Template

Add this to `.gitignore` at the root of the repository:

```gitignore
# API keys — never commit
.env
*.env
.env.local

# Memory store — created at runtime by Chapter 5
.memory/

# Jupyter checkpoints
.ipynb_checkpoints/
*/.ipynb_checkpoints/

# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.egg-info/
dist/
build/

# Virtual environments
.venv/
venv/
env/

# Evaluation outputs — created at runtime by Chapters 7 & 8
eval_results_*.json
eval_history.json
preferences.jsonl
audit_log.json

# MCP server (auto-generated by diet_coach_v4.py if not present)
# Uncomment if you want to exclude the auto-generated version:
# chapter_04_tools/diet_coach_mcp_server.py

# OS
.DS_Store
Thumbs.db
```

---

## Package Summary by Chapter

| Chapter | Core packages | Optional packages |
|---|---|---|
| 1 · Introduction | `anthropic`, `python-dotenv` | — |
| 2 · Frameworks | + `crewai` | `openai` (CrewAI backend) |
| 3 · Prompting | + `pydantic`, `dspy-ai` | `litellm` (DSPy with Anthropic) |
| 4 · Tools | + `mcp`, `requests` | `composio-crewai` |
| 5 · Memory | (no new core deps) | `pinecone-client`, `sentence-transformers`, `redis` |
| 6 · Communication | (no new core deps) | `redis` (production MessageBus) |
| 7 · Eval Fundamentals | (no new core deps) | — |
| 8 · Eval Practice | (no new core deps) | `sentry-sdk` |
| 9 · Reinforcement Learning | (no new core deps) | `trl`, `datasets`, `transformers`, `wandb` |
| 10 · Security | (no new core deps) | `sentry-sdk` |

**Total core packages:** 7  
**Total optional packages:** 13  
**Standard library only chapters (no new installs):** 5, 6, 7, 8, 9, 10

---

## Google Colab

Every notebook is compatible with Google Colab. To run in Colab:

1. Upload the `.ipynb` file to Colab (or open from GitHub)
2. In the first cell, uncomment the `# !pip install ...` line
3. Add your API key: `import os; os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."` (do not save this to a shared Colab)
4. Run cells top to bottom

For SKILL.md (Chapter 3 and 4 notebooks), upload the file to Colab and update the path:

```python
SKILL_PATH = "/content/SKILL.md"   # Colab path after upload
```

---

*"A repository that requires a PhD to set up will not be used. Invest in the README, the requirements file, and the first five minutes of setup. That is where most readers make the decision to continue or give up."*

— Mastering Agentic AI, Chapter 1
