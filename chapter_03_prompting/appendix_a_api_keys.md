# Appendix A · API Keys and Environment Setup
## Chapter 3 · Prompting with DSPy

*Mastering Agentic AI — Manning Publications*

Chapter 3 uses DSPy with OpenAI models:

- `gpt-4o-mini` as the main model
- `gpt-4o` as the judge model

---

## What Chapter 3 Needs

| Key | Required? | Used for |
|-----|-----------|----------|
| `OPENAI_API_KEY` | Required | DSPy main model and judge model |

---

## Step 1 · Create an OpenAI API Key

1. Go to [platform.openai.com](https://platform.openai.com)
2. Sign in to your account
3. Open the API keys page
4. Create a new secret key
5. Copy it immediately

---

## Step 2 · Store the Key in a `.env` File

Create a file named `.env` inside `chapter_03_prompting/`:

```bash
OPENAI_API_KEY=sk-proj-YOUR-KEY-HERE
```

Then load it in Python:

```python
from dotenv import load_dotenv
load_dotenv()
```

---

## Step 3 · Install Dependencies

```bash
pip install dspy-ai openai python-dotenv
```

---

## Step 4 · Verify the Setup

```python
import os
import dspy
from dotenv import load_dotenv

load_dotenv()

lm = dspy.LM(model="openai/gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
print(lm("Say hello in one sentence.", max_tokens=20))
```

If this works, the chapter setup is ready.

---

## Files Used in This Chapter

Chapter 3 uses a local nutrition database file:

```text
chapter_03_prompting/foodDB.json
```

The DSPy tool loads this file directly during the examples.

---

## Common Issues

### `OPENAI_API_KEY` is missing

- The `.env` file was not created
- `load_dotenv()` was not called
- The notebook is running from a different working directory

### `ModuleNotFoundError: dspy`

- Install the package:

```bash
pip install dspy-ai openai python-dotenv
```

### `foodDB.json` not found

- Run the notebook from the `chapter_03_prompting/` directory
- Make sure [foodDB.json](/home/sai/Dev/Mastering-Agentic-AI/chapter_03_prompting/foodDB.json) exists beside the notebook and script

---

Minimal configuration keeps the chapter focused on prompting logic rather than environment debugging.
