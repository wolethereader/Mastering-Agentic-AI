# Appendix A · API Keys and Environment Setup
## Chapter 2 · Building Agents with CrewAI

*Mastering Agentic AI — Manning Publications*

Chapter 2 uses the same diet coach use case in two forms:

1. A CrewAI agent team.
2. A plain LLM pipeline built with direct OpenAI API calls.

Both versions use the same model backend, so the setup is intentionally simple.

---

## What Chapter 2 Needs

| Key | Required? | Used for |
|-----|-----------|----------|
| `OPENAI_API_KEY` | Required | CrewAI agents and raw OpenAI API calls |
| `OPENAI_MODEL` | Optional | Override the default model (`gpt-4o-mini`) |

---

## Step 1 · Create an OpenAI API Key

1. Go to [platform.openai.com](https://platform.openai.com)
2. Sign in to your account
3. Open the API keys page
4. Create a new secret key
5. Copy it immediately

If you are using a fresh account, confirm that billing or trial credits are available before running the chapter.

---

## Step 2 · Store the Key in a `.env` File

Create a file named `.env` inside `chapter_02_hands_on/`:

```bash
OPENAI_API_KEY=sk-proj-YOUR-KEY-HERE
OPENAI_MODEL=gpt-4o-mini
```

Then load it in Python:

```python
from dotenv import load_dotenv
load_dotenv()
```

Do not commit `.env` to version control.

---

## Step 3 · Install Dependencies

```bash
pip install crewai crewai-tools openai python-dotenv
```

The chapter script imports CrewAI for the agent workflow and the official OpenAI SDK for the no-framework baseline.

---

## Step 4 · Verify the Setup

```python
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

assert os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY is not set"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
response = client.chat.completions.create(
    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    temperature=0.2,
    messages=[{"role": "user", "content": "Say hello in one sentence."}],
)
print(response.choices[0].message.content)
```

If this succeeds, the OpenAI portion of the chapter is ready.

---

## Common Issues

### `OPENAI_API_KEY is not set`

- The `.env` file is missing
- `load_dotenv()` was not called
- The notebook is running from a different working directory than expected

### `AuthenticationError`

- The key was copied incorrectly
- The key was revoked
- The account has no active billing or credit

### `ModuleNotFoundError: crewai` or `ModuleNotFoundError: openai`

- Install the required packages:

```bash
pip install crewai crewai-tools openai python-dotenv
```

### CrewAI starts but model calls fail

- Confirm `OPENAI_API_KEY` is visible in the same shell or notebook kernel
- Confirm the selected model exists on your account
- If needed, set `OPENAI_MODEL=gpt-4o-mini`

---

## Minimal `.env` Template

```bash
# chapter_02_hands_on/.env
OPENAI_API_KEY=sk-proj-YOUR-KEY-HERE
OPENAI_MODEL=gpt-4o-mini
```

---

Good credential habits are part of agent engineering. The chapter is about orchestration, but safe configuration is what makes the examples reusable outside the notebook.
