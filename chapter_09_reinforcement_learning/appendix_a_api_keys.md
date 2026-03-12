# Appendix A · API Keys and Environment Setup
## Chapter 1 · The AI Diet Coach: A Simple Conversational Agent

*Mastering Agentic AI — Manning Publications*

Treat credential management as a design decision, not an afterthought. A system that cannot be safely handed to a teammate is not production-ready.

---

## What Chapter 1 Needs

Chapter 1 requires only one API key:

| Key | Where to get it | Used for |
|-----|----------------|----------|
| `ANTHROPIC_API_KEY` | console.anthropic.com | Claude API (the Diet Coach LLM) |

---

## Step 1 · Get Your Anthropic API Key

1. Go to [console.anthropic.com](https://console.anthropic.com)
2. Sign in or create a free account
3. Navigate to **API Keys** in the left sidebar
4. Click **Create Key** — give it a descriptive name (e.g., `mastering-agentic-ai-ch1`)
5. Copy the key immediately — it is shown only once

**Free tier:** Anthropic provides starter credits. Chapter 1 uses minimal tokens (the conversational loop is short), so the free tier comfortably covers all examples.

---

## Step 2 · Store the Key Safely

### Option A: `.env` file (recommended for notebooks)

Create a file named `.env` in this chapter directory:

```
ANTHROPIC_API_KEY=sk-ant-api03-YOUR-KEY-HERE
```

Then in your notebook:

```python
from dotenv import load_dotenv
load_dotenv()  # reads .env and sets environment variables
```

> **Important:** Add `.env` to your `.gitignore` immediately. Never commit API keys to version control.

### Option B: Shell environment variable

```bash
export ANTHROPIC_API_KEY=sk-ant-api03-YOUR-KEY-HERE
```

Add this to your `~/.bashrc`, `~/.zshrc`, or `~/.bash_profile` to persist it across terminal sessions.

### Option C: IDE settings

VS Code, PyCharm, and Jupyter Lab all support environment variable configuration through their settings or launch configurations. Refer to your IDE documentation.

---

## Step 3 · Verify the Setup

```python
import os
from anthropic import Anthropic

assert os.getenv("ANTHROPIC_API_KEY"), "ANTHROPIC_API_KEY is not set"

client = Anthropic()
response = client.messages.create(
    model="claude-opus-4-5",
    max_tokens=50,
    messages=[{"role": "user", "content": "Say hello in one sentence."}]
)
print(response.content[0].text)
# Expected: a short greeting confirming the API connection works
```

If this runs without errors, you are ready to work through the entire chapter.

---

## Common Issues

### `AuthenticationError: invalid x-api-key`
- The key was copied incorrectly — copy it again from the console
- The key was revoked — generate a new one
- There is a leading/trailing space in the `.env` file — remove it

### `APIConnectionError`
- Check your internet connection
- If behind a corporate proxy, set `HTTPS_PROXY` environment variable

### `RateLimitError`
- You have exceeded the free tier rate limit
- Wait 60 seconds and retry
- Consider upgrading to a paid tier for uninterrupted development

---

## Model Used in Chapter 1

```python
MODEL = "claude-opus-4-5"
```

This is Claude's most capable model. If you want to reduce costs during experimentation, substitute `"claude-haiku-4-5-20251001"` — it is significantly faster and cheaper, and perfectly adequate for Chapter 1 exercises.

---

## What Chapter 2 Adds

Chapter 2 introduces CrewAI. Some CrewAI configurations require an **OpenAI API key** depending on the LLM backend you choose. See `chapter_02_hands_on/appendix_a_api_keys.md` for details.

---

"The habit of storing credentials in environment variables rather than code is one of the most valuable engineering habits you can build. It costs you thirty seconds and saves you from disasters."


## Chapter 5 · Memory Systems for Agents

---

## What Chapter 5 Needs

| Key | Required? | Where to get it | Used for |
|-----|-----------|----------------|----------|
| `ANTHROPIC_API_KEY` | **Required** | console.anthropic.com | Claude API (all memory sections) |
| `PINECONE_API_KEY` | Optional | pinecone.io | Production vector store (upgrade path) |
| `REDIS_URL` | Optional | redis.io / Upstash | Production semantic memory store |

Chapter 5 runs entirely from disk (JSON files) by default. No additional keys are required. The Pinecone and Redis notes below describe the production upgrade path discussed in the chapter.

---

## Step 1 · Anthropic API Key

```bash
ANTHROPIC_API_KEY=sk-ant-api03-YOUR-KEY-HERE
```

See `chapter_01_introduction/appendix_a_api_keys.md` for full setup walkthrough.

---

## Step 2 · Local Memory Store (Default — No Keys Needed)

Chapter 5 uses JSON files on disk by default:

```
.memory/
  alex_semantic.json    ← user profile facts
  alex_episodes.json    ← session summaries
```

These are created automatically when `SemanticMemory` and `EpisodicMemory` are instantiated. No configuration required.

To reset memory between notebook runs:

```python
import shutil
shutil.rmtree(".memory", ignore_errors=True)
print("Memory cleared.")
```

---

## Step 3 · Production Upgrade — Pinecone (Optional)

When your episode store grows beyond ~1,000 entries, keyword retrieval becomes inadequate. The production upgrade uses vector embeddings + Pinecone:

```bash
pip install pinecone-client sentence-transformers
```

```bash
# .env
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_ENVIRONMENT=us-east-1-aws   # from your Pinecone dashboard
```

Setup:
1. Create a free account at [pinecone.io](https://pinecone.io)
2. Create an index with dimension=384 (matches `all-MiniLM-L6-v2`)
3. Copy your API key from the Pinecone dashboard

The `EpisodicMemory.retrieve()` method in the chapter shows the interface that the production version must conform to. Only the backend changes.

---

## Step 4 · Production Upgrade — Redis (Optional)

For semantic memory (user profile facts) at scale:

```bash
pip install redis
```

Free managed Redis: [Upstash](https://upstash.com) — 10,000 commands/day free tier.

```bash
REDIS_URL=redis://default:YOUR_PASSWORD@your-endpoint.upstash.io:PORT
```

```python
import redis, os
r = redis.from_url(os.getenv("REDIS_URL"))
r.set("alex:weight_kg", "78")
print(r.get("alex:weight_kg"))  # b'78'
```

---

## Step 5 · Full .env Template for Chapter 5

```bash
# chapter_05_memory/.env

# Required
ANTHROPIC_API_KEY=sk-ant-api03-YOUR-KEY-HERE

# Optional: production vector store (Section 5.3 upgrade path)
PINECONE_API_KEY=your-pinecone-key
PINECONE_ENVIRONMENT=us-east-1-aws

# Optional: production key-value store for semantic memory
REDIS_URL=redis://default:password@endpoint:port
```

---

## Step 6 · Install Dependencies

```bash
pip install anthropic python-dotenv

# Optional production upgrades
pip install pinecone-client sentence-transformers redis
```

---

## Memory File Locations

The notebook creates files in `.memory/` relative to the working directory. If running from a different directory than the notebook, specify the absolute path:

```python
from pathlib import Path
store = Path(__file__).parent / ".memory"
mem = SemanticMemory("alex", store_path=store)
```

---

## Common Issues

### `FileNotFoundError` on `.memory/` directory
The directory is created automatically on first `SemanticMemory` or `EpisodicMemory` instantiation. If it fails, check write permissions: `ls -la .` and ensure the notebook working directory is writable.

### Memory persists between notebook runs (unexpected)
This is intentional — persistence is the point of Chapter 5. To start fresh: `shutil.rmtree(".memory", ignore_errors=True)`.

### Retrieval returning irrelevant episodes
The chapter uses keyword overlap, which is coarse. Add more specific keywords to your `query` argument, or switch to the Pinecone vector upgrade for semantic similarity.

---

"Start with the simplest possible memory store. A JSON file teaches you the interface. Once you understand what you need from a memory system, choosing the right backend becomes much easier."*

