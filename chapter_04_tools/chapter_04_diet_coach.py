"""
Mastering Agentic AI — Chapter 4
Tools and Interoperability

Sections covered:
  4.1  What Are Tools and How Do They Work Internally
  4.2  Ways of Building a Tool
  4.3  What is MCP (Model Context Protocol)
  4.4  Building an MCP Client
  4.5  Building an MCP Server
  4.6  Composio and Other Tooling Ecosystems

A tool is just a function with a description the
model can read. The genius of tool-use is not the API call — it is that
the model decides *when* and *how* to call it.

KEY CONTRAST IN THIS CHAPTER:
  • Chapter 2 tools    → give the coach DATABASE ACCESS (declarative: what)
  • Chapter 3 SKILL.md → gives the coach JUDGMENT (procedural: how)

Neither is sufficient alone. A coach with tools but no skill will look
up every food with no coherent strategy. A coach with skill but no tools
will give advice it cannot ground in data. Together they create a
capable, trustworthy agent.
"""

import json
import asyncio
import inspect
from typing import Any, Callable, get_type_hints
import anthropic

# ─────────────────────────────────────────────────────────────────────────────
# 4.1  What Are Tools and How Do They Work Internally
# ─────────────────────────────────────────────────────────────────────────────

"""
Under the hood, tool-use in Claude works as follows:

  1. The API receives your `tools` parameter — a JSON Schema array.
  2. During the forward pass the model decides to emit a `tool_use` block
     instead of (or alongside) a `text` block.
  3. The API returns stop_reason="tool_use".
  4. YOUR code executes the function and returns the result.
  5. You append a `tool_result` block and call the API again.
  6. Repeat until stop_reason="end_turn".

The model never directly executes code. It only *describes* what it
wants to call. You are the runtime.
"""

# ─────────────────────────────────────────────────────────────────────────────
# 4.2  Ways of Building a Tool
# ─────────────────────────────────────────────────────────────────────────────

# ── Approach A: Manual JSON Schema (most explicit, most control) ──────────────

TOOL_NUTRITION_LOOKUP_MANUAL = {
    "name": "lookup_nutrition",
    "description": (
        "Return macro-nutrient data (calories, protein, carbs, fat, fibre) "
        "for a given food item per 100 g serving."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "food_item": {
                "type": "string",
                "description": "Common name of the food (e.g. 'chicken breast', 'oats')",
            }
        },
        "required": ["food_item"],
    },
}


# ── Approach B: Auto-generate schema from Python type hints ──────────────────

def _py_type_to_json(py_type: type) -> dict:
    mapping = {str: "string", int: "integer", float: "number", bool: "boolean"}
    return {"type": mapping.get(py_type, "string")}


def tool_from_function(fn: Callable) -> dict:
    """
    Section 4.2: Generate a tool schema from a Python function's
    signature and docstring. Works for simple scalar parameters.
    """
    hints = get_type_hints(fn)
    sig   = inspect.signature(fn)
    props: dict = {}
    required: list[str] = []

    for name, param in sig.parameters.items():
        annotation = hints.get(name, str)
        props[name] = _py_type_to_json(annotation)
        props[name]["description"] = name.replace("_", " ").capitalize()
        if param.default is inspect.Parameter.empty:
            required.append(name)

    return {
        "name": fn.__name__,
        "description": (fn.__doc__ or "").strip(),
        "input_schema": {"type": "object", "properties": props, "required": required},
    }


# ── Example tool functions (reusing Chapter 2 DB) ────────────────────────────
NUTRITION_DB: dict[str, dict] = {
    "apple":          {"calories": 95,  "protein_g": 0.5,  "carbs_g": 25, "fat_g": 0.3, "fibre_g": 4.4},
    "chicken breast": {"calories": 165, "protein_g": 31.0, "carbs_g": 0,  "fat_g": 3.6, "fibre_g": 0},
    "brown rice":     {"calories": 216, "protein_g": 5.0,  "carbs_g": 45, "fat_g": 1.8, "fibre_g": 3.5},
    "broccoli":       {"calories": 55,  "protein_g": 3.7,  "carbs_g": 11, "fat_g": 0.6, "fibre_g": 5.1},
    "salmon":         {"calories": 208, "protein_g": 28.0, "carbs_g": 0,  "fat_g": 10,  "fibre_g": 0},
    "oats":           {"calories": 150, "protein_g": 5.0,  "carbs_g": 27, "fat_g": 2.5, "fibre_g": 4.0},
}


def lookup_nutrition(food_item: str) -> str:
    """Look up macro-nutrient data for a food item per 100 g serving."""
    key = food_item.strip().lower()
    data = NUTRITION_DB.get(key)
    if data:
        return json.dumps({"food": key, **data})
    matches = [k for k in NUTRITION_DB if key in k or k in key]
    if matches:
        return json.dumps({"food": matches[0], "note": f"closest match for '{food_item}'", **NUTRITION_DB[matches[0]]})
    return json.dumps({"error": f"'{food_item}' not found"})


def suggest_meal(goal: str, max_calories: int = 600) -> str:
    """Suggest a meal from the database that fits the user's goal and calorie cap."""
    suitable = [
        {"food": k, **v}
        for k, v in NUTRITION_DB.items()
        if v["calories"] <= max_calories
    ]
    if goal.lower() == "high protein":
        suitable.sort(key=lambda x: x["protein_g"], reverse=True)
    elif goal.lower() == "low carb":
        suitable.sort(key=lambda x: x["carbs_g"])

    top = suitable[:3]
    return json.dumps({"goal": goal, "suggestions": top})


# Auto-generate schemas
TOOL_LOOKUP_AUTO    = tool_from_function(lookup_nutrition)
TOOL_SUGGEST_AUTO   = tool_from_function(suggest_meal)


# ─────────────────────────────────────────────────────────────────────────────
# 4.3  What is MCP (Model Context Protocol)
# ─────────────────────────────────────────────────────────────────────────────

"""
MCP is an open standard (Anthropic, 2024) that decouples tool *servers*
from tool *clients*. Key ideas:

  • Server  — exposes capabilities (tools, resources, prompts) over a
              standardised JSON-RPC 2.0 transport (stdio or HTTP/SSE).
  • Client  — discovers capabilities at runtime by calling
              `tools/list`, then invokes them with `tools/call`.
  • Benefit — agents can connect to any MCP server without knowing
              implementation details in advance.

Think of MCP as USB-C for agent tools: one standard plug, many devices.
"""

# ─────────────────────────────────────────────────────────────────────────────
# 4.4  Building an MCP Client (minimal, stdio transport)
# ─────────────────────────────────────────────────────────────────────────────

import subprocess


class SimpleMCPClient:
    """
    Section 4.4: A minimal MCP client over stdio transport.

    In production use the official `mcp` Python SDK:
        pip install mcp
    This implementation is for pedagogical clarity.
    """

    def __init__(self, server_command: list[str]):
        self.proc = subprocess.Popen(
            server_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        self._id = 0

    def _rpc(self, method: str, params: dict | None = None) -> dict:
        self._id += 1
        request = json.dumps({"jsonrpc": "2.0", "id": self._id, "method": method, "params": params or {}})
        self.proc.stdin.write(request + "\n")      # type: ignore
        self.proc.stdin.flush()                    # type: ignore
        raw = self.proc.stdout.readline()          # type: ignore
        return json.loads(raw).get("result", {})

    def list_tools(self) -> list[dict]:
        return self._rpc("tools/list").get("tools", [])

    def call_tool(self, name: str, arguments: dict) -> Any:
        return self._rpc("tools/call", {"name": name, "arguments": arguments})

    def close(self):
        self.proc.terminate()


# ─────────────────────────────────────────────────────────────────────────────
# 4.5  Building an MCP Server
# ─────────────────────────────────────────────────────────────────────────────

MCP_SERVER_CODE = '''#!/usr/bin/env python3
"""
diet_coach_mcp_server.py
A minimal MCP server that exposes the AI Diet Coach's nutrition tools.
Run standalone: python diet_coach_mcp_server.py

Install SDK first: pip install mcp
"""

import json
import sys
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

NUTRITION_DB = {
    "apple":          {"calories": 95,  "protein_g": 0.5,  "carbs_g": 25},
    "chicken breast": {"calories": 165, "protein_g": 31.0, "carbs_g": 0},
    "oats":           {"calories": 150, "protein_g": 5.0,  "carbs_g": 27},
    "salmon":         {"calories": 208, "protein_g": 28.0, "carbs_g": 0},
}

server = Server("diet-coach-nutrition")

@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="lookup_nutrition",
            description="Return macro-nutrient data for a food item.",
            inputSchema={
                "type": "object",
                "properties": {
                    "food_item": {"type": "string", "description": "Food name"}
                },
                "required": ["food_item"],
            },
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    if name == "lookup_nutrition":
        food = arguments.get("food_item", "").lower()
        data = NUTRITION_DB.get(food, {"error": f"Not found: {food}"})
        return [types.TextContent(type="text", text=json.dumps(data))]
    raise ValueError(f"Unknown tool: {name}")

async def main():
    async with stdio_server() as (r, w):
        await server.run(r, w, server.create_initialization_options())

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
'''


def write_mcp_server():
    """Write the MCP server file to disk so it can be run independently."""
    path = "diet_coach_mcp_server.py"
    with open(path, "w") as f:
        f.write(MCP_SERVER_CODE)
    print(f"MCP server written to {path}")
    print("Install MCP SDK: pip install mcp")
    print(f"Run server:       python {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 4.6  Tools vs Skills — the core contrast of this chapter
# ─────────────────────────────────────────────────────────────────────────────

SKILL_VS_TOOL_EXPLAINER = """
TOOLS (Chapter 2) vs SKILLS (Chapter 3) — Side-by-Side

┌──────────────────────┬────────────────────────────────┬────────────────────────────────────┐
│                      │ TOOL (lookup_nutrition)        │ SKILL (SKILL.md protocol)          │
├──────────────────────┼────────────────────────────────┼────────────────────────────────────┤
│ Nature               │ Declarative — what to call     │ Procedural — how to think          │
│ Defined as           │ JSON Schema + Python function  │ Natural language instructions      │
│ Executed by          │ Your runtime (you call the fn) │ The model (it follows the steps)   │
│ Gives the agent      │ Access to external data        │ Judgment about using that data     │
│ Example              │ look up "chicken breast" → 165 cal │ Assess baseline first, then deficits │
│ Without the other    │ Floods user with raw numbers   │ Gives advice with no data to back it│
│ Together             │ Grounded, structured, strategic nutrition coaching                  │
└──────────────────────┴────────────────────────────────┴────────────────────────────────────┘

The SKILL tells the coach WHEN to call the tool and WHAT to do with the result.
The TOOL gives the coach something real to say.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Full agent: SKILL-guided tool use
# ─────────────────────────────────────────────────────────────────────────────

from pathlib import Path

SKILL_PATH = Path(__file__).parent.parent / "chapter_03_prompting" / "SKILL.md"

TOOLS = [TOOL_LOOKUP_AUTO, tool_from_function(suggest_meal)]
TOOL_FNS = {"lookup_nutrition": lookup_nutrition, "suggest_meal": suggest_meal}


def run_skill_guided_agent(user_message: str) -> str:
    """
    The Diet Coach using BOTH the SKILL (system prompt) and TOOLS (API).
    This is the fullest version of the coach so far.
    """
    client = anthropic.Anthropic()

    skill_text = SKILL_PATH.read_text() if SKILL_PATH.exists() else ""

    system = (
        "You are an AI Diet Coach.\n\n"
        f"[Nutrition Assessment Skill]\n{skill_text}\n\n"
        "Use your tools to look up nutritional data and suggest meals. "
        "Apply your Skill protocol when conducting full assessments."
    )

    messages = [{"role": "user", "content": user_message}]

    while True:
        response = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=1024,
            system=system,
            tools=TOOLS,
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            return " ".join(b.text for b in response.content if b.type == "text")

        if response.stop_reason != "tool_use":
            break

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                fn = TOOL_FNS.get(block.name)
                result = fn(**block.input) if fn else json.dumps({"error": "unknown tool"})
                tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": result})

        messages.append({"role": "user", "content": tool_results})

    return "[loop ended unexpectedly]"


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(SKILL_VS_TOOL_EXPLAINER)

    print("\n── Auto-generated tool schemas ──────────────────────────────")
    print(json.dumps(TOOL_LOOKUP_AUTO, indent=2))

    print("\n── Chapter 4 Agent: Skill-Guided Tool Use ───────────────────")
    answer = run_skill_guided_agent(
        "I want to build more muscle. What should I eat for dinner tonight? "
        "I want something under 500 calories and high in protein."
    )
    print(f"Coach: {answer}")

    print("\n── MCP Server (write to disk) ────────────────────────────────")
    write_mcp_server()
