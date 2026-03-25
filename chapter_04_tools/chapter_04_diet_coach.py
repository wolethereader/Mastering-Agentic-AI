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
import os
import sys
import inspect
from pathlib import Path
from typing import Any, Callable, get_type_hints

try:
    import anyio
    from mcp import ClientSession, StdioServerParameters, stdio_client
    from mcp.server import FastMCP
    from mcp.shared.memory import create_connected_server_and_client_session
except ImportError:
    anyio = None
    ClientSession = None
    StdioServerParameters = None
    stdio_client = None
    FastMCP = None
    create_connected_server_and_client_session = None

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    from composio import Composio
    from composio_crewai import CrewAIProvider
except ImportError:
    Composio = None
    CrewAIProvider = None

try:
    from crewai import Agent, Crew, Task
except ImportError:
    Agent = None
    Crew = None
    Task = None

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None

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
# 4.4  Building an MCP Client (official SDK, stdio transport)
# ─────────────────────────────────────────────────────────────────────────────

MCP_SERVER_FILENAME = "diet_coach_mcp_server.py"


def _require_mcp_sdk():
    if anyio is None or ClientSession is None or StdioServerParameters is None or stdio_client is None:
        raise RuntimeError(
            "Install MCP SDK first: pip install mcp\n"
            "This chapter's MCP client/server examples use the official SDK."
        )


def _tool_to_dict(tool: Any) -> dict[str, Any]:
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.inputSchema,
    }


async def list_mcp_tools_async(server_script_path: str) -> list[dict[str, Any]]:
    """
    Section 4.4: Connect to an MCP server over stdio using the
    official SDK, perform the required handshake, then discover tools.
    """
    _require_mcp_sdk()

    server = StdioServerParameters(
        command=sys.executable,
        args=[server_script_path],
        cwd=str(Path(server_script_path).parent),
    )

    async with stdio_client(server) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.list_tools()
            return [_tool_to_dict(tool) for tool in result.tools]


async def call_mcp_tool_async(
    server_script_path: str,
    tool_name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """
    Section 4.4: Call an MCP tool over stdio using the official client.
    """
    _require_mcp_sdk()

    server = StdioServerParameters(
        command=sys.executable,
        args=[server_script_path],
        cwd=str(Path(server_script_path).parent),
    )

    async with stdio_client(server) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments)
            text_blocks = [block.text for block in result.content if getattr(block, "type", None) == "text"]
            return {
                "is_error": result.isError,
                "content": text_blocks,
                "structured_content": result.structuredContent,
            }


def list_mcp_tools(server_script_path: str) -> list[dict[str, Any]]:
    _require_mcp_sdk()
    return anyio.run(list_mcp_tools_async, server_script_path)


def call_mcp_tool(
    server_script_path: str,
    tool_name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    _require_mcp_sdk()
    return anyio.run(call_mcp_tool_async, server_script_path, tool_name, arguments)


# ─────────────────────────────────────────────────────────────────────────────
# 4.5  Building an MCP Server
# ─────────────────────────────────────────────────────────────────────────────


def build_mcp_server():
    """
    Section 4.5: Build a real MCP server using the official SDK.

    We keep the same nutrition domain as earlier chapters, but now expose it
    through the MCP standard so any MCP-compatible host can discover and call
    the tool at runtime.
    """
    _require_mcp_sdk()

    server = FastMCP(
        "diet-coach-nutrition",
        instructions="Nutrition lookup tools for the AI Diet Coach.",
    )

    @server.tool(
        name="lookup_nutrition",
        description="Return macro-nutrient data for a food item per 100 g serving.",
        structured_output=True,
    )
    def lookup_nutrition_mcp(food_item: str) -> dict[str, Any]:
        key = food_item.strip().lower()
        data = NUTRITION_DB.get(key)
        if data:
            return {"food": key, **data}

        matches = [k for k in NUTRITION_DB if key in k or k in key]
        if matches:
            match = matches[0]
            return {"food": match, "note": f"closest match for '{food_item}'", **NUTRITION_DB[match]}

        return {"error": f"'{food_item}' not found"}

    return server


MCP_SERVER_CODE = '''#!/usr/bin/env python3
"""
diet_coach_mcp_server.py
A real MCP server that exposes the AI Diet Coach's nutrition tools.
Run standalone: python diet_coach_mcp_server.py

Install SDK first: pip install mcp
"""

import json
from mcp.server import FastMCP

NUTRITION_DB = {
    "apple":          {"calories": 95,  "protein_g": 0.5,  "carbs_g": 25, "fat_g": 0.3, "fibre_g": 4.4},
    "chicken breast": {"calories": 165, "protein_g": 31.0, "carbs_g": 0,  "fat_g": 3.6, "fibre_g": 0},
    "brown rice":     {"calories": 216, "protein_g": 5.0,  "carbs_g": 45, "fat_g": 1.8, "fibre_g": 3.5},
    "broccoli":       {"calories": 55,  "protein_g": 3.7,  "carbs_g": 11, "fat_g": 0.6, "fibre_g": 5.1},
    "salmon":         {"calories": 208, "protein_g": 28.0, "carbs_g": 0,  "fat_g": 10,  "fibre_g": 0},
    "oats":           {"calories": 150, "protein_g": 5.0,  "carbs_g": 27, "fat_g": 2.5, "fibre_g": 4.0},
}

server = FastMCP(
    "diet-coach-nutrition",
    instructions="Nutrition lookup tools for the AI Diet Coach.",
)

@server.tool(
    name="lookup_nutrition",
    description="Return macro-nutrient data for a food item per 100 g serving.",
    structured_output=True,
)
def lookup_nutrition(food_item: str) -> dict:
    key = food_item.strip().lower()
    data = NUTRITION_DB.get(key)
    if data:
        return {"food": key, **data}

    matches = [k for k in NUTRITION_DB if key in k or k in key]
    if matches:
        match = matches[0]
        return {"food": match, "note": f"closest match for '{food_item}'", **NUTRITION_DB[match]}

    return {"error": f"'{food_item}' not found"}

if __name__ == "__main__":
    server.run(transport="stdio")
'''


def write_mcp_server():
    """Write the MCP server file to disk so it can be run independently."""
    path = Path(__file__).with_name(MCP_SERVER_FILENAME)
    with open(path, "w") as f:
        f.write(MCP_SERVER_CODE)
    print(f"MCP server written to {path}")
    print("Install MCP SDK: pip install mcp")
    print(f"Run server:       python {path}")


async def verify_mcp_locally_async() -> dict[str, Any]:
    """
    Verify the MCP server end-to-end without spawning a subprocess.

    This still uses the official MCP protocol, but connects the client and
    server through in-memory transport so the chapter demo remains reliable.
    """
    _require_mcp_sdk()
    server = build_mcp_server()

    async with create_connected_server_and_client_session(server) as session:
        tools = await session.list_tools()
        result = await session.call_tool("lookup_nutrition", {"food_item": "oats"})
        text_blocks = [block.text for block in result.content if getattr(block, "type", None) == "text"]
        return {
            "tools": [_tool_to_dict(tool) for tool in tools.tools],
            "tool_call": {
                "is_error": result.isError,
                "content": text_blocks,
                "structured_content": result.structuredContent,
            },
        }


def verify_mcp_locally() -> dict[str, Any]:
    _require_mcp_sdk()
    return anyio.run(verify_mcp_locally_async)


# ─────────────────────────────────────────────────────────────────────────────
# 4.6  Composio and Other Tooling Ecosystems
# ─────────────────────────────────────────────────────────────────────────────

COMPOSIO_EXPLAINER = """
COMPOSIO — Managed Tooling Ecosystem

Where MCP standardises HOW tools are exposed, Composio helps you
ACCESS many real-world integrations without wiring each API yourself.

Typical workflow:
  1. Create or reuse an auth config for a toolkit (e.g. GitHub)
  2. Link a user account to that toolkit
  3. Fetch provider-formatted tools for your framework
  4. Give those tools to an agent

This reduces boilerplate around authentication, schemas, and execution.
"""


def _require_composio_sdk():
    if Composio is None or CrewAIProvider is None:
        raise RuntimeError(
            "Install Composio first: pip install composio composio_crewai\n"
            "This chapter's Composio example uses the current provider-based SDK."
        )


def create_composio_github_auth_link(
    user_id: str,
    auth_config_id: str,
    callback_url: str | None = None,
) -> str:
    """
    Create a Composio connection link so a user can connect GitHub.

    Args:
        user_id: Your application's user identifier.
        auth_config_id: The Composio auth config ID (for example: ac_xxx).
        callback_url: Optional redirect URL after the auth flow completes.

    Returns:
        str: A redirect URL the user can open to complete authentication.
    """
    _require_composio_sdk()

    composio = Composio()
    request = composio.connected_accounts.link(
        user_id=user_id,
        auth_config_id=auth_config_id,
        callback_url=callback_url,
    )
    return request.redirect_url


def get_composio_github_tools(user_id: str = "default") -> list[Any]:
    """
    Fetch GitHub tools from Composio, already formatted for CrewAI.

    Note:
        The user must have an authenticated GitHub connection in Composio.
    """
    _require_composio_sdk()

    composio = Composio(provider=CrewAIProvider())
    return composio.tools.get(
        user_id=user_id,
        toolkits=["GITHUB"],
    )


def build_composio_github_crew(user_id: str = "default") -> Any:
    """
    Build a small CrewAI example using Composio's GitHub toolkit.

    This is optional chapter code: it requires Composio auth plus an OpenAI
    key for the CrewAI LLM wrapper shown in current Composio docs.
    """
    _require_composio_sdk()

    if Agent is None or Crew is None or Task is None:
        raise RuntimeError("Install CrewAI first: pip install crewai")

    if ChatOpenAI is None:
        raise RuntimeError("Install langchain-openai first: pip install langchain-openai")

    tools = get_composio_github_tools(user_id=user_id)
    llm = ChatOpenAI()

    github_agent = Agent(
        role="GitHub Research Agent",
        goal="Understand recent activity in a GitHub repository",
        backstory=(
            "An expert developer agent that inspects repositories, "
            "reads commits and pull requests, and summarises changes."
        ),
        verbose=True,
        tools=tools,
        llm=llm,
    )

    task = Task(
        description=(
            "Inspect the latest activity in a GitHub repository and produce "
            "a short summary of what changed."
        ),
        expected_output="A concise summary of recent repository activity.",
        agent=github_agent,
    )

    return Crew(agents=[github_agent], tasks=[task])


# ─────────────────────────────────────────────────────────────────────────────
# Tools vs Skills — the core contrast of this chapter
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

SKILL_PATH = Path(__file__).parent.parent / "chapter_03_prompting" / "SKILL.md"

TOOLS = [TOOL_LOOKUP_AUTO, tool_from_function(suggest_meal)]
TOOL_FNS = {"lookup_nutrition": lookup_nutrition, "suggest_meal": suggest_meal}


def run_skill_guided_agent(user_message: str) -> str:
    """
    The Diet Coach using BOTH the SKILL (system prompt) and TOOLS (API).
    This is the fullest version of the coach so far.
    """
    if anthropic is None:
        raise RuntimeError("Install anthropic first: pip install anthropic")

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

    print("\n── MCP Verification (local protocol round-trip) ─────────────")
    try:
        verification = verify_mcp_locally()
        print(json.dumps(verification, indent=2))
    except Exception as exc:
        print(f"MCP demo skipped: {exc}")

    print("\n── MCP Server (write to disk) ────────────────────────────────")
    write_mcp_server()

    print("\n── Composio Toolkit Example ─────────────────────────────────")
    print(COMPOSIO_EXPLAINER)
    if Composio is not None and CrewAIProvider is not None and os.getenv("COMPOSIO_API_KEY"):
        github_user_id = os.getenv("COMPOSIO_USER_ID", "default")
        try:
            github_tools = get_composio_github_tools(user_id=github_user_id)
            print(f"Fetched {len(github_tools)} GitHub tool(s) for user_id='{github_user_id}'.")
            print("You can now attach these tools to a CrewAI agent with build_composio_github_crew(...).")
        except Exception as exc:
            print(f"Composio demo skipped: {exc}")
    else:
        print("Skipping Composio demo. Install `composio` + `composio_crewai` and set `COMPOSIO_API_KEY`.")

    if anthropic is not None and os.getenv("ANTHROPIC_API_KEY"):
        print("\n── Chapter 4 Agent: Skill-Guided Tool Use ───────────────────")
        answer = run_skill_guided_agent(
            "I want to build more muscle. What should I eat for dinner tonight? "
            "I want something under 500 calories and high in protein."
        )
        print(f"Coach: {answer}")
    else:
        print("\n── Chapter 4 Agent: Skill-Guided Tool Use ───────────────────")
        print("Skipping agent demo. Install `anthropic` and set `ANTHROPIC_API_KEY` to run it.")
