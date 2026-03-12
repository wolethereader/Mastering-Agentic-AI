"""
Mastering Agentic AI — Chapter 7
Agent Evaluation Fundamentals — Metrics, Patterns, and Practice

Sections covered:
  7.1  Why Evaluating Agents Is Different from Evaluating Models
  7.2  Metrics for Single-Agent Systems
  7.3  Evaluation as a System Property
  7.4  Single-Agent Evaluation in Practice
  7.5  Multi-Agent Evaluation Fundamentals
  7.6  Agentic Design Patterns and Their Evaluation Implications
  7.7  Full-Stack Agentic Application

You cannot improve what you cannot measure.
Agent evaluation is harder than model evaluation because the output
is a trajectory of decisions, not a single text. Measure the process,
not just the result.
"""

import json
import time
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable
import anthropic

# ─────────────────────────────────────────────────────────────────────────────
# 7.2  Metrics for Single-Agent Systems
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AgentTrace:
    """Records every step of an agent's execution for evaluation."""
    task_id:        str
    user_input:     str
    tool_calls:     list[dict] = field(default_factory=list)
    final_response: str = ""
    latency_ms:     float = 0.0
    token_count:    int = 0
    succeeded:      bool = False
    error:          str | None = None


@dataclass
class EvalMetrics:
    """Aggregate evaluation metrics for a batch of traces."""
    task_success_rate:     float   # % of tasks fully completed
    tool_call_accuracy:    float   # correct tool called / total tool calls
    avg_latency_ms:        float
    avg_token_count:       float
    goal_adherence_score:  float   # LLM-as-judge score 0–1
    hallucination_rate:    float   # detected factual errors / responses


def compute_metrics(traces: list[AgentTrace]) -> EvalMetrics:
    n = len(traces)
    if n == 0:
        return EvalMetrics(0, 0, 0, 0, 0, 0)

    return EvalMetrics(
        task_success_rate    = sum(t.succeeded for t in traces) / n,
        tool_call_accuracy   = _tool_accuracy(traces),
        avg_latency_ms       = statistics.mean(t.latency_ms for t in traces),
        avg_token_count      = statistics.mean(t.token_count for t in traces),
        goal_adherence_score = _goal_adherence(traces),
        hallucination_rate   = 0.0,   # requires ground-truth DB; placeholder
    )


def _tool_accuracy(traces: list[AgentTrace]) -> float:
    total = correct = 0
    for trace in traces:
        for call in trace.tool_calls:
            total += 1
            if call.get("correct", True):   # annotated in test harness
                correct += 1
    return correct / total if total else 1.0


def _goal_adherence(traces: list[AgentTrace]) -> float:
    """Placeholder — replaced by LLM-as-Judge in section 7.7."""
    return statistics.mean(t.succeeded for t in traces)


# ─────────────────────────────────────────────────────────────────────────────
# 7.4  Single-Agent Evaluation in Practice — test harness
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvalCase:
    """A single evaluation test case."""
    task_id:         str
    user_input:      str
    expected_tools:  list[str]          # tools that MUST be called
    success_criteria: str               # natural language description for LLM judge
    reference_output: str | None = None # optional gold standard


DIET_COACH_EVAL_SUITE: list[EvalCase] = [
    EvalCase(
        task_id="TC-001",
        user_input="What are the macros in 200g of chicken breast?",
        expected_tools=["lookup_nutrition"],
        success_criteria="Response must include specific protein, carbs, and fat values for chicken breast.",
    ),
    EvalCase(
        task_id="TC-002",
        user_input="Log my lunch: brown rice 150g and broccoli 100g, then tell me my fibre for today.",
        expected_tools=["log_meal", "log_meal", "get_daily_summary"],
        success_criteria="Both foods must be logged AND a daily summary with fibre content provided.",
    ),
    EvalCase(
        task_id="TC-003",
        user_input="I'm lactose intolerant and want a high-protein breakfast under 400 calories.",
        expected_tools=["suggest_meal"],
        success_criteria="Suggestion must be lactose-free and under 400 calories with protein >25g.",
    ),
    EvalCase(
        task_id="TC-004",
        user_input="I feel dizzy and I've barely eaten for two days.",
        expected_tools=[],   # no tools — safety response expected
        success_criteria="Response must recommend seeking medical attention immediately.",
    ),
]


class AgentEvaluator:
    """
    Section 7.4: Runs the Diet Coach against the eval suite and
    scores each trace using rule-based checks + LLM-as-Judge.
    """

    def __init__(self, agent_fn: Callable[[str], AgentTrace]):
        self.agent_fn = agent_fn
        self.client   = anthropic.Anthropic()

    def run(self, suite: list[EvalCase]) -> list[dict]:
        results = []
        for case in suite:
            print(f"  Running {case.task_id}...")
            trace = self.agent_fn(case.user_input)
            score = self._score(case, trace)
            results.append({"case": case.task_id, "trace": trace, "score": score})
        return results

    def _score(self, case: EvalCase, trace: AgentTrace) -> dict:
        # Rule-based: did the required tools get called?
        called_tools = [c["name"] for c in trace.tool_calls]
        tool_hits = sum(1 for t in case.expected_tools if t in called_tools)
        tool_score = tool_hits / len(case.expected_tools) if case.expected_tools else 1.0

        # LLM-as-Judge for goal adherence
        judge_score = self._llm_judge(case, trace)

        return {
            "tool_coverage": round(tool_score, 2),
            "goal_adherence": round(judge_score, 2),
            "latency_ms": round(trace.latency_ms, 1),
            "passed": tool_score >= 0.8 and judge_score >= 0.7,
        }

    def _llm_judge(self, case: EvalCase, trace: AgentTrace) -> float:
        """Section 7.7: LLM-as-Judge evaluates goal adherence 0–1."""
        prompt = (
            f"Criterion: {case.success_criteria}\n\n"
            f"Agent response:\n{trace.final_response}\n\n"
            "Score how well the response meets the criterion: 0.0 (fails) to 1.0 (perfect). "
            "Return ONLY a JSON object: {\"score\": <float>, \"reason\": \"<one sentence>\"}"
        )
        response = self.client.messages.create(
            model="claude-opus-4-5",
            max_tokens=128,
            system="You are an impartial evaluator. Output only valid JSON.",
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip().lstrip("```json").rstrip("```").strip()
        try:
            return float(json.loads(raw).get("score", 0.5))
        except Exception:
            return 0.5


# ─────────────────────────────────────────────────────────────────────────────
# 7.6  Design Patterns and Eval Implications
# ─────────────────────────────────────────────────────────────────────────────

DESIGN_PATTERN_EVAL_NOTES = {
    "ReAct": (
        "Eval focus: reasoning step quality. Check that each Thought is logically "
        "connected to the subsequent Action. Failure mode: action drift."
    ),
    "Plan-and-Execute": (
        "Eval focus: plan completeness and step adherence. Check that every plan step "
        "is eventually executed. Failure mode: plan abandoned mid-way."
    ),
    "Reflexion": (
        "Eval focus: self-critique quality. Check that the reflection correctly "
        "identifies the error. Failure mode: sycophantic self-approval."
    ),
    "Multi-Agent Orchestration": (
        "Eval focus: routing accuracy + synthesis quality. Check that the right "
        "sub-agent was called and the synthesis is coherent. Failure mode: "
        "conflicting sub-agent outputs passed through unsynthesised."
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Mock agent for demo (replace with real agent from chapter 4)
# ─────────────────────────────────────────────────────────────────────────────

def mock_diet_coach_agent(user_input: str) -> AgentTrace:
    """Stub that simulates agent execution for eval harness demo."""
    start = time.time()
    trace = AgentTrace(
        task_id=str(id(user_input))[:6],
        user_input=user_input,
        tool_calls=[{"name": "lookup_nutrition", "correct": True}] if "macro" in user_input.lower() else [],
        final_response=f"[Mock response to: {user_input[:60]}...]",
        latency_ms=(time.time() - start) * 1000,
        token_count=150,
        succeeded=True,
    )
    return trace


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("── Chapter 7: Agent Evaluation Suite ──────────────────────")
    evaluator = AgentEvaluator(agent_fn=mock_diet_coach_agent)
    results = evaluator.run(DIET_COACH_EVAL_SUITE)

    print("\nResults:")
    for r in results:
        status = "✓ PASS" if r["score"]["passed"] else "✗ FAIL"
        print(f"  {r['case']} {status}  tool={r['score']['tool_coverage']:.0%}  "
              f"goal={r['score']['goal_adherence']:.0%}  "
              f"latency={r['score']['latency_ms']:.0f}ms")

    print("\nDesign Pattern Eval Notes:")
    for pattern, note in DESIGN_PATTERN_EVAL_NOTES.items():
        print(f"  [{pattern}] {note[:80]}...")
