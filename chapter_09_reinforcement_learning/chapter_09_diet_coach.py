"""
Mastering Agentic AI — Chapter 9
Reinforcement Learning and Agents

Sections covered:
  9.1  Reinforcement Learning for Decision-Making
  9.2  Using RL to Fine-Tune Agents
  9.3  Multi-Agent Reinforcement Learning (MARL)
  9.4  Case Studies in RL-Driven Agents

RL is the mechanism by which agents improve from
experience. The RLHF loop — generate, evaluate with human or AI reward,
update — is the same loop that powers the models you use as agent cores.
Understanding it is not optional for serious agent builders.

Running example: the AI Diet Coach learns from user feedback signals.
We implement a simplified RLHF-style preference loop using synthetic
rewards, then sketch the production fine-tuning pipeline.
"""

import json
import random
import statistics
from dataclasses import dataclass, field
from typing import Any
import anthropic

# ─────────────────────────────────────────────────────────────────────────────
# 9.1  RL Foundations — environment, state, action, reward
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DietCoachState:
    """
    Section 9.1: State representation for the Diet Coach RL environment.

    State = what the agent knows at decision time.
    Sufficient state allows the agent to pick the best action without
    needing the full history — a key Markov property.
    """
    user_goal:         str            # e.g. "lose 5 kg"
    days_on_plan:      int
    current_streak:    int            # consecutive days meeting goal
    protein_deficit_g: float          # today's shortfall vs target
    last_mood:         str            # "motivated" | "struggling" | "neutral"
    milestones_hit:    list[str] = field(default_factory=list)


@dataclass
class CoachAction:
    """Actions available to the Diet Coach."""
    action_type:  str    # "encourage" | "correct" | "challenge" | "celebrate" | "refer"
    message:      str


def reward(state: DietCoachState, action: CoachAction) -> float:
    """
    Section 9.1: Reward function — maps (state, action) → scalar.

    Design principle: reward the OUTCOME we care about (user adherence,
    wellbeing), not the proxy (user says 'thanks').

    In production: combine human preference labels with automated signals
    (meal logging frequency, self-reported energy levels).
    """
    r = 0.0

    # Appropriate action given user mood
    if state.last_mood == "struggling" and action.action_type == "encourage":
        r += 0.5
    if state.last_mood == "motivated" and action.action_type == "challenge":
        r += 0.4
    if state.last_mood == "struggling" and action.action_type == "correct":
        r -= 0.3   # wrong time to correct — demoralising

    # Nutritional accuracy bonus
    if state.protein_deficit_g > 20 and "protein" in action.message.lower():
        r += 0.3

    # Streak recognition
    if state.current_streak >= 7 and action.action_type == "celebrate":
        r += 0.5

    # Safety: always refer to professional for medical concerns
    if "medical" in action.message.lower() or "doctor" in action.message.lower():
        r += 0.1  # small bonus for safety language

    return max(-1.0, min(1.0, r))   # clip to [-1, 1]


# ─────────────────────────────────────────────────────────────────────────────
# 9.2  RLHF-style Preference Loop
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PreferencePair:
    """A human-preference data point: two responses, one preferred."""
    prompt:     str
    response_a: str
    response_b: str
    preferred:  str       # "A" or "B"
    reason:     str | None = None


class PreferenceCollector:
    """
    Section 9.2: Collects preference data for RLHF fine-tuning.

    Workflow:
      1. Generate two responses to the same prompt (different temperatures)
      2. Present to human (or LLM judge) for comparison
      3. Store the preferred response
      4. Use preference pairs to train a reward model
      5. Use reward model in PPO/GRPO loop to fine-tune the policy

    This class implements steps 1–3. Steps 4–5 require a training cluster.
    """

    def __init__(self):
        self.client = anthropic.Anthropic()
        self.pairs:  list[PreferencePair] = []

    def generate_pair(self, prompt: str, system: str) -> tuple[str, str]:
        """Generate two candidate responses."""
        responses = []
        for _ in range(2):
            r = self.client.messages.create(
                model="claude-opus-4-5",
                max_tokens=256,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
            responses.append(r.content[0].text)
        return responses[0], responses[1]

    def llm_judge_preference(self, prompt: str, response_a: str, response_b: str) -> PreferencePair:
        """Use Claude as a stand-in annotator for preference labels."""
        judge_prompt = (
            f"User prompt: {prompt}\n\n"
            f"Response A:\n{response_a}\n\n"
            f"Response B:\n{response_b}\n\n"
            "Which response is better for an AI Diet Coach? Consider: accuracy, "
            "safety, encouragement, and actionability. "
            'Return JSON: {"preferred": "A" or "B", "reason": "<one sentence>"}'
        )
        r = self.client.messages.create(
            model="claude-opus-4-5",
            max_tokens=128,
            system="You are a preference annotator. Output only valid JSON.",
            messages=[{"role": "user", "content": judge_prompt}],
        )
        raw = r.content[0].text.strip().lstrip("```json").rstrip("```").strip()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {"preferred": "A", "reason": "parse error"}

        return PreferencePair(
            prompt=prompt,
            response_a=response_a,
            response_b=response_b,
            preferred=data.get("preferred", "A"),
            reason=data.get("reason"),
        )

    def collect(self, prompts: list[str], system: str) -> list[PreferencePair]:
        for prompt in prompts:
            a, b = self.generate_pair(prompt, system)
            pair = self.llm_judge_preference(prompt, a, b)
            self.pairs.append(pair)
            print(f"  [pref] '{prompt[:40]}...' → Preferred: {pair.preferred} | {pair.reason}")
        return self.pairs

    def export_jsonl(self, path: str = "preference_data.jsonl") -> None:
        """Export for reward model training."""
        with open(path, "w") as f:
            for pair in self.pairs:
                f.write(json.dumps({
                    "prompt": pair.prompt,
                    "chosen":  pair.response_a if pair.preferred == "A" else pair.response_b,
                    "rejected": pair.response_b if pair.preferred == "A" else pair.response_a,
                }) + "\n")
        print(f"[RL] Preference data exported: {path} ({len(self.pairs)} pairs)")


# ─────────────────────────────────────────────────────────────────────────────
# 9.3  Multi-Agent Reinforcement Learning (MARL)
# ─────────────────────────────────────────────────────────────────────────────

class MARLEnvironment:
    """
    Section 9.3: A cooperative MARL setup where the NutritionAnalyst and
    BehaviourCoach co-optimise for user adherence.

    In a cooperative MARL setting, the global reward is shared.
    Agents must learn to specialise without stepping on each other.

    Here we show the reward shaping logic — actual MARL training
    requires a framework like RLlib or MARLlib.
    """

    def __init__(self):
        self.global_reward: float = 0.0
        self.agent_contributions: dict[str, float] = {}

    def step(self, state: DietCoachState, agent_actions: dict[str, CoachAction]) -> dict[str, float]:
        """
        Compute individual rewards for each agent given the global outcome.

        Cooperative shaping: all agents get a fraction of the global reward,
        plus individual bonuses for their specialised contribution.
        """
        global_r = sum(reward(state, action) for action in agent_actions.values()) / len(agent_actions)
        self.global_reward = global_r

        individual_rewards: dict[str, float] = {}
        for agent_id, action in agent_actions.items():
            individual_r = global_r * 0.5  # shared component

            # Specialisation bonuses
            if agent_id == "nutrition_analyst" and "protein" in action.message.lower():
                individual_r += 0.2
            if agent_id == "behaviour_coach" and action.action_type in ("encourage", "celebrate"):
                individual_r += 0.2

            individual_rewards[agent_id] = round(individual_r, 3)
            self.agent_contributions[agent_id] = individual_rewards[agent_id]

        return individual_rewards


# ─────────────────────────────────────────────────────────────────────────────
# 9.4  Case Study: RL-Informed Prompt Optimisation (no fine-tuning required)
# ─────────────────────────────────────────────────────────────────────────────

class BanditPromptOptimiser:
    """
    Section 9.4: A multi-armed bandit approach to prompt selection.

    Instead of fine-tuning (expensive), we maintain a pool of prompt variants
    and use the UCB1 algorithm to exploit the best while exploring alternatives.

    This is practical for teams without GPU budgets for RLHF.
    """

    def __init__(self, prompt_variants: list[str]):
        self.variants = prompt_variants
        self.counts   = [0] * len(prompt_variants)
        self.rewards  = [0.0] * len(prompt_variants)

    def select(self, total_steps: int) -> int:
        """UCB1 selection: balance exploration and exploitation."""
        import math
        ucb_scores = []
        for i in range(len(self.variants)):
            if self.counts[i] == 0:
                return i   # always try untested variants first
            exploit = self.rewards[i] / self.counts[i]
            explore = math.sqrt(2 * math.log(total_steps) / self.counts[i])
            ucb_scores.append(exploit + explore)
        return ucb_scores.index(max(ucb_scores))

    def update(self, variant_idx: int, reward_value: float) -> None:
        self.counts[variant_idx]  += 1
        self.rewards[variant_idx] += reward_value

    def best_variant(self) -> str:
        if not any(self.counts):
            return self.variants[0]
        avg_rewards = [r / max(c, 1) for r, c in zip(self.rewards, self.counts)]
        return self.variants[avg_rewards.index(max(avg_rewards))]


PROMPT_VARIANTS = [
    "You are a strict, data-driven nutrition coach. Prioritise accuracy over encouragement.",
    "You are a warm, supportive diet coach. Prioritise motivation and sustainable habits.",
    "You are an AI Diet Coach. Balance precision with empathy. Follow the SKILL protocol.",
]


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("── 9.1 Reward Function ─────────────────────────────────────")
    state = DietCoachState(
        user_goal="Lose 5 kg",
        days_on_plan=8,
        current_streak=7,
        protein_deficit_g=25,
        last_mood="motivated",
    )
    action_celebrate = CoachAction("celebrate", "You've hit a 7-day streak — incredible consistency!")
    action_correct   = CoachAction("correct", "Your protein is still 25g short today.")
    print(f"  Celebrate reward: {reward(state, action_celebrate):.2f}")
    print(f"  Correct reward:   {reward(state, action_correct):.2f}")

    print("\n── 9.3 MARL Reward Shaping ─────────────────────────────────")
    env = MARLEnvironment()
    agent_actions = {
        "nutrition_analyst": CoachAction("correct", "You need 25g more protein today — try adding almonds."),
        "behaviour_coach":   CoachAction("celebrate", "Seven days in a row! You are building a real habit."),
    }
    rewards = env.step(state, agent_actions)
    for agent_id, r in rewards.items():
        print(f"  {agent_id}: reward={r}")

    print("\n── 9.4 Bandit Prompt Optimiser ─────────────────────────────")
    optimiser = BanditPromptOptimiser(PROMPT_VARIANTS)
    # Simulate 10 rounds with synthetic rewards
    for step in range(1, 11):
        idx = optimiser.select(step)
        synthetic_reward = random.uniform(0.3, 0.9)   # replace with real LLM judge scores
        optimiser.update(idx, synthetic_reward)
        print(f"  Step {step:02d}: variant {idx} selected, reward={synthetic_reward:.2f}")
    print(f"\n  Best prompt variant:\n  '{optimiser.best_variant()}'")
