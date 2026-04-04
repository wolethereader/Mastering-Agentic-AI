"""
Mastering Agentic AI — Chapter 9
Reinforcement Learning and Agents

Implements a simplified GRPO (Group Relative Policy Optimization) loop
that fine-tunes a small language model (Qwen2.5-0.5B-Instruct) to generate
correct Python code, using test-case pass rates as the reward signal.
"""

import ast
import textwrap

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
GROUP_SIZE = 4
MAX_NEW_TOKENS = 128
LEARNING_RATE = 1e-5
TRAIN_STEPS = 50


DATASET = [
    {
        "prompt": (
            "Write a Python function `sum_evens(numbers)` that returns the sum "
            "of all even numbers in the list."
        ),
        "starter_code": "def sum_evens(numbers):\n    pass\n",
        "tests": [
            ("sum_evens([1, 2, 3, 4])", 6),
            ("sum_evens([])", 0),
            ("sum_evens([1, 3, 5])", 0),
        ],
    },
    {
        "prompt": (
            "Write a Python function `reverse_string(text)` that returns the "
            "reversed string."
        ),
        "starter_code": "def reverse_string(text):\n    pass\n",
        "tests": [
            ('reverse_string("hello")', "olleh"),
            ('reverse_string("")', ""),
            ('reverse_string("a")', "a"),
        ],
    },
    {
        "prompt": (
            "Write a Python function `find_max(numbers)` that returns the largest "
            "number in a non-empty list."
        ),
        "starter_code": "def find_max(numbers):\n    pass\n",
        "tests": [
            ("find_max([1, 5, 3])", 5),
            ("find_max([10, 2, 8])", 10),
            ("find_max([-4, -2, -9])", -2),
        ],
    },
    {
        "prompt": (
            "Write a Python function `count_vowels(text)` that returns how many "
            "vowels appear in the string."
        ),
        "starter_code": "def count_vowels(text):\n    pass\n",
        "tests": [
            ('count_vowels("hello")', 2),
            ('count_vowels("sky")', 0),
            ('count_vowels("AEIOU")', 5),
        ],
    },
]


def build_prompt(example: dict) -> str:
    return textwrap.dedent(
        f"""
        You are a careful Python coding assistant.
        Solve the task by returning only valid Python code.

        Task:
        {example["prompt"]}

        Starter code:
        {example["starter_code"]}
        """
    ).strip()


def extract_python(response: str) -> str:
    if "```python" in response:
        return response.split("```python", 1)[1].split("```", 1)[0].strip()
    if "```" in response:
        return response.split("```", 1)[1].split("```", 1)[0].strip()
    return response.strip()


def run_tests(code: str, tests: list[tuple[str, object]]) -> float:
    try:
        ast.parse(code)
    except SyntaxError:
        return -0.5

    namespace: dict[str, object] = {}
    try:
        exec(code, namespace, namespace)
    except Exception:
        return -0.5

    passed = 0
    for expression, expected in tests:
        try:
            result = eval(expression, namespace, namespace)
        except Exception:
            continue
        if result == expected:
            passed += 1

    if passed == len(tests):
        return 1.0
    if passed == 0:
        return 0.0
    return passed / len(tests)


def reward_fn(response: str, example: dict) -> float:
    code = extract_python(response)
    return run_tests(code, example["tests"])


def compute_group_advantages(rewards: torch.Tensor) -> torch.Tensor:
    mean_reward = rewards.mean(dim=1, keepdim=True)
    return rewards - mean_reward


def score_rollouts(
    model: AutoModelForCausalLM,
    prompt_tokens: torch.Tensor,
    response_tokens: torch.Tensor,
) -> torch.Tensor:
    full_tokens = torch.cat([prompt_tokens, response_tokens], dim=1)
    outputs = model(full_tokens)
    logits = outputs.logits[:, :-1, :]
    targets = full_tokens[:, 1:]

    log_probs = torch.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    prompt_length = prompt_tokens.shape[1]
    response_log_probs = token_log_probs[:, prompt_length - 1 :]
    return response_log_probs.sum(dim=1)


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    for step in range(TRAIN_STEPS):
        batch = DATASET[:BATCH_SIZE]
        prompts = [build_prompt(example) for example in batch]

        expanded_prompts = []
        expanded_examples = []
        for prompt, example in zip(prompts, batch):
            for _ in range(GROUP_SIZE):
                expanded_prompts.append(prompt)
                expanded_examples.append(example)

        prompt_batch = tokenizer(
            expanded_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(DEVICE)

        with torch.no_grad():
            generated = model.generate(
                **prompt_batch,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                max_new_tokens=MAX_NEW_TOKENS,
                pad_token_id=tokenizer.pad_token_id,
            )

        prompt_length = prompt_batch["input_ids"].shape[1]
        response_tokens = generated[:, prompt_length:]
        responses = tokenizer.batch_decode(response_tokens, skip_special_tokens=True)

        rewards = torch.tensor(
            [reward_fn(response, example) for response, example in zip(responses, expanded_examples)],
            dtype=torch.float32,
            device=DEVICE,
        ).view(BATCH_SIZE, GROUP_SIZE)

        advantages = compute_group_advantages(rewards).reshape(-1)
        sequence_log_probs = score_rollouts(
            model,
            prompt_batch["input_ids"],
            response_tokens,
        )

        loss = -(advantages * sequence_log_probs).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_reward = rewards.mean().item()
        print(f"step={step:03d}  reward={avg_reward:.3f}  loss={loss.item():.3f}")
