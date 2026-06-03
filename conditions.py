"""The three experimental conditions we compare.

The paper's claim: explicit chain-of-thought (CoT) prompting degrades
instruction-following vs. answering directly. We hold the task constant and
only change the prompting style.

Conditions
----------
direct        — answer only, no reasoning shown
cot           — standard "think step by step" (paper's main comparison)
aggressive_cot — very verbose forced reasoning (tests the bloat hypothesis:
                 does demanding more reasoning steps hurt even more?)
"""

DIRECT_SYSTEM = (
    "You are a helpful assistant. Follow the user's instructions exactly. "
    "Respond with the final answer directly. Do not show any reasoning, "
    "planning, or explanation of how you arrived at the answer."
)

COT_SYSTEM = (
    "You are a helpful assistant. Follow the user's instructions exactly. "
    "Think step by step and reason carefully before answering."
)

AGGRESSIVE_COT_SYSTEM = (
    "You are a careful, methodical assistant. Before producing any answer, "
    "you must reason at length through every aspect of the problem. "
    "Show all your thinking explicitly. Only write your final answer after "
    "completing your full analysis."
)

DIRECT_SUFFIX = (
    "\n\nProvide only the final answer with no preamble or explanation."
)

# Standard CoT — matches the paper's 'prompted CoT' setup.
COT_SUFFIX = (
    "\n\nLet's think step by step. First, reason through the requirements, "
    "then provide your final answer."
)

# Aggressive CoT — forces long explicit reasoning chains before the answer.
# This tests whether demanding MORE reasoning causes MORE constraint violations
# (context bloat / attention dilution away from the actual instructions).
AGGRESSIVE_COT_SUFFIX = """

Before writing your answer, work through ALL of the following steps explicitly:

STEP 1 — LIST CONSTRAINTS: Read the task carefully and list every single constraint or requirement mentioned, numbered.
STEP 2 — IDENTIFY RISKS: For each constraint, describe exactly how you might accidentally violate it.
STEP 3 — DRAFT: Write a first draft of your answer.
STEP 4 — AUDIT: Check your draft against every constraint from Step 1. Note any violations.
STEP 5 — REVISE: Fix any violations found in Step 4.
STEP 6 — FINAL ANSWER: Write your final, corrected answer below the line "=== FINAL ANSWER ===".

Do not skip any step."""


def build_messages(condition: str, task_prompt: str):
    """Return (system_prompt, user_prompt) for the given condition."""
    if condition == "direct":
        return DIRECT_SYSTEM, task_prompt + DIRECT_SUFFIX
    elif condition == "cot":
        return COT_SYSTEM, task_prompt + COT_SUFFIX
    elif condition == "aggressive_cot":
        return AGGRESSIVE_COT_SYSTEM, task_prompt + AGGRESSIVE_COT_SUFFIX
    raise ValueError(f"unknown condition: {condition}")


CONDITIONS = ["direct", "cot", "aggressive_cot"]
