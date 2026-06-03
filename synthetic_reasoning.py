#!/usr/bin/env python3
"""
synthetic_reasoning.py — Can a cheap model with an explicit reasoning loop
match a native reasoning model on logic tasks?

The hypothesis: o-series models (o4-mini, o3) internally run a multi-stage
thinking loop — explore → plan → solve → critique → refine — before emitting
their final answer.  This script replicates that loop explicitly via multiple
API calls on a cheap model (gpt-4o-mini) and compares the result against:

  • o4-mini  direct prompt  (native reasoning, effort=high)
  • gpt-5.5  direct prompt  (flagship chat, no explicit reasoning)

The loop for the "synthetic reasoner":
  Stage 1 — EXPLORE  : Understand the problem, restate constraints
  Stage 2 — PLAN     : Outline an approach (no solving yet)
  Stage 3 — SOLVE    : Execute the plan step by step
  Stage 4 — CRITIQUE : Self-review — find mistakes, gaps, or wrong assumptions
  Stage 5 — REFINE   : Produce a final, corrected, clean answer

Output shows every stage, time, tokens, cost per stage, and a side-by-side
summary comparing all three approaches.

Usage:
  python synthetic_reasoning.py
  python synthetic_reasoning.py --task "Is 1033 a prime number? Show your work."
  python synthetic_reasoning.py --loop-model gpt-4o-mini --stages explore plan solve
  python synthetic_reasoning.py --no-save
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

# ── Models ─────────────────────────────────────────────────────────────────────
DEFAULT_LOOP_MODEL   = os.getenv("LOOP_MODEL",   "gpt-4o-mini")  # cheap "dumb" model
DEFAULT_NATIVE_MODEL = os.getenv("NATIVE_MODEL", "o4-mini")      # native reasoner
DEFAULT_CHAT_MODEL   = os.getenv("CHAT_MODEL",   "gpt-5.5")      # flagship chat

# ── Pricing (USD / 1M tokens) ──────────────────────────────────────────────────
PRICES: dict[str, tuple[float, float]] = {
    "gpt-4o-mini":  ( 0.15,   0.60),
    "gpt-4o":       ( 2.50,  10.00),
    "gpt-5.5":      ( 5.00,  30.00),
    "gpt-5.4":      ( 2.50,  15.00),
    "gpt-5.4-mini": ( 0.75,   4.50),
    "o4-mini":      ( 1.10,   4.40),
    "o3-mini":      ( 1.10,   4.40),
    "o3":           (10.00,  40.00),
}

def cost_usd(model: str, prompt_tok: int, completion_tok: int) -> float | None:
    key = next((k for k in PRICES if model.startswith(k)), None)
    if key is None:
        return None
    inp, out = PRICES[key]
    return (prompt_tok * inp + completion_tok * out) / 1_000_000

# ── Default task ───────────────────────────────────────────────────────────────
DEFAULT_TASK = (
    "Three logicians walk into a bar. The bartender asks: "
    "'Do all three of you want a drink?' "
    "The first logician says 'I don't know.' "
    "The second logician says 'I don't know.' "
    "The third logician says 'Yes.' "
    "Explain why the third logician can be certain, step by step."
)

# ── Stage definitions ──────────────────────────────────────────────────────────
# Each stage sees the original task + all prior stage outputs as context.
# The system prompt frames the model as a careful logical reasoner.
STAGE_SYSTEM = (
    "You are a careful, precise logical reasoner. "
    "Follow the instructions for each stage exactly. "
    "Do not skip ahead or combine stages."
)

STAGES: list[tuple[str, str]] = [
    (
        "EXPLORE",
        "Read the problem carefully. Do NOT attempt to solve it yet.\n"
        "1. Restate the problem in your own words.\n"
        "2. List every constraint and given fact.\n"
        "3. Identify what exactly is being asked.",
    ),
    (
        "PLAN",
        "Do NOT solve the problem yet.\n"
        "Outline the logical steps you will take to solve it. "
        "Number each step. Be specific about what you will check at each step.",
    ),
    (
        "SOLVE",
        "Execute your plan from Stage 2 step by step. "
        "Show your full reasoning. Do not skip steps.",
    ),
    (
        "CRITIQUE",
        "Review your solution from Stage 3 critically.\n"
        "- Did you make any logical errors or unsupported assumptions?\n"
        "- Are there edge cases you missed?\n"
        "- Is your conclusion actually supported by your reasoning?\n"
        "Be honest. If you find errors, describe them clearly.",
    ),
    (
        "REFINE",
        "Based on your critique in Stage 4, write your final answer.\n"
        "Fix any errors you found. Be concise but complete.\n"
        "This is the answer you would give to the user.",
    ),
]

ALL_STAGE_NAMES = [s[0].lower() for s in STAGES]

# ── Data classes ───────────────────────────────────────────────────────────────
@dataclass
class StageResult:
    name: str
    prompt_tokens: int
    completion_tokens: int
    elapsed_sec: float
    output: str
    error: str | None = None

    def cost(self, model: str) -> float | None:
        return cost_usd(model, self.prompt_tokens, self.completion_tokens)


@dataclass
class LoopResult:
    """Result for the synthetic reasoning loop (multi-stage, single cheap model)."""
    model: str
    task: str
    stages: list[StageResult] = field(default_factory=list)

    @property
    def total_prompt_tokens(self) -> int:
        return sum(s.prompt_tokens for s in self.stages)

    @property
    def total_completion_tokens(self) -> int:
        return sum(s.completion_tokens for s in self.stages)

    @property
    def total_elapsed_sec(self) -> float:
        return sum(s.elapsed_sec for s in self.stages)

    @property
    def final_answer(self) -> str:
        good = [s for s in self.stages if not s.error]
        return good[-1].output if good else ""

    def total_cost(self) -> float | None:
        costs = [s.cost(self.model) for s in self.stages]
        if any(c is None for c in costs):
            return None
        return sum(costs)  # type: ignore[arg-type]


@dataclass
class VotingResult:
    """N independent loop runs + a consensus aggregation call."""
    model: str
    task: str
    n_runs: int
    runs: list[LoopResult] = field(default_factory=list)
    consensus_answer: str = ""
    consensus_prompt_tokens: int = 0
    consensus_completion_tokens: int = 0
    consensus_elapsed_sec: float = 0.0
    consensus_error: str | None = None

    @property
    def total_prompt_tokens(self) -> int:
        return sum(r.total_prompt_tokens for r in self.runs) + self.consensus_prompt_tokens

    @property
    def total_completion_tokens(self) -> int:
        return sum(r.total_completion_tokens for r in self.runs) + self.consensus_completion_tokens

    @property
    def total_elapsed_sec(self) -> float:
        # runs are parallel; use max of run times + consensus
        run_wall = max((r.total_elapsed_sec for r in self.runs), default=0.0)
        return run_wall + self.consensus_elapsed_sec

    def total_cost(self) -> float | None:
        run_costs = [r.total_cost() for r in self.runs]
        cons_cost = cost_usd(self.model, self.consensus_prompt_tokens, self.consensus_completion_tokens)
        if any(c is None for c in run_costs) or cons_cost is None:
            return None
        return sum(run_costs) + cons_cost  # type: ignore[arg-type]

    @property
    def final_answer(self) -> str:
        return self.consensus_answer or (self.runs[0].final_answer if self.runs else "")

    @property
    def label(self) -> str:
        return f"{self.model} × {self.n_runs} runs (self-consistency)"


@dataclass
class DirectResult:
    """Result for a single direct-prompt call (native reasoning or chat model)."""
    label: str          # e.g. "o4-mini (native reasoning)"
    model: str
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int   # only from o-series; 0 otherwise
    elapsed_sec: float
    output: str
    error: str | None = None

    def cost(self) -> float | None:
        return cost_usd(self.model, self.prompt_tokens, self.completion_tokens)


# ── OpenAI client (shared lazy init) ──────────────────────────────────────────
_openai_client = None

def _client():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI()
    return _openai_client


# ── API call helpers ───────────────────────────────────────────────────────────
def _chat(model: str, messages: list[dict], reasoning_effort: str | None = None,
          temperature: float | None = None) -> tuple[str, int, int, int]:
    """Returns (text, prompt_tokens, completion_tokens, reasoning_tokens)."""
    kwargs: dict = dict(model=model, messages=messages)
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    if temperature is not None:
        kwargs["temperature"] = temperature
    resp = _client().chat.completions.create(**kwargs)
    usage = resp.usage
    details = getattr(usage, "completion_tokens_details", None)
    rtoks = getattr(details, "reasoning_tokens", 0) or 0
    return (
        resp.choices[0].message.content or "",
        usage.prompt_tokens,
        usage.completion_tokens,
        rtoks,
    )


# ── Run the synthetic reasoning loop ──────────────────────────────────────────
def run_loop(task: str, model: str, stage_names: list[str]) -> LoopResult:
    result = LoopResult(model=model, task=task)

    # conversation history grows with each stage
    # system message sets the frame; user messages inject each stage prompt
    history: list[dict] = [{"role": "system", "content": STAGE_SYSTEM}]

    # Inject the original task once, before any stage
    history.append({
        "role": "user",
        "content": f"Here is the problem you will work through in stages:\n\n{task}",
    })
    history.append({
        "role": "assistant",
        "content": "Understood. I will work through this problem stage by stage as instructed.",
    })

    active_stages = [s for s in STAGES if s[0].lower() in stage_names]

    for stage_name, stage_prompt in active_stages:
        user_msg = f"--- Stage: {stage_name} ---\n{stage_prompt}"
        history.append({"role": "user", "content": user_msg})

        t0 = time.perf_counter()
        try:
            text, ptok, ctok, _ = _chat(model, history)
            elapsed = time.perf_counter() - t0
            sr = StageResult(
                name=stage_name,
                prompt_tokens=ptok,
                completion_tokens=ctok,
                elapsed_sec=elapsed,
                output=text,
            )
            # Feed the stage output back so the next stage has full context
            history.append({"role": "assistant", "content": text})
        except Exception as e:  # noqa: BLE001
            elapsed = time.perf_counter() - t0
            sr = StageResult(
                name=stage_name,
                prompt_tokens=0, completion_tokens=0,
                elapsed_sec=elapsed, output="", error=str(e),
            )
            history.append({"role": "assistant", "content": f"[Error in stage {stage_name}]"})

        result.stages.append(sr)

    return result


# ── Run a direct prompt ────────────────────────────────────────────────────────
def run_direct(task: str, model: str, label: str,
               reasoning_effort: str | None = None) -> DirectResult:
    messages = [{"role": "user", "content": task}]
    t0 = time.perf_counter()
    try:
        text, ptok, ctok, rtok = _chat(model, messages, reasoning_effort)
        elapsed = time.perf_counter() - t0
        return DirectResult(
            label=label, model=model,
            prompt_tokens=ptok, completion_tokens=ctok, reasoning_tokens=rtok,
            elapsed_sec=elapsed, output=text,
        )
    except Exception as e:  # noqa: BLE001
        return DirectResult(
            label=label, model=model,
            prompt_tokens=0, completion_tokens=0, reasoning_tokens=0,
            elapsed_sec=time.perf_counter() - t0, output="", error=str(e),
        )


# ── Self-consistency: run loop N times + consensus ─────────────────────────────
CONSENSUS_SYSTEM = """You are a careful reasoning aggregator.
You will receive the same logic problem and N independent answers from the same model.
Your job:
1. Read all answers carefully.
2. Identify which conclusion appears most often (majority vote).
3. Among answers with the majority conclusion, pick the one with the clearest reasoning.
4. Output a single, clean, definitive final answer — do NOT mention the voting process.
   Just give the best unified explanation as if you solved it yourself."""


def run_voting_loop(task: str, model: str, stage_names: list[str],
                    n_runs: int, temperature: float = 0.8) -> VotingResult:
    """Run N independent reasoning loops in parallel, then aggregate via consensus."""
    vr = VotingResult(model=model, task=task, n_runs=n_runs)

    # ── Parallel runs ─────────────────────────────────────────────────────────
    def _single_run(run_idx: int) -> tuple[int, LoopResult]:
        return run_idx, _run_single_loop(task, model, stage_names, temperature=temperature)

    with ThreadPoolExecutor(max_workers=n_runs) as pool:
        futures = {pool.submit(_single_run, i): i for i in range(n_runs)}
        ordered: dict[int, LoopResult] = {}
        for fut in as_completed(futures):
            idx, result = fut.result()
            ordered[idx] = result

    vr.runs = [ordered[i] for i in range(n_runs)]

    # ── Consensus aggregation ─────────────────────────────────────────────────
    answers_block = "\n\n".join(
        f"--- Answer {i+1} ---\n{r.final_answer}"
        for i, r in enumerate(vr.runs)
    )
    user_msg = f"PROBLEM:\n{task}\n\nINDEPENDENT ANSWERS ({n_runs} runs):\n{answers_block}"

    t0 = time.perf_counter()
    try:
        text, ptok, ctok, _ = _chat(
            model,
            [
                {"role": "system", "content": CONSENSUS_SYSTEM},
                {"role": "user",   "content": user_msg},
            ],
        )
        vr.consensus_answer = text
        vr.consensus_prompt_tokens = ptok
        vr.consensus_completion_tokens = ctok
    except Exception as e:  # noqa: BLE001
        vr.consensus_error = str(e)
    vr.consensus_elapsed_sec = time.perf_counter() - t0

    return vr


def _run_single_loop(task: str, model: str, stage_names: list[str],
                     temperature: float = 0.8) -> LoopResult:
    """One loop run — like _run_loop_with_progress but silent (for parallel use)."""
    result = LoopResult(model=model, task=task)
    history: list[dict] = [{"role": "system", "content": STAGE_SYSTEM}]
    history.append({"role": "user",      "content": f"Here is the problem you will work through in stages:\n\n{task}"})
    history.append({"role": "assistant", "content": "Understood. I will work through this problem stage by stage as instructed."})

    active_stages = [s for s in STAGES if s[0].lower() in stage_names]
    for stage_name, stage_prompt in active_stages:
        history.append({"role": "user", "content": f"--- Stage: {stage_name} ---\n{stage_prompt}"})
        t0 = time.perf_counter()
        try:
            text, ptok, ctok, _ = _chat(model, history, temperature=temperature)
            elapsed = time.perf_counter() - t0
            sr = StageResult(name=stage_name, prompt_tokens=ptok,
                             completion_tokens=ctok, elapsed_sec=elapsed, output=text)
            history.append({"role": "assistant", "content": text})
        except Exception as e:  # noqa: BLE001
            elapsed = time.perf_counter() - t0
            sr = StageResult(name=stage_name, prompt_tokens=0, completion_tokens=0,
                             elapsed_sec=elapsed, output="", error=str(e))
            history.append({"role": "assistant", "content": f"[Error in stage {stage_name}]"})
        result.stages.append(sr)
    return result


# ── Display helpers ────────────────────────────────────────────────────────────
W = 74

def banner(title: str, char: str = "═") -> str:
    return f"\n{char*W}\n  {title}\n{char*W}"

def section(title: str) -> str:
    return f"\n  {'─'*70}\n  {title}\n  {'─'*70}"

def wrap(text: str, width: int = 70, indent: int = 4) -> str:
    import textwrap
    return textwrap.fill(text, width=width, initial_indent=" "*indent,
                         subsequent_indent=" "*indent)


def print_loop_result(lr: LoopResult) -> None:
    cost_str = f"${lr.total_cost():.4f}" if lr.total_cost() is not None else "n/a"
    print(banner(f"SYNTHETIC REASONING LOOP  —  model: {lr.model}"))
    print(f"\n  Total time : {lr.total_elapsed_sec:.1f}s")
    print(f"  Total cost : {cost_str}")
    print(f"  API calls  : {len(lr.stages)}  (one per stage)")
    print(f"  Tokens     : {lr.total_prompt_tokens:,} prompt  /  {lr.total_completion_tokens:,} completion")

    for sr in lr.stages:
        stage_cost = sr.cost(lr.model)
        cstr = f"${stage_cost:.5f}" if stage_cost else "n/a"
        print(section(f"Stage: {sr.name}  [{sr.elapsed_sec:.1f}s  {sr.completion_tokens} tokens  {cstr}]"))
        if sr.error:
            print(f"    ❌ ERROR: {sr.error}")
        else:
            print(wrap(sr.output, width=70, indent=4))

    print(section("FINAL ANSWER  (last stage output)"))
    print(wrap(lr.final_answer, width=70, indent=4))
    print()


def print_direct_result(dr: DirectResult) -> None:
    cost_str = f"${dr.cost():.4f}" if dr.cost() is not None else "n/a"
    print(banner(f"DIRECT PROMPT  —  {dr.label}"))
    print(f"\n  Time   : {dr.elapsed_sec:.1f}s")
    print(f"  Cost   : {cost_str}")
    print(f"  Tokens : {dr.prompt_tokens:,} prompt  /  {dr.completion_tokens:,} completion"
          + (f"  /  {dr.reasoning_tokens:,} reasoning (hidden)" if dr.reasoning_tokens else ""))
    print(section("ANSWER"))
    if dr.error:
        print(f"    ❌ ERROR: {dr.error}")
    else:
        print(wrap(dr.output, width=70, indent=4))
    print()


def print_voting_result(vr: VotingResult) -> None:
    cost_str = f"${vr.total_cost():.4f}" if vr.total_cost() is not None else "n/a"
    total_api_calls = sum(len(r.stages) for r in vr.runs) + 1  # +1 for consensus
    print(banner(f"SELF-CONSISTENCY LOOP  —  model: {vr.model}  ×{vr.n_runs} runs"))
    print(f"\n  Runs       : {vr.n_runs}  (parallel, temperature=0.8 for diversity)")
    print(f"  Stages/run : {len(vr.runs[0].stages) if vr.runs else 0}")
    print(f"  API calls  : {total_api_calls}  ({vr.n_runs} × loop + 1 consensus)")
    print(f"  Total time : {vr.total_elapsed_sec:.1f}s  (parallel run wall time + consensus)")
    print(f"  Total cost : {cost_str}")
    print(f"  Tokens     : {vr.total_prompt_tokens:,} prompt  /  {vr.total_completion_tokens:,} completion")

    for i, run in enumerate(vr.runs, 1):
        run_cost = f"${run.total_cost():.5f}" if run.total_cost() else "n/a"
        print(section(f"Run {i}/{vr.n_runs}  [{run.total_elapsed_sec:.1f}s  {run.total_completion_tokens} tokens  {run_cost}]"))
        print(wrap(run.final_answer, width=70, indent=4))

    print(section(f"CONSENSUS ANSWER  (aggregated from {vr.n_runs} runs)"))
    if vr.consensus_error:
        print(f"    ❌ Consensus error: {vr.consensus_error}")
    else:
        print(wrap(vr.consensus_answer, width=70, indent=4))
    print()


def print_comparison(loop_entry: LoopResult | VotingResult, directs: list[DirectResult]) -> None:
    print(banner("COMPARISON SUMMARY", "═"))

    hdr = f"  {'Approach':<42} {'Time':>7} {'API calls':>10} {'Tokens':>8} {'Cost':>9}"
    print(hdr)
    print(f"  {'─'*76}")

    # Loop / voting row
    if isinstance(loop_entry, VotingResult):
        vr = loop_entry
        cost_str = f"${vr.total_cost():.4f}" if vr.total_cost() is not None else "   n/a"
        total_tok = vr.total_prompt_tokens + vr.total_completion_tokens
        total_calls = sum(len(r.stages) for r in vr.runs) + 1
        lbl = f"{vr.model} × {vr.n_runs} runs + consensus"
        print(f"  {lbl:<42} {vr.total_elapsed_sec:>6.1f}s {total_calls:>10} {total_tok:>8,} {cost_str:>9}")
    else:
        lr = loop_entry
        cost_str = f"${lr.total_cost():.4f}" if lr.total_cost() is not None else "   n/a"
        total_tok = lr.total_prompt_tokens + lr.total_completion_tokens
        lbl = f"{lr.model} + reasoning loop ({len(lr.stages)} stages)"
        print(f"  {lbl:<42} {lr.total_elapsed_sec:>6.1f}s {len(lr.stages):>10} {total_tok:>8,} {cost_str:>9}")

    for dr in directs:
        cost_str = f"${dr.cost():.4f}" if dr.cost() is not None else "   n/a"
        total_tok = dr.prompt_tokens + dr.completion_tokens
        print(f"  {dr.label:<42} {dr.elapsed_sec:>6.1f}s {'1':>10} {total_tok:>8,} {cost_str:>9}")

    print(f"  {'─'*76}")
    print("""
  WHAT TO LOOK FOR
  ─────────────────
  • Does majority-voting lift the weak model to match native reasoning?
  • How much does N runs cost vs a single o4-mini call?
  • Does consensus answer clearly pick up the best run's reasoning?
  • Where does the weak model consistently go wrong across all runs?
""")


# ── Judge / correctness evaluator ─────────────────────────────────────────────
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-4o")  # separate neutral model as judge

JUDGE_SYSTEM = """You are an impartial expert evaluator. 
You will be given a reasoning/logic problem and multiple answers from different AI systems.
Evaluate each answer on three dimensions (score 1-10):
  • correctness   : Is the final conclusion factually/logically correct?
  • reasoning     : Is the step-by-step reasoning valid, clear, and complete?
  • completeness  : Does the answer fully address everything the question asks?

Respond ONLY with valid JSON in this exact structure (no markdown, no extra text):
{
  "evaluations": [
    {
      "label": "<label>",
      "correctness": <1-10>,
      "reasoning": <1-10>,
      "completeness": <1-10>,
      "verdict": "correct" | "partially_correct" | "incorrect",
      "explanation": "<2-3 sentence summary of strengths/weaknesses>"
    }
  ],
  "winner": "<label of best overall answer>",
  "judge_notes": "<1-2 sentences on key differences between approaches>"
}"""


@dataclass
class JudgeScore:
    label: str
    correctness: int
    reasoning: int
    completeness: int
    verdict: str
    explanation: str

    @property
    def total(self) -> int:
        return self.correctness + self.reasoning + self.completeness


@dataclass
class JudgeResult:
    judge_model: str
    scores: list[JudgeScore]
    winner: str
    judge_notes: str
    prompt_tokens: int
    completion_tokens: int
    elapsed_sec: float
    error: str | None = None

    def cost(self) -> float | None:
        return cost_usd(self.judge_model, self.prompt_tokens, self.completion_tokens)


def run_judge(task: str, loop_result: LoopResult | VotingResult,
              direct_results: list[DirectResult]) -> JudgeResult:
    """Call a neutral judge model to score all answers for correctness."""
    loop_label = loop_result.label if isinstance(loop_result, VotingResult) \
        else f"{loop_result.model} + reasoning loop"

    answers_text = f"=== ANSWER 1: {loop_label} ===\n{loop_result.final_answer}\n\n"
    for i, dr in enumerate(direct_results, 2):
        answers_text += f"=== ANSWER {i}: {dr.label} ===\n{dr.output}\n\n"

    labels = [loop_label] + [dr.label for dr in direct_results]

    user_msg = (
        f"PROBLEM:\n{task}\n\n"
        f"ANSWERS TO EVALUATE:\n{answers_text}"
        f"Labels to use exactly: {json.dumps(labels)}"
    )

    t0 = time.perf_counter()
    raw_text = ""
    try:
        # Use JSON mode so the API guarantees a parseable response
        resp = _client().chat.completions.create(
            model=JUDGE_MODEL,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user",   "content": user_msg},
            ],
        )
        elapsed = time.perf_counter() - t0
        raw_text = resp.choices[0].message.content or ""
        usage = resp.usage

        # Strip markdown fences in case they slip through anyway
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```", 2)[1]          # drop opening fence
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]                      # drop language tag
            cleaned = cleaned.rsplit("```", 1)[0].strip() # drop closing fence

        data = json.loads(cleaned)
        scores = [
            JudgeScore(
                label=e["label"],
                correctness=int(e["correctness"]),
                reasoning=int(e["reasoning"]),
                completeness=int(e["completeness"]),
                verdict=e["verdict"],
                explanation=e["explanation"],
            )
            for e in data["evaluations"]
        ]
        return JudgeResult(
            judge_model=JUDGE_MODEL,
            scores=scores,
            winner=data.get("winner", ""),
            judge_notes=data.get("judge_notes", ""),
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            elapsed_sec=elapsed,
        )
    except Exception as e:  # noqa: BLE001
        preview = raw_text[:300].replace("\n", "↵") if raw_text else "<empty response>"
        return JudgeResult(
            judge_model=JUDGE_MODEL, scores=[], winner="", judge_notes="",
            prompt_tokens=0, completion_tokens=0,
            elapsed_sec=time.perf_counter() - t0,
            error=f"{type(e).__name__}: {e}\n  Raw response preview: {preview}",
        )


def print_judge_verdict(jr: JudgeResult) -> None:
    print(banner("⚖️   CORRECTNESS EVALUATION  (judge: " + jr.judge_model + ")", "═"))

    if jr.error:
        print(f"\n  ❌ Judge error: {jr.error}\n")
        return

    cost_str = f"${jr.cost():.4f}" if jr.cost() else "n/a"
    print(f"\n  Judge cost: {cost_str}  ({jr.elapsed_sec:.1f}s)\n")

    # Score table
    hdr = f"  {'Approach':<40} {'Correct':>8} {'Reason':>8} {'Complete':>9} {'TOTAL':>6}  Verdict"
    print(hdr)
    print(f"  {'─'*80}")
    for s in jr.scores:
        bar = "█" * s.total + "░" * (30 - s.total)
        verdict_icon = {"correct": "✅", "partially_correct": "⚠️ ", "incorrect": "❌"}.get(s.verdict, "?")
        print(f"  {s.label:<40} {s.correctness:>8}/10 {s.reasoning:>8}/10 {s.completeness:>9}/10 {s.total:>6}/30  {verdict_icon} {s.verdict}")
    print(f"  {'─'*80}")

    # Winner
    print(f"\n  🏆  Winner : {jr.winner}")
    print(f"\n  Judge notes: {jr.judge_notes}\n")

    # Per-answer explanations
    for s in jr.scores:
        print(f"  [{s.label}]")
        print(wrap(s.explanation, width=70, indent=4))
        print()


RESULTS_DIR = Path(__file__).parent / "reasoning_results"

def save_run(loop_entry: LoopResult | VotingResult, directs: list[DirectResult],
             task: str, judge: JudgeResult | None = None) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    mode = "vote" if isinstance(loop_entry, VotingResult) else "loop"
    run_dir = RESULTS_DIR / f"{mode}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Build loop section of payload
    if isinstance(loop_entry, VotingResult):
        vr = loop_entry
        loop_payload: dict = {
            "type": "voting",
            "model": vr.model,
            "n_runs": vr.n_runs,
            "total_elapsed_sec": round(vr.total_elapsed_sec, 2),
            "total_prompt_tokens": vr.total_prompt_tokens,
            "total_completion_tokens": vr.total_completion_tokens,
            "total_cost_usd": vr.total_cost(),
            "consensus_answer": vr.consensus_answer,
            "consensus_error": vr.consensus_error,
            "runs": [
                {
                    "run_index": i,
                    "total_elapsed_sec": round(r.total_elapsed_sec, 2),
                    "total_cost_usd": r.total_cost(),
                    "final_answer": r.final_answer,
                    "stages": [
                        {"name": s.name, "output": s.output,
                         "completion_tokens": s.completion_tokens, "error": s.error}
                        for s in r.stages
                    ],
                }
                for i, r in enumerate(vr.runs, 1)
            ],
        }
    else:
        lr = loop_entry
        loop_payload = {
            "type": "single",
            "model": lr.model,
            "total_elapsed_sec": round(lr.total_elapsed_sec, 2),
            "total_prompt_tokens": lr.total_prompt_tokens,
            "total_completion_tokens": lr.total_completion_tokens,
            "total_cost_usd": lr.total_cost(),
            "stages": [
                {"name": s.name, "elapsed_sec": round(s.elapsed_sec, 2),
                 "prompt_tokens": s.prompt_tokens, "completion_tokens": s.completion_tokens,
                 "cost_usd": s.cost(lr.model), "output": s.output, "error": s.error}
                for s in lr.stages
            ],
        }

    payload = {
        "run_timestamp": ts,
        "task": task,
        "loop": loop_payload,
        "direct": [
            {
                "label": d.label, "model": d.model,
                "elapsed_sec": round(d.elapsed_sec, 2),
                "prompt_tokens": d.prompt_tokens,
                "completion_tokens": d.completion_tokens,
                "reasoning_tokens": d.reasoning_tokens,
                "cost_usd": d.cost(), "output": d.output, "error": d.error,
            }
            for d in directs
        ],
        "judge": {
            "model": judge.judge_model if judge else None,
            "elapsed_sec": round(judge.elapsed_sec, 2) if judge else None,
            "cost_usd": judge.cost() if judge else None,
            "winner": judge.winner if judge else None,
            "judge_notes": judge.judge_notes if judge else None,
            "error": judge.error if judge else None,
            "scores": [
                {"label": s.label, "correctness": s.correctness,
                 "reasoning": s.reasoning, "completeness": s.completeness,
                 "total": s.total, "verdict": s.verdict, "explanation": s.explanation}
                for s in (judge.scores if judge else [])
            ],
        },
    }
    (run_dir / "results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    # human-readable txt
    import io, sys
    buf = io.StringIO()
    old = sys.stdout; sys.stdout = buf
    if isinstance(loop_entry, VotingResult):
        print_voting_result(loop_entry)
    else:
        print_loop_result(loop_entry)
    for dr in directs:
        print_direct_result(dr)
    print_comparison(loop_entry, directs)
    if judge:
        print_judge_verdict(judge)
    sys.stdout = old
    (run_dir / "results.txt").write_text(buf.getvalue())

    return run_dir


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthetic reasoning loop vs native reasoning models"
    )
    parser.add_argument("--task", default=DEFAULT_TASK,
                        help="Logic/reasoning problem to test")
    parser.add_argument("--loop-model", default=DEFAULT_LOOP_MODEL,
                        help=f"Cheap model for the reasoning loop (default: {DEFAULT_LOOP_MODEL})")
    parser.add_argument("--native-model", default=DEFAULT_NATIVE_MODEL,
                        help=f"Native reasoning model for direct comparison (default: {DEFAULT_NATIVE_MODEL})")
    parser.add_argument("--chat-model", default=DEFAULT_CHAT_MODEL,
                        help=f"Flagship chat model for direct comparison (default: {DEFAULT_CHAT_MODEL})")
    parser.add_argument("--runs", type=int, default=1,
                        help="Number of voting runs for self-consistency (default: 1 = single loop)")
    parser.add_argument("--stages", nargs="+", choices=ALL_STAGE_NAMES,
                        default=ALL_STAGE_NAMES,
                        help="Which loop stages to run (default: all 5)")
    parser.add_argument("--skip-native", action="store_true",
                        help="Skip the native reasoning model call")
    parser.add_argument("--skip-chat", action="store_true",
                        help="Skip the flagship chat model call")
    parser.add_argument("--skip-judge", action="store_true",
                        help=f"Skip correctness evaluation (judge: {JUDGE_MODEL})")
    parser.add_argument("--no-save", action="store_true",
                        help="Skip saving results to reasoning_results/")
    args = parser.parse_args()

    print(banner("SYNTHETIC REASONING DEMO", "═"))
    print(f"\n  Task        : {args.task[:100]}{'...' if len(args.task) > 100 else ''}")
    if args.runs > 1:
        print(f"  Loop model  : {args.loop_model}  ({len(args.stages)} stages × {args.runs} runs — majority vote)")
    else:
        print(f"  Loop model  : {args.loop_model}  ({len(args.stages)} stages: {', '.join(s.upper() for s in args.stages)})")
    print(f"  Compare vs  : ", end="")
    comparisons = []
    if not args.skip_native: comparisons.append(f"{args.native_model} (native reasoning)")
    if not args.skip_chat:   comparisons.append(f"{args.chat_model} (direct prompt)")
    print("  +  ".join(comparisons) if comparisons else "nothing")

    # ── Run the loop / voting ──────────────────────────────────────────────────
    from openai import OpenAI  # noqa: F401 — ensure importable before we start

    loop_entry: LoopResult | VotingResult
    if args.runs > 1:
        print(f"\n🗳️   Running self-consistency: {args.runs} parallel runs on {args.loop_model}…")
        voting_result = run_voting_loop(args.task, args.loop_model, args.stages, args.runs)
        loop_entry = voting_result
    else:
        print(f"\n🔄  Running reasoning loop on {args.loop_model} ({len(args.stages)} stages)...")
        loop_entry = _run_loop_with_progress(args.task, args.loop_model, args.stages)

    # ── Run direct comparisons ─────────────────────────────────────────────────
    direct_results: list[DirectResult] = []

    if not args.skip_native:
        print(f"\n🎯  Running direct prompt on {args.native_model} (effort=high)…", end="", flush=True)
        dr = run_direct(args.task, args.native_model,
                        label=f"{args.native_model} direct (native reasoning)",
                        reasoning_effort="high")
        print(f" ✓ {dr.elapsed_sec:.1f}s" if not dr.error else " ✗ ERROR")
        direct_results.append(dr)

    if not args.skip_chat:
        print(f"\n💬  Running direct prompt on {args.chat_model} (no reasoning)…", end="", flush=True)
        dr = run_direct(args.task, args.chat_model,
                        label=f"{args.chat_model} direct (chat only)")
        print(f" ✓ {dr.elapsed_sec:.1f}s" if not dr.error else " ✗ ERROR")
        direct_results.append(dr)

    # ── Print results ──────────────────────────────────────────────────────────
    if isinstance(loop_entry, VotingResult):
        print_voting_result(loop_entry)
    else:
        print_loop_result(loop_entry)
    for dr in direct_results:
        print_direct_result(dr)
    print_comparison(loop_entry, direct_results)

    # ── Judge evaluation ───────────────────────────────────────────────────────
    judge_result: JudgeResult | None = None
    if not args.skip_judge and (loop_entry.final_answer or any(d.output for d in direct_results)):
        print(f"\n⚖️   Running judge evaluation ({JUDGE_MODEL})…", end="", flush=True)
        judge_result = run_judge(args.task, loop_entry, direct_results)
        status = f" ✓ {judge_result.elapsed_sec:.1f}s" if not judge_result.error else f" ✗ {judge_result.error}"
        print(status)
        print_judge_verdict(judge_result)

    # ── Save ───────────────────────────────────────────────────────────────────
    if not args.no_save:
        run_dir = save_run(loop_entry, direct_results, args.task, judge_result)
        print(f"  💾  Saved to: {run_dir}/")
        print(f"       • results.json  — full data (scores, all stage outputs)")
        print(f"       • results.txt   — human-readable report\n")


def _run_loop_with_progress(task: str, model: str, stage_names: list[str]) -> LoopResult:
    """Like run_loop but prints live progress per stage."""
    result = LoopResult(model=model, task=task)

    history: list[dict] = [{"role": "system", "content": STAGE_SYSTEM}]
    history.append({
        "role": "user",
        "content": f"Here is the problem you will work through in stages:\n\n{task}",
    })
    history.append({
        "role": "assistant",
        "content": "Understood. I will work through this problem stage by stage as instructed.",
    })

    active_stages = [s for s in STAGES if s[0].lower() in stage_names]
    total = len(active_stages)

    for i, (stage_name, stage_prompt) in enumerate(active_stages, 1):
        print(f"   [{i}/{total}] {stage_name:<10} … ", end="", flush=True)
        history.append({"role": "user", "content": f"--- Stage: {stage_name} ---\n{stage_prompt}"})

        t0 = time.perf_counter()
        try:
            text, ptok, ctok, _ = _chat(model, history)
            elapsed = time.perf_counter() - t0
            sr = StageResult(name=stage_name, prompt_tokens=ptok,
                             completion_tokens=ctok, elapsed_sec=elapsed, output=text)
            history.append({"role": "assistant", "content": text})
            stage_cost = sr.cost(model)
            cstr = f"${stage_cost:.5f}" if stage_cost else "n/a"
            print(f"✓  {elapsed:.1f}s  {ctok} tokens  {cstr}")
        except Exception as e:  # noqa: BLE001
            elapsed = time.perf_counter() - t0
            sr = StageResult(name=stage_name, prompt_tokens=0, completion_tokens=0,
                             elapsed_sec=elapsed, output="", error=str(e))
            history.append({"role": "assistant", "content": f"[Error in stage {stage_name}]"})
            print(f"✗ ERROR: {e}")

        result.stages.append(sr)

    return result


if __name__ == "__main__":
    main()
