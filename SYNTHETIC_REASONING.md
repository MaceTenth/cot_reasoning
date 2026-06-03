# Synthetic Reasoning — `synthetic_reasoning.py`

## What This Experiment Shows

This script tests a provocative idea: **can you turn a cheap, "dumb" model into a reasoning model by explicitly building the reasoning loop yourself?**

Native reasoning models like `o4-mini` internally run a multi-stage thinking process before answering. This experiment replicates that loop as explicit API calls on `gpt-4o-mini` — and compares the result against the native models.

---

## The Hypothesis

> *o-series models (o4-mini, o3) internally run something like: explore the problem → make a plan → solve it → critique the solution → refine the answer. What if we do that manually with a much cheaper model?*

Based on Wang et al. (2022) **Self-Consistency** paper: running the same reasoning problem multiple times and taking the majority/consensus answer significantly improves accuracy — especially for weak models.

---

## The Three Approaches Compared

| Approach | Model | Method | API calls |
|---|---|---|---|
| **Synthetic loop** | `gpt-4o-mini` | 5-stage explicit loop | 5 per run |
| **Self-consistency** | `gpt-4o-mini` | N parallel loop runs + consensus vote | 5N + 1 |
| **Native reasoning** | `o4-mini` | Single call, effort=high | 1 |
| **Direct chat** | `gpt-5.5` | Single call, no reasoning | 1 |

---

## The 5-Stage Reasoning Loop

Each stage builds on the previous, with the full conversation history carried forward:

```
Stage 1 — EXPLORE  : Read the problem carefully. List all constraints.
                     What are the key entities? What makes this hard?

Stage 2 — PLAN     : Outline an approach. Don't solve yet.
                     Break it into steps. Identify potential pitfalls.

Stage 3 — SOLVE    : Execute the plan step by step.
                     Show all intermediate reasoning. Reach a conclusion.

Stage 4 — CRITIQUE : Review your solution critically.
                     Are there mistakes? Gaps? Wrong assumptions?

Stage 5 — REFINE   : Produce a final, clean, corrected answer.
                     Be definitive — no hedging.
```

---

## Self-Consistency Mode (`--runs N`)

When `--runs N` is passed (N > 1):

1. **N independent runs** are executed **in parallel** (using `ThreadPoolExecutor`) at temperature 0.8 to introduce diversity
2. Each run produces its own final answer via the 5-stage loop
3. A **consensus call** reads all N answers and synthesises the majority conclusion
4. The consensus answer is what goes to the judge for scoring

This mirrors the Wang et al. self-consistency approach: diverse reasoning paths → majority vote → more reliable answer.

---

## Correctness Evaluation (Judge)

After all runs complete, a neutral judge model (`gpt-4o`) evaluates every answer on three dimensions (1–10 each):

| Dimension | What it measures |
|---|---|
| **Correctness** | Is the final conclusion factually/logically right? |
| **Reasoning** | Is the step-by-step logic valid, clear, and complete? |
| **Completeness** | Does the answer address everything the question asks? |

Maximum score: **30/30**. The judge picks a winner and provides per-answer explanations.

---

## Experimental Results

Using the default task (Three Logicians puzzle):

| Approach | Score | Cost | Time |
|---|---|---|---|
| gpt-4o-mini single loop | 19/30 ❌ | $0.0016 | 62s |
| gpt-4o-mini ×3 self-consistency | **29/30** ✅ | $0.0053 | 63s |
| o4-mini native (effort=high) | **30/30** ✅ | $0.0099 | 11s |
| gpt-5.5 direct chat | **29/30** ✅ | $0.0177 | 11s |

### Key Findings

1. **Self-consistency works** — jumping from 19/30 to 29/30 with 3 parallel runs
2. **3× gpt-4o-mini loops ≈ half the cost of 1× o4-mini** — same quality, ~2× cheaper
3. **Wall time stays similar** — parallel runs mean N loops barely adds latency vs 1
4. **The single loop fails due to weak deductive logic** in the SOLVE stage — the model hedges instead of committing to conclusions; majority vote overrides this
5. **Native reasoning (o4-mini) remains the gold standard** — cleaner reasoning, slightly higher score, faster

---

## Cost Comparison

| Model | Input | Output | Relative |
|---|---|---|---|
| gpt-4o-mini | $0.15/MTok | $0.60/MTok | 1× |
| o4-mini | $1.10/MTok | $4.40/MTok | ~7× |
| gpt-4o | $2.50/MTok | $10.00/MTok | ~17× |
| gpt-5.5 | $5.00/MTok | $30.00/MTok | ~47× |

---

## Usage

```bash
# Default: single loop, compare vs o4-mini + gpt-5.5
python3 synthetic_reasoning.py

# Self-consistency: 3 parallel runs + majority vote
python3 synthetic_reasoning.py --runs 3

# Custom task
python3 synthetic_reasoning.py --task "Is 1033 prime? Show your work."

# Only specific stages (skip CRITIQUE)
python3 synthetic_reasoning.py --stages explore plan solve refine

# Skip direct model comparisons
python3 synthetic_reasoning.py --skip-native --skip-chat

# Skip judge evaluation
python3 synthetic_reasoning.py --skip-judge

# Don't save results
python3 synthetic_reasoning.py --no-save
```

## Results

Results are saved to `reasoning_results/loop_<timestamp>/` (single run) or `reasoning_results/vote_<timestamp>/` (voting):
- `results.json` — full data: all stage outputs, scores, costs, judge evaluation
- `results.txt` — human-readable terminal output

---

## Why This Matters

This is a practical demonstration of two research ideas:

1. **Explicit reasoning loops can compensate for model weakness** — you don't always need a smarter model; sometimes a structured prompt pipeline gets you there at lower cost
2. **Self-consistency (majority voting) is a powerful, underused technique** — it's model-agnostic, requires no fine-tuning, and significantly boosts reliability on logic tasks
