# Reasoning Demo — `reasoning_demo.py`

## What This Experiment Shows

This script demonstrates **how reasoning effort levels affect model quality, latency, and cost**.

The core question: *If a model can "think more", does it actually get better answers? And what does that cost?*

---

## The Setup

Two providers are compared side-by-side, each run at three effort levels (low / medium / high):

| Provider | Model | How effort is controlled |
|---|---|---|
| **OpenAI** | `o4-mini` | `reasoning_effort = low \| medium \| high` |
| **Anthropic** | `claude-sonnet-4-6` | `budget_tokens = 1,024 \| 5,000 \| 16,000` |

### The Task (default)
A classic river-crossing logic puzzle:
> *A farmer needs to cross a river with a fox, a chicken, and a bag of grain. The boat holds only the farmer + one item. The fox eats the chicken; the chicken eats the grain. How does the farmer get everything across safely?*

You can pass any problem with `--task "..."`.

---

## What the API Actually Returns

This is the key difference between the two providers:

### OpenAI (`o4-mini`, `o3`)
- Returns **reasoning token count only** — the actual thinking text is never exposed
- You see: *"Used 847 reasoning tokens"* but not what the model was thinking
- This is by design; OpenAI keeps the scratchpad private

### Anthropic (`claude-sonnet-4-6`)
- Returns the **full thinking text** in `thinking` blocks inside the response
- You see the model's actual internal monologue before it writes its answer
- This is unique to models with Extended Thinking enabled
- **Note:** `claude-opus-4-8` uses "adaptive thinking" (`effort=`) but does *not* expose its reasoning text. Use `claude-sonnet-4-6` to see the actual thinking.

---

## What the Output Shows

For each model × effort level combination:

```
[ o4-mini  |  effort: high ]
  Time     : 8.3s
  Tokens   : 420 prompt  / 1,204 output  (847 reasoning)
  Cost     : $0.0059
  ── Answer ──
  [final answer text]
```

For Anthropic, a `── Thinking ──` block is also printed showing the raw internal reasoning.

A **summary table** at the end compares all runs:
```
Model              Effort    Time    R.Tokens  Out.Tokens    Cost
claude-sonnet-4-6  low       3.1s       1,024         312  $0.0059
claude-sonnet-4-6  medium    6.2s       4,891         298  $0.0187
claude-sonnet-4-6  high     14.8s      15,203         341  $0.0512
```

---

## Key Findings

1. **More effort = longer wall time** — reasoning tokens are generated serially
2. **More effort ≠ always better answer** — for simple problems, `low` is often sufficient
3. **Anthropic's thinking text shows the model genuinely exploring** — trying wrong paths and self-correcting
4. **OpenAI's reasoning token count is a proxy** — more tokens generally means more careful analysis
5. **Cost scales linearly with reasoning tokens** — `high` can be 5–10× the cost of `low`

---

## Usage

```bash
# Default: both providers, all 3 effort levels, farmer puzzle
python3 reasoning_demo.py

# Only OpenAI
python3 reasoning_demo.py --providers openai

# Custom task
python3 reasoning_demo.py --task "Is 1033 a prime number? Show your work."

# Only low and high effort
python3 reasoning_demo.py --efforts low high

# Print full Anthropic thinking text (not truncated)
python3 reasoning_demo.py --full-thinking

# Don't save results
python3 reasoning_demo.py --no-save
```

## Results

Results are saved to `reasoning_results/<timestamp>/`:
- `results.json` — full data including all thinking text, token counts, costs
- `results.txt` — human-readable version of the terminal output
