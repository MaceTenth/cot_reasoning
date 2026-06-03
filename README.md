# Does Chain-of-Thought Hurt Instruction-Following?

A reproduction harness for the core claim of
**"When Thinking Fails: The Pitfalls of Reasoning for Instruction-Following in
LLMs"** ([arXiv:2505.11423](https://arxiv.org/abs/2505.11423), Harvard / Amazon).

> **Claim:** Explicit chain-of-thought (CoT) reasoning can *degrade* a model's
> ability to follow simple, verifiable instructions.

This project tests that claim across three frontier providers — OpenAI, Anthropic
and Google — by running the same instruction-following tasks under different
prompting conditions and scoring the outputs with deterministic verifiers (no
LLM judge required).

See [`REPORT.md`](REPORT.md) for the full write-up of findings.

---

## How it works

For every **task × model × condition × repeat**, the harness:

1. Sends the task prompt to the model under one of three conditions.
2. Captures the raw output, token usage, and response length.
3. Scores the output with a Python verifier (pass/fail per constraint).
4. Aggregates accuracy, response-length "bloat", and estimated cost.

### The three conditions (`conditions.py`)

| Condition | What the model is told |
|---|---|
| `direct` | "Answer directly. Do not show any reasoning." |
| `cot` | "Think step by step and reason before answering." (the paper's setup) |
| `aggressive_cot` | A 6-step forced reasoning protocol before the final answer. |

### The tasks (`ifeval_prompts.py`)

16 IFEval-style tasks, each with a base request plus a **machine-checkable**
constraint, e.g.:

- output valid JSON
- exactly three bullet points
- all lowercase / all uppercase
- no commas anywhere
- at most 30 words
- end with the exact phrase "Keep going."
- avoid specific forbidden words

A task **passes only if all its constraints pass**. Verification is pure code,
so results are 100% reproducible from the raw output.

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt`:
```
openai
anthropic
google-genai
```

> The SDKs are imported lazily, so you only need the ones for the providers you
> actually run.

### 2. Set API keys

The official SDKs read these from the environment automatically. Export the
ones you need:

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="..."        # Gemini (or GEMINI_API_KEY)
```

There is **no `.env` loading** built in — just export in your shell.

---

## Running

### Offline smoke test (no keys needed)

Uses a built-in `mock` provider to exercise the whole pipeline:

```bash
python run_experiment.py --providers mock --repeats 1
```

### Full 3-provider run

```bash
python run_experiment.py --providers openai anthropic gemini --repeats 3
```

### Conservative run (strict rate limits)

```bash
python run_experiment.py --providers openai anthropic gemini \
  --repeats 3 --concurrency 2 --max-retries 6 --retry-base 5
```

---

## Command-line options

| Flag | Default | Description |
|---|---|---|
| `--providers` | `mock` | One or more of `openai anthropic gemini mock` |
| `--openai-model` | `gpt-5.5` | Override the OpenAI model ID |
| `--anthropic-model` | `claude-opus-4-8` | Override the Anthropic model ID |
| `--gemini-model` | `gemini-3` | Override the Gemini model ID |
| `--repeats` | `1` | Runs per task/condition (use ≥3 to average noise) |
| `--temperature` | `0.0` | Sampling temperature (auto-dropped for models that reject it) |
| `--concurrency` | `5` | Max simultaneous requests **per provider** |
| `--max-retries` | `4` | Retries on rate-limit (429) errors |
| `--retry-base` | `2.0` | Base seconds for exponential backoff |
| `--outdir` | `results` | Where to write the results CSV |

Model IDs are configurable because they evolve quickly. If a default ID is
rejected, override it, e.g.:

```bash
python run_experiment.py --providers openai \
  --openai-model gpt-4o
```

---

## Parallelism & rate limiting

- **All providers run simultaneously** (one thread pool each), so total wall
  time is roughly the slowest single provider rather than the sum of all.
- **Within a provider**, a semaphore caps concurrent requests at
  `--concurrency`.
- **Rate-limit errors** (HTTP 429, "rate limit", "quota", "overloaded") trigger
  **exponential backoff with jitter** and are retried up to `--max-retries`.
- Console output is serialized through a lock so concurrent threads don't
  produce garbled lines.

---

## Output

### Console summary

1. **Accuracy table** — accuracy per condition per model, with Δ vs. `direct`.
2. **Response length table** — average words per reply (the "bloat" effect).
3. **Token usage & estimated cost** — per model, plus a total.
4. **Per-task breakdown** — which constraints CoT breaks vs. leaves intact.

### Raw CSV

Each run writes `results/results_<timestamp>.csv` with one row per call:

| column | meaning |
|---|---|
| `provider`, `model` | who answered |
| `task_id`, `condition`, `repeat` | what was asked |
| `passed` | 1 if all constraints satisfied |
| `error` | API error string, if any |
| `prompt_tokens`, `completion_tokens` | usage from the API |
| `resp_words` | length of the response |
| `per_constraint` | JSON: pass/fail for each individual constraint |
| `output` | the raw model response |

---

## Pricing

Cost estimates use the `PRICE_TABLE` in `providers.py` (USD per 1M tokens).
Update those numbers to match current pricing; unknown models show `unknown`.

---

## Project layout

```
.
├── run_experiment.py    # orchestration, parallelism, scoring, reporting
├── conditions.py        # the 3 prompting conditions (direct / cot / aggressive)
├── ifeval_prompts.py    # 16 tasks + deterministic verifiers
├── providers.py         # OpenAI / Anthropic / Gemini wrappers + pricing + mock
├── requirements.txt
├── README.md
├── REPORT.md            # full findings write-up
└── results/             # timestamped CSV outputs
```

---

## Extending

- **Add a task:** append a `Task(...)` to `TASKS` in `ifeval_prompts.py` with a
  verifier function. Helpers like `word_count_at_most`, `no_commas`,
  `bullet_count_exactly`, `is_valid_json` are provided.
- **Add a condition:** add a system/suffix pair in `conditions.py` and register
  it in `CONDITIONS`.
- **Add a provider:** implement a class with a
  `generate(system, user, temperature) -> ModelResult` method and register it in
  `PROVIDERS`.
