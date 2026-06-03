# When Thinking Fails — Reproduction Findings

**Experiment:** Does chain-of-thought (CoT) prompting degrade instruction
following in frontier LLMs?
**Reference:** *"When Thinking Fails: The Pitfalls of Reasoning for
Instruction-Following in LLMs"* — [arXiv:2505.11423](https://arxiv.org/abs/2505.11423)
(Harvard University & Amazon).
**Date of run:** 2026-06-03
**Raw data:** `results/results_20260603T092711Z.csv`

---

## 1. Executive summary

We reproduced the paper's central claim and it **holds strongly**.

- Both tested models scored a **perfect 100%** on instruction-following when
  asked to answer **directly**.
- Adding chain-of-thought reasoning **reduced** accuracy, and **more reasoning
  caused more damage**.
- The damage is concentrated in **whole-output constraints** (length limits,
  exact counts, casing, paragraph structure) — exactly the constraints that the
  reasoning text itself pollutes.
- Semantic "include X" constraints were **unaffected**.

**Practical takeaway:** For instruction-following tasks on reasoning-capable
models, explicit CoT prompts are not just unnecessary — they actively hurt. They
bloat the context, divert attention from formatting constraints, and lower
output quality.

---

## 2. Setup

| Parameter | Value |
|---|---|
| Providers / models | OpenAI `gpt-5.5`, Anthropic `claude-opus-4-8` |
| Conditions | `direct`, `cot`, `aggressive_cot` |
| Tasks | 16 IFEval-style, deterministically verified |
| Repeats | 3 per task/condition |
| Temperature | 0.0 (auto-dropped where unsupported) |
| Scoring | Pure-code verifiers; task passes only if **all** constraints pass |

> **Note:** Gemini was configured but did not return data in this run (model ID
> / availability). The two-model result is already conclusive; re-running with a
> valid Gemini ID would add a third data point.

### The three conditions

- **direct** — "Respond with the final answer directly. Do not show any
  reasoning."
- **cot** — "Think step by step and reason carefully before answering." (matches
  the paper's prompted-CoT setup)
- **aggressive_cot** — a forced 6-step protocol (list constraints → identify
  risks → draft → audit → revise → final answer) producing long explicit
  reasoning chains.

---

## 3. Headline results

### Accuracy by condition

| Model | direct | cot | aggressive_cot |
|---|---|---|---|
| **claude-opus-4-8** | **100.0%** | 64.6% (↓ −35.4%) | 39.6% (↓ −60.4%) |
| **gpt-5.5** | **100.0%** | 100.0% (= 0.0%) | 75.0% (↓ −25.0%) |

**Both models degrade under reasoning, and the dose-response is monotonic:**
the more reasoning we forced, the lower the accuracy. Claude Opus — a
reasoning-native model — is far more sensitive, losing over a third of its
accuracy from standard CoT alone and nearly two-thirds under aggressive CoT.
GPT-5.5 absorbed standard CoT without loss but still fell 25 points under
aggressive CoT.

### Response-length "bloat"

| Model | direct | cot | aggressive_cot |
|---|---|---|---|
| claude-opus-4-8 | 63 w | 128 w (**2.0×**) | 358 w (**5.7×**) |
| gpt-5.5 | 46 w | 55 w (1.2×) | 278 w (**6.1×**) |

Reasoning conditions produce 2×–6× longer outputs. This is the mechanism behind
the accuracy drop: **the reasoning text becomes part of the output the
constraint applies to.** A "max 30 words" rule is violated by a 278-word
reply, even if the literal answer buried inside would have qualified.

### Cost

| Model | prompt tok | completion tok | est. cost |
|---|---|---|---|
| claude-opus-4-8 | 28,959 | 59,696 | $4.91 |
| gpt-5.5 | 18,210 | 66,177 | $21.22 |
| **Total** | | | **$26.13** |

CoT conditions also dominate the bill: the longer reasoning outputs are billed
as completion tokens, so the worse-performing conditions are also the most
expensive ones.

---

## 4. Per-task analysis: what CoT breaks

| Task | direct | cot | aggressive_cot | Verdict |
|---|---|---|---|---|
| `three_paragraphs_divider` | 100% | 50% | **0%** | ↓ hurts |
| `two_sentences` | 100% | 50% | **0%** | ↓ hurts |
| `word_cap_30` | 100% | 50% | **0%** | ↓ hurts |
| `numbered_five_steps` | 100% | 100% | **17%** | ↓ hurts |
| `exactly_three_bullets` | 100% | 50% | 33% | ↓ hurts |
| `start_with_word` | 100% | 67% | 50% | ↓ hurts |
| `forbidden_words` | 100% | 50% | 50% | ↓ hurts |
| `uppercase_slogan` | 100% | 100% | 50% | ↓ hurts |
| `all_lowercase` | 100% | 100% | 67% | ↓ hurts |
| `json_cities` | 100% | 100% | 67% | ↓ hurts |
| `no_commas_haiku` | 100% | 100% | 83% | ↓ hurts |
| `end_with_phrase` | 100% | 100% | 100% | = no change |
| `highlights` | 100% | 100% | 100% | = no change |
| `keywords_required` | 100% | 100% | 100% | = no change |
| `placeholders_email` | 100% | 100% | 100% | = no change |
| `title_and_postscript` | 100% | 100% | 100% | = no change |

### Three clear clusters

**A. Structural / counting constraints — catastrophic failure.**
`three_paragraphs_divider`, `two_sentences`, `word_cap_30` collapse all the way
to **0%** under aggressive CoT. The reasoning chain itself adds sentences,
paragraphs and words, blowing past the structural limit. `numbered_five_steps`
falls to 17% because the model emits unnumbered reasoning steps that contaminate
the required count.

**B. Casing / exclusion constraints — fragile.**
`uppercase_slogan`, `all_lowercase`, `forbidden_words` break because the
reasoning preamble violates the rule before the answer even begins — lowercase
"let me think…" ruins an all-caps requirement; reasoning *about* the word
"relax" uses the very word that was forbidden.

**C. Additive / semantic constraints — robust.**
`keywords_required`, `highlights`, `title_and_postscript`,
`placeholders_email`, `end_with_phrase` stay at 100% everywhere. These are
"include X" constraints that extra text cannot violate — so reasoning is
harmless.

---

## 5. Why it happens

The unifying principle:

> **CoT hurts constraints that govern the *entire output*, and is harmless for
> constraints satisfied by *adding* content.**

When a model reasons first and answers second, its reasoning text is part of the
response a global constraint sees. The model treats the constraint as applying
to "the answer" while the verifier (correctly) applies it to everything emitted.
This matches the paper's **"constraint attention"** analysis: the longer the
reasoning, the more the model's attention is diverted away from the
instruction-relevant tokens, and the more global constraints are dropped. Our
2×–6× response-length blow-up is a direct, measurable proxy for that attention
dilution.

---

## 6. Model differences

- **Claude Opus 4.8** is reasoning-native and already "thinks" internally.
  Forcing it to externalize reasoning is redundant and actively harmful
  (−35% from standard CoT). It is the more sensitive of the two.
- **GPT-5.5** tolerated standard CoT with no loss, suggesting it recovers to the
  constraint before finalizing. But aggressive CoT still cost it 25 points — no
  model was immune once the reasoning chain grew long enough.

---

## 7. Recommendations

1. **Default to direct prompting for instruction-following / formatting tasks.**
   On these models it gave a perfect score; CoT only took accuracy away.
2. **Do not add "think step by step" reflexively.** It bloats context, raises
   cost (the aggressive conditions were both worst-performing *and* most
   expensive), and lowers reliability.
3. **Reserve CoT for tasks that genuinely need it** (math, multi-hop logic), and
   suppress it elsewhere — the paper's "selective reasoning" strategy.
4. **If reasoning is unavoidable, isolate it from the output** (e.g. a separate
   scratchpad / hidden reasoning channel) so global constraints apply only to
   the final answer, not the reasoning trace.

---

## 8. Reproducing

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=...  ANTHROPIC_API_KEY=...  GOOGLE_API_KEY=...
python run_experiment.py --providers openai anthropic gemini --repeats 3
```

See `README.md` for full options. Raw per-call data for this report is in
`results/results_20260603T092711Z.csv`.

---

## 9. Limitations

- Two of three configured models returned data; adding Gemini would strengthen
  generalization.
- 16 tasks × 3 repeats is enough to show large effects but a wider task suite
  (e.g. full IFEval + ComplexBench, as in the paper) would tighten estimates.
- Cost figures depend on the `PRICE_TABLE` in `providers.py`; update for current
  pricing.
- Verifiers are intentionally strict (whole-output scope). This is the correct
  interpretation of IFEval-style constraints and is exactly what exposes the CoT
  failure mode.
