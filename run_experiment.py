"""Run the CoT-vs-direct instruction-following experiment.

Reproduces the core comparison from "When Thinking Fails: The Pitfalls of
Reasoning for Instruction-Following in LLMs" (arXiv:2505.11423):
each task is run under two conditions (direct, cot, aggressive_cot) for each
model, then scored with deterministic IFEval-style verifiers.

Parallelism
-----------
* All three providers run simultaneously (one thread pool each).
* Within a provider, up to --concurrency requests run at the same time.
* Rate-limit errors (HTTP 429 / "rate limit" messages) trigger exponential
  backoff with jitter before retrying (up to --max-retries attempts).

Examples
--------
# Smoke-test offline:
python run_experiment.py --providers mock --repeats 1

# Real run, 5 concurrent requests per provider, auto-retry on rate limits:
python run_experiment.py --providers openai anthropic gemini --repeats 3
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from conditions import CONDITIONS, build_messages
from ifeval_prompts import get_tasks
from providers import build_provider, estimate_cost

# ── rate-limit keywords we detect across all three SDKs ──────────────────────
_RATE_LIMIT_SIGNALS = ("429", "rate limit", "rate_limit", "too many requests",
                       "ratelimit", "quota", "overloaded")


def _is_rate_limit(error_str: str) -> bool:
    low = error_str.lower()
    return any(sig in low for sig in _RATE_LIMIT_SIGNALS)


def _call_with_retry(provider, system, user, temperature,
                     semaphore, max_retries, retry_base):
    """Acquire semaphore, call provider.generate, retry on rate-limit errors."""
    attempt = 0
    while True:
        with semaphore:
            res = provider.generate(system, user, temperature=temperature)

        if res.error and _is_rate_limit(res.error) and attempt < max_retries:
            # Exponential backoff with full jitter.
            wait = retry_base * (2 ** attempt) * (0.5 + random.random() * 0.5)
            attempt += 1
            _print_lock_print(
                f"  ⏳ Rate-limited by {provider.name}, "
                f"retry {attempt}/{max_retries} in {wait:.1f}s …"
            )
            time.sleep(wait)
            continue

        return res


# Thread-safe print via a module-level lock.
_print_lock = threading.Lock()


def _print_lock_print(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


def _preflight(provider, temperature: float) -> bool:
    """Send a trivial request. Returns True if the provider is reachable."""
    res = provider.generate(
        "You are a helpful assistant.",
        "Reply with the single word: OK",
        temperature=temperature,
    )
    if res.error:
        _print_lock_print(f"  ✗ Preflight FAILED: {res.error}")
        return False
    _print_lock_print(f"  ✓ Preflight OK — model replied: {res.text.strip()[:60]!r}")
    return True


def _run_provider(pname, provider, tasks, repeats, temperature,
                  concurrency, max_retries, retry_base):
    """Run all tasks for one provider in parallel. Returns list of row dicts."""
    semaphore = threading.Semaphore(concurrency)
    rows: list[dict] = []
    rows_lock = threading.Lock()

    # Build the full work list so we can show a progress counter.
    work = [
        (task, condition, rep)
        for task in tasks
        for condition in CONDITIONS
        for rep in range(repeats)
    ]
    total = len(work)
    done_count = [0]  # mutable counter shared across threads

    def _do_one(task, condition, rep):
        system, user = build_messages(condition, task.prompt)
        res = _call_with_retry(
            provider, system, user, temperature,
            semaphore, max_retries, retry_base,
        )
        score = task.score(res.text)
        resp_words = len(res.text.split())
        row = {
            "provider": pname,
            "model": provider.model,
            "task_id": task.id,
            "condition": condition,
            "repeat": rep,
            "passed": int(score["all_passed"]),
            "error": res.error or "",
            "prompt_tokens": res.prompt_tokens,
            "completion_tokens": res.completion_tokens,
            "resp_words": resp_words,
            "output": res.text,
            "per_constraint": json.dumps(score["per_constraint"]),
        }
        with rows_lock:
            rows.append(row)
            done_count[0] += 1
            n = done_count[0]

        flag = "ok " if score["all_passed"] else "FAIL"
        if res.error:
            flag = "ERR "
        _print_lock_print(
            f"  [{flag}] [{pname}] {task.id:24s} {condition:14s} "
            f"rep{rep}  ({resp_words}w)  [{n}/{total}]"
        )
        if res.error:
            _print_lock_print(f"         └─ ERROR: {res.error}")

    with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix=pname) as ex:
        futures = {ex.submit(_do_one, task, cond, rep): (task.id, cond, rep)
                   for task, cond, rep in work}
        for fut in as_completed(futures):
            exc = fut.exception()
            if exc:
                tid, cond, rep = futures[fut]
                _print_lock_print(
                    f"  [CRASH] [{pname}] {tid} {cond} rep{rep}: {exc}"
                )

    return rows


def run(providers, models, repeats, temperature, outdir,
        concurrency, max_retries, retry_base):
    tasks = get_tasks()
    os.makedirs(outdir, exist_ok=True)
    all_rows: list[dict] = []
    all_rows_lock = threading.Lock()

    # Preflight all providers first (sequential — before spawning pools).
    active_providers = []
    for pname in providers:
        provider = build_provider(pname, models.get(pname))
        print(f"\n=== Provider: {pname} (model={provider.model}) ===")
        if not _preflight(provider, temperature):
            print(f"  Skipping {pname} — fix the error above and re-run.")
        else:
            active_providers.append((pname, provider))

    if not active_providers:
        print("\nNo providers available. Exiting.")
        return []

    n_tasks = len(tasks) * len(CONDITIONS) * repeats
    print(
        f"\nRunning {n_tasks} calls × {len(active_providers)} provider(s) "
        f"[concurrency={concurrency}, max_retries={max_retries}] …\n"
    )

    # Each provider gets its own thread pool; all run in parallel.
    provider_futures = {}
    provider_executor = ThreadPoolExecutor(
        max_workers=len(active_providers),
        thread_name_prefix="provider",
    )
    for pname, provider in active_providers:
        fut = provider_executor.submit(
            _run_provider,
            pname, provider, tasks, repeats, temperature,
            concurrency, max_retries, retry_base,
        )
        provider_futures[fut] = pname

    for fut in as_completed(provider_futures):
        pname = provider_futures[fut]
        try:
            rows = fut.result()
            with all_rows_lock:
                all_rows.extend(rows)
            print(f"\n  ✓ {pname} finished ({len(rows)} results)")
        except Exception as e:  # noqa: BLE001
            print(f"\n  ✗ {pname} crashed: {e}")

    provider_executor.shutdown(wait=False)

    _write_outputs(all_rows, outdir)
    _summarize(all_rows)
    return all_rows


def _write_outputs(rows, outdir):
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw_path = os.path.join(outdir, f"results_{ts}.csv")
    fields = [
        "provider", "model", "task_id", "condition", "repeat",
        "passed", "error", "prompt_tokens", "completion_tokens", "resp_words",
        "per_constraint", "output",
    ]
    with open(raw_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nRaw results written to {raw_path}")


def _summarize(rows):
    # accuracy[(provider, condition)] = [passes, total]
    agg: dict = defaultdict(lambda: [0, 0])
    # avg response words[(provider, condition)] = [total_words, count]
    words_agg: dict = defaultdict(lambda: [0, 0])
    # token tracking[(provider, model)] = [prompt_tok, completion_tok]
    tok_agg: dict = defaultdict(lambda: [0, 0])

    for r in rows:
        key = (r["provider"], r["condition"])
        agg[key][0] += r["passed"]
        agg[key][1] += 1
        words_agg[key][0] += r.get("resp_words", 0)
        words_agg[key][1] += 1
        model_key = (r["provider"], r["model"])
        tok_agg[model_key][0] += r.get("prompt_tokens", 0)
        tok_agg[model_key][1] += r.get("completion_tokens", 0)

    model_for = {r["provider"]: r["model"] for r in rows}
    providers = sorted({r["provider"] for r in rows})
    conditions = ["direct", "cot", "aggressive_cot"]

    # ---- Accuracy table ----
    print("\n================ SUMMARY: instruction-following accuracy ================")
    header = f"{'provider':12s} {'model':22s}" + "".join(f" {c:>14s}" for c in conditions)
    print(header)
    print("-" * (12 + 22 + 15 * len(conditions)))
    for p in providers:
        mdl = model_for.get(p, "?")
        accs = []
        for c in conditions:
            ps, tot = agg[(p, c)]
            accs.append(ps / tot if tot else None)
        base = accs[0]  # direct is the baseline
        parts = f"{p:12s} {mdl:22s}"
        for acc in accs:
            if acc is None:
                parts += f"       (n/a)"
            else:
                parts += f"      {acc:6.1%}"
        print(parts)
        # Print deltas vs direct on a sub-line.
        delta_parts = f"{'  Δ vs direct':34s}"
        for i, acc in enumerate(accs):
            if i == 0 or acc is None or base is None:
                delta_parts += f"{'':14s}"
            else:
                d = acc - base
                arrow = "↓" if d < 0 else ("↑" if d > 0 else "=")
                delta_parts += f"    {arrow} {d:+.1%}    "
        print(delta_parts)
    print("-" * (12 + 22 + 15 * len(conditions)))
    print("↓ = CoT hurt accuracy vs direct (supports paper's claim)")

    # ---- Response length (bloat) table ----
    print("\n================ RESPONSE LENGTH (avg words per reply) ================")
    print("Longer responses in CoT conditions = context bloat / attention dilution.")
    header2 = f"{'provider':12s}" + "".join(f" {c:>16s}" for c in conditions)
    print(header2)
    print("-" * (12 + 17 * len(conditions)))
    for p in providers:
        parts = f"{p:12s}"
        base_words = None
        word_vals = []
        for c in conditions:
            tw, cnt = words_agg[(p, c)]
            avg = tw / cnt if cnt else None
            word_vals.append(avg)
            if c == "direct":
                base_words = avg
        for avg in word_vals:
            if avg is None:
                parts += f"{'n/a':>16s}"
            elif base_words and avg != base_words:
                ratio = avg / base_words
                parts += f"  {avg:6.0f}w ({ratio:.1f}x)"
            else:
                parts += f"  {avg:6.0f}w (base)"
        print(parts)
    print("-" * (12 + 17 * len(conditions)))

    # ---- Token usage and cost ----
    print("\n================ TOKEN USAGE & ESTIMATED COST ================")
    print(f"{'provider':12s} {'model':24s} {'prompt_tok':>12s} {'compl_tok':>10s} {'est_cost':>10s}")
    print("-" * 70)
    total_cost = 0.0
    for p in providers:
        mdl = model_for.get(p, "?")
        pt, ct = tok_agg[(p, mdl)]
        cost = estimate_cost(mdl, pt, ct)
        cost_str = f"${cost:.4f}" if cost is not None else "unknown"
        if cost:
            total_cost += cost
        print(f"{p:12s} {mdl:24s} {pt:12,d} {ct:10,d} {cost_str:>10s}")
    print("-" * 70)
    print(f"{'TOTAL':38s} ${total_cost:.4f}")

    # ---- Per-task breakdown ----
    print("\n--- Per-task breakdown (accuracy per condition, averaged over providers) ---")
    task_agg: dict = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for r in rows:
        task_agg[r["task_id"]][r["condition"]][0] += r["passed"]
        task_agg[r["task_id"]][r["condition"]][1] += 1
    hdr = f"  {'task':24s}" + "".join(f" {c:>14s}" for c in conditions) + "  verdict"
    print(hdr)
    for tid in sorted(task_agg):
        accs = []
        for c in conditions:
            a = task_agg[tid][c]
            accs.append(a[0] / a[1] if a[1] else None)
        base = accs[0]
        row_str = f"  {tid:24s}"
        for acc in accs:
            row_str += f"  {'n/a':>6s}" if acc is None else f"  {acc:6.0%}"
        # Verdict: did any CoT condition hurt vs direct?
        worse = any(
            accs[i] is not None and base is not None and accs[i] < base
            for i in range(1, len(accs))
        )
        better = any(
            accs[i] is not None and base is not None and accs[i] > base
            for i in range(1, len(accs))
        )
        verdict = "↓ CoT hurts" if worse else ("↑ CoT helps" if better else "= no change")
        print(row_str + f"  {verdict}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--providers", nargs="+", default=["mock"],
        choices=["openai", "anthropic", "gemini", "mock"],
        help="which providers to run",
    )
    p.add_argument("--openai-model", default=None)
    p.add_argument("--anthropic-model", default=None)
    p.add_argument("--gemini-model", default=None)
    p.add_argument("--repeats", type=int, default=1,
                   help="runs per task/condition (use >=3 to average noise)")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--outdir", default="results")
    p.add_argument(
        "--concurrency", type=int, default=5,
        help="max simultaneous requests per provider (default: 5)",
    )
    p.add_argument(
        "--max-retries", type=int, default=4,
        help="retries on rate-limit errors before giving up (default: 4)",
    )
    p.add_argument(
        "--retry-base", type=float, default=2.0,
        help="base seconds for exponential backoff (default: 2.0)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    models = {
        "openai": args.openai_model,
        "anthropic": args.anthropic_model,
        "gemini": args.gemini_model,
    }
    run(
        args.providers, models, args.repeats, args.temperature, args.outdir,
        concurrency=args.concurrency,
        max_retries=args.max_retries,
        retry_base=args.retry_base,
    )


if __name__ == "__main__":
    main()
