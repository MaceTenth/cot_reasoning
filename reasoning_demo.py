#!/usr/bin/env python3
"""
reasoning_demo.py — Compare internal reasoning across effort levels.

Shows the model's reasoning process, final answer, wall-clock time,
token counts, and estimated cost for each effort level.

API differences:
  • OpenAI  (o4-mini / o3): reasoning_effort=low|medium|high
            → Returns REASONING TOKEN COUNT only (text is never exposed)
  • Anthropic (claude-sonnet-4-6): extended thinking with budget_tokens
            → Returns the FULL THINKING TEXT in response blocks
            Note: Opus 4.8 uses adaptive thinking (effort=) but does NOT
            expose reasoning text — use Sonnet 4.6 to see actual thinking.

Usage:
  python reasoning_demo.py
  python reasoning_demo.py --efforts low high
  python reasoning_demo.py --providers openai
  python reasoning_demo.py --task "Your custom problem here"
  python reasoning_demo.py --openai-model o3 --anthropic-model claude-sonnet-4-6
  python reasoning_demo.py --full-thinking   # print complete Anthropic thinking text
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# ── Default models ─────────────────────────────────────────────────────────────
# Override with env vars or --openai-model / --anthropic-model flags
OPENAI_MODEL    = os.getenv("OPENAI_REASONING_MODEL",    "o4-mini")
# Sonnet 4.6 is the latest Anthropic model that returns the *full thinking text* via
# extended thinking (budget_tokens).  Opus 4.8 uses adaptive thinking (effort=) but
# does NOT expose its reasoning text through the API.
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_REASONING_MODEL", "claude-sonnet-4-6")

# ── Effort level → Anthropic budget_tokens ─────────────────────────────────────
# OpenAI uses the string directly; Anthropic uses a token budget.
ANTHROPIC_BUDGET: dict[str, int] = {
    "low":    1_024,
    "medium": 5_000,
    "high":  16_000,
}

# ── Pricing table (USD / 1M tokens) ───────────────────────────────────────────
PRICES: dict[str, tuple[float, float]] = {
    # OpenAI reasoning models
    "o4-mini":                    ( 1.10,   4.40),
    "o3-mini":                    ( 1.10,   4.40),
    "o3":                         (10.00,  40.00),
    "o1":                         (15.00,  60.00),
    # Anthropic — output includes thinking tokens
    # Sonnet 4.6 = latest model that exposes full thinking text via extended thinking API
    "claude-sonnet-4-6":          ( 3.00,  15.00),
    "claude-sonnet-4-5":          ( 3.00,  15.00),
    "claude-haiku-4-5":           ( 1.00,   5.00),
    "claude-3-7-sonnet-20250219": ( 3.00,  15.00),
    "claude-3-7-sonnet":          ( 3.00,  15.00),
    "claude-opus-4-8":            ( 5.00,  25.00),
}

# ── Demo task ──────────────────────────────────────────────────────────────────
DEFAULT_TASK = (
    "A farmer needs to cross a river with a fox, a chicken, and a bag of grain. "
    "His boat can carry only himself and one other item. "
    "If left alone together: the fox eats the chicken, and the chicken eats the grain. "
    "How does the farmer get everything across safely? "
    "List every single crossing step, and explain why each step is necessary."
)


# ── Result dataclass ───────────────────────────────────────────────────────────
@dataclass
class ReasoningResult:
    provider: str
    model: str
    effort: str
    thinking_text: str | None   # Anthropic: actual reasoning text; OpenAI: None
    thinking_tokens: int        # reasoning tokens used (count only for OpenAI)
    output_text: str
    prompt_tokens: int
    completion_tokens: int      # includes thinking tokens on Anthropic
    elapsed_sec: float
    error: str | None = None

    def cost(self) -> float | None:
        key = next((k for k in PRICES if self.model.startswith(k)), None)
        if key is None:
            return None
        inp, out = PRICES[key]
        return (self.prompt_tokens * inp + self.completion_tokens * out) / 1_000_000

    def thinking_word_count(self) -> int:
        if not self.thinking_text:
            return 0
        return len(self.thinking_text.split())

    def output_word_count(self) -> int:
        return len(self.output_text.split())


# ── OpenAI ─────────────────────────────────────────────────────────────────────
def run_openai(task: str, effort: str, model: str) -> ReasoningResult:
    from openai import OpenAI
    client = OpenAI()
    t0 = time.perf_counter()
    try:
        resp = client.chat.completions.create(
            model=model,
            reasoning_effort=effort,
            messages=[{"role": "user", "content": task}],
        )
        elapsed = time.perf_counter() - t0
        usage = resp.usage
        details = getattr(usage, "completion_tokens_details", None)
        rtoks = getattr(details, "reasoning_tokens", 0) or 0
        return ReasoningResult(
            provider="openai",
            model=model,
            effort=effort,
            thinking_text=None,          # API never exposes the reasoning text
            thinking_tokens=rtoks,
            output_text=resp.choices[0].message.content or "",
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            elapsed_sec=elapsed,
        )
    except Exception as e:  # noqa: BLE001
        return ReasoningResult(
            provider="openai", model=model, effort=effort,
            thinking_text=None, thinking_tokens=0, output_text="",
            prompt_tokens=0, completion_tokens=0,
            elapsed_sec=time.perf_counter() - t0, error=str(e),
        )


# ── Anthropic ──────────────────────────────────────────────────────────────────
def run_anthropic(task: str, effort: str, model: str) -> ReasoningResult:
    from anthropic import Anthropic
    client = Anthropic()
    budget = ANTHROPIC_BUDGET[effort]
    # max_tokens must exceed budget_tokens; leave room for the output text
    max_tokens = min(budget + 4096, 32_000)
    t0 = time.perf_counter()
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            thinking={"type": "enabled", "budget_tokens": budget},
            messages=[{"role": "user", "content": task}],
        )
        elapsed = time.perf_counter() - t0

        thinking_parts: list[str] = []
        text_parts: list[str] = []
        for block in resp.content:
            btype = getattr(block, "type", "")
            if btype == "thinking":
                thinking_parts.append(getattr(block, "thinking", "") or "")
            elif btype == "text":
                text_parts.append(getattr(block, "text", "") or "")

        usage = resp.usage
        return ReasoningResult(
            provider="anthropic",
            model=model,
            effort=effort,
            thinking_text="\n\n".join(thinking_parts) if thinking_parts else None,
            thinking_tokens=budget,          # Anthropic = budget allocated (used ≤ budget)
            output_text="\n".join(text_parts),
            prompt_tokens=getattr(usage, "input_tokens", 0),
            completion_tokens=getattr(usage, "output_tokens", 0),  # includes thinking
            elapsed_sec=elapsed,
        )
    except Exception as e:  # noqa: BLE001
        return ReasoningResult(
            provider="anthropic", model=model, effort=effort,
            thinking_text=None, thinking_tokens=0, output_text="",
            prompt_tokens=0, completion_tokens=0,
            elapsed_sec=time.perf_counter() - t0, error=str(e),
        )


# ── Display helpers ────────────────────────────────────────────────────────────
W = 72  # display width


def banner(title: str, char: str = "═") -> str:
    return f"\n{char*W}\n  {title}\n{char*W}"


def section(title: str) -> str:
    return f"\n  {'─'*66}\n  {title}\n  {'─'*66}"


def indent(text: str, spaces: int = 6, max_chars: int = 2000) -> str:
    prefix = " " * spaces
    truncated = text[:max_chars]
    overflow = len(text) - max_chars
    lines = "\n".join(prefix + ln for ln in truncated.splitlines())
    if overflow > 0:
        lines += f"\n{prefix}… [{overflow:,} more characters — run with --full-thinking to see all]"
    return lines


def print_result(r: ReasoningResult, show_full_thinking: bool = False) -> None:
    tag = f"{r.provider}  model={r.model}  effort={r.effort}"
    print(banner(tag))

    if r.error:
        print(f"\n  ❌  ERROR: {r.error}\n")
        return

    cost_str = f"${r.cost():.4f}" if r.cost() is not None else "n/a"
    print(f"\n  ⏱  Time : {r.elapsed_sec:.1f}s")
    print(f"  💰  Cost : {cost_str}")
    print(f"  🔢  Tokens → prompt: {r.prompt_tokens:,}   completion: {r.completion_tokens:,}   reasoning: {r.thinking_tokens:,}")

    # ── Reasoning section ─────────────────────────────────────────────────────
    print(section("🧠  REASONING"))
    if r.thinking_text:
        wc = r.thinking_word_count()
        print(f"  Anthropic extended thinking  ({wc:,} words, full text exposed by API)\n")
        limit = None if show_full_thinking else 2000
        txt = r.thinking_text if limit is None else r.thinking_text[:limit]
        overflow = 0 if limit is None else max(0, len(r.thinking_text) - limit)
        print(indent(txt, max_chars=999_999))
        if overflow:
            print(f"\n      … [{overflow:,} more chars — pass --full-thinking to see all]")
    else:
        print(f"  OpenAI  {r.thinking_tokens:,} reasoning tokens consumed internally.")
        print("  (The reasoning text itself is NOT returned by the OpenAI API.)")
        print("  You only see the final answer below.")

    # ── Final answer ──────────────────────────────────────────────────────────
    print(section("📝  FINAL ANSWER"))
    print(indent(r.output_text, max_chars=999_999))
    print()


def print_summary_table(results: list[ReasoningResult]) -> None:
    print(banner("SUMMARY — effort level comparison", "═"))
    hdr = (
        f"  {'Provider':<12} {'Model':<28} {'Effort':<8}"
        f" {'Time':>7} {'ThinkTok':>9} {'OutTok':>7} {'Words':>6} {'Cost':>8}"
    )
    print(hdr)
    print(f"  {'─'*68}")
    for r in results:
        if r.error:
            print(f"  {r.provider:<12} {r.model:<28} {r.effort:<8}  {'ERROR':>7}")
            continue
        cost_str = f"${r.cost():.4f}" if r.cost() is not None else "  n/a"
        out_wc = r.output_word_count()
        print(
            f"  {r.provider:<12} {r.model:<28} {r.effort:<8}"
            f" {r.elapsed_sec:>6.1f}s {r.thinking_tokens:>9,}"
            f" {r.completion_tokens:>7,} {out_wc:>6} {cost_str:>8}"
        )
    print(f"  {'─'*68}")

    # Highlight key insights
    good = [r for r in results if not r.error]
    if len(good) >= 2:
        print("\n  KEY OBSERVATIONS")
        print("  ─────────────────")
        for provider in {"openai", "anthropic"} & {r.provider for r in good}:
            subset = [r for r in good if r.provider == provider]
            if len(subset) > 1:
                low_r  = next((r for r in subset if r.effort == "low"),  None)
                high_r = next((r for r in subset if r.effort == "high"), None)
                if low_r and high_r:
                    t_ratio = high_r.elapsed_sec / low_r.elapsed_sec if low_r.elapsed_sec else 0
                    print(f"  • {provider}: high effort is {t_ratio:.1f}× slower than low effort")
                    if low_r.cost() and high_r.cost():
                        c_ratio = high_r.cost() / low_r.cost()
                        print(f"  • {provider}: high effort is {c_ratio:.1f}× more expensive than low effort")
    print()


# ── Save results ───────────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).parent / "reasoning_results"


def save_results(results: list[ReasoningResult], task: str) -> Path:
    """Save run to reasoning_results/<timestamp>/  as JSON + human-readable txt."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RESULTS_DIR / ts
    run_dir.mkdir(parents=True, exist_ok=True)

    # ── JSON (full data, including complete thinking text) ────────────────────
    payload = {
        "run_timestamp": ts,
        "task": task,
        "results": [
            {
                "provider":          r.provider,
                "model":             r.model,
                "effort":            r.effort,
                "elapsed_sec":       round(r.elapsed_sec, 2),
                "prompt_tokens":     r.prompt_tokens,
                "completion_tokens": r.completion_tokens,
                "thinking_tokens":   r.thinking_tokens,
                "estimated_cost_usd": r.cost(),
                "thinking_text":     r.thinking_text,   # full text, no truncation
                "output_text":       r.output_text,
                "error":             r.error,
            }
            for r in results
        ],
    }
    json_path = run_dir / "results.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    # ── Human-readable text (same layout as terminal, full thinking text) ─────
    import io, sys
    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf

    print(banner("REASONING MODEL DEMO", "═"))
    print(f"\n  Run      : {ts}")
    print(f"  Task     : {task}")
    for r in results:
        print_result(r, show_full_thinking=True)   # always full in saved file
    print_summary_table(results)

    sys.stdout = old_stdout
    txt_path = run_dir / "results.txt"
    txt_path.write_text(buf.getvalue())

    return run_dir


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show reasoning model internals across effort levels"
    )
    parser.add_argument("--task", default=DEFAULT_TASK,
                        help="Problem to send to the models")
    parser.add_argument("--efforts", nargs="+",
                        choices=["low", "medium", "high"],
                        default=["low", "medium", "high"])
    parser.add_argument("--providers", nargs="+",
                        choices=["openai", "anthropic"],
                        default=["openai", "anthropic"])
    parser.add_argument("--openai-model",    default=OPENAI_MODEL,
                        help=f"OpenAI model (default: {OPENAI_MODEL})")
    parser.add_argument("--anthropic-model", default=ANTHROPIC_MODEL,
                        help=f"Anthropic model (default: {ANTHROPIC_MODEL})")
    parser.add_argument("--full-thinking", action="store_true",
                        help="Print complete Anthropic thinking text (can be very long)")
    parser.add_argument("--no-save", action="store_true",
                        help="Skip saving results to reasoning_results/ folder")
    args = parser.parse_args()

    print(banner("REASONING MODEL DEMO", "═"))
    print(f"\n  Task     : {args.task[:100]}{'...' if len(args.task) > 100 else ''}")
    print(f"  Efforts  : {args.efforts}")
    print(f"  Providers: {args.providers}")
    print(f"\n  Note: OpenAI returns reasoning TOKEN COUNT only.")
    print(f"        Anthropic returns the FULL THINKING TEXT.")

    results: list[ReasoningResult] = []

    for provider in args.providers:
        for effort in args.efforts:
            model = args.openai_model if provider == "openai" else args.anthropic_model
            print(f"\n⏳  {provider}/{model}  effort={effort} … ", end="", flush=True)
            if provider == "openai":
                r = run_openai(args.task, effort, model)
            else:
                r = run_anthropic(args.task, effort, model)
            status = f"✓ {r.elapsed_sec:.1f}s" if not r.error else "✗ ERROR"
            print(status)
            results.append(r)

    # Detailed results
    for r in results:
        print_result(r, show_full_thinking=args.full_thinking)

    print_summary_table(results)

    if not args.no_save:
        run_dir = save_results(results, args.task)
        print(f"  💾  Results saved to: {run_dir}/")
        print(f"       • results.json  — full data (machine-readable)")
        print(f"       • results.txt   — human-readable report (full thinking text)\n")


if __name__ == "__main__":
    main()
