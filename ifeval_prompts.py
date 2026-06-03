"""IFEval-style instruction-following prompts with deterministic verifiers.

Each task is a base instruction plus one or more *programmatically verifiable*
constraints (the same idea as Google's IFEval). No LLM judge is needed: the
verifier returns True/False from the raw model output.

This lets us measure the paper's core claim ("When Thinking Fails", 2505.11423):
does CoT prompting reduce instruction-following accuracy?
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable, List


@dataclass
class Task:
    id: str
    prompt: str
    # Each verifier checks one constraint. A task passes only if ALL pass.
    verifiers: List[Callable[[str], bool]] = field(default_factory=list)
    # Human-readable description of the constraints (for reporting).
    constraints: List[str] = field(default_factory=list)

    def score(self, output: str) -> dict:
        per = []
        for desc, fn in zip(self.constraints, self.verifiers):
            try:
                ok = bool(fn(output))
            except Exception:
                ok = False
            per.append({"constraint": desc, "passed": ok})
        all_pass = all(p["passed"] for p in per) if per else False
        return {"all_passed": all_pass, "per_constraint": per}


# --------------------------------------------------------------------------- #
# Verifier helpers
# --------------------------------------------------------------------------- #
def _words(text: str) -> List[str]:
    return re.findall(r"\b[\w']+\b", text)


def _strip_md(text: str) -> str:
    return text.strip()


def word_count_at_least(n: int) -> Callable[[str], bool]:
    return lambda t: len(_words(t)) >= n


def word_count_at_most(n: int) -> Callable[[str], bool]:
    return lambda t: len(_words(t)) <= n


def sentence_count_exactly(n: int) -> Callable[[str], bool]:
    def f(t: str) -> bool:
        sentences = [s for s in re.split(r"[.!?]+", t) if s.strip()]
        return len(sentences) == n
    return f


def contains_all_keywords(keywords: List[str]) -> Callable[[str], bool]:
    return lambda t: all(k.lower() in t.lower() for k in keywords)


def forbids_words(words: List[str]) -> Callable[[str], bool]:
    def f(t: str) -> bool:
        low = t.lower()
        return not any(re.search(r"\b" + re.escape(w.lower()) + r"\b", low) for w in words)
    return f


def is_valid_json() -> Callable[[str], bool]:
    def f(t: str) -> bool:
        s = t.strip()
        # Allow fenced code blocks.
        m = re.search(r"```(?:json)?\s*(.*?)```", s, re.DOTALL)
        if m:
            s = m.group(1).strip()
        try:
            json.loads(s)
            return True
        except Exception:
            return False
    return f


def all_lowercase() -> Callable[[str], bool]:
    return lambda t: t == t.lower() and any(c.isalpha() for c in t)


def all_uppercase() -> Callable[[str], bool]:
    return lambda t: t == t.upper() and any(c.isalpha() for c in t)


def bullet_count_exactly(n: int) -> Callable[[str], bool]:
    def f(t: str) -> bool:
        bullets = re.findall(r"(?m)^\s*[\*\-]\s+\S", t)
        return len(bullets) == n
    return f


def paragraph_count_exactly(n: int) -> Callable[[str], bool]:
    # Paragraphs separated by the literal divider '***' (IFEval style).
    return lambda t: len([p for p in t.split("***") if p.strip()]) == n


def ends_with_phrase(phrase: str) -> Callable[[str], bool]:
    return lambda t: t.strip().rstrip('"').strip().endswith(phrase)


def starts_with_word(word: str) -> Callable[[str], bool]:
    def f(t: str) -> bool:
        w = _words(t.strip())
        return bool(w) and w[0].lower() == word.lower()
    return f


def has_title_in_double_angle() -> Callable[[str], bool]:
    return lambda t: bool(re.search(r"<<[^<>]+>>", t))


def contains_postscript() -> Callable[[str], bool]:
    return lambda t: bool(re.search(r"(?im)^\s*p\.?\s*s\.?", t))


def no_commas() -> Callable[[str], bool]:
    return lambda t: "," not in t


def placeholder_count_at_least(n: int) -> Callable[[str], bool]:
    return lambda t: len(re.findall(r"\[[^\[\]]+\]", t)) >= n


def highlight_count_at_least(n: int) -> Callable[[str], bool]:
    # Markdown *highlighted* sections.
    return lambda t: len(re.findall(r"\*[^*\n]+\*", t)) >= n


def letter_frequency_at_least(letter: str, n: int) -> Callable[[str], bool]:
    return lambda t: t.lower().count(letter.lower()) >= n


def numbered_list_exactly(n: int) -> Callable[[str], bool]:
    def f(t: str) -> bool:
        items = re.findall(r"(?m)^\s*\d+[\.\)]\s+\S", t)
        return len(items) == n
    return f


# --------------------------------------------------------------------------- #
# Task set
# --------------------------------------------------------------------------- #
TASKS: List[Task] = [
    Task(
        id="json_cities",
        prompt=(
            "List three large cities and their countries. "
            "Respond ONLY with valid JSON (an array of objects with keys "
            '"city" and "country"). Do not include any text outside the JSON.'
        ),
        verifiers=[is_valid_json()],
        constraints=["output is valid JSON"],
    ),
    Task(
        id="no_commas_haiku",
        prompt=(
            "Write a short description of the ocean at sunset. "
            "Do not use any commas anywhere in your response."
        ),
        verifiers=[no_commas(), word_count_at_least(8)],
        constraints=["no commas", "at least 8 words"],
    ),
    Task(
        id="exactly_three_bullets",
        prompt=(
            "Give me tips for staying productive. "
            "Answer using exactly three bullet points, each starting with '* '."
        ),
        verifiers=[bullet_count_exactly(3)],
        constraints=["exactly 3 bullet points"],
    ),
    Task(
        id="all_lowercase",
        prompt=(
            "Explain what photosynthesis is in one or two sentences. "
            "Your entire response must be in all lowercase letters; "
            "no capital letters are allowed."
        ),
        verifiers=[all_lowercase()],
        constraints=["entire response lowercase"],
    ),
    Task(
        id="title_and_postscript",
        prompt=(
            "Write a brief note inviting a friend to dinner. "
            "Include a title wrapped in double angle brackets, like <<Title>>, "
            "and add a postscript starting with 'P.S.' at the end."
        ),
        verifiers=[has_title_in_double_angle(), contains_postscript()],
        constraints=["title in <<...>>", "contains a P.S. postscript"],
    ),
    Task(
        id="word_cap_30",
        prompt=(
            "Summarize the plot of Romeo and Juliet in no more than 30 words."
        ),
        verifiers=[word_count_at_most(30), word_count_at_least(5)],
        constraints=["at most 30 words", "at least 5 words"],
    ),
    Task(
        id="keywords_required",
        prompt=(
            "Write two sentences about healthy breakfasts. "
            "You must include the words 'protein', 'fiber', and 'energy'."
        ),
        verifiers=[contains_all_keywords(["protein", "fiber", "energy"])],
        constraints=["contains 'protein', 'fiber', 'energy'"],
    ),
    Task(
        id="forbidden_words",
        prompt=(
            "Describe a relaxing weekend. "
            "Do not use the words 'relax', 'rest', or 'calm' anywhere."
        ),
        verifiers=[forbids_words(["relax", "rest", "calm"]), word_count_at_least(15)],
        constraints=["avoids 'relax'/'rest'/'calm'", "at least 15 words"],
    ),
    Task(
        id="three_paragraphs_divider",
        prompt=(
            "Write about the benefits of exercise. "
            "Your response must contain exactly three paragraphs, and the "
            "paragraphs must be separated by a line containing only '***'."
        ),
        verifiers=[paragraph_count_exactly(3)],
        constraints=["exactly 3 paragraphs separated by ***"],
    ),
    Task(
        id="end_with_phrase",
        prompt=(
            "Give a one-sentence motivational quote. "
            "Finish your entire response with the exact phrase: "
            "Keep going."
        ),
        verifiers=[ends_with_phrase("Keep going.")],
        constraints=["ends with 'Keep going.'"],
    ),
    Task(
        id="numbered_five_steps",
        prompt=(
            "Explain how to make a paper airplane. "
            "Use a numbered list with exactly five steps (1. 2. 3. 4. 5.)."
        ),
        verifiers=[numbered_list_exactly(5)],
        constraints=["exactly 5 numbered steps"],
    ),
    Task(
        id="placeholders_email",
        prompt=(
            "Write a short template email confirming a meeting. "
            "Include at least three placeholders written in square brackets, "
            "such as [name] or [date]."
        ),
        verifiers=[placeholder_count_at_least(3)],
        constraints=["at least 3 [bracketed] placeholders"],
    ),
    Task(
        id="start_with_word",
        prompt=(
            "Answer the question: why is sleep important? "
            "Your response must begin with the word 'Sleep'."
        ),
        verifiers=[starts_with_word("Sleep")],
        constraints=["starts with the word 'Sleep'"],
    ),
    Task(
        id="two_sentences",
        prompt=(
            "Describe your favorite season. "
            "Write exactly two sentences."
        ),
        verifiers=[sentence_count_exactly(2)],
        constraints=["exactly 2 sentences"],
    ),
    Task(
        id="highlights",
        prompt=(
            "Recommend a book and explain why. "
            "Highlight at least two key phrases using markdown asterisks, "
            "like *this*."
        ),
        verifiers=[highlight_count_at_least(2)],
        constraints=["at least 2 *highlighted* phrases"],
    ),
    Task(
        id="uppercase_slogan",
        prompt=(
            "Create a marketing slogan for a coffee shop. "
            "Your entire response must be in ALL CAPITAL LETTERS."
        ),
        verifiers=[all_uppercase()],
        constraints=["entire response uppercase"],
    ),
]


def get_tasks() -> List[Task]:
    return TASKS
