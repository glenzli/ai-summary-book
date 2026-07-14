#!/usr/bin/env python3
"""Validate the Proof, Explanation, and Rhetoric textbook."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parent
CHAPTERS = [ROOT / "00_preface_and_scope.md"] + [
    ROOT / f"{n:02d}_{slug}.md"
    for n, slug in [
        (1, "claims_arguments_and_obligations"), (2, "validity_and_countermodels"),
        (3, "definitions_types_and_quantifiers"), (4, "proof_methods"),
        (5, "formal_and_informal_proof"), (6, "proof_and_mathematical_explanation"),
        (7, "scientific_explanation_and_evidence"), (8, "diagrams_analogies_and_rhetoric"),
        (9, "citation_and_sources"), (10, "ai_reasoning_and_cot"),
        (11, "verification_workshop"),
    ]
]
REQUIRED = CHAPTERS + [ROOT / n for n in (
    "README.md", "SKILL.md", "GLOSSARY.md", "SOURCES.md",
    "CLAIM_LEDGER.md", "SOLUTIONS.md", "CLOSURE_AUDIT.md",
)]
EXERCISE = re.compile(r"\*\*练习\s+(\d+\.\d+)\.")
AUTHOR_LINE = "**作者：Dr. Stochastic Parrot**"
LEGACY_HEADING = re.compile(
    r"^\s*(?:\*\*)?##\s+(?:\d+(?:\.\d+)*\.?\s+)?"
    r"(本章目标|依赖|主线|本章小结)(?:\s*[:：].*)?(?:\*\*)?\s*$",
    re.MULTILINE,
)
FORMAL_RESULT = re.compile(r"\*\*(定理|命题|推论|外部输入)\s+(\d+\.\d+)")
MARKDOWN_LINK = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
MALFORMED_TEX = (
    (
        "bare TeX spacing command",
        re.compile(r"(?<!\\)\b(?:quad|qquad)\b"),
    ),
)
CASE_CUE = re.compile(
    r"案例|例子|原句|开头|章首|贯穿|下面|这段|这份|两份证明|一道题|研究|摘要|报告|程序|需求"
)
MIN_INTRO_CHARS = 120


def chapter_intro(text: str) -> str:
    """Return prose between the chapter title and its first level-two heading."""
    lines = text.splitlines()
    if not lines:
        return ""
    first_h2 = next(
        (index for index, line in enumerate(lines[1:], start=1) if line.startswith("## ")),
        len(lines),
    )
    return "\n".join(lines[1:first_h2]).strip()


def local_link_target(source: Path, raw_target: str) -> Path | None:
    """Resolve a local Markdown target; return None for external or in-page links."""
    target = raw_target.strip()
    if target.startswith("<") and target.endswith(">"):
        target = target[1:-1]
    target = target.split(maxsplit=1)[0]
    if target.startswith(("http://", "https://", "mailto:", "#")):
        return None
    target = unquote(target.split("#", maxsplit=1)[0])
    if not target:
        return None
    return (source.parent / target).resolve()


def main() -> int:
    errors: list[str] = []
    for path in REQUIRED:
        if not path.is_file():
            errors.append(f"missing required file: {path.name}")
    if errors:
        print("\n".join(errors))
        return 1
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    if readme.splitlines().count(AUTHOR_LINE) != 1:
        errors.append(
            f"README.md: expected exactly one author line equal to: {AUTHOR_LINE}"
        )

    solutions = (ROOT / "SOLUTIONS.md").read_text(encoding="utf-8")
    ledger = (ROOT / "CLAIM_LEDGER.md").read_text(encoding="utf-8")
    seen: set[str] = set()
    formal_seen: dict[str, str] = {}
    for chapter in CHAPTERS:
        text = chapter.read_text(encoding="utf-8")
        if not text.startswith("# "):
            errors.append(f"{chapter.name}: first line must be a chapter title")
        for match in LEGACY_HEADING.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            errors.append(
                f"{chapter.name}:{line}: legacy template heading is forbidden: "
                f"{match.group(1)}"
            )
        intro = chapter_intro(text)
        prose_chars = len(re.sub(r"[`*_>#\s$\\]", "", intro))
        if prose_chars < MIN_INTRO_CHARS:
            errors.append(
                f"{chapter.name}: chapter opening is too short to establish a "
                f"readable problem ({prose_chars} < {MIN_INTRO_CHARS} characters)"
            )
        if not CASE_CUE.search(intro):
            errors.append(
                f"{chapter.name}: chapter opening lacks a concrete case or "
                "reviewable argument cue"
            )
        if text.count("## 练习") != 1:
            errors.append(f"{chapter.name}: expected exactly one exercise section")
        chapter_results: set[str] = set()
        for kind, number in FORMAL_RESULT.findall(text):
            if number in chapter_results:
                errors.append(f"{chapter.name}: duplicate formal result: {number}")
            chapter_results.add(number)
            if number in formal_seen:
                errors.append(
                    f"duplicate formal result across chapters: {number} "
                    f"({formal_seen[number]}, {chapter.name})"
                )
            formal_seen[number] = chapter.name
            if number not in ledger:
                errors.append(
                    f"{chapter.name}: {kind} {number} is missing from CLAIM_LEDGER.md"
                )
        chapter_exercises = EXERCISE.findall(text)
        if not chapter_exercises:
            errors.append(f"{chapter.name}: no numbered exercises")
        for number in chapter_exercises:
            if number in seen:
                errors.append(f"duplicate exercise: {number}")
            seen.add(number)
            if f"**练习 {number}.**" not in solutions:
                errors.append(f"missing solution: {number}")
    extras = sorted(set(EXERCISE.findall(solutions)) - seen)
    if extras:
        errors.append("solutions without exercises: " + ", ".join(extras))
    for source in sorted(ROOT.glob("*.md")):
        text = source.read_text(encoding="utf-8")
        for label, pattern in MALFORMED_TEX:
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                errors.append(
                    f"{source.name}:{line}: {label}: {match.group(0)!r}; "
                    f"expected '\\{match.group(0)}'"
                )
        for raw_target in MARKDOWN_LINK.findall(text):
            target = local_link_target(source, raw_target)
            if target is not None and not target.exists():
                errors.append(
                    f"{source.name}: broken local link: {raw_target}"
                )
    if errors:
        print("\n".join(errors))
        return 1
    print(
        "proof-explanation-and-rhetoric validation: "
        f"chapters={len(CHAPTERS)} exercises={len(seen)} "
        f"formal_results={len(formal_seen)} errors=0"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
