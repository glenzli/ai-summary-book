#!/usr/bin/env python3
"""Validate structure and local closure of the textbook."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parent
CHAPTERS = [ROOT / "00_preface_and_scope.md"] + [
    ROOT / f"{number:02d}_{slug}.md"
    for number, slug in [
        (1, "relations_of_sameness"),
        (2, "state_rng_and_execution"),
        (3, "floating_point_and_error"),
        (4, "parallel_and_distributed"),
        (5, "training_reproducibility"),
        (6, "inference_reproducibility"),
        (7, "data_environment_and_artifacts"),
        (8, "scientific_reproduction"),
        (9, "statistical_replication"),
        (10, "failure_modes"),
        (11, "reproducibility_contract"),
    ]
]
SUPPORT = [
    ROOT / name
    for name in (
        "README.md",
        "SKILL.md",
        "GLOSSARY.md",
        "SOURCES.md",
        "CLAIM_LEDGER.md",
        "SOLUTIONS.md",
        "CLOSURE_AUDIT.md",
    )
]
REQUIRED = CHAPTERS + SUPPORT
EXERCISE = re.compile(r"\*\*练习\s+(\d+\.\d+)\.\*\*")
STATEMENT = re.compile(r"^\*\*(?:定理|命题|引理|推论)\s+(\d+\.\d+)", re.MULTILINE)
CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
OLD_TEMPLATE_HEADING = re.compile(
    r"^##\s+(?:(?:\d+(?:\.\d+)*)\s+)?(?:本章目标|依赖|主线|本章小结)\s*$",
    re.MULTILINE,
)
MARKDOWN_LINK = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
BEGIN_ENV = re.compile(r"\\begin\{([^{}]+)\}")
END_ENV = re.compile(r"\\end\{([^{}]+)\}")
FUTURE_CLOSURE_PLACEHOLDER = re.compile(
    r"最终结果记录将在|冻结后(?:记录|验证)|待最终验证|待收口|待冻结"
)


def iter_math_segments(text: str) -> list[tuple[int, str]]:
    """Return (starting line, content) for inline and display-dollar math."""
    segments: list[tuple[int, str]] = []
    lines = text.splitlines()
    in_fence = False
    in_display = False
    display_start = 0
    display_parts: list[str] = []

    for line_number, line in enumerate(lines, 1):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue

        cursor = 0
        while cursor < len(line):
            if in_display:
                end = line.find("$$", cursor)
                if end < 0:
                    display_parts.append(line[cursor:])
                    break
                display_parts.append(line[cursor:end])
                segments.append((display_start, "\n".join(display_parts)))
                display_parts = []
                in_display = False
                cursor = end + 2
                continue

            start_display = line.find("$$", cursor)
            start_inline = -1
            scan = cursor
            while scan < len(line):
                if line[scan] == "$" and (scan == 0 or line[scan - 1] != "\\"):
                    if scan + 1 >= len(line) or line[scan + 1] != "$":
                        start_inline = scan
                    break
                scan += 1

            if start_display >= 0 and (start_inline < 0 or start_display < start_inline):
                in_display = True
                display_start = line_number
                cursor = start_display + 2
                continue
            if start_inline < 0:
                break

            end = start_inline + 1
            while end < len(line):
                if line[end] == "$" and line[end - 1] != "\\":
                    break
                end += 1
            if end >= len(line):
                break
            segments.append((line_number, line[start_inline + 1 : end]))
            cursor = end + 1

    return segments


def check_math_structure(path: Path, text: str, errors: list[str]) -> None:
    for line_number, segment in iter_math_segments(text):
        depth = 0
        for index, char in enumerate(segment):
            escaped = index > 0 and segment[index - 1] == "\\"
            if char == "{" and not escaped:
                depth += 1
            elif char == "}" and not escaped:
                depth -= 1
                if depth < 0:
                    errors.append(f"{path.name}:{line_number}: unmatched }} in math")
                    break
        if depth > 0:
            errors.append(f"{path.name}:{line_number}: unmatched {{ in math")

        env_stack: list[str] = []
        tokens = sorted(
            [(m.start(), "begin", m.group(1)) for m in BEGIN_ENV.finditer(segment)]
            + [(m.start(), "end", m.group(1)) for m in END_ENV.finditer(segment)]
        )
        for _, kind, environment in tokens:
            if kind == "begin":
                env_stack.append(environment)
            elif not env_stack or env_stack.pop() != environment:
                errors.append(
                    f"{path.name}:{line_number}: mismatched LaTeX environment {environment}"
                )
                break
        if env_stack:
            errors.append(
                f"{path.name}:{line_number}: unclosed LaTeX environment {env_stack[-1]}"
            )


def check_local_links(path: Path, text: str, errors: list[str]) -> None:
    for target in MARKDOWN_LINK.findall(text):
        target = target.strip()
        if target.startswith("<") and target.endswith(">"):
            target = target[1:-1]
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        file_part = unquote(target.split("#", 1)[0])
        if not file_part:
            continue
        resolved = (path.parent / file_part).resolve()
        if not resolved.exists():
            errors.append(f"{path.name}: broken local link: {target}")


def check_balanced_markup(path: Path, text: str, errors: list[str]) -> None:
    if text.count("$$") % 2:
        errors.append(f"{path.name}: odd number of $$ delimiters")
    if sum(line.lstrip().startswith("```") for line in text.splitlines()) % 2:
        errors.append(f"{path.name}: odd number of fenced-code delimiters")
    match = CONTROL.search(text)
    if match:
        errors.append(f"{path.name}: forbidden control character U+{ord(match.group()):04X}")

    in_fence = False
    for line_number, line in enumerate(text.splitlines(), 1):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if line.endswith((" ", "\t")):
            errors.append(f"{path.name}:{line_number}: trailing whitespace")
        if not in_fence:
            unescaped_dollars = len(re.findall(r"(?<!\\)\$", line))
            if unescaped_dollars % 2:
                errors.append(
                    f"{path.name}:{line_number}: odd number of unescaped $ delimiters"
                )
    check_math_structure(path, text, errors)
    check_local_links(path, text, errors)


def main() -> int:
    errors: list[str] = []
    for path in REQUIRED:
        if not path.is_file():
            errors.append(f"missing required file: {path.name}")
    if errors:
        print("\n".join(errors))
        return 1

    solutions_text = (ROOT / "SOLUTIONS.md").read_text(encoding="utf-8")
    ledger_text = (ROOT / "CLAIM_LEDGER.md").read_text(encoding="utf-8")
    readme_text = (ROOT / "README.md").read_text(encoding="utf-8")
    closure_text = (ROOT / "CLOSURE_AUDIT.md").read_text(encoding="utf-8")
    seen_exercises: set[str] = set()
    seen_statements: set[str] = set()

    for path in REQUIRED:
        check_balanced_markup(path, path.read_text(encoding="utf-8"), errors)

    for chapter_index, chapter in enumerate(CHAPTERS):
        text = chapter.read_text(encoding="utf-8")
        match = OLD_TEMPLATE_HEADING.search(text)
        if match:
            errors.append(
                f"{chapter.name}: forbidden outline-style heading: {match.group().strip()}"
            )
        if "## 练习" not in text:
            errors.append(f"{chapter.name}: missing ## 练习")

        lines = text.splitlines()
        first_section = next(
            (index for index, line in enumerate(lines[1:], 1) if line.startswith("## ")),
            len(lines),
        )
        introduction = "\n".join(lines[1:first_section]).strip()
        introduction_text = re.sub(r"[`*_#>\[\]()$\\]", "", introduction)
        if len(introduction_text) < 100:
            errors.append(
                f"{chapter.name}: chapter introduction before first section is too short"
            )

        expected_prefix = f"{chapter_index}."
        chapter_exercises = EXERCISE.findall(text)
        if not chapter_exercises:
            errors.append(f"{chapter.name}: no exercises")
        for number in chapter_exercises:
            if not number.startswith(expected_prefix):
                errors.append(f"{chapter.name}: exercise has wrong prefix: {number}")
            if number in seen_exercises:
                errors.append(f"duplicate exercise: {number}")
            seen_exercises.add(number)
            count = len(re.findall(rf"\*\*练习\s+{re.escape(number)}\.\*\*", solutions_text))
            if count != 1:
                errors.append(f"solution count for {number}: expected 1, got {count}")

        for number in STATEMENT.findall(text):
            if number in seen_statements:
                errors.append(f"duplicate theorem-style statement: {number}")
            seen_statements.add(number)
            locator = re.compile(
                rf"(?:定理|命题|引理|推论)\s+{re.escape(number)}(?:\D|$)"
            )
            if not locator.search(ledger_text):
                errors.append(f"claim ledger missing statement: {number}")

    solution_numbers = EXERCISE.findall(solutions_text)
    if len(solution_numbers) != len(set(solution_numbers)):
        errors.append("SOLUTIONS.md contains duplicate exercise answers")
    extras = sorted(set(solution_numbers) - seen_exercises)
    if extras:
        errors.append("solutions without exercises: " + ", ".join(extras))

    if readme_text.count("**作者：Dr. Stochastic Parrot**") != 1:
        errors.append("README.md: exact author attribution must appear exactly once")
    if FUTURE_CLOSURE_PLACEHOLDER.search(closure_text):
        errors.append("CLOSURE_AUDIT.md: contains a future-tense closure placeholder")
    expected_count = f"正文共有 {len(seen_statements)} 个定理、命题、引理或推论型陈述"
    if expected_count not in closure_text:
        errors.append(
            "CLOSURE_AUDIT.md: theorem-style statement count is stale; "
            f"expected text containing: {expected_count}"
        )

    if errors:
        print("\n".join(errors))
        return 1
    print(
        "illusion-of-reproducibility validation: "
        f"chapters={len(CHAPTERS)} "
        f"statements={len(seen_statements)} "
        f"exercises={len(seen_exercises)} errors=0"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
