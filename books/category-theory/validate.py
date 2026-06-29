#!/usr/bin/env python3
"""Repository-local consistency checks for the category theory textbook."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CHAPTER_FILE = re.compile(r"^(?:[0-9][0-9]|[ABCE])_.*\.md$")
MARKDOWN_LINK = re.compile(r"\[[^\]\n]+\]\(([^)]+)\)")
EXERCISE = re.compile(r"^\*\*练习\s+([0-9]+|[A-Z])\.([0-9]+)\.\*\*", re.M)
ANSWER = re.compile(r"^\*\*答案\s+([0-9]+|[A-Z])\.([0-9]+)\.\*\*", re.M)
COMPREHENSIVE = re.compile(r"^\*\*综合题\s+([0-9]+)\.\*\*", re.M)
COMPREHENSIVE_ANSWER = re.compile(r"^## 综合题\s+([0-9]+)$", re.M)
FORBIDDEN = (
    "证明草图",
    "完整证明需要",
    "审查标记",
    "[TODO",
    "TODO",
    "placeholder",
)


def fail(message: str, failures: list[str]) -> None:
    failures.append(message)


def check_links(md_files: list[Path], failures: list[str]) -> None:
    for path in md_files:
        text = path.read_text(encoding="utf-8")
        for match in MARKDOWN_LINK.finditer(text):
            target = match.group(1)
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            target_path = target.split("#", 1)[0]
            if not target_path:
                continue
            if not (path.parent / target_path).resolve().exists():
                fail(f"missing link in {path.name}: {target}", failures)


def check_structure(chapter_files: list[Path], failures: list[str]) -> None:
    required = ("## 本章目标", "本章小结", "## 练习")
    for path in chapter_files:
        text = path.read_text(encoding="utf-8")
        for marker in required:
            if marker not in text:
                fail(f"missing section in {path.name}: {marker}", failures)


def check_forbidden_markers(md_files: list[Path], failures: list[str]) -> None:
    for path in md_files:
        text = path.read_text(encoding="utf-8")
        for marker in FORBIDDEN:
            if marker in text:
                fail(f"forbidden marker in {path.name}: {marker}", failures)


def check_exercise_answers(chapter_files: list[Path], failures: list[str]) -> tuple[int, int]:
    solutions = (ROOT / "SOLUTIONS.md").read_text(encoding="utf-8")
    answer_keys = {f"{a}.{b}" for a, b in ANSWER.findall(solutions)}
    exercises: list[str] = []
    for path in chapter_files:
        exercises.extend(f"{a}.{b}" for a, b in EXERCISE.findall(path.read_text(encoding="utf-8")))

    for key in exercises:
        if key not in answer_keys:
            fail(f"missing solution: exercise {key}", failures)

    comprehensive = (ROOT / "COMPREHENSIVE_EXERCISES.md").read_text(encoding="utf-8")
    comprehensive_solutions = (ROOT / "COMPREHENSIVE_SOLUTIONS.md").read_text(encoding="utf-8")
    comp_keys = set(COMPREHENSIVE.findall(comprehensive))
    comp_answer_keys = set(COMPREHENSIVE_ANSWER.findall(comprehensive_solutions))
    for key in sorted(comp_keys, key=int):
        if key not in comp_answer_keys:
            fail(f"missing comprehensive solution: problem {key}", failures)

    return len(exercises), len(comp_keys)


def main() -> int:
    md_files = sorted(ROOT.glob("*.md"))
    chapter_files = [path for path in md_files if CHAPTER_FILE.match(path.name)]
    failures: list[str] = []

    check_links(md_files, failures)
    check_structure(chapter_files, failures)
    check_forbidden_markers(md_files, failures)
    exercise_count, comprehensive_count = check_exercise_answers(chapter_files, failures)

    print(f"markdown_files={len(md_files)}")
    print(f"structured_chapters={len(chapter_files)}")
    print(f"chapter_and_appendix_exercises={exercise_count}")
    print(f"comprehensive_exercises={comprehensive_count}")

    if failures:
        print("failures:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("validation=ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
