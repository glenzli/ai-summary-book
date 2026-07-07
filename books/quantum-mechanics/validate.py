#!/usr/bin/env python3
"""Repository-local consistency checks for the quantum mechanics textbook."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CHAPTER_FILE = re.compile(r"^(?:[0-9][0-9]|[ABC])_.*\.md$")
MARKDOWN_LINK = re.compile(r"\[[^\]\n]+\]\(([^)]+)\)")
EXERCISE = re.compile(r"^\*\*练习\s+([0-9]+|[A-Z])\.([0-9]+)\.\*\*", re.M)
ANSWER = re.compile(r"^\*\*答案\s+([0-9]+|[A-Z])\.([0-9]+)\.\*\*", re.M)
HINT = re.compile(r"^\*\*提示\s+([0-9]+|[A-Z])\.([0-9]+)\.\*\*", re.M)
COMPREHENSIVE = re.compile(r"^\*\*综合题\s+([0-9]+)\.\*\*", re.M)
COMPREHENSIVE_HINT = re.compile(r"^\*\*综合题\s+([0-9]+)\s+提示\.\*\*", re.M)
COMPREHENSIVE_ANSWER = re.compile(r"^## 综合题\s+([0-9]+)$", re.M)
EXTERNAL_INPUT = re.compile(r"QM-EXT-([0-9]+)")
EXTERNAL_INPUT_ROW = re.compile(r"^\|\s*QM-EXT-([0-9]+)\s*\|", re.M)
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
    required = ("## 本章目标", "## 依赖前置知识", "本章小结", "## 练习")
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


def check_layout_hygiene(md_files: list[Path], failures: list[str]) -> None:
    for path in md_files:
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if line.rstrip() != line:
                fail(f"trailing whitespace in {path.name}:{lineno}", failures)


def check_term_index(failures: list[str]) -> int:
    path = ROOT / "TERM_INDEX.md"
    rows: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|") or line.startswith("|---"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 3 or cells[0] == "中文":
            continue
        rows.append((cells[0], cells[1]))

    cn_terms = [row[0] for row in rows]
    en_terms = [row[1] for row in rows]
    if len(cn_terms) != len(set(cn_terms)):
        fail("duplicate Chinese term in TERM_INDEX.md", failures)
    if len(en_terms) != len(set(en_terms)):
        fail("duplicate English term in TERM_INDEX.md", failures)
    return len(rows)


def check_external_inputs(chapter_files: list[Path], failures: list[str]) -> int:
    defined = EXTERNAL_INPUT_ROW.findall((ROOT / "THEOREM_DEPENDENCIES.md").read_text(encoding="utf-8"))
    defined_set = set(defined)
    if len(defined) != len(defined_set):
        fail("duplicate QM-EXT label in THEOREM_DEPENDENCIES.md", failures)

    if defined_set:
        numbers = sorted(int(label) for label in defined_set)
        expected = set(str(n) for n in range(1, numbers[-1] + 1))
        missing = sorted(expected - defined_set, key=int)
        if missing:
            fail(f"non-consecutive QM-EXT labels: missing {', '.join(missing)}", failures)

    used_set: set[str] = set()
    for path in chapter_files:
        used_set.update(EXTERNAL_INPUT.findall(path.read_text(encoding="utf-8")))

    unused = sorted(defined_set - used_set, key=int)
    undefined = sorted(used_set - defined_set, key=int)
    if unused:
        fail(f"QM-EXT labels defined but unused in structured chapters: {', '.join(unused)}", failures)
    if undefined:
        fail(f"QM-EXT labels used but undefined: {', '.join(undefined)}", failures)
    return len(defined_set)


def check_exercise_support(chapter_files: list[Path], failures: list[str]) -> tuple[int, int, int, int]:
    solutions = (ROOT / "SOLUTIONS.md").read_text(encoding="utf-8")
    hints = (ROOT / "HINTS.md").read_text(encoding="utf-8")
    answer_keys = {f"{a}.{b}" for a, b in ANSWER.findall(solutions)}
    hint_keys = {f"{a}.{b}" for a, b in HINT.findall(hints)}
    exercises: list[str] = []
    for path in chapter_files:
        exercises.extend(f"{a}.{b}" for a, b in EXERCISE.findall(path.read_text(encoding="utf-8")))

    for key in exercises:
        if key not in answer_keys:
            fail(f"missing solution: exercise {key}", failures)
        if key not in hint_keys:
            fail(f"missing hint: exercise {key}", failures)

    comprehensive = (ROOT / "COMPREHENSIVE_EXERCISES.md").read_text(encoding="utf-8")
    comprehensive_solutions = (ROOT / "COMPREHENSIVE_SOLUTIONS.md").read_text(encoding="utf-8")
    comp_keys = set(COMPREHENSIVE.findall(comprehensive))
    comp_hint_keys = set(COMPREHENSIVE_HINT.findall(hints))
    comp_answer_keys = set(COMPREHENSIVE_ANSWER.findall(comprehensive_solutions))
    for key in sorted(comp_keys, key=int):
        if key not in comp_hint_keys:
            fail(f"missing comprehensive hint: problem {key}", failures)
        if key not in comp_answer_keys:
            fail(f"missing comprehensive solution: problem {key}", failures)

    return len(exercises), len(hint_keys), len(comp_keys), len(comp_hint_keys)


def main() -> int:
    md_files = sorted(ROOT.glob("*.md"))
    chapter_files = [path for path in md_files if CHAPTER_FILE.match(path.name)]
    failures: list[str] = []

    check_links(md_files, failures)
    check_structure(chapter_files, failures)
    check_forbidden_markers(md_files, failures)
    check_layout_hygiene(md_files, failures)
    term_count = check_term_index(failures)
    external_count = check_external_inputs(chapter_files, failures)
    exercise_count, hint_count, comprehensive_count, comprehensive_hint_count = check_exercise_support(chapter_files, failures)

    print(f"markdown_files={len(md_files)}")
    print(f"structured_chapters={len(chapter_files)}")
    print(f"term_index_rows={term_count}")
    print(f"external_inputs={external_count}")
    print(f"chapter_and_appendix_exercises={exercise_count}")
    print(f"chapter_and_appendix_hints={hint_count}")
    print(f"comprehensive_exercises={comprehensive_count}")
    print(f"comprehensive_hints={comprehensive_hint_count}")

    if failures:
        print("failures:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("validation=ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
