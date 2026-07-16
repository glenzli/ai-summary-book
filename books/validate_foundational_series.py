#!/usr/bin/env python3
"""Mechanical checks for the foundational textbook series."""

from __future__ import annotations

import argparse
import collections
import pathlib
import re
import sys


ROOT = pathlib.Path(__file__).resolve().parent
BOOKS = [
    "mathematical-physics-foundations",
    "probability-stochastic-information",
    "computation-types-semantics",
    "textbook-writing-methodology",
]

REQUIRED = {
    "README.md",
    "SKILL.md",
    "NOTATION.md",
    "SOURCES.md",
    "THEOREM_INDEX.md",
    "DEPENDENCY_GRAPH.md",
    "CONTENT_CLOSURE_AUDIT.md",
    "SOLUTIONS.md",
}

LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+\.md(?:#[^)]+)?)\)")
LABEL_RE = re.compile(
    r"\*\*(定理|命题|引理|推论|定义|公理|约定|例|练习|原则|猜想|外部输入) "
    r"([A-Z]*[0-9]+(?:\.[0-9]+)+(?:[A-Z])?|[A-Z]+(?:\.[0-9]+)+)"
)
INDEX_RE = re.compile(
    r"^\|\s*([A-Z]*[0-9]+(?:\.[0-9]+)+(?:[A-Z])?|[A-Z]+(?:\.[0-9]+)+)\s*\|"
)
EXERCISE_RE = re.compile(r"\*\*练习 ([A-Z]*[0-9]+\.[0-9]+|[A-Z]+\.[0-9]+)")
SOLUTION_RE = re.compile(
    r"^(?:##\s*(?:练习\s*)?|\*\*(?:练习\s*)?)"
    r"([A-Z]*[0-9]+\.[0-9]+|[A-Z]+\.[0-9]+)"
    r"(?:[.。]|\*\*|\s*$)",
    re.M,
)
SKETCH_RE = re.compile(r"^\s*\*\*(?:证明草图|证明思路|推导草图)[.。:]?", re.M)
EXAMPLE_RE = re.compile(
    r"\*\*(?:例子?|边界例|反例|案例|计算)\s*[A-Z]*[0-9]+(?:\.[0-9]+)+"
)
CHAPTER_RE = re.compile(r"[0-9]{2}_.+\.md$")


def check_book(name: str) -> list[str]:
    errors: list[str] = []
    root = ROOT / name
    if not root.exists():
        return [f"{name}: missing directory"]

    files = sorted(root.glob("*.md"))
    content_files = [path for path in files if path.name != "SKILL.md"]
    present = {p.name for p in files}
    for required in sorted(REQUIRED - present):
        errors.append(f"{name}: missing {required}")

    links = []
    for path in files:
        text = path.read_text()
        for match in LINK_RE.finditer(text):
            target = match.group(1).split("#", 1)[0]
            if "://" in target or target.startswith("/"):
                continue
            links.append((path.name, target))
            if not (path.parent / target).exists():
                errors.append(f"{name}: broken link {path.name} -> {target}")

    labels = []
    for path in content_files:
        if path.name in {"THEOREM_INDEX.md", "SOLUTIONS.md"}:
            continue
        for line_no, line in enumerate(path.read_text().splitlines(), 1):
            for match in LABEL_RE.finditer(line):
                labels.append((match.group(1), match.group(2), path.name, line_no))

    grouped: dict[tuple[str, str], list[tuple[str, int]]] = collections.defaultdict(list)
    for kind, num, file_name, line_no in labels:
        grouped[(kind, num)].append((file_name, line_no))
    for key, locations in grouped.items():
        if len(locations) > 1:
            errors.append(f"{name}: duplicate label {key} at {locations}")

    index_path = root / "THEOREM_INDEX.md"
    if index_path.exists():
        ids = []
        for line in index_path.read_text().splitlines():
            match = INDEX_RE.match(line)
            if match:
                ids.append(match.group(1))
        body = "\n".join(
            path.read_text()
            for path in content_files
            if path.name != "THEOREM_INDEX.md"
        )
        for theorem_id in ids:
            if theorem_id.startswith("EI-"):
                continue
            body_id = theorem_id[1:] if re.match(r"^T[0-9]", theorem_id) else theorem_id
            if not (
                re.search(
                    r"(?:定理|命题|引理|推论|定义|公理|约定|例|原则|猜想|外部输入) "
                    + re.escape(body_id)
                    + r"(?:\.|\b)",
                    body,
                )
                or re.search(r"\b" + re.escape(theorem_id) + r"\b", body)
            ):
                errors.append(f"{name}: theorem index id not found: {theorem_id}")

    chapter_exercises = []
    for path in files:
        if re.match(r"(?:[0-9]{2}|[A-Z])_", path.name):
            chapter_exercises.extend(EXERCISE_RE.findall(path.read_text()))
        if CHAPTER_RE.fullmatch(path.name):
            chapter_text = path.read_text()
            if SKETCH_RE.search(chapter_text):
                errors.append(
                    f"{name}: non-final proof sketch remains in {path.name}"
                )
            if not EXERCISE_RE.search(chapter_text):
                errors.append(f"{name}: chapter has no numbered exercise: {path.name}")
            if not EXAMPLE_RE.search(chapter_text):
                errors.append(
                    f"{name}: chapter has no numbered worked example/case: {path.name}"
                )
    solution_path = root / "SOLUTIONS.md"
    if solution_path.exists():
        solutions = SOLUTION_RE.findall(solution_path.read_text())
        missing = sorted(set(chapter_exercises) - set(solutions))
        if missing:
            errors.append(f"{name}: exercises without solutions: {missing}")

    audit_path = root / "CONTENT_CLOSURE_AUDIT.md"
    if audit_path.exists():
        audit_text = audit_path.read_text()
        for required_heading in ("机械闭合", "内容闭合"):
            if required_heading not in audit_text:
                errors.append(
                    f"{name}: closure audit does not separate {required_heading}"
                )

    print(
        f"{name}: files={len(files)} links={len(links)} "
        f"labels={len(labels)} exercises={len(chapter_exercises)}"
    )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("books", nargs="*", choices=BOOKS)
    args = parser.parse_args()
    selected = args.books or BOOKS
    errors: list[str] = []
    for book in selected:
        errors.extend(check_book(book))
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
