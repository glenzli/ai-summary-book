#!/usr/bin/env python3
"""Structural validation for the six-volume after-output textbook."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
VOLUMES = tuple(ROOT / f"vol-{number:02d}" for number in range(1, 7))
LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
EXERCISE = re.compile(r"^\*\*练习 ([A-Z]+\d*(?:\.\d+)+|\d+\.\d+)\.\*\*", re.MULTILINE)
SOLUTION_FILES = {
    "root": ROOT / "SOLUTIONS.md",
    "vol2": ROOT / "vol-02" / "SOLUTIONS.md",
    "P": ROOT / "vol-03" / "PROBABILITY_SOLUTIONS.md",
    "R": ROOT / "vol-03" / "REPRODUCIBILITY_SOLUTIONS.md",
    "M": ROOT / "vol-04" / "INTERPRETABILITY_SOLUTIONS.md",
    "V": ROOT / "vol-04" / "VERIFICATION_SOLUTIONS.md",
}


def fail(message: str, errors: list[str]) -> None:
    errors.append(message)


def exercise_group(label: str) -> str:
    if label[0].isdigit():
        return "vol2"
    if label.startswith("P"):
        return "P"
    if label.startswith("R"):
        return "R"
    if label.startswith("M"):
        return "M"
    if label.startswith("V"):
        return "V"
    return "root"


def main() -> int:
    errors: list[str] = []

    for volume in VOLUMES:
        if not volume.is_dir():
            fail(f"missing volume directory: {volume.relative_to(ROOT)}", errors)
            continue
        if not (volume / "README.md").is_file():
            fail(f"missing volume README: {volume.relative_to(ROOT)}", errors)
        chapters = sorted(volume.glob("ch[0-9][0-9]_*.md"))
        if not chapters:
            fail(f"volume has no chapter files: {volume.relative_to(ROOT)}", errors)
        for chapter in chapters:
            text = chapter.read_text(encoding="utf-8")
            if not text.startswith("# "):
                fail(f"chapter lacks leading H1: {chapter.relative_to(ROOT)}", errors)

    legacy_body = sorted(ROOT.glob("[0-9][0-9]*_volume_*.md"))
    if legacy_body:
        fail(f"legacy flat body files remain: {legacy_body}", errors)

    markdown_files = sorted(ROOT.rglob("*.md"))
    for path in markdown_files:
        text = path.read_text(encoding="utf-8")
        for target in LINK.findall(text):
            target = target.strip().removeprefix("<").removesuffix(">")
            target = target.split("#", 1)[0]
            if not target or target.startswith(("http://", "https://", "mailto:")):
                continue
            resolved = (path.parent / target).resolve()
            if not resolved.exists():
                fail(f"broken local link: {path.relative_to(ROOT)} -> {target}", errors)

    content_labels: dict[str, list[str]] = {key: [] for key in SOLUTION_FILES}
    for path in markdown_files:
        if path in SOLUTION_FILES.values() or path.name.endswith("SOLUTIONS.md"):
            continue
        for label in EXERCISE.findall(path.read_text(encoding="utf-8")):
            content_labels[exercise_group(label)].append(label)

    for group, solution_path in SOLUTION_FILES.items():
        if not solution_path.is_file():
            fail(f"missing solution file for {group}: {solution_path.relative_to(ROOT)}", errors)
            continue
        solutions = EXERCISE.findall(solution_path.read_text(encoding="utf-8"))
        content = content_labels[group]
        if len(content) != len(set(content)):
            fail(f"duplicate exercise locator in {group}", errors)
        if len(solutions) != len(set(solutions)):
            fail(f"duplicate solution locator in {group}", errors)
        missing = sorted(set(content) - set(solutions))
        extra = sorted(set(solutions) - set(content))
        if missing:
            fail(f"missing solutions in {group}: {missing}", errors)
        if extra:
            fail(f"orphan solutions in {group}: {extra}", errors)

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        print(f"after-output validation failed: {len(errors)} error(s)")
        return 1

    exercise_count = sum(len(labels) for labels in content_labels.values())
    chapter_count = sum(len(list(volume.glob("ch[0-9][0-9]_*.md"))) for volume in VOLUMES)
    print(
        "after-output validation passed: "
        f"volumes={len(VOLUMES)} chapters={chapter_count} exercises={exercise_count}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
