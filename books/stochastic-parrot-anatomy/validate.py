#!/usr/bin/env python3
"""Structural validation for the six-volume Stochastic Parrot Anatomy."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
VOLUMES = tuple(ROOT / f"vol-{number:02d}" for number in range(1, 7))
LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
LABELED_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
HTML_IMAGE = re.compile(r"<img\b[^>]*\bsrc=[\"']([^\"']+)[\"']", re.IGNORECASE)
EXERCISE = re.compile(r"^\*\*练习 ([A-Z]+\d*(?:\.\d+)+|\d+\.\d+)\.\*\*", re.MULTILINE)
SOLUTION_FILES = {
    "root": ROOT / "SOLUTIONS.md",
    "vol2": ROOT / "vol-02" / "SOLUTIONS.md",
    "P": ROOT / "vol-03" / "SOLUTIONS.md",
    "R": ROOT / "vol-03" / "SOLUTIONS.md",
    "M": ROOT / "vol-04" / "SOLUTIONS.md",
    "V": ROOT / "vol-04" / "SOLUTIONS.md",
}

LOCATOR_HEADING = re.compile(
    r"^#{2,6} ((?:[SPRMV][0-9]+|[C-H])\.[0-9]+(?:\.[0-9]+)*)",
    re.MULTILINE,
)
INTERNAL_LOCATOR = re.compile(
    r"\b(?:[SPRMV]|[C-H])[0-9]+\.[0-9]+(?:\.[0-9]+)*\b"
)
LOCATOR_DEFINITION = re.compile(
    r"^(?:#{2,6}\s+|\*\*(?:定义|定理|命题|引理|推论|反例|"
    r"外部输入(?:定理)?|练习)\s+)"
    r"((?:[SPRMV]|[C-H])[0-9]+\.[0-9]+(?:\.[0-9]+)*)",
    re.MULTILINE,
)
HEADING = re.compile(r"^(#{1,6})\s+")
HEADING_TEXT = re.compile(r"^#{1,6}\s+(.+?)\s*$", re.MULTILINE)
EXPLICIT_ANCHOR = re.compile(r"<a\s+(?:id|name)=[\"']([^\"']+)[\"']", re.IGNORECASE)
FENCE = re.compile(r"^\s*(`{3,}|~{3,})")
DISPLAY_MATH = re.compile(r"(?<!\\)\$\$")
CHAPTER_TITLE_PREFIX = re.compile(
    r"^(?:第[零一二三四五六七八九十百0-9]+章|"
    r"卷[一二三四五六]导论|导论|序章|序言)\s*"
)
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"}


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


def heading_level_skips(text: str) -> list[tuple[int, int, int]]:
    skips: list[tuple[int, int, int]] = []
    previous_level = 0
    in_fence = False
    for line_number, line in enumerate(text.splitlines(), 1):
        if line.startswith("```") or line.startswith("~~~"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        match = HEADING.match(line)
        if match is None:
            continue
        level = len(match.group(1))
        if previous_level and level > previous_level + 1:
            skips.append((line_number, previous_level, level))
        previous_level = level
    return skips


def nonterminal_exercise_sections(text: str) -> list[int]:
    h2: list[tuple[int, str]] = []
    in_fence = False
    for line_number, line in enumerate(text.splitlines(), 1):
        if line.startswith("```") or line.startswith("~~~"):
            in_fence = not in_fence
            continue
        if not in_fence and line.startswith("## "):
            h2.append((line_number, line[3:].strip()))
    exercise_titles = {"练习", "习题", "综合练习"}
    return [line_number for line_number, title in h2[:-1] if title in exercise_titles]


def markup_balance_issues(text: str) -> list[str]:
    issues: list[str] = []
    fence_character: str | None = None
    fence_length = 0
    fence_line = 0
    display_math_delimiters = 0

    for line_number, line in enumerate(text.splitlines(), 1):
        match = FENCE.match(line)
        if match is not None:
            marker = match.group(1)
            if fence_character is None:
                fence_character = marker[0]
                fence_length = len(marker)
                fence_line = line_number
            elif marker[0] == fence_character and len(marker) >= fence_length:
                fence_character = None
                fence_length = 0
                fence_line = 0
            continue
        if fence_character is None:
            display_math_delimiters += len(DISPLAY_MATH.findall(line))

    if fence_character is not None:
        issues.append(f"unclosed code fence opened at line {fence_line}")
    if display_math_delimiters % 2:
        issues.append("unbalanced display-math delimiters ($$)")
    return issues


def chapter_display_title(text: str) -> str:
    first_line = text.splitlines()[0].removeprefix("# ").strip()
    return CHAPTER_TITLE_PREFIX.sub("", first_line).strip()


def markdown_anchor_ids(text: str) -> set[str]:
    anchors = set(EXPLICIT_ANCHOR.findall(text))
    occurrences: dict[str, int] = {}
    for heading in HEADING_TEXT.findall(text):
        normalized = re.sub(r"<[^>]+>", "", heading)
        normalized = re.sub(r"[`*_~]", "", normalized).strip().lower()
        normalized = "".join(
            character
            for character in normalized
            if character.isalnum() or character in " -_"
        )
        base = re.sub(r"\s+", "-", normalized)
        count = occurrences.get(base, 0)
        anchors.add(base if count == 0 else f"{base}-{count}")
        occurrences[base] = count + 1
    return anchors


def main() -> int:
    errors: list[str] = []

    root_readme = (ROOT / "README.md").read_text(encoding="utf-8")
    if "(00_preface_and_scope.md)" not in root_readme:
        fail("root README does not link the book preface", errors)

    for volume in VOLUMES:
        if not volume.is_dir():
            fail(f"missing volume directory: {volume.relative_to(ROOT)}", errors)
            continue
        volume_readme_path = volume / "README.md"
        if not volume_readme_path.is_file():
            fail(f"missing volume README: {volume.relative_to(ROOT)}", errors)
            volume_readme = ""
        else:
            volume_readme = volume_readme_path.read_text(encoding="utf-8")
        readme_chapter_labels = {
            target: label
            for label, target in LABELED_LINK.findall(volume_readme)
            if re.fullmatch(r"ch[0-9][0-9]_[^)]+\.md", target)
        }
        chapters = sorted(volume.glob("ch[0-9][0-9]_*.md"))
        if not chapters:
            fail(f"volume has no chapter files: {volume.relative_to(ROOT)}", errors)
        for chapter in chapters:
            text = chapter.read_text(encoding="utf-8")
            if not text.startswith("# "):
                fail(f"chapter lacks leading H1: {chapter.relative_to(ROOT)}", errors)
            for line_number in nonterminal_exercise_sections(text):
                fail(
                    f"nonterminal H2 exercise section: {chapter.relative_to(ROOT)}:{line_number}",
                    errors,
                )
            if f"({chapter.name})" not in volume_readme:
                fail(
                    f"volume README omits chapter: {chapter.relative_to(ROOT)}",
                    errors,
                )
            elif readme_chapter_labels.get(chapter.name) != chapter_display_title(text):
                fail(
                    "volume README title differs from chapter H1: "
                    f"{chapter.relative_to(ROOT)}",
                    errors,
                )

    legacy_body = sorted(ROOT.glob("[0-9][0-9]*_volume_*.md"))
    if legacy_body:
        fail(f"legacy flat body files remain: {legacy_body}", errors)

    markdown_files = sorted(ROOT.rglob("*.md"))
    locators: dict[str, list[str]] = {}
    defined_locators: set[str] = set()
    referenced_locators: dict[str, set[str]] = {}
    anchor_cache: dict[Path, set[str]] = {}
    referenced_assets: set[Path] = set()
    for path in markdown_files:
        text = path.read_text(encoding="utf-8")
        if "editorial" not in path.parts:
            defined_locators.update(LOCATOR_DEFINITION.findall(text))
            for locator in INTERNAL_LOCATOR.findall(text):
                referenced_locators.setdefault(locator, set()).add(
                    str(path.relative_to(ROOT))
                )
            for issue in markup_balance_issues(text):
                fail(f"{issue}: {path.relative_to(ROOT)}", errors)
            for line_number, previous, current in heading_level_skips(text):
                fail(
                    "heading level skip: "
                    f"{path.relative_to(ROOT)}:{line_number} H{previous} -> H{current}",
                    errors,
                )
            for locator in LOCATOR_HEADING.findall(text):
                locators.setdefault(locator, []).append(str(path.relative_to(ROOT)))
        for target in LINK.findall(text):
            target = target.strip().removeprefix("<").removesuffix(">")
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            target_path, separator, fragment = target.partition("#")
            resolved = (path.parent / target_path).resolve() if target_path else path.resolve()
            if not resolved.exists():
                fail(f"broken local link: {path.relative_to(ROOT)} -> {target_path}", errors)
                continue
            if resolved.suffix.lower() in IMAGE_SUFFIXES:
                referenced_assets.add(resolved)
            if separator and fragment and resolved.suffix.lower() == ".md":
                if resolved not in anchor_cache:
                    anchor_cache[resolved] = markdown_anchor_ids(
                        resolved.read_text(encoding="utf-8")
                    )
                if fragment not in anchor_cache[resolved]:
                    fail(
                        f"broken local anchor: {path.relative_to(ROOT)} -> {target}",
                        errors,
                    )
        for target in HTML_IMAGE.findall(text):
            if target.startswith(("http://", "https://", "data:")):
                continue
            resolved = (path.parent / target.split("#", 1)[0]).resolve()
            if not resolved.exists():
                fail(f"broken HTML image: {path.relative_to(ROOT)} -> {target}", errors)
            else:
                referenced_assets.add(resolved)

    image_assets = {
        path.resolve()
        for path in ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }
    for orphan in sorted(image_assets - referenced_assets):
        fail(f"orphan image asset: {orphan.relative_to(ROOT)}", errors)

    for locator, paths in sorted(locators.items()):
        if len(paths) > 1:
            fail(f"duplicate heading locator {locator}: {paths}", errors)

    for locator in sorted(set(referenced_locators) - defined_locators):
        fail(
            f"unresolved internal locator {locator}: "
            f"{sorted(referenced_locators[locator])}",
            errors,
        )

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
        solutions = [
            label
            for label in EXERCISE.findall(solution_path.read_text(encoding="utf-8"))
            if exercise_group(label) == group
        ]
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
        print(f"stochastic-parrot-anatomy validation failed: {len(errors)} error(s)")
        return 1

    exercise_count = sum(len(labels) for labels in content_labels.values())
    chapter_count = sum(len(list(volume.glob("ch[0-9][0-9]_*.md"))) for volume in VOLUMES)
    print(
        "stochastic-parrot-anatomy validation passed: "
        f"volumes={len(VOLUMES)} chapters={chapter_count} exercises={exercise_count}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
