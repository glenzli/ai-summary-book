#!/usr/bin/env python3
"""Structural validation for the five-volume Stochastic Parrot Anatomy."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
VOLUMES = tuple(ROOT / f"vol-{number:02d}" for number in range(1, 6))
LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
LABELED_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
HTML_IMAGE = re.compile(r"<img\b[^>]*\bsrc=[\"']([^\"']+)[\"']", re.IGNORECASE)
HEADING = re.compile(r"^(#{1,6})\s+")
HEADING_TEXT = re.compile(r"^#{1,6}\s+(.+?)\s*$", re.MULTILINE)
EXPLICIT_ANCHOR = re.compile(r"<a\s+(?:id|name)=[\"']([^\"']+)[\"']", re.IGNORECASE)
FENCE = re.compile(r"^\s*(`{3,}|~{3,})")
DISPLAY_MATH = re.compile(r"(?<!\\)\$\$")
DISPLAY_MATH_BLOCK = re.compile(r"(?s)(?<!\\)\$\$(.*?)(?<!\\)\$\$")
INLINE_MATH = re.compile(r"(?<![\\$])\$(?!\$)(.+?)(?<!\\)\$(?!\$)")
CONTROL_CHARACTER = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
MALFORMED_LATEX_COMMAND = re.compile(
    r"(?<![A-Za-z\\])(?:operatorname|frac|sqrt|mathbb|mathbf|mathrm|text|begin|end)\{"
    r"|(?<![A-Za-z\\])(?:sum|prod|int|nabla|partial|Delta|Theta|alpha|beta|gamma|"
    r"sigma|varepsilon|epsilon|infty|odot|approx|leq|geq|mid)(?:_|\()"
)
CHAPTER_TITLE_PREFIX = re.compile(
    r"^(?:第[零一二三四五六七八九十百0-9]+章|导论|序章|序言)\s*"
)
NUMBERED_EXERCISE = re.compile(
    r"^(?:#{2,6}\s+(?:练习|习题|综合练习)\b|"
    r"\*\*(?:练习|习题)\s+[A-Z0-9]+(?:\.[0-9]+)+\.)",
    re.MULTILINE,
)
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"}
RETIRED_READER_PATTERNS = {
    "vol-06": re.compile(r"vol-06|卷六"),
    "solutions": re.compile(r"\bSOLUTIONS\.md\b|习题解答"),
    "claim ledger": re.compile(r"\bCLAIM_LEDGER\.md\b|主张责任表"),
    "proof kernels": re.compile(r"app-[c-h]_[^)\s]+\.md"),
}


def fail(message: str, errors: list[str]) -> None:
    errors.append(message)


def heading_level_skips(text: str) -> list[tuple[int, int, int]]:
    skips: list[tuple[int, int, int]] = []
    previous_level = 0
    fence_character: str | None = None
    fence_length = 0

    for line_number, line in enumerate(text.splitlines(), 1):
        match = FENCE.match(line)
        if match is not None:
            marker = match.group(1)
            if fence_character is None:
                fence_character = marker[0]
                fence_length = len(marker)
            elif marker[0] == fence_character and len(marker) >= fence_length:
                fence_character = None
                fence_length = 0
            continue
        if fence_character is not None:
            continue
        match = HEADING.match(line)
        if match is None:
            continue
        level = len(match.group(1))
        if previous_level and level > previous_level + 1:
            skips.append((line_number, previous_level, level))
        previous_level = level
    return skips


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


def malformed_latex_issues(text: str) -> list[tuple[int, str]]:
    issues: list[tuple[int, str]] = []

    for block in DISPLAY_MATH_BLOCK.finditer(text):
        content = block.group(1)
        content_start = block.start(1)
        for match in MALFORMED_LATEX_COMMAND.finditer(content):
            line_number = text.count("\n", 0, content_start + match.start()) + 1
            issues.append((line_number, match.group(0)))

    for line_number, line in enumerate(text.splitlines(), 1):
        for fragment in INLINE_MATH.finditer(line):
            for match in MALFORMED_LATEX_COMMAND.finditer(fragment.group(1)):
                issues.append((line_number, match.group(0)))

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


def validate_volume(volume: Path, errors: list[str]) -> int:
    readme_path = volume / "README.md"
    if not readme_path.is_file():
        fail(f"missing volume README: {volume.relative_to(ROOT)}", errors)
        return 0

    readme = readme_path.read_text(encoding="utf-8")
    linked_chapters = {
        target: label
        for label, target in LABELED_LINK.findall(readme)
        if re.fullmatch(r"ch[0-9][0-9]_[^)]+\.md", target)
    }
    chapters = sorted(volume.glob("ch[0-9][0-9]_*.md"))
    if not chapters:
        fail(f"volume has no chapter files: {volume.relative_to(ROOT)}", errors)
        return 0

    numbers = [int(chapter.name[2:4]) for chapter in chapters]
    expected = list(range(numbers[-1] + 1))
    if numbers != expected:
        fail(
            f"non-contiguous chapter numbering in {volume.relative_to(ROOT)}: "
            f"found={numbers} expected={expected}",
            errors,
        )

    for chapter in chapters:
        text = chapter.read_text(encoding="utf-8")
        if not text.startswith("# "):
            fail(f"chapter lacks leading H1: {chapter.relative_to(ROOT)}", errors)
            continue
        if chapter.name not in linked_chapters:
            fail(f"volume README omits chapter: {chapter.relative_to(ROOT)}", errors)
        elif linked_chapters[chapter.name] != chapter_display_title(text):
            fail(
                f"volume README title differs from chapter H1: {chapter.relative_to(ROOT)}",
                errors,
            )

    for target in linked_chapters:
        if not (volume / target).is_file():
            fail(f"volume README links missing chapter: {volume.name}/{target}", errors)
    return len(chapters)


def main() -> int:
    errors: list[str] = []
    root_readme_path = ROOT / "README.md"
    if not root_readme_path.is_file():
        fail("missing root README", errors)
        root_readme = ""
    else:
        root_readme = root_readme_path.read_text(encoding="utf-8")

    if "(00_preface_and_scope.md)" not in root_readme:
        fail("root README does not link the book introduction", errors)

    chapter_count = 0
    for volume in VOLUMES:
        if not volume.is_dir():
            fail(f"missing volume directory: {volume.relative_to(ROOT)}", errors)
            continue
        if f"({volume.name}/README.md)" not in root_readme:
            fail(f"root README does not link {volume.name}", errors)
        chapter_count += validate_volume(volume, errors)

    if (ROOT / "vol-06").exists():
        fail("retired vol-06 directory still exists", errors)

    markdown_files = sorted(ROOT.rglob("*.md"))
    anchor_cache: dict[Path, set[str]] = {}
    referenced_assets: set[Path] = set()

    for path in markdown_files:
        text = path.read_text(encoding="utf-8")
        relative = path.relative_to(ROOT)
        reader_facing = "editorial" not in path.parts

        if reader_facing:
            if NUMBERED_EXERCISE.search(text):
                fail(f"numbered exercise remains: {relative}", errors)
            for name, pattern in RETIRED_READER_PATTERNS.items():
                if pattern.search(text):
                    fail(f"retired {name} reference remains: {relative}", errors)

        for match in CONTROL_CHARACTER.finditer(text):
            line_number = text.count("\n", 0, match.start()) + 1
            fail(f"control character in Markdown: {relative}:{line_number}", errors)
        for line_number, command in malformed_latex_issues(text):
            fail(
                f"possible missing LaTeX backslash: {relative}:{line_number} "
                f"({command})",
                errors,
            )

        for issue in markup_balance_issues(text):
            fail(f"{issue}: {relative}", errors)
        for line_number, previous, current in heading_level_skips(text):
            fail(
                f"heading level skip: {relative}:{line_number} H{previous} -> H{current}",
                errors,
            )

        for target in LINK.findall(text):
            target = target.strip().removeprefix("<").removesuffix(">")
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            target_path, separator, fragment = target.partition("#")
            resolved = (path.parent / target_path).resolve() if target_path else path.resolve()
            if not resolved.exists():
                fail(f"broken local link: {relative} -> {target_path}", errors)
                continue
            if resolved.suffix.lower() in IMAGE_SUFFIXES:
                referenced_assets.add(resolved)
            if separator and fragment and resolved.suffix.lower() == ".md":
                if resolved not in anchor_cache:
                    anchor_cache[resolved] = markdown_anchor_ids(
                        resolved.read_text(encoding="utf-8")
                    )
                if fragment not in anchor_cache[resolved]:
                    fail(f"broken local anchor: {relative} -> {target}", errors)

        for target in HTML_IMAGE.findall(text):
            if target.startswith(("http://", "https://", "data:")):
                continue
            resolved = (path.parent / target.split("#", 1)[0]).resolve()
            if not resolved.exists():
                fail(f"broken HTML image: {relative} -> {target}", errors)
            else:
                referenced_assets.add(resolved)

    image_assets = {
        path.resolve()
        for path in ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }
    for orphan in sorted(image_assets - referenced_assets):
        fail(f"orphan image asset: {orphan.relative_to(ROOT)}", errors)

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        print(f"stochastic-parrot-anatomy validation failed: {len(errors)} error(s)")
        return 1

    print(
        "stochastic-parrot-anatomy validation passed: "
        f"volumes={len(VOLUMES)} chapters={chapter_count} "
        f"markdown_files={len(markdown_files)} images={len(image_assets)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
