#!/usr/bin/env python3
"""Validate the Machine Hermeneutics textbook and its narrative closure."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CHAPTERS = [ROOT / "00_preface_and_scope.md"] + [
    ROOT / f"{n:02d}_{slug}.md"
    for n, slug in [
        (1, "levels_and_claims"),
        (2, "behavior_and_identification"),
        (3, "gradient_attribution"),
        (4, "attention_and_attribution"),
        (5, "probes_and_representation"),
        (6, "interventions_and_patching"),
        (7, "circuits_and_sparse_features"),
        (8, "robustness_and_underdetermination"),
        (9, "emergence_and_benchmarks"),
        (10, "psychological_vocabulary"),
        (11, "protocol_and_cases"),
    ]
]
REQUIRED = CHAPTERS + [ROOT / name for name in (
    "README.md", "SKILL.md", "GLOSSARY.md", "SOURCES.md",
    "CLAIM_LEDGER.md", "SOLUTIONS.md", "CLOSURE_AUDIT.md",
)]
EXERCISE = re.compile(r"\*\*练习\s+(\d+\.\d+)\.")
SOURCE_REF = re.compile(r"\[S(\d{2})\]\(SOURCES\.md#s(\d{2})\)")
SOURCE_ANCHOR = re.compile(r'<a id="s(\d{2})"></a>')
CLAIM = re.compile(r"\|\s*(MH-\d{2})\s*\|")
FORMAL = re.compile(r"\*\*(?:定理|命题|引理)\s+(\d+\.\d+)[^*]*\*\*")
DEFINITION = re.compile(r"\*\*定义\s+(\d+\.\d+)[^*]*\*\*")
LOCAL_MD_LINK = re.compile(r"\]\((?!https?://)([^)#]+\.md)(?:#[^)]+)?\)")
OLD_TEMPLATE_HEADING = re.compile(
    r"^##\s+(?:本章目标|依赖|主线|本章小结)\s*$", re.MULTILINE
)
SECTION_HEADING = re.compile(r"^##\s+.+$", re.MULTILINE)
H2_TITLE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
CASE_HEADING = re.compile(
    r"^#{2,3}\s+.*(?:案例|实验|计算|研究|评估).*$", re.MULTILINE
)
NUMBERED_SECTION_TITLE = re.compile(r"^\d+(?:\.\d+)*\s+")
CHAPTER_PREVIEW = re.compile(r"下一章|下章")


def main() -> int:
    errors: list[str] = []
    for path in REQUIRED:
        if not path.is_file():
            errors.append(f"missing required file: {path.name}")
    if errors:
        print("\n".join(errors))
        return 1

    solutions = (ROOT / "SOLUTIONS.md").read_text(encoding="utf-8")
    sources = (ROOT / "SOURCES.md").read_text(encoding="utf-8")
    source_ids = SOURCE_ANCHOR.findall(sources)
    if len(source_ids) != len(set(source_ids)):
        errors.append("duplicate source anchors")
    if source_ids != [f"{i:02d}" for i in range(1, len(source_ids) + 1)]:
        errors.append("source anchors must be consecutive from s01")

    ledger = (ROOT / "CLAIM_LEDGER.md").read_text(encoding="utf-8")
    claims = CLAIM.findall(ledger)
    if len(claims) != len(set(claims)):
        errors.append("duplicate claim IDs")
    expected_claims = [f"MH-{i:02d}" for i in range(1, len(claims) + 1)]
    if claims != expected_claims:
        errors.append("claim IDs must be consecutive from MH-01")

    seen: set[str] = set()
    seen_formal: set[str] = set()
    referenced_sources: set[str] = set()
    case_chapters = 0
    standalone_closings: list[str] = []
    for chapter in CHAPTERS:
        chapter_text = chapter.read_text(encoding="utf-8")
        if "## 练习" not in chapter_text:
            errors.append(f"{chapter.name}: missing ## 练习")
        old_headings = OLD_TEMPLATE_HEADING.findall(chapter_text)
        if old_headings:
            errors.append(
                f"{chapter.name}: legacy template heading(s): "
                + ", ".join(old_headings)
            )
        first_section = SECTION_HEADING.search(chapter_text)
        if first_section is None:
            errors.append(f"{chapter.name}: no level-2 section heading")
        else:
            first_line_end = chapter_text.find("\n")
            intro = chapter_text[first_line_end + 1:first_section.start()].strip()
            intro_paragraphs = [p for p in re.split(r"\n\s*\n", intro) if p.strip()]
            if len(intro_paragraphs) < 2 or len(intro) < 100:
                errors.append(
                    f"{chapter.name}: chapter opening must contain at least "
                    "two substantive narrative paragraphs"
                )
        if CASE_HEADING.search(chapter_text):
            case_chapters += 1
        else:
            errors.append(
                f"{chapter.name}: missing a headed case, experiment, calculation, "
                "research design, or evaluation"
            )
        h2_titles = H2_TITLE.findall(chapter_text)
        if "练习" in h2_titles:
            exercise_index = h2_titles.index("练习")
            before_exercises = h2_titles[:exercise_index]
            if before_exercises:
                closing_title = before_exercises[-1]
                if not NUMBERED_SECTION_TITLE.match(closing_title):
                    standalone_closings.append(chapter.name)
        if chapter != CHAPTERS[0] and CHAPTER_PREVIEW.search(chapter_text):
            errors.append(f"{chapter.name}: chapter-preview phrasing remains")
        if chapter_text.count("$$") % 2:
            errors.append(f"{chapter.name}: unbalanced display-math delimiters")
        if chapter_text.count("```") % 2:
            errors.append(f"{chapter.name}: unbalanced fenced-code delimiters")
        for bad in (r"\mathbb R_{≥", ",ldots", "R≥", r"\!left", "$mathcal"):
            if bad in chapter_text:
                errors.append(f"{chapter.name}: malformed token {bad!r}")
        for left, right in SOURCE_REF.findall(chapter_text):
            if left != right:
                errors.append(f"{chapter.name}: mismatched source link S{left}/s{right}")
            if left not in source_ids:
                errors.append(f"{chapter.name}: unknown source S{left}")
            referenced_sources.add(left)
        formal_matches = list(FORMAL.finditer(chapter_text))
        definition_numbers = DEFINITION.findall(chapter_text)
        for number in definition_numbers:
            if number in seen_formal:
                errors.append(f"duplicate numbered statement: {number}")
            seen_formal.add(number)
        for index, match in enumerate(formal_matches):
            number = match.group(1)
            if number in seen_formal:
                errors.append(f"duplicate numbered statement: {number}")
            seen_formal.add(number)
            end = formal_matches[index + 1].start() if index + 1 < len(formal_matches) else len(chapter_text)
            section_end = chapter_text.find("\n## ", match.end(), end)
            if section_end != -1:
                end = section_end
            if "**证明." not in chapter_text[match.end():end]:
                errors.append(f"{chapter.name}: formal result {number} lacks proof")
        for number in EXERCISE.findall(chapter_text):
            if number in seen:
                errors.append(f"duplicate exercise: {number}")
            seen.add(number)
            if f"**练习 {number}.**" not in solutions:
                errors.append(f"missing solution: {number}")

    extras = sorted(set(EXERCISE.findall(solutions)) - seen)
    if extras:
        errors.append("solutions without exercises: " + ", ".join(extras))
    unused_sources = sorted(set(source_ids) - referenced_sources)
    if unused_sources:
        errors.append("sources without chapter references: " + ", ".join(f"S{x}" for x in unused_sources))
    if len(standalone_closings) > 3:
        errors.append(
            "standalone pre-exercise synthesis headings remain in too many chapters: "
            + ", ".join(standalone_closings)
        )

    for path in REQUIRED:
        document = path.read_text(encoding="utf-8")
        for target in LOCAL_MD_LINK.findall(document):
            resolved = (path.parent / target).resolve()
            if not resolved.is_file():
                errors.append(f"{path.name}: broken local link {target}")
    readme_lines = (ROOT / "README.md").read_text(encoding="utf-8").splitlines()
    author_line = "**作者：Dr. Stochastic Parrot**"
    if readme_lines.count(author_line) != 1:
        errors.append("README.md: exact author attribution must appear once")
    if errors:
        print("\n".join(errors))
        return 1
    print(
        "machine-hermeneutics validation: "
        f"chapters={len(CHAPTERS)} case_chapters={case_chapters} "
        f"exercises={len(seen)} errors=0"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
