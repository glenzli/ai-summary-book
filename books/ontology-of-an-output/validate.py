#!/usr/bin/env python3
"""Validate the Ontology of an Output textbook."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CHAPTERS = [ROOT / "00_preface_and_scope.md"] + [
    ROOT / f"{number:02d}_{slug}.md"
    for number, slug in [
        (1, "bytes_text_and_tokens"),
        (2, "functions_relations_and_partiality"),
        (3, "states_events_and_traces"),
        (4, "operational_semantics_of_generation"),
        (5, "probabilistic_generation"),
        (6, "tools_and_external_world"),
        (7, "concurrency_and_streaming"),
        (8, "provenance_and_identity"),
        (9, "reference_and_truth"),
        (10, "agency_authorship_and_responsibility"),
        (11, "complete_decomposition"),
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
CLAIM = re.compile(r"\*\*(?:定理|命题|推论)\s+(\d+\.\d+)")
EXTERNAL = re.compile(r"\*\*外部输入\s+(\d+\.[A-Z])")
LOCAL_MD_LINK = re.compile(r"\]\((?!https?://)([^)#]+\.md)(?:#[^)]+)?\)")
FORBIDDEN = re.compile(
    r"\b(?:TODO|TBD|FIXME)\b|待补|证明略|来源待核|显然|不难|容易看出|同理",
    re.IGNORECASE,
)
AUTHOR_LINE = "**作者：Dr. Stochastic Parrot**"
OLD_TEMPLATE_HEADING = re.compile(
    r"^##\s+(?:本章目标|依赖|本章小结|(?:\d+(?:\.\d+)*\s+)?主线)\s*$",
    re.MULTILINE,
)
SECTION_HEADING = re.compile(r"^##\s+", re.MULTILINE)
LIFECYCLE_MARKERS = (
    "轨迹",
    "反例",
    "状态",
    "字节",
    "token",
    r"u_\star",
    r"v_\star",
    r"\operatorname",
    "|---",
)


def check_fences(path: Path, text: str, errors: list[str]) -> None:
    fence_count = sum(line.lstrip().startswith("```") for line in text.splitlines())
    if fence_count % 2:
        errors.append(f"{path.name}: unbalanced fenced code blocks")
    if text.count("$$") % 2:
        errors.append(f"{path.name}: unbalanced display-math delimiters")


def check_links(path: Path, text: str, errors: list[str]) -> None:
    for target in LOCAL_MD_LINK.findall(text):
        resolved = (path.parent / target).resolve()
        if not resolved.is_file():
            errors.append(f"{path.name}: broken local Markdown link: {target}")


def solution_blocks(text: str) -> dict[str, str]:
    matches = list(EXERCISE.finditer(text))
    blocks: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        blocks[match.group(1)] = text[match.end():end].strip()
    return blocks


def check_chapter_narrative(
    path: Path,
    text: str,
    chapter_index: int,
    errors: list[str],
) -> None:
    lines = text.splitlines()
    if not lines or not lines[0].startswith("# "):
        errors.append(f"{path.name}: chapter must begin with a level-one title")
        return

    old_heading = OLD_TEMPLATE_HEADING.search(text)
    if old_heading:
        errors.append(
            f"{path.name}: forbidden outline-style heading: "
            f"{old_heading.group(0).strip()}"
        )

    first_section = SECTION_HEADING.search(text)
    if first_section is None:
        errors.append(f"{path.name}: missing substantive section headings")
        return

    title_end = text.find("\n")
    introduction = text[title_end + 1:first_section.start()].strip()
    introduction_prose = re.sub(r"[`*_>#|$\\]", "", introduction)
    introduction_prose = re.sub(r"\s+", "", introduction_prose)
    if len(introduction_prose) < 100:
        errors.append(
            f"{path.name}: opening introduction is too short to establish the problem"
        )
    if "SP404" not in introduction:
        errors.append(f"{path.name}: opening does not continue the SP404 lifecycle")

    if not any(marker in text for marker in LIFECYCLE_MARKERS):
        errors.append(
            f"{path.name}: missing worked lifecycle fragment, trace, or counterexample"
        )

    exercise_heading = "## 练习"
    if exercise_heading not in text:
        errors.append(f"{path.name}: missing {exercise_heading}")
        return

    if chapter_index < len(CHAPTERS) - 1:
        before_exercises = text.rsplit(exercise_heading, 1)[0]
        closing_window = before_exercises[-800:]
        if not any(marker in closing_window for marker in ("下一章", "最后一章")):
            errors.append(
                f"{path.name}: closing prose does not bridge to the next layer"
            )


def main() -> int:
    errors: list[str] = []

    for path in REQUIRED:
        if not path.is_file():
            errors.append(f"missing required file: {path.name}")
    if errors:
        print("\n".join(errors))
        return 1

    texts = {path: path.read_text(encoding="utf-8") for path in REQUIRED}
    for path, content in texts.items():
        check_fences(path, content, errors)
        check_links(path, content, errors)
        if FORBIDDEN.search(content):
            errors.append(f"{path.name}: unresolved placeholder or proof omission")

    expected_numbers = {
        *(f"0.{index}" for index in range(1, 5)),
        *(f"{chapter}.{index}" for chapter in range(1, 12) for index in range(1, 6)),
    }
    seen: set[str] = set()
    for index, chapter in enumerate(CHAPTERS):
        content = texts[chapter]
        check_chapter_narrative(chapter, content, index, errors)
        for number in EXERCISE.findall(content):
            if number in seen:
                errors.append(f"duplicate exercise: {number}")
            seen.add(number)

    if seen != expected_numbers:
        missing = sorted(expected_numbers - seen)
        extra = sorted(seen - expected_numbers)
        if missing:
            errors.append("missing exercises: " + ", ".join(missing))
        if extra:
            errors.append("unexpected exercises: " + ", ".join(extra))

    solutions = texts[ROOT / "SOLUTIONS.md"]
    blocks = solution_blocks(solutions)
    if set(blocks) != seen:
        missing = sorted(seen - set(blocks))
        extra = sorted(set(blocks) - seen)
        if missing:
            errors.append("missing solutions: " + ", ".join(missing))
        if extra:
            errors.append("solutions without exercises: " + ", ".join(extra))
    for number, block in blocks.items():
        if len(block) < 80:
            errors.append(f"solution {number}: answer is too short to close the exercise")

    ledger = texts[ROOT / "CLAIM_LEDGER.md"]
    for chapter in CHAPTERS:
        chapter_text = texts[chapter]
        for number in CLAIM.findall(chapter_text):
            if number not in ledger:
                errors.append(f"{chapter.name}: claim {number} absent from CLAIM_LEDGER.md")
        for number in EXTERNAL.findall(chapter_text):
            if number not in ledger:
                errors.append(f"{chapter.name}: external input {number} absent from CLAIM_LEDGER.md")
            if number not in texts[ROOT / "SOURCES.md"]:
                errors.append(f"{chapter.name}: external input {number} absent from SOURCES.md")

    sources = texts[ROOT / "SOURCES.md"]
    for required_term in (
        "UAX #15",
        "UAX #29",
        "RFC 3629",
        "Ionescu--Tulcea",
        "随机化引理",
        "Lamport",
        "PROV-CONSTRAINTS",
        "Tarski",
        "AI RMF",
    ):
        if required_term not in sources:
            errors.append(f"SOURCES.md: missing required source/interface: {required_term}")

    readme = texts[ROOT / "README.md"]
    author_count = readme.splitlines().count(AUTHOR_LINE)
    if author_count != 1:
        errors.append(
            "README.md: expected exactly one author line: " + AUTHOR_LINE
        )

    closure = texts[ROOT / "CLOSURE_AUDIT.md"]
    if "待最终验证" in closure:
        errors.append("CLOSURE_AUDIT.md: unresolved final-verification status")

    chapter_one = texts[ROOT / "01_bytes_text_and_tokens.md"]
    if (
        r"\operatorname{AdmIn}_\Theta\to\operatorname{AdmTok}_\Theta"
        not in chapter_one
    ):
        errors.append("chapter 1: tokenizer encoder codomain is not AdmTok")

    chapter_five = texts[ROOT / "05_probabilistic_generation.md"]
    for required_term in (
        "(Y,\\mathcal Y)",
        "O:(\\Omega_n,\\mathcal F_n)\\to(Y,\\mathcal Y)",
        "B\\in\\mathcal Y",
        "实现映射",
        "固定完整初态",
    ):
        if required_term not in chapter_five:
            errors.append(f"chapter 5: missing measurable/randomness interface: {required_term}")

    chapter_eleven = texts[ROOT / "11_complete_decomposition.md"]
    for required_term in (
        r"\operatorname{Field}(X)",
        r"\mathsf{AbsenceReason}",
        r"\mathsf{OutputRec}_\Sigma",
    ):
        if required_term not in chapter_eleven:
            errors.append(f"chapter 11: missing typed record interface: {required_term}")

    if errors:
        print("\n".join(errors))
        return 1

    print(
        "ontology-of-an-output validation: "
        f"chapters={len(CHAPTERS)} exercises={len(seen)} "
        f"claims={sum(len(CLAIM.findall(texts[item])) for item in CHAPTERS)} "
        "errors=0"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
