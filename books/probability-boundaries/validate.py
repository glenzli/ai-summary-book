#!/usr/bin/env python3
"""Structural checks for the Probability Boundaries textbook."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CHAPTERS = [ROOT / "00_preface_and_scope.md"] + [
    ROOT / f"{number:02d}_{name}.md"
    for number, name in [
        (1, "probability_spaces"),
        (2, "random_variables_and_integration"),
        (3, "independence_and_products"),
        (4, "conditioning_and_information"),
        (5, "convergence_and_limit_theorems"),
        (6, "entropy_divergence_and_scoring"),
        (7, "calibration_and_decision"),
        (8, "randomized_algorithms"),
        (9, "observation_intervention_and_causality"),
        (10, "language_model_probability"),
        (11, "boundaries"),
    ]
]
REQUIRED = CHAPTERS + [
    ROOT / name
    for name in [
        "README.md",
        "SKILL.md",
        "NOTATION.md",
        "SOURCES.md",
        "THEOREM_LEDGER.md",
        "SOLUTIONS.md",
        "CLOSURE_AUDIT.md",
    ]
]
EXERCISE = re.compile(r"\*\*练习\s+(\d+\.\d+)\.")
STATEMENT = re.compile(
    r"\*\*(?:定义|定理|命题|引理|推论|外部输入)\s+(\d+\.\d+)"
)
LOCAL_LINK = re.compile(r"\[[^\]]+\]\((?!https?://)([^)#]+\.md)(?:#[^)]+)?\)")
FORBIDDEN_TEMPLATE_HEADING = re.compile(
    r"^#{2,6}\s+(?:\d+(?:\.\d+)*\s+)?"
    r"(?:本章目标|依赖|主线|本章小结)\s*$",
    re.MULTILINE,
)
SECTION_HEADING = re.compile(r"^##\s+.+$", re.MULTILINE)
MALFORMED_LATEX = re.compile(
    r"(?<!\\)\b(?:left|right|bigl|bigr|Bigl|Bigr|qquad|quad|"
    r"frac|tfrac|dfrac|sqrt|sum|prod|lim|sup|inf|"
    r"leq|geq|neq|approx|mapsto|notin|subseteq|supseteq|cup|cap|"
    r"mid|vert|Vert|langle|rangle|infty|partial|nabla|"
    r"ldots|cdots|dots|text|mathrm|mathbb|mathbf|mathcal|mathsf|"
    r"operatorname|overline|underline|widehat|widetilde|begin|end)\b"
)
NARRATIVE_ANCHORS = {
    "00_preface_and_scope.md": "F(S)",
    "01_probability_spaces.md": "只记录骰子的奇偶",
    "02_random_variables_and_integration.md": "一个三点分布的完整计算",
    "03_independence_and_products.md": "| 0 | 0 | 0 | $1/4$ |",
    "04_conditioning_and_information.md": "有限划分上的条件平均",
    "05_convergence_and_limit_theorems.md": "尖峰序列把这条边界算得很清楚",
    "06_entropy_divergence_and_scoring.md": "章首二元例子的这一分解",
    "07_calibration_and_decision.md": "分解一个失准但有分辨率的预测器",
    "08_randomized_algorithms.md": "直到首次正面的随机带表示",
    "09_observation_intervention_and_causality.md": "观察条件可不同于干预分布",
    "10_language_model_probability.md": r"\left(\frac17,\frac27,\frac47\right)",
    "11_boundaries.md": "F(0)=F(1)=F(2)",
}


def main() -> int:
    errors: list[str] = []
    for path in REQUIRED:
        if not path.is_file():
            errors.append(f"missing required file: {path.name}")

    if errors:
        print("\n".join(errors))
        return 1

    author_line = "**作者：Dr. Stochastic Parrot**"
    readme_lines = (ROOT / "README.md").read_text(encoding="utf-8").splitlines()
    if readme_lines.count(author_line) != 1:
        errors.append("README.md: exact author attribution must appear once")

    solutions = (ROOT / "SOLUTIONS.md").read_text(encoding="utf-8")
    seen: set[str] = set()
    statements: set[str] = set()
    narrative_examples = 0
    for chapter in CHAPTERS:
        text = chapter.read_text(encoding="utf-8")
        if "## 练习" not in text:
            errors.append(f"{chapter.name}: missing heading ## 练习")

        old_headings = FORBIDDEN_TEMPLATE_HEADING.findall(text)
        if old_headings:
            errors.append(
                f"{chapter.name}: legacy template heading(s): "
                + ", ".join(old_headings)
            )

        first_section = SECTION_HEADING.search(text)
        if first_section is None:
            errors.append(f"{chapter.name}: no level-2 section heading")
        else:
            first_line_end = text.find("\n")
            intro = text[first_line_end + 1 : first_section.start()].strip()
            paragraphs = [p for p in re.split(r"\n\s*\n", intro) if p.strip()]
            prose = re.sub(r"[`*_>#\[\]()$\\\s]", "", intro)
            if len(paragraphs) < 2 or len(prose) < 100:
                errors.append(
                    f"{chapter.name}: chapter opening must contain at least "
                    "two substantive narrative paragraphs"
                )

        anchor = NARRATIVE_ANCHORS[chapter.name]
        if anchor not in text:
            errors.append(
                f"{chapter.name}: missing retained worked-example anchor {anchor!r}"
            )
        else:
            narrative_examples += 1

        for number in EXERCISE.findall(text):
            if number in seen:
                errors.append(f"duplicate exercise number: {number}")
            seen.add(number)
            if f"**练习 {number}.**" not in solutions:
                errors.append(f"missing solution: {number}")

        for number in STATEMENT.findall(text):
            if number in statements:
                errors.append(f"duplicate statement number: {number}")
            statements.add(number)

    for path in REQUIRED:
        if path.suffix != ".md":
            continue
        text = path.read_text(encoding="utf-8")
        bad_controls = sorted(
            {f"U+{ord(char):04X}" for char in text if ord(char) < 32 and char not in "\n\r\t"}
        )
        if bad_controls:
            errors.append(
                f"{path.name}: control character " + ", ".join(bad_controls)
            )
        for line_number, line in enumerate(text.splitlines(), start=1):
            match = MALFORMED_LATEX.search(line)
            if match:
                errors.append(
                    f"{path.name}:{line_number}: possible missing LaTeX backslash "
                    f"before {match.group(0)!r}"
                )

        if text.count("$$") % 2:
            errors.append(f"{path.name}: unmatched $$ math fence")
        if text.count("```") % 2:
            errors.append(f"{path.name}: unmatched fenced code block")

        for target in LOCAL_LINK.findall(text):
            linked = (path.parent / target).resolve()
            if not linked.is_file():
                errors.append(f"{path.name}: broken local link: {target}")

    solution_numbers = set(EXERCISE.findall(solutions))
    extras = sorted(solution_numbers - seen)
    if extras:
        errors.append("solutions without exercises: " + ", ".join(extras))

    if errors:
        print("\n".join(errors))
        return 1

    print(
        f"probability-boundaries validation: chapters={len(CHAPTERS)} "
        f"worked_examples={narrative_examples} exercises={len(seen)} errors=0"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
