#!/usr/bin/env python3
"""Book-local structural and regression checks for the HoTT textbook."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


BOOK = Path(__file__).resolve().parent
REPO = BOOK.parent.parent
FORBIDDEN_HEADING = re.compile(
    r"^#{2,6}\s+(?:\d+(?:\.\d+)*\s+)?"
    r"(?:本章目标|目标|依赖(?:前置知识)?|主线|本章小结)\s*$",
    re.MULTILINE,
)
MARKDOWN_NOISE = re.compile(r"[`*_>#\[\]()]|!\[[^\]]*\]\([^)]*\)")
CONTROL_CHAR = re.compile(r"[\x00-\x09\x0b\x0c\x0e-\x1f\x7f]")


def chapter_intro(text: str) -> str:
    lines = text.splitlines()
    h1 = next((i for i, line in enumerate(lines) if line.startswith("# ")), None)
    if h1 is None:
        return ""
    result: list[str] = []
    in_fence = False
    for line in lines[h1 + 1 :]:
        if line.startswith("```"):
            in_fence = not in_fence
            continue
        if not in_fence and line.startswith("## "):
            break
        if not in_fence:
            result.append(line)
    return "\n".join(result).strip()


def prose_length(text: str) -> int:
    cleaned = MARKDOWN_NOISE.sub("", text)
    return len(re.sub(r"\s+", "", cleaned))


def check_numbered_chapters(errors: list[str]) -> None:
    chapters = sorted(BOOK.glob("[0-9][0-9]_*.md"))
    prefixes = [path.name[:2] for path in chapters]
    expected = [f"{number:02d}" for number in range(18)]
    if prefixes != expected:
        errors.append(f"numbered chapters are {prefixes}, expected {expected}")

    for path in chapters:
        text = path.read_text(encoding="utf-8")
        if len(re.findall(r"^#\s+", text, re.MULTILINE)) != 1:
            errors.append(f"{path.name}: expected exactly one H1")
        intro_length = prose_length(chapter_intro(text))
        if intro_length < 120:
            errors.append(
                f"{path.name}: opening has {intro_length} prose characters; need at least 120"
            )
        for match in FORBIDDEN_HEADING.finditer(text):
            errors.append(f"{path.name}: legacy template heading {match.group(0).strip()}")
        if not re.search(r"^##\s+(?:\d+(?:\.\d+)*\s+)?练习\s*$", text, re.MULTILINE):
            errors.append(f"{path.name}: missing exercise section")


def check_regressions(errors: list[str]) -> None:
    for path in BOOK.glob("*"):
        if path.is_file() and path.suffix in {".md", ".py"}:
            for line_number, line in enumerate(
                path.read_text(encoding="utf-8").split("\n"), start=1
            ):
                if CONTROL_CHAR.search(line):
                    errors.append(f"{path.name}:{line_number}: control character in source")

    skill = (BOOK / "SKILL.md").read_text(encoding="utf-8")
    stale_skill_rules = (
        "每章必须先列出“本章目标”和“依赖前置知识”",
        "每章末尾必须有“本章小结”",
    )
    for phrase in stale_skill_rules:
        if phrase in skill:
            errors.append(f"SKILL.md: stale template requirement remains: {phrase}")

    bo = (BOOK / "BO_constructive_metric_spaces_series_and_integration.md").read_text(
        encoding="utf-8"
    )
    for required in (
        r"\mathsf B_q(x,y)",
        r"\|X\|",
        r"\mathsf{Limit}(a)",
        r"\mathsf{isContr}(\mathsf{Fix}(T))",
        "极限的 mere existence 可消去",
    ):
        if required not in bo:
            errors.append(f"BO appendix: missing Banach/completeness guard {required!r}")
    for forbidden in ("d(Tx,Ty)\\le", "取任意 $x_0:X$", "若 $X$ 完备且 $T$ 是 contraction"):
        if forbidden in bo:
            errors.append(f"BO appendix: stale metric/nonempty formulation {forbidden!r}")

    rezk_checks = {
        "14_yoneda_limits_adjunctions_rezk.md": (
            "外部输入定理 14.11",
            "Theorem 8.4",
        ),
        "AA_rezk_universal_property_schema.md": (
            "外部输入定理 AA.8",
            "Theorem 8.4",
            "未重证边界",
        ),
        "R_rezk_completion_input.md": (
            "外部输入定理 R.11",
            "Theorem 8.4",
        ),
    }
    for filename, required_phrases in rezk_checks.items():
        text = (BOOK / filename).read_text(encoding="utf-8")
        for phrase in required_phrases:
            if phrase not in text:
                errors.append(f"{filename}: missing precise Rezk boundary {phrase!r}")
        if "书内归约" in text:
            errors.append(f"{filename}: Rezk universal property still claims book-internal reduction")


def run_command(command: list[str], errors: list[str]) -> None:
    print(f"+ {' '.join(command)}")
    result = subprocess.run(command, cwd=REPO, text=True)
    if result.returncode:
        errors.append(f"command failed ({result.returncode}): {' '.join(command)}")


def main() -> int:
    errors: list[str] = []
    check_numbered_chapters(errors)
    check_regressions(errors)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        print(f"hott-local-validation: errors={len(errors)}")
        return 1

    run_command(
        [sys.executable, "books/audit_textbook_narrative.py", "homotopy-type-theory", "--strict"],
        errors,
    )
    run_command(
        [sys.executable, "books/audit_oet_rigor.py", "homotopy-type-theory", "--strict"],
        errors,
    )
    run_command(
        ["git", "diff", "--check", "--", "books/homotopy-type-theory"],
        errors,
    )

    for error in errors:
        print(f"ERROR: {error}")
    print(f"hott-local-validation: errors={len(errors)}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
