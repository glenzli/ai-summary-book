#!/usr/bin/env python3
"""Validate the structure, numbering, links, and layered format of the book."""

from __future__ import annotations

import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MONTHS = sorted(path for path in ROOT.glob("[01][0-9]_*.md") if not path.name.startswith("00_"))
DAY_RE = re.compile(r"^## 第 (\d{3}) 天｜(.+)$", re.MULTILINE)
ENTRY_RE = re.compile(
    r"^## 第 (\d{3}) 天｜(.+?)\n\n"
    r"!\[[^\]]*\]\(images/days/day-\d{3}\.svg\)\n\n"
    r"(.+?)\n\n"
    r"\*\*再想一步：\*\* (.+?)\n\n"
    r"(?:!\[原理图：[^\]]+\]\(images/explainers/day-\d{3}\.svg\)\n\n"
    r"\*图解：[^\n]+\*\n\n)?"
    r"\*\*一起[^*]*：\*\* (.+?)(?=\n\n---|\Z)",
    re.MULTILINE | re.DOTALL,
)
EXPLAINER_RE = re.compile(
    r"!\[原理图：([^\]]+)\]\(images/explainers/day-(\d{3})\.svg\)\n\n"
    r"\*图解：([^\n]+)\*"
)
LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
EXPECTED_MONTH_COUNTS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
EXPECTED_EXPLAINERS_PER_MONTH = 12


def fail(errors: list[str], message: str) -> None:
    errors.append(message)


def main() -> int:
    errors: list[str] = []
    entries: list[tuple[int, str, Path]] = []
    sections: list[tuple[int, str, str, str, str, Path]] = []
    explainer_days: list[int] = []

    if len(MONTHS) != 12:
        fail(errors, f"expected 12 month files, found {len(MONTHS)}")

    for index, path in enumerate(MONTHS):
        text = path.read_text(encoding="utf-8")
        heading_matches = list(DAY_RE.finditer(text))
        headings = [(int(match.group(1)), match.group(2)) for match in heading_matches]
        entries.extend((day, title, path) for day, title in headings)
        parsed_sections = [
            (int(day), title, first, principle, activity)
            for day, title, first, principle, activity in ENTRY_RE.findall(text)
        ]
        sections.extend((*section, path) for section in parsed_sections)
        if index < len(EXPECTED_MONTH_COUNTS) and len(headings) != EXPECTED_MONTH_COUNTS[index]:
            fail(errors, f"{path.name}: expected {EXPECTED_MONTH_COUNTS[index]} days, found {len(headings)}")
        if [(day, title) for day, title, *_ in parsed_sections] != headings:
            fail(errors, f"{path.name}: one or more entries do not match the full layered page structure")
        principles = text.count("**再想一步：**")
        activities = len(re.findall(r"^\*\*一起[^*]*：\*\*", text, re.MULTILINE))
        if principles != len(headings):
            fail(errors, f"{path.name}: {len(headings)} entries but {principles} principle layers")
        if activities != len(headings):
            fail(errors, f"{path.name}: {len(headings)} entries but {activities} activities")

        explainer_matches = EXPLAINER_RE.findall(text)
        if len(explainer_matches) != EXPECTED_EXPLAINERS_PER_MONTH:
            fail(errors, f"{path.name}: expected {EXPECTED_EXPLAINERS_PER_MONTH} explainers, found {len(explainer_matches)}")
        heading_days = {day for day, _ in headings}
        for alt, day_text, caption in explainer_matches:
            day = int(day_text)
            explainer_days.append(day)
            if day not in heading_days:
                fail(errors, f"{path.name}: explainer day {day:03d} is outside the chapter's entries")
            if alt != caption:
                fail(errors, f"{path.name}: explainer alt text and caption differ on day {day:03d}")
        for position, match in enumerate(heading_matches):
            block_end = heading_matches[position + 1].start() if position + 1 < len(heading_matches) else len(text)
            block = text[match.end():block_end]
            block_explainers = EXPLAINER_RE.findall(block)
            if len(block_explainers) > 1:
                fail(errors, f"{path.name}: multiple explainers in day {int(match.group(1)):03d}")
            if block_explainers and int(block_explainers[0][1]) != int(match.group(1)):
                fail(errors, f"{path.name}: wrong explainer number inside day {int(match.group(1)):03d}")

    days = [day for day, _, _ in entries]
    if days != list(range(1, 366)):
        fail(errors, "day sequence is not exactly 001..365")
    titles = [title for _, title, _ in entries]
    duplicates = [title for title, count in Counter(titles).items() if count > 1]
    if duplicates:
        fail(errors, "duplicate titles: " + ", ".join(duplicates))

    first_layers = [re.sub(r"\s+", "", first) for _, _, first, _, _, _ in sections]
    principle_layers = [re.sub(r"\s+", "", principle) for _, _, _, principle, _, _ in sections]
    activities = [re.sub(r"\s+", "", activity) for _, _, _, _, activity, _ in sections]
    for day, _, first, principle, activity, path in sections:
        if len(re.sub(r"\s+", "", first)) < 40:
            fail(errors, f"{path.name}: first explanation is too short on day {day:03d}")
        if len(re.sub(r"\s+", "", principle)) < 40:
            fail(errors, f"{path.name}: principle layer is too short on day {day:03d}")
        if len(re.sub(r"\s+", "", activity)) < 10:
            fail(errors, f"{path.name}: activity is too short on day {day:03d}")
    if len(first_layers) != len(set(first_layers)):
        fail(errors, "duplicate first-layer explanations found")
    if len(principle_layers) != len(set(principle_layers)):
        fail(errors, "duplicate principle explanations found")
    if len(activities) != len(set(activities)):
        fail(errors, "duplicate activities found")

    if len(explainer_days) != 144:
        fail(errors, f"expected exactly 144 linked explainers, found {len(explainer_days)}")
    if len(explainer_days) != len(set(explainer_days)):
        fail(errors, "duplicate explainer day links found")

    daily_art = sorted((ROOT / "images" / "days").glob("day-*.svg"))
    if len(daily_art) != 365:
        fail(errors, f"expected exactly 365 daily SVG files, found {len(daily_art)}")
    for day, title, path in entries:
        expected = ROOT / "images" / "days" / f"day-{day:03d}.svg"
        if not expected.exists():
            fail(errors, f"missing daily art for {day:03d} {title}")
        marker = f"(images/days/day-{day:03d}.svg)"
        if marker not in path.read_text(encoding="utf-8"):
            fail(errors, f"{path.name}: wrong or missing daily image link for {day:03d}")
        if expected.exists():
            try:
                root = ET.parse(expected).getroot()
                if not root.tag.endswith("svg") or "viewBox" not in root.attrib:
                    fail(errors, f"invalid SVG root or viewBox in {expected.name}")
            except ET.ParseError as exc:
                fail(errors, f"unparseable SVG {expected.name}: {exc}")

    explainer_art = sorted((ROOT / "images" / "explainers").glob("day-*.svg"))
    if len(explainer_art) != 144:
        fail(errors, f"expected exactly 144 explainer SVG files, found {len(explainer_art)}")
    linked_explainer_names = {f"day-{day:03d}.svg" for day in explainer_days}
    actual_explainer_names = {path.name for path in explainer_art}
    if linked_explainer_names != actual_explainer_names:
        missing = sorted(linked_explainer_names - actual_explainer_names)
        orphaned = sorted(actual_explainer_names - linked_explainer_names)
        if missing:
            fail(errors, "missing explainer files: " + ", ".join(missing))
        if orphaned:
            fail(errors, "orphaned explainer files: " + ", ".join(orphaned))
    for expected in explainer_art:
        try:
            root = ET.parse(expected).getroot()
            if not root.tag.endswith("svg") or root.attrib.get("viewBox") != "0 0 1200 680":
                fail(errors, f"invalid SVG root or viewBox in explainer {expected.name}")
                continue
            if root.attrib.get("role") != "img" or root.attrib.get("aria-labelledby") != "title desc":
                fail(errors, f"missing explainer accessibility attributes in {expected.name}")
            children = {child.tag.rsplit("}", 1)[-1]: child for child in root}
            if "title" not in children or not "".join(children["title"].itertext()).startswith("原理图："):
                fail(errors, f"missing explainer title in {expected.name}")
            if "desc" not in children or not "".join(children["desc"].itertext()).strip():
                fail(errors, f"missing explainer description in {expected.name}")
            visible_text = "".join(root.itertext())
            if "示意图" not in visible_text or "不按比例" not in visible_text:
                fail(errors, f"missing not-to-scale notice in {expected.name}")
        except ET.ParseError as exc:
            fail(errors, f"unparseable explainer SVG {expected.name}: {exc}")

    for path in [
        ROOT / "README.md",
        ROOT / "00_for_grownups.md",
        ROOT / "STYLE_GUIDE.md",
        ROOT / "SOURCES.md",
        ROOT / "IMAGE_QA.md",
        *MONTHS,
    ]:
        text = path.read_text(encoding="utf-8")
        for target in LINK_RE.findall(text):
            if target.startswith(("http://", "https://", "#")):
                continue
            target_path = (path.parent / target).resolve()
            if not target_path.exists():
                fail(errors, f"{path.name}: broken local link {target}")

    for required in [ROOT / "images" / "cover.png", *[ROOT / "images" / "months" / f"{month:02d}-{name}.png" for month, name in enumerate([
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december",
    ], start=1)]]:
        if not required.exists():
            fail(errors, f"missing generated illustration {required.relative_to(ROOT)}")

    if errors:
        print("VALIDATION FAILED")
        for error in errors:
            print(f"- {error}")
        return 1

    print("VALIDATION OK")
    print(f"months={len(MONTHS)} days={len(entries)} principles={len(entries)} activities={len(entries)}")
    print("daily_art=365 explainer_art=144 cover=1 month_plates=12 local_links=OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
