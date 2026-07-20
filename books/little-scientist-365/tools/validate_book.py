#!/usr/bin/env python3
"""Validate the content-led structure, question articles, links, and visuals."""

from __future__ import annotations

import re
import struct
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from book_structure import (
    CHAPTERS,
    CHAPTER_BY_QUESTION,
    OBSERVATION_IMAGES,
    PARTS,
    PART_BY_NUMBER,
    REALISTIC_MAIN_IMAGES,
    REALISTIC_PRINCIPLE_IMAGES,
)
from earth_weather_illustrations import EARTH_WEATHER_KEYS, EARTH_WEATHER_QUESTIONS
from invertebrate_illustrations import INVERTEBRATE_KEYS, INVERTEBRATE_QUESTIONS
from machine_illustrations import MACHINE_KEYS, MACHINE_QUESTIONS
from plant_illustrations import PLANT_KEYS, PLANT_QUESTIONS
from question_scene_specs import SUBJECT_ICONS
from textbook_illustrations import TEXTBOOK_KEYS, TEXTBOOK_QUESTIONS
from vertebrate_illustrations import VERTEBRATE_KEYS, VERTEBRATE_QUESTIONS


ROOT = Path(__file__).resolve().parents[1]
CHAPTER_PATHS = [ROOT / chapter.filename for chapter in CHAPTERS]
QUESTION_RE = re.compile(r"^### 第 (\d{3}) 问｜(.+)$", re.MULTILINE)
ENTRY_RE = re.compile(
    r"^### 第 (\d{3}) 问｜(.+?)\n\n"
    r"!\[[^\]]*\]\((images/(?:questions/question-\d{3}\.svg|questions-real/question-\d{3}\.png|observations/question-\d{3}[-a-z0-9]*\.png))\)\n\n"
    r"(?:\*(?:写实观察图|科学写实图)：[^\n]+\*\n\n)?"
    r"(.+?)\n\n"
    r"\*\*再想一步：\*\* (.+?)\n\n"
    r"\*\*把知识连起来：\*\* (.+?)\n\n"
    r"\*\*再往外看：\*\* (.+?)\n\n"
    r"\*\*容易弄混：\*\* (.+?)\n\n"
    r"(?:!\[写实原理图：[^\]]+\]\(images/principles-realistic/question-\d{3}[-a-z0-9]*\.png\)\n\n"
    r"\*(?:写实原理图|科学写实原理图)：[^\n]+\*\n\n)?"
    r"(?:!\[原理图：[^\]]+\]\(images/explainers/question-\d{3}\.svg\)\n\n"
    r"\*图解：[^\n]+\*\n\n)?"
    r"\*\*一起[^*]*：\*\* (.+?)(?=\n\n(?:---|## )|\Z)",
    re.MULTILINE | re.DOTALL,
)
EXPLAINER_RE = re.compile(
    r"!\[原理图：([^\]]+)\]\(images/explainers/question-(\d{3})\.svg\)\n\n"
    r"\*图解：([^\n]+)\*"
)
REALISTIC_RE = re.compile(
    r"!\[写实原理图：([^\]]+)\]"
    r"\(images/principles-realistic/(question-(\d{3})[-a-z0-9]*\.png)\)\n\n"
    r"\*((?:写实原理图|科学写实原理图)：[^\n]+)\*"
)
MAIN_IMAGE_RE = re.compile(
    r"!\[[^\]]*\]\((images/(?:questions/question-\d{3}\.svg|questions-real/question-\d{3}\.png|observations/question-\d{3}[-a-z0-9]*\.png))\)"
)
LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
EXPECTED_EXPLAINERS = 154


def fail(errors: list[str], message: str) -> None:
    errors.append(message)


def png_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        signature = handle.read(24)
    if len(signature) != 24 or signature[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("invalid PNG signature")
    return struct.unpack(">II", signature[16:24])


def check_png_set(
    errors: list[str],
    directory: Path,
    expected_names: set[str],
    label: str,
) -> None:
    actual_names = {path.name for path in directory.glob("*.png")}
    missing = sorted(expected_names - actual_names)
    orphaned = sorted(actual_names - expected_names)
    if missing:
        fail(errors, f"missing {label} images: " + ", ".join(missing))
    if orphaned:
        fail(errors, f"orphaned {label} images: " + ", ".join(orphaned))
    for name in sorted(expected_names & actual_names):
        path = directory / name
        try:
            width, height = png_dimensions(path)
            if width < 1200 or height < 675:
                fail(errors, f"{label} image is too small: {name} ({width}x{height})")
            ratio = width / height
            if not 1.7 <= ratio <= 1.9:
                fail(errors, f"{label} image has unexpected aspect ratio: {name} ({ratio:.3f})")
        except ValueError as exc:
            fail(errors, f"invalid {label} image {name}: {exc}")


def main() -> int:
    errors: list[str] = []
    entries: list[tuple[int, str, Path]] = []
    sections: list[tuple[int, str, str, str, str, str, str, str, Path]] = []
    explainer_questions: list[int] = []
    realistic_questions: list[int] = []

    expected_chapter_names = {chapter.filename for chapter in CHAPTERS}
    actual_chapter_names = {
        path.name
        for path in ROOT.glob("[0-9][0-9]_*.md")
        if path.name != "00_for_grownups.md"
    }
    if actual_chapter_names != expected_chapter_names:
        missing = sorted(expected_chapter_names - actual_chapter_names)
        unexpected = sorted(actual_chapter_names - expected_chapter_names)
        if missing:
            fail(errors, "missing interest-unit files: " + ", ".join(missing))
        if unexpected:
            fail(errors, "unexpected interest-unit files: " + ", ".join(unexpected))

    first_unit_by_part: dict[int, int] = {}
    for chapter in CHAPTERS:
        first_unit_by_part.setdefault(chapter.part, chapter.number)

    for chapter, path in zip(CHAPTERS, CHAPTER_PATHS):
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        expected_heading = f"# 兴趣单元 {chapter.number:02d}｜{chapter.title}"
        if not text.startswith(expected_heading + "\n"):
            fail(errors, f"{path.name}: missing canonical interest-unit heading")
        part = PART_BY_NUMBER[chapter.part]
        if f"> **探索领域 {part.number:02d}：** {part.title}" not in text:
            fail(errors, f"{path.name}: missing canonical exploration-part label")
        opener_count = text.count(f"]({part.opener})")
        expected_opener_count = 1 if first_unit_by_part[part.number] == chapter.number else 0
        if opener_count != expected_opener_count:
            fail(errors, f"{path.name}: part opener count is {opener_count}, expected {expected_opener_count}")
        if text.count("> **核心问题：**") != 1:
            fail(errors, f"{path.name}: expected exactly one core question")
        if len(re.findall(r"^## 单元综合观察｜", text, re.MULTILINE)) != 1:
            fail(errors, f"{path.name}: expected exactly one synthesis observation")

        expected_routes = [
            (f"{chapter.number:02d}.{index}", route.title)
            for index, route in enumerate(chapter.routes, start=1)
        ]
        actual_routes = re.findall(r"^## 追问链 (\d{2}\.\d+)｜(.+)$", text, re.MULTILINE)
        if actual_routes != expected_routes:
            fail(errors, f"{path.name}: question trails differ from the canonical structure")

        heading_matches = list(QUESTION_RE.finditer(text))
        headings = [(int(match.group(1)), match.group(2)) for match in heading_matches]
        expected_questions = [
            question
            for route in chapter.routes
            for question in range(route.start, route.end + 1)
        ]
        if [question for question, _ in headings] != expected_questions:
            fail(errors, f"{path.name}: question range/order differs from its question trails")
        entries.extend((question, title, path) for question, title in headings)

        parsed_sections = [
            (int(question), title, first, principle, connection, outward, confusion, activity)
            for question, title, _, first, principle, connection, outward, confusion, activity
            in ENTRY_RE.findall(text)
        ]
        sections.extend((*section, path) for section in parsed_sections)
        if [(question, title) for question, title, *_ in parsed_sections] != headings:
            fail(errors, f"{path.name}: one or more questions do not match the full layered article structure")

        principles = text.count("**再想一步：**")
        activities = len(re.findall(r"^\*\*一起[^*]*：\*\*", text, re.MULTILINE))
        if principles != len(headings):
            fail(errors, f"{path.name}: {len(headings)} questions but {principles} principle layers")
        if activities != len(headings):
            fail(errors, f"{path.name}: {len(headings)} questions but {activities} activities")
        article_layer_counts = {
            label: text.count(f"**{label}：**")
            for label in ("把知识连起来", "再往外看", "容易弄混")
        }
        if set(article_layer_counts.values()) != {len(headings)}:
            details = ", ".join(f"{label}={count}" for label, count in article_layer_counts.items())
            fail(errors, f"{path.name}: incomplete article-layer coverage ({details})")

        heading_questions = {question for question, _ in headings}
        for alt, question_text, caption in EXPLAINER_RE.findall(text):
            question = int(question_text)
            explainer_questions.append(question)
            if question not in heading_questions:
                fail(errors, f"{path.name}: explainer question {question:03d} is outside the unit")
            if alt != caption:
                fail(errors, f"{path.name}: explainer alt text and caption differ on question {question:03d}")

        for _, filename, question_text, caption in REALISTIC_RE.findall(text):
            question = int(question_text)
            realistic_questions.append(question)
            policy = REALISTIC_PRINCIPLE_IMAGES.get(question)
            if question not in heading_questions:
                fail(errors, f"{path.name}: realistic principle question {question:03d} is outside the unit")
            if policy is None:
                fail(errors, f"{path.name}: unexpected realistic principle image on question {question:03d}")
            elif filename != policy.filename or caption != policy.caption:
                fail(errors, f"{path.name}: noncanonical realistic principle block on question {question:03d}")

        for position, match in enumerate(heading_matches):
            block_end = heading_matches[position + 1].start() if position + 1 < len(heading_matches) else len(text)
            block = text[match.end():block_end]
            question = int(match.group(1))
            main_image = MAIN_IMAGE_RE.search(block)
            observation = OBSERVATION_IMAGES.get(question)
            expected_main = (
                f"images/questions-real/{REALISTIC_MAIN_IMAGES[question]}"
                if question in REALISTIC_MAIN_IMAGES
                else f"images/observations/{observation.filename}"
                if observation
                else f"images/questions/question-{question:03d}.svg"
            )
            if main_image is None or main_image.group(1) != expected_main:
                found = main_image.group(1) if main_image else "none"
                fail(errors, f"{path.name}: question {question:03d} main image is {found}, expected {expected_main}")
            if observation and f"*{observation.caption}*" not in block:
                fail(errors, f"{path.name}: missing canonical observation caption on question {question:03d}")
            principle_image = REALISTIC_PRINCIPLE_IMAGES.get(question)
            principle_matches = REALISTIC_RE.findall(block)
            if principle_image is None and principle_matches:
                fail(errors, f"{path.name}: unexpected realistic principle block on question {question:03d}")
            if principle_image is not None:
                expected_path = f"images/principles-realistic/{principle_image.filename}"
                if len(principle_matches) != 1 or expected_path not in block or f"*{principle_image.caption}*" not in block:
                    fail(errors, f"{path.name}: missing canonical realistic principle block on question {question:03d}")
            block_explainers = EXPLAINER_RE.findall(block)
            if len(block_explainers) > 1:
                fail(errors, f"{path.name}: multiple explainers in question {question:03d}")
            if block_explainers and int(block_explainers[0][1]) != question:
                fail(errors, f"{path.name}: wrong explainer number inside question {question:03d}")

    expected_question_sequence = list(CHAPTER_BY_QUESTION)
    questions = [question for question, _, _ in entries]
    if questions != expected_question_sequence:
        fail(errors, "question sequence differs from the canonical extensible index")
    titles = [title for _, title, _ in entries]
    duplicates = [title for title, count in Counter(titles).items() if count > 1]
    if duplicates:
        fail(errors, "duplicate titles: " + ", ".join(duplicates))

    first_layers = [re.sub(r"\s+", "", first) for _, _, first, _, _, _, _, _, _ in sections]
    principle_layers = [re.sub(r"\s+", "", principle) for _, _, _, principle, _, _, _, _, _ in sections]
    connection_layers = [re.sub(r"\s+", "", layer) for _, _, _, _, layer, _, _, _, _ in sections]
    outward_layers = [re.sub(r"\s+", "", layer) for _, _, _, _, _, layer, _, _, _ in sections]
    confusion_layers = [re.sub(r"\s+", "", layer) for _, _, _, _, _, _, layer, _, _ in sections]
    activities = [re.sub(r"\s+", "", activity) for _, _, _, _, _, _, _, activity, _ in sections]
    for question, _, first, principle, connection, outward, confusion, activity, path in sections:
        if len(re.sub(r"\s+", "", first)) < 40:
            fail(errors, f"{path.name}: first explanation is too short on question {question:03d}")
        if len(re.sub(r"\s+", "", principle)) < 40:
            fail(errors, f"{path.name}: principle layer is too short on question {question:03d}")
        if len(re.sub(r"\s+", "", connection)) < 70:
            fail(errors, f"{path.name}: system-connection layer is too short on question {question:03d}")
        if len(re.sub(r"\s+", "", outward)) < 28:
            fail(errors, f"{path.name}: outward-extension layer is too short on question {question:03d}")
        if len(re.sub(r"\s+", "", confusion)) < 35:
            fail(errors, f"{path.name}: misconception layer is too short on question {question:03d}")
        if len(re.sub(r"\s+", "", activity)) < 10:
            fail(errors, f"{path.name}: activity is too short on question {question:03d}")
        if not 3 <= len(re.findall(r"[。！？]", first)) <= 5:
            fail(errors, f"{path.name}: first explanation needs 3--5 sentences on question {question:03d}")
        if not 2 <= len(re.findall(r"[。！？]", principle)) <= 4:
            fail(errors, f"{path.name}: principle layer needs 2--4 sentences on question {question:03d}")
        if not 2 <= len(re.findall(r"[。！？]", connection)) <= 4:
            fail(errors, f"{path.name}: system-connection layer needs 2--4 sentences on question {question:03d}")
        if not 1 <= len(re.findall(r"[。！？]", outward)) <= 3:
            fail(errors, f"{path.name}: outward-extension layer needs 1--3 sentences on question {question:03d}")
        if not 1 <= len(re.findall(r"[。！？]", confusion)) <= 3:
            fail(errors, f"{path.name}: misconception layer needs 1--3 sentences on question {question:03d}")
    if len(first_layers) != len(set(first_layers)):
        fail(errors, "duplicate first-layer explanations found")
    if len(principle_layers) != len(set(principle_layers)):
        fail(errors, "duplicate principle explanations found")
    if len(connection_layers) != len(set(connection_layers)):
        fail(errors, "duplicate system-connection layers found")
    if len(outward_layers) != len(set(outward_layers)):
        fail(errors, "duplicate outward-extension layers found")
    if len(confusion_layers) != len(set(confusion_layers)):
        fail(errors, "duplicate misconception layers found")
    if len(activities) != len(set(activities)):
        fail(errors, "duplicate activities found")

    if len(explainer_questions) != EXPECTED_EXPLAINERS:
        fail(errors, f"expected {EXPECTED_EXPLAINERS} linked explainers, found {len(explainer_questions)}")
    if len(explainer_questions) != len(set(explainer_questions)):
        fail(errors, "duplicate explainer question links found")
    body_explainers = set(explainer_questions) & set(TEXTBOOK_QUESTIONS)
    if body_explainers != {153}:
        fail(errors, f"body explainers should contain only the fingerprint structure plate: {sorted(body_explainers)}")
    invertebrate_explainers = set(explainer_questions) & set(INVERTEBRATE_QUESTIONS)
    if invertebrate_explainers:
        fail(errors, f"invertebrate articles duplicate their annotated main plates: {sorted(invertebrate_explainers)}")
    expected_realistic_questions = sorted(REALISTIC_PRINCIPLE_IMAGES)
    if sorted(realistic_questions) != expected_realistic_questions:
        fail(errors, "realistic principle links differ from the canonical selection")

    question_art = sorted((ROOT / "images" / "questions").glob("question-*.svg"))
    if len(question_art) != len(expected_question_sequence):
        fail(errors, f"expected {len(expected_question_sequence)} question SVG files, found {len(question_art)}")
    question_scene_signatures: list[tuple[str, str]] = []
    for question, title, _ in entries:
        expected = ROOT / "images" / "questions" / f"question-{question:03d}.svg"
        if not expected.exists():
            fail(errors, f"missing question art for question {question:03d} {title}")
            continue
        try:
            root = ET.parse(expected).getroot()
            if not root.tag.endswith("svg") or "viewBox" not in root.attrib:
                fail(errors, f"invalid SVG root or viewBox in {expected.name}")
                continue
            if root.attrib.get("role") != "img" or root.attrib.get("aria-labelledby") != "title desc":
                fail(errors, f"missing question-art accessibility attributes in {expected.name}")
            children = {child.tag.rsplit("}", 1)[-1]: child for child in root}
            if "title" not in children or "".join(children["title"].itertext()).strip() != title:
                fail(errors, f"question-art title does not match question {question:03d}")
            if "desc" not in children or not "".join(children["desc"].itertext()).strip():
                fail(errors, f"missing question-art description in {expected.name}")
            scene_groups = [node for node in root.iter() if "data-question" in node.attrib]
            if len(scene_groups) != 1:
                fail(errors, f"expected one explicit scene group in {expected.name}, found {len(scene_groups)}")
                continue
            scene = scene_groups[0]
            if scene.attrib.get("data-question") != f"{question:03d}":
                fail(errors, f"question number mismatch in {expected.name}")
            layout = scene.attrib.get("data-layout", "")
            presentation = scene.attrib.get("data-presentation", "")
            source = scene.attrib.get("data-source", "")
            icons = scene.attrib.get("data-icons", "")
            icon_keys = tuple(key for key in icons.split(",") if key)
            if question in EARTH_WEATHER_QUESTIONS:
                if layout != "earth-weather" or presentation != "earth-weather" or source != "earth-weather":
                    fail(errors, f"earth/weather question lacks its annotated plate in {expected.name}")
                if icon_keys != EARTH_WEATHER_KEYS[question]:
                    fail(errors, f"earth/weather question has wrong plate key in {expected.name}: {icons}")
            elif question in PLANT_QUESTIONS:
                if layout != "plant" or presentation != "plant" or source != "plant":
                    fail(errors, f"plant question lacks its annotated plate in {expected.name}")
                if icon_keys != PLANT_KEYS[question]:
                    fail(errors, f"plant question has wrong plate key in {expected.name}: {icons}")
            elif question in TEXTBOOK_QUESTIONS:
                if layout != "textbook" or presentation != "textbook" or source != "textbook":
                    fail(errors, f"body question lacks textbook presentation in {expected.name}")
                if icon_keys != TEXTBOOK_KEYS[question]:
                    fail(errors, f"body question has wrong anatomical structures in {expected.name}: {icons}")
            elif question in INVERTEBRATE_QUESTIONS:
                if layout != "invertebrate" or presentation != "invertebrate" or source != "invertebrate":
                    fail(errors, f"invertebrate question lacks annotated biological presentation in {expected.name}")
                if icon_keys != INVERTEBRATE_KEYS[question]:
                    fail(errors, f"invertebrate question has wrong biological structures in {expected.name}: {icons}")
            elif question in VERTEBRATE_QUESTIONS:
                if layout != "vertebrate" or presentation != "vertebrate" or source != "vertebrate":
                    fail(errors, f"vertebrate question lacks its annotated plate in {expected.name}")
                if icon_keys != VERTEBRATE_KEYS[question]:
                    fail(errors, f"vertebrate question has wrong plate key in {expected.name}: {icons}")
            elif question in MACHINE_QUESTIONS:
                if layout != "machine" or presentation != "machine" or source != "machine":
                    fail(errors, f"machine question lacks its annotated plate in {expected.name}")
                if icon_keys != MACHINE_KEYS[question]:
                    fail(errors, f"machine question has wrong plate key in {expected.name}: {icons}")
            elif question in SUBJECT_ICONS:
                if layout != "concrete" or presentation != "concrete" or source != "subject":
                    fail(errors, f"observable subject lacks direct presentation in {expected.name}")
                if icon_keys != (SUBJECT_ICONS[question],):
                    fail(errors, f"observable subject has wrong main object in {expected.name}: {icons}")
            else:
                if layout != "concrete" or presentation != "concrete" or source not in {"explainer", "manual"}:
                    fail(errors, f"question lacks a concrete generated scene in {expected.name}")
                if not icon_keys:
                    fail(errors, f"question-art scene has no semantic objects in {expected.name}")
            question_scene_signatures.append((layout, icons))
        except ET.ParseError as exc:
            fail(errors, f"unparseable SVG {expected.name}: {exc}")

    unique_question_scenes = len(set(question_scene_signatures))
    minimum_unique = max(1, int(len(expected_question_sequence) * 0.9))
    if unique_question_scenes < minimum_unique:
        fail(errors, f"question-art semantic variety is too low: {unique_question_scenes}/{len(expected_question_sequence)}")
    repeated_question_scenes = Counter(question_scene_signatures)
    if repeated_question_scenes and max(repeated_question_scenes.values()) > 2:
        fail(errors, "a question-art semantic scene is reused more than twice")

    check_png_set(
        errors,
        ROOT / "images" / "questions-real",
        set(REALISTIC_MAIN_IMAGES.values()),
        "realistic main",
    )
    check_png_set(
        errors,
        ROOT / "images" / "observations",
        {image.filename for image in OBSERVATION_IMAGES.values()},
        "observation",
    )
    check_png_set(
        errors,
        ROOT / "images" / "principles-realistic",
        {image.filename for image in REALISTIC_PRINCIPLE_IMAGES.values()},
        "realistic principle",
    )

    explainer_art = sorted((ROOT / "images" / "explainers").glob("question-*.svg"))
    if len(explainer_art) != EXPECTED_EXPLAINERS:
        fail(errors, f"expected {EXPECTED_EXPLAINERS} explainer SVG files, found {len(explainer_art)}")
    linked_explainer_names = {f"question-{question:03d}.svg" for question in explainer_questions}
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

    linked_files = [
        ROOT / "README.md",
        ROOT / "00_for_grownups.md",
        ROOT / "STYLE_GUIDE.md",
        ROOT / "SOURCES.md",
        ROOT / "IMAGE_QA.md",
        *CHAPTER_PATHS,
    ]
    for path in linked_files:
        text = path.read_text(encoding="utf-8")
        for target in LINK_RE.findall(text):
            if target.startswith(("http://", "https://", "#")):
                continue
            target_path = (path.parent / target).resolve()
            if not target_path.exists():
                fail(errors, f"{path.name}: broken local link {target}")

    required_generated = [
        ROOT / "images" / "cover-modern.png",
        *[ROOT / part.opener for part in PARTS],
    ]
    for required in required_generated:
        if not required.exists():
            fail(errors, f"missing generated illustration {required.relative_to(ROOT)}")

    if errors:
        print("VALIDATION FAILED")
        for error in errors:
            print(f"- {error}")
        return 1

    print("VALIDATION OK")
    print(
        f"parts={len(PARTS)} units={len(CHAPTERS)} "
        f"questions={len(entries)} trails={sum(len(chapter.routes) for chapter in CHAPTERS)}"
    )
    print(
        f"question_art={len(question_art)} observation_art={len(OBSERVATION_IMAGES)} "
        f"realistic_main_art={len(REALISTIC_MAIN_IMAGES)} "
        f"realistic_principle_art={len(REALISTIC_PRINCIPLE_IMAGES)} "
        f"explainer_art={len(explainer_art)}"
    )
    print(f"cover=1 part_openers={len(PARTS)} local_links=OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
