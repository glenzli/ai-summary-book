#!/usr/bin/env python3
"""Validate the structure and local assets of the train encyclopedia."""

from __future__ import annotations

import html
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parent

NUMBERED_CHAPTERS = (
    "00_for_grownups.md",
    "01_how_trains_move.md",
    "02_steam_pioneers.md",
    "03_electric_diesel_streamliners.md",
    "04_world_high_speed.md",
    "05_shinkansen_family.md",
    "06_japan_everyday.md",
    "07_japan_special_trains.md",
    "08_work_trains.md",
    "09_unusual_railways.md",
    "10_spotter_games.md",
    "11_more_trains.md",
    "12_nankai_trains.md",
    "13_china_trains.md",
)
CARD_CHAPTERS = NUMBERED_CHAPTERS[2:10] + NUMBERED_CHAPTERS[11:]
EXPECTED_CARD_NUMBERS = {f"{number:03d}" for number in range(1, 160)}

ATX_HEADING_RE = re.compile(r"^ {0,3}(#{1,6})(?:[ \t]+|$)")
CARD_HEADING_RE = re.compile(
    r"^ {0,3}##[ \t]+(?P<number>\d{3})(?=[^\d]|$)", re.MULTILINE
)
FENCE_RE = re.compile(r"^ {0,3}(?P<mark>\x60{3,}|~{3,})(?P<rest>.*)$")
INLINE_LINK_RE = re.compile(
    r"!?\[[^\]\n]*\]\(\s*(?P<target><[^>\n]+>|(?:\\.|[^)\s])+)",
)
REFERENCE_LINK_RE = re.compile(
    r"^ {0,3}\[[^\]\n]+\]:[ \t]*(?P<target><[^>\n]+>|(?:\\.|\S)+)"
)
HTML_LINK_RE = re.compile(
    r"<(?:a|img)\b[^>]*?\b(?:href|src)[ \t]*=[ \t]*"
    r"(?P<quote>['\"])(?P<target>.*?)(?P=quote)",
    re.IGNORECASE,
)
TRAIN_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_.-])(?:\./)?images/trains/"
    r"(?P<filename>[^\s)'\"<>]+\.webp)",
    re.IGNORECASE,
)
CREDIT_LINK_RE = re.compile(
    r"\[[^\]\n]+\]\(\s*(?:\./)?IMAGE_CREDITS\.md#img-"
    r"(?P<id>[A-Za-z0-9._-]+)(?:\s+[^)]*)?\)",
)
BIRTH_CARD_RE = re.compile(
    r"^ {0,3}(?:\*\*)?出生卡[：:](?:\*\*)?", re.MULTILINE
)
LOOK_FOR_RE = re.compile(
    r"^ {0,3}(?:\*\*)?找找看[：:](?:\*\*)?", re.MULTILINE
)
GROWNUP_RE = re.compile(r"^ {0,3}>[ \t]*给大人[：:]", re.MULTILINE)
WORK_MARKER_RE = re.compile(r"\b(?:TODO|FIXME)\b", re.IGNORECASE)
HTML_ID_RE = re.compile(r"\bid[ \t]*=[ \t]*(['\"])(?P<id>.*?)\1", re.IGNORECASE)


@dataclass(frozen=True)
class Problem:
    path: str
    line: int | None
    message: str


@dataclass(frozen=True)
class Reference:
    line: int
    target: str
    resolved: Path | None


@dataclass(frozen=True)
class Card:
    number: str
    path: Path
    line: int
    image_id: str | None
    credit_id: str | None


class Reporter:
    def __init__(self) -> None:
        self.problems: list[Problem] = []

    def add(self, path: Path | str, message: str, line: int | None = None) -> None:
        if isinstance(path, Path):
            try:
                label = path.resolve().relative_to(ROOT).as_posix()
            except ValueError:
                label = str(path)
        else:
            label = path
        self.problems.append(Problem(label, line, message))

    def finish(self, *, cards: int, images: int, principles: int) -> int:
        if self.problems:
            for problem in sorted(
                self.problems,
                key=lambda item: (item.path, item.line or 0, item.message),
            ):
                location = problem.path
                if problem.line is not None:
                    location += f":{problem.line}"
                print(f"ERROR {location}: {problem.message}")
            count = len(self.problems)
            noun = "problem" if count == 1 else "problems"
            print(f"\nFAILED: {count} {noun} found.")
            return 1

        print(
            "OK: validated "
            f"{len(NUMBERED_CHAPTERS)} numbered chapters, "
            f"{cards} train cards, {images} train WebPs, "
            f"and {principles} principle SVGs."
        )
        return 0


def read_utf8(path: Path, reporter: Reporter) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        reporter.add(path, "file is missing")
    except UnicodeDecodeError as exc:
        reporter.add(path, f"is not valid UTF-8 ({exc})")
    except OSError as exc:
        reporter.add(path, f"could not be read ({exc})")
    return None


def scan_markdown(
    path: Path, text: str, reporter: Reporter
) -> tuple[list[tuple[int, int]], list[tuple[int, str]]]:
    """Check Markdown hygiene and return headings and non-fenced content lines."""
    lines = text.splitlines()
    headings: list[tuple[int, int]] = []
    content_lines: list[tuple[int, str]] = []
    previous_heading_level: int | None = None
    fence_char: str | None = None
    fence_length = 0
    fence_open_line: int | None = None

    for line_number, line in enumerate(lines, start=1):
        if re.search(r"[ \t]+$", line):
            reporter.add(path, "trailing whitespace", line_number)
        if WORK_MARKER_RE.search(line):
            reporter.add(path, "unfinished-work marker is not allowed", line_number)

        if fence_char is not None:
            closing = re.match(
                rf"^ {{0,3}}{re.escape(fence_char)}{{{fence_length},}}[ \t]*$",
                line,
            )
            if closing:
                fence_char = None
                fence_length = 0
                fence_open_line = None
            continue

        fence = FENCE_RE.match(line)
        if fence:
            mark = fence.group("mark")
            fence_char = mark[0]
            fence_length = len(mark)
            fence_open_line = line_number
            continue

        content_lines.append((line_number, line))
        heading = ATX_HEADING_RE.match(line)
        if not heading:
            continue
        level = len(heading.group(1))
        headings.append((line_number, level))
        if previous_heading_level is not None and level > previous_heading_level + 1:
            reporter.add(
                path,
                f"heading level jumps from H{previous_heading_level} to H{level}",
                line_number,
            )
        previous_heading_level = level

    if fence_char is not None:
        reporter.add(
            path,
            f"unclosed {fence_char * fence_length} code fence",
            fence_open_line,
        )

    return headings, content_lines


def clean_destination(raw_target: str) -> str:
    target = html.unescape(raw_target.strip())
    if target.startswith("<") and target.endswith(">"):
        target = target[1:-1]
    return re.sub(r"\\([\\\x60*{}\[\]()#+.! _>~-])", r"\1", target)


def resolve_local_reference(source: Path, target: str) -> Path | None:
    target = clean_destination(target)
    if not target or target.startswith("#") or target.startswith("//"):
        return None
    parsed = urlsplit(target)
    if parsed.scheme or not parsed.path:
        return None
    decoded_path = unquote(parsed.path)
    local_path = Path(decoded_path)
    if not local_path.is_absolute():
        local_path = source.parent / local_path
    return local_path.resolve()


def collect_references(
    path: Path,
    content_lines: list[tuple[int, str]],
    reporter: Reporter,
) -> list[Reference]:
    references: list[Reference] = []
    for line_number, line in content_lines:
        raw_targets = [match.group("target") for match in INLINE_LINK_RE.finditer(line)]
        definition = REFERENCE_LINK_RE.match(line)
        if definition:
            raw_targets.append(definition.group("target"))
        raw_targets.extend(match.group("target") for match in HTML_LINK_RE.finditer(line))

        for raw_target in raw_targets:
            target = clean_destination(raw_target)
            resolved = resolve_local_reference(path, target)
            references.append(Reference(line_number, target, resolved))
            if resolved is not None and not resolved.exists():
                reporter.add(path, f"local link target does not exist: {target}", line_number)
    return references


def check_chapters(
    texts: dict[Path, str],
    headings: dict[Path, list[tuple[int, int]]],
    references: dict[Path, list[Reference]],
    reporter: Reporter,
) -> None:
    readme = ROOT / "README.md"
    if readme not in texts:
        if not readme.exists():
            reporter.add(readme, "file is missing")
        return

    linked_files = {
        reference.resolved
        for reference in references.get(readme, [])
        if reference.resolved is not None
    }
    for chapter_name in NUMBERED_CHAPTERS:
        chapter = (ROOT / chapter_name).resolve()
        if chapter not in linked_files:
            reporter.add(readme, f"does not link numbered chapter {chapter_name}")

    for chapter_name in NUMBERED_CHAPTERS[1:]:
        chapter = ROOT / chapter_name
        if chapter not in texts:
            if not chapter.exists():
                reporter.add(chapter, "numbered chapter is missing")
            continue
        h1_lines = [line for line, level in headings.get(chapter, []) if level == 1]
        if len(h1_lines) != 1:
            detail = "none" if not h1_lines else ", ".join(map(str, h1_lines))
            reporter.add(
                chapter,
                f"must contain exactly one H1 heading (found {len(h1_lines)}; lines: {detail})",
            )


def line_at(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def check_marker_count(
    reporter: Reporter,
    path: Path,
    card_number: str,
    card_line: int,
    label: str,
    count: int,
) -> None:
    if count != 1:
        reporter.add(
            path,
            f"card {card_number} must have exactly one {label} (found {count})",
            card_line,
        )


def collect_cards(texts: dict[Path, str], reporter: Reporter) -> list[Card]:
    cards: list[Card] = []
    number_locations: dict[str, list[str]] = {}

    for chapter_name in CARD_CHAPTERS:
        path = ROOT / chapter_name
        text = texts.get(path)
        if text is None:
            continue
        matches = list(CARD_HEADING_RE.finditer(text))
        for index, match in enumerate(matches):
            number = match.group("number")
            card_line = line_at(text, match.start())
            block_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            block = text[match.start():block_end]
            number_locations.setdefault(number, []).append(f"{chapter_name}:{card_line}")

            train_paths = [item.group("filename") for item in TRAIN_PATH_RE.finditer(block)]
            credit_ids = [item.group("id") for item in CREDIT_LINK_RE.finditer(block)]
            check_marker_count(
                reporter, path, number, card_line, "images/trains WebP path", len(train_paths)
            )
            check_marker_count(
                reporter, path, number, card_line, "image-credit anchor link", len(credit_ids)
            )
            check_marker_count(
                reporter,
                path,
                number,
                card_line,
                "出生卡",
                len(BIRTH_CARD_RE.findall(block)),
            )
            check_marker_count(
                reporter,
                path,
                number,
                card_line,
                "找找看",
                len(LOOK_FOR_RE.findall(block)),
            )
            check_marker_count(
                reporter,
                path,
                number,
                card_line,
                "给大人 note",
                len(GROWNUP_RE.findall(block)),
            )

            image_id = Path(train_paths[0]).stem if len(train_paths) == 1 else None
            credit_id = credit_ids[0] if len(credit_ids) == 1 else None
            if image_id is not None and credit_id is not None and image_id != credit_id:
                reporter.add(
                    path,
                    f"card {number} image basename {image_id!r} does not match "
                    f"credit anchor {credit_id!r}",
                    card_line,
                )
            if image_id is not None and not image_id.startswith(f"t{number}-"):
                reporter.add(
                    path,
                    f"card {number} image ID {image_id!r} does not start with t{number}-",
                    card_line,
                )
            cards.append(Card(number, path, card_line, image_id, credit_id))

    found_numbers = set(number_locations)
    missing_numbers = sorted(EXPECTED_CARD_NUMBERS - found_numbers)
    unexpected_numbers = sorted(found_numbers - EXPECTED_CARD_NUMBERS)
    if missing_numbers:
        reporter.add(
            "train-card chapters",
            "missing train card numbers: " + ", ".join(missing_numbers),
        )
    if unexpected_numbers:
        reporter.add(
            "train-card chapters",
            "unexpected train card numbers: " + ", ".join(unexpected_numbers),
        )
    for number, locations in sorted(number_locations.items()):
        if len(locations) != 1:
            reporter.add(
                "train-card chapters",
                f"train card {number} occurs {len(locations)} times: " + ", ".join(locations),
            )

    return cards


def load_json_ids(path: Path, label: str, reporter: Reporter) -> tuple[set[str], bool]:
    text = read_utf8(path, reporter)
    if text is None:
        return set(), False
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        reporter.add(path, f"invalid JSON: {exc.msg}", exc.lineno)
        return set(), False
    if not isinstance(data, list):
        reporter.add(path, f"{label} must be a JSON array")
        return set(), False

    ids: list[str] = []
    for index, record in enumerate(data):
        if not isinstance(record, dict):
            reporter.add(path, f"{label} item {index + 1} is not an object")
            continue
        identifier = record.get("id")
        if not isinstance(identifier, str) or not identifier.strip():
            reporter.add(path, f"{label} item {index + 1} has no non-empty string id")
            continue
        ids.append(identifier)

    counts = Counter(ids)
    for identifier, count in sorted(counts.items()):
        if count > 1:
            reporter.add(path, f"duplicate ID {identifier!r} occurs {count} times")
    return set(ids), True


def compare_ids(
    reporter: Reporter,
    path: Path | str,
    actual: set[str],
    expected: set[str],
    actual_label: str,
    expected_label: str,
) -> None:
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing:
        reporter.add(
            path,
            f"{actual_label} missing IDs from {expected_label}: " + ", ".join(missing),
        )
    if extra:
        reporter.add(
            path,
            f"{actual_label} has IDs absent from {expected_label}: " + ", ".join(extra),
        )


def check_image_inventory(cards: list[Card], reporter: Reporter) -> int:
    card_ids_list = [card.image_id for card in cards if card.image_id is not None]
    card_ids = set(card_ids_list)
    for identifier, count in sorted(Counter(card_ids_list).items()):
        if count > 1:
            reporter.add("train cards", f"image ID {identifier!r} is used by {count} cards")

    targets_path = ROOT / "tools" / "image_targets.json"
    target_ids, targets_loaded = load_json_ids(targets_path, "image target", reporter)
    if targets_loaded:
        compare_ids(
            reporter,
            targets_path,
            target_ids,
            card_ids,
            "image_targets.json",
            "train cards",
        )
    inventory_ids = target_ids if targets_loaded else card_ids

    metadata_path = ROOT / "image_metadata.json"
    metadata_ids, metadata_loaded = load_json_ids(metadata_path, "image metadata", reporter)
    if metadata_loaded:
        compare_ids(
            reporter,
            metadata_path,
            metadata_ids,
            inventory_ids,
            "image metadata",
            "image targets",
        )

    train_dir = ROOT / "images" / "trains"
    files: list[Path] = []
    if not train_dir.is_dir():
        reporter.add(train_dir, "train image directory is missing")
    else:
        files = sorted(
            path for path in train_dir.iterdir() if path.is_file() and path.suffix.lower() == ".webp"
        )
    file_ids_list = [path.stem for path in files]
    file_ids = set(file_ids_list)
    for identifier, count in sorted(Counter(file_ids_list).items()):
        if count > 1:
            reporter.add(train_dir, f"multiple WebP files have ID {identifier!r}")
    compare_ids(
        reporter,
        train_dir,
        file_ids,
        inventory_ids,
        "train WebP files",
        "image targets",
    )
    return len(files)


def check_credit_anchors(cards: list[Card], texts: dict[Path, str], reporter: Reporter) -> None:
    credits_path = ROOT / "IMAGE_CREDITS.md"
    credits_text = texts.get(credits_path)
    if credits_text is None:
        return
    anchor_ids = [match.group("id") for match in HTML_ID_RE.finditer(credits_text)]
    anchor_counts = Counter(anchor_ids)
    for card in cards:
        if card.credit_id is None:
            continue
        expected_anchor = f"img-{card.credit_id}"
        count = anchor_counts[expected_anchor]
        if count != 1:
            reporter.add(
                card.path,
                f"card {card.number} links anchor #{expected_anchor}, "
                f"but IMAGE_CREDITS.md defines it {count} times",
                card.line,
            )


def check_principle_svgs(
    references: dict[Path, list[Reference]], reporter: Reporter
) -> int:
    principle_dir = ROOT / "images" / "principles"
    if not principle_dir.is_dir():
        reporter.add(principle_dir, "principle image directory is missing")
        return 0
    svgs = sorted(principle_dir.glob("*.svg"))
    if not svgs:
        reporter.add(principle_dir, "no principle SVG files found")
        return 0
    referenced_paths = {
        reference.resolved
        for file_references in references.values()
        for reference in file_references
        if reference.resolved is not None
    }
    for svg in svgs:
        if svg.resolve() not in referenced_paths:
            reporter.add(svg, "principle SVG is not referenced by any Markdown file")
    return len(svgs)


def main() -> int:
    reporter = Reporter()
    texts: dict[Path, str] = {}
    headings: dict[Path, list[tuple[int, int]]] = {}
    references: dict[Path, list[Reference]] = {}

    markdown_paths = sorted(ROOT.rglob("*.md"))
    if not markdown_paths:
        reporter.add(ROOT, "no Markdown files found")
    for path in markdown_paths:
        text = read_utf8(path, reporter)
        if text is None:
            continue
        texts[path] = text
        file_headings, content_lines = scan_markdown(path, text, reporter)
        headings[path] = file_headings
        references[path] = collect_references(path, content_lines, reporter)

    check_chapters(texts, headings, references, reporter)
    cards = collect_cards(texts, reporter)
    check_credit_anchors(cards, texts, reporter)
    image_count = check_image_inventory(cards, reporter)
    principle_count = check_principle_svgs(references, reporter)
    return reporter.finish(cards=len(cards), images=image_count, principles=principle_count)


if __name__ == "__main__":
    raise SystemExit(main())
