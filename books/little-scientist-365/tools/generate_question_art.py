#!/usr/bin/env python3
"""Generate explicit, topic-specific SVG spot art for every question entry."""

from __future__ import annotations

import html
import math
import re
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = Path(__file__).resolve().parent
OUT = ROOT / "images" / "questions"
QUESTION_RE = re.compile(r"^### 第 (\d{3}) 问｜(.+)$")

if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import generate_explainer_art as explainer  # noqa: E402
from book_structure import CHAPTERS, CHAPTER_BY_QUESTION  # noqa: E402
from earth_weather_illustrations import (  # noqa: E402
    EARTH_WEATHER_KEYS,
    EARTH_WEATHER_QUESTIONS,
    earth_weather_body,
)
from invertebrate_illustrations import (  # noqa: E402
    INVERTEBRATE_KEYS,
    INVERTEBRATE_QUESTIONS,
    invertebrate_body,
)
from machine_illustrations import MACHINE_KEYS, MACHINE_QUESTIONS, machine_body  # noqa: E402
from plant_illustrations import PLANT_KEYS, PLANT_QUESTIONS, plant_body  # noqa: E402
from question_scene_specs import CUSTOM_ICON_KEYS, MANUAL_SCENES, SUBJECT_ICONS  # noqa: E402
from textbook_illustrations import TEXTBOOK_KEYS, TEXTBOOK_QUESTIONS, textbook_body  # noqa: E402
from vertebrate_illustrations import (  # noqa: E402
    VERTEBRATE_KEYS,
    VERTEBRATE_QUESTIONS,
    vertebrate_body,
)


# Bright, clean cel-animation colors. Each exploration part keeps an accent
# while white, ink blue, warm yellow, coral, and green recur across the book.
PALETTES = [
    ("#F4F9FF", "#2563EB", "#F5B83D", "#FFFFFF", "#17213A"),
    ("#F0FDFA", "#0F8A8A", "#FB7185", "#FFFFFF", "#173746"),
    ("#F5FBEF", "#3B8C5A", "#F3A712", "#FFFFFF", "#203D2D"),
    ("#FFF8ED", "#E66A3D", "#2F8F83", "#FFFFFF", "#3D2E35"),
    ("#FFF4F7", "#2563A6", "#F06A72", "#FFFFFF", "#24324A"),
    ("#EFFAFA", "#147D88", "#F06B61", "#FFFFFF", "#23374D"),
    ("#FFF7ED", "#3B78A8", "#E9633F", "#FFFFFF", "#293747"),
    ("#F3FAF5", "#2F7D5B", "#3D7DB2", "#FFFFFF", "#263B35"),
    ("#EEF9FC", "#147A96", "#F06C57", "#FFFFFF", "#173B55"),
    ("#F3F2FF", "#5364C7", "#F0B43C", "#FFFFFF", "#25284A"),
    ("#FFF7F2", "#B8584A", "#397A91", "#FFFFFF", "#42343A"),
    ("#F2FAF5", "#2D8065", "#3478AA", "#FFFFFF", "#263D38"),
    ("#F7F7F3", "#4F6B5C", "#D68A3A", "#FFFFFF", "#283B33"),
]

DIAGRAMS_BY_QUESTION = {diagram.question: diagram for diagram in explainer.DIAGRAMS}
SUBJECT_ICON_KEYS = set(SUBJECT_ICONS.values())


# The main picture in a read-aloud science book should normally show the
# phenomenon or object itself. These overrides select that observable focus
# when the mechanism specification begins with an invisible particle, force,
# field, or other explanatory abstraction.
CONCRETE_FOCUS_OVERRIDES: dict[int, str] = {
    1: "sun", 2: "sun-ground", 4: "sun-ground", 5: "shadow", 6: "shadow",
    7: "mirror", 8: "window-light", 9: "drop-rainbow", 10: "half-earth",
    12: "moon", 14: "star-field", 15: "half-earth", 16: "sun-ground",
    17: "tilted-earth", 18: "tilted-earth", 20: "ice", 21: "snow",
    23: "metal", 26: "clock", 27: "calendar", 28: "week", 31: "earth-air",
    32: "balloon", 33: "wind", 34: "wind", 37: "straw", 38: "cloud",
    39: "rain", 41: "rain-snow", 42: "hailstone", 44: "lightning",
    45: "cloud", 47: "water", 49: "cloud", 50: "ice-lattice",
    51: "cold-water", 52: "river", 53: "water", 54: "water", 55: "home",
    56: "clean-water", 57: "water", 58: "cloud", 60: "seed", 61: "seed",
    63: "root", 64: "stem", 66: "leaf", 67: "leaf", 68: "leaf",
    70: "flower", 71: "bee", 73: "seed-parachute", 74: "hook-seed",
    75: "seed", 76: "growth-ring", 77: "tree", 78: "tree", 79: "tree",
    80: "leaf", 82: "stem", 83: "bent-plant", 84: "mushroom",
    85: "leaf", 86: "leaf", 87: "stem", 88: "stem", 89: "root",
    182: "lever", 183: "lever", 184: "seesaw", 185: "pulley",
    186: "wheel-cart", 187: "gear-contact", 189: "screw", 190: "wedge",
    191: "ramp", 192: "spring", 194: "magnet", 195: "earth-field",
    196: "coil", 197: "lamp", 198: "battery", 199: "motor", 201: "rotor",
    202: "water", 203: "faucet-handle", 204: "lever", 205: "hooklets",
    207: "lever", 208: "truss", 209: "arch", 210: "pulley", 211: "lift",
    212: "receiver", 213: "earth", 215: "earth-person", 216: "earth-cutaway",
    244: "ocean", 245: "wind-wave", 246: "earth", 247: "warm-current",
    248: "ocean-light", 249: "deep-object", 250: "ocean-light",
    251: "coral", 252: "healthy-coral", 305: "liquid-water", 306: "matter",
    307: "ice-lattice", 308: "drop", 309: "water-molecules",
    310: "salt-solution", 311: "water", 312: "warm-water",
    313: "liquid-particles", 314: "bread", 315: "protein-network",
    316: "folded-protein", 317: "fruit", 318: "popcorn",
    319: "evaporator", 320: "ice-lattice", 321: "microwave", 322: "bread",
    323: "thermos", 324: "wood", 325: "wet-paper", 326: "paper-fibers",
    327: "glass", 328: "iron", 329: "matter", 330: "spring",
    331: "igneous-rock", 332: "web", 333: "rotation", 334: "pressure",
    335: "step-one", 336: "bicycle", 337: "motor", 338: "train-wheel",
    339: "seatbelt", 340: "traffic-light", 341: "train-wheel",
    342: "train-wheel", 343: "boat", 344: "sailboat", 345: "airplane-wing",
    346: "rotor", 347: "hot-balloon", 348: "surface-sub", 349: "sun",
    350: "solar-cell", 351: "rotor", 352: "rotor", 353: "home",
    354: "light", 355: "home", 356: "matter", 357: "leaf", 358: "tree",
    359: "earth-air", 360: "tree", 361: "eye", 362: "measure",
    363: "experiment", 364: "model", 365: "question",
}


def part_for(question: int) -> int:
    return CHAPTER_BY_QUESTION[question].part - 1


def entries() -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []
    for source in (ROOT / chapter.filename for chapter in CHAPTERS):
        for line_text in source.read_text(encoding="utf-8").splitlines():
            match = QUESTION_RE.match(line_text)
            if match:
                found.append((int(match.group(1)), match.group(2)))
    return found


def circle(cx: float, cy: float, radius: float, fill: str, stroke: str = "none", width: float = 0) -> str:
    return f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{radius:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="{width:.1f}"/>'


def ellipse(cx: float, cy: float, rx: float, ry: float, fill: str, stroke: str = "none", width: float = 0) -> str:
    return f'<ellipse cx="{cx:.1f}" cy="{cy:.1f}" rx="{rx:.1f}" ry="{ry:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="{width:.1f}"/>'


def rect(x: float, y: float, width: float, height: float, fill: str, radius: float = 8, stroke: str = "none", stroke_width: float = 0) -> str:
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" rx="{radius:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width:.1f}"/>'


def line(x1: float, y1: float, x2: float, y2: float, stroke: str, width: float = 7, marker: bool = False, dash: str = "") -> str:
    marker_attr = ' marker-end="url(#arrow)"' if marker else ""
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{width:.1f}" stroke-linecap="round"{marker_attr}{dash_attr}/>'


def path(data: str, fill: str = "none", stroke: str = "none", width: float = 0, marker: bool = False, dash: str = "") -> str:
    marker_attr = ' marker-end="url(#arrow)"' if marker else ""
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<path d="{data}" fill="{fill}" stroke="{stroke}" stroke-width="{width:.1f}" stroke-linecap="round" stroke-linejoin="round"{marker_attr}{dash_attr}/>'


def polygon(points: list[tuple[float, float]], fill: str, stroke: str = "none", width: float = 0) -> str:
    data = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f'<polygon points="{data}" fill="{fill}" stroke="{stroke}" stroke-width="{width:.1f}" stroke-linejoin="round"/>'


def insect_subject(kind: str, cx: float, cy: float, s: float, palette: tuple[str, ...]) -> str:
    _, primary, accent, light, dark = palette
    bits: list[str] = []

    if kind == "spiderweb":
        for radius in (28, 52, 78, 104):
            bits.append(circle(cx, cy, radius * s, "none", primary if radius % 52 else accent, 3 * s))
        for index in range(8):
            angle = index * math.tau / 8
            bits.append(line(cx, cy, cx + math.cos(angle) * 108 * s, cy + math.sin(angle) * 108 * s, dark, 3 * s))
        bits.append(circle(cx + 25 * s, cy - 18 * s, 13 * s, accent, dark, 3 * s))
        return "".join(bits)

    if kind == "spider":
        bits.extend([circle(cx, cy - 24 * s, 30 * s, primary, dark, 5 * s), ellipse(cx, cy + 30 * s, 43 * s, 53 * s, accent, dark, 5 * s)])
        for side in (-1, 1):
            for index, offset in enumerate((-50, -20, 17, 50)):
                knee_x = cx + side * (62 + index * 6) * s
                knee_y = cy + offset * s
                bits.append(path(f"M{cx+side*30*s},{cy+(offset*.45)*s} L{knee_x},{knee_y} L{cx+side*(105+index*4)*s},{cy+(offset+(index-1.5)*16)*s}", "none", dark, 5 * s))
        bits.append(circle(cx - 10 * s, cy - 31 * s, 4 * s, light))
        bits.append(circle(cx + 10 * s, cy - 31 * s, 4 * s, light))
        return "".join(bits)

    if kind == "ant-nest":
        bits.append(path(f"M{cx-120*s},{cy+75*s} Q{cx},{cy-55*s} {cx+120*s},{cy+75*s} Z", accent, dark, 5 * s))
        bits.append(path(f"M{cx-88*s},{cy+50*s} Q{cx-30*s},{cy+5*s} {cx-3*s},{cy+52*s} T{cx+82*s},{cy+15*s} M{cx-5*s},{cy+52*s} V{cy+90*s}", "none", light, 10 * s))
        for dx, dy in [(-63, 35), (38, 42), (0, -30)]:
            bits.append(circle(cx + dx * s, cy + dy * s, 8 * s, dark))
            bits.append(circle(cx + (dx + 12) * s, cy + dy * s, 7 * s, primary))
        return "".join(bits)

    if kind in {"ant", "general"}:
        sizes = (24, 31, 39) if kind == "ant" else (29, 38, 47)
        xs = (-60, -10, 58)
        for index, (x, radius) in enumerate(zip(xs, sizes)):
            bits.append(circle(cx + x * s, cy, radius * s, accent if index == 1 else primary, dark, 4 * s))
        for side in (-1, 1):
            for offset in (-28, 0, 28):
                bits.append(line(cx - 5 * s, cy + offset * s, cx + side * 82 * s, cy + (offset + side * 20) * s, dark, 4 * s))
        bits.append(path(f"M{cx-76*s},{cy-20*s} Q{cx-100*s},{cy-65*s} {cx-116*s},{cy-42*s} M{cx-53*s},{cy-22*s} Q{cx-62*s},{cy-72*s} {cx-35*s},{cy-60*s}", "none", dark, 4 * s))
        return "".join(bits)

    if kind == "bee":
        bits.append(ellipse(cx, cy, 78 * s, 48 * s, "#F5B83D", dark, 5 * s))
        for offset in (-35, 0, 35):
            bits.append(line(cx + offset * s, cy - 42 * s, cx + offset * s, cy + 42 * s, dark, 9 * s))
        bits.append(circle(cx + 72 * s, cy - 5 * s, 29 * s, primary, dark, 5 * s))
        bits.append(ellipse(cx - 24 * s, cy - 48 * s, 48 * s, 28 * s, light, primary, 4 * s))
        bits.append(ellipse(cx + 22 * s, cy - 52 * s, 48 * s, 28 * s, light, primary, 4 * s))
        return "".join(bits)

    if kind == "butterfly":
        bits.append(ellipse(cx - 52 * s, cy - 25 * s, 60 * s, 78 * s, accent, dark, 5 * s))
        bits.append(ellipse(cx + 52 * s, cy - 25 * s, 60 * s, 78 * s, primary, dark, 5 * s))
        bits.append(ellipse(cx - 45 * s, cy + 55 * s, 43 * s, 51 * s, primary, dark, 4 * s))
        bits.append(ellipse(cx + 45 * s, cy + 55 * s, 43 * s, 51 * s, accent, dark, 4 * s))
        bits.append(rect(cx - 10 * s, cy - 78 * s, 20 * s, 165 * s, dark, 10 * s))
        for side in (-1, 1):
            bits.append(path(f"M{cx+side*5*s},{cy-70*s} Q{cx+side*38*s},{cy-110*s} {cx+side*60*s},{cy-92*s}", "none", dark, 4 * s))
        return "".join(bits)

    if kind == "caterpillar":
        for index in range(6):
            x = cx + (-72 + index * 29) * s
            y = cy + (8 if index % 2 else -5) * s
            bits.append(circle(x, y, (29 if index < 5 else 34) * s, primary if index % 2 else accent, dark, 4 * s))
            bits.append(line(x - 9 * s, y + 26 * s, x - 18 * s, y + 48 * s, dark, 3 * s))
            bits.append(line(x + 9 * s, y + 26 * s, x + 18 * s, y + 48 * s, dark, 3 * s))
        bits.append(circle(cx + 84 * s, cy - 12 * s, 4 * s, light))
        return "".join(bits)

    if kind in {"beetle", "ladybug", "firefly"}:
        body_color = "#F05252" if kind == "ladybug" else primary
        bits.append(ellipse(cx, cy + 12 * s, 72 * s, 92 * s, body_color, dark, 6 * s))
        bits.append(circle(cx, cy - 72 * s, 34 * s, dark, dark, 4 * s))
        bits.append(line(cx, cy - 55 * s, cx, cy + 98 * s, light, 4 * s))
        if kind == "ladybug":
            for dx, dy in [(-30, -20), (30, -20), (-37, 30), (37, 30), (-18, 66), (18, 66)]:
                bits.append(circle(cx + dx * s, cy + dy * s, 9 * s, dark))
        elif kind == "firefly":
            bits.append(ellipse(cx, cy + 62 * s, 53 * s, 35 * s, "#FFE66D", accent, 4 * s))
            for radius in (88, 106):
                bits.append(ellipse(cx, cy + 45 * s, radius * .65 * s, radius * .45 * s, "none", "#FFE66D", 3 * s))
        else:
            bits.append(path(f"M{cx-58*s},{cy-8*s} Q{cx},{cy-45*s} {cx+58*s},{cy-8*s}", "none", accent, 7 * s))
        for side in (-1, 1):
            for offset in (-35, 5, 48):
                bits.append(line(cx + side * 48 * s, cy + offset * s, cx + side * 102 * s, cy + (offset + side * 14) * s, dark, 4 * s))
        return "".join(bits)

    if kind == "mosquito":
        bits.append(ellipse(cx, cy, 18 * s, 76 * s, primary, dark, 4 * s))
        bits.append(circle(cx, cy - 82 * s, 21 * s, accent, dark, 4 * s))
        bits.append(line(cx, cy - 100 * s, cx + 70 * s, cy - 125 * s, dark, 4 * s))
        bits.append(ellipse(cx - 42 * s, cy - 28 * s, 51 * s, 25 * s, light, primary, 4 * s))
        bits.append(ellipse(cx + 42 * s, cy - 28 * s, 51 * s, 25 * s, light, primary, 4 * s))
        for side in (-1, 1):
            for offset in (-35, 5, 38):
                bits.append(path(f"M{cx+side*12*s},{cy+offset*s} L{cx+side*72*s},{cy+(offset+35)*s} L{cx+side*104*s},{cy+(offset+20)*s}", "none", dark, 3.5 * s))
        return "".join(bits)

    if kind == "dragonfly":
        for index in range(7):
            bits.append(circle(cx, cy + (-70 + index * 23) * s, (18 - index * 1.3) * s, primary if index % 2 else accent, dark, 3 * s))
        for side in (-1, 1):
            bits.append(ellipse(cx + side * 58 * s, cy - 33 * s, 69 * s, 23 * s, light, primary, 4 * s))
            bits.append(ellipse(cx + side * 52 * s, cy + 12 * s, 62 * s, 20 * s, light, accent, 4 * s))
        bits.append(circle(cx, cy - 93 * s, 23 * s, dark))
        return "".join(bits)

    if kind in {"grasshopper", "cricket", "mantis", "stick"}:
        body_width = 16 if kind == "stick" else 26
        bits.append(ellipse(cx, cy, body_width * s, 82 * s, primary, dark, 4 * s))
        bits.append(circle(cx, cy - 85 * s, 24 * s, accent, dark, 4 * s))
        if kind == "mantis":
            for side in (-1, 1):
                bits.append(path(f"M{cx+side*10*s},{cy-55*s} L{cx+side*72*s},{cy-92*s} L{cx+side*42*s},{cy-25*s}", "none", dark, 8 * s))
                bits.append(line(cx + side * 8 * s, cy + 40 * s, cx + side * 76 * s, cy + 85 * s, dark, 6 * s))
        elif kind in {"grasshopper", "cricket"}:
            for side in (-1, 1):
                bits.append(path(f"M{cx+side*10*s},{cy+28*s} L{cx+side*82*s},{cy+82*s} L{cx+side*120*s},{cy+42*s}", "none", dark, 9 * s))
                bits.append(line(cx + side * 8 * s, cy - 12 * s, cx + side * 70 * s, cy + 12 * s, dark, 5 * s))
            if kind == "cricket":
                bits.append(path(f"M{cx-12*s},{cy+10*s} Q{cx+55*s},{cy+45*s} {cx+75*s},{cy-25*s}", "none", accent, 6 * s))
        else:
            for side in (-1, 1):
                for offset in (-45, 5, 48):
                    bits.append(line(cx, cy + offset * s, cx + side * 95 * s, cy + (offset + side * 18) * s, dark, 4 * s))
        return "".join(bits)

    raise KeyError(f"unknown insect subject: {kind}")


def small_subject(kind: str, cx: float, cy: float, s: float, palette: tuple[str, ...]) -> str:
    _, primary, accent, light, dark = palette
    bits: list[str] = []
    if kind in {"snail", "snail-trail"}:
        if kind == "snail-trail":
            bits.append(path(f"M{cx-135*s},{cy+92*s} Q{cx},{cy+65*s} {cx+138*s},{cy+92*s}", "none", light, 14 * s))
        bits.append(path(f"M{cx-110*s},{cy+55*s} Q{cx-30*s},{cy+20*s} {cx+90*s},{cy+48*s} Q{cx+125*s},{cy+58*s} {cx+98*s},{cy+78*s} H{cx-102*s} Z", primary, dark, 5 * s))
        bits.append(circle(cx - 32 * s, cy - 2 * s, 62 * s, accent, dark, 5 * s))
        bits.append(path(f"M{cx-32*s},{cy-2*s} q{38*s},{-34*s} {42*s},{12*s} q{3*s},{36*s} {-31*s},{33*s}", "none", dark, 5 * s))
        for offset in (42, 72):
            bits.append(line(cx + offset * s, cy + 42 * s, cx + (offset + 10) * s, cy - 8 * s, dark, 3 * s))
            bits.append(circle(cx + (offset + 10) * s, cy - 12 * s, 5 * s, dark))
        return "".join(bits)

    if kind in {"earthworm", "earthworm-soil"}:
        if kind == "earthworm-soil":
            bits.append(path(f"M{cx-130*s},{cy+85*s} Q{cx},{cy+45*s} {cx+130*s},{cy+85*s} V{cy+125*s} H{cx-130*s} Z", accent, dark, 4 * s))
        bits.append(path(f"M{cx-120*s},{cy+25*s} C{cx-70*s},{cy-75*s} {cx-5*s},{cy+105*s} {cx+48*s},{cy+15*s} C{cx+78*s},{cy-40*s} {cx+108*s},{cy-5*s} {cx+122*s},{cy+42*s}", "none", primary, 28 * s))
        for index in range(-4, 5):
            bits.append(line(cx + index * 22 * s, cy + (15 if index % 2 else -2) * s, cx + index * 22 * s, cy + (34 if index % 2 else 17) * s, dark, 2.5 * s))
        return "".join(bits)

    if kind == "centipede":
        for index in range(9):
            x = cx + (-88 + index * 22) * s
            y = cy + math.sin(index * .8) * 22 * s
            bits.append(circle(x, y, 18 * s, primary if index % 2 else accent, dark, 3 * s))
            for side in (-1, 1):
                bits.append(line(x, y + side * 14 * s, x + (8 if index % 2 else -8) * s, y + side * 45 * s, dark, 3 * s))
        return "".join(bits)

    if kind == "pillbug":
        bits.append(ellipse(cx, cy, 103 * s, 72 * s, primary, dark, 6 * s))
        for offset in (-68, -34, 0, 34, 68):
            bits.append(path(f"M{cx+offset*s},{cy-62*s} Q{cx+(offset+18)*s},{cy} {cx+offset*s},{cy+62*s}", "none", light, 5 * s))
        bits.append(path(f"M{cx-82*s},{cy-35*s} Q{cx},{cy-105*s} {cx+82*s},{cy-35*s}", "none", accent, 5 * s))
        return "".join(bits)

    if kind == "rock-habitat":
        bits.append(path(f"M{cx-125*s},{cy+38*s} Q{cx-95*s},{cy-82*s} {cx},{cy-94*s} Q{cx+96*s},{cy-77*s} {cx+125*s},{cy+38*s} Z", primary, dark, 6 * s))
        bits.append(path(f"M{cx-135*s},{cy+45*s} Q{cx},{cy+20*s} {cx+135*s},{cy+45*s}", "none", accent, 8 * s))
        for dx, dy in [(-75, 70), (-8, 62), (62, 72)]:
            bits.append(circle(cx + dx * s, cy + dy * s, 13 * s, dark))
            bits.append(line(cx + (dx - 15) * s, cy + (dy + 10) * s, cx + (dx - 35) * s, cy + (dy + 24) * s, dark, 3 * s))
            bits.append(line(cx + (dx + 15) * s, cy + (dy + 10) * s, cx + (dx + 35) * s, cy + (dy + 24) * s, dark, 3 * s))
        return "".join(bits)

    if kind in {"frog", "frog-cycle"}:
        bits.append(ellipse(cx, cy + 24 * s, 83 * s, 62 * s, primary, dark, 6 * s))
        bits.append(circle(cx - 42 * s, cy - 45 * s, 34 * s, primary, dark, 5 * s))
        bits.append(circle(cx + 42 * s, cy - 45 * s, 34 * s, primary, dark, 5 * s))
        bits.append(circle(cx - 42 * s, cy - 50 * s, 10 * s, dark))
        bits.append(circle(cx + 42 * s, cy - 50 * s, 10 * s, dark))
        for side in (-1, 1):
            bits.append(path(f"M{cx+side*55*s},{cy+48*s} L{cx+side*118*s},{cy+95*s} L{cx+side*82*s},{cy+104*s}", "none", dark, 12 * s))
        bits.append(path(f"M{cx-38*s},{cy+45*s} Q{cx},{cy+68*s} {cx+38*s},{cy+45*s}", "none", light, 5 * s))
        if kind == "frog-cycle":
            bits.append(circle(cx - 105 * s, cy - 78 * s, 13 * s, accent, dark, 3 * s))
            bits.append(path(f"M{cx-92*s},{cy-78*s} Q{cx-62*s},{cy-62*s} {cx-72*s},{cy-40*s}", "none", accent, 8 * s))
        return "".join(bits)

    if kind in {"gecko", "lizard", "chameleon"}:
        body_color = accent if kind == "chameleon" else primary
        bits.append(ellipse(cx - 15 * s, cy, 76 * s, 38 * s, body_color, dark, 5 * s))
        bits.append(circle(cx - 88 * s, cy - 4 * s, 32 * s, accent, dark, 5 * s))
        if kind == "chameleon":
            bits.append(path(f"M{cx+52*s},{cy} Q{cx+135*s},{cy-55*s} {cx+135*s},{cy+25*s} Q{cx+135*s},{cy+72*s} {cx+92*s},{cy+47*s}", "none", body_color, 18 * s))
            bits.append(circle(cx - 98 * s, cy - 12 * s, 10 * s, light, dark, 3 * s))
            for dx, dy in [(-35, -8), (5, 12), (35, -12)]:
                bits.append(circle(cx + dx * s, cy + dy * s, 8 * s, primary))
        else:
            bits.append(path(f"M{cx+52*s},{cy} Q{cx+125*s},{cy-28*s} {cx+142*s},{cy+18*s}", "none", body_color, 17 * s))
        for dx in (-55, 20):
            for side in (-1, 1):
                bits.append(line(cx + dx * s, cy + side * 24 * s, cx + (dx - 25) * s, cy + side * 72 * s, dark, 5 * s))
                if kind == "gecko":
                    bits.append(circle(cx + (dx - 25) * s, cy + side * 72 * s, 8 * s, accent, dark, 2 * s))
        return "".join(bits)

    raise KeyError(f"unknown small-animal subject: {kind}")


def bird_subject(kind: str, cx: float, cy: float, s: float, palette: tuple[str, ...]) -> str:
    _, primary, accent, light, dark = palette
    bits: list[str] = []
    if kind == "feather":
        bits.append(path(f"M{cx-85*s},{cy+80*s} Q{cx-45*s},{cy-105*s} {cx+90*s},{cy-90*s} Q{cx+40*s},{cy+42*s} {cx-85*s},{cy+80*s} Z", light, dark, 5 * s))
        bits.append(line(cx - 68 * s, cy + 66 * s, cx + 72 * s, cy - 74 * s, primary, 7 * s))
        for index in range(6):
            bits.append(line(cx - 38 * s + index * 18 * s, cy + 35 * s - index * 18 * s, cx - 82 * s + index * 11 * s, cy - index * 15 * s, accent, 3 * s))
        return "".join(bits)

    if kind == "migration":
        for index, (x, y, scale) in enumerate([(-72, 28, 1.0), (0, -42, 1.25), (76, 22, .9)]):
            bits.append(path(f"M{cx+(x-48*scale)*s},{cy+y*s} Q{cx+(x-22*scale)*s},{cy+(y-35*scale)*s} {cx+x*s},{cy+y*s} Q{cx+(x+22*scale)*s},{cy+(y-35*scale)*s} {cx+(x+48*scale)*s},{cy+y*s}", "none", primary if index % 2 == 0 else accent, 8 * scale * s))
        bits.append(path(f"M{cx-122*s},{cy+95*s} Q{cx},{cy+60*s} {cx+125*s},{cy+95*s}", "none", dark, 4 * s, True))
        return "".join(bits)

    if kind == "owl":
        return custom_icon("owl", cx, cy, s * .88, palette)

    if kind == "flight":
        bits.append(ellipse(cx, cy + 18 * s, 48 * s, 65 * s, primary, dark, 5 * s))
        bits.append(path(f"M{cx-25*s},{cy} Q{cx-105*s},{cy-95*s} {cx-132*s},{cy+8*s} Q{cx-85*s},{cy+28*s} {cx-20*s},{cy+52*s} Z", light, dark, 5 * s))
        bits.append(path(f"M{cx+25*s},{cy} Q{cx+105*s},{cy-95*s} {cx+132*s},{cy+8*s} Q{cx+85*s},{cy+28*s} {cx+20*s},{cy+52*s} Z", light, dark, 5 * s))
        bits.append(circle(cx, cy - 58 * s, 34 * s, accent, dark, 4 * s))
        bits.append(polygon([(cx + 28 * s, cy - 60 * s), (cx + 65 * s, cy - 48 * s), (cx + 29 * s, cy - 38 * s)], "#F5B83D", dark, 3 * s))
        return "".join(bits)

    if kind == "penguin":
        bits.append(ellipse(cx, cy, 73 * s, 108 * s, dark, dark, 5 * s))
        bits.append(ellipse(cx, cy + 20 * s, 51 * s, 78 * s, light, dark, 3 * s))
        bits.append(circle(cx, cy - 78 * s, 42 * s, dark))
        bits.append(polygon([(cx - 54 * s, cy - 15 * s), (cx - 120 * s, cy + 58 * s), (cx - 62 * s, cy + 42 * s)], primary, dark, 4 * s))
        bits.append(polygon([(cx + 54 * s, cy - 15 * s), (cx + 120 * s, cy + 58 * s), (cx + 62 * s, cy + 42 * s)], primary, dark, 4 * s))
        bits.append(polygon([(cx, cy - 74 * s), (cx + 38 * s, cy - 60 * s), (cx, cy - 48 * s)], "#F5B83D", dark, 3 * s))
        bits.append(ellipse(cx - 35 * s, cy + 102 * s, 36 * s, 13 * s, accent, dark, 3 * s))
        bits.append(ellipse(cx + 35 * s, cy + 102 * s, 36 * s, 13 * s, accent, dark, 3 * s))
        return "".join(bits)

    if kind == "flamingo":
        bits.append(ellipse(cx - 25 * s, cy - 5 * s, 70 * s, 48 * s, "#F06A8A", dark, 5 * s))
        bits.append(path(f"M{cx+28*s},{cy-20*s} Q{cx+105*s},{cy-55*s} {cx+62*s},{cy-112*s} Q{cx+35*s},{cy-145*s} {cx+18*s},{cy-112*s}", "none", "#F06A8A", 20 * s))
        bits.append(circle(cx + 20 * s, cy - 118 * s, 22 * s, "#F06A8A", dark, 4 * s))
        bits.append(polygon([(cx + 34 * s, cy - 124 * s), (cx + 76 * s, cy - 115 * s), (cx + 35 * s, cy - 102 * s)], light, dark, 3 * s))
        bits.append(line(cx - 42 * s, cy + 35 * s, cx - 42 * s, cy + 118 * s, "#F06A8A", 9 * s))
        bits.append(path(f"M{cx+5*s},{cy+35*s} L{cx+2*s},{cy+82*s} L{cx+52*s},{cy+110*s}", "none", "#F06A8A", 9 * s))
        return "".join(bits)

    # Duck and woodpecker share a bird body but retain distinctive posture.
    if kind == "woodpecker":
        bits.append(rect(cx - 90 * s, cy - 125 * s, 42 * s, 250 * s, accent, 10 * s, dark, 5 * s))
        bits.append(ellipse(cx, cy, 50 * s, 78 * s, primary, dark, 5 * s))
        bits.append(circle(cx + 15 * s, cy - 72 * s, 34 * s, light, dark, 4 * s))
        bits.append(polygon([(cx + 43 * s, cy - 78 * s), (cx + 115 * s, cy - 61 * s), (cx + 43 * s, cy - 52 * s)], "#F5B83D", dark, 3 * s))
        bits.append(polygon([(cx - 5 * s, cy - 104 * s), (cx + 20 * s, cy - 132 * s), (cx + 35 * s, cy - 96 * s)], "#F05252", dark, 3 * s))
        bits.append(line(cx - 38 * s, cy - 10 * s, cx - 70 * s, cy + 18 * s, dark, 6 * s))
        return "".join(bits)

    bits.append(ellipse(cx - 15 * s, cy + 18 * s, 82 * s, 58 * s, primary, dark, 5 * s))
    bits.append(circle(cx + 72 * s, cy - 25 * s, 39 * s, accent, dark, 5 * s))
    bits.append(polygon([(cx + 106 * s, cy - 30 * s), (cx + 150 * s, cy - 10 * s), (cx + 106 * s, cy + 4 * s)], "#F5B83D", dark, 3 * s))
    bits.append(path(f"M{cx-52*s},{cy+10*s} Q{cx-5*s},{cy-62*s} {cx+38*s},{cy+25*s} Q{cx-10*s},{cy+58*s} {cx-52*s},{cy+10*s} Z", light, dark, 4 * s))
    bits.append(polygon([(cx - 88 * s, cy + 16 * s), (cx - 142 * s, cy - 28 * s), (cx - 130 * s, cy + 55 * s)], accent, dark, 4 * s))
    if kind == "duck":
        bits.append(path(f"M{cx-145*s},{cy+88*s} Q{cx},{cy+58*s} {cx+150*s},{cy+88*s}", "none", primary, 7 * s))
    return "".join(bits)


def mammal_subject(kind: str, cx: float, cy: float, s: float, palette: tuple[str, ...]) -> str:
    _, primary, accent, light, dark = palette
    bits: list[str] = []
    if kind == "bat":
        bits.append(path(f"M{cx},{cy-20*s} C{cx-55*s},{cy-115*s} {cx-142*s},{cy-75*s} {cx-135*s},{cy+55*s} Q{cx-92*s},{cy+15*s} {cx-58*s},{cy+82*s} Q{cx-22*s},{cy+35*s} {cx},{cy+72*s} Q{cx+22*s},{cy+35*s} {cx+58*s},{cy+82*s} Q{cx+92*s},{cy+15*s} {cx+135*s},{cy+55*s} C{cx+142*s},{cy-75*s} {cx+55*s},{cy-115*s} {cx},{cy-20*s} Z", primary, dark, 6 * s))
        bits.append(ellipse(cx, cy, 35 * s, 68 * s, accent, dark, 5 * s))
        bits.append(polygon([(cx - 25 * s, cy - 52 * s), (cx - 12 * s, cy - 92 * s), (cx, cy - 55 * s)], accent, dark, 4 * s))
        bits.append(polygon([(cx + 25 * s, cy - 52 * s), (cx + 12 * s, cy - 92 * s), (cx, cy - 55 * s)], accent, dark, 4 * s))
        return "".join(bits)

    if kind == "footprints":
        for x, y, turn in [(-62, 35, -1), (12, -52, 1), (72, 55, -1)]:
            bits.append(ellipse(cx + x * s, cy + y * s, 35 * s, 45 * s, primary if turn < 0 else accent, dark, 4 * s))
            for offset in (-25, 0, 25):
                bits.append(circle(cx + (x + offset) * s, cy + (y - 52) * s, 12 * s, primary if turn < 0 else accent, dark, 3 * s))
        return "".join(bits)

    if kind == "group":
        for x, scale, color in [(-75, .66, primary), (0, .9, accent), (80, .6, primary)]:
            bits.append(ellipse(cx + x * s, cy + 18 * s, 58 * scale * s, 38 * scale * s, color, dark, 4 * s))
            bits.append(circle(cx + (x + 48 * scale) * s, cy - 8 * s, 28 * scale * s, color, dark, 4 * s))
            bits.append(line(cx + (x - 28 * scale) * s, cy + 42 * s, cx + (x - 30 * scale) * s, cy + 78 * s, dark, 6 * scale * s))
            bits.append(line(cx + (x + 18 * scale) * s, cy + 42 * s, cx + (x + 20 * scale) * s, cy + 78 * s, dark, 6 * scale * s))
        return "".join(bits)

    if kind == "sloth":
        bits.append(line(cx - 135 * s, cy - 78 * s, cx + 135 * s, cy - 78 * s, dark, 14 * s))
        bits.append(ellipse(cx, cy + 2 * s, 70 * s, 88 * s, primary, dark, 6 * s))
        bits.append(circle(cx + 35 * s, cy + 50 * s, 42 * s, accent, dark, 5 * s))
        bits.append(ellipse(cx + 22 * s, cy + 44 * s, 15 * s, 10 * s, dark))
        bits.append(ellipse(cx + 51 * s, cy + 44 * s, 15 * s, 10 * s, dark))
        for side in (-1, 1):
            bits.append(path(f"M{cx+side*42*s},{cy-34*s} Q{cx+side*82*s},{cy-72*s} {cx+side*72*s},{cy-86*s}", "none", dark, 13 * s))
        return "".join(bits)

    if kind == "kangaroo":
        bits.append(ellipse(cx, cy + 10 * s, 58 * s, 86 * s, primary, dark, 6 * s))
        bits.append(circle(cx + 35 * s, cy - 74 * s, 38 * s, primary, dark, 5 * s))
        bits.append(ellipse(cx + 15 * s, cy - 125 * s, 14 * s, 48 * s, accent, dark, 4 * s))
        bits.append(ellipse(cx + 50 * s, cy - 125 * s, 14 * s, 48 * s, accent, dark, 4 * s))
        bits.append(path(f"M{cx-48*s},{cy+22*s} Q{cx-120*s},{cy+72*s} {cx-150*s},{cy+90*s}", "none", primary, 22 * s))
        bits.append(path(f"M{cx-28*s},{cy+70*s} L{cx-75*s},{cy+122*s} H{cx-32*s} M{cx+28*s},{cy+70*s} L{cx+82*s},{cy+122*s} H{cx+125*s}", "none", dark, 15 * s))
        bits.append(circle(cx + 4 * s, cy + 34 * s, 23 * s, accent, dark, 4 * s))
        return "".join(bits)

    body_color = light if kind in {"polar-bear", "panda"} else primary
    bits.append(ellipse(cx - 22 * s, cy + 22 * s, 105 * s, 62 * s, body_color, dark, 6 * s))
    head_y = -16
    neck_height = 0
    if kind == "giraffe":
        neck_height = 108
        head_y = -118
        bits.append(rect(cx + 48 * s, cy - 102 * s, 48 * s, 122 * s, body_color, 22 * s, dark, 6 * s))
    bits.append(circle(cx + 82 * s, cy + head_y * s, 47 * s, body_color, dark, 6 * s))
    for x in (-82, -15, 48, 82):
        leg_top = 60 if x < 60 else 47
        bits.append(line(cx + x * s, cy + leg_top * s, cx + (x - 4) * s, cy + 125 * s, dark, 12 * s))
    bits.append(path(f"M{cx-120*s},{cy+2*s} Q{cx-160*s},{cy-15*s} {cx-150*s},{cy-58*s}", "none", dark, 10 * s))
    bits.append(circle(cx + 97 * s, cy + (head_y - 8) * s, 6 * s, dark))

    if kind == "elephant":
        bits.append(ellipse(cx + 58 * s, cy - 14 * s, 38 * s, 52 * s, accent, dark, 4 * s))
        bits.append(path(f"M{cx+120*s},{cy-4*s} Q{cx+155*s},{cy+75*s} {cx+112*s},{cy+90*s}", "none", body_color, 20 * s))
    elif kind == "giraffe":
        for dx, dy in [(-65, 5), (-15, 34), (28, -10), (72, -65), (62, -110)]:
            bits.append(circle(cx + dx * s, cy + dy * s, 13 * s, accent))
        for dx in (62, 94):
            bits.append(line(cx + dx * s, cy - 150 * s, cx + dx * s, cy - 172 * s, dark, 5 * s))
    elif kind == "zebra":
        for offset in (-75, -35, 5, 45):
            bits.append(line(cx + offset * s, cy - 22 * s, cx + (offset + 20) * s, cy + 58 * s, dark, 9 * s))
        bits.append(path(f"M{cx+55*s},{cy-52*s} L{cx+110*s},{cy-28*s} M{cx+54*s},{cy-26*s} L{cx+112*s},{cy+2*s}", "none", dark, 7 * s))
    elif kind == "camel":
        bits.append(path(f"M{cx-105*s},{cy-8*s} Q{cx-66*s},{cy-100*s} {cx-22*s},{cy-5*s} Q{cx+26*s},{cy-104*s} {cx+62*s},{cy-8*s}", accent, dark, 5 * s))
    elif kind == "panda":
        bits.append(circle(cx + 52 * s, cy - 50 * s, 20 * s, dark))
        bits.append(circle(cx + 112 * s, cy - 50 * s, 20 * s, dark))
        bits.append(ellipse(cx + 66 * s, cy - 20 * s, 13 * s, 18 * s, dark))
        bits.append(ellipse(cx + 98 * s, cy - 20 * s, 13 * s, 18 * s, dark))
        bits.append(ellipse(cx - 15 * s, cy + 25 * s, 70 * s, 48 * s, dark, "none", 0))
    elif kind == "hedgehog":
        for index in range(9):
            angle = math.pi + index * math.pi / 8
            x = cx - 25 * s + math.cos(angle) * 102 * s
            y = cy + 15 * s + math.sin(angle) * 75 * s
            bits.append(polygon([(x - 12 * s, y + 15 * s), (x, y - 32 * s), (x + 14 * s, y + 15 * s)], accent, dark, 3 * s))
    elif kind == "rabbit":
        bits.append(ellipse(cx + 58 * s, cy - 78 * s, 17 * s, 64 * s, accent, dark, 4 * s))
        bits.append(ellipse(cx + 104 * s, cy - 78 * s, 17 * s, 64 * s, accent, dark, 4 * s))
        bits.append(circle(cx - 132 * s, cy + 22 * s, 23 * s, light, dark, 4 * s))
    elif kind in {"cat", "dog"}:
        if kind == "cat":
            bits.append(polygon([(cx + 42 * s, cy - 45 * s), (cx + 53 * s, cy - 96 * s), (cx + 78 * s, cy - 53 * s)], accent, dark, 4 * s))
            bits.append(polygon([(cx + 86 * s, cy - 53 * s), (cx + 116 * s, cy - 96 * s), (cx + 122 * s, cy - 42 * s)], accent, dark, 4 * s))
            for offset in (-12, 8):
                bits.append(line(cx + 105 * s, cy + offset * s, cx + 152 * s, cy + (offset - 8) * s, dark, 3 * s))
        else:
            bits.append(ellipse(cx + 51 * s, cy - 47 * s, 20 * s, 43 * s, accent, dark, 4 * s))
            bits.append(ellipse(cx + 116 * s, cy - 47 * s, 20 * s, 43 * s, accent, dark, 4 * s))
    elif kind == "cow":
        bits.append(path(f"M{cx+45*s},{cy-42*s} L{cx+22*s},{cy-78*s} M{cx+118*s},{cy-42*s} L{cx+142*s},{cy-78*s}", "none", dark, 6 * s))
        for dx, dy in [(-72, -8), (-12, 34), (33, -15), (75, 22)]:
            bits.append(ellipse(cx + dx * s, cy + dy * s, 22 * s, 14 * s, accent))
    elif kind == "horse":
        bits.append(path(f"M{cx+45*s},{cy-52*s} Q{cx+75*s},{cy-95*s} {cx+116*s},{cy-52*s}", "none", accent, 11 * s))
        bits.append(polygon([(cx + 105 * s, cy - 52 * s), (cx + 120 * s, cy - 95 * s), (cx + 133 * s, cy - 52 * s)], accent, dark, 4 * s))
    return "".join(bits)


def aquatic_subject(kind: str, cx: float, cy: float, s: float, palette: tuple[str, ...]) -> str:
    _, primary, accent, light, dark = palette
    bits: list[str] = []
    if kind == "octopus":
        bits.append(path(f"M{cx-75*s},{cy+5*s} Q{cx-60*s},{cy-105*s} {cx},{cy-112*s} Q{cx+60*s},{cy-105*s} {cx+75*s},{cy+5*s} Q{cx+42*s},{cy+45*s} {cx},{cy+45*s} Q{cx-42*s},{cy+45*s} {cx-75*s},{cy+5*s} Z", primary, dark, 6 * s))
        for index in range(8):
            x = cx + (-70 + index * 20) * s
            bits.append(path(f"M{x},{cy+25*s} Q{x+(20 if index%2 else -20)*s},{cy+85*s} {x+(8 if index%2 else -8)*s},{cy+125*s}", "none", accent if index % 2 else primary, 12 * s))
        bits.append(circle(cx - 28 * s, cy - 18 * s, 9 * s, light, dark, 3 * s))
        bits.append(circle(cx + 28 * s, cy - 18 * s, 9 * s, light, dark, 3 * s))
        return "".join(bits)

    if kind == "whale":
        bits.append(ellipse(cx - 10 * s, cy + 15 * s, 115 * s, 63 * s, primary, dark, 6 * s))
        bits.append(path(f"M{cx-112*s},{cy+15*s} Q{cx-155*s},{cy-38*s} {cx-145*s},{cy+42*s} Q{cx-150*s},{cy+92*s} {cx-105*s},{cy+48*s}", accent, dark, 4 * s))
        bits.append(path(f"M{cx+58*s},{cy+45*s} Q{cx+82*s},{cy+92*s} {cx+98*s},{cy+43*s}", accent, dark, 4 * s))
        bits.append(circle(cx + 65 * s, cy, 7 * s, dark))
        bits.append(path(f"M{cx+25*s},{cy-48*s} Q{cx+8*s},{cy-95*s} {cx-15*s},{cy-68*s} M{cx+25*s},{cy-48*s} Q{cx+45*s},{cy-95*s} {cx+67*s},{cy-67*s}", "none", light, 7 * s))
        return "".join(bits)

    body = path(f"M{cx-105*s},{cy+5*s} Q{cx-20*s},{cy-70*s} {cx+105*s},{cy-8*s} Q{cx+12*s},{cy+75*s} {cx-105*s},{cy+5*s} Z", primary, dark, 6 * s)
    bits.append(body)
    bits.append(polygon([(cx - 96 * s, cy + 4 * s), (cx - 150 * s, cy - 55 * s), (cx - 145 * s, cy + 67 * s)], accent, dark, 5 * s))
    bits.append(circle(cx + 65 * s, cy - 15 * s, 7 * s, dark))
    if kind == "dolphin":
        bits.append(path(f"M{cx+85*s},{cy-15*s} Q{cx+135*s},{cy-26*s} {cx+145*s},{cy-5*s}", "none", primary, 15 * s))
        bits.append(polygon([(cx - 5 * s, cy - 50 * s), (cx + 28 * s, cy - 105 * s), (cx + 50 * s, cy - 40 * s)], accent, dark, 4 * s))
    elif kind == "shark":
        bits.append(polygon([(cx - 5 * s, cy - 50 * s), (cx + 22 * s, cy - 118 * s), (cx + 55 * s, cy - 40 * s)], accent, dark, 5 * s))
        bits.append(path(f"M{cx+38*s},{cy+20*s} Q{cx+72*s},{cy+38*s} {cx+92*s},{cy+16*s}", "none", light, 4 * s))
        for offset in (48, 63, 78):
            bits.append(line(cx + offset * s, cy + 18 * s, cx + (offset + 5) * s, cy + 35 * s, light, 3 * s))
    if kind == "fish-bladder":
        bits.append(ellipse(cx, cy + 5 * s, 42 * s, 20 * s, light, dark, 4 * s))
    return "".join(bits)


def geo_subject(kind: str, cx: float, cy: float, s: float, palette: tuple[str, ...]) -> str:
    _, primary, accent, light, dark = palette
    bits: list[str] = []

    if kind == "continents":
        bits.append(circle(cx, cy, 92 * s, primary, dark, 6 * s))
        bits.append(path(f"M{cx-68*s},{cy-45*s} Q{cx-28*s},{cy-92*s} {cx+5*s},{cy-48*s} L{cx-12*s},{cy-5*s} L{cx-72*s},{cy+12*s} Z M{cx+18*s},{cy+8*s} Q{cx+88*s},{cy-18*s} {cx+70*s},{cy+52*s} L{cx+16*s},{cy+70*s} L{cx-8*s},{cy+30*s} Z", accent, dark, 3 * s))
        bits.append(ellipse(cx, cy, 92 * s, 35 * s, "none", light, 4 * s))
        return "".join(bits)

    if kind == "earthquake":
        bits.append(path(f"M{cx-135*s},{cy+72*s} L{cx-55*s},{cy+62*s} L{cx-28*s},{cy+8*s} L{cx+4*s},{cy+75*s} L{cx+45*s},{cy+48*s} L{cx+135*s},{cy+65*s}", "none", dark, 10 * s))
        bits.append(rect(cx - 55 * s, cy - 50 * s, 110 * s, 105 * s, light, 6 * s, dark, 5 * s))
        bits.append(polygon([(cx - 72 * s, cy - 50 * s), (cx, cy - 112 * s), (cx + 72 * s, cy - 50 * s)], accent, dark, 5 * s))
        for offset in (-110, 110):
            bits.append(path(f"M{cx+offset*s},{cy-35*s} q{-18*s},{20*s} 0,40 q{18*s},{20*s} 0,40", "none", primary, 5 * s))
        return "".join(bits)

    if kind in {"volcano", "mountain", "valley", "glacier", "cave"}:
        if kind == "valley":
            bits.append(polygon([(cx - 140 * s, cy + 95 * s), (cx - 62 * s, cy - 85 * s), (cx + 5 * s, cy + 95 * s)], primary, dark, 5 * s))
            bits.append(polygon([(cx - 5 * s, cy + 95 * s), (cx + 70 * s, cy - 70 * s), (cx + 140 * s, cy + 95 * s)], accent, dark, 5 * s))
            bits.append(path(f"M{cx},{cy+25*s} Q{cx-18*s},{cy+65*s} {cx+15*s},{cy+105*s}", "none", light, 11 * s))
            return "".join(bits)
        bits.append(polygon([(cx - 135 * s, cy + 98 * s), (cx, cy - 105 * s), (cx + 135 * s, cy + 98 * s)], primary, dark, 6 * s))
        if kind == "volcano":
            bits.append(path(f"M{cx-23*s},{cy-70*s} Q{cx},{cy-110*s} {cx+25*s},{cy-70*s} M{cx},{cy-68*s} L{cx+42*s},{cy+88*s}", "none", accent, 14 * s))
            bits.append(circle(cx - 35 * s, cy - 125 * s, 18 * s, accent))
            bits.append(circle(cx + 20 * s, cy - 145 * s, 24 * s, accent))
        elif kind == "glacier":
            bits.append(path(f"M{cx-28*s},{cy-62*s} L{cx-82*s},{cy+90*s} H{cx+80*s} L{cx+26*s},{cy-62*s} Z", light, primary, 5 * s))
            bits.append(path(f"M{cx-68*s},{cy+70*s} Q{cx},{cy+40*s} {cx+68*s},{cy+78*s}", "none", accent, 6 * s, True))
        elif kind == "cave":
            bits.append(path(f"M{cx-82*s},{cy+98*s} Q{cx-60*s},{cy-22*s} {cx},{cy-38*s} Q{cx+62*s},{cy-22*s} {cx+82*s},{cy+98*s} Z", dark, dark, 4 * s))
            for x in (-35, 5, 40):
                bits.append(polygon([(cx + x * s, cy - 32 * s), (cx + (x + 12) * s, cy + 18 * s), (cx + (x + 25) * s, cy - 30 * s)], light))
        else:
            bits.append(polygon([(cx - 38 * s, cy - 47 * s), (cx, cy - 105 * s), (cx + 42 * s, cy - 42 * s), (cx + 12 * s, cy - 58 * s)], light))
        return "".join(bits)

    if kind == "rock-mineral":
        bits.append(path(f"M{cx-118*s},{cy+48*s} Q{cx-105*s},{cy-55*s} {cx-28*s},{cy-72*s} Q{cx+55*s},{cy-60*s} {cx+70*s},{cy+40*s} Q{cx+25*s},{cy+92*s} {cx-72*s},{cy+85*s} Z", primary, dark, 6 * s))
        for x, y, color in [(-45, -8, accent), (5, 20, light), (48, -22, accent)]:
            bits.append(polygon([(cx + (x-18)*s, cy + (y+25)*s), (cx + x*s, cy + (y-30)*s), (cx + (x+18)*s, cy + (y+25)*s)], color, dark, 3 * s))
        return "".join(bits)

    if kind == "soil":
        for index, color in enumerate((accent, primary, dark)):
            bits.append(rect(cx - 125 * s, cy + (-65 + index * 45) * s, 250 * s, 45 * s, color, 2 * s, dark, 2 * s))
        bits.append(path(f"M{cx-65*s},{cy-65*s} V{cy+60*s} M{cx-65*s},{cy-5*s} L{cx-105*s},{cy+25*s} M{cx-65*s},{cy+20*s} L{cx-20*s},{cy+52*s}", "none", light, 6 * s))
        bits.append(ellipse(cx + 65 * s, cy + 15 * s, 36 * s, 13 * s, "#F06A72", dark, 3 * s))
        return "".join(bits)

    if kind == "fossil":
        bits.append(ellipse(cx, cy, 125 * s, 88 * s, primary, dark, 6 * s))
        bits.append(circle(cx - 58 * s, cy - 10 * s, 20 * s, light, dark, 4 * s))
        bits.append(circle(cx + 58 * s, cy + 22 * s, 20 * s, light, dark, 4 * s))
        bits.append(line(cx - 43 * s, cy + 2 * s, cx + 43 * s, cy + 12 * s, light, 22 * s))
        bits.append(line(cx - 43 * s, cy + 2 * s, cx + 43 * s, cy + 12 * s, dark, 4 * s))
        return "".join(bits)

    if kind in {"map", "projection"}:
        bits.append(polygon([(cx - 125 * s, cy - 80 * s), (cx - 42 * s, cy - 55 * s), (cx + 42 * s, cy - 82 * s), (cx + 125 * s, cy - 52 * s), (cx + 115 * s, cy + 88 * s), (cx + 35 * s, cy + 60 * s), (cx - 45 * s, cy + 90 * s), (cx - 125 * s, cy + 62 * s)], light, dark, 5 * s))
        bits.append(line(cx - 42 * s, cy - 52 * s, cx - 45 * s, cy + 88 * s, primary, 4 * s))
        bits.append(line(cx + 42 * s, cy - 78 * s, cx + 35 * s, cy + 58 * s, accent, 4 * s))
        bits.append(path(f"M{cx-92*s},{cy+35*s} Q{cx-30*s},{cy-15*s} {cx+12*s},{cy+20*s} T{cx+92*s},{cy-18*s}", "none", primary, 8 * s))
        if kind == "projection":
            bits.append(circle(cx - 105 * s, cy - 112 * s, 34 * s, primary, dark, 4 * s))
            bits.append(line(cx - 62 * s, cy - 102 * s, cx - 22 * s, cy - 78 * s, accent, 5 * s, True))
        return "".join(bits)

    if kind == "compass":
        bits.append(circle(cx, cy, 105 * s, light, dark, 6 * s))
        bits.append(circle(cx, cy, 14 * s, dark))
        bits.append(polygon([(cx, cy - 88 * s), (cx - 23 * s, cy + 10 * s), (cx, cy), (cx + 23 * s, cy + 10 * s)], accent, dark, 4 * s))
        bits.append(polygon([(cx, cy + 88 * s), (cx - 23 * s, cy - 10 * s), (cx, cy), (cx + 23 * s, cy - 10 * s)], primary, dark, 4 * s))
        return "".join(bits)

    if kind == "gps":
        bits.append(rect(cx - 35 * s, cy - 30 * s, 70 * s, 60 * s, accent, 6 * s, dark, 4 * s))
        bits.append(rect(cx - 118 * s, cy - 45 * s, 72 * s, 90 * s, primary, 4 * s, dark, 4 * s))
        bits.append(rect(cx + 46 * s, cy - 45 * s, 72 * s, 90 * s, primary, 4 * s, dark, 4 * s))
        for radius in (75, 105):
            bits.append(path(f"M{cx-radius*s},{cy+90*s} Q{cx},{cy+(20-radius*.55)*s} {cx+radius*s},{cy+90*s}", "none", accent, 4 * s))
        return "".join(bits)

    if kind == "latitude":
        return custom_icon("latitude", cx, cy, s, palette)
    if kind == "border":
        return custom_icon("border", cx, cy, s, palette)

    if kind in {"equator", "poles", "timezones"}:
        bits.append(circle(cx, cy, 88 * s, primary, dark, 6 * s))
        bits.append(path(f"M{cx-62*s},{cy-38*s} Q{cx-18*s},{cy-82*s} {cx+8*s},{cy-42*s} L{cx-5*s},{cy} L{cx-68*s},{cy+8*s} Z", accent, dark, 3 * s))
        if kind == "equator":
            bits.append(line(cx - 92 * s, cy, cx + 92 * s, cy, "#F06A57", 9 * s))
            bits.append(circle(cx + 125 * s, cy - 75 * s, 30 * s, "#F5B83D", dark, 3 * s))
        elif kind == "poles":
            bits.append(path(f"M{cx-52*s},{cy-60*s} Q{cx},{cy-95*s} {cx+52*s},{cy-60*s} M{cx-54*s},{cy+62*s} Q{cx},{cy+95*s} {cx+54*s},{cy+62*s}", "none", light, 20 * s))
        else:
            for offset in (-30, 30):
                bits.append(ellipse(cx + offset * s, cy, 22 * s, 82 * s, "none", light, 4 * s))
            bits.append(circle(cx + 118 * s, cy - 40 * s, 30 * s, light, dark, 4 * s))
            bits.append(line(cx + 118 * s, cy - 40 * s, cx + 118 * s, cy - 60 * s, primary, 4 * s))
        return "".join(bits)

    if kind in {"island", "peninsula", "lake-sea", "delta"}:
        bits.append(rect(cx - 140 * s, cy + 18 * s, 280 * s, 88 * s, primary, 22 * s, "none", 0))
        if kind == "island":
            bits.append(ellipse(cx, cy + 10 * s, 82 * s, 40 * s, accent, dark, 4 * s))
        elif kind == "peninsula":
            bits.append(path(f"M{cx-130*s},{cy-92*s} H{cx+15*s} Q{cx+100*s},{cy-40*s} {cx+50*s},{cy+45*s} H{cx-130*s} Z", accent, dark, 4 * s))
        elif kind == "lake-sea":
            bits.append(path(f"M{cx-130*s},{cy-70*s} Q{cx-45*s},{cy-5*s} {cx+10*s},{cy-62*s} L{cx+120*s},{cy-82*s}", "none", accent, 30 * s))
            bits.append(ellipse(cx + 65 * s, cy + 18 * s, 40 * s, 25 * s, light, dark, 3 * s))
        else:
            bits.append(path(f"M{cx-15*s},{cy-105*s} V{cy-5*s} M{cx-15*s},{cy-5*s} L{cx-105*s},{cy+88*s} M{cx-15*s},{cy-5*s} L{cx},{cy+100*s} M{cx-15*s},{cy-5*s} L{cx+105*s},{cy+88*s}", "none", light, 13 * s))
        return "".join(bits)

    if kind in {"desert", "rainforest", "grassland"}:
        if kind == "desert":
            bits.append(path(f"M{cx-145*s},{cy+65*s} Q{cx-65*s},{cy-35*s} {cx+15*s},{cy+65*s} T{cx+150*s},{cy+65*s} V{cy+115*s} H{cx-145*s} Z", accent, dark, 4 * s))
            bits.append(path(f"M{cx+55*s},{cy+40*s} V{cy-60*s} M{cx+55*s},{cy-22*s} L{cx+90*s},{cy-48*s} M{cx+55*s},{cy+5*s} L{cx+22*s},{cy-18*s}", "none", primary, 13 * s))
        elif kind == "rainforest":
            for x, height, color in [(-90, 80, primary), (-30, 125, accent), (35, 105, primary), (95, 72, accent)]:
                bits.append(rect(cx + (x-7)*s, cy + (75-height)*s, 14*s, height*s, dark, 5*s))
                bits.append(circle(cx + x*s, cy + (55-height)*s, 45*s, color, dark, 4*s))
        else:
            for x in range(-120, 121, 24):
                bits.append(path(f"M{cx+x*s},{cy+90*s} Q{cx+(x-8)*s},{cy+35*s} {cx+(x+4)*s},{cy+8*s}", "none", primary if x % 48 else accent, 5*s))
            bits.append(circle(cx+90*s, cy-18*s, 35*s, accent, dark, 4*s))
            bits.append(rect(cx+84*s, cy+12*s, 12*s, 75*s, dark, 5*s))
        return "".join(bits)

    raise KeyError(f"unknown geography subject: {kind}")


def ocean_subject(kind: str, cx: float, cy: float, s: float, palette: tuple[str, ...]) -> str:
    _, primary, accent, light, dark = palette
    bits: list[str] = []
    if kind == "octopus":
        return aquatic_subject("octopus", cx, cy, s, palette)
    if kind == "shell":
        return custom_icon("shell", cx, cy, s, palette)
    if kind == "plastic":
        return custom_icon("plastic-bottle", cx, cy, s, palette)
    if kind == "jellyfish":
        bits.append(path(f"M{cx-82*s},{cy+10*s} Q{cx-65*s},{cy-105*s} {cx},{cy-112*s} Q{cx+65*s},{cy-105*s} {cx+82*s},{cy+10*s} Z", primary, dark, 6*s))
        for x in (-55, -20, 20, 55):
            bits.append(path(f"M{cx+x*s},{cy+10*s} Q{cx+(x-22)*s},{cy+70*s} {cx+x*s},{cy+125*s}", "none", accent, 8*s))
        return "".join(bits)
    if kind == "starfish":
        points=[]
        for index in range(10):
            angle=-math.pi/2+index*math.pi/5
            radius=112 if index%2==0 else 46
            points.append((cx+math.cos(angle)*radius*s,cy+math.sin(angle)*radius*s))
        bits.append(polygon(points,accent,dark,6*s))
        return "".join(bits)
    if kind == "sea-cucumber":
        bits.append(ellipse(cx,cy,118*s,48*s,primary,dark,6*s))
        for x in (-75,-38,0,38,75):
            bits.append(circle(cx+x*s,cy-28*s,8*s,accent,dark,2*s))
            bits.append(line(cx+x*s,cy+38*s,cx+(x+6)*s,cy+62*s,dark,3*s))
        return "".join(bits)
    if kind in {"crab","hermit-crab"}:
        if kind=="hermit-crab":
            bits.append(circle(cx-45*s,cy-12*s,70*s,accent,dark,6*s))
            bits.append(path(f"M{cx-45*s},{cy-12*s} q{38*s},{-32*s} 42,12 q{2*s},35 {-33*s},33", "none", light,5*s))
        bits.append(ellipse(cx+25*s,cy+18*s,68*s,45*s,primary,dark,5*s))
        for side in (-1,1):
            for offset in (-20,8,35):
                bits.append(line(cx+(25+side*46)*s,cy+offset*s,cx+(25+side*112)*s,cy+(offset+side*18)*s,dark,5*s))
            bits.append(path(f"M{cx+(25+side*45)*s},{cy-12*s} Q{cx+(25+side*92)*s},{cy-72*s} {cx+(25+side*115)*s},{cy-40*s}","none",dark,7*s))
        return "".join(bits)
    if kind == "squid":
        bits.append(path(f"M{cx},{cy-125*s} Q{cx-76*s},{cy-35*s} {cx-55*s},{cy+38*s} Q{cx},{cy+85*s} {cx+55*s},{cy+38*s} Q{cx+76*s},{cy-35*s} {cx},{cy-125*s} Z",primary,dark,6*s))
        for x in (-42,-14,14,42): bits.append(path(f"M{cx+x*s},{cy+48*s} Q{cx+(x-20)*s},{cy+92*s} {cx+x*s},{cy+125*s}","none",accent,8*s))
        bits.append(circle(cx-22*s,cy-5*s,8*s,light,dark,3*s)); bits.append(circle(cx+22*s,cy-5*s,8*s,light,dark,3*s))
        return "".join(bits)
    if kind == "whale-shark":
        bits.append(ellipse(cx,cy,128*s,58*s,primary,dark,6*s))
        bits.append(polygon([(cx-120*s,cy),(cx-170*s,cy-62*s),(cx-160*s,cy+64*s)],accent,dark,5*s))
        bits.append(rect(cx+92*s,cy-35*s,45*s,70*s,light,15*s,dark,4*s))
        for dx,dy in [(-70,-18),(-35,15),(0,-22),(38,15),(72,-12)]: bits.append(circle(cx+dx*s,cy+dy*s,6*s,light))
        return "".join(bits)
    if kind == "seahorse":
        bits.append(path(f"M{cx-10*s},{cy-112*s} Q{cx+72*s},{cy-90*s} {cx+35*s},{cy-28*s} Q{cx-5*s},{cy+15*s} {cx+45*s},{cy+55*s} Q{cx+82*s},{cy+88*s} {cx+32*s},{cy+105*s} Q{cx-28*s},{cy+110*s} {cx-42*s},{cy+45*s} Q{cx-70*s},{cy-22*s} {cx-10*s},{cy-112*s} Z",primary,dark,6*s))
        bits.append(path(f"M{cx-35*s},{cy-45*s} L{cx-105*s},{cy-78*s} L{cx-42*s},{cy-5*s} Z",accent,dark,4*s))
        return "".join(bits)
    if kind in {"clownfish","bioluminescent-fish"}:
        bits.append(aquatic_subject("fish",cx,cy,s,palette))
        if kind=="clownfish":
            for x in (-42,18,66): bits.append(line(cx+x*s,cy-44*s,cx+(x+15)*s,cy+44*s,light,9*s))
            for x in (-110,-70,-30,10,50,90,130): bits.append(path(f"M{cx+x*s},{cy+100*s} Q{cx+(x-15)*s},{cy+42*s} {cx+(x+4)*s},{cy+18*s}","none",accent,5*s))
        else:
            for x in (-55,-12,34,70): bits.append(circle(cx+x*s,cy+8*s,8*s,"#FFE66D"))
            bits.append(ellipse(cx,cy,145*s,82*s,"none","#FFE66D",4*s))
        return "".join(bits)
    if kind == "turtle":
        bits.append(ellipse(cx,cy,82*s,58*s,primary,dark,6*s)); bits.append(circle(cx+98*s,cy-5*s,25*s,accent,dark,4*s))
        for dx,dy in [(-58,-52),(-58,52),(52,-52),(52,52)]: bits.append(ellipse(cx+dx*s,cy+dy*s,35*s,14*s,accent,dark,3*s))
        bits.append(path(f"M{cx-48*s},{cy-20*s} L{cx},{cy-48*s} L{cx+48*s},{cy-20*s} L{cx+30*s},{cy+35*s} L{cx-30*s},{cy+35*s} Z",light,dark,3*s))
        return "".join(bits)
    if kind == "mangrove":
        bits.append(rect(cx-18*s,cy-72*s,36*s,105*s,accent,10*s,dark,4*s)); bits.append(circle(cx-62*s,cy-85*s,52*s,primary,dark,4*s)); bits.append(circle(cx+38*s,cy-105*s,62*s,primary,dark,4*s)); bits.append(circle(cx+90*s,cy-62*s,44*s,primary,dark,4*s))
        for x in (-45,-12,18,48): bits.append(line(cx+x*s,cy+12*s,cx+(x-35)*s,cy+110*s,dark,6*s)); bits.append(line(cx+x*s,cy+12*s,cx+(x+35)*s,cy+110*s,dark,6*s))
        return "".join(bits)
    if kind in {"seagrass","seaweed","algae"}:
        for x in range(-100,101,28):
            height=75+(x%3)*12
            bits.append(path(f"M{cx+x*s},{cy+100*s} Q{cx+(x-30)*s},{cy+(100-height/2)*s} {cx+(x+8)*s},{cy+(100-height)*s}","none",primary if x%2 else accent,9*s))
        if kind=="algae":
            for x,y in [(-65,-60),(-15,-88),(45,-55),(78,-95)]: bits.append(circle(cx+x*s,cy+y*s,15*s,primary,light,3*s))
        return "".join(bits)
    if kind == "plankton":
        for index in range(14):
            angle=index*math.tau/14; radius=45+(index%3)*27; x=cx+math.cos(angle)*radius*s; y=cy+math.sin(angle)*radius*.75*s
            bits.append(circle(x,y,(8+index%4*3)*s,primary if index%2 else accent,dark,2*s)); bits.append(line(x,y,x+math.cos(angle)*24*s,y+math.sin(angle)*24*s,dark,2*s))
        return "".join(bits)
    if kind in {"sand","shore-drift"}:
        for row in range(5):
            for col in range(8): bits.append(circle(cx+(-105+col*30+(row%2)*10)*s,cy+(-48+row*28)*s,(6+(row+col)%3)*s,accent if (row+col)%2 else primary))
        if kind=="shore-drift": bits.append(path(f"M{cx-130*s},{cy-92*s} Q{cx},{cy-135*s} {cx+130*s},{cy-92*s}","none",light,12*s,True))
        return "".join(bits)
    raise KeyError(f"unknown ocean subject: {kind}")


def space_subject(kind: str, cx: float, cy: float, s: float, palette: tuple[str, ...]) -> str:
    _, primary, accent, light, dark = palette
    bits: list[str] = []
    if kind in {"universe","galaxy"}:
        bits.append(custom_icon("galaxy",cx,cy,s*.82,palette))
        for x,y in [(-115,-88),(105,-72),(-105,82),(112,65)]: bits.append(circle(cx+x*s,cy+y*s,7*s,light,accent,2*s))
        return "".join(bits)
    if kind in {"solar-system","planets"}:
        bits.append(circle(cx,cy,34*s,accent,dark,4*s))
        for index,(rx,ry,color) in enumerate([(70,34,primary),(105,54,accent),(140,75,light)]):
            bits.append(ellipse(cx,cy,rx*s,ry*s,"none",color,4*s)); angle=(index+1)*1.4; bits.append(circle(cx+math.cos(angle)*rx*s,cy+math.sin(angle)*ry*s,(10+index*3)*s,color,dark,3*s))
        return "".join(bits)
    if kind == "scale":
        bits.append(circle(cx-42*s,cy,82*s,accent,dark,6*s)); bits.append(circle(cx+102*s,cy+48*s,18*s,primary,dark,3*s)); bits.append(line(cx+40*s,cy-72*s,cx+110*s,cy+22*s,dark,4*s,True))
        return "".join(bits)
    if kind == "star":
        for radius,color in [(78,accent),(54,light),(30,primary)]: bits.append(circle(cx,cy,radius*s,color,dark if radius==78 else "none",5*s if radius==78 else 0))
        for index in range(12):
            angle=index*math.tau/12; bits.append(line(cx+math.cos(angle)*90*s,cy+math.sin(angle)*90*s,cx+math.cos(angle)*125*s,cy+math.sin(angle)*125*s,accent,7*s))
        return "".join(bits)
    if kind == "asteroids": return custom_icon("asteroids",cx,cy,s,palette)
    if kind == "comet":
        bits.append(path(f"M{cx-145*s},{cy+70*s} Q{cx-25*s},{cy-95*s} {cx+82*s},{cy-35*s}","none",accent,26*s)); bits.append(path(f"M{cx-135*s},{cy+92*s} Q{cx-15*s},{cy-48*s} {cx+88*s},{cy-32*s}","none",light,10*s)); bits.append(circle(cx+95*s,cy-35*s,43*s,primary,dark,5*s)); return "".join(bits)
    if kind == "meteor":
        bits.append(path(f"M{cx-135*s},{cy-92*s} L{cx+38*s},{cy+38*s}","none",accent,25*s)); bits.append(path(f"M{cx-120*s},{cy-110*s} L{cx+50*s},{cy+18*s}","none",light,8*s)); bits.append(circle(cx+65*s,cy+52*s,38*s,primary,dark,5*s)); return "".join(bits)
    if kind == "crater":
        bits.append(ellipse(cx,cy+22*s,125*s,58*s,primary,dark,7*s)); bits.append(ellipse(cx,cy+18*s,72*s,30*s,dark,accent,5*s)); bits.append(circle(cx-78*s,cy-60*s,20*s,primary,dark,4*s)); bits.append(circle(cx+72*s,cy-75*s,14*s,primary,dark,3*s)); return "".join(bits)
    if kind in {"solar-eclipse","lunar-eclipse"}:
        bits.append(circle(cx-98*s,cy,52*s,accent,dark,5*s)); bits.append(circle(cx,cy,40*s,dark,dark,4*s)); bits.append(circle(cx+102*s,cy,50*s,"#F06A57" if kind=="lunar-eclipse" else primary,dark,5*s)); bits.append(line(cx-48*s,cy,cx-42*s,cy,dark,12*s,True)); bits.append(line(cx+42*s,cy,cx+52*s,cy,dark,12*s,True)); return "".join(bits)
    if kind == "orbit":
        bits.append(ellipse(cx,cy,135*s,78*s,"none",primary,6*s)); bits.append(circle(cx,cy,55*s,primary,dark,5*s)); bits.append(circle(cx+125*s,cy-22*s,24*s,light,dark,4*s)); bits.append(path(f"M{cx-125*s},{cy+30*s} A{135*s},{78*s} 0 0 0 {cx+45*s},{cy+72*s}","none",accent,6*s,True)); return "".join(bits)
    if kind == "rocket":
        bits.append(path(f"M{cx},{cy-125*s} C{cx-65*s},{cy-55*s} {cx-58*s},{cy+55*s} {cx},{cy+92*s} C{cx+58*s},{cy+55*s} {cx+65*s},{cy-55*s} {cx},{cy-125*s} Z",light,dark,6*s)); bits.append(circle(cx,cy-25*s,27*s,primary,dark,4*s)); bits.append(polygon([(cx-25*s,cy+82*s),(cx,cy+145*s),(cx+25*s,cy+82*s)],accent,dark,3*s)); return "".join(bits)
    if kind in {"astronaut","suit"}:
        bits.append(circle(cx,cy-72*s,52*s,light,dark,7*s)); bits.append(ellipse(cx,cy-72*s,35*s,25*s,primary,dark,4*s)); bits.append(rect(cx-62*s,cy-20*s,124*s,110*s,light,22*s,dark,6*s)); bits.append(rect(cx-28*s,cy+5*s,56*s,38*s,accent,6*s,dark,3*s)); bits.append(line(cx-48*s,cy+80*s,cx-65*s,cy+130*s,dark,14*s)); bits.append(line(cx+48*s,cy+80*s,cx+65*s,cy+130*s,dark,14*s));
        if kind=="suit": bits.append(rect(cx+62*s,cy-5*s,52*s,75*s,primary,8*s,dark,4*s))
        return "".join(bits)
    if kind == "satellite":
        bits.append(rect(cx-38*s,cy-32*s,76*s,64*s,accent,7*s,dark,5*s)); bits.append(rect(cx-135*s,cy-48*s,82*s,96*s,primary,4*s,dark,5*s)); bits.append(rect(cx+53*s,cy-48*s,82*s,96*s,primary,4*s,dark,5*s)); bits.append(path(f"M{cx},{cy+32*s} Q{cx+65*s},{cy+75*s} {cx+85*s},{cy+125*s}","none",light,7*s)); return "".join(bits)
    if kind == "telescope":
        bits.append(path(f"M{cx-110*s},{cy-55*s} L{cx+85*s},{cy-105*s} L{cx+112*s},{cy-38*s} L{cx-85*s},{cy+14*s} Z",primary,dark,6*s)); bits.append(circle(cx+100*s,cy-72*s,32*s,light,dark,5*s)); bits.append(line(cx,cy-8*s,cx-65*s,cy+120*s,dark,12*s)); bits.append(line(cx+22*s,cy-12*s,cx+88*s,cy+120*s,dark,12*s)); return "".join(bits)
    if kind == "black-hole":
        bits.append(ellipse(cx,cy,145*s,55*s,"none",accent,22*s)); bits.append(ellipse(cx,cy,105*s,40*s,"none",primary,14*s)); bits.append(circle(cx,cy,50*s,dark)); bits.append(path(f"M{cx-140*s},{cy-75*s} Q{cx},{cy-145*s} {cx+140*s},{cy-75*s}","none",light,5*s,True)); return "".join(bits)
    if kind == "vacuum":
        bits.append(circle(cx-75*s,cy-45*s,8*s,light,accent,2*s)); bits.append(circle(cx+88*s,cy+32*s,6*s,light,accent,2*s)); bits.append(circle(cx+10*s,cy+85*s,5*s,light,accent,2*s)); bits.append(path(f"M{cx-120*s},{cy} Q{cx},{cy-85*s} {cx+120*s},{cy}","none",primary,4*s,False,"9 12")); return "".join(bits)
    if kind == "moon-gravity":
        bits.append(circle(cx,cy,85*s,light,dark,6*s)); bits.append(circle(cx-30*s,cy-22*s,18*s,primary)); bits.append(circle(cx+34*s,cy+30*s,23*s,primary)); bits.append(circle(cx-120*s,cy-80*s,18*s,accent,dark,3*s)); bits.append(path(f"M{cx-110*s},{cy-55*s} Q{cx-40*s},{cy-5*s} {cx-15*s},{cy+80*s}","none",accent,5*s,True)); return "".join(bits)

    # Planet portraits share a sphere but differ by atmosphere, surface, rings,
    # axial tilt, storms, or color.  These are qualitative, not to scale.
    planet_color={"planet":primary,"mercury":"#A9A9A9","venus":"#E3A83B","earth-water":"#2E78C7","mars":"#D76542","jupiter":"#D7A46A","saturn":"#E5C56E","uranus":"#76C7D6","neptune":"#4169C1","pluto":"#B9A58D"}.get(kind,primary)
    bits.append(circle(cx,cy,86*s,planet_color,dark,6*s))
    if kind=="mercury":
        for x,y,r in [(-35,-28,14),(30,18,20),(18,-45,9),(-18,45,12)]: bits.append(circle(cx+x*s,cy+y*s,r*s,light,dark,2*s))
    elif kind=="venus":
        for offset in (-25,10,45): bits.append(path(f"M{cx-65*s},{cy+offset*s} Q{cx},{cy+(offset-24)*s} {cx+65*s},{cy+offset*s}","none",light,7*s)); bits.append(circle(cx,cy,108*s,"none",accent,5*s))
    elif kind=="earth-water": bits.append(path(f"M{cx-62*s},{cy-35*s} Q{cx-20*s},{cy-75*s} {cx+8*s},{cy-35*s} L{cx-8*s},{cy+5*s} L{cx-64*s},{cy+15*s} Z",accent,dark,3*s))
    elif kind=="mars":
        for x,y,r in [(-35,-18,13),(28,22,17),(18,-42,8)]: bits.append(circle(cx+x*s,cy+y*s,r*s,accent,dark,2*s))
    elif kind=="jupiter":
        for offset,color in [(-45,light),(-15,accent),(20,light),(50,accent)]: bits.append(path(f"M{cx-72*s},{cy+offset*s} Q{cx},{cy+(offset-10)*s} {cx+72*s},{cy+offset*s}","none",color,9*s)); bits.append(ellipse(cx+38*s,cy+22*s,20*s,12*s,"#D94F45",dark,2*s))
    elif kind=="saturn": bits.append(ellipse(cx,cy,145*s,38*s,"none",accent,18*s)); bits.append(ellipse(cx,cy,145*s,38*s,"none",dark,4*s))
    elif kind=="uranus": bits.append(ellipse(cx,cy,42*s,140*s,"none",light,15*s)); bits.append(line(cx-65*s,cy+115*s,cx+65*s,cy-115*s,dark,5*s))
    elif kind=="neptune":
        for offset in (-35,0,35): bits.append(path(f"M{cx-72*s},{cy+offset*s} Q{cx},{cy+(offset-18)*s} {cx+72*s},{cy+offset*s}","none",light,7*s,True))
    elif kind=="pluto": bits.append(path(f"M{cx-38*s},{cy-18*s} C{cx-70*s},{cy-55*s} {cx-5*s},{cy-75*s} {cx},{cy-35*s} C{cx+5*s},{cy-75*s} {cx+70*s},{cy-55*s} {cx+38*s},{cy-18*s} Q{cx},{cy+35*s} {cx-38*s},{cy-18*s} Z",light,dark,3*s))
    return "".join(bits)


def custom_icon(key: str, cx: float, cy: float, scale: float, palette: tuple[str, ...]) -> str:
    """Draw the small set of subjects not present in the explainer icon set."""
    _, primary, accent, light, dark = palette
    s = scale
    bits: list[str] = []

    if key in {"force-up", "force-down"}:
        direction = -1 if key == "force-up" else 1
        bits.append(line(cx, cy - 72 * s * direction, cx, cy + 66 * s * direction, primary if direction < 0 else accent, 11 * s, True))
        bits.append(circle(cx, cy, 18 * s, light, dark, 4 * s))
        return "".join(bits)

    if key == "owl":
        # A recognizable anime-style owl: facial disks, forward-facing eyes,
        # ear tufts, hooked beak, layered wings, talons, and a branch.
        bits.append(line(cx - 115 * s, cy + 108 * s, cx + 125 * s, cy + 108 * s, dark, 15 * s))
        bits.append(ellipse(cx, cy + 22 * s, 76 * s, 104 * s, primary, dark, 6 * s))
        bits.append(path(f"M{cx-68*s},{cy+30*s} Q{cx-120*s},{cy+78*s} {cx-46*s},{cy+92*s} Q{cx-8*s},{cy+80*s} {cx-2*s},{cy+28*s} Z", accent, dark, 5 * s))
        bits.append(path(f"M{cx+68*s},{cy+30*s} Q{cx+120*s},{cy+78*s} {cx+46*s},{cy+92*s} Q{cx+8*s},{cy+80*s} {cx+2*s},{cy+28*s} Z", accent, dark, 5 * s))
        bits.append(circle(cx, cy - 50 * s, 78 * s, accent, dark, 6 * s))
        bits.append(polygon([(cx - 65 * s, cy - 98 * s), (cx - 36 * s, cy - 145 * s), (cx - 8 * s, cy - 103 * s)], primary, dark, 5 * s))
        bits.append(polygon([(cx + 65 * s, cy - 98 * s), (cx + 36 * s, cy - 145 * s), (cx + 8 * s, cy - 103 * s)], primary, dark, 5 * s))
        for offset in (-34, 34):
            bits.append(circle(cx + offset * s, cy - 52 * s, 31 * s, light, dark, 5 * s))
            bits.append(circle(cx + offset * s, cy - 52 * s, 15 * s, dark))
            bits.append(circle(cx + (offset - 5) * s, cy - 60 * s, 5 * s, light))
        bits.append(polygon([(cx, cy - 35 * s), (cx - 15 * s, cy - 7 * s), (cx + 15 * s, cy - 7 * s)], "#F59E0B", dark, 4 * s))
        for offset in (-25, 25):
            bits.append(path(f"M{cx+offset*s},{cy+104*s} q{-12*s},{22*s} {-28*s},{7*s} M{cx+offset*s},{cy+104*s} q{12*s},{22*s} {28*s},{7*s}", "none", dark, 4 * s))
        return "".join(bits)

    if key == "star-field":
        for dx, dy, radius in [(-62, 12, 34), (2, -48, 24), (62, 28, 29), (5, 58, 16)]:
            points = []
            for index in range(10):
                angle = -math.pi / 2 + index * math.pi / 5
                rr = radius if index % 2 == 0 else radius * 0.43
                points.append((cx + (dx + math.cos(angle) * rr) * s, cy + (dy + math.sin(angle) * rr) * s))
            bits.append(polygon(points, accent if dx < 0 else primary, dark, 3 * s))
        return "".join(bits)

    if key == "frost-leaf":
        bits.append(path(f"M{cx-90*s},{cy+58*s} C{cx-75*s},{cy-72*s} {cx+72*s},{cy-98*s} {cx+94*s},{cy-42*s} C{cx+66*s},{cy+70*s} {cx-15*s},{cy+95*s} {cx-90*s},{cy+58*s} Z", primary, dark, 5 * s))
        bits.append(line(cx - 68 * s, cy + 50 * s, cx + 62 * s, cy - 48 * s, light, 5 * s))
        for dx, dy in [(-48, -35), (10, 2), (55, 42)]:
            for angle in (0, math.pi / 3, 2 * math.pi / 3):
                bits.append(line(cx + (dx - math.cos(angle) * 18) * s, cy + (dy - math.sin(angle) * 18) * s, cx + (dx + math.cos(angle) * 18) * s, cy + (dy + math.sin(angle) * 18) * s, light, 3 * s))
        return "".join(bits)

    if key == "breath":
        bits.append(path(f"M{cx-75*s},{cy+78*s} Q{cx-105*s},{cy+8*s} {cx-72*s},{cy-62*s} Q{cx-20*s},{cy-118*s} {cx+18*s},{cy-68*s} L{cx+38*s},{cy-25*s} L{cx+18*s},{cy-10*s} Q{cx+38*s},{cy+22*s} {cx+3*s},{cy+30*s} Q{cx-20*s},{cy+63*s} {cx-75*s},{cy+78*s} Z", light, dark, 5 * s))
        bits.append(path(f"M{cx-35*s},{cy-37*s} Q{cx-13*s},{cy-50*s} {cx+2*s},{cy-35*s}", "none", dark, 4 * s))
        bits.append(circle(cx + 60 * s, cy + 4 * s, 24 * s, light, primary, 4 * s))
        bits.append(circle(cx + 96 * s, cy - 12 * s, 33 * s, light, primary, 4 * s))
        bits.append(circle(cx + 132 * s, cy + 5 * s, 22 * s, light, primary, 4 * s))
        return "".join(bits)

    if key == "coat":
        bits.append(path(f"M{cx-48*s},{cy-94*s} L{cx-105*s},{cy-55*s} L{cx-132*s},{cy+52*s} L{cx-80*s},{cy+70*s} L{cx-65*s},{cy+120*s} H{cx+65*s} L{cx+80*s},{cy+70*s} L{cx+132*s},{cy+52*s} L{cx+105*s},{cy-55*s} L{cx+48*s},{cy-94*s} L{cx},{cy-50*s} Z", primary, dark, 6 * s))
        bits.append(path(f"M{cx-48*s},{cy-94*s} L{cx},{cy-50*s} L{cx+48*s},{cy-94*s}", light, dark, 4 * s))
        bits.append(line(cx, cy - 48 * s, cx, cy + 118 * s, accent, 5 * s))
        for offset in (-12, 28, 68):
            bits.append(circle(cx + 20 * s, cy + offset * s, 5 * s, light, dark, 2 * s))
        return "".join(bits)

    if key == "hat":
        bits.append(circle(cx, cy + 28 * s, 64 * s, light, dark, 5 * s))
        bits.append(path(f"M{cx-74*s},{cy+12*s} Q{cx-66*s},{cy-88*s} {cx},{cy-100*s} Q{cx+66*s},{cy-88*s} {cx+74*s},{cy+12*s} Z", primary, dark, 6 * s))
        bits.append(rect(cx - 86 * s, cy - 2 * s, 172 * s, 30 * s, accent, 13 * s, dark, 5 * s))
        bits.append(circle(cx, cy - 108 * s, 18 * s, accent, dark, 4 * s))
        bits.append(path(f"M{cx-35*s},{cy+34*s} Q{cx},{cy+52*s} {cx+35*s},{cy+34*s}", "none", dark, 4 * s))
        return "".join(bits)

    if key == "calendar":
        bits.append(rect(cx - 95 * s, cy - 92 * s, 190 * s, 190 * s, light, 14 * s, dark, 6 * s))
        bits.append(rect(cx - 95 * s, cy - 92 * s, 190 * s, 45 * s, primary, 12 * s, dark, 5 * s))
        for dx in (-50, 50):
            bits.append(line(cx + dx * s, cy - 113 * s, cx + dx * s, cy - 72 * s, dark, 8 * s))
        for row in range(3):
            for column in range(4):
                bits.append(circle(cx + (-55 + column * 37) * s, cy + (-20 + row * 40) * s, 7 * s, accent if (row + column) % 3 == 0 else primary))
        bits.append(path(f"M{cx+54*s},{cy+98*s} L{cx+95*s},{cy+58*s} V{cy+98*s} Z", accent, dark, 3 * s))
        return "".join(bits)

    if key == "week":
        positions = [(-72, -40), (-24, -60), (28, -52), (70, -18), (55, 35), (5, 60), (-52, 42)]
        for index, (dx, dy) in enumerate(positions):
            bits.append(rect(cx + (dx - 20) * s, cy + (dy - 20) * s, 40 * s, 40 * s, primary if index % 2 == 0 else accent, 7 * s, dark, 3 * s))
            bits.append(circle(cx + dx * s, cy + dy * s, 5 * s, light))
        bits.append(path(f"M{cx-100*s},{cy} A{100*s},{85*s} 0 1 1 {cx+92*s},{cy-25*s}", "none", dark, 5 * s, True))
        return "".join(bits)

    if key == "sleep":
        bits.append(rect(cx - 125 * s, cy + 38 * s, 250 * s, 70 * s, primary, 18 * s, dark, 6 * s))
        bits.append(rect(cx - 112 * s, cy - 8 * s, 92 * s, 62 * s, light, 24 * s, dark, 5 * s))
        bits.append(circle(cx - 44 * s, cy - 34 * s, 52 * s, "#FFD8B5", dark, 5 * s))
        bits.append(path(f"M{cx-92*s},{cy-52*s} Q{cx-45*s},{cy-104*s} {cx+5*s},{cy-60*s} L{cx-2*s},{cy-20*s} Q{cx-38*s},{cy-58*s} {cx-92*s},{cy-52*s} Z", dark, dark, 3 * s))
        bits.append(path(f"M{cx-65*s},{cy-25*s} Q{cx-48*s},{cy-12*s} {cx-30*s},{cy-25*s}", "none", dark, 4 * s))
        bits.append(path(f"M{cx+5*s},{cy+22*s} Q{cx+70*s},{cy-8*s} {cx+112*s},{cy+43*s} V{cy+101*s} H{cx-2*s} Z", accent, dark, 5 * s))
        bits.append(path(f"M{cx+72*s},{cy-55*s} q{28*s},{-25*s} {48*s},0 q{-25*s},{5*s} {-36*s},{30*s} q{10*s},{-4*s} {25*s},{4*s}", "none", primary, 5 * s))
        return "".join(bits)

    if key == "lock-key":
        bits.append(path(f"M{cx-42*s},{cy-25*s} V{cy-60*s} A{42*s},{42*s} 0 0 1 {cx+42*s},{cy-60*s} V{cy-25*s}", "none", dark, 12 * s))
        bits.append(rect(cx - 65 * s, cy - 28 * s, 130 * s, 105 * s, primary, 12 * s, dark, 5 * s))
        bits.append(circle(cx, cy + 12 * s, 12 * s, light, dark, 3 * s))
        bits.append(line(cx, cy + 23 * s, cx, cy + 48 * s, light, 8 * s))
        bits.append(circle(cx + 102 * s, cy - 34 * s, 25 * s, "none", accent, 9 * s))
        bits.append(line(cx + 82 * s, cy - 18 * s, cx + 28 * s, cy + 42 * s, accent, 10 * s))
        return "".join(bits)

    if key == "truss":
        bits.extend([line(cx - 115 * s, cy + 65 * s, cx + 115 * s, cy + 65 * s, dark, 10 * s), line(cx - 100 * s, cy - 45 * s, cx + 100 * s, cy - 45 * s, primary, 9 * s)])
        points = [-100, -50, 0, 50, 100]
        for left, right in zip(points, points[1:]):
            bits.append(line(cx + left * s, cy + 65 * s, cx + right * s, cy - 45 * s, accent, 7 * s))
            bits.append(line(cx + right * s, cy + 65 * s, cx + left * s, cy - 45 * s, primary, 7 * s))
        return "".join(bits)

    if key == "arch":
        bits.append(path(f"M{cx-120*s},{cy+90*s} V{cy+45*s} A{120*s},{120*s} 0 0 1 {cx+120*s},{cy+45*s} V{cy+90*s} H{cx+65*s} V{cy+40*s} A{65*s},{65*s} 0 0 0 {cx-65*s},{cy+40*s} V{cy+90*s} Z", primary, dark, 6 * s))
        for angle in range(200, 341, 28):
            radians = math.radians(angle)
            bits.append(line(cx + math.cos(radians) * 68 * s, cy + 45 * s + math.sin(radians) * 68 * s, cx + math.cos(radians) * 116 * s, cy + 45 * s + math.sin(radians) * 116 * s, light, 4 * s))
        return "".join(bits)

    if key == "latitude":
        bits.append(circle(cx, cy, 82 * s, primary, dark, 5 * s))
        bits.append(ellipse(cx, cy, 82 * s, 31 * s, "none", light, 5 * s))
        bits.append(ellipse(cx, cy, 36 * s, 82 * s, "none", light, 5 * s))
        bits.append(line(cx - 82 * s, cy, cx + 82 * s, cy, accent, 7 * s))
        return "".join(bits)

    if key == "border":
        bits.append(path(f"M{cx-95*s},{cy+70*s} L{cx-72*s},{cy-58*s} L{cx-5*s},{cy-88*s} L{cx+65*s},{cy-48*s} L{cx+98*s},{cy+54*s} L{cx+22*s},{cy+84*s} Z", light, dark, 5 * s))
        bits.append(path(f"M{cx-8*s},{cy-78*s} Q{cx+38*s},{cy-38*s} {cx+2*s},{cy+2*s} T{cx+42*s},{cy+77*s}", "none", accent, 7 * s, False, "11 10"))
        return "".join(bits)

    if key == "shell":
        bits.append(path(f"M{cx-92*s},{cy+65*s} C{cx-86*s},{cy-58*s} {cx-42*s},{cy-95*s} {cx},{cy-95*s} C{cx+42*s},{cy-95*s} {cx+86*s},{cy-58*s} {cx+92*s},{cy+65*s} Z", accent, dark, 6 * s))
        for endpoint in (-70, -35, 0, 35, 70):
            bits.append(line(cx, cy - 82 * s, cx + endpoint * s, cy + 58 * s, light, 5 * s))
        return "".join(bits)

    if key == "plastic-bottle":
        bits.append(rect(cx - 30 * s, cy - 110 * s, 60 * s, 36 * s, accent, 5 * s, dark, 4 * s))
        bits.append(path(f"M{cx-25*s},{cy-72*s} L{cx-62*s},{cy-25*s} V{cy+94*s} Q{cx},{cy+118*s} {cx+62*s},{cy+94*s} V{cy-25*s} L{cx+25*s},{cy-72*s} Z", light, dark, 6 * s))
        bits.append(path(f"M{cx-56*s},{cy+30*s} Q{cx},{cy+5*s} {cx+56*s},{cy+30*s}", "none", primary, 9 * s))
        return "".join(bits)

    if key == "galaxy":
        for radius, color, width in [(118, primary, 16), (82, accent, 12), (48, light, 9)]:
            bits.append(path(f"M{cx-radius*s},{cy+15*s} C{cx-radius*.45*s},{cy-radius*.8*s} {cx+radius*.65*s},{cy-radius*.55*s} {cx+radius*s},{cy} C{cx+radius*.45*s},{cy+radius*.78*s} {cx-radius*.6*s},{cy+radius*.55*s} {cx-radius*.78*s},{cy+8*s}", "none", color, width * s))
        bits.append(circle(cx, cy, 18 * s, light, dark, 3 * s))
        return "".join(bits)

    if key == "asteroids":
        for dx, dy, radius in [(-70, -32, 32), (-5, 22, 44), (64, -22, 25), (82, 55, 18), (-78, 62, 17)]:
            points = []
            for index in range(8):
                angle = index * math.tau / 8
                rr = radius * (0.78 if index % 2 else 1.0)
                points.append((cx + (dx + math.cos(angle) * rr) * s, cy + (dy + math.sin(angle) * rr) * s))
            bits.append(polygon(points, primary if dx < 0 else accent, dark, 4 * s))
        return "".join(bits)

    if key == "wood":
        bits.append(ellipse(cx, cy, 105 * s, 78 * s, accent, dark, 6 * s))
        for radius in (22, 43, 65):
            bits.append(ellipse(cx, cy, radius * 1.25 * s, radius * s, "none", light if radius == 43 else dark, 3 * s))
        bits.append(path(f"M{cx-16*s},{cy-8*s} L{cx+34*s},{cy+22*s} L{cx+72*s},{cy+5*s}", "none", dark, 4 * s))
        return "".join(bits)

    if key == "traffic-light":
        bits.append(rect(cx - 54 * s, cy - 122 * s, 108 * s, 244 * s, dark, 18 * s, dark, 5 * s))
        for offset, color in [(-76, "#F05252"), (0, "#F5B83D"), (76, "#39A96B")]:
            bits.append(circle(cx, cy + offset * s, 30 * s, color, light, 4 * s))
        return "".join(bits)

    if key == "measure":
        bits.append(rect(cx - 118 * s, cy - 22 * s, 236 * s, 52 * s, accent, 6 * s, dark, 5 * s))
        for index in range(11):
            height = 30 if index % 5 == 0 else 18
            bits.append(line(cx + (-100 + index * 20) * s, cy - 18 * s, cx + (-100 + index * 20) * s, cy + (-18 + height) * s, dark, 3 * s))
        bits.append(line(cx, cy - 34 * s, cx, cy - 92 * s, dark, 6 * s))
        bits.append(line(cx - 76 * s, cy - 72 * s, cx + 76 * s, cy - 72 * s, primary, 7 * s))
        bits.append(path(f"M{cx-76*s},{cy-70*s} L{cx-112*s},{cy+5*s} H{cx-40*s} Z M{cx+76*s},{cy-70*s} L{cx+40*s},{cy+5*s} H{cx+112*s} Z", light, dark, 4 * s))
        return "".join(bits)

    if key == "experiment":
        for offset, color in [(-52, primary), (52, accent)]:
            bits.append(path(f"M{cx+(offset-28)*s},{cy-100*s} H{cx+(offset+28)*s} V{cy-18*s} L{cx+(offset+62)*s},{cy+92*s} H{cx+(offset-62)*s} L{cx+(offset-28)*s},{cy-18*s} Z", light, dark, 5 * s))
            bits.append(path(f"M{cx+(offset-48)*s},{cy+40*s} Q{cx+offset*s},{cy+20*s} {cx+(offset+48)*s},{cy+40*s} L{cx+(offset+60)*s},{cy+87*s} H{cx+(offset-60)*s} Z", color, "none"))
        return "".join(bits)

    if key == "model":
        vertices = [(cx - 70 * s, cy - 35 * s), (cx, cy - 85 * s), (cx + 70 * s, cy - 35 * s), (cx - 70 * s, cy + 55 * s), (cx, cy + 100 * s), (cx + 70 * s, cy + 55 * s)]
        for start, end in [(0, 1), (1, 2), (0, 3), (2, 5), (3, 4), (4, 5), (0, 5), (2, 3)]:
            bits.append(line(*vertices[start], *vertices[end], primary if start % 2 else accent, 6 * s, False, "10 7"))
        for index, (x, y) in enumerate(vertices):
            bits.append(circle(x, y, 14 * s, light, dark, 4 * s))
        return "".join(bits)

    if key == "question":
        bits.append(path(f"M{cx-55*s},{cy-44*s} C{cx-55*s},{cy-118*s} {cx+72*s},{cy-120*s} {cx+72*s},{cy-42*s} C{cx+72*s},{cy+15*s} {cx+8*s},{cy+18*s} {cx+8*s},{cy+65*s}", "none", primary, 20 * s))
        bits.append(circle(cx + 8 * s, cy + 105 * s, 13 * s, accent, dark, 3 * s))
        return "".join(bits)

    raise KeyError(f"no custom icon renderer for {key}")


def topic_icon(key: str, cx: float, cy: float, scale: float, palette: tuple[str, ...]) -> str:
    if key in SUBJECT_ICON_KEYS and key.startswith("insect-"):
        return insect_subject(key.removeprefix("insect-"), cx, cy, scale, palette)
    if key in SUBJECT_ICON_KEYS and key.startswith("small-"):
        return small_subject(key.removeprefix("small-"), cx, cy, scale, palette)
    if key in SUBJECT_ICON_KEYS and key.startswith("bird-"):
        return bird_subject(key.removeprefix("bird-"), cx, cy, scale, palette)
    if key in SUBJECT_ICON_KEYS and key.startswith("mammal-"):
        return mammal_subject(key.removeprefix("mammal-"), cx, cy, scale, palette)
    if key in SUBJECT_ICON_KEYS and key.startswith("aquatic-"):
        return aquatic_subject(key.removeprefix("aquatic-"), cx, cy, scale, palette)
    if key in SUBJECT_ICON_KEYS and key.startswith("geo-"):
        return geo_subject(key.removeprefix("geo-"), cx, cy, scale, palette)
    if key in SUBJECT_ICON_KEYS and key.startswith("ocean-"):
        return ocean_subject(key.removeprefix("ocean-"), cx, cy, scale, palette)
    if key in SUBJECT_ICON_KEYS and key.startswith("space-"):
        return space_subject(key.removeprefix("space-"), cx, cy, scale, palette)
    question_aliases = {
        "chromatophore": "octopus-skin",
        "crust-mantle": "earth-cutaway",
        "dermis": "skin",
        "epidermis": "skin",
        "insect-body": "insect",
        "molecular-contact": "molecule",
        "reflector": "octopus-skin",
        "setae": "hair",
        "six-legs": "insect",
        "soap": "micelle",
        "thick-air": "air-layers",
        "tide": "water",
    }
    key = question_aliases.get(key, key)
    if key in CUSTOM_ICON_KEYS or key in {"force-up", "force-down"}:
        return custom_icon(key, cx, cy, scale, palette)
    _, primary, accent, light, dark = palette
    return explainer.icon(key, cx, cy, scale, primary, accent, dark, light)


def scene_spec(question: int) -> tuple[str, tuple[str, ...], str]:
    if question in EARTH_WEATHER_QUESTIONS:
        return "earth-weather", EARTH_WEATHER_KEYS[question], "earth-weather"
    if question in TEXTBOOK_QUESTIONS:
        return "textbook", TEXTBOOK_KEYS[question], "textbook"
    if question in INVERTEBRATE_QUESTIONS:
        return "invertebrate", INVERTEBRATE_KEYS[question], "invertebrate"
    if question in VERTEBRATE_QUESTIONS:
        return "vertebrate", VERTEBRATE_KEYS[question], "vertebrate"
    if question in PLANT_QUESTIONS:
        return "plant", PLANT_KEYS[question], "plant"
    if question in MACHINE_QUESTIONS:
        return "machine", MACHINE_KEYS[question], "machine"
    if question in SUBJECT_ICONS:
        return "concrete", (SUBJECT_ICONS[question],), "subject"
    diagram = DIAGRAMS_BY_QUESTION.get(question)
    if diagram is not None:
        layout = diagram.kind
        keys = tuple(key for key, _ in diagram.nodes)
        source = "explainer"
    else:
        layout, keys = MANUAL_SCENES[question]
        source = "manual"
    return "concrete", keys, source


def icon_stage(key: str, x: float, y: float, scale: float, palette: tuple[str, ...], index: int) -> str:
    _, primary, accent, light, dark = palette
    stage_color = primary if index % 2 == 0 else accent
    return "".join([
        circle(x, y, 102 * scale, light, stage_color, 3.5 * scale),
        circle(x - 31 * scale, y - 36 * scale, 13 * scale, stage_color, "none", 0),
        topic_icon(key, x, y + 3 * scale, 0.72 * scale, palette),
        path(f"M{x-64*scale},{y+74*scale} Q{x},{y+96*scale} {x+64*scale},{y+74*scale}", "none", dark, 3.5 * scale),
    ])


def open_subject(key: str, x: float, y: float, scale: float, palette: tuple[str, ...], index: int = 0) -> str:
    """Draw a subject at readable scale without enclosing it in an abstract card."""
    _, primary, accent, _, dark = palette
    color = primary if index % 2 == 0 else accent
    return "".join([
        ellipse(x, y + 112 * scale, 92 * scale, 16 * scale, dark, "none", 0).replace('/>', ' opacity="0.10"/>'),
        path(f"M{x-78*scale},{y+96*scale} Q{x},{y+118*scale} {x+78*scale},{y+96*scale}", "none", color, 3.5 * scale),
        topic_icon(key, x, y, scale, palette),
    ])


def render_flow(keys: tuple[str, ...], palette: tuple[str, ...]) -> str:
    _, primary, accent, _, dark = palette
    count = len(keys)
    xs = [480] if count == 1 else [285, 675] if count == 2 else [175, 480, 785] if count == 3 else [125, 365, 595, 835]
    scale = 1.55 if count == 1 else 1.22 if count == 2 else 0.94 if count == 3 else 0.72
    bits = [open_subject(key, x, 270, scale, palette, index) for index, (key, x) in enumerate(zip(keys, xs))]
    for index, (left, right) in enumerate(zip(xs, xs[1:])):
        reach = 118 * scale
        bits.append(line(left + reach, 270, right - reach, 270, primary if index % 2 == 0 else accent, 7, True))
    bits.append(path("M82,452 Q480,490 878,452", "none", dark, 3).replace('/>', ' opacity="0.18"/>'))
    return "".join(bits)


def render_compare(keys: tuple[str, ...], palette: tuple[str, ...]) -> str:
    _, primary, accent, light, dark = palette
    count = len(keys)
    xs = [310, 650] if count == 2 else [185, 480, 775]
    scale = 1.28 if count == 2 else 0.92
    bits = [path("M480,100 V440", "none", dark, 2, False, "9 11").replace('/>', ' opacity="0.28"/>') if count == 2 else ""]
    for index, (key, x) in enumerate(zip(keys, xs)):
        bits.append(open_subject(key, x, 270, scale, palette, index))
    bits.append(path("M105,445 Q480,505 855,445", "none", primary, 5))
    bits.append(path("M105,462 Q480,522 855,462", "none", accent, 3))
    return "".join(bits)


def render_cycle(keys: tuple[str, ...], palette: tuple[str, ...]) -> str:
    _, primary, accent, _, _ = palette
    positions = [(480, 125), (735, 285), (480, 425), (225, 285)] if len(keys) == 4 else [(480, 135), (700, 390), (260, 390)]
    scale = 0.70 if len(keys) == 4 else 0.82
    bits = [ellipse(480, 280, 285, 175, "none", primary, 4).replace('/>', ' opacity="0.45"/>')]
    for index, (key, (x, y)) in enumerate(zip(keys, positions)):
        bits.append(open_subject(key, x, y, scale, palette, index))
        nx, ny = positions[(index + 1) % len(positions)]
        dx, dy = nx - x, ny - y
        length = math.hypot(dx, dy)
        bits.append(line(x + dx / length * 103, y + dy / length * 103, nx - dx / length * 111, ny - dy / length * 111, primary if index % 2 == 0 else accent, 5, True))
    return "".join(bits)


def render_forces(keys: tuple[str, ...], palette: tuple[str, ...]) -> str:
    _, primary, accent, light, dark = palette
    center_key = keys[0]
    bits = [ellipse(480, 430, 220, 35, primary, "none", 0).replace('/>', ' opacity="0.16"/>'), open_subject(center_key, 480, 265, 1.42, palette, 0)]
    bits.append(line(335, 290, 335, 92, primary, 9, True))
    bits.append(line(625, 190, 625, 424, accent, 9, True))
    for index, key in enumerate(keys[1:]):
        x = 214 if index % 2 == 0 else 746
        y = 155 if index < 2 else 405
        bits.append(topic_icon(key, x, y, 0.58, palette))
    return "".join(bits)


def render_orbit(keys: tuple[str, ...], palette: tuple[str, ...]) -> str:
    _, primary, accent, light, dark = palette
    positions = [(480, 280), (245, 175), (730, 180), (665, 425)]
    bits = [ellipse(480, 280, 310, 180, "none", primary, 4).replace('/>', ' opacity="0.65"/>'), ellipse(480, 280, 215, 122, "none", accent, 3).replace('/>', ' opacity="0.55"/>')]
    for index, (key, (x, y)) in enumerate(zip(keys, positions)):
        bits.append(open_subject(key, x, y, 1.05 if index == 0 else 0.64, palette, index))
    bits.append(path("M177,315 A310,180 0 0 0 430,457", "none", primary, 7, True))
    return "".join(bits)


def render_network(keys: tuple[str, ...], palette: tuple[str, ...]) -> str:
    _, primary, accent, light, dark = palette
    center = (480, 285)
    satellites = [(190, 155), (770, 155), (230, 425), (730, 425)]
    bits = [open_subject(keys[0], *center, 1.28, palette, 0)]
    for index, (key, position) in enumerate(zip(keys[1:], satellites)):
        bits.append(path(f"M{center[0]},{center[1]} Q480,{135 if position[1] < 285 else 445} {position[0]},{position[1]}", "none", primary if index % 2 == 0 else accent, 4).replace('/>', ' opacity="0.55"/>'))
        bits.append(open_subject(key, *position, 0.58, palette, index + 1))
    return "".join(bits)


def render_layers(keys: tuple[str, ...], palette: tuple[str, ...]) -> str:
    _, primary, accent, light, dark = palette
    main_x, main_y = 305, 275
    detail_count = len(keys) - 1
    ys = [280] if detail_count == 1 else [190, 360] if detail_count == 2 else [135, 280, 425]
    bits = [open_subject(keys[0], main_x, main_y, 1.45, palette, 0), path("M92,445 Q320,485 555,445", "none", primary, 5)]
    for index, (key, y) in enumerate(zip(keys[1:], ys), 1):
        bits.append(path(f"M405,{main_y + (index-detail_count/2)*20:.1f} Q535,{y:.1f} 630,{y:.1f}", "none", primary if index % 2 else accent, 3, False, "8 8"))
        bits.append(open_subject(key, 745, y, 0.66 if detail_count < 3 else 0.55, palette, index))
    return "".join(bits)


def render_subject(keys: tuple[str, ...], palette: tuple[str, ...]) -> str:
    """Present an observable subject directly instead of forcing it into a relation template."""
    key = keys[0]
    _, primary, accent, light, dark = palette
    if key.startswith(("geo-", "ocean-", "space-")):
        scale = 1.48
    elif key.startswith(("insect-", "small-")):
        scale = 1.42
    else:
        scale = 1.36
    return "".join([
        ellipse(480, 438, 245, 28, dark, "none", 0).replace('/>', ' opacity="0.10"/>'),
        path("M135,430 Q480,485 825,430", "none", primary, 5),
        path("M215,457 Q480,493 745,457", "none", accent, 3),
        topic_icon(key, 480, 270, scale, palette),
        circle(765, 116, 9, accent),
        circle(804, 92, 5, primary),
        circle(193, 126, 6, primary),
    ])


INVISIBLE_CONTEXT_KEYS = {
    "air", "air-layers", "atom", "balance", "brain", "capillary", "cell",
    "charge", "collision", "cool-air", "counter", "crystal-spacing", "energy",
    "enzyme", "field", "force-down", "force-up", "forward", "fusion",
    "gas-exchange", "gravity-center", "heat", "heat-out", "hydration",
    "ionized-air", "light", "light-air", "matter", "micelle", "molecule",
    "motion", "motion-arrow", "nutrient", "orbit", "pressure", "rotation",
    "salt", "sound-wave", "spring", "sugar", "sugar-oxygen", "trapped-heat",
    "vapor", "wave", "wavy-air", "wind", "water-air", "water-particles",
}


def concrete_focus(question: int, keys: tuple[str, ...]) -> str:
    if question in CONCRETE_FOCUS_OVERRIDES:
        return CONCRETE_FOCUS_OVERRIDES[question]
    for key in keys:
        if key not in INVISIBLE_CONTEXT_KEYS:
            return key
    return keys[0]


def concrete_backdrop(question: int, palette: tuple[str, ...]) -> str:
    """Create one continuous place instead of several disconnected icon stages."""
    chapter = CHAPTER_BY_QUESTION[question].number
    _, primary, accent, light, dark = palette
    bits: list[str] = []
    if chapter <= 4:
        bits.extend([
            rect(18, 18, 924, 420, "#DDEEFF", 6),
            path("M18,390 Q180,330 330,390 T640,390 T942,390 V438 H18 Z", "#B8D49B"),
            path("M18,420 Q220,380 410,420 T790,420 T980,420", "none", "#7DAA6D", 5),
        ])
    elif chapter <= 7:
        bits.extend([
            rect(18, 18, 924, 330, "#DDF3F4", 6),
            path("M18,340 Q180,300 350,340 T680,340 T980,340 V438 H18 Z", "#99C9B3"),
            path("M18,405 Q170,365 330,405 T650,405 T970,405", "none", primary, 7),
        ])
    elif chapter <= 10:
        bits.extend([
            rect(18, 18, 924, 300, "#EAF5DC", 6),
            rect(18, 318, 924, 120, "#B9875A", 0),
            path("M18,318 Q150,292 290,318 T570,318 T850,318 T1000,318", "none", "#5F8B55", 8),
        ])
    elif chapter <= 16:
        bits.extend([
            rect(18, 18, 924, 330, "#EAF3F6", 6),
            path("M18,340 Q180,285 340,340 T670,340 T980,340 V438 H18 Z", "#BCD49B"),
            path("M18,413 Q200,375 380,413 T740,413 T1040,413", "none", "#6E9B64", 6),
        ])
    elif chapter <= 22:
        bits.extend([
            rect(18, 18, 924, 320, "#F2EFE8", 6),
            rect(18, 338, 924, 100, "#C99C70", 0),
            line(18, 338, 942, 338, dark, 6),
            line(90, 92, 870, 92, primary, 4),
        ])
    elif chapter <= 25:
        bits.extend([
            rect(18, 18, 924, 305, "#DDEEF1", 6),
            path("M18,330 Q160,190 320,330 Q470,150 650,330 Q790,235 942,330 V438 H18 Z", "#8AB584"),
            path("M18,410 Q200,365 390,410 T770,410 T1040,410", "none", "#4E8D70", 7),
        ])
    elif chapter <= 27:
        bits.extend([
            rect(18, 18, 924, 420, "#BDE9EE", 6),
            path("M18,92 Q160,68 300,92 T580,92 T860,92 T1000,92", "none", "#FFFFFF", 7),
            path("M18,390 Q170,350 330,390 T650,390 T970,390 V438 H18 Z", "#D7C28A"),
        ])
        for x, y, radius in ((96, 165, 7), (145, 125, 5), (835, 180, 8), (790, 245, 5), (875, 290, 4)):
            bits.append(circle(x, y, radius, "none", "#FFFFFF", 2))
    elif chapter <= 29:
        bits.append(rect(18, 18, 924, 420, "#151A36", 6))
        for index in range(24):
            x = 45 + (index * 83) % 860
            y = 45 + (index * 47) % 340
            bits.append(circle(x, y, 2 + index % 3, light if index % 2 else "#F4B940"))
    elif chapter <= 31:
        bits.extend([
            rect(18, 18, 924, 315, "#F4EFE8", 6),
            rect(18, 333, 924, 105, "#A9B6B0", 0),
            line(18, 333, 942, 333, dark, 5),
            path("M75,80 H885 M75,145 H885", "none", "#D8CFC4", 4),
        ])
    elif chapter <= 33:
        bits.extend([
            rect(18, 18, 924, 285, "#DCEFF2", 6),
            path("M18,310 Q160,240 320,310 T620,310 T940,310 V438 H18 Z", "#9CC492"),
            rect(18, 365, 924, 73, "#67747A", 0),
            line(18, 400, 942, 400, "#F3D15A", 5, False, "18 18"),
        ])
    else:
        bits.extend([
            rect(18, 18, 924, 315, "#F4F5EE", 6),
            rect(18, 333, 924, 105, "#B88B63", 0),
            line(18, 333, 942, 333, dark, 5),
        ])
    return "".join(bits)


def concrete_special(question: int, palette: tuple[str, ...]) -> str:
    """Whole-scene treatments for familiar phenomena that should not look like a diagram."""
    _, primary, accent, light, dark = palette
    if question == 3:
        return "".join([
            topic_icon("cloud", 330, 235, 1.45, palette),
            topic_icon("cloud", 650, 270, 1.6, palette),
            path("M535,248 Q650,330 765,248", "#788590", "none"),
            circle(180, 105, 46, "#F5B83D", dark, 4),
        ])
    if question == 4:
        return "".join([
            circle(480, 325, 63, "#F06B50", dark, 5),
            path("M18,190 H942 V438 H18 Z", "#F6B36C", "none").replace('/>', ' opacity="0.45"/>'),
            path("M80,225 Q250,195 410,225 T740,225 T960,225", "none", "#E96658", 12),
            path("M18,355 H942 V438 H18 Z", "#526D67"),
        ])
    if question == 5:
        return "".join([
            topic_icon("lamp", 170, 230, 1.1, palette),
            path("M245,205 L430,175 L430,355 L245,285 Z", "#F5D46B").replace('/>', ' opacity="0.38"/>'),
            rect(430, 205, 95, 150, accent, 5, dark, 5),
            path("M525,345 Q700,390 850,355 Q700,330 525,315 Z", dark).replace('/>', ' opacity="0.55"/>'),
        ])
    if question == 7:
        return "".join([
            topic_icon("face", 260, 255, 1.15, palette),
            rect(465, 90, 26, 300, "#C8D5DF", 3, dark, 5),
            topic_icon("face", 690, 255, 1.15, palette),
            line(490, 105, 490, 385, "#FFFFFF", 4),
        ])
    if question == 8:
        return "".join([
            rect(250, 75, 460, 310, "#D9F2F6", 4, dark, 8),
            line(480, 78, 480, 382, dark, 6), line(253, 230, 707, 230, dark, 6),
            circle(355, 145, 38, "#F5B83D", dark, 4),
            path("M260,340 Q355,235 445,340 Q540,250 700,340", "#7EAD73", dark, 4),
            path("M260,365 H700", "none", "#4387A8", 16),
        ])
    if question == 9:
        return "".join(path(f"M160,{405-index*7} Q480,{80-index*5} 800,{405-index*7}", "none", color, 12) for index, color in enumerate(("#E75858", "#F09A3E", "#E7C84A", "#54A66B", "#4F7FC4", "#745FA8")))
    if question == 12:
        phases = ("moon", "half-moon", "full-moon", "half-moon")
        return "".join(topic_icon(key, 175 + index * 205, 245, 0.88, palette) for index, key in enumerate(phases))
    if question == 17:
        bits = []
        for index, color in enumerate(("#93C86A", "#4D9A55", "#D98A42", "#D7E2E5")):
            x = 155 + index * 215
            bits.extend([line(x, 310, x, 185, "#6C4A34", 15), circle(x, 165, 58, color, dark, 4)])
        return "".join(bits)
    if question == 31:
        return "".join([
            path("M45,410 Q160,250 280,410 Q400,180 520,410 Q650,225 790,410 Q865,300 935,410", "none", "#66D8A5", 24),
            path("M65,420 Q205,285 330,420 Q470,240 610,420 Q760,260 900,420", "none", "#7A8DE8", 18),
            path("M18,405 Q190,350 350,405 T680,405 T980,405 V438 H18 Z", "#263C48"),
        ])
    return ""


def render_concrete_scene(question: int, keys: tuple[str, ...], palette: tuple[str, ...]) -> str:
    """Render one coherent, child-readable scene; reserve symbols for a small inset."""
    _, primary, accent, light, dark = palette
    focus = concrete_focus(question, keys)
    bits = [concrete_backdrop(question, palette)]
    special = concrete_special(question, palette)
    if special:
        bits.append(special)
        return "".join(bits)

    chapter = CHAPTER_BY_QUESTION[question].number
    cx, cy, scale = 480, 255, 1.22
    if chapter in {8, 9, 10}:
        cy, scale = 245, 1.22
    elif chapter in {13, 14, 15, 16}:
        cy, scale = 245, 1.16
    elif chapter in {20, 21, 22, 30, 31, 34}:
        cy, scale = 235, 1.10
    elif chapter in {23, 24, 25}:
        cy, scale = 245, 1.18
    elif chapter in {26, 27}:
        cy, scale = 245, 1.14
    elif chapter in {28, 29}:
        cy, scale = 240, 1.15
    elif chapter in {32, 33}:
        cy, scale = 230, 1.08

    contexts = [key for key in keys if key != focus and key not in INVISIBLE_CONTEXT_KEYS]
    contexts = list(dict.fromkeys(contexts))[:2]
    if contexts:
        cx = 430
    bits.append(ellipse(cx, cy + 125 * scale, 135 * scale, 18 * scale, dark).replace('/>', ' opacity="0.12"/>'))
    bits.append(topic_icon(focus, cx, cy, scale, palette))

    positions = [(735, 170), (755, 335)]
    for index, (key, (x, y)) in enumerate(zip(contexts, positions)):
        bits.append(topic_icon(key, x, y, 0.62 if len(contexts) > 1 else 0.72, palette))

    # A few environmental cues make the picture read as a place, not a symbol shelf.
    if chapter in {8, 9, 10}:
        bits.extend([line(120, 318, 105, 270, "#4E8A4B", 8), line(835, 318, 850, 260, "#4E8A4B", 8)])
    elif chapter in {13, 14, 15, 16}:
        bits.extend([path("M85,340 q35,-80 70,0 M790,340 q35,-75 70,0", "none", "#5F9860", 8)])
    elif chapter in {20, 21, 22, 30, 31, 34}:
        bits.extend([circle(90, 375, 11, accent), circle(865, 380, 9, primary)])
    elif chapter in {26, 27}:
        bits.extend([path("M85,365 q20,-70 40,0 M835,360 q18,-65 36,0", "none", "#4A8D79", 8)])
    return "".join(bits)


RENDERERS = {
    "compare": render_compare,
    "cycle": render_cycle,
    "flow": render_flow,
    "forces": render_forces,
    "layers": render_layers,
    "network": render_network,
    "orbit": render_orbit,
    "subject": render_subject,
}


def part_environment(question: int, palette: tuple[str, ...]) -> str:
    """Add quiet, domain-specific context without competing with the subject."""
    part = part_for(question)
    _, primary, accent, light, dark = palette
    opacity = ' opacity="0.16"'
    if part == 0:
        return f'<g{opacity}>' + path("M45,95 H215 M745,92 H910 M80,455 H260 M700,458 H900", "none", primary, 5) + circle(875, 145, 7, accent) + circle(825, 105, 5, accent) + "</g>"
    if part == 1:
        return f'<g{opacity}>' + path("M30,125 Q145,80 260,125 T490,125 M610,450 Q735,405 900,450", "none", primary, 8) + "</g>"
    if part == 2:
        return f'<g{opacity}>' + path("M50,470 Q155,335 285,470 M675,470 Q790,330 915,470", "none", primary, 13) + "</g>"
    if part == 3:
        return f'<g{opacity}>' + path("M40,460 Q160,390 280,460 T520,460 T760,460 T1000,460", "none", primary, 8) + "</g>"
    if part == 4:
        return f'<g{opacity}>' + path("M45,445 C170,390 265,505 390,445 S615,390 745,445 S900,500 955,430", "none", accent, 7) + "</g>"
    if part == 5:
        return f'<g{opacity}>' + "".join(circle(95 + index * 95, 475 - (index % 2) * 26, 16, primary if index % 2 else accent) for index in range(9)) + "</g>"
    if part == 6:
        return f'<g{opacity}>' + line(45, 455, 915, 455, dark, 5) + "".join(circle(120 + index * 145, 455, 25, "none", primary, 5) for index in range(6)) + "</g>"
    if part == 7:
        return f'<g{opacity}>' + path("M35,450 Q155,400 275,450 T515,450 T755,450 T995,450", "none", primary, 7) + path("M65,475 Q185,425 305,475 T545,475 T785,475", "none", accent, 5) + "</g>"
    if part == 8:
        return f'<g{opacity}>' + path("M0,430 Q120,370 240,430 T480,430 T720,430 T960,430", "none", primary, 12) + path("M0,470 Q120,410 240,470 T480,470 T720,470 T960,470", "none", accent, 7) + "</g>"
    if part == 9:
        return f'<g{opacity}>' + "".join(circle(75 + index * 105, 80 + (index % 3) * 22, 5 + index % 2 * 3, light, accent, 2) for index in range(9)) + "</g>"
    if part == 10:
        return f'<g{opacity}>' + "".join(circle(75 + (index % 9) * 105, 445 + (index // 9) * 38, 10, primary if index % 2 else accent) for index in range(18)) + "</g>"
    return f'<g{opacity}>' + line(35, 450, 925, 450, primary, 6, True) + line(90, 485, 870, 485, accent, 4, True) + "</g>"


def make_svg(question: int, title: str) -> str:
    palette = PALETTES[part_for(question)]
    background, primary, accent, light, dark = palette
    layout, keys, source = scene_spec(question)
    if layout == "textbook":
        body = textbook_body(question, palette)
    elif layout == "earth-weather":
        body = earth_weather_body(question, palette)
    elif layout == "invertebrate":
        body = invertebrate_body(question, palette)
    elif layout == "vertebrate":
        body = vertebrate_body(question, palette)
    elif layout == "plant":
        body = plant_body(question, palette)
    elif layout == "machine":
        body = machine_body(question, palette)
    elif layout == "concrete":
        body = render_concrete_scene(question, keys, palette)
    else:
        body = RENDERERS[layout](keys, palette)
    presentation = layout if layout in {"concrete", "earth-weather", "invertebrate", "machine", "plant", "textbook", "vertebrate"} else "open"
    safe_title = html.escape(title, quote=True)
    safe_icons = html.escape(",".join(keys), quote=True)
    description = html.escape(f"围绕“{title}”绘制的清晰主题图；具体对象优先，必要关系用箭头表示。", quote=True)
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="960" height="540" viewBox="0 0 960 540" role="img" aria-labelledby="title desc">
  <title id="title">{safe_title}</title>
  <desc id="desc">{description}</desc>
  <defs>
    <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="{dark}"/></marker>
  </defs>
  <rect width="960" height="540" fill="{background}"/>
  <path d="M0,455 Q185,410 355,455 T700,455 T1040,455 V540 H0 Z" fill="{primary}" opacity="0.07"/>
  {"" if layout in {"concrete", "earth-weather", "machine", "plant", "vertebrate"} else part_environment(question, palette)}
  <g data-question="{question:03d}" data-layout="{layout}" data-presentation="{presentation}" data-source="{source}" data-icons="{safe_icons}">{body}</g>
  <rect x="12" y="12" width="936" height="516" rx="8" fill="none" stroke="{dark}" stroke-width="3" opacity="0.18"/>
</svg>
'''


def validate_specs(all_entries: list[tuple[int, str]]) -> None:
    questions = [question for question, _ in all_entries]
    expected_questions = list(CHAPTER_BY_QUESTION)
    if questions != expected_questions:
        raise RuntimeError("question headings differ from the canonical structure")

    diagram_questions = set(DIAGRAMS_BY_QUESTION)
    manual_questions = set(MANUAL_SCENES)
    if diagram_questions & manual_questions:
        raise RuntimeError(f"question scene sources overlap: {sorted(diagram_questions & manual_questions)}")
    covered_questions = diagram_questions | manual_questions | set(EARTH_WEATHER_QUESTIONS) | set(TEXTBOOK_QUESTIONS) | set(INVERTEBRATE_QUESTIONS) | set(MACHINE_QUESTIONS) | set(PLANT_QUESTIONS) | set(VERTEBRATE_QUESTIONS)
    if covered_questions != set(questions):
        missing = sorted(set(questions) - covered_questions)
        extra = sorted((diagram_questions | manual_questions) - set(questions))
        raise RuntimeError(f"question scene coverage mismatch: missing={missing}, extra={extra}")

    unsupported_layouts = sorted({scene_spec(question)[0] for question in questions} - set(RENDERERS) - {"concrete", "earth-weather", "invertebrate", "machine", "plant", "textbook", "vertebrate"})
    if unsupported_layouts:
        raise RuntimeError(f"unsupported question-art layouts: {unsupported_layouts}")

    signatures = [(scene_spec(question)[0], scene_spec(question)[1]) for question in questions]
    unique_count = len(set(signatures))
    minimum_unique = max(1, int(len(questions) * 0.9))
    if unique_count < minimum_unique:
        repeated = Counter(signatures).most_common(10)
        raise RuntimeError(f"question scene variety too low: {unique_count}/{len(questions)}; repeated={repeated}")


def main() -> None:
    all_entries = entries()
    validate_specs(all_entries)
    explainer.FALLBACK_KEYS.clear()
    rendered = [(question, title, make_svg(question, title)) for question, title in all_entries]
    if explainer.FALLBACK_KEYS:
        raise RuntimeError("unsupported explainer icons in question art: " + ", ".join(sorted(explainer.FALLBACK_KEYS)))
    OUT.mkdir(parents=True, exist_ok=True)
    for question, _, svg in rendered:
        (OUT / f"question-{question:03d}.svg").write_text(svg, encoding="utf-8")
    signatures = {(scene_spec(question)[0], scene_spec(question)[1]) for question, _ in all_entries}
    print(f"generated={len(rendered)} explicit_scenes={len(signatures)} output={OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
