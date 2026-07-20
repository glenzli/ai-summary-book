#!/usr/bin/env python3
"""Annotated, specimen-led plates for vertebrate and migration questions."""

from __future__ import annotations

import math

from textbook_illustrations import INK, circle, ellipse, line, panel_title, path, rect, text


VERTEBRATE_QUESTIONS = frozenset(range(116, 152))
VERTEBRATE_KEYS: dict[int, tuple[str, ...]] = {
    question: (f"vertebrate-plate-{question:03d}",) for question in VERTEBRATE_QUESTIONS
}

BLUE = "#4C86B8"
BLUE_DARK = "#2D618F"
TEAL = "#2D8079"
TEAL_LIGHT = "#8BC9BF"
GREEN = "#5C9659"
GREEN_LIGHT = "#A8CA83"
ORANGE = "#D9693F"
YELLOW = "#E6B542"
RED = "#C95555"
PURPLE = "#786AA6"
BROWN = "#8B6549"
BROWN_DARK = "#57483D"
CREAM = "#F6E9CF"
WATER = "#BDE4EA"
SKY = "#DDEEF4"
SOIL = "#B98A63"
WHITE = "#FFFFFF"


def polygon(points: list[tuple[float, float]], fill: str, stroke: str = INK, width: float = 3) -> str:
    values = " ".join(f"{x},{y}" for x, y in points)
    return f'<polygon points="{values}" fill="{fill}" stroke="{stroke}" stroke-width="{width}" stroke-linejoin="round"/>'


def label(tx: float, ty: float, x: float, y: float, value: str, color: str = INK) -> str:
    """Compact two-line callouts that remain legible in contact-sheet review."""
    end_x = x - 12 if x >= tx else x + 12
    chunks = [value]
    if len(value) > 15:
        split = min(range(7, len(value) - 5), key=lambda index: abs(index - len(value) / 2))
        chunks = [value[:split], value[split:]]
    copy = "".join(text(x, y + index * 22, chunk, 19, color, "start", 650) for index, chunk in enumerate(chunks))
    return line(tx, ty, end_x, y - 6, color, 2.5) + circle(tx, ty, 4.5, color) + copy


def arrow(data: str, color: str = INK, width: float = 4) -> str:
    return path(data, "none", color, width, True)


def land_backdrop(water: bool = False, night: bool = False) -> str:
    sky = "#253452" if night else SKY
    bits = [rect(34, 82, 892, 382, sky, "none", 0, 4)]
    if night:
        for index in range(18):
            bits.append(circle(70 + (index * 83) % 820, 105 + (index * 47) % 195, 2 + index % 3, WHITE if index % 2 else YELLOW))
    if water:
        bits.extend([
            path("M34,260 Q180,230 320,260 T610,260 T926,260 V464 H34 Z", WATER),
            path("M34,278 Q175,248 320,278 T610,278 T926,278", "none", WHITE, 5),
            path("M34,425 Q200,385 360,425 T690,425 T1010,425 V464 H34 Z", "#D7C58F"),
        ])
    else:
        bits.extend([
            path("M34,350 Q180,285 335,350 T650,350 T960,350 V464 H34 Z", GREEN_LIGHT),
            path("M34,425 Q190,382 360,425 T700,425 T1040,425 V464 H34 Z", GREEN),
        ])
    return "".join(bits)


def pond_backdrop() -> str:
    return "".join([
        rect(34, 82, 892, 382, SKY, "none", 0, 4),
        path("M34,330 Q190,270 340,330 T650,330 T960,330 V464 H34 Z", GREEN_LIGHT),
        ellipse(430, 390, 300, 63, WATER, BLUE, 3),
        path("M160,370 q18,-72 36,0 M680,378 q20,-88 42,0 M735,380 q15,-62 30,0", "none", GREEN, 8),
    ])


def underwater_backdrop() -> str:
    bits = [rect(34, 82, 892, 382, WATER, "none", 0, 4)]
    bits.extend([
        path("M34,420 Q190,375 350,420 T680,420 T1000,420 V464 H34 Z", "#D4C28A"),
        path("M95,425 q18,-85 36,0 M770,425 q20,-95 40,0 M830,425 q17,-70 34,0", "none", TEAL, 8),
    ])
    for x, y, r in ((120, 160, 7), (150, 125, 4), (810, 175, 7), (845, 135, 4), (750, 240, 5)):
        bits.append(circle(x, y, r, "none", WHITE, 2))
    return "".join(bits)


def frog(cx: float, cy: float, s: float = 1.0, jumping: bool = False) -> str:
    bits = [
        ellipse(cx, cy, 78 * s, 58 * s, GREEN, INK, 4 * s),
        circle(cx - 45 * s, cy - 46 * s, 25 * s, GREEN, INK, 4 * s),
        circle(cx + 45 * s, cy - 46 * s, 25 * s, GREEN, INK, 4 * s),
        circle(cx - 46 * s, cy - 50 * s, 8 * s, INK),
        circle(cx + 46 * s, cy - 50 * s, 8 * s, INK),
        path(f"M{cx-34*s},{cy+24*s} Q{cx},{cy+45*s} {cx+34*s},{cy+24*s}", "none", CREAM, 5 * s),
    ]
    if jumping:
        bits.extend([
            path(f"M{cx-45*s},{cy+28*s} L{cx-135*s},{cy+76*s} L{cx-175*s},{cy+48*s}", "none", INK, 15 * s),
            path(f"M{cx+45*s},{cy+28*s} L{cx+135*s},{cy+76*s} L{cx+178*s},{cy+48*s}", "none", INK, 15 * s),
            line(cx - 175 * s, cy + 48 * s, cx - 215 * s, cy + 40 * s, GREEN, 9 * s),
            line(cx + 178 * s, cy + 48 * s, cx + 218 * s, cy + 40 * s, GREEN, 9 * s),
        ])
    else:
        bits.extend([
            path(f"M{cx-42*s},{cy+25*s} L{cx-105*s},{cy+78*s} L{cx-145*s},{cy+55*s}", "none", INK, 12 * s),
            path(f"M{cx+42*s},{cy+25*s} L{cx+105*s},{cy+78*s} L{cx+145*s},{cy+55*s}", "none", INK, 12 * s),
        ])
    return "".join(bits)


def tadpole(cx: float, cy: float, s: float = 1.0, legs: bool = False) -> str:
    bits = [ellipse(cx, cy, 35 * s, 29 * s, BROWN_DARK, INK, 2.5 * s), path(f"M{cx+28*s},{cy} C{cx+80*s},{cy-42*s} {cx+118*s},{cy+36*s} {cx+150*s},{cy-5*s}", "none", BROWN_DARK, 15 * s)]
    if legs:
        bits.extend([line(cx - 8 * s, cy + 18 * s, cx - 35 * s, cy + 48 * s, GREEN, 7 * s), line(cx + 10 * s, cy + 18 * s, cx + 40 * s, cy + 45 * s, GREEN, 7 * s)])
    return "".join(bits)


def lizard(cx: float, cy: float, s: float = 1.0, gecko: bool = False, chameleon: bool = False) -> str:
    color = TEAL if chameleon else GREEN if gecko else "#B97A48"
    bits = [
        path(f"M{cx-105*s},{cy} C{cx-75*s},{cy-48*s} {cx+35*s},{cy-48*s} {cx+96*s},{cy-3*s} C{cx+35*s},{cy+45*s} {cx-70*s},{cy+48*s} {cx-105*s},{cy} Z", color, INK, 4 * s),
        ellipse(cx + 105 * s, cy - 4 * s, 48 * s, 36 * s, color, INK, 4 * s),
        circle(cx + 120 * s, cy - 12 * s, 5 * s, INK),
    ]
    if chameleon:
        bits.extend([
            path(f"M{cx-102*s},{cy} C{cx-175*s},{cy-70*s} {cx-220*s},{cy+40*s} {cx-150*s},{cy+52*s} C{cx-95*s},{cy+62*s} {cx-95*s},{cy+8*s} {cx-130*s},{cy+10*s}", "none", color, 16 * s),
            polygon([(cx - 25 * s, cy - 45 * s), (cx + 10 * s, cy - 83 * s), (cx + 45 * s, cy - 42 * s)], ORANGE, INK, 3 * s),
        ])
    else:
        bits.append(path(f"M{cx-102*s},{cy} C{cx-170*s},{cy-20*s} {cx-220*s},{cy-78*s} {cx-270*s},{cy-55*s}", "none", color, 17 * s))
    for side in (-1, 1):
        bits.extend([
            path(f"M{cx-45*s},{cy+side*20*s} L{cx-92*s},{cy+side*65*s} L{cx-125*s},{cy+side*75*s}", "none", INK, 8 * s),
            path(f"M{cx+38*s},{cy+side*18*s} L{cx+70*s},{cy+side*64*s} L{cx+110*s},{cy+side*72*s}", "none", INK, 8 * s),
        ])
        if gecko:
            for x in (cx - 125 * s, cx + 110 * s):
                bits.append(ellipse(x, cy + side * 75 * s, 18 * s, 7 * s, CREAM, INK, 2 * s))
    return "".join(bits)


def feather(cx: float, cy: float, s: float = 1.0, owl: bool = False) -> str:
    bits = [path(f"M{cx},{cy+170*s} Q{cx+10*s},{cy} {cx+45*s},{cy-160*s}", "none", BROWN_DARK, 8 * s)]
    bits.extend([
        path(f"M{cx+8*s},{cy+100*s} Q{cx-90*s},{cy+45*s} {cx-135*s},{cy-55*s} Q{cx-55*s},{cy-118*s} {cx+40*s},{cy-145*s}", CREAM, INK, 4 * s),
        path(f"M{cx+12*s},{cy+96*s} Q{cx+95*s},{cy+35*s} {cx+120*s},{cy-70*s} Q{cx+82*s},{cy-126*s} {cx+42*s},{cy-150*s}", WHITE, INK, 4 * s),
    ])
    for offset in range(-110, 85, 30):
        bits.append(line(cx + 20 * s, cy + offset * s, cx - (80 + offset / 5) * s, cy + (offset - 35) * s, BLUE if owl else RED, 2.5 * s))
    if owl:
        for index in range(8):
            x = cx - 130 * s + index * 15 * s
            bits.append(path(f"M{x},{cy-55*s} q{8*s},{-16*s} {16*s},0", "none", BROWN_DARK, 2.5 * s))
    return "".join(bits)


def bird(cx: float, cy: float, s: float = 1.0, kind: str = "bird") -> str:
    if kind == "penguin":
        return "".join([
            ellipse(cx, cy, 73 * s, 118 * s, "#263A58", INK, 4 * s),
            ellipse(cx, cy + 22 * s, 48 * s, 85 * s, WHITE, "none"),
            polygon([(cx - 52 * s, cy - 65 * s), (cx - 110 * s, cy + 55 * s), (cx - 35 * s, cy + 24 * s)], BLUE, INK, 3 * s),
            polygon([(cx + 52 * s, cy - 65 * s), (cx + 110 * s, cy + 55 * s), (cx + 35 * s, cy + 24 * s)], BLUE, INK, 3 * s),
            polygon([(cx + 12 * s, cy - 72 * s), (cx + 48 * s, cy - 56 * s), (cx + 12 * s, cy - 45 * s)], YELLOW, INK, 2.5 * s),
            ellipse(cx - 34 * s, cy + 118 * s, 38 * s, 13 * s, ORANGE, INK, 3 * s),
            ellipse(cx + 34 * s, cy + 118 * s, 38 * s, 13 * s, ORANGE, INK, 3 * s),
        ])
    if kind == "flamingo":
        return "".join([
            ellipse(cx - 25 * s, cy, 74 * s, 53 * s, "#EA718C", INK, 4 * s),
            path(f"M{cx+10*s},{cy-35*s} C{cx+85*s},{cy-95*s} {cx+95*s},{cy-180*s} {cx+46*s},{cy-190*s}", "none", "#EA718C", 18 * s),
            circle(cx + 40 * s, cy - 192 * s, 20 * s, "#EA718C", INK, 3 * s),
            path(f"M{cx+50*s},{cy-188*s} q{45*s},{5*s} {55*s},{25*s} q{-20*s},{18*s} {-50*s},{4*s}", CREAM, INK, 3 * s),
            path(f"M{cx-50*s},{cy+38*s} L{cx-52*s},{cy+180*s} L{cx-20*s},{cy+180*s} M{cx},{cy+42*s} L{cx+26*s},{cy+120*s} L{cx+65*s},{cy+120*s}", "none", "#D95678", 7 * s),
        ])
    body = "#3E77AF" if kind != "duck" else "#7D8B55"
    bits = [
        ellipse(cx, cy, 105 * s, 64 * s, body, INK, 4 * s),
        circle(cx + 84 * s, cy - 48 * s, 42 * s, ORANGE if kind == "woodpecker" else body, INK, 4 * s),
        polygon([(cx + 118 * s, cy - 52 * s), (cx + 170 * s, cy - 35 * s), (cx + 119 * s, cy - 20 * s)], YELLOW, INK, 3 * s),
        polygon([(cx - 95 * s, cy - 10 * s), (cx - 165 * s, cy - 55 * s), (cx - 142 * s, cy + 25 * s)], RED, INK, 3 * s),
        path(f"M{cx-62*s},{cy-8*s} Q{cx},{cy-90*s} {cx+48*s},{cy-2*s} Q{cx},{cy+38*s} {cx-62*s},{cy-8*s} Z", CREAM, INK, 3 * s),
        circle(cx + 92 * s, cy - 55 * s, 5 * s, INK),
    ]
    if kind == "woodpecker":
        bits.append(polygon([(cx + 70 * s, cy - 88 * s), (cx + 92 * s, cy - 126 * s), (cx + 112 * s, cy - 82 * s)], RED, INK, 3 * s))
    return "".join(bits)


def owl(cx: float, cy: float, s: float = 1.0) -> str:
    bits = [
        ellipse(cx, cy + 30 * s, 88 * s, 113 * s, BROWN, INK, 4 * s),
        polygon([(cx - 70 * s, cy - 45 * s), (cx - 65 * s, cy - 120 * s), (cx - 20 * s, cy - 67 * s)], BROWN, INK, 4 * s),
        polygon([(cx + 70 * s, cy - 45 * s), (cx + 65 * s, cy - 120 * s), (cx + 20 * s, cy - 67 * s)], BROWN, INK, 4 * s),
        circle(cx - 34 * s, cy - 25 * s, 35 * s, CREAM, INK, 4 * s),
        circle(cx + 34 * s, cy - 25 * s, 35 * s, CREAM, INK, 4 * s),
        circle(cx - 34 * s, cy - 25 * s, 12 * s, INK), circle(cx + 34 * s, cy - 25 * s, 12 * s, INK),
        polygon([(cx, cy + 5 * s), (cx - 14 * s, cy + 28 * s), (cx + 14 * s, cy + 28 * s)], YELLOW, INK, 2.5 * s),
    ]
    for side in (-1, 1):
        bits.append(path(f"M{cx+side*55*s},{cy+20*s} Q{cx+side*115*s},{cy+55*s} {cx+side*68*s},{cy+118*s}", "none", "#6D503D", 18 * s))
    return "".join(bits)


def quadruped(cx: float, cy: float, s: float, kind: str) -> str:
    colors = {
        "elephant": "#7D8B91", "giraffe": "#E2B55B", "zebra": WHITE,
        "camel": "#C59663", "polar-bear": WHITE, "panda": WHITE,
        "kangaroo": "#B98058", "sloth": "#8C765E", "hedgehog": "#8A674A",
        "rabbit": "#C6B8A4", "cat": "#D38A4A", "dog": "#B68155",
        "cow": WHITE, "horse": "#9A6649",
    }
    color = colors[kind]
    body_y = cy
    body_rx, body_ry = 120 * s, 62 * s
    neck_w, neck_h = 28 * s, 68 * s
    if kind == "giraffe":
        neck_h, neck_w = 190 * s, 38 * s
    elif kind == "kangaroo":
        body_rx, body_ry, neck_h = 78 * s, 98 * s, 90 * s
    elif kind == "rabbit":
        body_rx, body_ry, neck_h = 92 * s, 62 * s, 48 * s
    bits = [ellipse(cx, body_y, body_rx, body_ry, color, INK, 4 * s)]
    if kind == "camel":
        bits.extend([path(f"M{cx-70*s},{cy-35*s} Q{cx-32*s},{cy-125*s} {cx+5*s},{cy-42*s} Q{cx+40*s},{cy-120*s} {cx+78*s},{cy-35*s}", color, INK, 4 * s)])
    head_x = cx + (105 if kind != "kangaroo" else 65) * s
    head_y = cy - neck_h
    bits.extend([
        path(f"M{cx+72*s},{cy-25*s} Q{head_x-20*s},{head_y+35*s} {head_x},{head_y+15*s}", "none", color, neck_w),
        ellipse(head_x + 18 * s, head_y, 48 * s, 39 * s, color, INK, 4 * s),
        circle(head_x + 35 * s, head_y - 8 * s, 5 * s, INK),
    ])
    leg_xs = (-72, -25, 42, 82)
    if kind == "kangaroo":
        leg_xs = (-40, 35)
    for index, dx in enumerate(leg_xs):
        lean = (-10 if index < len(leg_xs) / 2 else 13) * s
        bits.append(path(f"M{cx+dx*s},{cy+42*s} L{cx+dx*s+lean},{cy+145*s} L{cx+dx*s+(lean-10*s)},{cy+153*s}", "none", INK, 11 * s))
    bits.append(path(f"M{cx-115*s},{cy-15*s} Q{cx-175*s},{cy-50*s} {cx-190*s},{cy+10*s}", "none", color, 14 * s))
    if kind == "elephant":
        bits.extend([
            ellipse(head_x - 15 * s, head_y + 5 * s, 48 * s, 59 * s, "#909CA1", INK, 3 * s),
            path(f"M{head_x+46*s},{head_y+5*s} C{head_x+85*s},{head_y+65*s} {head_x+62*s},{head_y+145*s} {head_x+28*s},{head_y+145*s}", "none", color, 23 * s),
        ])
    elif kind == "giraffe":
        bits.extend([polygon([(head_x - 8 * s, head_y - 33 * s), (head_x - 18 * s, head_y - 70 * s), (head_x + 1 * s, head_y - 36 * s)], color, INK, 3 * s), polygon([(head_x + 22 * s, head_y - 35 * s), (head_x + 30 * s, head_y - 72 * s), (head_x + 40 * s, head_y - 32 * s)], color, INK, 3 * s)])
        for dx, dy in [(-70, -10), (-25, 20), (20, -15), (65, 10), (102, -95), (108, -145)]:
            bits.append(circle(cx + dx * s, cy + dy * s, 12 * s, BROWN))
    elif kind == "zebra":
        for dx in (-80, -45, -10, 25, 60):
            bits.append(path(f"M{cx+dx*s},{cy-48*s} L{cx+(dx+18)*s},{cy+48*s}", "none", INK, 9 * s))
    elif kind == "polar-bear":
        bits.extend([circle(head_x - 12 * s, head_y - 34 * s, 14 * s, WHITE, INK, 3 * s), circle(head_x + 22 * s, head_y - 35 * s, 14 * s, WHITE, INK, 3 * s)])
    elif kind == "panda":
        bits.extend([ellipse(cx - 20 * s, cy, 55 * s, 60 * s, INK), circle(head_x - 10 * s, head_y - 34 * s, 18 * s, INK), circle(head_x + 28 * s, head_y - 34 * s, 18 * s, INK), ellipse(head_x + 1 * s, head_y - 5 * s, 15 * s, 23 * s, INK), ellipse(head_x + 36 * s, head_y - 5 * s, 15 * s, 23 * s, INK)])
    elif kind == "kangaroo":
        bits.extend([path(f"M{cx-72*s},{cy+15*s} Q{cx-190*s},{cy+75*s} {cx-250*s},{cy+130*s}", "none", color, 24 * s), ellipse(cx + 35 * s, cy + 20 * s, 42 * s, 52 * s, CREAM, INK, 3 * s), circle(cx + 42 * s, cy + 10 * s, 18 * s, BROWN, INK, 3 * s)])
    elif kind == "hedgehog":
        for index in range(12):
            x = cx - 100 * s + index * 18 * s
            bits.append(polygon([(x, cy - 35 * s), (x + 10 * s, cy - (80 + index % 3 * 18) * s), (x + 22 * s, cy - 30 * s)], BROWN_DARK, INK, 2 * s))
    elif kind == "rabbit":
        bits.extend([ellipse(head_x - 8 * s, head_y - 58 * s, 16 * s, 67 * s, color, INK, 3 * s), ellipse(head_x + 25 * s, head_y - 60 * s, 16 * s, 69 * s, color, INK, 3 * s), circle(cx - 112 * s, cy - 12 * s, 23 * s, WHITE, INK, 3 * s)])
    elif kind == "cat":
        bits.extend([polygon([(head_x - 18 * s, head_y - 28 * s), (head_x - 8 * s, head_y - 70 * s), (head_x + 7 * s, head_y - 28 * s)], color, INK, 3 * s), polygon([(head_x + 18 * s, head_y - 28 * s), (head_x + 34 * s, head_y - 70 * s), (head_x + 42 * s, head_y - 25 * s)], color, INK, 3 * s)])
        for dy in (-12, 4):
            bits.extend([line(head_x + 42 * s, head_y + dy * s, head_x + 100 * s, head_y + (dy - 8) * s, INK, 2 * s), line(head_x + 42 * s, head_y + dy * s, head_x + 100 * s, head_y + (dy + 12) * s, INK, 2 * s)])
    elif kind == "cow":
        bits.extend([path(f"M{head_x-15*s},{head_y-35*s} q{-35*s},{-35*s} {-65*s},{-10*s} M{head_x+15*s},{head_y-35*s} q{35*s},{-35*s} {65*s},{-10*s}", "none", YELLOW, 7 * s), ellipse(cx - 55 * s, cy - 5 * s, 35 * s, 28 * s, INK), ellipse(cx + 35 * s, cy + 10 * s, 38 * s, 25 * s, INK)])
    elif kind == "horse":
        bits.append(path(f"M{head_x-40*s},{head_y-35*s} Q{head_x-75*s},{head_y-12*s} {head_x-48*s},{head_y+50*s}", "none", BROWN_DARK, 16 * s))
    return "".join(bits)


def whale(cx: float, cy: float, s: float = 1.0, dolphin: bool = False) -> str:
    color = BLUE if dolphin else BLUE_DARK
    snout = 155 if dolphin else 118
    bits = [
        path(f"M{cx-150*s},{cy} C{cx-70*s},{cy-95*s} {cx+80*s},{cy-82*s} {cx+snout*s},{cy-15*s} C{cx+58*s},{cy+72*s} {cx-90*s},{cy+80*s} {cx-150*s},{cy} Z", color, INK, 4 * s),
        polygon([(cx - 145 * s, cy), (cx - 225 * s, cy - 70 * s), (cx - 210 * s, cy + 10 * s)], RED, INK, 3 * s),
        polygon([(cx - 145 * s, cy), (cx - 220 * s, cy + 75 * s), (cx - 205 * s, cy - 4 * s)], ORANGE, INK, 3 * s),
        polygon([(cx - 15 * s, cy + 38 * s), (cx + 40 * s, cy + 105 * s), (cx + 60 * s, cy + 42 * s)], color, INK, 3 * s),
        circle(cx + 85 * s, cy - 24 * s, 5 * s, INK),
    ]
    if not dolphin:
        bits.extend([path(f"M{cx+8*s},{cy-65*s} q{-8*s},{-58*s} {-30*s},{-95*s} M{cx+8*s},{cy-65*s} q{18*s},{-58*s} {45*s},{-92*s}", "none", WHITE, 5 * s)])
    return "".join(bits)


def bat(cx: float, cy: float, s: float = 1.0) -> str:
    return "".join([
        ellipse(cx, cy, 35 * s, 80 * s, BROWN_DARK, INK, 4 * s),
        polygon([(cx - 25 * s, cy - 35 * s), (cx - 110 * s, cy - 105 * s), (cx - 210 * s, cy - 70 * s), (cx - 160 * s, cy + 15 * s), (cx - 90 * s, cy + 62 * s), (cx - 18 * s, cy + 25 * s)], PURPLE, INK, 4 * s),
        polygon([(cx + 25 * s, cy - 35 * s), (cx + 110 * s, cy - 105 * s), (cx + 210 * s, cy - 70 * s), (cx + 160 * s, cy + 15 * s), (cx + 90 * s, cy + 62 * s), (cx + 18 * s, cy + 25 * s)], PURPLE, INK, 4 * s),
        polygon([(cx - 22 * s, cy - 68 * s), (cx - 10 * s, cy - 120 * s), (cx + 3 * s, cy - 72 * s)], BROWN_DARK, INK, 3 * s),
        polygon([(cx + 22 * s, cy - 68 * s), (cx + 10 * s, cy - 120 * s), (cx - 3 * s, cy - 72 * s)], BROWN_DARK, INK, 3 * s),
    ])


def fish(cx: float, cy: float, s: float = 1.0, shark: bool = False) -> str:
    color = "#657D8B" if shark else BLUE
    bits = [
        path(f"M{cx-135*s},{cy} C{cx-65*s},{cy-82*s} {cx+70*s},{cy-72*s} {cx+142*s},{cy} C{cx+62*s},{cy+72*s} {cx-68*s},{cy+78*s} {cx-135*s},{cy} Z", color, INK, 4 * s),
        polygon([(cx - 132 * s, cy), (cx - 205 * s, cy - 78 * s), (cx - 195 * s, cy + 76 * s)], RED if not shark else color, INK, 3 * s),
        polygon([(cx - 18 * s, cy - 65 * s), (cx + 20 * s, cy - 130 * s), (cx + 50 * s, cy - 54 * s)], color, INK, 3 * s),
        polygon([(cx + 5 * s, cy + 45 * s), (cx + 62 * s, cy + 105 * s), (cx + 78 * s, cy + 38 * s)], color, INK, 3 * s),
        circle(cx + 92 * s, cy - 17 * s, 6 * s, INK),
    ]
    if shark:
        for index in range(5):
            bits.append(line(cx + (48 + index * 11) * s, cy + 8 * s, cx + (42 + index * 11) * s, cy + 43 * s, CREAM, 3 * s))
    return "".join(bits)


def octopus(cx: float, cy: float, s: float = 1.0) -> str:
    bits = [path(f"M{cx-72*s},{cy+20*s} Q{cx-78*s},{cy-112*s} {cx},{cy-145*s} Q{cx+78*s},{cy-112*s} {cx+72*s},{cy+20*s} Z", ORANGE, INK, 4 * s)]
    bits.extend([circle(cx - 25 * s, cy - 50 * s, 9 * s, WHITE, INK, 2 * s), circle(cx + 25 * s, cy - 50 * s, 9 * s, WHITE, INK, 2 * s)])
    for index in range(8):
        start_x = cx - 60 * s + index * 17 * s
        end_x = cx + (-170 + index * 48) * s
        end_y = cy + (95 + (index % 3) * 35) * s
        bits.append(path(f"M{start_x},{cy+5*s} C{start_x-10*s},{cy+60*s} {end_x+25*s},{end_y-25*s} {end_x},{end_y}", "none", ORANGE if index % 2 else RED, 16 * s))
        for dot in range(3):
            t = (dot + 1) / 4
            bits.append(circle(start_x * (1 - t) + end_x * t, (cy + 10 * s) * (1 - t) + end_y * t, 4 * s, CREAM, INK, 1 * s))
    return "".join(bits)


def _plate_116() -> str:
    bits = [panel_title("青蛙的一生跨过水中和陆上两个环境"), pond_backdrop()]
    bits.extend([
        circle(145, 340, 12, CREAM, INK, 2), circle(170, 352, 12, CREAM, INK, 2), circle(196, 338, 12, CREAM, INK, 2),
        tadpole(315, 360, 0.48), tadpole(495, 350, 0.50, True), frog(705, 320, 0.72),
        arrow("M205,340 Q250,310 275,345", TEAL), arrow("M380,350 Q425,315 455,342", TEAL), arrow("M560,342 Q610,300 640,322", TEAL),
        text(165, 445, "卵", 21, INK, "middle"), text(345, 445, "有鳃和尾的蝌蚪", 21, INK, "middle"),
        text(505, 445, "长出四肢", 21, INK, "middle"), text(720, 445, "用肺和皮肤呼吸", 21, INK, "middle"),
    ])
    return "".join(bits)


def _plate_117() -> str:
    bits = [panel_title("粗壮后肢先折叠储能，再快速伸直推地"), land_backdrop(), frog(430, 285, 1.0, True)]
    bits.extend([
        arrow("M250,390 Q190,315 230,235", RED, 6),
        label(320, 335, 675, 155, "长后腿提供更长的推地距离"),
        label(247, 347, 675, 260, "脚掌向后推地，身体受到向前上方的力"),
        label(438, 235, 675, 385, "落地时前肢帮助缓冲"),
    ])
    return "".join(bits)


def _plate_118() -> str:
    bits = [panel_title("壁虎脚趾下密集的细毛贴近墙面"), rect(34, 82, 892, 382, "#E9E4D8", "none", 0, 4)]
    bits.extend([
        rect(80, 100, 470, 330, "#D7D0C3", INK, 3),
        f'<g transform="translate(340 270) rotate(-22)">{lizard(0, 0, 0.72, gecko=True)}</g>',
        rect(640, 135, 215, 205, CREAM, INK, 3, 12),
    ])
    for row in range(5):
        for col in range(8):
            x = 662 + col * 23
            y = 172 + row * 30
            bits.append(path(f"M{x},{y+12} q7,-18 14,0", "none", TEAL, 2.2))
    bits.extend([
        label(300, 335, 650, 100, "每个脚趾都有许多褶皱"),
        label(735, 245, 620, 395, "褶皱上是成排的微小刚毛"),
        text(474, 487, "不是吸盘，也不是胶水；关键是极大的贴近面积", 21, INK, "middle", 650),
    ])
    return "".join(bits)


def _plate_119() -> str:
    bits = [panel_title("断尾处有预设的薄弱面，尾巴肌肉还能短时抽动"), land_backdrop(), lizard(405, 260, 0.68)]
    bits.extend([
        path("M185,235 C130,210 105,240 80,275", "none", ORANGE, 15),
        line(192, 220, 203, 264, RED, 5, False, "6 6"),
        path("M105,245 q-30,-25 -50,5 q25,20 45,35", "none", RED, 4),
        label(197, 242, 660, 150, "尾椎间有容易分开的区域"),
        label(105, 245, 660, 260, "分离后的尾段会暂时扭动，吸引捕食者"),
        label(340, 265, 660, 385, "新长出的尾通常不完全等同原来的尾"),
    ])
    return "".join(bits)


def _plate_120() -> str:
    bits = [panel_title("变色龙皮肤会调色，也会传递温度和情绪信息"), land_backdrop(), path("M105,350 Q360,275 620,350", "none", BROWN, 20), lizard(365, 270, 0.75, chameleon=True)]
    bits.extend([
        rect(650, 130, 220, 225, CREAM, INK, 3, 12),
        text(760, 165, "皮肤放大", 20, INK, "middle", 720),
    ])
    for row, color in enumerate((YELLOW, RED, BLUE, TEAL)):
        for col in range(5):
            bits.append(circle(680 + col * 38, 205 + row * 36, 12, color, INK, 2))
    bits.extend([
        label(390, 245, 660, 405, "色素与反光晶体共同改变看到的颜色"),
        text(405, 490, "颜色还会随光照、体温、紧张和求偶状态改变", 20, INK, "middle", 650),
    ])
    return "".join(bits)


def _plate_121() -> str:
    return "".join([
        panel_title("一根羽毛由羽轴、羽枝和互相扣住的小钩组成"),
        feather(335, 285, 1.02),
        rect(625, 130, 240, 225, CREAM, INK, 3, 12),
        line(660, 315, 820, 175, BROWN_DARK, 5),
        line(690, 292, 650, 245, BLUE, 3), line(720, 265, 680, 217, BLUE, 3), line(750, 238, 710, 190, BLUE, 3),
        line(720, 265, 770, 300, RED, 3), line(750, 238, 800, 273, RED, 3),
        label(355, 300, 675, 405, "羽轴支撑整片羽毛"),
        label(738, 250, 650, 105, "小钩把相邻羽枝扣成连续的羽片"),
        text(410, 500, "飞羽推空气，绒羽留住空气，体表羽毛也能防水和展示", 20, INK, "middle", 650),
    ])


def _plate_122() -> str:
    bits = [panel_title("翅膀向下后方推空气，空气也把鸟托向上前方"), land_backdrop(), bird(420, 255, 1.05)]
    bits.extend([
        path("M320,255 Q405,125 520,210 Q435,250 335,290 Z", CREAM, INK, 4),
        arrow("M385,230 Q390,120 430,90", BLUE, 6), arrow("M420,285 Q430,385 385,420", RED, 6),
        label(395, 215, 665, 135, "弯曲的翼面让气流发生偏转"),
        label(400, 115, 665, 260, "向上的合力必须抵消体重"),
        label(430, 335, 665, 385, "拍翼同时提供向前的推力"),
    ])
    return "".join(bits)


def _plate_123() -> str:
    bits = [panel_title("企鹅把翅膀变成鳍状肢，在水下像飞行一样划水"), underwater_backdrop(), bird(370, 270, 1.0, "penguin")]
    bits.extend([
        arrow("M260,275 Q170,230 120,275", TEAL, 5),
        label(270, 255, 650, 145, "短而扁的翅膀不适合空中飞行"),
        label(330, 340, 650, 270, "坚实翼骨和肌肉推动鳍状肢"),
        label(470, 300, 650, 395, "流线身体和防水羽毛减少阻力"),
    ])
    return "".join(bits)


def _plate_124() -> str:
    bits = [panel_title("鸭子整理羽毛时把油脂涂开，重叠羽片挡住水"), pond_backdrop(), bird(365, 310, 0.95, "duck")]
    bits.extend([
        circle(560, 300, 18, BLUE, WHITE, 3), circle(600, 270, 13, BLUE, WHITE, 3),
        rect(650, 140, 215, 205, CREAM, INK, 3, 12),
        path("M675,285 Q720,160 760,285 Q800,160 840,285", "none", BROWN, 16),
        circle(720, 225, 17, BLUE, WHITE, 3), circle(795, 215, 14, BLUE, WHITE, 3),
        label(752, 225, 640, 405, "水珠停在羽片外，不容易进入绒羽层"),
        text(390, 490, "羽毛不是完全不湿；外层结构与油脂共同减慢进水", 20, INK, "middle", 650),
    ])
    return "".join(bits)


def _plate_125() -> str:
    bits = [panel_title("啄木鸟用直线啄击、强壮颈肌和特殊头部结构分散冲击"), land_backdrop(), rect(475, 90, 92, 365, BROWN, INK, 4, 35), bird(360, 265, 0.82, "woodpecker")]
    bits.extend([
        path("M500,235 q40,-18 0,-36 q42,-18 0,-36", "none", ORANGE, 6),
        rect(665, 135, 205, 205, CREAM, INK, 3, 100),
        circle(765, 230, 68, "#E7D9BF", INK, 4), ellipse(775, 235, 42, 27, "#D88D9A", INK, 3),
        label(765, 230, 625, 400, "紧贴头骨的大脑与缓冲组织减少晃动"),
        label(415, 235, 630, 110, "喙尽量沿直线撞击树干"),
        text(405, 495, "这不是“完全不怕震”，而是多种结构一起降低损伤风险", 19, INK, "middle", 650),
    ])
    return "".join(bits)


def _plate_126() -> str:
    bits = [panel_title("猫头鹰飞羽边缘像梳子，把湍急气流拆成更小的涡"), land_backdrop(night=True), owl(300, 285, 0.85), feather(585, 280, 0.82, True)]
    bits.extend([
        label(475, 230, 715, 145, "前缘梳齿减少尖锐的破风声"),
        label(560, 330, 715, 270, "柔软绒面吸收一部分高频振动"),
        label(305, 310, 690, 405, "安静飞行帮助接近猎物，也帮助听清地面声音"),
    ])
    return "".join(bits)


def _plate_127() -> str:
    bits = [panel_title("火烈鸟从藻类和小甲壳动物中得到红橙色色素"), pond_backdrop(), bird(315, 315, 0.82, "flamingo")]
    bits.extend([
        ellipse(670, 240, 80, 50, WATER, BLUE, 3),
        path("M620,245 q25,-35 50,0 q25,35 50,0", "none", ORANGE, 9),
        circle(645, 205, 8, GREEN), circle(680, 205, 7, GREEN), circle(715, 215, 9, GREEN),
        arrow("M670,305 Q585,345 515,325", RED, 5),
        label(670, 240, 650, 120, "食物中的类胡萝卜素进入身体"),
        label(330, 260, 650, 405, "色素沉积在新长出的羽毛里"),
    ])
    return "".join(bits)


def _plate_128() -> str:
    bits = [panel_title("哺乳动物共同特征不是外形，而是毛和乳汁等身体结构"), land_backdrop()]
    bits.extend([
        quadruped(190, 320, 0.42, "cat"), bat(430, 270, 0.38), whale(700, 300, 0.43),
        label(195, 300, 90, 145, "身体至少在一生某阶段有毛"),
        label(430, 260, 365, 115, "蝙蝠会飞，仍是哺乳动物"),
        label(700, 300, 665, 130, "鲸住在海里，也用肺呼吸"),
        text(480, 485, "雌性有乳腺，幼仔喝乳汁；大多数种类还会生下幼仔", 21, INK, "middle", 680),
    ])
    return "".join(bits)


def _plate_129() -> str:
    bits = [panel_title("鲸用肺呼吸，必须把头顶的呼吸孔带到水面"), underwater_backdrop(), whale(390, 300, 0.9)]
    bits.extend([
        path("M410,230 q-20,-70 -42,-110 M410,230 q20,-72 50,-105", "none", WHITE, 7),
        line(34, 175, 926, 175, BLUE, 5),
        label(405, 230, 650, 115, "呼吸孔连接肺，不连接鳃"),
        label(390, 300, 650, 260, "潜水时储存氧气并减慢部分器官活动"),
        label(375, 175, 650, 400, "浮到水面快速呼气，再吸入新鲜空气"),
    ])
    return "".join(bits)


def _plate_130() -> str:
    bits = [panel_title("海豚发出短促声脉冲，再比较回声到达的方向和时间"), underwater_backdrop(), whale(350, 285, 0.72, True)]
    for radius in (70, 115, 160):
        bits.append(path(f"M{510},{230-radius/8:.1f} Q{510+radius},{285} {510},{340+radius/8:.1f}", "none", TEAL, 4))
    bits.extend([
        polygon([(820, 230), (875, 285), (815, 345)], SOIL, INK, 3),
        arrow("M790,285 Q660,205 535,250", RED, 4),
        label(470, 250, 650, 110, "额部聚焦咔嗒声"),
        label(820, 285, 650, 410, "回声从物体反射，再由下颌附近接收"),
    ])
    return "".join(bits)


def _plate_131() -> str:
    bits = [panel_title("蝙蝠能看见，也常用超声回声在黑暗中测距"), land_backdrop(night=True), bat(360, 285, 0.78)]
    for radius in (70, 110, 150):
        bits.append(path(f"M{470},{215-radius/8:.1f} Q{470+radius},{285} {470},{355+radius/8:.1f}", "none", TEAL_LIGHT, 4))
    bits.extend([
        circle(795, 250, 18, YELLOW, INK, 3),
        label(360, 205, 650, 115, "眼睛仍能感受光"),
        label(525, 285, 650, 250, "喉部发声，耳朵接收回声"),
        label(795, 250, 650, 405, "回声变化帮助判断昆虫的位置和运动"),
    ])
    return "".join(bits)


def _plate_quadruped(question: int, kind: str, title_value: str, labels: tuple[str, str, str]) -> str:
    bits = [panel_title(title_value), land_backdrop(), quadruped(360, 310, 0.78, kind)]
    anchors = {
        "elephant": ((475, 255), (290, 310), (385, 390)),
        "giraffe": ((450, 125), (330, 285), (305, 390)),
        "zebra": ((340, 285), (265, 325), (430, 350)),
        "camel": ((330, 205), (310, 295), (400, 380)),
        "polar-bear": ((450, 215), (330, 290), (280, 345)),
        "panda": ((410, 270), (300, 295), (380, 365)),
        "kangaroo": ((390, 330), (310, 255), (210, 390)),
        "sloth": ((430, 260), (320, 310), (250, 355)),
        "hedgehog": ((300, 245), (360, 310), (255, 365)),
        "rabbit": ((440, 190), (350, 285), (285, 360)),
        "cat": ((455, 205), (420, 260), (330, 320)),
        "dog": ((455, 220), (425, 265), (330, 320)),
        "cow": ((410, 260), (330, 300), (295, 365)),
        "horse": ((430, 215), (350, 315), (270, 390)),
    }
    for (tx, ty), y, copy in zip(anchors[kind], (130, 260, 395), labels):
        bits.append(label(tx, ty, 650, y, copy))
    return "".join(bits)


def _plate_146() -> str:
    bits = [panel_title("鱼让水从口进入、穿过鳃丝，再从鳃盖后流出"), underwater_backdrop(), fish(350, 285, 0.8)]
    bits.extend([
        path("M455,270 Q510,285 455,300", "none", RED, 16),
        arrow("M180,285 Q245,250 300,270", BLUE, 5), arrow("M455,285 Q525,275 565,300", BLUE, 5),
        rect(650, 145, 210, 205, CREAM, INK, 3, 12),
    ])
    for index in range(7):
        bits.append(path(f"M680,{180+index*22} q45,25 95,0", "none", RED, 5))
    bits.extend([label(455, 285, 640, 405, "鳃丝很薄，里面有密集血管交换氧和二氧化碳"), text(755, 375, "鳃丝放大", 19, INK, "middle")])
    return "".join(bits)


def _plate_147() -> str:
    bits = [panel_title("许多硬骨鱼调节鳔里的气体，改变平均密度"), underwater_backdrop(), fish(365, 285, 0.82)]
    bits.extend([
        ellipse(340, 282, 72, 31, CREAM, RED, 4),
        rect(650, 145, 210, 210, CREAM, INK, 3, 12),
        ellipse(705, 250, 32, 18, WHITE, RED, 3), ellipse(805, 250, 63, 31, WHITE, RED, 3),
        arrow("M745,250 L765,250", TEAL, 4),
        label(340, 282, 620, 410, "鳔变大时整体密度下降，较容易上浮"),
        text(755, 382, "气体少　　气体多", 19, INK, "middle"),
    ])
    return "".join(bits)


def _plate_148() -> str:
    bits = [panel_title("鲨鱼骨架主要由软骨组成，较轻且有韧性"), underwater_backdrop(), fish(335, 285, 0.78, True)]
    bits.extend([
        rect(625, 120, 245, 260, CREAM, INK, 3, 12),
        path("M665,250 Q730,150 820,215 Q760,320 665,250", "none", TEAL, 16),
        circle(710, 235, 10, "#D4E6D8", INK, 2), circle(755, 205, 10, "#D4E6D8", INK, 2), circle(780, 260, 10, "#D4E6D8", INK, 2),
        label(700, 230, 625, 420, "软骨有细胞和基质，但矿化程度通常低于硬骨"),
        text(750, 150, "软骨组织放大", 19, INK, "middle"),
    ])
    return "".join(bits)


def _plate_149() -> str:
    bits = [panel_title("章鱼腕没有骨头，三组不同方向的肌肉互相配合"), underwater_backdrop(), octopus(330, 260, 0.72)]
    bits.extend([
        rect(645, 135, 220, 220, CREAM, INK, 3, 12),
        ellipse(755, 245, 65, 65, "#E58C6C", INK, 4),
        circle(755, 245, 40, "none", RED, 9),
        line(690, 245, 820, 245, TEAL, 8), line(755, 180, 755, 310, TEAL, 8),
        path("M705,205 Q755,170 805,205 Q840,245 805,285 Q755,320 705,285 Q670,245 705,205", "none", YELLOW, 6),
        label(755, 245, 620, 405, "环形、纵向和斜向肌肉改变腕的长度、粗细和弯曲"),
    ])
    return "".join(bits)


def _plate_150() -> str:
    bits = [panel_title("一串脚印要和步距、方向、地面与生活环境一起读"), rect(34, 82, 892, 382, "#E8E1D4", "none", 0, 4)]
    tracks = [(150, 350, -18), (270, 300, 12), (390, 250, -16), (515, 205, 14), (650, 165, -12)]
    for x, y, angle in tracks:
        bits.append(f'<g transform="translate({x} {y}) rotate({angle})">{ellipse(0, 10, 25, 34, BROWN, INK, 3)}{circle(-24, -25, 12, BROWN, INK, 2)}{circle(0, -32, 12, BROWN, INK, 2)}{circle(24, -25, 12, BROWN, INK, 2)}</g>')
    bits.extend([
        arrow("M120,410 Q380,285 700,115", TEAL, 4),
        label(270, 300, 670, 225, "形状帮助判断脚掌和趾的类型"),
        label(515, 205, 670, 335, "前后间距与左右排列反映步态"),
        text(445, 495, "脚印能给出线索，通常不能单靠一枚印记确定物种", 20, INK, "middle", 650),
    ])
    return "".join(bits)


def _plate_151() -> str:
    bits = [panel_title("迁徙把繁殖地、食物地和季节性路线连接起来"), rect(34, 82, 892, 382, "#DCECF1", "none", 0, 4)]
    bits.extend([
        path("M90,170 Q210,105 330,165 Q420,225 510,170 Q610,90 715,150 Q820,205 890,155", GREEN_LIGHT, INK, 3),
        path("M130,360 Q220,300 315,365 Q420,420 515,355 Q625,285 760,350", GREEN, INK, 3),
        path("M205,330 C250,250 330,220 390,250 C465,280 500,185 585,175 C665,165 720,225 790,205", "none", RED, 6, True, "12 10"),
    ])
    for x, y in ((215, 320), (390, 250), (585, 175), (790, 205)):
        bits.append(path(f"M{x-18},{y} q18,-24 36,0 M{x+18},{y} q18,-24 36,0", "none", BLUE_DARK, 4))
    bits.extend([
        label(215, 320, 90, 440, "食物丰富的越冬或觅食地"),
        label(790, 205, 650, 110, "适合繁殖和育幼的地点"),
        label(500, 220, 650, 360, "太阳、星空、地标、气味和地磁都可能参与导航"),
    ])
    return "".join(bits)


RENDERERS = {
    116: _plate_116,
    117: _plate_117,
    118: _plate_118,
    119: _plate_119,
    120: _plate_120,
    121: _plate_121,
    122: _plate_122,
    123: _plate_123,
    124: _plate_124,
    125: _plate_125,
    126: _plate_126,
    127: _plate_127,
    128: _plate_128,
    129: _plate_129,
    130: _plate_130,
    131: _plate_131,
    132: lambda: _plate_quadruped(132, "elephant", "象鼻是一束没有骨头的肌肉，能缩短、伸长和弯曲", ("两类肌肉沿不同方向排列", "鼻端像手指一样抓取和感触", "也能吸入水后再喷出，但水不会进入肺")),
    133: lambda: _plate_quadruped(133, "giraffe", "长颈鹿喝水时张开前腿，循环系统要稳住脑部血压", ("长颈和头部形成很大的高度差", "强壮心脏把血送向高处", "瓣膜和血管调节低头、抬头时的压力变化")),
    134: lambda: _plate_quadruped(134, "zebra", "黑白条纹会打乱轮廓，也可能减少叮咬昆虫落脚", ("条纹由皮肤中色素细胞的分布形成", "群体站在一起时轮廓互相交叠", "条纹不是年龄刻度，每只个体图案不同")),
    135: lambda: _plate_quadruped(135, "camel", "驼峰主要储存脂肪，不是装水的水袋", ("脂肪集中在背部的驼峰", "代谢脂肪能提供能量并产生少量水", "节水还依靠浓缩尿液、耐受体温变化等能力")),
    136: lambda: _plate_quadruped(136, "polar-bear", "北极熊毛大多近乎透明，白色来自光的散射", ("外层护毛中空或半透明", "浓密绒毛夹住空气，减少散热", "毛下的皮肤颜色较深，有助吸收辐射")),
    137: lambda: _plate_quadruped(137, "panda", "大熊猫主要吃竹子，但消化道仍保留食肉目特点", ("宽大的臼齿碾碎竹纤维", "腕部增大的籽骨像拇指夹住竹竿", "竹子营养密度低，所以一天要花很久取食")),
    138: lambda: _plate_quadruped(138, "kangaroo", "袋鼠幼仔出生很小，会爬进育儿袋继续发育", ("幼仔沿母体腹部爬向袋口", "在育儿袋里含住乳头", "袋子保护幼仔，但不是另一个子宫")),
    139: lambda: _plate_quadruped(139, "sloth", "树懒以低能量叶片为食，代谢慢、消化时间长", ("弯曲长爪把身体挂在树枝下", "多室胃缓慢分解难消化的叶片", "动作慢可以减少能量消耗，也不等于永远不动")),
    140: lambda: _plate_quadruped(140, "hedgehog", "刺猬的刺是变硬的毛，受到威胁时竖起并蜷身", ("每根刺固定在皮肤里，不能主动射出", "背部肌肉收紧，使刺朝外", "柔软的脸和腹部被包在刺球里面")),
    141: lambda: _plate_quadruped(141, "rabbit", "兔子长耳既收集声音，也帮助身体散热", ("大耳廓把更多声音引向耳道", "耳皮很薄，里面有密集血管", "天气热时更多血液流过耳朵并向外散热")),
    142: lambda: _plate_quadruped(142, "cat", "猫胡须根部有丰富感觉神经，会感知弯曲和触碰", ("胡须比普通被毛更粗、更深地扎入皮肤", "空气和物体让胡须轻微弯曲", "大脑用多根胡须共同判断近处空间")),
    143: lambda: _plate_quadruped(143, "dog", "狗鼻腔内有大片嗅上皮，湿鼻面也能捕捉气味分子", ("鼻孔能让进入和呼出的气流分开", "卷曲鼻甲增加气味接触面积", "气味图样送入大脑，不是鼻尖单独在思考")),
    144: lambda: _plate_quadruped(144, "cow", "牛把草先吞入瘤胃，再反刍重新细嚼", ("瘤胃中的微生物分解植物纤维", "一团食物回到口中再次咀嚼", "随后进入其他胃室和肠道继续消化")),
    145: lambda: _plate_quadruped(145, "horse", "马腿部的肌腱和韧带能锁住部分关节，站立时较省力", ("站立装置减少肌肉持续用力", "马可以站着浅睡，便于快速逃跑", "进入较深睡眠时仍需要躺下")),
    146: _plate_146,
    147: _plate_147,
    148: _plate_148,
    149: _plate_149,
    150: _plate_150,
    151: _plate_151,
}


def vertebrate_body(question: int, palette: tuple[str, ...]) -> str:
    del palette
    try:
        return RENDERERS[question]()
    except KeyError as exc:
        raise KeyError(f"no vertebrate illustration for question {question}") from exc
