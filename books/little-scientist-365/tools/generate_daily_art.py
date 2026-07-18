#!/usr/bin/env python3
"""Generate deterministic, text-free SVG spot art for all 365 entries."""

from __future__ import annotations

import html
import math
import random
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "images" / "days"
DAY_RE = re.compile(r"^## 第 (\d{3}) 天｜(.+)$")

PALETTES = [
    ("#dff3ff", "#2864b7", "#f5c84c", "#ffffff", "#172d4d"),
    ("#e2f6f3", "#177f87", "#ef7d63", "#ffffff", "#213b4a"),
    ("#eef7df", "#3c8b4f", "#f1b64a", "#f28ba8", "#264c36"),
    ("#f4f5d7", "#6d913d", "#db704d", "#f4c84a", "#344525"),
    ("#fff0dc", "#3f7f76", "#dc6c55", "#f4b642", "#3f453f"),
    ("#e8f5f0", "#147d88", "#e66f67", "#f0b941", "#263e51"),
    ("#fff1df", "#3f6f99", "#dd7444", "#f2b544", "#303d4c"),
    ("#f4ead7", "#357761", "#2f78a6", "#d7984c", "#3f3a32"),
    ("#dff4f5", "#176f8c", "#ef745f", "#f4c74e", "#163c59"),
    ("#e9e5f7", "#46468c", "#f0b84b", "#6fb7c7", "#25264c"),
    ("#f8eadf", "#a84f45", "#3f7692", "#e0a63f", "#4b3734"),
    ("#e8f1e2", "#2f7863", "#3276a2", "#e3b33e", "#33413d"),
]


def month_for(day: int) -> int:
    starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    return max(index for index, start in enumerate(starts) if day >= start)


def entries() -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []
    for path in sorted(ROOT.glob("[01][0-9]_*.md")):
        if path.name.startswith("00_"):
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            match = DAY_RE.match(line)
            if match:
                found.append((int(match.group(1)), match.group(2)))
    return found


def circle(x: float, y: float, r: float, fill: str, stroke: str = "none", sw: int = 0) -> str:
    return f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'


def ellipse(x: float, y: float, rx: float, ry: float, fill: str, stroke: str = "none", sw: int = 0) -> str:
    return f'<ellipse cx="{x}" cy="{y}" rx="{rx}" ry="{ry}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'


def rect(x: float, y: float, w: float, h: float, fill: str, rx: float = 0, stroke: str = "none", sw: int = 0) -> str:
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'


def line(x1: float, y1: float, x2: float, y2: float, stroke: str, sw: int = 8, dash: str = "") -> str:
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{sw}" stroke-linecap="round"{extra}/>'


def path(d: str, fill: str = "none", stroke: str = "none", sw: int = 0) -> str:
    return f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" stroke-linecap="round" stroke-linejoin="round"/>'


def polygon(points: list[tuple[float, float]], fill: str, stroke: str = "none", sw: int = 0) -> str:
    pts = " ".join(f"{x},{y}" for x, y in points)
    return f'<polygon points="{pts}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" stroke-linejoin="round"/>'


def cloud(x: float, y: float, scale: float, white: str, outline: str) -> str:
    return "".join([
        circle(x - 55 * scale, y, 42 * scale, white),
        circle(x, y - 26 * scale, 60 * scale, white),
        circle(x + 60 * scale, y, 45 * scale, white),
        rect(x - 95 * scale, y, 190 * scale, 48 * scale, white, 24 * scale),
        path(f"M{x-82*scale},{y+36*scale} H{x+82*scale}", "none", outline, max(2, int(5 * scale))),
    ])


def rays(cx: int, cy: int, radius: int, color: str) -> str:
    bits = []
    for i in range(12):
        angle = math.tau * i / 12
        bits.append(line(
            cx + math.cos(angle) * (radius + 14),
            cy + math.sin(angle) * (radius + 14),
            cx + math.cos(angle) * (radius + 48),
            cy + math.sin(angle) * (radius + 48),
            color,
            10,
        ))
    return "".join(bits)


def gear(cx: int, cy: int, radius: int, fill: str, hole: str, outline: str) -> str:
    points = []
    for i in range(24):
        angle = math.tau * i / 24 - math.pi / 2
        rr = radius * (1.16 if i % 2 == 0 else 0.9)
        points.append((cx + math.cos(angle) * rr, cy + math.sin(angle) * rr))
    return polygon(points, fill, outline, 6) + circle(cx, cy, radius * 0.34, hole, outline, 5)


def drop(cx: int, cy: int, scale: float, fill: str, outline: str) -> str:
    d = f"M{cx},{cy-105*scale} C{cx-75*scale},{cy-15*scale} {cx-95*scale},{cy+40*scale} {cx},{cy+100*scale} C{cx+95*scale},{cy+40*scale} {cx+75*scale},{cy-15*scale} {cx},{cy-105*scale} Z"
    return path(d, fill, outline, 6)


def leaf(cx: int, cy: int, scale: float, fill: str, outline: str) -> str:
    d = f"M{cx-110*scale},{cy+55*scale} C{cx-100*scale},{cy-90*scale} {cx+80*scale},{cy-120*scale} {cx+115*scale},{cy-55*scale} C{cx+95*scale},{cy+65*scale} {cx-20*scale},{cy+110*scale} {cx-110*scale},{cy+55*scale} Z"
    return path(d, fill, outline, 6) + line(cx - 75 * scale, cy + 50 * scale, cx + 75 * scale, cy - 55 * scale, outline, 6)


def fish(cx: int, cy: int, scale: float, fill: str, accent: str, outline: str) -> str:
    body = ellipse(cx, cy, 125 * scale, 72 * scale, fill, outline, 7)
    tail = polygon([(cx - 110 * scale, cy), (cx - 205 * scale, cy - 80 * scale), (cx - 195 * scale, cy + 82 * scale)], accent, outline, 7)
    eye = circle(cx + 62 * scale, cy - 18 * scale, 10 * scale, outline)
    fin = polygon([(cx, cy + 15 * scale), (cx - 25 * scale, cy + 88 * scale), (cx + 45 * scale, cy + 47 * scale)], accent, outline, 5)
    return tail + body + fin + eye


def bird(cx: int, cy: int, scale: float, fill: str, accent: str, outline: str) -> str:
    body = ellipse(cx, cy + 20 * scale, 110 * scale, 82 * scale, fill, outline, 7)
    head = circle(cx + 82 * scale, cy - 48 * scale, 52 * scale, fill, outline, 7)
    beak = polygon([(cx + 128 * scale, cy - 52 * scale), (cx + 190 * scale, cy - 30 * scale), (cx + 128 * scale, cy - 16 * scale)], accent, outline, 5)
    wing = path(f"M{cx-65*scale},{cy+15*scale} Q{cx},{cy-70*scale} {cx+48*scale},{cy+35*scale} Q{cx-15*scale},{cy+75*scale} {cx-65*scale},{cy+15*scale} Z", accent, outline, 6)
    return body + head + beak + wing + circle(cx + 95 * scale, cy - 58 * scale, 8 * scale, outline)


def bug(cx: int, cy: int, scale: float, fill: str, accent: str, outline: str, legs: int = 6) -> str:
    bits = [ellipse(cx, cy + 20 * scale, 70 * scale, 105 * scale, fill, outline, 7), circle(cx, cy - 88 * scale, 42 * scale, accent, outline, 7)]
    per_side = legs // 2
    for i in range(per_side):
        yy = cy - 30 * scale + i * 47 * scale
        bits.append(line(cx - 55 * scale, yy, cx - 130 * scale, yy + (i - 1) * 25 * scale, outline, 8))
        bits.append(line(cx + 55 * scale, yy, cx + 130 * scale, yy + (i - 1) * 25 * scale, outline, 8))
    bits.append(path(f"M{cx-20*scale},{cy-120*scale} Q{cx-55*scale},{cy-175*scale} {cx-90*scale},{cy-150*scale}", "none", outline, 6))
    bits.append(path(f"M{cx+20*scale},{cy-120*scale} Q{cx+55*scale},{cy-175*scale} {cx+90*scale},{cy-150*scale}", "none", outline, 6))
    return "".join(bits)


def mammal(cx: int, cy: int, scale: float, fill: str, accent: str, outline: str, title: str) -> str:
    body = ellipse(cx - 25 * scale, cy + 25 * scale, 130 * scale, 78 * scale, fill, outline, 7)
    head = circle(cx + 105 * scale, cy - 25 * scale, 62 * scale, fill, outline, 7)
    bits = [body, head]
    if "象" in title:
        bits.append(path(f"M{cx+150*scale},{cy-10*scale} Q{cx+190*scale},{cy+70*scale} {cx+145*scale},{cy+105*scale}", "none", outline, 24))
        bits.append(ellipse(cx + 75 * scale, cy - 28 * scale, 45 * scale, 62 * scale, accent, outline, 5))
    elif "长颈鹿" in title:
        bits[1] = circle(cx + 105 * scale, cy - 120 * scale, 45 * scale, fill, outline, 7)
        bits.append(rect(cx + 70 * scale, cy - 105 * scale, 58 * scale, 120 * scale, fill, 25, outline, 7))
        for dx, dy in [(-40, 5), (20, 35), (50, -25)]:
            bits.append(circle(cx + dx * scale, cy + dy * scale, 18 * scale, accent))
    elif "斑马" in title:
        for dx in [-95, -45, 5, 55]:
            bits.append(line(cx + dx * scale, cy - 25 * scale, cx + (dx + 20) * scale, cy + 65 * scale, outline, 10))
    elif "骆驼" in title:
        bits.append(path(f"M{cx-120*scale},{cy-5*scale} Q{cx-70*scale},{cy-100*scale} {cx-20*scale},{cy-10*scale} Q{cx+25*scale},{cy-105*scale} {cx+70*scale},{cy-5*scale}", accent, outline, 6))
    elif "熊猫" in title:
        bits.append(circle(cx + 72 * scale, cy - 67 * scale, 24 * scale, outline))
        bits.append(circle(cx + 138 * scale, cy - 67 * scale, 24 * scale, outline))
        bits.append(ellipse(cx + 83 * scale, cy - 30 * scale, 17 * scale, 23 * scale, outline))
        bits.append(ellipse(cx + 127 * scale, cy - 30 * scale, 17 * scale, 23 * scale, outline))
    elif "兔" in title:
        bits.append(ellipse(cx + 80 * scale, cy - 105 * scale, 20 * scale, 70 * scale, accent, outline, 5))
        bits.append(ellipse(cx + 130 * scale, cy - 105 * scale, 20 * scale, 70 * scale, accent, outline, 5))
    elif "刺猬" in title:
        spikes = [(cx - 145 * scale, cy + 5 * scale), (cx - 105 * scale, cy - 80 * scale), (cx - 45 * scale, cy - 95 * scale), (cx + 10 * scale, cy - 75 * scale)]
        bits.extend(polygon([(x - 22 * scale, y + 35 * scale), (x, y - 30 * scale), (x + 25 * scale, y + 35 * scale)], accent, outline, 4) for x, y in spikes)
    elif "猫" in title or "狗" in title:
        bits.append(polygon([(cx + 58 * scale, cy - 62 * scale), (cx + 72 * scale, cy - 125 * scale), (cx + 103 * scale, cy - 75 * scale)], accent, outline, 5))
        bits.append(polygon([(cx + 112 * scale, cy - 75 * scale), (cx + 145 * scale, cy - 125 * scale), (cx + 153 * scale, cy - 55 * scale)], accent, outline, 5))
    bits.extend([
        line(cx - 90 * scale, cy + 70 * scale, cx - 95 * scale, cy + 145 * scale, outline, 13),
        line(cx + 35 * scale, cy + 70 * scale, cx + 40 * scale, cy + 145 * scale, outline, 13),
        circle(cx + 122 * scale, cy - 38 * scale, 7 * scale, outline),
    ])
    return "".join(bits)


def organ_art(title: str, p: tuple[str, ...]) -> str:
    _, primary, accent, light, dark = p
    if any(k in title for k in ("平衡", "闭眼站立")):
        return path("M500,120 C350,90 300,210 350,300 C385,355 420,330 420,410 C420,465 520,455 520,372 C520,290 455,298 455,240 C455,195 540,185 555,255", "none", dark, 24) + path("M500,195 C425,180 410,250 455,270", "none", accent, 16)
    if any(k in title for k in ("眼", "瞳孔", "眼泪")):
        return path("M275,270 Q480,90 685,270 Q480,450 275,270 Z", light, dark, 8) + circle(480, 270, 92, primary, dark, 7) + circle(480, 270, 38, dark) + circle(450, 235, 13, light)
    if any(k in title for k in ("耳", "平衡")):
        return path("M500,120 C350,90 300,210 350,300 C385,355 420,330 420,410 C420,465 520,455 520,372 C520,290 455,298 455,240 C455,195 540,185 555,255", "none", dark, 24) + path("M500,195 C425,180 410,250 455,270", "none", accent, 16)
    if any(k in title for k in ("骨头", "关节", "肌肉")):
        return circle(355, 160, 38, light, dark, 7) + circle(605, 380, 38, light, dark, 7) + line(380, 185, 580, 355, light, 62) + line(380, 185, 580, 355, dark, 8)
    if any(k in title for k in ("心脏", "心跳", "血液")):
        return path("M480,420 C360,330 300,270 320,190 C340,105 445,110 480,185 C515,110 620,105 640,190 C660,270 600,330 480,420 Z", accent, dark, 8) + line(480, 182, 480, 360, light, 8)
    if any(k in title for k in ("肺", "呼吸", "吸气", "打嗝")):
        return line(480, 120, 480, 220, dark, 22) + path("M462,205 C380,165 300,220 305,340 C310,430 410,430 455,365 Z", primary, dark, 8) + path("M498,205 C580,165 660,220 655,340 C650,430 550,430 505,365 Z", accent, dark, 8)
    if "大脑" in title or "反射" in title:
        bits = [circle(420, 250, 105, primary, dark, 8), circle(535, 235, 115, accent, dark, 8), circle(495, 330, 105, primary, dark, 8)]
        bits += [path("M375,220 Q430,175 480,220 T585,220", "none", light, 10), path("M390,310 Q455,265 520,315", "none", light, 10)]
        return "".join(bits)
    if any(k in title for k in ("牙", "舌", "味", "食物", "肚子", "尿")):
        return path("M370,150 Q480,95 590,150 L565,340 Q540,450 480,390 Q420,450 395,340 Z", light, dark, 8) + path("M410,280 Q480,340 550,280 Q545,385 480,395 Q415,385 410,280 Z", accent, dark, 6)
    if "指纹" in title:
        return "".join(path(f"M{330+i*22},390 C{285+i*16},215 {380+i*8},105 {480+i*3},125 C{600-i*4},145 {650-i*16},265 {590-i*14},390", "none", primary if i % 2 else dark, 8) for i in range(7))
    return path("M480,110 C390,110 350,175 350,255 C350,350 405,430 480,445 C555,430 610,350 610,255 C610,175 570,110 480,110 Z", primary, dark, 8) + circle(430, 220, 18, light) + circle(530, 220, 18, light) + path("M415,325 Q480,380 545,325", "none", light, 10)


def machine_art(title: str, p: tuple[str, ...]) -> str:
    _, primary, accent, light, dark = p
    if "桥" in title:
        return line(220, 390, 740, 390, dark, 18) + path("M250,390 Q480,120 710,390", "none", primary, 26) + "".join(line(x, 390, x, 390 - 175 * math.sin((x - 250) / 460 * math.pi), accent, 8) for x in range(300, 701, 50))
    if "杠杆" in title or "跷跷板" in title:
        return polygon([(450, 360), (510, 360), (480, 260)], accent, dark, 6) + line(235, 245, 725, 365, primary, 28) + circle(280, 220, 40, light, dark, 6) + rect(650, 330, 90, 70, accent, 12, dark, 6)
    if "滑轮" in title or "起重机" in title:
        return circle(480, 165, 72, light, dark, 15) + line(480, 235, 480, 405, dark, 12) + rect(420, 395, 120, 75, accent, 10, dark, 7) + path("M330,470 V145 H480", "none", primary, 16)
    if "齿轮" in title or "钟" in title:
        return gear(405, 275, 105, primary, light, dark) + gear(590, 330, 78, accent, light, dark)
    if "轮" in title or "自行车" in title or "链条" in title:
        return circle(345, 355, 112, "none", dark, 15) + circle(615, 355, 112, "none", dark, 15) + path("M345,355 L455,180 L535,355 Z M455,180 L615,355 M420,355 H555", "none", primary, 16)
    if "磁" in title or "指南针" in title:
        return path("M315,145 V300 C315,430 645,430 645,300 V145 H555 V300 C555,330 405,330 405,300 V145 Z", primary, dark, 8) + rect(315, 145, 90, 90, accent, 0, dark, 5) + rect(555, 145, 90, 90, light, 0, dark, 5)
    if any(k in title for k in ("电池", "电路", "电动机", "发电机", "LED")):
        return rect(300, 190, 150, 160, primary, 24, dark, 8) + rect(335, 155, 80, 38, accent, 8, dark, 6) + path("M450,270 C540,270 535,145 640,160 C735,175 700,365 590,360 C520,355 535,320 450,320", "none", dark, 12) + circle(590, 360, 42, accent, dark, 7)
    if "机器人" in title:
        return rect(350, 155, 260, 230, light, 35, dark, 9) + circle(425, 245, 24, primary, dark, 5) + circle(535, 245, 24, accent, dark, 5) + line(400, 330, 560, 330, dark, 10) + line(480, 155, 480, 95, dark, 9) + circle(480, 80, 18, accent, dark, 4)
    return gear(410, 285, 100, primary, light, dark) + gear(585, 305, 75, accent, light, dark) + line(250, 420, 710, 420, dark, 14)


def earth_art(title: str, p: tuple[str, ...]) -> str:
    _, primary, accent, light, dark = p
    if any(k in title for k in ("山", "山谷", "火山", "冰川", "洞穴", "沙漠", "草原", "雨林")):
        bits = [polygon([(170, 410), (400, 125), (600, 410)], primary, dark, 8), polygon([(385, 145), (400, 125), (485, 230), (430, 210)], light)]
        if "火山" in title:
            bits.append(path("M410,145 Q480,80 540,145", "none", accent, 28))
            bits.append(path("M430,145 L520,390", "none", accent, 20))
        elif "洞穴" in title:
            bits.append(path("M360,410 Q400,285 470,410 Z", dark, dark, 5))
        elif "冰川" in title:
            bits.append(path("M260,410 Q380,285 545,410 Z", light, primary, 6))
        else:
            bits.append(polygon([(420, 410), (650, 195), (820, 410)], accent, dark, 8))
        return "".join(bits)
    globe = circle(480, 270, 165, primary, dark, 9)
    land = path("M365,150 Q420,115 470,165 L455,235 L385,250 L340,210 Z M515,255 Q620,205 640,285 L590,360 L510,345 L475,300 Z", accent, dark, 5)
    grid = path("M315,270 H645 M480,105 V435 M345,175 Q480,235 615,175 M345,365 Q480,305 615,365", "none", light, 5)
    return globe + land + grid


def space_art(title: str, p: tuple[str, ...]) -> str:
    _, primary, accent, light, dark = p
    if "火箭" in title or "宇航服" in title:
        return path("M480,95 C390,175 390,330 480,400 C570,330 570,175 480,95 Z", light, dark, 8) + circle(480, 235, 48, primary, dark, 7) + polygon([(410, 330), (335, 420), (430, 390)], accent, dark, 6) + polygon([(550, 330), (625, 420), (530, 390)], accent, dark, 6) + polygon([(440, 400), (480, 485), (520, 400)], accent)
    if "黑洞" in title:
        return ellipse(480, 275, 205, 88, accent, dark, 12) + ellipse(480, 275, 130, 55, primary, dark, 5) + circle(480, 275, 70, dark)
    if "彗星" in title or "流星" in title:
        return path("M180,370 Q360,170 650,175", "none", accent, 70) + path("M190,390 Q360,235 650,175", "none", light, 26) + circle(675, 170, 58, primary, dark, 7)
    if "月" in title or "日食" in title:
        return circle(510, 270, 145, light, dark, 8) + circle(450, 230, 28, primary) + circle(565, 330, 38, primary) + circle(575, 195, 17, primary)
    if "土星" in title:
        return ellipse(480, 280, 250, 65, "none", light, 24) + circle(480, 280, 125, accent, dark, 8) + ellipse(480, 280, 250, 65, "none", dark, 7)
    if "太阳" in title or "恒星" in title or "星星" in title:
        return rays(480, 270, 112, accent) + circle(480, 270, 112, accent, dark, 8)
    if "望远镜" in title:
        return path("M300,210 L610,125 L660,220 L350,300 Z", primary, dark, 8) + line(460, 270, 370, 445, dark, 16) + line(500, 260, 600, 445, dark, 16)
    return circle(480, 270, 135, primary, dark, 8) + ellipse(480, 270, 245, 80, "none", accent, 18) + circle(690, 218, 28, light, dark, 5)


def home_art(title: str, p: tuple[str, ...]) -> str:
    _, primary, accent, light, dark = p
    if any(k in title for k in ("冰箱", "微波炉", "洗衣机", "吸尘器")):
        return rect(330, 105, 300, 340, light, 28, dark, 9) + rect(380, 165, 200, 185, primary, 16, dark, 7) + circle(480, 258, 70, accent, dark, 7) + circle(575, 135, 11, accent)
    if any(k in title for k in ("面包", "面团", "鸡蛋", "苹果", "玉米")):
        return ellipse(480, 320, 210, 110, accent, dark, 8) + path("M315,310 Q330,130 480,135 Q630,130 645,310", light, dark, 8) + circle(400, 240, 24, primary) + circle(485, 210, 30, primary) + circle(565, 250, 20, primary)
    if any(k in title for k in ("水", "盐", "糖", "油", "牛奶")):
        return path("M350,120 H610 L580,430 H380 Z", light, dark, 9) + path("M372,285 H588 L580,430 H380 Z", primary, dark, 5) + circle(430, 335, 18, accent) + circle(520, 370, 13, accent)
    if any(k in title for k in ("纸", "布", "陶瓷", "塑料", "橡胶", "玻璃", "铁")):
        return rect(310, 145, 340, 250, light, 28, dark, 9) + path("M330,340 L430,240 L510,315 L590,190 L635,330", "none", primary, 18) + circle(415, 210, 32, accent)
    return circle(405, 270, 88, primary, dark, 7) + circle(555, 270, 88, accent, dark, 7) + line(470, 270, 490, 270, dark, 10)


def transport_art(title: str, p: tuple[str, ...]) -> str:
    _, primary, accent, light, dark = p
    if "走路" in title:
        return circle(480, 135, 55, accent, dark, 7) + line(480, 195, 480, 330, dark, 22) + line(480, 235, 365, 300, primary, 20) + line(480, 235, 595, 300, primary, 20) + line(480, 330, 390, 450, dark, 22) + line(480, 330, 590, 430, dark, 22)
    if "自行车" in title:
        return machine_art(title, p)
    if "红绿灯" in title:
        return rect(410, 80, 140, 330, dark, 32, dark, 8) + circle(480, 165, 45, "#df5b52", light, 5) + circle(480, 245, 45, "#edbd43", light, 5) + circle(480, 325, 45, "#4ca66a", light, 5) + line(480, 410, 480, 490, dark, 24)
    if "安全带" in title:
        return rect(310, 115, 340, 320, primary, 50, dark, 9) + line(365, 135, 590, 410, light, 36) + rect(525, 330, 95, 78, accent, 16, dark, 7) + line(375, 355, 530, 355, light, 30)
    if "火车" in title or "地铁" in title:
        return rect(260, 145, 440, 245, primary, 42, dark, 9) + rect(310, 190, 105, 90, light, 12, dark, 6) + rect(455, 190, 105, 90, light, 12, dark, 6) + circle(345, 398, 45, dark) + circle(615, 398, 45, dark) + line(190, 455, 770, 455, dark, 13) + line(220, 490, 740, 490, dark, 13)
    if "飞机" in title:
        return path("M195,285 L420,245 L510,115 L560,120 L525,240 L745,275 L760,320 L525,300 L570,410 L520,420 L430,305 L200,330 Z", light, dark, 8)
    if "直升机" in title:
        return ellipse(485, 290, 145, 92, light, dark, 8) + rect(330, 265, 145, 45, accent, 12, dark, 7) + line(480, 195, 480, 130, dark, 12) + line(250, 130, 710, 130, dark, 13) + line(615, 280, 740, 230, dark, 22)
    if "潜水艇" in title:
        return ellipse(480, 315, 245, 105, primary, dark, 9) + rect(440, 185, 105, 75, primary, 20, dark, 7) + line(490, 185, 490, 125, dark, 14) + line(490, 125, 560, 125, dark, 14) + polygon([(250, 315), (170, 245), (175, 385)], accent, dark, 7) + circle(420, 300, 24, light, dark, 5) + circle(500, 300, 24, light, dark, 5) + circle(580, 300, 24, light, dark, 5)
    if "船" in title:
        return path("M230,310 H720 L660,405 H300 Z", primary, dark, 9) + polygon([(455,305), (455,120), (610,290)], light, dark, 7) + path("M160,430 Q280,385 400,430 T640,430 T880,430", "none", accent, 18)
    if "热气球" in title:
        return ellipse(480, 210, 135, 160, accent, dark, 8) + line(425, 340, 450, 405, dark, 8) + line(535, 340, 510, 405, dark, 8) + rect(438, 400, 84, 62, primary, 10, dark, 7)
    if any(k in title for k in ("太阳能", "风力", "流水", "能源", "电网")):
        return circle(270, 175, 70, accent, dark, 6) + rays(270, 175, 70, accent) + line(610, 150, 610, 430, dark, 16) + "".join(polygon([(610, 200), (610 + math.cos(a) * 170, 200 + math.sin(a) * 170), (610 + math.cos(a + 0.34) * 55, 200 + math.sin(a + 0.34) * 55)], light, dark, 5) for a in [0, math.tau / 3, 2 * math.tau / 3])
    if "LED" in title:
        return circle(480, 220, 120, accent, dark, 8) + path("M405,300 Q430,365 430,410 H530 Q530,365 555,300", light, dark, 8) + line(430, 420, 530, 420, dark, 18) + rays(480, 220, 120, accent)
    if "房子" in title:
        return polygon([(245, 275), (480, 90), (715, 275)], accent, dark, 9) + rect(300, 270, 360, 215, light, 12, dark, 9) + rect(425, 350, 110, 135, primary, 10, dark, 7) + line(330, 305, 630, 305, primary, 22)
    if "回收" in title:
        return "".join([
            path("M480,115 L570,255 H520", "none", primary, 32), polygon([(515,220), (630,255), (520,290)], primary),
            path("M620,300 L535,440 L510,395", "none", accent, 32), polygon([(545,400), (470,485), (485,365)], accent),
            path("M420,440 L325,300 H380", "none", dark, 32), polygon([(370,340), (280,270), (400,260)], dark),
        ])
    if "堆肥" in title:
        return rect(340, 205, 280, 250, primary, 24, dark, 9) + rect(315, 165, 330, 55, accent, 18, dark, 7) + leaf(425, 310, 0.55, light, dark) + leaf(540, 350, 0.45, accent, dark)
    if any(k in title for k in ("树", "生物", "大气", "气候")):
        if "大气" in title or "气候" in title:
            return earth_art(title, p) + path("M265,155 Q480,35 695,155", "none", accent, 20) + path("M300,390 Q480,505 660,390", "none", light, 16)
        return rect(445, 250, 70, 225, accent, 22, dark, 7) + circle(365, 230, 100, primary, dark, 6) + circle(485, 160, 125, primary, dark, 6) + circle(600, 245, 95, primary, dark, 6) + circle(305, 410, 24, light, dark, 4) + bird(680, 360, 0.38, accent, light, dark)
    if any(k in title for k in ("科学", "观察", "尺", "实验", "模型", "为什么")):
        return circle(415, 235, 105, "none", primary, 22) + line(490, 315, 590, 420, primary, 30) + rect(555, 130, 155, 220, light, 18, dark, 8) + line(585, 190, 675, 190, accent, 10) + line(585, 240, 655, 240, accent, 10)
    return rect(270, 240, 420, 130, primary, 34, dark, 9) + path("M350,240 L420,155 H580 L640,240", light, dark, 8) + circle(380, 380, 62, dark) + circle(590, 380, 62, dark) + circle(380, 380, 28, accent) + circle(590, 380, 28, accent)


def scene_for(day: int, title: str, p: tuple[str, ...]) -> str:
    _, primary, accent, light, dark = p
    if day <= 31:
        if any(k in title for k in ("月", "星", "夜")):
            return space_art(title, p)
        if any(k in title for k in ("冰", "雪", "霜", "冷", "冬")):
            return cloud(480, 180, 1.1, light, dark) + "".join(circle(355 + i * 62, 330 + (i % 2) * 45, 18, primary, dark, 3) for i in range(5))
        if any(k in title for k in ("钟", "日历", "星期")):
            return machine_art(title, p)
        if "彩虹" in title or "晚霞" in title:
            return "".join(path(f"M{270-i*15},380 Q480,{90-i*12} {690+i*15},380", "none", c, 26) for i, c in enumerate(["#e75f55", "#ed9c42", "#eed85a", "#55a96f", "#4a83c5", "#7664ae"]))
        return rays(480, 250, 105, accent) + circle(480, 250, 105, accent, dark, 7) + ellipse(480, 440, 275, 45, primary)
    if day <= 59:
        if any(k in title for k in ("水", "雨", "河", "露", "冰", "雪", "杯", "衣")):
            return drop(480, 275, 1.2, primary, dark) + circle(440, 260, 18, light)
        if any(k in title for k in ("雷", "闪电")):
            return cloud(480, 170, 1.1, light, dark) + polygon([(495, 230), (430, 350), (495, 340), (445, 465), (570, 300), (505, 310)], accent, dark, 6)
        return cloud(480, 210, 1.1, light, dark) + path("M245,345 Q340,300 430,345 T625,345 T780,345", "none", primary, 18)
    if day <= 90:
        if "蘑菇" in title:
            return path("M340,280 Q480,95 620,280 Z", accent, dark, 8) + rect(440, 275, 80, 160, light, 35, dark, 7)
        if any(k in title for k in ("花", "蜜蜂", "花粉", "果实")):
            petals = "".join(circle(480 + math.cos(a) * 78, 235 + math.sin(a) * 78, 55, accent, dark, 5) for a in [i * math.tau / 6 for i in range(6)])
            return line(480, 285, 480, 455, dark, 18) + petals + circle(480, 235, 48, primary, dark, 6)
        if any(k in title for k in ("树", "竹", "藤")):
            return rect(438, 210, 84, 250, accent, 25, dark, 7) + circle(380, 205, 95, primary, dark, 6) + circle(500, 155, 110, primary, dark, 6) + circle(595, 225, 92, primary, dark, 6)
        return line(480, 300, 480, 450, dark, 16) + leaf(400, 260, 0.75, primary, dark) + leaf(555, 325, 0.65, accent, dark) + ellipse(480, 440, 210, 38, accent)
    if day <= 120:
        if "蜘蛛" in title:
            return bug(480, 285, 1.0, dark, accent, dark, 8)
        if any(k in title for k in ("蜗牛", "蚯蚓")):
            return circle(440, 260, 95, accent, dark, 8) + path("M440,260 Q440,205 495,215 Q555,235 535,300 Q510,350 440,330", "none", light, 12) + path("M330,350 Q480,290 650,355", "none", primary, 55)
        if any(k in title for k in ("青蛙", "壁虎", "蜥蜴", "变色龙")):
            return ellipse(480, 290, 135, 85, primary, dark, 8) + circle(420, 220, 48, primary, dark, 6) + circle(540, 220, 48, primary, dark, 6) + circle(420, 215, 12, dark) + circle(540, 215, 12, dark) + line(390, 340, 300, 430, dark, 18) + line(570, 340, 660, 430, dark, 18)
        return bug(480, 280, 1.0, primary, accent, dark, 6)
    if day <= 151:
        if any(k in title for k in ("鸟", "鸭", "企鹅", "猫头鹰", "啄木鸟", "火烈鸟")):
            return bird(480, 275, 1.15, primary, accent, dark)
        if any(k in title for k in ("鲸", "海豚", "鱼", "鲨", "章鱼")):
            return fish(500, 280, 1.15, primary, accent, dark)
        if "蝙蝠" in title:
            return path("M480,260 C390,135 285,175 250,310 C330,250 390,330 480,390 C570,330 630,250 710,310 C675,175 570,135 480,260 Z", primary, dark, 8) + circle(480, 270, 48, accent, dark, 6)
        return mammal(465, 270, 1.0, primary, accent, dark, title)
    if day <= 181:
        return organ_art(title, p)
    if day <= 212:
        return machine_art(title, p)
    if day <= 243:
        return earth_art(title, p)
    if day <= 273:
        if "珊瑚" in title:
            return path("M470,455 V285 M470,340 L370,245 M470,365 L575,255 M390,265 L350,190 M560,275 L620,190 M470,310 L500,205", "none", accent, 34) + circle(350, 190, 24, light, dark, 4) + circle(620, 190, 24, light, dark, 4) + circle(500, 205, 24, light, dark, 4) + path("M110,455 Q240,405 370,455 T630,455 T890,455", "none", primary, 18)
        if any(k in title for k in ("浮游", "小藻")):
            bits = []
            for i in range(14):
                a = i * math.tau / 14
                x = 480 + math.cos(a) * (105 + (i % 3) * 22)
                y = 280 + math.sin(a) * (85 + (i % 2) * 25)
                bits.append(circle(x, y, 14 + (i % 4) * 5, primary if i % 2 else accent, dark, 3))
                bits.append(line(x, y, x + math.cos(a) * 35, y + math.sin(a) * 35, dark, 4))
            return circle(480, 280, 70, light, dark, 6) + "".join(bits)
        if "章鱼" in title or "鱿鱼" in title:
            head = path("M365,285 Q380,105 480,105 Q580,105 595,285 Q555,345 480,345 Q405,345 365,285 Z", primary, dark, 8)
            arms = "".join(path(f"M{390+i*30},325 Q{350+i*40},390 {365+i*35},465", "none", accent if i % 2 else dark, 14) for i in range(7))
            return head + arms + circle(440, 235, 16, light, dark, 4) + circle(520, 235, 16, light, dark, 4)
        if "贝壳" in title or "沙" in title:
            return path("M300,400 C305,210 385,120 480,120 C575,120 655,210 660,400 Z", accent, dark, 8) + "".join(line(480, 130, 335 + i * 48, 395, light, 9) for i in range(7))
        if "塑料" in title:
            return path("M410,120 H550 L535,180 L585,245 V445 H375 V245 L425,180 Z", light, dark, 8) + rect(430, 85, 100, 45, accent, 8, dark, 5) + path("M105,455 Q240,400 375,455 T645,455 T915,455", "none", primary, 22)
        if "发光" in title:
            return fish(500, 300, 1.0, dark, primary, dark) + "".join(circle(300 + i * 70, 140 + (i % 3) * 45, 12 + (i % 2) * 8, light) for i in range(7))
        if "水母" in title:
            return path("M340,285 Q365,105 480,105 Q595,105 620,285 Z", primary, dark, 8) + "".join(path(f"M{x},285 Q{x-35},365 {x+10},445", "none", accent, 12) for x in [370, 425, 480, 535, 590])
        if "海星" in title:
            pts = []
            for i in range(10):
                a = -math.pi / 2 + i * math.pi / 5
                r = 155 if i % 2 == 0 else 70
                pts.append((480 + math.cos(a) * r, 285 + math.sin(a) * r))
            return polygon(pts, accent, dark, 8)
        if any(k in title for k in ("鱼", "鲸", "海马", "海龟", "小丑")):
            return fish(500, 285, 1.15, primary, accent, dark)
        if "螃蟹" in title or "寄居蟹" in title:
            return ellipse(480, 300, 125, 85, accent, dark, 8) + "".join(line(390 if side < 0 else 570, 290 + i * 30, 270 if side < 0 else 690, 245 + i * 55, dark, 12) for side in [-1, 1] for i in range(3)) + circle(430, 250, 16, dark) + circle(530, 250, 16, dark)
        return path("M120,420 Q240,355 360,420 T600,420 T840,420", "none", primary, 24) + leaf(480, 270, 1.1, accent, dark)
    if day <= 304:
        return space_art(title, p)
    if day <= 334:
        return home_art(title, p)
    return transport_art(title, p)


def make_svg(day: int, title: str) -> str:
    p = PALETTES[month_for(day)]
    background, primary, accent, light, dark = p
    rng = random.Random(day * 1009)
    decorations = []
    for _ in range(10):
        x = rng.randint(45, 915)
        y = rng.randint(40, 500)
        r = rng.randint(5, 15)
        decorations.append(circle(x, y, r, light if rng.random() < 0.55 else accent, "none", 0))
    scene = scene_for(day, title, p)
    safe_title = html.escape(title)
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="960" height="540" viewBox="0 0 960 540" role="img" aria-labelledby="title">
  <title id="title">{safe_title}</title>
  <rect width="960" height="540" fill="{background}"/>
  <path d="M0,430 Q180,370 340,430 T680,430 T1020,430 V540 H0 Z" fill="{primary}" opacity="0.13"/>
  <g opacity="0.38">{''.join(decorations)}</g>
  <g>{scene}</g>
  <rect x="12" y="12" width="936" height="516" rx="30" fill="none" stroke="{dark}" stroke-width="5" opacity="0.18"/>
</svg>
'''


def main() -> None:
    all_entries = entries()
    if [day for day, _ in all_entries] != list(range(1, 366)):
        raise SystemExit("day headings are not the exact sequence 001..365")
    OUT.mkdir(parents=True, exist_ok=True)
    for day, title in all_entries:
        (OUT / f"day-{day:03d}.svg").write_text(make_svg(day, title), encoding="utf-8")
    print(f"generated {len(all_entries)} SVG files in {OUT}")


if __name__ == "__main__":
    main()
