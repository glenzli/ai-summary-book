#!/usr/bin/env python3
"""Textbook-style annotated plates for insects and soil invertebrates."""

from __future__ import annotations

import math

from textbook_illustrations import INK, circle, ellipse, line, panel_title, path, rect, text


INVERTEBRATE_QUESTIONS = frozenset(range(91, 116))

INVERTEBRATE_KEYS: dict[int, tuple[str, ...]] = {
    91: ("insect-body", "head", "thorax", "abdomen", "six-legs"),
    92: ("insect-spider-comparison", "six-legs", "eight-legs", "body-regions"),
    93: ("ant-trail", "nest", "pheromone", "food"),
    94: ("ant-nest-cutaway", "queen", "workers", "larvae", "foragers"),
    95: ("bee-side-view", "wings", "flight-muscles", "air-wave"),
    96: ("butterfly-feeding", "proboscis", "flower", "nectar"),
    97: ("butterfly-life-cycle", "egg", "caterpillar", "chrysalis", "adult"),
    98: ("beetle-wing-cutaway", "elytron", "hindwing", "thorax"),
    99: ("ladybug-patterns", "elytra", "species-pattern", "adult-spots"),
    100: ("firefly-side-view", "light-organ", "abdomen", "chemical-light"),
    101: ("mosquito-mouthparts", "female-proboscis", "skin", "male-antenna"),
    102: ("dragonfly-top-view", "forewings", "hindwings", "independent-motion"),
    103: ("grasshopper-leg", "hind-leg", "muscle", "elastic-cuticle"),
    104: ("cricket-wing", "file", "scraper", "sound-wave"),
    105: ("mantis-foreleg", "raptorial-leg", "spines", "joint"),
    106: ("stick-insect-on-twig", "body-outline", "jointed-legs", "camouflage"),
    107: ("snail-anatomy", "shell", "mantle", "soft-body", "foot"),
    108: ("snail-movement", "muscular-foot", "mucus", "motion-wave"),
    109: ("earthworm-anatomy", "segments", "circular-muscle", "longitudinal-muscle", "setae"),
    110: ("earthworm-soil-cutaway", "burrow", "roots", "leaf-litter", "castings"),
    111: ("centipede-millipede-comparison", "one-leg-pair", "two-leg-pairs", "body-shape"),
    112: ("pillbug-rolling", "overlapping-plates", "soft-underside", "rolled-ball"),
    113: ("orb-web", "frame-thread", "radial-thread", "capture-spiral", "anchor"),
    114: ("spider-on-web", "dry-route", "sticky-spiral", "claws"),
    115: ("under-rock-habitat", "moist-soil", "pillbug", "centipede", "earthworm"),
}


ORANGE = "#D9693F"
ORANGE_DARK = "#9B432F"
TEAL = "#287F78"
TEAL_LIGHT = "#78B9AC"
YELLOW = "#E7B33F"
CREAM = "#F7E8C8"
SOIL = "#87654A"
SOIL_LIGHT = "#C99D71"
GREEN = "#568A58"
RED = "#C94F4F"
BLUE = "#4C86B8"
PURPLE = "#8267A8"


def polygon(points: list[tuple[float, float]], fill: str, stroke: str = INK, width: float = 3) -> str:
    coordinates = " ".join(f"{x},{y}" for x, y in points)
    return f'<polygon points="{coordinates}" fill="{fill}" stroke="{stroke}" stroke-width="{width}" stroke-linejoin="round"/>'


def dashed_curve(data: str, color: str, width: float = 4) -> str:
    return path(data, "none", color, width, False, "8 8")


def label(tx: float, ty: float, x: float, y: float, value: str, color: str = INK) -> str:
    """Use compact specimen labels so Chinese annotations stay inside the plate."""
    end_x = x - 12 if x >= tx else x + 12
    max_chars = max(6, int((930 - x) / 19))
    line_count = max(1, math.ceil(len(value) / max_chars))
    line_length = math.ceil(len(value) / line_count)
    lines = [value[index:index + line_length] for index in range(0, len(value), line_length)]
    first_y = y if len(lines) == 1 else y - 8
    copy = "".join(text(x, first_y + index * 21, item, 19, color, "start", 650) for index, item in enumerate(lines))
    return line(tx, ty, end_x, y - 6, color, 2.5) + circle(tx, ty, 4.5, color) + copy


def insect_side(cx: float, cy: float, scale: float = 1.0, wings: bool = False) -> str:
    s = scale
    bits = [
        ellipse(cx - 104 * s, cy, 42 * s, 48 * s, ORANGE, INK, 4 * s),
        ellipse(cx - 36 * s, cy + 2 * s, 51 * s, 58 * s, TEAL, INK, 4 * s),
        path(
            f"M{cx+4*s},{cy-45*s} C{cx+78*s},{cy-65*s} {cx+145*s},{cy-36*s} {cx+154*s},{cy} "
            f"C{cx+145*s},{cy+42*s} {cx+78*s},{cy+62*s} {cx+4*s},{cy+45*s} Z",
            ORANGE,
            INK,
            4 * s,
        ),
        circle(cx - 115 * s, cy - 8 * s, 9 * s, "#24384A"),
        circle(cx - 119 * s, cy - 12 * s, 3 * s, "#FFFFFF"),
        path(f"M{cx-126*s},{cy-35*s} Q{cx-160*s},{cy-80*s} {cx-178*s},{cy-62*s}", "none", INK, 3 * s),
        path(f"M{cx-108*s},{cy-40*s} Q{cx-125*s},{cy-90*s} {cx-100*s},{cy-82*s}", "none", INK, 3 * s),
    ]
    for index, ox in enumerate((-70, -36, -4)):
        near_y = cy + (17 + index * 5) * s
        bits.append(path(f"M{cx+ox*s},{near_y} L{cx+(ox-34)*s},{cy+(82+index*10)*s} L{cx+(ox-6)*s},{cy+(105+index*3)*s}", "none", INK, 6 * s))
        bits.append(path(f"M{cx+ox*s},{cy-14*s} L{cx+(ox+25)*s},{cy-72*s} L{cx+(ox+58)*s},{cy-82*s}", "none", "#50687A", 4 * s))
    if wings:
        bits.extend([
            path(f"M{cx-45*s},{cy-42*s} Q{cx+32*s},{cy-128*s} {cx+92*s},{cy-55*s} Q{cx+35*s},{cy-10*s} {cx-35*s},{cy-16*s} Z", "#DCEFF1", BLUE, 3 * s),
            path(f"M{cx-28*s},{cy-33*s} Q{cx+80*s},{cy-78*s} {cx+125*s},{cy-8*s} Q{cx+46*s},{cy+15*s} {cx-20*s},{cy-5*s} Z", "#EDF7F6", TEAL, 3 * s),
        ])
    return "".join(bits)


def ant(cx: float, cy: float, scale: float = 1.0, angle: float = 0) -> str:
    s = scale
    body = [
        circle(-48 * s, 0, 18 * s, ORANGE, INK, 2.5 * s),
        ellipse(-8 * s, 0, 20 * s, 16 * s, TEAL, INK, 2.5 * s),
        ellipse(39 * s, 0, 30 * s, 23 * s, ORANGE_DARK, INK, 2.5 * s),
        path(f"M{-61*s},{-10*s} Q{-79*s},{-34*s} {-90*s},{-18*s}", "none", INK, 2 * s),
        path(f"M{-50*s},{-17*s} Q{-57*s},{-40*s} {-40*s},{-37*s}", "none", INK, 2 * s),
    ]
    for ox in (-23, -4, 14):
        body.append(path(f"M{ox*s},{6*s} L{(ox-19)*s},{31*s} L{(ox-4)*s},{43*s}", "none", INK, 3 * s))
        body.append(path(f"M{ox*s},{-6*s} L{(ox+18)*s},{-30*s} L{(ox+34)*s},{-38*s}", "none", INK, 2.5 * s))
    return f'<g transform="translate({cx} {cy}) rotate({angle})">{"".join(body)}</g>'


def spider(cx: float, cy: float, scale: float = 1.0) -> str:
    s = scale
    bits = [circle(cx - 26 * s, cy, 32 * s, ORANGE, INK, 4 * s), ellipse(cx + 35 * s, cy, 52 * s, 45 * s, TEAL, INK, 4 * s)]
    for index, offset in enumerate((-33, -12, 12, 33)):
        bits.append(path(f"M{cx+5*s},{cy+offset*s} L{cx-54*s},{cy+(offset-30+index*8)*s} L{cx-92*s},{cy+(offset-45+index*14)*s}", "none", INK, 5 * s))
        bits.append(path(f"M{cx+48*s},{cy+offset*s} L{cx+96*s},{cy+(offset-31+index*8)*s} L{cx+130*s},{cy+(offset-48+index*14)*s}", "none", INK, 5 * s))
    for dx in (-36, -22, -8, 6):
        bits.append(circle(cx + dx * s, cy - 8 * s, 3.3 * s, "#FFFFFF"))
    return "".join(bits)


def ladybug(cx: float, cy: float, scale: float, spots: tuple[tuple[int, int], ...]) -> str:
    s = scale
    bits = [ellipse(cx, cy + 8 * s, 62 * s, 72 * s, RED, INK, 4 * s), circle(cx, cy - 57 * s, 31 * s, INK), line(cx, cy - 36 * s, cx, cy + 78 * s, INK, 3 * s)]
    for dx, dy in spots:
        bits.append(circle(cx + dx * s, cy + dy * s, 8 * s, INK))
    for side in (-1, 1):
        for dy in (-23, 9, 38):
            bits.append(line(cx + side * 44 * s, cy + dy * s, cx + side * 83 * s, cy + (dy + side * 7) * s, INK, 3 * s))
    return "".join(bits)


def worm(cx: float, cy: float, scale: float = 1.0, color: str = "#C87569") -> str:
    s = scale
    data = f"M{cx-150*s},{cy} C{cx-110*s},{cy-70*s} {cx-45*s},{cy+70*s} {cx},{cy} C{cx+45*s},{cy-68*s} {cx+110*s},{cy+66*s} {cx+150*s},{cy}"
    bits = [path(data, "none", color, 30 * s)]
    for index in range(-6, 7):
        x = cx + index * 22 * s
        y = cy + math.sin(index * 0.9) * 28 * s
        bits.append(line(x, y - 12 * s, x, y + 12 * s, ORANGE_DARK, 1.8 * s))
    return "".join(bits)


def question_91() -> str:
    bits = [panel_title("昆虫的三段身体和六条腿都能在实物上找到"), insect_side(390, 275, 1.15, True)]
    bits.extend([
        label(270, 260, 715, 125, "头：触角、眼和口器"),
        label(352, 275, 715, 225, "胸：翅和三对足都接在这里"),
        label(510, 275, 715, 325, "腹：消化、生殖和呼吸开口"),
        label(350, 378, 715, 425, "三对足，共六条"),
    ])
    return "".join(bits)


def question_92() -> str:
    bits = [panel_title("把真实身体并排比较，腿数和体区最可靠"), insect_side(240, 245, 0.66, False), spider(650, 255, 0.74)]
    bits.extend([
        line(470, 95, 470, 455, "#8FA0AB", 2, False, "8 10"),
        text(240, 442, "昆虫：头、胸、腹｜六条腿", 23, INK, "middle", 720),
        text(665, 442, "蜘蛛：两大体区｜八条腿", 23, INK, "middle", 720),
        label(203, 250, 75, 135, "三段身体"),
        label(690, 260, 735, 130, "两大体区"),
    ])
    return "".join(bits)


def question_93() -> str:
    bits = [panel_title("蚂蚁沿着同伴留下的信息素气味行走")]
    bits.extend([
        path("M70,395 Q140,270 240,315 Q305,355 360,370", SOIL_LIGHT, INK, 3),
        circle(135, 380, 34, "#6B4C38", INK, 3),
        path("M165,374 C300,320 470,365 650,245 S820,175 870,150", "none", "#5FA78B", 16).replace('/>', ' opacity="0.20"/>'),
        dashed_curve("M165,374 C300,320 470,365 650,245 S820,175 870,150", TEAL, 4),
        polygon([(852, 124), (885, 142), (870, 180), (835, 166)], GREEN),
        circle(870, 146, 9, RED),
        ant(300, 340, 0.62, -10), ant(455, 320, 0.62, -18), ant(610, 250, 0.62, -24), ant(758, 190, 0.62, -26),
        label(138, 380, 65, 475, "蚁巢入口"),
        label(485, 305, 535, 455, "反复经过会加强气味路线"),
        label(862, 145, 735, 90, "食物"),
    ])
    return "".join(bits)


def question_94() -> str:
    bits = [panel_title("蚁巢剖面里，不同成员做不同工作")]
    bits.extend([
        path("M80,172 Q360,75 650,172 L710,500 H52 Z", SOIL_LIGHT, INK, 4),
        path("M230,180 Q220,260 300,275 T285,395", "none", CREAM, 30),
        ellipse(205, 255, 80, 48, CREAM, INK, 3), ellipse(435, 218, 88, 52, CREAM, INK, 3), ellipse(475, 365, 105, 58, CREAM, INK, 3),
        ant(200, 255, 0.54), ant(430, 218, 0.50), ant(515, 360, 0.52),
        ellipse(430, 368, 37, 26, ORANGE_DARK, INK, 3), circle(388, 368, 22, ORANGE, INK, 3),
        circle(450, 232, 13, "#F6F1D5", INK, 2), circle(475, 225, 12, "#F6F1D5", INK, 2), circle(498, 237, 11, "#F6F1D5", INK, 2),
        ant(705, 150, 0.52, -18),
        label(430, 367, 700, 160, "蚁后：主要负责产卵"),
        label(475, 230, 700, 260, "幼虫和蛹由工蚁照料"),
        label(500, 355, 700, 360, "工蚁搬运、清理和照料"),
        label(705, 150, 700, 455, "外出工蚁寻找食物"),
    ])
    return "".join(bits)


def question_95() -> str:
    bits = [panel_title("蜜蜂的翅和胸部飞行肌让空气产生振动"), insect_side(390, 285, 1.0, True)]
    bits.extend([
        path("M330,145 Q390,92 470,155 M345,165 Q405,112 485,176", "none", BLUE, 4, True),
        path("M585,220 Q640,185 695,220 M600,255 Q670,210 740,255 M615,292 Q700,235 785,292", "none", TEAL, 5),
        ellipse(354, 287, 32, 38, RED, "none"),
        label(355, 288, 700, 115, "胸部飞行肌反复收缩"),
        label(420, 158, 700, 220, "翅快速拍动和扭转"),
        label(700, 255, 700, 335, "空气压力波传到耳朵"),
        label(286, 285, 700, 430, "嗡声主要来自飞行振动"),
    ])
    return "".join(bits)


def question_96() -> str:
    bits = [panel_title("蝴蝶把卷曲的口器伸进花里吸取花蜜")]
    bits.extend([
        ellipse(315, 230, 62, 92, ORANGE, INK, 4), ellipse(455, 230, 62, 92, TEAL, INK, 4),
        ellipse(325, 355, 52, 65, YELLOW, INK, 4), ellipse(445, 355, 52, 65, ORANGE, INK, 4),
        rect(376, 180, 18, 210, INK, INK, 2, 9), circle(385, 164, 24, INK),
        path("M382,174 C430,175 500,245 520,326 C529,365 551,360 555,334", "none", "#57483D", 8),
        path("M555,334 q22,28 39,0 q14,-22 30,1", "none", "#57483D", 6),
        line(565, 360, 565, 440, GREEN, 13),
        circle(565, 330, 35, RED, INK, 3), circle(530, 350, 30, ORANGE, INK, 3), circle(600, 350, 30, YELLOW, INK, 3), circle(565, 370, 30, "#E889AA", INK, 3),
        circle(565, 345, 20, "#F2C75C", INK, 2),
        label(505, 302, 705, 145, "口器平时卷起"),
        label(555, 334, 705, 255, "吸食时伸成长管"),
        label(565, 350, 705, 370, "花蜜藏在花的深处"),
        text(330, 485, "这根长管叫喙，不是吸管一样的硬管", 20, INK, "middle", 600),
    ])
    return "".join(bits)


def question_97() -> str:
    bits = [panel_title("蝴蝶的一生会经过四种可辨认的身体阶段")]
    xs = [125, 360, 590, 825]
    bits.extend([
        ellipse(xs[0], 285, 80, 55, GREEN, INK, 3),
        circle(xs[0] - 20, 275, 10, "#F4E9B9", INK, 2), circle(xs[0] + 4, 260, 10, "#F4E9B9", INK, 2), circle(xs[0] + 27, 280, 10, "#F4E9B9", INK, 2),
    ])
    for i in range(6):
        bits.append(circle(xs[1] - 60 + i * 24, 285 + (8 if i % 2 else -4), 25, GREEN if i % 2 else YELLOW, INK, 3))
    bits.extend([
        path(f"M{xs[2]-36},210 Q{xs[2]},175 {xs[2]+36},210 L{xs[2]+18},338 Q{xs[2]},370 {xs[2]-18},338 Z", CREAM, INK, 4),
        line(xs[2], 185, xs[2], 130, SOIL, 8),
        ellipse(xs[3] - 55, 245, 60, 84, ORANGE, INK, 4), ellipse(xs[3] + 55, 245, 60, 84, TEAL, INK, 4),
        ellipse(xs[3] - 44, 345, 44, 52, YELLOW, INK, 3), ellipse(xs[3] + 44, 345, 44, 52, ORANGE, INK, 3),
        rect(xs[3] - 8, 195, 16, 185, INK, INK, 1, 8),
    ])
    for left, right in zip(xs, xs[1:]):
        bits.append(line(left + 92, 285, right - 92, 285, TEAL, 5, True))
    for x, value in zip(xs, ("卵", "幼虫（毛毛虫）", "蛹", "成虫（蝴蝶）")):
        bits.append(text(x, 465, value, 21, INK, "middle", 700))
    return "".join(bits)


def question_98() -> str:
    bits = [panel_title("甲虫的硬壳是变硬的前翅，下面藏着飞行后翅")]
    bits.extend([
        circle(400, 160, 55, INK), ellipse(400, 305, 130, 165, ORANGE, INK, 5),
        path("M400,150 C315,165 280,235 285,380 Q340,455 400,468 Z", ORANGE_DARK, INK, 4),
        path("M400,150 C500,115 610,115 690,180 C615,230 525,270 410,300 Z", RED, INK, 4),
        path("M410,285 C520,205 650,225 705,345 C600,375 495,350 410,310 Z", "#DDEFF0", BLUE, 4),
    ])
    for offset in range(0, 6):
        bits.append(path(f"M{440+offset*35},275 Q{500+offset*22},315 {465+offset*35},350", "none", TEAL, 2))
    for side in (-1, 1):
        for y in (250, 315, 375):
            bits.append(line(400 + side * 75, y, 400 + side * 175, y + side * 25, INK, 5))
    bits.extend([
        label(580, 150, 700, 145, "鞘翅：坚硬的前翅"),
        label(610, 320, 700, 275, "后翅：薄而能折叠"),
        label(400, 165, 700, 395, "飞行前，鞘翅先打开"),
    ])
    return "".join(bits)


def question_99() -> str:
    bits = [panel_title("瓢虫成虫的斑点不会每天增加，不同种图案可不同")]
    bits.extend([
        ladybug(210, 285, 0.95, ((-25, -8), (25, -8), (-30, 30), (30, 30))),
        ladybug(480, 285, 0.95, ((-25, -16), (25, -16), (0, 16), (-25, 44), (25, 44))),
        ladybug(750, 285, 0.95, ((-28, 4), (28, 4))),
        text(210, 450, "四斑型", 21, INK, "middle", 700), text(480, 450, "多斑型", 21, INK, "middle", 700), text(750, 450, "少斑型", 21, INK, "middle", 700),
        text(480, 493, "斑点数量主要与种类和遗传有关，不是年龄刻度", 21, ORANGE_DARK, "middle", 700),
    ])
    return "".join(bits)


def question_100() -> str:
    bits = [panel_title("萤火虫腹部末端有专门的发光器")]
    bits.append(insect_side(390, 280, 1.15, True))
    bits.extend([
        ellipse(535, 290, 80, 48, "#F8E66D", YELLOW, 4),
        ellipse(535, 290, 118, 75, "none", "#F8E66D", 5),
        circle(720, 250, 42, "#F8E66D", INK, 3),
        circle(704, 242, 7, ORANGE), circle(728, 264, 7, TEAL), circle(741, 235, 7, "#FFFFFF"),
        label(535, 290, 715, 130, "腹部发光器"),
        label(720, 250, 675, 260, "发光物质、酶和氧参与反应"),
        label(470, 250, 715, 390, "化学能大多变成光，热很少"),
    ])
    return "".join(bits)


def question_101() -> str:
    bits = [panel_title("会吸血的是部分种类的雌蚊，口器由多根细结构组成")]
    bits.extend([
        rect(70, 385, 540, 90, "#EFAE98", INK, 3),
        ellipse(360, 205, 32, 90, ORANGE, INK, 4), circle(360, 110, 32, TEAL, INK, 4),
        ellipse(292, 175, 74, 34, "#E9F5F3", BLUE, 3), ellipse(428, 175, 74, 34, "#E9F5F3", BLUE, 3),
        line(360, 130, 360, 385, INK, 6),
    ])
    for side in (-1, 1):
        for y in (180, 225, 270):
            bits.append(path(f"M{360+side*18},{y} L{360+side*95},{y+45} L{360+side*135},{y+25}", "none", INK, 4))
    bits.extend([
        line(352, 385, 345, 435, RED, 3), line(360, 385, 360, 438, ORANGE_DARK, 3), line(368, 385, 375, 435, TEAL, 3),
        circle(780, 205, 34, TEAL, INK, 4),
    ])
    for angle in range(0, 360, 20):
        radians = math.radians(angle)
        bits.append(line(780, 205, 780 + math.cos(radians) * 70, 205 + math.sin(radians) * 70, INK, 2))
    bits.extend([
        label(360, 385, 690, 105, "雌蚊细长口器刺入皮肤"),
        label(360, 435, 690, 325, "吸血可为产卵提供蛋白质"),
        label(780, 205, 690, 445, "雄蚊触角常更蓬松，多取食花蜜"),
    ])
    return "".join(bits)


def question_102() -> str:
    bits = [panel_title("蜻蜓的四片翅能分别改变拍动时机和角度")]
    bits.extend([
        circle(480, 120, 38, TEAL, INK, 4),
        path("M480,150 Q464,255 480,445 Q496,255 480,150 Z", ORANGE, INK, 4),
        path("M455,190 Q280,75 115,185 Q260,260 455,245 Z", "#E5F2F1", TEAL, 4),
        path("M505,190 Q680,75 845,185 Q700,260 505,245 Z", "#E5F2F1", TEAL, 4),
        path("M458,255 Q300,220 160,380 Q330,410 470,310 Z", "#EDF5FB", BLUE, 4),
        path("M502,255 Q660,220 800,380 Q630,410 490,310 Z", "#EDF5FB", BLUE, 4),
        path("M468,190 L405,150 L370,165 M492,190 L555,150 L590,165", "none", INK, 5),
        path("M466,225 L395,250 L360,285 M494,225 L565,250 L600,285", "none", INK, 5),
        path("M470,260 L415,315 L380,345 M490,260 L545,315 L580,345", "none", INK, 5),
        path("M220,120 Q145,80 110,135", "none", ORANGE, 5, True),
        path("M740,120 Q815,80 850,135", "none", ORANGE, 5, True),
        path("M205,420 Q130,455 120,390", "none", BLUE, 5, True),
        path("M755,420 Q830,455 840,390", "none", BLUE, 5, True),
        label(325, 185, 75, 105, "前翅"), label(330, 320, 75, 425, "后翅"),
        label(480, 285, 690, 260, "四片翅不必完全同步"),
        label(480, 400, 690, 390, "左右差异帮助急转和悬停"),
    ])
    return "".join(bits)


def question_103() -> str:
    bits = [panel_title("蚱蜢粗大的后腿先储能，再突然伸直")]
    bits.extend([
        insect_side(340, 235, 0.9, True),
        path("M300,285 L180,390 L95,405", "none", ORANGE_DARK, 24),
        path("M300,285 L180,390 L95,405", "none", INK, 4),
        path("M315,270 Q235,315 190,370", "none", RED, 11),
        path("M130,410 Q220,495 340,425", "none", TEAL, 6, True),
        ellipse(690, 250, 105, 62, CREAM, INK, 3),
        path("M620,270 Q690,185 760,270", "none", RED, 18),
        path("M620,275 Q690,210 760,275", "none", YELLOW, 7),
        label(250, 340, 705, 120, "大腿里的肌肉很粗"),
        label(185, 390, 705, 225, "关节和外骨骼也能储存弹性能"),
        label(320, 425, 705, 340, "突然伸腿把身体推离地面"),
        label(690, 240, 705, 450, "局部放大：肌肉牵拉关节"),
    ])
    return "".join(bits)


def question_104() -> str:
    bits = [panel_title("雄蟋蟀摩擦前翅上的锉纹和刮片发声")]
    bits.extend([
        insect_side(345, 300, 1.05, False),
        path("M315,255 Q405,185 505,285 Q405,325 315,292 Z", CREAM, ORANGE_DARK, 4),
        path("M228,263 Q135,135 76,175 M246,250 Q175,112 118,130", "none", INK, 4),
        path("M390,342 L285,435 L165,420", "none", ORANGE_DARK, 22),
        path("M390,342 L285,435 L165,420", "none", INK, 4),
        path("M305,215 Q370,160 445,220", "none", ORANGE_DARK, 8),
        rect(660, 145, 200, 165, CREAM, INK, 3, 10),
        line(685, 245, 815, 190, ORANGE_DARK, 8),
    ])
    for index in range(9):
        x = 690 + index * 14
        bits.append(line(x, 225 - index * 6, x + 9, 243 - index * 6, INK, 2))
    bits.extend([
        path("M650,355 Q700,325 750,355 M665,395 Q735,350 805,395 M700,435 Q780,380 860,435", "none", TEAL, 5),
        label(370, 205, 710, 105, "两片前翅互相摩擦"),
        label(748, 215, 710, 330, "一边像小锉，一边像刮片"),
        label(775, 395, 710, 485, "翅面振动推动空气形成声波"),
    ])
    return "".join(bits)


def question_105() -> str:
    bits = [panel_title("螳螂举起的是带刺、能快速折叠的捕捉前足")]
    bits.extend([
        ellipse(410, 275, 45, 115, GREEN, INK, 4), polygon([(390, 165), (430, 165), (460, 95), (410, 70), (360, 95)], TEAL),
        circle(390, 100, 8, "#FFFFFF", INK, 2), circle(430, 100, 8, "#FFFFFF", INK, 2),
        path("M382,205 L275,135 L215,235 L315,260", "none", GREEN, 18),
        path("M438,205 L545,135 L605,235 L505,260", "none", GREEN, 18),
        path("M275,135 L215,235 M545,135 L605,235", "none", INK, 4),
    ])
    for side in (-1, 1):
        base = 410 + side * 195
        for index in range(5):
            x = base - side * index * 15
            bits.append(polygon([(x, 220 + index * 3), (x + side * 17, 239 + index * 2), (x + side * 4, 244 + index * 3)], ORANGE, INK, 1.5))
    bits.extend([
        path("M392,330 L300,435 L245,425 M428,330 L520,435 L575,425", "none", INK, 10),
        label(275, 135, 705, 115, "前足的多个关节能迅速折叠"),
        label(230, 235, 705, 235, "内侧尖刺帮助夹住猎物"),
        label(410, 185, 705, 355, "静止举起是在等待，不是在祈祷"),
    ])
    return "".join(bits)


def question_106() -> str:
    bits = [panel_title("竹节虫的身体和腿在真实树枝旁很难被发现")]
    bits.extend([
        path("M90,430 Q280,335 455,360 T865,190", "none", SOIL, 24),
        path("M310,345 Q260,270 215,220 M590,310 Q640,235 700,190", "none", SOIL, 14),
        ellipse(475, 300, 18, 128, GREEN, INK, 4), circle(475, 158, 23, GREEN, INK, 4),
        path("M468,142 Q450,108 430,118 M482,142 Q505,110 520,125", "none", INK, 3),
    ])
    for y, spread in ((215, 105), (280, 135), (350, 115)):
        bits.append(path(f"M475,{y} L{475-spread},{y-45} L{475-spread-55},{y-25}", "none", INK, 5))
        bits.append(path(f"M475,{y} L{475+spread},{y+35} L{475+spread+55},{y+15}", "none", INK, 5))
    bits.extend([
        dashed_curve("M445,145 Q410,300 455,430", ORANGE, 3),
        label(475, 240, 720, 115, "细长身体像一段嫩枝"),
        label(610, 315, 720, 250, "关节腿也像分叉小枝"),
        label(475, 380, 720, 385, "颜色和静止姿势共同形成伪装"),
    ])
    return "".join(bits)


def snail(cx: float, cy: float, scale: float = 1.0) -> str:
    s = scale
    bits = [
        path(f"M{cx-135*s},{cy+55*s} Q{cx-40*s},{cy+15*s} {cx+105*s},{cy+45*s} Q{cx+145*s},{cy+58*s} {cx+120*s},{cy+78*s} H{cx-120*s} Z", TEAL_LIGHT, INK, 4 * s),
        circle(cx - 35 * s, cy - 10 * s, 78 * s, ORANGE, INK, 5 * s),
        path(f"M{cx-35*s},{cy-10*s} q{48*s},{-45*s} {55*s},{10*s} q{4*s},{48*s} {-40*s},{42*s} q{-34*s},{-4*s} {-26*s},{-34*s}", "none", ORANGE_DARK, 5 * s),
    ]
    for offset in (60, 92):
        bits.append(line(cx + offset * s, cy + 45 * s, cx + (offset + 10) * s, cy - 20 * s, INK, 3 * s))
        bits.append(circle(cx + (offset + 10) * s, cy - 24 * s, 5 * s, INK))
    return "".join(bits)


def question_107() -> str:
    bits = [panel_title("蜗牛壳是身体分泌并不断加大的保护结构"), snail(350, 285, 1.25)]
    bits.extend([
        path("M306,263 q58,-55 68,12 q5,57 -48,50", "none", CREAM, 12),
        label(315, 220, 715, 115, "壳：坚硬、会随身体长大"),
        label(370, 330, 715, 225, "外套膜：分泌形成壳的材料"),
        label(445, 355, 715, 335, "柔软身体可以缩回壳口"),
        label(325, 430, 715, 440, "宽大的足贴着地面移动"),
    ])
    return "".join(bits)


def question_108() -> str:
    bits = [panel_title("蜗牛用肌肉波推地，并在黏液上滑行"), snail(365, 245, 1.15)]
    bits.extend([
        path("M105,420 Q350,375 640,420", "none", "#9ED8D5", 20),
        path("M150,405 C205,365 260,445 315,405 S425,365 480,405 S590,445 645,405", "none", TEAL, 7, True),
        circle(195, 432, 7, "#CDEDEC"), circle(275, 425, 6, "#CDEDEC"), circle(530, 432, 8, "#CDEDEC"),
        label(335, 350, 710, 130, "足底肌肉一段段收缩"),
        label(430, 410, 710, 255, "收缩波沿足部向后传"),
        label(260, 424, 710, 380, "黏液减少擦伤，也帮助传力"),
        text(365, 492, "亮亮的痕迹是很薄的黏液层，不是蜗牛融化了", 20, INK, "middle", 600),
    ])
    return "".join(bits)


def question_109() -> str:
    bits = [panel_title("蚯蚓靠两层肌肉改变体形，再用刚毛固定身体")]
    bits.extend([worm(330, 270, 1.0)])
    for index in range(-5, 6):
        x = 330 + index * 26
        bits.append(line(x, 310 + math.sin(index * 0.9) * 25, x - 9, 333 + math.sin(index * 0.9) * 25, INK, 2))
    bits.extend([
        circle(720, 245, 105, "#C87569", INK, 4), circle(720, 245, 70, CREAM, INK, 3),
        path("M630,245 A90,90 0 0 1 810,245", "none", RED, 18),
        path("M720,155 A90,90 0 0 1 720,335", "none", BLUE, 18),
        label(330, 255, 80, 115, "许多相似体节"),
        label(300, 325, 80, 435, "短刚毛抓住土粒"),
        label(660, 185, 710, 145, "环形肌收缩：变细变长"),
        label(765, 285, 710, 350, "纵向肌收缩：变粗变短"),
    ])
    return "".join(bits)


def question_110() -> str:
    bits = [panel_title("蚯蚓洞道让空气和水进入土壤，也混合有机物")]
    bits.extend([
        rect(65, 125, 620, 360, SOIL_LIGHT, INK, 4),
        rect(65, 125, 620, 62, "#6F8D4D", INK, 2),
        path("M145,175 Q170,245 155,480 M290,175 Q255,270 300,480 M505,175 Q465,260 480,480", "none", CREAM, 24),
        path("M105,135 q35,-36 70,0 M230,142 q42,-45 85,0 M520,138 q40,-38 80,0", "none", "#9A6A3F", 10),
        worm(300, 332, 0.55),
        path("M120,126 Q145,70 170,126 M245,126 Q280,60 315,126 M500,126 Q540,70 580,126", "none", GREEN, 7),
        circle(150, 465, 8, "#5A3D2C"), circle(166, 469, 7, "#5A3D2C"), circle(520, 460, 8, "#5A3D2C"),
        label(160, 265, 730, 125, "洞道增加通气和排水通路"),
        label(300, 330, 730, 245, "蚯蚓吞食土和腐烂碎屑"),
        label(540, 132, 730, 355, "落叶碎屑被带入土中"),
        label(520, 460, 730, 455, "排出的蚓粪仍是土壤的一部分"),
    ])
    return "".join(bits)


def many_legged(cx: float, cy: float, segments: int, pairs: int, color: str, scale: float = 1.0) -> str:
    s = scale
    bits = []
    for index in range(segments):
        x = cx + (index - segments / 2) * 30 * s
        y = cy + math.sin(index * 0.55) * 10 * s
        bits.append(circle(x, y, 22 * s, color if index % 2 else ORANGE, INK, 2.5 * s))
        for pair_index in range(pairs):
            offset = (pair_index - (pairs - 1) / 2) * 9 * s
            bits.append(line(x + offset, y + 16 * s, x + offset - 10 * s, y + 48 * s, INK, 2.5 * s))
    bits.append(circle(cx - (segments / 2 + 0.55) * 30 * s, cy, 26 * s, TEAL, INK, 3 * s))
    return "".join(bits)


def question_111() -> str:
    bits = [panel_title("蜈蚣和马陆都分节，但身体形状和腿的排列不同")]
    bits.extend([
        many_legged(260, 245, 7, 1, RED, 1.0),
        many_legged(700, 245, 7, 2, "#6D7659", 1.0),
        line(480, 90, 480, 465, "#94A2A9", 2, False, "8 10"),
        text(260, 410, "蜈蚣", 26, ORANGE_DARK, "middle", 760),
        text(700, 410, "马陆", 26, TEAL, "middle", 760),
        text(260, 448, "身体较扁｜每节一对足｜行动较快", 20, INK, "middle", 600),
        text(700, 448, "身体较圆｜多数体节两对足｜常吃腐屑", 20, INK, "middle", 600),
        label(125, 245, 70, 115, "头后有捕食用的毒爪"),
        label(575, 285, 700, 115, "很多短足依次摆动"),
    ])
    return "".join(bits)


def pillbug_open(cx: float, cy: float, scale: float = 1.0) -> str:
    s = scale
    bits = [ellipse(cx, cy, 130 * s, 78 * s, "#6D7B78", INK, 5 * s)]
    for index in range(-3, 4):
        x = cx + index * 30 * s
        bits.append(path(f"M{x},{cy-68*s} Q{x+18*s},{cy} {x},{cy+68*s}", "none", CREAM, 3 * s))
        bits.append(line(x, cy + 55 * s, x - 10 * s, cy + 98 * s, INK, 3 * s))
        bits.append(line(x + 9 * s, cy + 52 * s, x + 20 * s, cy + 92 * s, INK, 3 * s))
    bits.append(circle(cx - 116 * s, cy - 4 * s, 18 * s, TEAL, INK, 3 * s))
    bits.append(path(f"M{cx-126*s},{cy-16*s} Q{cx-155*s},{cy-47*s} {cx-168*s},{cy-30*s}", "none", INK, 2.5 * s))
    return "".join(bits)


def question_112() -> str:
    bits = [panel_title("潮虫把一片片背甲合拢，保护柔软腹面")]
    bits.extend([
        pillbug_open(285, 265, 0.9),
        circle(700, 270, 108, "#6D7B78", INK, 5),
    ])
    for radius in (28, 51, 75, 98):
        bits.append(path(f"M{700-radius},270 A{radius},{radius} 0 0 1 {700+radius},270", "none", CREAM, 4))
        bits.append(path(f"M{700-radius},270 A{radius},{radius} 0 0 0 {700+radius},270", "none", "#596966", 2))
    bits.extend([
        path("M440,270 Q500,225 560,270", "none", TEAL, 6, True),
        text(285, 445, "展开：背甲重叠，腹面有许多足", 21, INK, "middle", 650),
        text(700, 445, "卷曲：头尾靠近，甲片围成保护球", 21, INK, "middle", 650),
        label(255, 206, 65, 115, "七对步足藏在腹面"),
        label(285, 320, 65, 355, "柔软腹面朝下"),
        label(700, 170, 815, 120, "硬背甲朝外"),
        label(700, 335, 815, 350, "腹面被包住"),
    ])
    return "".join(bits)


def orb_web(cx: float, cy: float, radius: float) -> str:
    bits = []
    for index in range(12):
        angle = index * math.tau / 12
        bits.append(line(cx, cy, cx + math.cos(angle) * radius, cy + math.sin(angle) * radius, BLUE, 2.4))
    points = []
    for turn in range(1, 9):
        rr = radius * turn / 9
        ring = []
        for index in range(49):
            angle = index * math.tau / 48
            ring.append((cx + math.cos(angle) * rr, cy + math.sin(angle) * rr))
        data = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in ring)
        bits.append(path(data, "none", TEAL if turn < 3 else "#77AFA9", 1.7))
        points.extend(ring)
    return "".join(bits)


def question_113() -> str:
    bits = [panel_title("圆网把外框、辐射丝和捕捉螺旋连成受力网络"), orb_web(420, 290, 205), spider(420, 290, 0.30)]
    bits.extend([
        line(215, 85, 625, 85, INK, 5), line(185, 105, 130, 30, INK, 5), line(650, 110, 725, 35, INK, 5),
        label(420, 85, 700, 110, "外框丝连接固定点"),
        label(420, 290, 700, 225, "辐射丝把拉力传向四周"),
        label(510, 340, 700, 345, "螺旋捕捉丝吸收撞击"),
        label(225, 175, 675, 455, "多条丝共同分担，不靠一根丝"),
    ])
    return "".join(bits)


def question_114() -> str:
    bits = [panel_title("蜘蛛多沿不黏的辐射丝和网心移动")]
    bits.append(orb_web(420, 290, 205))
    for radius in (95, 130, 165):
        bits.append(ellipse(420, 290, radius, radius, "none", RED, 4))
    bits.extend([
        spider(420, 290, 0.42),
        path("M420,290 L420,95 M420,290 L595,390", "none", "#F3D36B", 8),
        label(420, 120, 695, 105, "辐射丝通常不带黏胶"),
        label(550, 290, 695, 240, "捕捉螺旋上有黏滴"),
        label(420, 290, 675, 365, "足端小爪和动作也减少粘住"),
        text(420, 500, "蜘蛛仍可能失足粘住，并不是完全不怕自己的网", 20, INK, "middle", 600),
    ])
    return "".join(bits)


def question_115() -> str:
    bits = [panel_title("石头下面较暗、较湿，形成小动物的微环境")]
    bits.extend([
        rect(50, 265, 650, 220, SOIL_LIGHT, INK, 4),
        path("M80,245 Q155,80 350,105 Q540,110 650,245 Z", "#7D8584", INK, 5),
        path("M105,270 Q320,230 635,270", "none", "#74A99A", 16),
        path("M155,260 Q135,205 100,180 M270,260 Q285,190 330,160 M520,260 Q535,195 580,170", "none", GREEN, 7),
        pillbug_open(210, 345, 0.32),
        many_legged(420, 350, 5, 1, RED, 0.38),
        worm(535, 425, 0.34),
        circle(160, 420, 8, "#8C6546"), circle(180, 430, 7, "#8C6546"),
        label(300, 260, 680, 105, "石头挡住阳光和干燥的风"),
        label(210, 345, 680, 220, "潮虫需要较湿环境帮助呼吸"),
        label(420, 350, 680, 335, "蜈蚣在缝隙中寻找小猎物"),
        label(535, 425, 665, 445, "蚯蚓和腐屑生物活动在土中"),
    ])
    return "".join(bits)


RENDERERS = {
    91: question_91,
    92: question_92,
    93: question_93,
    94: question_94,
    95: question_95,
    96: question_96,
    97: question_97,
    98: question_98,
    99: question_99,
    100: question_100,
    101: question_101,
    102: question_102,
    103: question_103,
    104: question_104,
    105: question_105,
    106: question_106,
    107: question_107,
    108: question_108,
    109: question_109,
    110: question_110,
    111: question_111,
    112: question_112,
    113: question_113,
    114: question_114,
    115: question_115,
}


def invertebrate_body(question: int, palette: tuple[str, ...]) -> str:
    del palette
    try:
        return RENDERERS[question]()
    except KeyError as exc:
        raise KeyError(f"no invertebrate illustration for question {question}") from exc
