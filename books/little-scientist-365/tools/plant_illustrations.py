#!/usr/bin/env python3
"""Botany plates that replace generic sprout icons with observable structures."""

from __future__ import annotations

import math

from textbook_illustrations import INK, circle, ellipse, line, panel_title, path, rect, text


PLANT_QUESTIONS = frozenset(range(60, 91))
PLANT_KEYS = {question: (f"plant-plate-{question:03d}",) for question in PLANT_QUESTIONS}

GREEN = "#4F8A55"
GREEN_DARK = "#2F6744"
GREEN_LIGHT = "#A7C97E"
LEAF_LIGHT = "#8FC676"
BROWN = "#8D6546"
BROWN_DARK = "#5A4032"
SOIL = "#B8845D"
SOIL_DARK = "#76523B"
CREAM = "#F7E9C9"
YELLOW = "#E6B542"
ORANGE = "#D9703D"
RED = "#C95858"
BLUE = "#4D86B5"
TEAL = "#2D8178"
PURPLE = "#7A68A4"
WHITE = "#FFFFFF"
SKY = "#DDEFF3"


def polygon(points: list[tuple[float, float]], fill: str, stroke: str = INK, width: float = 3) -> str:
    values = " ".join(f"{x},{y}" for x, y in points)
    return f'<polygon points="{values}" fill="{fill}" stroke="{stroke}" stroke-width="{width}" stroke-linejoin="round"/>'


def label(tx: float, ty: float, x: float, y: float, value: str, color: str = INK) -> str:
    end_x = x - 12 if x >= tx else x + 12
    chunks = [value]
    if len(value) > 15:
        cut = round(len(value) / 2)
        chunks = [value[:cut], value[cut:]]
    copy = "".join(text(x, y + index * 22, chunk, 19, color, "start", 650) for index, chunk in enumerate(chunks))
    return line(tx, ty, end_x, y - 6, color, 2.5) + circle(tx, ty, 4.5, color) + copy


def arrow(data: str, color: str = INK, width: float = 4) -> str:
    return path(data, "none", color, width, True)


def garden_backdrop() -> str:
    return "".join([
        rect(34, 82, 892, 382, SKY, "none", 0, 4),
        path("M34,330 Q170,280 315,330 T610,330 T920,330 V464 H34 Z", GREEN_LIGHT),
        rect(34, 360, 892, 104, SOIL, "none"),
        path("M34,360 Q175,328 320,360 T610,360 T930,360", "none", GREEN, 7),
    ])


def soil_cutaway() -> str:
    return "".join([
        rect(34, 82, 892, 190, SKY, "none", 0, 4),
        rect(34, 272, 892, 192, SOIL, "none"),
        path("M34,272 Q165,245 300,272 T580,272 T860,272 T980,272", "none", GREEN, 7),
        circle(115, 335, 8, SOIL_DARK), circle(175, 402, 6, SOIL_DARK), circle(820, 330, 9, SOIL_DARK), circle(745, 420, 7, SOIL_DARK),
    ])


def seed(cx: float, cy: float, s: float = 1.0, cutaway: bool = False) -> str:
    shell = path(f"M{cx-92*s},{cy} C{cx-76*s},{cy-85*s} {cx+38*s},{cy-100*s} {cx+92*s},{cy-16*s} C{cx+65*s},{cy+77*s} {cx-52*s},{cy+88*s} {cx-92*s},{cy} Z", "#C68B50", INK, 4 * s)
    if not cutaway:
        return shell
    return "".join([
        shell,
        path(f"M{cx-62*s},{cy-2*s} C{cx-45*s},{cy-58*s} {cx+27*s},{cy-70*s} {cx+62*s},{cy-12*s} C{cx+40*s},{cy+47*s} {cx-35*s},{cy+52*s} {cx-62*s},{cy-2*s} Z", CREAM, BROWN_DARK, 3 * s),
        ellipse(cx - 4 * s, cy - 3 * s, 38 * s, 47 * s, YELLOW, BROWN_DARK, 3 * s),
        path(f"M{cx+22*s},{cy+20*s} q{33*s},{30*s} {48*s},{52*s}", "none", GREEN_DARK, 7 * s),
        circle(cx + 22 * s, cy + 18 * s, 8 * s, GREEN_DARK),
    ])


def sprout(cx: float, cy: float, s: float = 1.0, roots: bool = True) -> str:
    bits = [line(cx, cy, cx, cy - 125 * s, GREEN_DARK, 10 * s)]
    bits.extend([
        path(f"M{cx},{cy-85*s} Q{cx-65*s},{cy-145*s} {cx-105*s},{cy-98*s} Q{cx-62*s},{cy-50*s} {cx},{cy-70*s} Z", GREEN, INK, 3 * s),
        path(f"M{cx},{cy-112*s} Q{cx+65*s},{cy-167*s} {cx+110*s},{cy-115*s} Q{cx+62*s},{cy-64*s} {cx},{cy-88*s} Z", LEAF_LIGHT, INK, 3 * s),
    ])
    if roots:
        bits.extend([
            path(f"M{cx},{cy} C{cx-4*s},{cy+55*s} {cx+10*s},{cy+110*s} {cx-12*s},{cy+155*s}", "none", CREAM, 9 * s),
            path(f"M{cx-2*s},{cy+55*s} q{-55*s},{28*s} {-72*s},{68*s} M{cx+3*s},{cy+73*s} q{52*s},{25*s} {67*s},{70*s}", "none", CREAM, 5 * s),
        ])
    return "".join(bits)


def leaf(cx: float, cy: float, s: float = 1.0, color: str = GREEN) -> str:
    return "".join([
        path(f"M{cx-150*s},{cy+15*s} Q{cx-40*s},{cy-125*s} {cx+145*s},{cy-20*s} Q{cx+35*s},{cy+120*s} {cx-150*s},{cy+15*s} Z", color, INK, 4 * s),
        path(f"M{cx-130*s},{cy+10*s} Q{cx},{cy} {cx+125*s},{cy-18*s}", "none", GREEN_DARK, 6 * s),
        path(f"M{cx-70*s},{cy+2*s} l{-25*s},-50 M{cx-25*s},{cy-3*s} l{-20*s},-62 M{cx+30*s},{cy-8*s} l{-10*s},-63 M{cx+65*s},{cy-12*s} l35,-45", "none", GREEN_DARK, 3 * s),
    ])


def flower(cx: float, cy: float, s: float = 1.0, color: str = RED) -> str:
    bits = [line(cx, cy + 55 * s, cx, cy + 175 * s, GREEN_DARK, 9 * s)]
    for index in range(6):
        angle = index * math.tau / 6
        bits.append(ellipse(cx + math.cos(angle) * 52 * s, cy + math.sin(angle) * 52 * s, 33 * s, 48 * s, color if index % 2 else ORANGE, INK, 3 * s))
    bits.append(circle(cx, cy, 36 * s, YELLOW, INK, 3 * s))
    return "".join(bits)


def bee(cx: float, cy: float, s: float = 1.0) -> str:
    bits = [ellipse(cx, cy, 48 * s, 30 * s, YELLOW, INK, 3 * s), circle(cx + 42 * s, cy - 2 * s, 19 * s, BROWN_DARK, INK, 2 * s)]
    for offset in (-20, 4, 24):
        bits.append(line(cx + offset * s, cy - 25 * s, cx + offset * s, cy + 25 * s, INK, 6 * s))
    bits.extend([ellipse(cx - 12 * s, cy - 34 * s, 34 * s, 20 * s, WHITE, BLUE, 2 * s), ellipse(cx + 20 * s, cy - 36 * s, 34 * s, 20 * s, WHITE, BLUE, 2 * s)])
    return "".join(bits)


def tree(cx: float, cy: float, s: float = 1.0, bare: bool = False, conifer: bool = False) -> str:
    bits = [rect(cx - 23 * s, cy - 10 * s, 46 * s, 190 * s, BROWN, INK, 3 * s, 10 * s)]
    if conifer:
        for y, width in ((cy - 155 * s, 70 * s), (cy - 105 * s, 105 * s), (cy - 48 * s, 135 * s)):
            bits.append(polygon([(cx, y - 75 * s), (cx - width, y + 75 * s), (cx + width, y + 75 * s)], GREEN_DARK, INK, 3 * s))
    else:
        bits.extend([
            path(f"M{cx},{cy+15*s} Q{cx-40*s},{cy-75*s} {cx-95*s},{cy-112*s} M{cx},{cy} Q{cx+35*s},{cy-80*s} {cx+105*s},{cy-118*s} M{cx-6*s},{cy-35*s} Q{cx-25*s},{cy-115*s} {cx},{cy-165*s}", "none", BROWN_DARK, 12 * s),
        ])
        if not bare:
            for dx, dy, radius in ((-90, -125, 67), (-22, -170, 78), (70, -135, 72), (15, -95, 76)):
                bits.append(circle(cx + dx * s, cy + dy * s, radius * s, GREEN if dx % 2 else LEAF_LIGHT, INK, 3 * s))
    return "".join(bits)


def _q60() -> str:
    bits = [panel_title("种皮里面有幼小的胚，也有供刚发芽时使用的食物"), soil_cutaway(), seed(350, 285, 1.22, True)]
    bits.extend([label(250, 260, 655, 125, "种皮保护内部，吸水后逐渐变软"), label(350, 280, 655, 255, "胚会长成新的根、茎和叶"), label(410, 330, 655, 385, "子叶或胚乳储存淀粉、油和蛋白质")])
    return "".join(bits)


def _q61() -> str:
    bits = [panel_title("水、氧气和合适温度共同启动种子的代谢"), soil_cutaway(), seed(370, 310, 1.0)]
    bits.extend([
        path("M180,155 q-35,55 0,90 q35,-35 0,-90", BLUE, INK, 3),
        path("M510,180 q25,-55 50,0 q25,55 50,0", "none", TEAL, 5),
        circle(750, 185, 48, YELLOW, INK, 3),
        arrow("M205,235 Q265,270 300,285", BLUE), arrow("M575,230 Q500,270 440,290", TEAL), arrow("M710,220 Q565,255 450,300", ORANGE),
        text(180, 125, "水", 22, INK, "middle"), text(560, 135, "氧气", 22, INK, "middle"), text(750, 125, "适宜温度", 22, INK, "middle"),
        text(480, 475, "光对不同种子的作用不同，许多种子在土中也能开始萌发", 20, INK, "middle", 650),
    ])
    return "".join(bits)


def _q62() -> str:
    bits = [panel_title("胚根先突破种皮，把幼苗固定并开始吸水"), soil_cutaway()]
    positions = [(150, 300), (340, 300), (540, 300), (760, 300)]
    bits.extend([seed(positions[0][0], 300, 0.52), seed(positions[1][0], 300, 0.52, True)])
    bits.extend([
        seed(540, 285, 0.45), path("M550,315 Q545,365 525,410", "none", CREAM, 8),
        sprout(760, 300, 0.62, True),
        arrow("M215,300 L275,300", TEAL), arrow("M405,300 L470,300", TEAL), arrow("M605,300 L665,300", TEAL),
        text(150, 448, "吸水", 20, INK, "middle"), text(340, 448, "种皮裂开", 20, INK, "middle"), text(540, 448, "胚根先向下", 20, INK, "middle"), text(760, 448, "胚芽向上", 20, INK, "middle"),
    ])
    return "".join(bits)


def _q63() -> str:
    bits = [panel_title("根尖向前生长，后方的根毛扩大与土壤水的接触"), soil_cutaway(), sprout(330, 270, 1.0, True)]
    bits.extend([rect(650, 145, 210, 220, CREAM, INK, 3, 12)])
    for index in range(8):
        y = 185 + index * 22
        bits.append(path(f"M700,{y} q{-42},-12 -58,5 M720,{y} q42,-12 58,5", "none", CREAM, 3))
    bits.extend([line(710, 170, 710, 335, GREEN_DARK, 15), label(710, 245, 620, 410, "根毛从表皮细胞伸出，水和矿物离子在这里进入"), label(315, 405, 655, 115, "根冠保护不断穿过土壤的根尖")])
    return "".join(bits)


def _q64() -> str:
    bits = [panel_title("茎靠细胞膨压和支撑组织站立，也把水与糖送到各处"), garden_backdrop(), sprout(300, 330, 1.05, False)]
    bits.extend([rect(620, 135, 240, 235, CREAM, INK, 3, 12), circle(740, 250, 82, "#D5E7B9", INK, 4)])
    for index in range(10):
        angle = index * math.tau / 10
        bits.append(circle(740 + math.cos(angle) * 58, 250 + math.sin(angle) * 58, 12, BLUE if index % 2 else ORANGE, INK, 2))
    bits.extend([label(300, 275, 650, 105, "纤维和木质化细胞帮助抗弯"), label(740, 250, 615, 415, "茎横切面中的维管束：木质部运水，韧皮部运糖")])
    return "".join(bits)


def _q65() -> str:
    bits = [panel_title("薄而平的叶片能接到更多光，也缩短气体进入细胞的距离"), garden_backdrop(), leaf(330, 280, 1.0)]
    bits.extend([
        rect(650, 145, 220, 205, CREAM, INK, 3, 12),
        rect(680, 190, 160, 22, GREEN_DARK, INK, 2, 8), rect(680, 212, 160, 55, LEAF_LIGHT, INK, 2), rect(680, 267, 160, 55, "#B9D893", INK, 2),
        circle(720, 285, 10, WHITE, INK, 2), circle(760, 292, 12, WHITE, INK, 2), circle(800, 280, 9, WHITE, INK, 2),
        label(760, 245, 620, 405, "横切面很薄，内部仍有表皮、叶肉、叶脉和气孔"),
        text(395, 485, "叶形很多，但“扩大受光面、方便交换”是常见原则", 20, INK, "middle", 650),
    ])
    return "".join(bits)


def _q66() -> str:
    bits = [panel_title("叶绿素主要吸收红光和蓝光，较多绿光被反射或透过"), garden_backdrop(), leaf(330, 280, 1.0)]
    bits.extend([rect(650, 145, 220, 210, CREAM, INK, 3, 12)])
    for row in range(4):
        for col in range(5):
            bits.append(ellipse(685 + col * 36, 190 + row * 38, 15, 9, GREEN_DARK, INK, 2))
    bits.extend([arrow("M160,120 Q230,175 270,220", RED), arrow("M230,105 Q285,175 305,220", BLUE), arrow("M350,220 Q440,150 505,120", GREEN), text(160, 105, "被吸收", 19, INK, "middle"), text(495, 105, "绿光返回眼睛", 19, INK, "middle"), label(740, 245, 620, 405, "叶肉细胞里的叶绿体含有叶绿素")])
    return "".join(bits)


def _q67() -> str:
    bits = [panel_title("植物用光能把二氧化碳和水重新组合成糖"), garden_backdrop(), sprout(470, 330, 0.95, False), circle(150, 145, 48, YELLOW, INK, 3)]
    bits.extend([
        arrow("M185,165 Q305,180 390,230", ORANGE, 6),
        arrow("M160,355 Q290,340 400,300", BLUE, 5),
        arrow("M780,250 Q650,230 555,250", TEAL, 5),
        arrow("M555,285 Q655,330 785,335", GREEN_DARK, 5),
        text(145, 230, "光能", 21, INK, "middle"), text(150, 390, "根吸收的水", 21, INK, "middle"), text(795, 225, "二氧化碳", 21, INK, "middle"), text(800, 375, "糖和氧气", 21, INK, "middle"),
        text(480, 485, "糖既能供呼吸释放能量，也能变成淀粉、纤维和新的身体", 20, INK, "middle", 680),
    ])
    return "".join(bits)


def _q68() -> str:
    bits = [panel_title("叶片通过气孔交换气体，白天和夜晚都进行细胞呼吸"), garden_backdrop(), leaf(300, 270, 0.92)]
    bits.extend([rect(630, 135, 250, 235, CREAM, INK, 3, 12), ellipse(755, 250, 90, 58, "#B8D895", INK, 3), ellipse(715, 250, 48, 35, GREEN, INK, 3), ellipse(795, 250, 48, 35, GREEN, INK, 3), ellipse(755, 250, 20, 38, WHITE, INK, 3)])
    bits.extend([arrow("M755,205 Q755,165 755,145", BLUE), arrow("M755,295 Q755,335 755,355", TEAL), label(755, 250, 620, 410, "一对保卫细胞调节孔隙大小，控制气体和水汽进出"), text(380, 490, "光合作用和呼吸不是同一件事：有光时两者可同时发生", 20, INK, "middle", 650)])
    return "".join(bits)


def _q69() -> str:
    bits = [panel_title("花的颜色、气味和形状常与传粉者的感官相匹配"), garden_backdrop(), flower(350, 270, 1.05, PURPLE), bee(575, 190, 0.75)]
    bits.extend([path("M470,180 Q540,140 610,175", "none", PURPLE, 4, False, "7 7"), label(350, 270, 650, 135, "花瓣和蜜导图案引导传粉者靠近"), label(570, 190, 650, 275, "挥发性气味分子可在空气中传播"), label(350, 330, 650, 405, "并非所有花都鲜艳；风媒花常更朴素")])
    return "".join(bits)


def _q70() -> str:
    bits = [panel_title("花粉是装着雄性生殖细胞的微小结构"), garden_backdrop(), flower(300, 300, 0.92, ORANGE)]
    bits.extend([rect(620, 125, 250, 250, CREAM, INK, 3, 12), circle(745, 250, 75, YELLOW, INK, 4)])
    for index in range(18):
        angle = index * math.tau / 18
        bits.append(line(745 + math.cos(angle) * 72, 250 + math.sin(angle) * 72, 745 + math.cos(angle) * 95, 250 + math.sin(angle) * 95, ORANGE, 4))
    bits.extend([circle(745, 250, 28, ORANGE, INK, 3), label(315, 285, 630, 105, "花药产生并释放花粉"), label(745, 250, 615, 415, "花粉壁保护里面的细胞；外形随植物种类不同")])
    return "".join(bits)


def _q71() -> str:
    bits = [panel_title("蜜蜂携带花粉到另一朵同种花，受精后子房发育成果实"), garden_backdrop(), flower(225, 300, 0.72, PURPLE), flower(690, 300, 0.72, PURPLE), bee(455, 205, 0.78)]
    bits.extend([path("M310,215 Q455,125 600,215", "none", TEAL, 5, True, "10 8"), circle(415, 235, 6, YELLOW), circle(435, 228, 6, YELLOW), circle(455, 233, 6, YELLOW), label(455, 205, 650, 110, "花粉沾在身体毛和足部结构上"), label(690, 300, 650, 405, "花粉到达柱头只是传粉；之后还要萌发、受精")])
    return "".join(bits)


def _q72() -> str:
    bits = [panel_title("果实由花的一部分发育而来，包住并帮助传播种子"), garden_backdrop()]
    bits.extend([
        circle(360, 280, 135, RED, INK, 5),
        path("M360,145 q15,-55 45,-65", "none", BROWN, 12), leaf(430, 100, 0.25),
        path("M360,150 A130,130 0 0 1 360,410 L360,150", "#F4B6A0", INK, 4),
        ellipse(410, 255, 18, 30, BROWN_DARK, INK, 2), ellipse(410, 315, 18, 30, BROWN_DARK, INK, 2),
        label(410, 255, 650, 140, "种子含下一代的胚"), label(365, 215, 650, 275, "果肉可吸引动物，也能保护种子"), label(360, 150, 650, 405, "果皮来自子房壁或与其他花部共同形成"),
    ])
    return "".join(bits)


def _q73() -> str:
    bits = [panel_title("蒲公英果实带着冠毛，增大空气阻力并随风飘远"), garden_backdrop()]
    bits.extend([line(260, 385, 260, 205, GREEN_DARK, 9), circle(260, 190, 68, CREAM, INK, 3)])
    for index in range(18):
        angle = index * math.tau / 18
        bits.append(line(260, 190, 260 + math.cos(angle) * 72, 190 + math.sin(angle) * 72, WHITE, 3))
    for index, (x, y) in enumerate(((470, 205), (610, 165), (760, 220))):
        bits.extend([line(x, y, x, y + 62, BROWN_DARK, 3), circle(x, y + 70, 7, BROWN_DARK)])
        for ray in range(9):
            angle = ray * math.tau / 9
            bits.append(line(x, y, x + math.cos(angle) * 34, y + math.sin(angle) * 34, WHITE, 2))
    bits.extend([arrow("M365,255 Q580,305 805,175", TEAL, 5), label(610, 165, 620, 405, "冠毛不是降落伞布，而是一圈细毛形成的空气阻力结构")])
    return "".join(bits)


def _q74() -> str:
    bits = [panel_title("苍耳等果实表面的钩刺会勾住毛或织物"), garden_backdrop(), rect(130, 155, 260, 230, "#5E7FA5", INK, 3, 10)]
    for row in range(5):
        bits.append(line(150, 180 + row * 40, 365, 180 + row * 40, WHITE, 4))
    bits.append(ellipse(420, 275, 68, 48, BROWN, INK, 4))
    for index in range(16):
        angle = index * math.tau / 16
        x = 420 + math.cos(angle) * 67
        y = 275 + math.sin(angle) * 48
        bits.append(path(f"M{x},{y} q{18*math.cos(angle):.1f},{18*math.sin(angle):.1f} {24*math.cos(angle):.1f},{24*math.sin(angle):.1f}", "none", BROWN_DARK, 3))
    bits.extend([rect(650, 145, 200, 220, CREAM, INK, 3, 12), path("M700,310 Q700,190 760,225 Q805,260 790,310", "none", BROWN_DARK, 8), path("M760,225 q-18,-30 -35,-12", "none", BROWN_DARK, 4), label(420, 275, 620, 410, "钩刺附着只是搭便车，不会主动钻进皮肤")])
    return "".join(bits)


def _q75() -> str:
    bits = [panel_title("椰子果皮中有厚纤维层和空气空隙，能浮水并抗碰撞"), rect(34, 82, 892, 382, SKY, "none", 0, 4), path("M34,285 Q180,250 330,285 T630,285 T940,285 V464 H34 Z", "#9BD4DE")]
    bits.extend([ellipse(330, 300, 125, 92, BROWN, INK, 5), path("M330,210 A125,92 0 0 1 330,392 L330,210", "#D7B475", INK, 4), ellipse(355, 300, 72, 62, WHITE, INK, 3), ellipse(370, 300, 47, 42, "#DDEEF5", INK, 2), line(34, 330, 926, 330, BLUE, 5), label(260, 280, 650, 130, "外层果皮耐磨"), label(330, 250, 650, 260, "纤维层夹着空气，平均密度较低"), label(380, 300, 650, 390, "坚硬内果皮和胚乳保护种子")])
    return "".join(bits)


def _q76() -> str:
    bits = [panel_title("温带树木每年形成一圈早材和晚材，宽窄记录生长条件"), garden_backdrop(), circle(360, 285, 170, "#D6A66F", INK, 5)]
    for radius in (30, 55, 82, 112, 143):
        bits.append(circle(360, 285, radius, "none", BROWN_DARK if radius % 2 else CREAM, 5))
    bits.extend([line(360, 285, 510, 210, RED, 4), label(425, 250, 650, 140, "一组浅色早材与深色晚材常组成一年轮"), label(300, 285, 650, 285, "最里面不是树的出生日期标签，而是较早形成的木材"), label(505, 330, 650, 410, "干旱、损伤等会改变宽度，所以年轮并不总规则")])
    return "".join(bits)


def _q77() -> str:
    bits = [panel_title("树皮不是一层壳：外层防护，内侧韧皮部运输糖"), garden_backdrop(), rect(170, 120, 340, 300, BROWN, INK, 4, 70)]
    bits.extend([rect(220, 120, 240, 300, "#C99664", INK, 3, 45), rect(275, 120, 130, 300, CREAM, INK, 3, 30), line(275, 140, 275, 400, ORANGE, 8), label(185, 260, 650, 130, "外树皮减慢失水并抵挡机械伤害"), label(250, 260, 650, 270, "内树皮中的韧皮部运输叶片制造的糖"), label(330, 260, 650, 405, "树干增粗时，新树皮在里面形成，外层会裂开脱落")])
    return "".join(bits)


def _q78() -> str:
    bits = [panel_title("落叶树在寒冷或干旱季节前回收养分并封住叶柄"), garden_backdrop()]
    colors = (GREEN, "#D98B3D", "#B96A3D", BROWN_DARK)
    for index, color in enumerate(colors):
        x = 130 + index * 220
        bits.append(tree(x, 330, 0.45, bare=index == 3))
        if index < 3:
            bits.append(circle(x, 230, 70, color, INK, 2))
    bits.extend([arrow("M190,225 L260,225", TEAL), arrow("M410,225 L480,225", TEAL), arrow("M630,225 L700,225", TEAL), text(130, 445, "生长", 19, INK, "middle"), text(350, 445, "叶绿素减少", 19, INK, "middle"), text(570, 445, "养分回收", 19, INK, "middle"), text(790, 445, "形成离层后落叶", 19, INK, "middle")])
    return "".join(bits)


def _q79() -> str:
    bits = [panel_title("松针面积小、表皮厚，能减慢冬季失水"), garden_backdrop(), tree(300, 330, 0.8, conifer=True)]
    bits.extend([rect(625, 135, 235, 230, CREAM, INK, 3, 12), ellipse(742, 250, 30, 92, GREEN_DARK, INK, 4), ellipse(742, 250, 17, 74, GREEN, INK, 3), circle(735, 290, 5, WHITE, INK, 2), circle(750, 290, 5, WHITE, INK, 2), label(742, 250, 620, 410, "针叶横切面：厚角质层、下陷气孔和紧密叶肉共同保水"), label(315, 180, 650, 110, "常绿不等于一片叶永远不落；针叶也会逐批更新")])
    return "".join(bits)


def _q80() -> str:
    bits = [panel_title("仙人掌把水储在肥厚茎里，叶退化成刺以减少蒸腾"), garden_backdrop()]
    bits.extend([path("M350,390 V190 Q350,130 410,130 Q470,130 470,190 V390 Z", GREEN, INK, 5), path("M350,270 Q285,270 285,220 V190 Q285,160 315,160 Q340,160 340,190", "none", GREEN, 40), path("M470,245 Q540,245 540,195", "none", GREEN, 40)])
    for x, y in ((365, 210), (440, 180), (390, 300), (455, 335), (300, 210), (525, 215)):
        bits.extend([line(x, y, x - 15, y - 15, CREAM, 3), line(x, y, x + 15, y - 15, CREAM, 3)])
    bits.extend([rect(650, 145, 210, 210, CREAM, INK, 3, 12), ellipse(755, 250, 72, 90, "#79B46C", INK, 4), ellipse(755, 250, 45, 62, "#B9D99E", INK, 3), circle(740, 235, 9, BLUE), circle(770, 270, 10, BLUE), label(755, 250, 620, 410, "储水组织含大量薄壁细胞；茎表皮与气孔还会控制失水"), label(350, 205, 650, 115, "刺是变态叶，也能遮阴和防止动物啃食")])
    return "".join(bits)


def _q81() -> str:
    bits = [panel_title("荷叶表面有微小凸起和蜡层，水珠难以铺展开"), rect(34, 82, 892, 382, SKY, "none", 0, 4), path("M34,330 Q180,300 320,330 T610,330 T930,330 V464 H34 Z", "#9BD4DE")]
    bits.extend([ellipse(330, 285, 180, 82, GREEN, INK, 5), path("M330,285 L470,240", "none", GREEN_DARK, 5), circle(300, 230, 35, BLUE, WHITE, 4), circle(410, 260, 24, BLUE, WHITE, 4), rect(640, 145, 220, 210, CREAM, INK, 3, 12)])
    for row in range(5):
        for col in range(7):
            bits.append(circle(670 + col * 27, 195 + row * 30, 8, GREEN, INK, 1.5))
    bits.extend([path("M690,180 q35,-70 70,0 q35,70 70,0", BLUE, INK, 3), label(750, 225, 620, 410, "粗糙凸起让水只接触少数高点，蜡层又不易被水润湿"), text(405, 490, "水珠滚动时常带走灰尘，这叫自清洁效应", 20, INK, "middle", 650)])
    return "".join(bits)


def _q82() -> str:
    bits = [panel_title("藤本植物用卷须或缠绕茎攀附支架，把叶片带到有光处"), garden_backdrop(), rect(420, 110, 35, 320, BROWN, INK, 3, 10)]
    bits.extend([path("M260,410 C270,330 420,350 335,255 C265,175 430,185 350,105", "none", GREEN_DARK, 12), path("M350,230 q95,-75 75,25 q-15,65 -60,25", "none", GREEN, 7), leaf(285, 305, 0.35), leaf(380, 170, 0.32), label(410, 245, 650, 125, "触碰支架的一侧生长改变，卷须逐渐弯曲"), label(290, 305, 650, 270, "攀爬减少自己制造粗壮支撑茎的成本"), label(350, 410, 650, 405, "根仍在土中吸水，藤不是靠支架“吸食”")])
    return "".join(bits)


def _q83() -> str:
    bits = [panel_title("单侧光照让幼茎两侧生长速度不同，于是逐渐向光弯曲"), garden_backdrop(), rect(75, 110, 240, 245, "#D7EEF2", INK, 5, 4), line(195, 110, 195, 355, INK, 4), line(75, 230, 315, 230, INK, 4)]
    bits.extend([circle(180, 170, 38, YELLOW, INK, 3), path("M520,395 C520,325 525,255 480,205", "none", GREEN_DARK, 14), leaf(470, 205, 0.33), path("M760,395 C750,310 700,245 610,210", "none", GREEN_DARK, 14), leaf(610, 210, 0.33), arrow("M330,235 Q430,210 470,210", ORANGE, 5), label(515, 300, 650, 115, "开始时茎近乎直立"), label(700, 270, 650, 340, "一段时间后，背光侧伸长更多，茎弯向窗户")])
    return "".join(bits)


def _q84() -> str:
    bits = [panel_title("蘑菇是真菌的繁殖结构，主体是土中细密的菌丝网"), soil_cutaway()]
    bits.extend([path("M300,275 Q350,145 450,275 Z", RED, INK, 4), rect(355, 275, 40, 105, CREAM, INK, 3, 12)])
    for row in range(4):
        for col in range(7):
            bits.append(path(f"M{185+col*72},{330+row*32} q35,-28 70,0", "none", CREAM, 3))
    bits.extend([label(375, 220, 650, 115, "地上的蘑菇会释放孢子"), label(405, 350, 650, 260, "菌丝分泌酶，把外界大分子分解后吸收"), label(260, 410, 650, 405, "真菌没有叶绿体，不属于植物")])
    return "".join(bits)


def _q85() -> str:
    bits = [panel_title("苔藓没有真正的维管束和深根，湿润环境帮助吸水与受精"), soil_cutaway()]
    for index in range(16):
        x = 100 + index * 28
        h = 35 + index % 4 * 12
        bits.extend([line(x, 310, x, 310 - h, GREEN_DARK, 5), circle(x, 300 - h, 10, GREEN)])
        if index % 4 == 0:
            bits.extend([line(x, 270 - h, x, 190 - h, BROWN, 3), ellipse(x, 180 - h, 10, 18, ORANGE, INK, 2)])
    bits.extend([path("M90,335 Q250,305 470,335", "none", BLUE, 8), rect(650, 145, 210, 205, CREAM, INK, 3, 12), path("M680,305 Q720,200 760,305 Q800,200 840,305", "none", GREEN, 10), circle(720, 240, 10, BLUE), circle(800, 250, 9, BLUE), label(750, 250, 620, 405, "叶状体只有一层或少数几层细胞，水可从表面进入"), label(180, 250, 650, 115, "孢蒴在细柄顶端释放孢子")])
    return "".join(bits)


def _q86() -> str:
    bits = [panel_title("蕨类不结果实，叶背的孢子囊群会释放孢子"), garden_backdrop()]
    bits.extend([path("M180,400 Q350,200 530,110", "none", GREEN_DARK, 12)])
    for index in range(9):
        t = index / 9
        x = 205 + 285 * t
        y = 375 - 235 * t
        bits.extend([leaf(x - 25, y + 15, 0.22), f'<g transform="translate({x+20} {y}) scale(-1 1)">{leaf(0,0,0.22)}</g>'])
    bits.extend([rect(640, 135, 230, 230, CREAM, INK, 3, 12), ellipse(755, 250, 75, 95, GREEN_LIGHT, INK, 3)])
    for row in range(4):
        for col in range(4):
            bits.append(circle(705 + col * 33, 205 + row * 35, 10, ORANGE, INK, 2))
    bits.extend([label(755, 250, 620, 410, "叶背褐色小点是许多孢子囊组成的孢子囊群"), text(390, 490, "孢子先长成很小的配子体，再经受精形成新的蕨", 20, INK, "middle", 650)])
    return "".join(bits)


def _q87() -> str:
    bits = [panel_title("竹子是禾本科草本，空心茎由一节一节的节间组成"), garden_backdrop()]
    for x in (210, 310, 410):
        bits.append(rect(x, 120, 52, 310, GREEN, INK, 4, 8))
        for y in (180, 260, 345):
            bits.append(rect(x - 8, y, 68, 14, GREEN_DARK, INK, 2, 5))
    bits.extend([path("M262,200 Q345,150 420,110", "none", GREEN_DARK, 8), leaf(420, 110, 0.28), rect(650, 150, 210, 200, CREAM, INK, 3, 12), circle(755, 250, 72, GREEN, INK, 4), circle(755, 250, 45, WHITE, INK, 3), label(755, 250, 620, 410, "许多竹秆节间中空，节部有横隔并承担连接"), label(320, 265, 650, 115, "地下茎能不断长出新竹笋")])
    return "".join(bits)


def _q88() -> str:
    bits = [panel_title("土豆是膨大的地下茎；“芽眼”能长出新的枝叶"), soil_cutaway(), sprout(300, 275, 0.82, True)]
    bits.extend([ellipse(390, 390, 82, 47, "#D2A064", INK, 4), ellipse(520, 410, 70, 42, "#D2A064", INK, 4)])
    for x, y in ((365, 380), (420, 405), (505, 398), (540, 420)):
        bits.append(circle(x, y, 7, BROWN_DARK))
    bits.extend([path("M300,330 Q360,340 390,365 M310,340 Q430,355 520,380", "none", GREEN_DARK, 7), label(390, 390, 650, 145, "块茎储存淀粉，是茎的一部分"), label(420, 405, 650, 280, "芽眼是缩短枝条上的芽"), label(305, 335, 650, 405, "连接块茎的是地下匍匐茎，不是主根")])
    return "".join(bits)


def _q89() -> str:
    bits = [panel_title("胡萝卜把叶片制造的糖储存在膨大的主根里"), soil_cutaway()]
    bits.extend([path("M360,250 Q270,300 335,445 Q365,495 395,445 Q460,300 360,250 Z", ORANGE, INK, 4)])
    for angle in (-70, -35, 0, 35, 70):
        x = 360 + math.sin(math.radians(angle)) * 20
        bits.append(path(f"M{x},255 Q{360+angle*2},160 {360+angle*2.3},110", "none", GREEN, 11))
    bits.extend([rect(650, 145, 210, 205, CREAM, INK, 3, 12), circle(755, 250, 72, ORANGE, INK, 4), circle(755, 250, 35, YELLOW, INK, 3), label(755, 250, 620, 410, "根横切面里，薄壁细胞储存糖和其他物质"), label(360, 340, 650, 115, "膨大的主根同时仍吸水、运输并固定植物")])
    return "".join(bits)


def _q90() -> str:
    bits = [panel_title("花园是一张交换物质和信息的网，不是植物单独生活"), garden_backdrop(), flower(250, 300, 0.65, PURPLE), bee(410, 200, 0.62), sprout(650, 340, 0.58, True)]
    bits.extend([
        path("M80,410 q70,-80 140,0 q70,-80 140,0", "none", BROWN, 18),
        path("M560,420 q40,-45 80,0 q40,-45 80,0", "none", CREAM, 9),
        arrow("M390,225 Q330,245 300,270", TEAL), arrow("M300,345 Q420,390 560,410", BROWN), arrow("M640,405 Q580,330 595,290", BLUE),
        text(430, 155, "传粉", 20, INK, "middle"), text(425, 430, "落叶和遗体成为分解者的食物", 20, INK, "middle"),
        label(650, 380, 650, 120, "菌根真菌可与根交换矿物和糖"),
        text(450, 495, "帮助、竞争、取食和分解同时存在，共同改变花园", 20, INK, "middle", 650),
    ])
    return "".join(bits)


RENDERERS = {
    60: _q60, 61: _q61, 62: _q62, 63: _q63, 64: _q64, 65: _q65,
    66: _q66, 67: _q67, 68: _q68, 69: _q69, 70: _q70, 71: _q71,
    72: _q72, 73: _q73, 74: _q74, 75: _q75, 76: _q76, 77: _q77,
    78: _q78, 79: _q79, 80: _q80, 81: _q81, 82: _q82, 83: _q83,
    84: _q84, 85: _q85, 86: _q86, 87: _q87, 88: _q88, 89: _q89,
    90: _q90,
}


def plant_body(question: int, palette: tuple[str, ...]) -> str:
    del palette
    try:
        return RENDERERS[question]()
    except KeyError as exc:
        raise KeyError(f"no plant illustration for question {question}") from exc
