#!/usr/bin/env python3
"""Mechanism-first plates for levers, circuits, structures, and robots."""

from __future__ import annotations

import math

from textbook_illustrations import INK, circle, ellipse, line, panel_title, path, rect, text


MACHINE_QUESTIONS = frozenset(range(182, 213))
MACHINE_KEYS = {question: (f"machine-plate-{question:03d}",) for question in MACHINE_QUESTIONS}

BLUE = "#4B82B3"
BLUE_DARK = "#2A5F89"
TEAL = "#2C8178"
RED = "#C95757"
ORANGE = "#DC713E"
YELLOW = "#E6B441"
GREEN = "#5B9159"
PURPLE = "#7769A4"
BROWN = "#896448"
WOOD = "#C69463"
METAL = "#AAB7BD"
CREAM = "#F7E9CA"
WHITE = "#FFFFFF"
SKY = "#DFEEF1"
WATER = "#A8D9E2"
TABLE = "#C99B70"


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


def arrow(data: str, color: str = INK, width: float = 4, dash: str = "") -> str:
    return path(data, "none", color, width, True, dash)


def workshop() -> str:
    return "".join([rect(34, 82, 892, 382, "#EFEAE1", "none", 0, 4), rect(34, 360, 892, 104, TABLE, "none"), line(34, 360, 926, 360, INK, 5)])


def gear(cx: float, cy: float, r: float, color: str, teeth: int = 12) -> str:
    bits = [circle(cx, cy, r, color, INK, 4), circle(cx, cy, r * .32, WHITE, INK, 3)]
    for index in range(teeth):
        angle = index * math.tau / teeth
        bits.append(line(cx + math.cos(angle) * r, cy + math.sin(angle) * r, cx + math.cos(angle) * (r + 17), cy + math.sin(angle) * (r + 17), INK, 9))
    return "".join(bits)


def wheel(cx: float, cy: float, r: float = 58) -> str:
    bits = [circle(cx, cy, r, "#E9EEF0", INK, 5), circle(cx, cy, 13, BLUE, INK, 3)]
    for index in range(8):
        angle = index * math.tau / 8
        bits.append(line(cx, cy, cx + math.cos(angle) * (r - 7), cy + math.sin(angle) * (r - 7), METAL, 3))
    return "".join(bits)


def battery(cx: float, cy: float, s: float = 1.0) -> str:
    return "".join([rect(cx - 70 * s, cy - 110 * s, 140 * s, 220 * s, BLUE, INK, 4 * s, 10 * s), rect(cx - 25 * s, cy - 130 * s, 50 * s, 20 * s, ORANGE, INK, 3 * s, 3 * s), text(cx, cy - 35 * s, "+", int(42 * s), WHITE, "middle", 750), text(cx, cy + 60 * s, "−", int(42 * s), WHITE, "middle", 750)])


def magnet(cx: float, cy: float, s: float = 1.0) -> str:
    return path(f"M{cx-90*s},{cy-90*s} V{cy+30*s} Q{cx-90*s},{cy+110*s} {cx},{cy+110*s} Q{cx+90*s},{cy+110*s} {cx+90*s},{cy+30*s} V{cy-90*s} H{cx+35*s} V{cy+28*s} Q{cx+35*s},{cy+55*s} {cx},{cy+55*s} Q{cx-35*s},{cy+55*s} {cx-35*s},{cy+28*s} V{cy-90*s} Z", RED, INK, 4 * s)


def lamp(cx: float, cy: float, s: float = 1.0, on: bool = True) -> str:
    bits = [circle(cx, cy - 25 * s, 50 * s, YELLOW if on else WHITE, INK, 4 * s), rect(cx - 28 * s, cy + 22 * s, 56 * s, 50 * s, METAL, INK, 3 * s, 5 * s)]
    if on:
        for index in range(8):
            angle = index * math.tau / 8
            bits.append(line(cx + math.cos(angle) * 66 * s, cy - 25 * s + math.sin(angle) * 66 * s, cx + math.cos(angle) * 92 * s, cy - 25 * s + math.sin(angle) * 92 * s, ORANGE, 4 * s))
    return "".join(bits)


def _q182() -> str:
    bits = [panel_title("机器用结构改变力的大小、方向或运动方式，不一定需要电"), workshop()]
    bits.extend([path("M100,300 L270,220", "none", BLUE, 14), polygon([(180, 300), (220, 245), (255, 300)], ORANGE, INK, 3), wheel(430, 290, 55), gear(650, 275, 58, BLUE), rect(760, 220, 90, 90, RED, INK, 3, 5)])
    bits.extend([text(190, 420, "杠杆", 20, INK, "middle"), text(430, 420, "轮轴", 20, INK, "middle"), text(650, 420, "齿轮", 20, INK, "middle"), text(805, 420, "组合成装置", 20, INK, "middle"), text(480, 495, "剪刀、开瓶器、自行车和电动机都是机器；复杂度不是判定标准", 19, INK, "middle", 720)])
    return "".join(bits)


def _q183() -> str:
    bits = [panel_title("支点离重物近、手柄长时，小手力可换成较大的抬举力"), workshop(), ellipse(270, 325, 105, 55, GRAY := "#79878C", INK, 4), path("M150,310 L570,210", "none", BLUE, 18), polygon([(315, 350), (355, 292), (395, 350)], ORANGE, INK, 3)]
    bits.extend([arrow("M545,130 L545,210", RED, 7), arrow("M250,295 L250,220", BLUE_DARK, 7), line(355, 295, 355, 350, INK, 5), label(355, 295, 620, 105, "支点"), label(250, 250, 620, 245, "重物端移动距离小、受力大"), label(545, 190, 620, 385, "手端移动更远，用较小力换取较大力")])
    return "".join(bits)


def _q184() -> str:
    bits = [panel_title("跷跷板是否平衡，取决于两边“重量 × 到支点距离”"), workshop(), path("M150,265 L810,330", "none", BLUE, 18), polygon([(430, 365), (480, 295), (530, 365)], ORANGE, INK, 4)]
    bits.extend([circle(270, 205, 42, RED, INK, 3), rect(235, 245, 70, 50, RED, INK, 3, 5), circle(700, 260, 30, YELLOW, INK, 3), rect(675, 295, 50, 38, YELLOW, INK, 3, 5), line(270, 350, 480, 350, TEAL, 4), line(480, 380, 700, 380, TEAL, 4), text(375, 340, "距离较长", 19, INK, "middle"), text(590, 420, "距离较短", 19, INK, "middle"), text(480, 490, "较轻的人坐得更远，也可能和较重的人平衡", 20, INK, "middle", 620)])
    return "".join(bits)


def _q185() -> str:
    bits = [panel_title("滑轮改变拉力方向；动滑轮让多段绳子共同承担重物"), workshop(), line(170, 115, 170, 205, INK, 8), circle(170, 240, 48, WHITE, INK, 5), path("M120,240 V400 M220,240 V400", "none", BLUE, 8), rect(75, 400, 90, 55, RED, INK, 3, 5)]
    bits.extend([line(530, 115, 530, 190, INK, 8), circle(530, 225, 45, WHITE, INK, 5), circle(650, 330, 45, WHITE, INK, 5), path("M485,225 V330 Q485,375 530,375 H650 Q695,375 695,330 V145", "none", BLUE, 8), rect(610, 380, 80, 60, RED, INK, 3, 5), arrow("M770,145 L770,260", RED, 6), text(170, 480, "定滑轮：主要改方向", 19, INK, "middle"), text(625, 480, "动滑轮：两段绳分担重量", 19, INK, "middle")])
    return "".join(bits)


def _q186() -> str:
    bits = [panel_title("轮子把大面积滑动摩擦变成轴承处较小的滚动与转动阻力"), workshop(), rect(90, 250, 220, 115, RED, INK, 4, 5), wheel(135, 395, 42), wheel(270, 395, 42), rect(580, 250, 220, 115, RED, INK, 4, 5)]
    bits.extend([path("M560,365 L820,365", "none", INK, 8), arrow("M75,305 L35,305", BLUE, 5), arrow("M565,305 L510,305", BLUE, 9), text(200, 470, "轮车：接触处滚动", 20, INK, "middle"), text(690, 470, "箱子：底面滑动", 20, INK, "middle"), text(480, 510, "轮轴也会变形和摩擦，所以车不会永远滚下去", 19, INK, "middle", 650)])
    return "".join(bits)


def _q187() -> str:
    bits = [panel_title("相邻齿轮的齿逐个接触传力，转向相反，转速由齿数比决定"), workshop(), gear(330, 275, 95, BLUE, 16), gear(580, 305, 62, ORANGE, 11)]
    bits.extend([arrow("M225,180 A130,130 0 0 1 330,135", BLUE_DARK, 5), arrow("M655,220 A95,95 0 0 1 675,320", RED, 5), label(455, 285, 650, 115, "接触点沿共同切线传递力"), label(330, 275, 650, 255, "大齿轮齿数多，同一时间转角较小"), label(580, 305, 650, 395, "小齿轮转得更快，但可用扭矩相应改变")])
    return "".join(bits)


def _q188() -> str:
    bits = [panel_title("链条把脚踏处的前齿盘转动传到后轮齿盘"), workshop(), wheel(230, 320, 92), wheel(650, 320, 92), gear(355, 300, 48, BLUE, 12), gear(590, 325, 30, ORANGE, 10)]
    bits.extend([path("M355,252 L590,295 Q620,300 620,325 Q620,350 590,355 L355,348 Q320,342 320,300 Q320,258 355,252 Z", "none", TEAL, 8), line(355, 300, 285, 180, INK, 7), line(285, 180, 405, 180, INK, 7), line(405, 180, 495, 320, INK, 7), line(355, 300, 230, 320, INK, 7), line(495, 320, 650, 320, INK, 7), line(285, 180, 230, 320, INK, 7), label(470, 270, 650, 110, "换挡改变前后齿盘的齿数比"), text(450, 495, "链条传力，不是链条把自行车向前拉；后轮推地才让车前进", 19, INK, "middle", 710)])
    return "".join(bits)


def _q189() -> str:
    bits = [panel_title("螺纹相当于绕在圆柱上的斜面，转很多圈换取较小的直线前进"), workshop(), rect(180, 145, 140, 275, METAL, INK, 4, 12)]
    for y in range(175, 410, 42):
        bits.append(path(f"M170,{y} L330,{y+35}", "none", BLUE_DARK, 8))
    bits.extend([polygon([(560, 380), (820, 380), (820, 160)], CREAM, INK, 4), path("M560,380 L820,160", "none", BLUE, 8), arrow("M250,120 A95,95 0 0 1 345,205", RED, 5), arrow("M250,420 L250,455", BLUE, 5), label(250, 260, 620, 105, "旋转使螺纹沿配合表面前进"), label(700, 270, 620, 405, "展开螺纹可看到一条长斜坡")])
    return "".join(bits)


def _q190() -> str:
    bits = [panel_title("刀刃是楔形：把向下的力分成推开材料两侧的力"), workshop(), circle(310, 315, 105, RED, INK, 4), polygon([(245, 315), (560, 210), (560, 315)], METAL, INK, 4)]
    bits.extend([arrow("M500,120 L500,210", RED, 7), arrow("M330,315 Q270,250 245,215", BLUE, 5), arrow("M330,315 Q270,375 245,410", BLUE, 5), label(360, 300, 620, 105, "尖端接触面积小，局部压力大"), label(275, 315, 620, 260, "斜面把材料推向两边，裂缝继续扩展"), text(420, 495, "更尖不总更安全或更耐用；刃角还要兼顾强度", 20, INK, "middle", 650)])
    return "".join(bits)


def _q191() -> str:
    bits = [panel_title("斜坡用更长的路换取较小的沿坡推力，理想功并没有减少"), workshop(), polygon([(110, 400), (610, 400), (610, 170)], CREAM, INK, 4), rect(315, 255, 100, 85, RED, INK, 4, 5)]
    bits.extend([arrow("M270,330 L420,260", BLUE, 7), arrow("M720,400 L720,190", RED, 8), line(650, 400, 790, 400, INK, 4), text(340, 445, "沿长斜面推", 20, INK, "middle"), text(720, 445, "直接抬", 20, INK, "middle"), label(410, 280, 620, 105, "同样抬高，斜坡方向所需力较小"), text(440, 500, "实际还要克服摩擦，消耗的总能量可能更多", 20, INK, "middle", 650)])
    return "".join(bits)


def _q192() -> str:
    bits = [panel_title("弹簧变形时储存弹性势能，回复力把它拉回原来长度"), workshop()]
    for start_x, compression in ((190, 1.0), (480, .55), (720, 1.35)):
        points = []
        turns = 7
        height = 200 * compression
        for index in range(turns * 2 + 1):
            x = start_x + (-42 if index % 2 else 42)
            y = 350 - height * index / (turns * 2)
            points.append((x, y))
        bits.append(path("M" + " L".join(f"{x},{y}" for x, y in points), "none", BLUE, 8))
    bits.extend([arrow("M480,110 L480,175", RED, 6), arrow("M720,135 L720,85", TEAL, 6), text(190, 430, "原长", 20, INK, "middle"), text(480, 430, "压缩", 20, INK, "middle"), text(720, 430, "拉伸", 20, INK, "middle"), text(480, 495, "超过弹性限度后可能不能完全回来，反复使用也会疲劳", 20, INK, "middle", 680)])
    return "".join(bits)


def _q193() -> str:
    bits = [panel_title("金属弹簧靠线材弯扭，橡皮筋靠长分子链重新蜷曲"), workshop(), path("M110,280 C155,170 200,390 245,280 S335,170 380,280", "none", RED, 20)]
    points = [(580 + (40 if i % 2 else -40), 380 - i * 25) for i in range(11)]
    bits.append(path("M" + " L".join(f"{x},{y}" for x, y in points), "none", BLUE, 9))
    bits.extend([rect(120, 130, 250, 85, CREAM, INK, 3, 12), path("M145,175 q25,-35 50,0 q25,35 50,0 q25,-35 50,0 q25,35 50,0", "none", PURPLE, 5), rect(650, 130, 210, 85, CREAM, INK, 3, 12), path("M680,175 H825", "none", BLUE_DARK, 8), text(245, 440, "橡皮筋：聚合物网络", 20, INK, "middle"), text(700, 440, "螺旋弹簧：金属线受扭", 20, INK, "middle")])
    return "".join(bits)


def _q194() -> str:
    bits = [panel_title("磁铁周围有磁场，铁磁材料中的磁畴会被重新排列"), workshop(), magnet(280, 255, .85)]
    for radius in (125, 165, 205):
        bits.append(path(f"M{280-radius},{250} Q280,{70-radius/4} {280+radius},{250} Q280,{430+radius/4} {280-radius},{250}", "none", BLUE, 3))
    for x, y, angle in ((650, 220, -20), (730, 270, 15), (805, 325, -10)):
        bits.append(f'<g transform="translate({x} {y}) rotate({angle})">{rect(-45,-8,90,16,METAL,INK,2,4)}</g>')
    bits.extend([label(280, 255, 620, 105, "磁场不需要接触就能对磁性材料施力"), label(730, 270, 620, 395, "普通铁钉靠近后会被暂时磁化并受到吸引")])
    return "".join(bits)


def _q195() -> str:
    bits = [panel_title("指南针磁针沿地球附近的磁场方向转动"), workshop(), circle(330, 275, 150, WHITE, INK, 5), polygon([(330, 125), (350, 275), (330, 245), (310, 275)], RED, INK, 3), polygon([(330, 425), (350, 275), (330, 305), (310, 275)], BLUE, INK, 3)]
    bits.extend([circle(700, 270, 95, BLUE, INK, 4), path("M650,210 Q700,175 750,210 M650,330 Q700,365 750,330", "none", GREEN, 6), path("M700,120 Q565,270 700,420 M700,120 Q835,270 700,420", "none", BLUE_DARK, 4), label(330, 150, 620, 105, "红端通常指向磁北附近，不是精确地理北极"), label(700, 270, 620, 405, "地核流动产生的地磁场会缓慢变化")])
    return "".join(bits)


def _q196() -> str:
    bits = [panel_title("电流通过绕在铁芯外的线圈，会产生可开关的电磁铁"), workshop(), rect(260, 205, 300, 80, METAL, INK, 4, 12)]
    for x in range(285, 550, 35):
        bits.append(path(f"M{x},180 q{-25},65 0,130 q25,-65 0,-130", "none", ORANGE, 7))
    bits.extend([battery(745, 265, .55), path("M285,180 Q285,120 745,120 V190 M545,310 Q545,420 745,420 V340", "none", BLUE_DARK, 6), rect(190, 360, 75, 18, METAL, INK, 2, 4), rect(290, 390, 75, 18, METAL, INK, 2, 4), label(420, 230, 620, 105, "线圈每一匝的磁场叠加"), label(260, 365, 620, 390, "铁芯磁化后吸引铁件；断电后磁性大幅减弱")])
    return "".join(bits)


def _q197() -> str:
    bits = [panel_title("闭合电路给电荷提供连续路径，灯丝或 LED 才能持续传递能量"), workshop(), battery(210, 270, .55), lamp(690, 245, .75, True), rect(420, 205, 90, 45, WHITE, INK, 3, 5)]
    bits.extend([circle(440, 227, 10, BLUE), circle(490, 227, 10, BLUE), path("M210,135 H690 V180 M210,405 H690 V315", "none", BLUE_DARK, 7), line(420, 227, 510, 227, GREEN, 7), label(465, 227, 620, 105, "开关闭合，整条回路连续"), label(690, 245, 620, 405, "能量从电源传到灯并转成光和热"), text(430, 495, "电流不是从电池里跑完就消失；电荷在整个回路中移动", 19, INK, "middle", 700)])
    return "".join(bits)


def _q198() -> str:
    bits = [panel_title("电池用化学反应维持两端电势差，不是把“电”装在壳里"), workshop(), rect(190, 125, 300, 295, BLUE, INK, 5, 12), rect(235, 145, 70, 250, RED, INK, 3, 6), rect(375, 145, 70, 250, METAL, INK, 3, 6), rect(305, 145, 70, 250, "#D8E9C5", INK, 3, 6)]
    bits.extend([text(270, 445, "负极", 20, INK, "middle"), text(340, 445, "电解质", 20, INK, "middle"), text(410, 445, "正极", 20, INK, "middle"), arrow("M305,210 L375,210", TEAL, 4), arrow("M375,320 L305,320", ORANGE, 4), label(340, 265, 620, 105, "离子在电解质中移动，电子走外部电路"), label(430, 180, 620, 285, "化学物质逐渐转变，能提供的电势差下降"), text(430, 495, "不同电池化学体系的材料、可充性和安全要求不同", 19, INK, "middle", 700)])
    return "".join(bits)


def _q199() -> str:
    bits = [panel_title("电动机中载流线圈与磁场相互作用，形成持续转矩"), workshop(), magnet(270, 255, .7), circle(500, 255, 95, WHITE, INK, 4), rect(455, 205, 90, 100, ORANGE, INK, 4, 8), line(500, 160, 500, 350, INK, 8), gear(720, 255, 60, BLUE, 12)]
    bits.extend([arrow("M420,170 A120,120 0 0 1 580,170", RED, 6), arrow("M580,340 A120,120 0 0 1 420,340", RED, 6), line(595, 255, 660, 255, INK, 10), label(500, 255, 620, 105, "换向或交变电流让转矩持续同一转向"), label(720, 255, 620, 405, "转轴把旋转传给齿轮、风扇或车轮")])
    return "".join(bits)


def _q200() -> str:
    bits = [panel_title("电动机把电能变成机械能，发电机把机械能变成电能"), workshop(), circle(250, 270, 95, WHITE, INK, 4), rect(215, 215, 70, 110, ORANGE, INK, 4, 8), circle(710, 270, 95, WHITE, INK, 4), rect(675, 215, 70, 110, BLUE, INK, 4, 8)]
    bits.extend([battery(80, 270, .35), arrow("M130,270 L155,270", TEAL, 5), arrow("M345,270 L420,270", RED, 5), gear(455, 270, 36, RED, 10), gear(545, 270, 36, BLUE, 10), arrow("M585,270 L615,270", BLUE, 5), lamp(870, 270, .42, True), arrow("M805,270 L830,270", TEAL, 5), text(250, 430, "电流进，转动出", 20, INK, "middle"), text(710, 430, "转动进，电流出", 20, INK, "middle"), text(480, 495, "两者核心都利用电流、磁场与运动之间的关系", 20, INK, "middle", 650)])
    return "".join(bits)


def _q201() -> str:
    bits = [panel_title("风扇让空气流动；人体觉得凉主要因为对流和汗液蒸发加快"), workshop(), circle(330, 255, 130, WHITE, INK, 5), circle(330, 255, 24, BLUE, INK, 3)]
    for angle in (0, math.tau / 3, 2 * math.tau / 3):
        x1, y1 = 330 + math.cos(angle) * 20, 255 + math.sin(angle) * 20
        bits.append(path(f"M{x1},{y1} Q{330+math.cos(angle+.55)*120:.1f},{255+math.sin(angle+.55)*120:.1f} {330+math.cos(angle+1.2)*95:.1f},{255+math.sin(angle+1.2)*95:.1f} Q{330+math.cos(angle+.5)*55:.1f},{255+math.sin(angle+.5)*55:.1f} {x1},{y1} Z", BLUE, INK, 3))
    bits.extend([arrow("M500,190 Q620,180 745,190", TEAL, 5), arrow("M500,255 Q620,245 745,255", TEAL, 5), arrow("M500,320 Q620,310 745,320", TEAL, 5), circle(815, 255, 45, "#E7B39E", INK, 3), circle(795, 315, 9, BLUE), label(815, 255, 620, 110, "风扇不主动降低整个房间空气温度"), text(430, 490, "关在无人房间的电风扇反而会把电能最终变成少量热", 20, INK, "middle", 650)])
    return "".join(bits)


def _q202() -> str:
    bits = [panel_title("水泵让叶轮或活塞对水做功，建立压力差并推动连续水流"), workshop(), rect(120, 210, 320, 190, WATER, INK, 4, 12), circle(280, 305, 78, WHITE, INK, 4)]
    for index in range(6):
        angle = index * math.tau / 6
        bits.append(path(f"M280,305 Q{280+math.cos(angle+.5)*65:.1f},{305+math.sin(angle+.5)*65:.1f} {280+math.cos(angle)*68:.1f},{305+math.sin(angle)*68:.1f}", "none", BLUE, 14))
    bits.extend([path("M440,305 H570 V130 H830", "none", BLUE, 28), arrow("M500,305 L555,305", WHITE, 4), arrow("M570,230 L570,165", WHITE, 4), rect(760, 90, 90, 70, WATER, INK, 3, 5), label(280, 305, 620, 105, "叶轮提高水的速度和压力"), label(570, 210, 620, 405, "管道约束水流，把压力传到较高处")])
    return "".join(bits)


def _q203() -> str:
    bits = [panel_title("水龙头转动阀芯，改变水能通过的开口面积"), workshop(), path("M160,315 H430 V205 H650 Q710,205 710,265 V315", "none", METAL, 42), path("M160,315 H430 V205 H650 Q710,205 710,265 V315", "none", WATER, 16), rect(360, 170, 140, 70, BLUE, INK, 4, 8), line(430, 170, 430, 120, INK, 8), rect(350, 95, 160, 35, RED, INK, 3, 15)]
    bits.extend([circle(430, 205, 28, CREAM, INK, 3), line(405, 205, 455, 205, RED, 8), arrow("M710,315 L710,395", BLUE, 5), label(430, 205, 620, 105, "阀芯开口与管道对齐时水可通过"), label(430, 120, 620, 250, "手柄通过螺纹、陶瓷片或球阀控制开度"), text(430, 495, "关上不是把水“消失”，而是阻断压差驱动的流路", 20, INK, "middle", 660)])
    return "".join(bits)


def _q204() -> str:
    bits = [panel_title("剪刀把两个一级杠杆和两片楔形刀刃装在同一转轴上"), workshop(), circle(250, 315, 70, "none", BLUE, 18), circle(390, 315, 70, "none", RED, 18), circle(320, 260, 18, METAL, INK, 4), polygon([(320, 250), (760, 135), (390, 275)], METAL, INK, 4), polygon([(320, 270), (760, 390), (390, 245)], METAL, INK, 4)]
    bits.extend([label(320, 260, 620, 105, "铆钉是共同支点"), label(245, 315, 620, 250, "长把手放大夹持力"), label(650, 180, 620, 405, "尖薄刀刃把材料推开并扩展裂口")])
    return "".join(bits)


def _q205() -> str:
    bits = [panel_title("拉链头把两排互相错开的链牙导入狭窄通道并扣合"), workshop()]
    for index in range(8):
        y = 130 + index * 38
        bits.extend([rect(240, y, 80, 25, RED, INK, 2, 5), rect(500, y + 18, 80, 25, BLUE, INK, 2, 5)])
    bits.extend([path("M320,150 Q420,220 500,165 L500,420 Q420,360 320,405 Z", CREAM, INK, 4), rect(380, 235, 65, 100, METAL, INK, 4, 12), arrow("M412,190 L412,235", TEAL, 5), label(410, 285, 620, 105, "滑块内部的 Y 形通道改变两排链牙间距"), label(500, 330, 620, 310, "链牙凸起进入相邻凹槽，互相限制侧向移动"), text(430, 495, "反向拉动时通道把链牙分开；不是滑块把布粘住", 20, INK, "middle", 660)])
    return "".join(bits)


def _q206() -> str:
    bits = [panel_title("合适钥匙把每一组锁芯弹子顶到同一剪切线，锁芯才能转动"), workshop(), circle(330, 275, 155, METAL, INK, 5), circle(330, 275, 105, CREAM, INK, 4)]
    for index, height in enumerate((55, 85, 42, 72, 60)):
        x = 250 + index * 40
        bits.extend([rect(x, 150, 22, height, BLUE, INK, 2, 4), rect(x, 150 + height, 22, 225 - height, RED, INK, 2, 4)])
    bits.extend([path("M170,330 H520 L560,300 L530,280 L560,260 L520,235 H170 Z", YELLOW, INK, 4), line(180, 250, 500, 250, TEAL, 4, False, "8 6"), label(350, 250, 620, 105, "正确钥匙让各接缝正好落在剪切线"), label(260, 180, 620, 280, "高度不对时弹子跨过边界，阻止锁芯旋转")])
    return "".join(bits)


def _q207() -> str:
    bits = [panel_title("门绕铰链轴转动；手把离轴越远，同样推力产生的转矩越大"), workshop(), rect(250, 110, 280, 300, BROWN, INK, 5, 4), line(250, 125, 250, 395, METAL, 14), circle(480, 260, 15, YELLOW, INK, 3)]
    bits.extend([path("M250,110 L650,190 L650,430 L250,410 Z", "none", BLUE, 5), arrow("M480,260 Q585,255 625,320", RED, 6), line(250, 260, 480, 260, TEAL, 4), label(250, 260, 620, 105, "铰链确定转轴并承受门的重量"), label(480, 260, 620, 280, "门把远离转轴，较省力"), text(420, 495, "在靠近铰链处推门，需要更大的力才能得到同样转动效果", 19, INK, "middle", 700)])
    return "".join(bits)


def _q208() -> str:
    bits = [panel_title("三角桁架把桥面荷载分成杆件中的拉力和压力"), rect(34, 82, 892, 382, SKY, "none", 0, 4), rect(80, 350, 800, 35, METAL, INK, 4, 4)]
    nodes = [(100 + i * 120, 350 if i % 2 == 0 else 190) for i in range(7)]
    for (x1, y1), (x2, y2) in zip(nodes, nodes[1:]):
        bits.extend([line(x1, y1, x2, y2, BLUE if len(bits) % 2 else RED, 14), circle(x1, y1, 9, INK)])
    for index in range(0, len(nodes) - 2, 2):
        bits.append(line(nodes[index][0], nodes[index][1], nodes[index + 2][0], nodes[index + 2][1], TEAL, 12))
    bits.extend([arrow("M480,105 L480,185", RED, 7), label(480, 190, 620, 105, "节点受力后，三角形不容易改变形状"), label(360, 280, 620, 305, "有些杆受拉、有些杆受压，共同把力送向桥墩"), text(430, 490, "工程师还要检查连接、材料屈曲、振动和疲劳", 20, INK, "middle", 650)])
    return "".join(bits)


def _q209() -> str:
    bits = [panel_title("拱把桥面重量沿弧形变成压力，并推向两侧桥墩和地基"), rect(34, 82, 892, 382, SKY, "none", 0, 4), path("M120,390 Q480,70 840,390", "none", METAL, 70), path("M120,390 Q480,135 840,390", "none", SKY, 40), rect(70, 370, 120, 70, BROWN, INK, 4, 4), rect(770, 370, 120, 70, BROWN, INK, 4, 4)]
    for t in (0.18, .33, .5, .67, .82):
        x = 120 + 720 * t
        y = 390 - 255 * (1 - ((t - .5) / .5) ** 2)
        bits.append(arrow(f"M{x},{y-65} L{x},{y-15}", RED, 5))
    bits.extend([arrow("M160,385 L80,420", BLUE, 6), arrow("M800,385 L880,420", BLUE, 6), label(480, 150, 620, 105, "石块互相挤压，把力沿拱圈传递"), label(800, 390, 620, 380, "桥台必须抵抗向外推力")])
    return "".join(bits)


def _q210() -> str:
    bits = [panel_title("起重机用桁架臂跨出距离，用滑轮组省力，并用配重防止倾覆"), rect(34, 82, 892, 382, SKY, "none", 0, 4), rect(170, 360, 160, 70, BLUE, INK, 4, 5), line(250, 360, 250, 110, INK, 18), path("M250,120 L760,180", "none", BLUE, 18), path("M250,120 L110,230", "none", BLUE, 18), rect(80, 225, 120, 95, RED, INK, 4, 5)]
    for x in range(300, 730, 75):
        bits.extend([line(x, 125 + (x-250)*.12, x+40, 180 + (x-250)*.12, TEAL, 5), line(x+40, 180 + (x-250)*.12, x+75, 130 + (x-250)*.12, TEAL, 5)])
    bits.extend([circle(720, 175, 32, WHITE, INK, 4), path("M720,205 V365", "none", INK, 7), rect(675, 365, 90, 60, ORANGE, INK, 3, 5), label(140, 260, 620, 105, "后方配重平衡长臂和载荷产生的转矩"), label(720, 175, 620, 285, "滑轮组改变绳路并分担拉力"), text(430, 495, "额定载荷随吊臂角度和伸出距离变化，不能只看物体重量", 19, INK, "middle", 700)])
    return "".join(bits)


def _q211() -> str:
    bits = [panel_title("电梯控制器结合楼层传感器、曳引机和制动器准确停车"), rect(34, 82, 892, 382, "#E8ECEC", "none", 0, 4), rect(180, 110, 360, 320, "none", INK, 5, 4), rect(230, 210, 150, 170, METAL, INK, 4, 5), circle(470, 155, 50, WHITE, INK, 4), path("M305,210 V155 H470 Q520,155 520,205 V375", "none", INK, 8), rect(475, 340, 90, 55, BLUE, INK, 3, 5)]
    for y in (150, 250, 350):
        bits.extend([circle(590, y, 10, RED), line(540, y, 590, y, TEAL, 3, False, "6 6")])
    bits.extend([rect(680, 145, 150, 210, CREAM, INK, 3, 12), circle(755, 195, 22, GREEN, INK, 2), circle(755, 255, 22, YELLOW, INK, 2), circle(755, 315, 22, RED, INK, 2), label(590, 250, 620, 105, "位置传感器告诉控制器轿厢经过哪里"), label(470, 155, 620, 395, "电机减速接近楼层，制动器最后保持静止")])
    return "".join(bits)


def _q212() -> str:
    bits = [panel_title("机器人用传感器取得信号，控制器判断，再驱动电机执行动作"), workshop(), rect(300, 175, 220, 180, BLUE, INK, 4, 15), circle(410, 135, 70, METAL, INK, 4), circle(385, 125, 14, WHITE, INK, 2), circle(435, 125, 14, WHITE, INK, 2), rect(350, 355, 35, 90, INK, INK, 3, 8), rect(435, 355, 35, 90, INK, INK, 3, 8), line(300, 225, 210, 315, INK, 15), line(520, 225, 610, 315, INK, 15)]
    bits.extend([path("M360,265 H460", "none", TEAL, 8), circle(360, 265, 10, YELLOW), circle(410, 265, 10, YELLOW), circle(460, 265, 10, YELLOW), rect(680, 145, 180, 210, CREAM, INK, 3, 12), arrow("M710,205 L815,205", BLUE, 4), arrow("M815,255 L710,255", RED, 4), arrow("M710,305 L815,305", TEAL, 4), text(760, 390, "感知—计算—动作", 19, INK, "middle"), label(410, 135, 620, 105, "摄像头、测距器和触觉器件把环境变成信号"), text(420, 500, "传感器并不“理解”世界；程序或学习模型解释测量结果", 19, INK, "middle", 700)])
    return "".join(bits)


RENDERERS = {
    182: _q182, 183: _q183, 184: _q184, 185: _q185, 186: _q186, 187: _q187,
    188: _q188, 189: _q189, 190: _q190, 191: _q191, 192: _q192, 193: _q193,
    194: _q194, 195: _q195, 196: _q196, 197: _q197, 198: _q198, 199: _q199,
    200: _q200, 201: _q201, 202: _q202, 203: _q203, 204: _q204, 205: _q205,
    206: _q206, 207: _q207, 208: _q208, 209: _q209, 210: _q210, 211: _q211,
    212: _q212,
}


def machine_body(question: int, palette: tuple[str, ...]) -> str:
    del palette
    try:
        return RENDERERS[question]()
    except KeyError as exc:
        raise KeyError(f"no machine illustration for question {question}") from exc
