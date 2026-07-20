#!/usr/bin/env python3
"""Observable sky, weather, air, heat, and water plates for questions 1-59."""

from __future__ import annotations

import math

from textbook_illustrations import INK, circle, ellipse, line, panel_title, path, rect, text


EARTH_WEATHER_QUESTIONS = frozenset(range(1, 60))
EARTH_WEATHER_KEYS = {question: (f"earth-weather-plate-{question:03d}",) for question in EARTH_WEATHER_QUESTIONS}

BLUE = "#4B86B7"
BLUE_DARK = "#295E88"
SKY = "#DCEEF4"
WATER = "#A9D9E2"
TEAL = "#2D8178"
GREEN = "#5A9258"
GREEN_LIGHT = "#A6C982"
YELLOW = "#E8B642"
ORANGE = "#DE753E"
RED = "#C85555"
PURPLE = "#7667A5"
BROWN = "#8A654A"
BROWN_DARK = "#584338"
SOIL = "#B88762"
CREAM = "#F7E9CA"
WHITE = "#FFFFFF"
GRAY = "#75848C"
DARK = "#273552"


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


def landscape(night: bool = False, water: bool = False) -> str:
    sky = DARK if night else SKY
    bits = [rect(34, 82, 892, 382, sky, "none", 0, 4)]
    if night:
        for index in range(24):
            bits.append(circle(55 + (index * 79) % 850, 100 + (index * 53) % 230, 2 + index % 3, WHITE if index % 2 else YELLOW))
    if water:
        bits.extend([
            path("M34,300 Q180,265 325,300 T620,300 T930,300 V464 H34 Z", WATER),
            path("M34,330 Q170,295 320,330 T620,330 T940,330", "none", WHITE, 5),
        ])
    else:
        bits.extend([
            path("M34,350 Q175,285 325,350 T635,350 T945,350 V464 H34 Z", GREEN_LIGHT),
            path("M34,425 Q190,380 355,425 T690,425 T1030,425 V464 H34 Z", GREEN),
        ])
    return "".join(bits)


def sun(cx: float, cy: float, r: float = 55, rays: bool = True) -> str:
    bits = [circle(cx, cy, r, YELLOW, ORANGE, 4)]
    if rays:
        for index in range(12):
            angle = index * math.tau / 12
            bits.append(line(cx + math.cos(angle) * (r + 10), cy + math.sin(angle) * (r + 10), cx + math.cos(angle) * (r + 38), cy + math.sin(angle) * (r + 38), ORANGE, 4))
    return "".join(bits)


def cloud(cx: float, cy: float, s: float = 1.0, dark: bool = False) -> str:
    color = "#75838C" if dark else WHITE
    return "".join([
        circle(cx - 65 * s, cy + 12 * s, 45 * s, color, INK, 3 * s),
        circle(cx - 15 * s, cy - 25 * s, 62 * s, color, INK, 3 * s),
        circle(cx + 50 * s, cy - 5 * s, 52 * s, color, INK, 3 * s),
        rect(cx - 108 * s, cy + 8 * s, 205 * s, 62 * s, color, INK, 3 * s, 28 * s),
    ])


def drop(cx: float, cy: float, s: float = 1.0, color: str = BLUE) -> str:
    return path(f"M{cx},{cy-65*s} C{cx-22*s},{cy-25*s} {cx-48*s},{cy+10*s} {cx-48*s},{cy+38*s} C{cx-48*s},{cy+98*s} {cx+48*s},{cy+98*s} {cx+48*s},{cy+38*s} C{cx+48*s},{cy+10*s} {cx+22*s},{cy-25*s} {cx},{cy-65*s} Z", color, INK, 3 * s)


def earth(cx: float, cy: float, r: float = 85, half_dark: bool = False) -> str:
    bits = [circle(cx, cy, r, BLUE, INK, 4)]
    bits.extend([
        path(f"M{cx-58},{cy-35} q35,-42 72,-22 q8,28 -14,45 q-32,18 -55,5 Z", GREEN, INK, 2),
        path(f"M{cx+10},{cy+18} q45,-20 62,12 q-12,48 -60,47 q-27,-18 -2,-59 Z", GREEN, INK, 2),
    ])
    if half_dark:
        bits.append(path(f"M{cx},{cy-r} A{r},{r} 0 0 1 {cx},{cy+r} A{r},{r} 0 0 0 {cx},{cy-r} Z", DARK, "none"))
    return "".join(bits)


def moon(cx: float, cy: float, r: float = 58, phase: float = 1.0) -> str:
    bits = [circle(cx, cy, r, CREAM, INK, 3)]
    if phase < 1:
        cover_x = cx - r * (1 - phase) * 1.2
        bits.append(ellipse(cover_x, cy, r * 0.92, r, DARK, "none"))
    bits.extend([circle(cx - r * .25, cy - r * .2, r * .12, "#D5C8AD"), circle(cx + r * .22, cy + r * .18, r * .09, "#D5C8AD")])
    return "".join(bits)


def person(cx: float, cy: float, s: float = 1.0, sleeping: bool = False) -> str:
    if sleeping:
        return "".join([
            rect(cx - 120 * s, cy - 20 * s, 250 * s, 90 * s, "#8AA6C4", INK, 3 * s, 10 * s),
            circle(cx - 60 * s, cy - 28 * s, 28 * s, "#E7B39E", INK, 3 * s),
            path(f"M{cx-28*s},{cy-10*s} Q{cx+45*s},{cy-48*s} {cx+105*s},{cy+16*s} V{cy+58*s} H{cx-20*s} Z", PURPLE, INK, 3 * s),
            path(f"M{cx-72*s},{cy-30*s} q12,8 24,0", "none", INK, 2 * s),
        ])
    return "".join([
        circle(cx, cy - 95 * s, 30 * s, "#E7B39E", INK, 3 * s),
        line(cx, cy - 62 * s, cx, cy + 45 * s, DARK, 12 * s),
        line(cx, cy - 20 * s, cx - 62 * s, cy + 15 * s, DARK, 9 * s), line(cx, cy - 20 * s, cx + 62 * s, cy + 15 * s, DARK, 9 * s),
        line(cx, cy + 42 * s, cx - 48 * s, cy + 115 * s, DARK, 10 * s), line(cx, cy + 42 * s, cx + 48 * s, cy + 115 * s, DARK, 10 * s),
    ])


def leaf(cx: float, cy: float, s: float = 1.0) -> str:
    return "".join([path(f"M{cx-90*s},{cy+15*s} Q{cx-20*s},{cy-70*s} {cx+90*s},{cy-10*s} Q{cx+20*s},{cy+78*s} {cx-90*s},{cy+15*s} Z", GREEN, INK, 3 * s), line(cx - 75 * s, cy + 10 * s, cx + 70 * s, cy - 7 * s, GREEN_LIGHT, 4 * s)])


def house(cx: float, cy: float, s: float = 1.0) -> str:
    return "".join([rect(cx - 80 * s, cy - 50 * s, 160 * s, 125 * s, CREAM, INK, 4 * s, 4 * s), polygon([(cx - 100 * s, cy - 50 * s), (cx, cy - 135 * s), (cx + 100 * s, cy - 50 * s)], RED, INK, 4 * s), rect(cx - 22 * s, cy + 5 * s, 44 * s, 70 * s, BROWN, INK, 3 * s, 3 * s)])


def _q1() -> str:
    bits = [panel_title("太阳核心的核聚变释放能量，经过很久才从表面辐射出来"), rect(34, 82, 892, 382, DARK, "none", 0, 4), sun(350, 275, 155, False)]
    bits.extend([circle(350, 275, 88, ORANGE, RED, 4), circle(350, 275, 42, WHITE, ORANGE, 4)])
    for index in range(10):
        angle = index * math.tau / 10
        bits.append(arrow(f"M{350+math.cos(angle)*48:.1f},{275+math.sin(angle)*48:.1f} L{350+math.cos(angle)*135:.1f},{275+math.sin(angle)*135:.1f}", YELLOW, 3))
    bits.extend([label(350, 275, 650, 125, "核心：氢原子核聚变成氦，少量质量转为能量"), label(430, 245, 650, 275, "能量在内部反复传递"), label(490, 275, 650, 405, "表面发出可见光、红外线等电磁辐射")])
    return "".join(bits)


def _q2() -> str:
    bits = [panel_title("空气分子更容易把蓝紫色短波光散向四周"), landscape(), sun(145, 145, 42)]
    bits.extend([line(190, 165, 650, 275, ORANGE, 7)])
    for x, y in ((350, 205), (470, 235), (590, 265)):
        bits.extend([circle(x, y, 8, BLUE), arrow(f"M{x},{y} Q{x-20},{y-70} {x-55},{y-105}", BLUE, 3), arrow(f"M{x},{y} Q{x+20},{y+70} {x+55},{y+105}", BLUE, 3)])
    bits.extend([person(760, 325, 0.55), label(470, 235, 650, 120, "蓝光被许多方向散射，天空各处都把蓝光送进眼睛"), text(420, 490, "天空不是蓝色墙面；向太空看，背景仍是黑的", 20, INK, "middle", 620)])
    return "".join(bits)


def _q3() -> str:
    bits = [panel_title("云顶受光会亮，厚云底部的光被多次散射和吸收后变暗"), landscape()]
    bits.extend([sun(135, 135, 38), cloud(360, 230, 1.0), cloud(700, 245, 1.15, True), line(180, 160, 310, 205, YELLOW, 7), line(190, 180, 630, 225, YELLOW, 5)])
    for x, y in ((650, 215), (700, 245), (750, 230), (680, 285), (735, 290)):
        bits.append(circle(x, y, 8, BLUE_DARK))
    bits.extend([label(360, 205, 650, 100, "较薄云层把各色可见光一起散开，看起来白"), label(700, 250, 650, 390, "云很厚或背光时，到达底部和眼睛的光更少")])
    return "".join(bits)


def _q4() -> str:
    bits = [panel_title("太阳接近地平线时，光要穿过更长的大气路程"), rect(34, 82, 892, 382, "#F3B078", "none", 0, 4), path("M34,360 Q180,320 330,360 T640,360 T950,360 V464 H34 Z", "#596D65"), sun(150, 330, 42)]
    bits.extend([line(195, 325, 750, 250, ORANGE, 8), person(805, 330, 0.5)])
    for x in range(300, 700, 85):
        bits.extend([circle(x, 300 - (x - 300) * .13, 7, BLUE), arrow(f"M{x},{300-(x-300)*.13:.1f} q0,-55 -35,-80", BLUE, 2.5)])
    bits.extend([label(500, 280, 620, 115, "蓝光更多地被散到别处"), label(740, 250, 620, 270, "剩下较多红橙光沿原方向到达眼睛"), text(420, 490, "空气里的尘埃和水滴也会改变晚霞颜色与亮度", 20, INK, "middle", 650)])
    return "".join(bits)


def _q5() -> str:
    return "".join([panel_title("不透明物体挡住沿直线传播的光，后方形成光较少的区域"), rect(34, 82, 892, 382, "#EFE9DE", "none", 0, 4), circle(155, 235, 48, YELLOW, INK, 4), path("M205,190 L430,150 L430,345 L205,280 Z", YELLOW, "none").replace('/>', ' opacity="0.30"/>'), rect(430, 190, 95, 165, RED, INK, 4, 5), path("M525,325 Q700,390 865,340 Q700,305 525,285 Z", DARK, "none").replace('/>', ' opacity="0.55"/>'), label(430, 250, 650, 125, "物体挡住直达光"), label(650, 330, 650, 390, "边缘仍会收到周围反射光，所以影子并非绝对黑")])


def _q6() -> str:
    bits = [panel_title("同一个物体的影长取决于光线照来的高度"), landscape()]
    for index, (sx, sy, px, length) in enumerate(((150, 125, 245, 150), (475, 105, 475, 70), (790, 180, 710, 115))):
        bits.extend([sun(sx, sy, 28), person(px, 350, 0.45), path(f"M{px},{425} L{px+length},{425} L{px+20},{405} Z", DARK).replace('/>', ' opacity="0.45"/>'), line(sx, sy, px, 260, YELLOW, 4)])
    bits.extend([text(245, 470, "太阳低：影子长", 19, INK, "middle"), text(475, 470, "太阳高：影子短", 19, INK, "middle"), text(710, 470, "下午方向又改变", 19, INK, "middle")])
    return "".join(bits)


def _q7() -> str:
    bits = [panel_title("镜面把来自脸的光有规则地反射进眼睛"), rect(34, 82, 892, 382, "#ECE8DF", "none", 0, 4), person(260, 315, 0.62), rect(475, 110, 30, 300, "#C7D7E0", INK, 4, 3), person(720, 315, 0.62)]
    bits.extend([arrow("M280,230 L490,190", ORANGE, 5), arrow("M490,190 L300,205", BLUE, 5), line(505, 110, 505, 410, WHITE, 3), label(490, 190, 620, 105, "入射角和反射角相等"), text(480, 490, "镜中人不是藏在玻璃后面，而是反射光形成的虚像", 20, INK, "middle", 650)])
    return "".join(bits)


def _q8() -> str:
    bits = [panel_title("透明玻璃让大部分可见光穿过，也反射一小部分"), landscape(), rect(260, 105, 440, 285, "#D6EEF2", INK, 5, 4), line(480, 105, 480, 390, INK, 4), line(260, 247, 700, 247, INK, 4), sun(365, 170, 32), path("M275,340 Q380,245 470,340 Q565,255 690,340", GREEN, INK, 3)]
    bits.extend([arrow("M120,220 L285,220", ORANGE, 5), arrow("M285,220 L650,220", ORANGE, 5), arrow("M285,220 L170,145", BLUE, 4), label(285, 220, 630, 120, "多数光穿过玻璃和室内空气"), label(225, 185, 630, 270, "少量光被表面反射，所以夜里窗上会见到室内倒影"), text(470, 485, "透明不等于对所有光都透明，厚度、颜色和波长都会影响", 19, INK, "middle", 690)])
    return "".join(bits)


def _q9() -> str:
    bits = [panel_title("阳光进入雨滴时折射并分色，内部反射后再折射出来"), landscape(), drop(340, 270, 1.45, "#CBE9F0")]
    colors = (RED, ORANGE, YELLOW, GREEN, BLUE, PURPLE)
    bits.append(line(90, 205, 295, 230, WHITE, 8))
    for index, color in enumerate(colors):
        bits.append(path(f"M300,{230+index*3} Q395,{255+index*5} 360,{345+index*2} Q490,{365-index*2} 610,{300+index*8}", "none", color, 5))
    bits.extend([person(790, 330, 0.5), label(360, 345, 630, 115, "不同颜色弯曲程度稍有不同"), label(610, 320, 630, 270, "许多雨滴把特定方向的颜色送进眼睛"), text(440, 490, "看到彩虹时，太阳通常在观察者身后，雨滴在前方", 20, INK, "middle", 650)])
    return "".join(bits)


def _q10() -> str:
    bits = [panel_title("地球背向太阳的一半收不到直射阳光，于是进入夜晚"), rect(34, 82, 892, 382, DARK, "none", 0, 4), sun(160, 270, 58), earth(650, 270, 125, True), path("M220,210 L525,160 L525,380 L220,330 Z", YELLOW, "none").replace('/>', ' opacity="0.23"/>')]
    bits.extend([label(590, 270, 610, 115, "向着太阳的一面是白天"), label(710, 270, 610, 410, "背向太阳的一面是夜晚"), text(430, 495, "夜晚变黑不是太阳熄灭，而是所在地点转进背光面", 20, WHITE, "middle", 650)])
    return "".join(bits)


def _q11() -> str:
    bits = [panel_title("月亮不靠自身发可见光；我们看到的是它反射的太阳光"), rect(34, 82, 892, 382, DARK, "none", 0, 4), sun(150, 260, 50), moon(650, 260, 90)]
    bits.extend([arrow("M210,230 L555,230", YELLOW, 6), arrow("M590,290 Q470,360 330,320", CREAM, 5), label(600, 230, 620, 105, "太阳照亮月面"), label(475, 345, 620, 390, "月面把一小部分光反射到地球和眼睛")])
    return "".join(bits)


def _q12() -> str:
    bits = [panel_title("月相来自我们看见的受光半球比例变化，不是月亮真的变形"), rect(34, 82, 892, 382, DARK, "none", 0, 4)]
    phases = (0.05, 0.45, 1.0, 0.45)
    for index, phase in enumerate(phases):
        x = 150 + index * 220
        bits.append(moon(x, 260, 62, phase))
    bits.extend([text(150, 390, "新月附近", 19, WHITE, "middle"), text(370, 390, "上弦附近", 19, WHITE, "middle"), text(590, 390, "满月附近", 19, WHITE, "middle"), text(810, 390, "下弦附近", 19, WHITE, "middle"), text(480, 480, "完整月球始终存在，改变的是太阳、地球和月球的相对位置", 20, WHITE, "middle", 700)])
    return "".join(bits)


def _q13() -> str:
    bits = [panel_title("星光穿过不停流动的空气层，方向和亮度会快速微变"), rect(34, 82, 892, 382, DARK, "none", 0, 4), circle(170, 150, 13, YELLOW)]
    bits.extend([path("M175,160 C240,185 205,220 290,245 S355,305 420,300 S505,345 560,330", "none", YELLOW, 5), path("M560,330 Q655,340 730,355", "none", CREAM, 5), person(810, 350, 0.45)])
    for x, y in ((250, 205), (360, 275), (500, 325)):
        bits.append(path(f"M{x-45},{y} q45,-28 90,0", "none", TEAL, 4))
    bits.extend([label(360, 275, 620, 115, "温度不同的空气团密度稍有差别，光路跟着弯动"), text(420, 490, "行星也会受大气影响，但视面较大，通常比恒星闪烁少", 20, WHITE, "middle", 650)])
    return "".join(bits)


def _q14() -> str:
    bits = [panel_title("白天星星仍在，只是散射的阳光让天空背景太亮"), rect(34, 82, 446, 382, DARK, "none", 0, 4), rect(480, 82, 446, 382, SKY, "none", 0, 4)]
    for x, y in ((95, 130), (180, 210), (300, 145), (410, 270), (550, 130), (650, 230), (790, 165), (860, 300)):
        bits.append(circle(x, y, 5, WHITE if x < 480 else "#C8D5D9"))
    bits.extend([moon(320, 330, 38), sun(650, 330, 44), text(250, 425, "夜晚：背景暗，星光容易分辨", 20, WHITE, "middle"), text(705, 425, "白天：天空散射光淹没微弱星光", 20, INK, "middle")])
    return "".join(bits)


def _q15() -> str:
    bits = [panel_title("地球大约一天自转一周，各地依次进入向光面和背光面"), rect(34, 82, 892, 382, DARK, "none", 0, 4), sun(150, 270, 55), earth(610, 270, 120, True)]
    bits.extend([path("M610,115 A155,155 0 1 1 480,185", "none", BLUE, 7, True), circle(545, 230, 8, RED), label(545, 230, 620, 105, "地点随地球自转进入白天"), label(670, 300, 620, 410, "继续转到背光面后进入夜晚")])
    return "".join(bits)


def _q16() -> str:
    bits = [panel_title("地球由西向东自转，所以太阳看起来从东方升起"), landscape(), sun(160, 305, 45), earth(700, 230, 92)]
    bits.extend([arrow("M735,130 A120,120 0 0 1 790,245", BLUE, 6), person(420, 340, 0.48), arrow("M420,250 Q300,245 205,290", ORANGE, 4), text(160, 390, "东方地平线", 20, INK, "middle"), label(700, 150, 610, 105, "真实运动是地球自西向东转"), text(450, 490, "我们随地面一起转，便看见太阳相对天空向西移动", 20, INK, "middle", 660)])
    return "".join(bits)


def _q17() -> str:
    bits = [panel_title("地轴倾斜让同一半球一年中接收阳光的角度和时长变化"), rect(34, 82, 892, 382, DARK, "none", 0, 4), sun(480, 270, 65)]
    for x, y in ((480, 115), (750, 270), (480, 425), (210, 270)):
        bits.extend([earth(x, y, 48), line(x - 18, y + 60, x + 18, y - 60, WHITE, 4)])
    bits.extend([ellipse(480, 270, 300, 170, "none", BLUE, 3), text(480, 72, "北半球夏季附近", 18, WHITE, "middle"), text(480, 505, "北半球冬季附近", 18, WHITE, "middle"), label(750, 270, 620, 105, "地轴方向在公转中大致保持不变"), label(480, 425, 620, 405, "冬季太阳较低、白昼较短，并非因为地球最远")])
    return "".join(bits)


def _q18() -> str:
    bits = [panel_title("冬季太阳较低、白昼较短，同一片地面得到的能量更少"), landscape(), sun(150, 250, 42), rect(430, 330, 250, 50, "#D9D4C8", INK, 3)]
    bits.extend([path("M195,250 L430,330 L680,330 L195,285 Z", YELLOW, "none").replace('/>', ' opacity="0.35"/>'), sun(760, 125, 30), path("M735,150 L620,330 L690,330 L790,155 Z", ORANGE, "none").replace('/>', ' opacity="0.25"/>'), text(245, 430, "低角度：能量铺在更大面积", 20, INK, "middle"), text(700, 430, "高角度：同样光束更集中", 20, INK, "middle"), text(470, 490, "冬季还常有更长夜晚，地面有更多时间向外散热", 20, INK, "middle", 650)])
    return "".join(bits)


def _q19() -> str:
    bits = [panel_title("晴冷夜里叶面降到冰点以下，水汽可直接沉积成冰晶"), landscape(night=True), leaf(330, 330, 1.15)]
    for x, y in ((250, 290), (300, 315), (350, 280), (410, 315)):
        bits.append(path(f"M{x-12},{y} H{x+12} M{x},{y-12} V{y+12} M{x-9},{y-9} L{x+9},{y+9} M{x+9},{y-9} L{x-9},{y+9}", "none", WHITE, 2.5))
    bits.extend([arrow("M180,160 Q245,220 285,275", WATER, 4), label(300, 300, 620, 110, "冰晶长在叶面，不是从云里落下的雪"), label(330, 350, 620, 300, "风小、云少时叶面更容易向天空辐射散热"), text(430, 490, "若水汽先变成小水滴，再冻结，就会形成不同的结冰形态", 19, WHITE, "middle", 690)])
    return "".join(bits)


def _q20() -> str:
    bits = [panel_title("水结冰时形成较疏松的晶格，所以同体积冰通常更轻"), rect(34, 82, 892, 382, "#EAE7DF", "none", 0, 4), rect(135, 150, 350, 260, "#D6EEF2", INK, 4, 8), rect(150, 255, 320, 140, WATER, "none"), rect(245, 205, 125, 105, "#EAF6F7", BLUE, 4, 5)]
    bits.extend([line(150, 255, 470, 255, BLUE, 4), rect(620, 140, 230, 230, CREAM, INK, 3, 12)])
    for row in range(4):
        for col in range(5):
            x = 660 + col * 38 + (row % 2) * 18
            y = 180 + row * 48
            bits.append(circle(x, y, 7, BLUE, INK, 1.5))
            if col < 4:
                bits.append(line(x, y, x + 38, y, BLUE_DARK, 2))
    bits.extend([label(305, 250, 620, 105, "冰的平均密度比液态水小，浮力可托住它"), label(730, 250, 620, 415, "六角晶格留下较多空隙；融化后分子能靠得更近")])
    return "".join(bits)


def _q21() -> str:
    bits = [panel_title("水分子在冰晶表面按六角对称继续排列，形成六个主枝"), rect(34, 82, 892, 382, SKY, "none", 0, 4)]
    cx, cy = 350, 270
    for index in range(6):
        angle = index * math.tau / 6
        bits.append(line(cx, cy, cx + math.cos(angle) * 165, cy + math.sin(angle) * 165, BLUE, 7))
        for offset in (70, 115):
            bx, by = cx + math.cos(angle) * offset, cy + math.sin(angle) * offset
            for side in (-1, 1):
                branch_angle = angle + side * math.pi / 3
                bits.append(line(bx, by, bx + math.cos(branch_angle) * 42, by + math.sin(branch_angle) * 42, BLUE, 4))
    bits.extend([rect(650, 145, 210, 205, CREAM, INK, 3, 12)])
    for index in range(6):
        angle = index * math.tau / 6
        bits.extend([circle(755 + math.cos(angle) * 55, 250 + math.sin(angle) * 55, 10, BLUE, INK, 2), line(755, 250, 755 + math.cos(angle) * 55, 250 + math.sin(angle) * 55, BLUE_DARK, 2)])
    bits.extend([label(755, 250, 620, 410, "水分子的结合方向带来六角对称"), text(420, 490, "温度和湿度变化让每片雪晶的分枝细节不同", 20, INK, "middle", 650)])
    return "".join(bits)


def _q22() -> str:
    bits = [panel_title("呼出的温暖湿气遇到冷空气，凝结成能看见的小水滴"), landscape(), person(280, 330, 0.65)]
    for index, (x, y, r) in enumerate(((390, 225, 10), (430, 205, 14), (475, 190, 18), (525, 180, 13), (565, 175, 9))):
        bits.append(circle(x, y, r, WHITE, BLUE, 2))
    bits.extend([arrow("M310,230 Q390,205 455,190", ORANGE, 4), label(475, 190, 620, 115, "白气是液态微滴，不是看得见的水蒸气"), label(380, 220, 620, 280, "空气越冷，能保持为气态的水越少"), text(430, 490, "微滴再蒸发后会消失；原理和云、雾相似", 20, INK, "middle", 650)])
    return "".join(bits)


def _q23() -> str:
    bits = [panel_title("金属导热快，会更快把手的热带走；温度未必比木头更低"), rect(34, 82, 892, 382, "#EEE9DF", "none", 0, 4), rect(120, 220, 270, 100, "#A9B5BC", INK, 4, 4), rect(560, 220, 270, 100, BROWN, INK, 4, 4)]
    bits.extend([circle(255, 160, 40, "#E7B39E", INK, 3), circle(695, 160, 40, "#E7B39E", INK, 3)])
    for x in (250, 285, 320, 355):
        bits.append(arrow(f"M{x},205 L{x},245", BLUE, 3))
    for x in (650, 685, 720):
        bits.append(arrow(f"M{x},205 L{x},235", ORANGE, 2.5))
    bits.extend([text(255, 375, "金属：热流较快", 21, INK, "middle"), text(695, 375, "木头：热流较慢", 21, INK, "middle"), text(480, 475, "在同一房间放久后两者温度可相同，触感仍不同", 20, INK, "middle", 650)])
    return "".join(bits)


def _q24() -> str:
    bits = [panel_title("衣服夹住许多不流动的空气，减慢身体向外传热"), rect(34, 82, 892, 382, "#E8EFF0", "none", 0, 4), person(310, 330, 0.75), path("M245,245 Q310,200 375,245 L360,370 H260 Z", RED, INK, 4)]
    for x, y in ((270, 255), (300, 235), (335, 250), (280, 290), (330, 300)):
        bits.append(circle(x, y, 10, WHITE, BLUE, 2))
    bits.extend([rect(620, 145, 230, 215, CREAM, INK, 3, 12), path("M655,320 Q700,180 745,320 Q790,180 835,320", "none", RED, 15)])
    for x, y in ((690, 260), (735, 225), (780, 270)):
        bits.append(circle(x, y, 13, WHITE, BLUE, 2))
    bits.extend([label(735, 250, 610, 405, "纤维间空气导热慢，也不易形成强对流"), text(415, 490, "衣服不制造热；身体产生热，衣服只是减慢热量跑掉", 20, INK, "middle", 650)])
    return "".join(bits)


def _q25() -> str:
    bits = [panel_title("帽子减少头部的对流、传导和辐射散热"), rect(34, 82, 892, 382, "#E2ECEF", "none", 0, 4), person(360, 335, 0.85), path("M285,225 Q360,145 435,225 Z", RED, INK, 4), rect(275, 220, 170, 35, RED, INK, 3, 15)]
    for data in ("M310,180 Q280,130 250,105", "M360,160 Q360,110 360,85", "M410,180 Q450,130 485,110"):
        bits.append(arrow(data, ORANGE, 4))
    bits.extend([label(355, 215, 620, 125, "帽内空气层减慢热流"), label(280, 230, 620, 275, "遮住耳朵和额头可减少迎风面积"), text(420, 490, "头并不会神奇地失去“大多数”热；哪里暴露，哪里就更容易散热", 19, INK, "middle", 700)])
    return "".join(bits)


def _q26() -> str:
    bits = [panel_title("钟表用规律振荡计时，再由齿轮把节拍传给指针"), rect(34, 82, 892, 382, "#EEE9DF", "none", 0, 4), circle(330, 275, 150, CREAM, INK, 5)]
    for index in range(12):
        angle = index * math.tau / 12 - math.pi / 2
        bits.append(line(330 + math.cos(angle) * 122, 275 + math.sin(angle) * 122, 330 + math.cos(angle) * 137, 275 + math.sin(angle) * 137, INK, 4))
    bits.extend([line(330, 275, 330, 175, BLUE_DARK, 8), line(330, 275, 410, 315, RED, 7), circle(330, 275, 10, INK), circle(680, 230, 70, BLUE, INK, 4), circle(785, 285, 45, ORANGE, INK, 4)])
    for cx, cy, r, teeth in ((680, 230, 70, 12), (785, 285, 45, 10)):
        for index in range(teeth):
            angle = index * math.tau / teeth
            bits.append(line(cx + math.cos(angle) * r, cy + math.sin(angle) * r, cx + math.cos(angle) * (r + 14), cy + math.sin(angle) * (r + 14), INK, 7))
    bits.extend([label(680, 230, 620, 115, "摆轮、摆或石英晶体提供稳定节拍"), label(785, 285, 620, 405, "齿轮把快速节拍换成秒、分和小时的转动")])
    return "".join(bits)


def _q27() -> str:
    bits = [panel_title("日历把地球自转、公转和人类约定整理成可重复的日期表"), rect(34, 82, 892, 382, "#EEE9DF", "none", 0, 4), rect(110, 130, 370, 270, WHITE, INK, 4, 10), rect(110, 130, 370, 60, RED, INK, 3, 10)]
    for row in range(4):
        for col in range(7):
            bits.append(circle(145 + col * 48, 225 + row * 48, 8, BLUE if (row + col) % 2 else ORANGE))
    bits.extend([sun(735, 265, 45), earth(735, 140, 42), ellipse(735, 265, 135, 125, "none", BLUE, 3), arrow("M735,140 Q820,190 855,265", BLUE, 4), label(735, 140, 620, 405, "一年接近地球绕太阳一周；月份和闰日用于校准"), text(310, 445, "日期格是人类约定", 20, INK, "middle")])
    return "".join(bits)


def _q28() -> str:
    bits = [panel_title("七天一周是长期形成的历法约定，不是自然界唯一的周期"), rect(34, 82, 892, 382, "#EEE9DF", "none", 0, 4)]
    colors = (RED, ORANGE, YELLOW, GREEN, TEAL, BLUE, PURPLE)
    for index, color in enumerate(colors):
        x = 95 + index * 120
        bits.extend([rect(x, 180, 92, 120, WHITE, INK, 3, 8), circle(x + 46, 240, 26, color, INK, 2), text(x + 46, 345, str(index + 1), 24, INK, "middle", 750)])
    bits.extend([path("M110,390 H845", "none", TEAL, 5, True), text(480, 465, "不同文化曾用过不同周长；七日制后来广泛沿用", 20, INK, "middle", 680)])
    return "".join(bits)


def _q29() -> str:
    bits = [panel_title("睡眠让大脑和身体按阶段调整，不是简单地“关机”"), rect(34, 82, 892, 382, DARK, "none", 0, 4), person(320, 320, 1.0, True)]
    bits.extend([path("M560,315 C590,240 620,360 650,285 S715,230 745,315 S805,360 845,275", "none", TEAL, 6), text(700, 370, "一夜中睡眠阶段反复循环", 20, WHITE, "middle"), label(260, 292, 620, 105, "身体修复、生长和免疫调节持续进行"), label(620, 285, 620, 210, "大脑整理记忆，也会出现梦境"), text(435, 490, "规律睡眠帮助注意、情绪和学习；不同年龄需要的时长不同", 20, WHITE, "middle", 690)])
    return "".join(bits)


def _q30() -> str:
    bits = [panel_title("猫头鹰的感官和捕猎节律适合低光环境，但白天并非一定睡着"), landscape(night=True)]
    bits.extend([ellipse(365, 300, 90, 115, BROWN, INK, 4), polygon([(300, 225), (310, 150), (345, 220)], BROWN, INK, 3), polygon([(430, 225), (420, 150), (385, 220)], BROWN, INK, 3), circle(330, 270, 35, CREAM, INK, 3), circle(400, 270, 35, CREAM, INK, 3), circle(330, 270, 13, INK), circle(400, 270, 13, INK), polygon([(365, 300), (350, 325), (380, 325)], YELLOW, INK, 2), path("M200,410 Q365,350 520,410", "none", BROWN_DARK, 18)])
    bits.extend([label(330, 270, 620, 110, "大眼和高感光能力适应昏暗环境"), label(400, 270, 620, 245, "不对称耳位和面盘帮助定位声音"), label(365, 360, 620, 385, "昼夜活动还受食物、季节和物种差异影响")])
    return "".join(bits)


def _q31() -> str:
    bits = [panel_title("太阳风粒子沿地磁场进入高纬大气，使气体发出彩色光"), rect(34, 82, 892, 382, DARK, "none", 0, 4), earth(310, 300, 105)]
    for offset, color in ((0, "#64D6A5"), (22, "#7488E7"), (44, "#D76DB2")):
        bits.append(path(f"M120,{380-offset} Q310,{80-offset} 500,{380-offset}", "none", color, 16))
    bits.extend([arrow("M850,145 Q660,170 500,245", ORANGE, 5), path("M265,150 Q155,280 275,435 M355,150 Q470,280 345,435", "none", BLUE, 4), label(540, 230, 620, 105, "带电粒子受地磁场引导"), label(340, 180, 620, 250, "氧和氮被激发后发出不同颜色的光"), text(440, 490, "极光高度很高，不是贴在云上的彩带", 20, WHITE, "middle", 620)])
    return "".join(bits)


def _q32() -> str:
    bits = [panel_title("空气占空间并能推压物体，虽然眼睛通常看不见它"), rect(34, 82, 892, 382, "#EEE9DF", "none", 0, 4), rect(130, 210, 320, 210, WATER, INK, 4, 8), rect(230, 135, 120, 190, "none", INK, 5, 6)]
    bits.extend([line(140, 250, 440, 250, BLUE, 4), arrow("M290,110 L290,165", DARK, 5)])
    for x, y in ((260, 180), (295, 205), (325, 175)):
        bits.append(circle(x, y, 8, TEAL))
    bits.extend([ellipse(690, 265, 90, 115, RED, INK, 4), rect(676, 375, 28, 40, BROWN, INK, 3, 4), arrow("M570,265 Q600,230 625,240", TEAL, 4), label(290, 220, 590, 115, "倒扣杯里的空气阻止水占据同一空间"), label(690, 265, 590, 405, "吹入更多空气，气球壁被向外推开")])
    return "".join(bits)


def _q33() -> str:
    bits = [panel_title("地表受热不均使空气密度和气压不同，空气便发生水平流动"), landscape(), sun(200, 130, 42)]
    bits.extend([rect(100, 355, 300, 70, "#D99155", INK, 3), rect(560, 355, 300, 70, WATER, INK, 3)])
    for x in (180, 260, 340):
        bits.append(arrow(f"M{x},350 Q{x-30},260 {x},190", ORANGE, 4))
    for x in (620, 700, 780):
        bits.append(arrow(f"M{x},205 Q{x+20},285 {x},345", BLUE, 4))
    bits.extend([arrow("M560,330 Q450,310 400,330", TEAL, 6), arrow("M400,180 Q500,155 560,180", TEAL, 4), text(250, 455, "暖地面上方空气上升", 20, INK, "middle"), text(710, 455, "较冷空气补过来", 20, INK, "middle"), label(470, 320, 620, 105, "这种循环的一部分在地面附近表现为风")])
    return "".join(bits)


def _q34() -> str:
    bits = [panel_title("风速不同会让风袋、树枝和旗帜呈现不同弯曲程度"), landscape()]
    for index, (x, bend, color) in enumerate(((250, 70, GREEN), (700, 150, ORANGE))):
        bits.extend([line(x, 390, x, 155, INK, 9), path(f"M{x},{170} Q{x+bend/2},{180} {x+bend},{220} Q{x+bend/2},{250} {x},{230} Z", color, INK, 3)])
        count = 2 if index == 0 else 5
        for row in range(count):
            bits.append(arrow(f"M{x-150},{190+row*35} L{x-45},{190+row*35}", BLUE, 3 + index))
    bits.extend([text(250, 455, "较小风速", 21, INK, "middle"), text(700, 455, "较大风速", 21, INK, "middle"), text(480, 500, "阵风会不断变化；测量风速要在规定条件下重复取样", 19, INK, "middle", 700)])
    return "".join(bits)


def _q35() -> str:
    bits = [panel_title("泡泡是被肥皂水薄膜包住的一团气体"), rect(34, 82, 892, 382, "#E7F2F3", "none", 0, 4), circle(340, 275, 150, "#D9F1F3", BLUE, 5)]
    bits.extend([path("M250,195 Q340,120 430,195", "none", WHITE, 16), rect(620, 145, 230, 220, CREAM, INK, 3, 12), line(650, 205, 820, 205, TEAL, 8), line(650, 295, 820, 295, TEAL, 8), rect(650, 213, 170, 74, WATER, "none")])
    for x in range(660, 820, 30):
        bits.extend([circle(x, 205, 8, RED), line(x, 213, x, 287, DARK, 2), circle(x, 295, 8, RED)])
    bits.extend([label(340, 275, 610, 110, "里面主要是吹进去的空气"), label(740, 250, 610, 410, "薄膜是水层夹在两层肥皂分子之间"), text(420, 490, "彩色条纹来自不同厚度薄膜对光的干涉", 20, INK, "middle", 650)])
    return "".join(bits)


def _q36() -> str:
    bits = [panel_title("吹气或打气会增加气球中的气体，内压把橡胶膜撑开"), rect(34, 82, 892, 382, "#EEE9DF", "none", 0, 4), ellipse(250, 270, 65, 85, RED, INK, 4), ellipse(620, 260, 145, 165, RED, INK, 4), rect(230, 350, 40, 38, BROWN, INK, 3, 4), rect(600, 420, 40, 28, BROWN, INK, 3, 4)]
    bits.extend([rect(340, 250, 120, 70, BLUE, INK, 3, 5), line(460, 285, 505, 285, DARK, 12), arrow("M330,285 L290,285", TEAL, 5), arrow("M500,285 L535,285", TEAL, 5)])
    for x, y in ((585, 220), (650, 200), (690, 280), (600, 330), (660, 350)):
        bits.append(circle(x, y, 7, WHITE, BLUE, 2))
    bits.extend([text(250, 435, "空气少", 21, INK, "middle"), text(620, 470, "空气多、橡胶拉伸", 21, INK, "middle"), label(620, 260, 620, 105, "气球停止变大时，内外压力和橡胶拉力达到平衡")])
    return "".join(bits)


def _q37() -> str:
    bits = [panel_title("吸管顶部压力降低后，杯面上的大气压把水推入吸管"), rect(34, 82, 892, 382, "#EEE9DF", "none", 0, 4), rect(150, 195, 310, 230, "#DCECF0", INK, 4, 10), rect(165, 285, 280, 125, WATER, "none"), line(310, 115, 310, 370, TEAL, 18), person(620, 285, 0.62)]
    bits.extend([path("M310,115 Q420,115 540,210", "none", TEAL, 18), arrow("M310,350 L310,205", BLUE, 6), arrow("M180,270 Q240,285 275,300", ORANGE, 4), label(310, 190, 610, 105, "嘴吸走部分空气，使吸管内压力下降"), label(210, 285, 610, 270, "外界空气压着水面，把水推向低压处"), text(430, 490, "真正把水送上去的主要是压力差，不是嘴在远处“拉”水", 20, INK, "middle", 680)])
    return "".join(bits)


def _q38() -> str:
    bits = [panel_title("湿空气上升冷却，水汽凝结在微粒周围，许多小滴组成云"), landscape(water=True), sun(150, 130, 38), cloud(650, 190, 1.05)]
    bits.extend([arrow("M260,320 Q320,220 390,170", BLUE, 5), arrow("M420,175 Q500,150 555,175", TEAL, 4)])
    for x, y in ((585, 190), (630, 165), (680, 195), (720, 175)):
        bits.append(circle(x, y, 7, BLUE, INK, 1.5))
    bits.extend([label(350, 220, 620, 330, "上升空气膨胀并降温"), label(650, 180, 620, 105, "凝结核周围长出微小水滴或冰晶"), text(420, 490, "单个云滴很小，下降慢；云是大量云滴共同散射光的结果", 19, INK, "middle", 700)])
    return "".join(bits)


def _q39() -> str:
    bits = [panel_title("云滴碰并长大，重力超过上升气流和空气阻力后落成雨"), landscape(), cloud(390, 190, 1.15, True)]
    for index, (x, y, scale) in enumerate(((300, 180, .18), (350, 205, .24), (400, 210, .32), (450, 220, .42), (500, 245, .58))):
        bits.append(drop(x, y, scale))
    for x, y in ((320, 330), (390, 365), (470, 340), (520, 385)):
        bits.append(drop(x, y, .25))
        bits.append(arrow(f"M{x},{y+15} L{x},{y+65}", BLUE, 3))
    bits.extend([label(420, 215, 620, 105, "小滴碰撞、合并，冰晶也可长大并融化"), label(470, 350, 620, 330, "雨滴离开云后仍会受空气阻力并可能蒸发"), text(420, 495, "不是云“装满了水”才漏下来，而是滴的大小和气流条件改变", 19, INK, "middle", 710)])
    return "".join(bits)


def _q40() -> str:
    bits = [panel_title("小雨滴近似圆球，大雨滴下落时底部变平，过大还会碎开"), rect(34, 82, 892, 382, SKY, "none", 0, 4)]
    bits.extend([circle(170, 255, 45, BLUE, INK, 3), ellipse(430, 270, 85, 65, BLUE, INK, 3), path("M640,220 Q750,160 830,245 Q800,330 710,315 Q650,300 640,220 Z", BLUE, INK, 4), line(690, 270, 780, 270, WHITE, 4)])
    bits.extend([text(170, 360, "小滴：表面张力使它接近球形", 19, INK, "middle"), text(430, 380, "较大滴：空气把底部压平", 19, INK, "middle"), text(745, 380, "过大：振动并碎成小滴", 19, INK, "middle"), text(480, 475, "“眼泪形”常是画法；自由下落的雨滴没有尖尾巴", 20, INK, "middle", 670)])
    return "".join(bits)


def _q41() -> str:
    bits = [panel_title("云中降水常先以冰晶形成，穿过暖层时可能融化成雨"), rect(34, 82, 892, 382, SKY, "none", 0, 4), cloud(420, 165, 1.05, True), rect(34, 285, 892, 65, "#F4C78A", "none")]
    for x in (250, 340, 430, 520):
        bits.append(path(f"M{x-12},225 H{x+12} M{x},213 V{x+24} M{x-9},216 L{x+9},234 M{x+9},216 L{x-9},234", "none", WHITE, 3))
    for x in (300, 410, 520):
        bits.append(drop(x, 385, .24))
    bits.extend([text(750, 255, "0°C 以下的冷层", 20, INK, "middle"), text(750, 330, "较暖空气层", 20, INK, "middle"), label(410, 325, 620, 410, "冰晶完全融化后成为雨；若地面附近仍冷，也可能形成冻雨")])
    return "".join(bits)


def _q42() -> str:
    bits = [panel_title("强上升气流让冰粒在雷暴云中反复碰到过冷水滴，长出多层冰"), landscape(), cloud(390, 185, 1.2, True)]
    bits.extend([arrow("M360,350 Q250,250 350,170", TEAL, 6), arrow("M450,170 Q560,250 465,360", BLUE, 6), circle(405, 245, 26, WHITE, BLUE, 4), rect(650, 135, 220, 225, CREAM, INK, 3, 12), circle(760, 250, 82, WHITE, INK, 4)])
    for radius, color in ((20, BLUE), (38, CREAM), (56, BLUE), (75, CREAM)):
        bits.append(circle(760, 250, radius, "none", color, 8))
    bits.extend([label(410, 245, 620, 105, "冰胚被上升气流托回高处"), label(760, 250, 620, 405, "透明和浑浊冰层记录冻结方式不同"), text(410, 490, "长到气流托不住时才落下；并非所有雷雨都会下冰雹", 20, INK, "middle", 660)])
    return "".join(bits)


def _q43() -> str:
    bits = [panel_title("闪电把空气瞬间加热膨胀，产生向外传播的冲击波和雷声"), landscape(night=True), cloud(350, 165, 1.0, True), path("M365,220 L315,310 L360,300 L305,410", "none", YELLOW, 12)]
    for radius in (45, 80, 120, 165):
        bits.append(path(f"M{360+radius},{300-radius*.55} Q{360+radius*1.3},{300} {360+radius},{300+radius*.55}", "none", TEAL if radius % 2 else BLUE, 4))
    bits.extend([label(345, 290, 620, 105, "闪电通道温度骤升，空气急剧膨胀"), label(570, 300, 620, 300, "声波被地面、云层和不同温度空气折射反射，形成轰隆延续"), text(420, 490, "先看见闪光再听到雷声，是因为光比声音快得多", 20, WHITE, "middle", 650)])
    return "".join(bits)


def _q44() -> str:
    bits = [panel_title("雷暴云中电荷分离，电场足够强时空气被击穿形成放电通道"), landscape(night=True), cloud(370, 170, 1.1, True)]
    for x in (280, 340, 400, 460):
        bits.append(circle(x, 145, 12, BLUE, INK, 2))
        bits.append(text(x, 151, "−", 18, WHITE, "middle", 750))
    for x in (300, 370, 440):
        bits.append(circle(x, 215, 12, RED, INK, 2))
        bits.append(text(x, 221, "+", 18, WHITE, "middle", 750))
    bits.extend([path("M390,230 L335,315 L380,305 L320,420", "none", YELLOW, 12), circle(320, 420, 16, RED, INK, 2), label(370, 180, 620, 105, "冰粒和水滴碰撞、气流搬运，使电荷区域分开"), label(350, 330, 620, 315, "空气电离后形成导电通道，电荷快速移动"), text(410, 490, "闪电可在云内、云间或云地之间发生", 20, WHITE, "middle", 620)])
    return "".join(bits)


def _q45() -> str:
    bits = [panel_title("地面附近的空气冷到接近饱和，微小水滴悬浮起来就是雾"), landscape()]
    for x, y, s in ((210, 310, .75), (430, 330, .8), (650, 305, .72)):
        bits.append(cloud(x, y, s))
    bits.extend([house(770, 350, .42), label(430, 330, 620, 105, "雾滴与云滴相似，区别主要是它接触地面"), label(250, 335, 620, 270, "夜间地面降温、暖湿空气移到冷地面等都能成雾"), text(420, 490, "看不清远处，是因为光被大量小滴散射", 20, INK, "middle", 620)])
    return "".join(bits)


def _q46() -> str:
    bits = [panel_title("叶面夜里变冷，附近空气中的水汽凝结成露珠"), landscape(), leaf(350, 320, 1.3)]
    for x, y, s in ((260, 280, .35), (340, 300, .28), (420, 275, .32)):
        bits.append(drop(x, y, s, "#BEE5EA"))
    bits.extend([arrow("M180,150 Q240,220 285,265", WATER, 4), label(340, 290, 620, 105, "露珠来自空气水汽，不是叶子把水挤出来"), label(250, 340, 620, 285, "表面温度低于露点时，凝结速度超过蒸发"), text(420, 490, "风、云、表面材料和位置都会影响露珠多少", 20, INK, "middle", 620)])
    return "".join(bits)


def _q47() -> str:
    bits = [panel_title("水洼表面不断有分子逃入空气，留下的水就逐渐变少"), landscape(), sun(150, 135, 35)]
    for index, (x, width) in enumerate(((220, 170), (500, 105), (760, 45))):
        bits.append(ellipse(x, 390, width, 25, WATER, BLUE, 3))
        bits.append(arrow(f"M{x},{360} Q{x-20},{280} {x+5},{230}", BLUE, 3))
        bits.append(text(x, 445, ("刚下雨", "过一会儿", "更久以后")[index], 19, INK, "middle"))
    bits.extend([text(470, 495, "阳光、风、较干空气和较大表面积都会加快蒸发", 20, INK, "middle", 650)])
    return "".join(bits)


def _q48() -> str:
    bits = [panel_title("湿衣服中的水从纤维表面蒸发，流动空气把水汽带走"), landscape(), line(105, 145, 835, 145, INK, 5)]
    for index, color in enumerate((RED, BLUE, YELLOW)):
        x = 190 + index * 230
        bits.extend([path(f"M{x-55},170 L{x-95},215 L{x-65},245 L{x-35},220 V365 H{x+35} V220 L{x+65},245 L{x+95},215 L{x+55},170 Z", color, INK, 3), line(x - 30, 145, x - 15, 170, INK, 3), line(x + 30, 145, x + 15, 170, INK, 3)])
        for row in range(3 - index):
            bits.append(drop(x - 25 + row * 25, 300, .16))
    bits.extend([arrow("M120,410 Q450,360 820,410", TEAL, 6), text(480, 490, "摊开比揉成一团干得快，因为更多湿纤维接触空气", 20, INK, "middle", 660)])
    return "".join(bits)


def _q49() -> str:
    bits = [panel_title("真正的水蒸气是看不见的；白雾是已经凝结的小水滴"), rect(34, 82, 892, 382, "#EEE9DF", "none", 0, 4), rect(120, 280, 250, 125, "#A9B4B8", INK, 4, 8), path("M150,280 Q245,220 340,280", "none", INK, 4)]
    for x, y in ((230, 235), (270, 205), (310, 185)):
        bits.append(circle(x, y, 5, TEAL))
    for x, y, r in ((390, 175, 12), (430, 155, 18), (475, 165, 23), (525, 145, 18), (565, 160, 12)):
        bits.append(circle(x, y, r, WHITE, BLUE, 2))
    bits.extend([label(275, 205, 620, 105, "壶嘴附近高温区：气态水分子不可见"), label(475, 160, 620, 285, "稍远处冷却凝结：微滴散射光，出现白雾"), text(420, 490, "云、雾和冷天白气都是微滴或冰晶，不是水蒸气本身", 20, INK, "middle", 680)])
    return "".join(bits)


def _q50() -> str:
    bits = [panel_title("冰吸收周围热量后，晶格被打乱，分子成为能流动的液态水"), rect(34, 82, 892, 382, "#EEE9DF", "none", 0, 4), rect(150, 190, 220, 190, "#E7F4F5", BLUE, 4, 8), ellipse(670, 365, 155, 32, WATER, BLUE, 3)]
    for row in range(4):
        for col in range(4):
            bits.append(circle(190 + col * 45 + (row % 2) * 20, 225 + row * 42, 7, BLUE, INK, 1.5))
    for index in range(18):
        angle = index * 2.4
        radius = 15 + index * 6
        bits.append(circle(670 + math.cos(angle) * radius, 315 + math.sin(angle) * radius * .45, 7, BLUE, INK, 1.5))
    bits.extend([arrow("M420,285 L535,285", ORANGE, 7), text(480, 255, "吸热", 21, INK, "middle"), label(260, 280, 620, 105, "固态中分子围绕固定位置振动"), label(670, 315, 620, 405, "液态中分子仍靠近，但能不断交换邻居")])
    return "".join(bits)


def _q51() -> str:
    bits = [panel_title("冷杯壁使附近空气降到露点以下，水汽便凝结在外表面"), rect(34, 82, 892, 382, "#EEE9DF", "none", 0, 4), rect(250, 145, 240, 260, "#D7E9ED", INK, 4, 8), rect(265, 190, 210, 200, WATER, "none")]
    for x, y in ((225, 215), (215, 275), (230, 340), (510, 205), (520, 275), (505, 340)):
        bits.append(drop(x, y, .18))
    bits.extend([circle(350, 250, 55, WHITE, BLUE, 3), arrow("M700,210 Q600,220 520,235", TEAL, 4), arrow("M700,300 Q610,300 520,290", TEAL, 4), label(510, 235, 620, 105, "杯外的水来自空气，不是穿过杯壁漏出"), label(350, 250, 620, 280, "冰水让杯壁和附近空气降温"), text(420, 490, "空气越湿，或杯壁越冷，通常越容易出现水珠", 20, INK, "middle", 650)])
    return "".join(bits)


def _q52() -> str:
    bits = [panel_title("水在海洋、空气、云、陆地、河流和地下不断交换"), landscape(water=True), sun(120, 130, 35), cloud(480, 155, .8), path("M640,330 Q720,205 815,330", GREEN, INK, 4)]
    bits.extend([arrow("M210,320 Q245,220 330,185", BLUE, 5), arrow("M520,205 Q610,240 660,300", BLUE, 5), path("M760,315 Q650,375 520,410 Q390,440 250,420", "none", BLUE, 10, True), arrow("M690,400 Q630,435 570,440", TEAL, 4), text(230, 250, "蒸发", 20, INK, "middle"), text(595, 245, "降水", 20, INK, "middle"), text(500, 455, "地表径流与地下水回到河海", 20, INK, "middle"), label(480, 155, 620, 105, "凝结形成云滴和冰晶")])
    return "".join(bits)


def _q53() -> str:
    bits = [panel_title("重力沿着地形坡度拉动水，河道把分散流水汇集起来"), landscape(), path("M110,180 Q220,150 330,215 Q450,285 570,315 Q690,345 850,390", "none", BLUE, 24)]
    bits.extend([arrow("M220,185 Q300,210 350,250", BLUE_DARK, 5), arrow("M470,285 Q555,325 620,335", BLUE_DARK, 5), arrow("M680,350 Q760,380 820,390", BLUE_DARK, 5), line(140, 425, 840, 425, INK, 3), line(140, 425, 140, 160, INK, 3), label(330, 220, 620, 105, "水从势能较高处向较低处运动"), text(430, 490, "河流会弯曲、侵蚀和堆积，但整体水面仍沿下游降低", 20, INK, "middle", 660)])
    return "".join(bits)


def _q54() -> str:
    bits = [panel_title("雨水会渗入土壤和岩石缝隙，在不透水层上方形成地下水"), rect(34, 82, 892, 170, SKY, "none", 0, 4), rect(34, 252, 892, 100, SOIL, "none"), rect(34, 352, 892, 112, "#9B8675", "none")]
    bits.extend([path("M34,335 Q180,305 330,335 T640,335 T950,335", WATER, "none").replace('/>', ' opacity="0.65"/>'), path("M34,420 Q190,380 350,420 T690,420 T1030,420", "none", DARK, 18)])
    for x in (220, 320, 420):
        bits.append(arrow(f"M{x},170 L{x+20},315", BLUE, 4))
    bits.extend([rect(700, 155, 45, 245, "none", INK, 4, 4), line(722, 330, 722, 190, BLUE, 12), label(360, 325, 610, 115, "孔隙和裂缝被水充满的区域叫饱和带"), label(720, 330, 610, 275, "井进入含水层后可以取水"), text(420, 495, "地下水也会缓慢流动，并从泉、河流或海岸排出", 20, INK, "middle", 650)])
    return "".join(bits)


def _q55() -> str:
    bits = [panel_title("自来水通常经历取水、处理、储存和管网输送"), landscape(water=True)]
    bits.extend([ellipse(150, 340, 90, 35, WATER, BLUE, 3), rect(300, 255, 130, 145, "#AAB7BC", INK, 4, 5), rect(505, 200, 115, 205, "#D6E7EA", INK, 4, 8), house(790, 340, .55)])
    bits.extend([arrow("M235,340 L285,330", BLUE, 5), arrow("M435,325 L490,300", BLUE, 5), arrow("M625,310 L700,330", BLUE, 5), text(150, 420, "河湖或地下水", 19, INK, "middle"), text(365, 445, "混凝、沉淀、过滤、消毒", 18, INK, "middle"), text(560, 445, "清水池或水塔", 19, INK, "middle"), text(790, 445, "管道送到家", 19, INK, "middle"), text(480, 495, "不同城市流程会不同；饮用水仍需持续检测", 20, INK, "middle", 650)])
    return "".join(bits)


def _q56() -> str:
    bits = [panel_title("肥皂分子一端喜欢水、一端喜欢油，把油污包成能被水带走的小团"), rect(34, 82, 892, 382, "#EEE9DF", "none", 0, 4), rect(110, 190, 280, 210, WHITE, INK, 4, 12), ellipse(250, 300, 100, 50, BROWN, INK, 3)]
    bits.extend([rect(570, 140, 260, 240, CREAM, INK, 3, 12), circle(700, 260, 60, BROWN, INK, 3)])
    for index in range(16):
        angle = index * math.tau / 16
        x, y = 700 + math.cos(angle) * 83, 260 + math.sin(angle) * 83
        bits.extend([circle(x, y, 8, RED, INK, 1.5), line(x - math.cos(angle) * 8, y - math.sin(angle) * 8, x - math.cos(angle) * 34, y - math.sin(angle) * 34, TEAL, 3)])
    bits.extend([arrow("M410,285 L520,285", BLUE, 6), label(700, 260, 610, 410, "形成胶束后，流水能把分散油滴带走"), text(310, 470, "搓洗让水、肥皂和污物充分接触", 20, INK, "middle")])
    return "".join(bits)


def _q57() -> str:
    bits = [panel_title("伞布纤维织得紧并带防水处理，水难以进入细小孔隙"), rect(34, 82, 892, 382, SKY, "none", 0, 4), path("M120,300 Q360,80 600,300 Z", RED, INK, 5), line(360, 180, 360, 430, INK, 9), path("M360,430 q0,55 45,25", "none", INK, 9)]
    for x, y in ((210, 225), (300, 180), (400, 175), (500, 225)):
        bits.append(drop(x, y, .22, "#BDE6EB"))
    bits.extend([rect(650, 145, 210, 210, CREAM, INK, 3, 12)])
    for row in range(6):
        bits.append(line(675, 175 + row * 28, 835, 175 + row * 28, BROWN, 5))
    for col in range(6):
        bits.append(line(685 + col * 28, 165, 685 + col * 28, 335, BROWN, 5))
    bits.extend([drop(755, 150, .28), label(755, 220, 610, 405, "孔隙小、表面不易润湿，水压不足时水珠留在外面"), text(420, 495, "久用磨损、接缝或强水压仍可能让雨伞进水", 20, INK, "middle", 650)])
    return "".join(bits)


def _q58() -> str:
    bits = [panel_title("天气描述短时间状态；气候来自许多年天气记录的统计"), rect(34, 82, 430, 382, SKY, "none", 0, 4), rect(496, 82, 430, 382, "#EEE9DF", "none", 0, 4), cloud(245, 200, .72, True), drop(210, 320, .2), sun(300, 320, 28)]
    for row in range(4):
        for col in range(5):
            color = (BLUE, YELLOW, GRAY, GREEN)[(row + col) % 4]
            bits.append(rect(535 + col * 66, 135 + row * 65, 48, 48, color, INK, 2, 5))
    bits.extend([text(250, 420, "今天到本周：温度、风、雨、云", 20, INK, "middle"), text(710, 420, "通常至少数十年的分布与趋势", 20, INK, "middle"), text(480, 495, "一次寒潮不能否定变暖趋势，一次热天也不能单独证明趋势", 19, INK, "middle", 720)])
    return "".join(bits)


def _q59() -> str:
    bits = [panel_title("天气预报把地面、雷达、气球和卫星观测送入物理模型"), landscape(), rect(100, 250, 120, 160, "#AAB7BC", INK, 4, 6), line(160, 250, 160, 135, INK, 6), path("M135,170 Q160,120 185,170", "none", BLUE, 5), circle(345, 225, 55, WHITE, INK, 3), line(345, 280, 345, 405, INK, 4), path("M345,225 Q430,170 500,225", "none", BLUE, 5), earth(735, 335, 75)]
    bits.extend([rect(650, 120, 170, 70, "#B8C5CC", INK, 4, 8), rect(700, 88, 70, 35, BLUE, INK, 3, 4), line(735, 190, 735, 245, INK, 4), arrow("M735,205 Q735,255 735,270", TEAL, 4), arrow("M220,250 Q300,180 335,185", TEAL, 4), arrow("M500,225 Q600,200 650,155", TEAL, 4), text(160, 445, "地面站", 19, INK, "middle"), text(350, 445, "气球和雷达", 19, INK, "middle"), text(735, 445, "卫星", 19, INK, "middle"), text(480, 495, "模型从当前状态计算未来变化；时间越远，不确定性通常越大", 20, INK, "middle", 700)])
    return "".join(bits)


RENDERERS = {
    1: _q1, 2: _q2, 3: _q3, 4: _q4, 5: _q5, 6: _q6, 7: _q7, 8: _q8, 9: _q9,
    10: _q10, 11: _q11, 12: _q12, 13: _q13, 14: _q14, 15: _q15, 16: _q16, 17: _q17, 18: _q18,
    19: _q19, 20: _q20, 21: _q21, 22: _q22, 23: _q23, 24: _q24, 25: _q25,
    26: _q26, 27: _q27, 28: _q28, 29: _q29, 30: _q30, 31: _q31,
    32: _q32, 33: _q33, 34: _q34, 35: _q35, 36: _q36, 37: _q37,
    38: _q38, 39: _q39, 40: _q40, 41: _q41, 42: _q42, 43: _q43, 44: _q44, 45: _q45, 46: _q46,
    47: _q47, 48: _q48, 49: _q49, 50: _q50, 51: _q51, 52: _q52, 53: _q53, 54: _q54, 55: _q55, 56: _q56, 57: _q57, 58: _q58, 59: _q59,
}


def earth_weather_body(question: int, palette: tuple[str, ...]) -> str:
    del palette
    try:
        return RENDERERS[question]()
    except KeyError as exc:
        raise KeyError(f"no earth/weather illustration for question {question}") from exc
