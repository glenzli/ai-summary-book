#!/usr/bin/env python3
"""Concrete textbook-style plates for the human-body questions."""

from __future__ import annotations

import html


TEXTBOOK_QUESTIONS = frozenset(range(152, 182))

TEXTBOOK_KEYS: dict[int, tuple[str, ...]] = {
    152: ("skin-cross-section", "epidermis", "dermis", "subcutis"),
    153: ("fingerprint-ridges", "fingertip", "touch-receptors"),
    154: ("hair-follicle", "hair-shaft", "growth-zone"),
    155: ("long-bone-cutaway", "compact-bone", "spongy-bone", "marrow"),
    156: ("shoulder-joint", "knee-joint", "cartilage", "ligament"),
    157: ("arm-muscles", "biceps", "triceps", "tendon"),
    158: ("heart-cutaway", "four-chambers", "valves", "conduction-path"),
    159: ("circulation", "heart", "lungs", "body-capillaries"),
    160: ("lungs", "bronchial-tree", "alveoli", "capillaries"),
    161: ("breathing", "rib-cage", "diaphragm", "airflow"),
    162: ("hiccup", "diaphragm", "airway", "glottis"),
    163: ("brain-side-view", "cerebrum", "cerebellum", "brainstem"),
    164: ("withdrawal-reflex", "skin", "spinal-cord", "arm-muscle"),
    165: ("eye-cutaway", "cornea", "lens", "retina"),
    166: ("pupil-light-response", "iris", "small-pupil", "large-pupil"),
    167: ("tear-system", "lacrimal-gland", "tear-film", "tear-duct"),
    168: ("ear-cutaway", "eardrum", "ossicles", "cochlea"),
    169: ("balance-system", "eyes", "inner-ear", "muscle-sense"),
    170: ("nose-cutaway", "airflow", "olfactory-region", "brain"),
    171: ("tongue", "taste-buds", "five-tastes"),
    172: ("whole-tongue-taste", "distributed-receptors", "brain"),
    173: ("tooth-shapes", "incisor", "canine", "molar"),
    174: ("tooth-replacement", "baby-tooth", "adult-tooth", "jaw"),
    175: ("digestive-system", "mouth", "stomach", "intestines"),
    176: ("gut-movement", "stomach", "small-intestine", "peristalsis"),
    177: ("urinary-system", "kidneys", "ureters", "bladder"),
    178: ("sweating-skin", "sweat-gland", "skin-vessels", "evaporation"),
    179: ("goosebumps-skin", "hair", "arrector-muscle", "nerve"),
    180: ("handwashing", "soap", "oil-dirt", "running-water"),
    181: ("wound-healing", "clot", "scab", "new-skin"),
}


FONT = "system-ui, PingFang SC, Noto Sans CJK SC, sans-serif"
INK = "#24384A"
SKIN = "#F2B39C"
SKIN_DARK = "#D98478"
FAT = "#F3D56B"
BONE = "#F3E5CB"
BONE_DARK = "#C9A96E"
MUSCLE = "#C95B63"
ARTERY = "#D84F55"
VEIN = "#487DB8"
NERVE = "#E0B52E"
ORGAN = "#E58D99"


def esc(value: str) -> str:
    return html.escape(value, quote=True)


def path(d: str, fill: str = "none", stroke: str = "none", width: float = 0, marker: bool = False, dash: str = "") -> str:
    marker_attr = ' marker-end="url(#arrow)"' if marker else ""
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round"{marker_attr}{dash_attr}/>'


def line(x1: float, y1: float, x2: float, y2: float, stroke: str = INK, width: float = 4, marker: bool = False, dash: str = "") -> str:
    return path(f"M{x1},{y1} L{x2},{y2}", "none", stroke, width, marker, dash)


def rect(x: float, y: float, w: float, h: float, fill: str, stroke: str = "none", width: float = 0, radius: float = 0) -> str:
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>'


def circle(cx: float, cy: float, r: float, fill: str, stroke: str = "none", width: float = 0) -> str:
    return f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>'


def ellipse(cx: float, cy: float, rx: float, ry: float, fill: str, stroke: str = "none", width: float = 0) -> str:
    return f'<ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>'


def text(x: float, y: float, value: str, size: int = 23, color: str = INK, anchor: str = "start", weight: int = 650) -> str:
    return f'<text x="{x}" y="{y}" fill="{color}" text-anchor="{anchor}" font-family="{FONT}" font-size="{size}" font-weight="{weight}">{esc(value)}</text>'


def label(tx: float, ty: float, x: float, y: float, value: str, color: str = INK) -> str:
    end_x = x - 14 if x >= tx else x + 14
    return line(tx, ty, end_x, y - 7, color, 3) + circle(tx, ty, 5, color) + text(x, y, value, 22, color)


def panel_title(value: str) -> str:
    return text(54, 54, value, 26, INK, "start", 760)


def _skin_base() -> str:
    bits = [
        path("M105,112 Q190,96 275,112 T445,112 T615,112 L615,166 L105,166 Z", SKIN, INK, 3),
        rect(105, 166, 510, 205, "#ECA089", INK, 3),
        rect(105, 371, 510, 112, "#F7E5A6", INK, 3),
    ]
    for x, y, r in [(145, 407, 28), (198, 443, 34), (260, 405, 31), (327, 447, 36), (397, 407, 30), (468, 445, 34), (548, 408, 31)]:
        bits.append(circle(x, y, r, FAT, "#D7B84D", 2))
    bits.extend([
        path("M278,54 Q286,146 298,244 Q303,315 325,350", "none", "#433A35", 10),
        ellipse(326, 319, 37, 75, "#D87D73", INK, 3),
        ellipse(326, 336, 21, 37, "#9C514F", "none"),
        path("M504,326 C458,286 548,268 518,333 C492,373 463,334 500,302 C535,277 560,320 532,352", "none", "#A65E86", 7),
        path("M515,285 Q525,215 521,136", "none", "#A65E86", 5),
        path("M133,393 C238,349 385,427 577,380", "none", ARTERY, 7),
        path("M132,420 C250,377 405,451 578,405", "none", VEIN, 7),
        path("M420,392 Q433,317 458,260 Q470,232 462,204", "none", NERVE, 5),
        circle(462, 199, 12, "#F5D95C", INK, 2),
    ])
    return "".join(bits)


def skin_plate(question: int) -> str:
    bits = [panel_title({152: "皮肤是有层次的器官", 153: "纹脊把细小触碰传给感受器", 154: "头发从毛囊底部生长", 178: "汗液蒸发会带走热", 179: "立毛肌收缩会拉起毛发"}.get(question, "皮肤结构"))]
    if question == 153:
        bits.extend([
            path("M95,180 C120,85 300,70 370,165 C414,225 382,353 274,401 C175,445 79,365 77,271 C76,235 82,205 95,180 Z", "#F2B39C", INK, 4),
            path("M132,203 C188,135 313,137 346,225 C374,303 288,370 207,351 C130,333 104,257 132,203", "none", "#B8645E", 7),
            path("M166,220 C210,173 294,181 314,239 C335,300 272,334 220,317 C169,301 142,255 166,220", "none", "#B8645E", 7),
            path("M198,234 C224,209 274,214 286,250 C298,286 261,307 231,293 C201,279 184,251 198,234", "none", "#B8645E", 7),
            rect(500, 120, 310, 245, "#F7D0BD", INK, 3, 12),
            path("M500,181 Q550,147 600,181 T700,181 T810,181", "none", "#B8645E", 12),
            path("M500,224 Q550,190 600,224 T700,224 T810,224", "none", "#D98478", 8),
            path("M610,325 Q630,280 650,250", "none", NERVE, 6),
            path("M700,325 Q685,275 675,225", "none", NERVE, 6),
            circle(650, 249, 10, NERVE, INK, 2), circle(675, 225, 10, NERVE, INK, 2),
            label(260, 165, 115, 115, "指腹纹脊"),
            label(675, 225, 825, 205, "触觉感受器"),
            label(620, 305, 825, 300, "感觉神经"),
        ])
        return "".join(bits)

    bits.append(_skin_base())
    if question == 152:
        bits.extend([
            label(588, 135, 690, 132, "表皮：外层屏障"),
            label(586, 236, 690, 235, "真皮：血管、神经和腺体"),
            label(565, 425, 690, 424, "皮下组织：缓冲和连接"),
            label(520, 310, 690, 325, "汗腺"),
        ])
    elif question == 154:
        bits.extend([
            line(280, 95, 270, 48, MUSCLE, 6, True),
            label(285, 78, 690, 105, "露出皮肤的发丝"),
            label(327, 295, 690, 245, "毛囊"),
            label(326, 340, 690, 346, "底部生长区"),
            label(175, 395, 690, 440, "血管供应活组织"),
        ])
    elif question == 178:
        bits.extend([
            circle(521, 92, 10, "#63B7D1"), circle(544, 78, 8, "#63B7D1"), circle(500, 69, 7, "#63B7D1"),
            line(520, 132, 520, 96, "#4B9EB8", 5, True),
            label(520, 300, 690, 175, "汗腺制造汗液"),
            label(521, 78, 690, 270, "水分蒸发带走热"),
            label(475, 394, 690, 382, "皮肤血流也会增加"),
        ])
    elif question == 179:
        bits.extend([
            path("M327,280 L382,220", "none", MUSCLE, 10),
            line(382, 220, 349, 184, MUSCLE, 4, True),
            label(365, 235, 690, 165, "立毛肌收缩"),
            label(284, 78, 690, 270, "毛发被拉得更直"),
            label(458, 260, 690, 380, "神经信号触发反应"),
        ])
    return "".join(bits)


def wound_plate() -> str:
    bits = [panel_title("伤口表面结痂，下面继续修复")]
    bits.extend([
        rect(90, 130, 570, 55, SKIN, INK, 3), rect(90, 185, 570, 210, "#ECA089", INK, 3), rect(90, 395, 570, 90, "#F7E5A6", INK, 3),
        path("M330,129 L360,219 L391,129 Z", "#F6F7F5", INK, 3),
        path("M320,126 Q360,92 402,126 Q382,149 360,148 Q338,149 320,126 Z", "#7F4C3D", INK, 3),
        path("M316,228 Q360,200 406,228 Q387,268 360,273 Q335,264 316,228 Z", "#D85F62", "none"),
        path("M126,360 C220,319 486,424 620,348", "none", ARTERY, 7),
        circle(330, 258, 9, "#F3F0DB", INK, 1), circle(360, 286, 9, "#F3F0DB", INK, 1), circle(392, 252, 9, "#F3F0DB", INK, 1),
        label(365, 111, 710, 125, "血块干燥形成痂"),
        label(365, 238, 710, 245, "免疫细胞清理碎屑"),
        label(360, 286, 710, 355, "新组织从下面填补"),
        label(542, 360, 710, 445, "血流运来修复材料"),
    ])
    return "".join(bits)


def bone_plate(question: int) -> str:
    if question == 155:
        bits = [panel_title("长骨不是实心石棒，也不是空吸管")]
        bits.extend([
            path("M170,245 Q190,130 275,135 L565,135 Q650,130 670,245 Q650,360 565,355 L275,355 Q190,360 170,245 Z", BONE, INK, 5),
            path("M255,190 Q290,165 335,183 L500,183 Q550,165 585,190 L585,300 Q550,325 500,307 L335,307 Q290,325 255,300 Z", "#D8BE8B", "none"),
            rect(336, 197, 164, 96, "#E8A54D", "#B97834", 3, 45),
        ])
        for x1, y1, x2, y2 in [(205,210,275,280),(205,280,280,205),(570,205,640,280),(568,282,642,210),(230,177,292,238),(548,252,622,325)]:
            bits.append(line(x1,y1,x2,y2,BONE_DARK,5))
        bits.extend([
            label(217, 150, 700, 145, "致密骨：坚硬外壳"),
            label(248, 236, 700, 245, "海绵骨：轻而有支架"),
            label(420, 245, 700, 345, "骨髓空间"),
            label(603, 315, 700, 445, "骨内也有血管和活细胞"),
        ])
        return "".join(bits)
    if question == 156:
        bits = [panel_title("不同关节用不同形状换取灵活或稳定")]
        bits.extend([
            text(235, 105, "肩：球窝关节", 23, INK, "middle", 700),
            circle(225, 235, 72, BONE, INK, 4),
            path("M95,365 Q115,235 165,207 Q126,270 144,355 Z", BONE, INK, 4),
            path("M255,285 Q310,306 345,390", "none", BONE_DARK, 48),
            path("M159,205 Q176,184 197,180", "none", "#7EC1C0", 18),
            path("M540,140 Q580,115 620,145 L620,250 Q582,265 544,245 Z", BONE, INK, 4),
            path("M545,270 Q585,250 625,270 L655,435 Q590,465 525,435 Z", BONE, INK, 4),
            ellipse(582, 258, 55, 18, "#7EC1C0", INK, 2),
            path("M515,184 Q474,251 521,344", "none", "#E7C45C", 12),
            path("M644,184 Q687,251 643,345", "none", "#E7C45C", 12),
            text(585, 105, "膝：铰链样关节", 23, INK, "middle", 700),
            line(405, 105, 405, 460, "#C9D5DC", 3, False, "8 9"),
            label(165, 201, 710, 145, "软骨覆盖骨端"),
            label(515, 270, 710, 250, "韧带限制过度移动"),
            label(585, 258, 710, 355, "关节面彼此配合"),
            label(625, 410, 710, 445, "周围肌肉主动稳定"),
        ])
        return "".join(bits)
    bits = [panel_title("肌肉收缩，经肌腱拉动骨头")]
    bits.extend([
        path("M205,150 Q260,120 327,151 L450,271 Q470,295 449,320 L417,336 Q392,346 371,322 L270,224 Q237,206 205,215 Z", "#F5C6AD", INK, 4),
        path("M283,187 Q338,164 387,220 Q419,258 395,298 Q340,275 296,228 Z", MUSCLE, "#983E49", 3),
        path("M254,211 Q300,253 371,321", "none", BONE_DARK, 22),
        path("M291,199 Q300,242 319,257", "none", "#EEE4C7", 10),
        path("M350,203 Q391,209 414,270", "none", "#A74752", 19),
        line(343, 180, 327, 118, MUSCLE, 7, True),
        path("M450,319 Q530,360 615,337", "none", "#F5C6AD", 55),
        path("M435,319 Q525,351 610,331", "none", BONE_DARK, 18),
        label(343, 205, 690, 145, "肱二头肌收缩变短"),
        label(396, 240, 690, 245, "肱三头肌配合稳定"),
        label(308, 246, 690, 345, "肌腱连接肌肉和骨"),
        label(440, 322, 690, 445, "关节角度随拉力改变"),
    ])
    return "".join(bits)


def heart_plate(question: int) -> str:
    if question == 159:
        bits = [panel_title("血液在心、肺和全身之间形成闭环")]
        bits.extend([
            ellipse(480, 275, 86, 110, ORGAN, INK, 4),
            ellipse(280, 170, 88, 70, "#F2A6B5", INK, 3), ellipse(680, 170, 88, 70, "#F2A6B5", INK, 3),
            path("M423,250 C350,235 350,180 365,170", "none", VEIN, 12, True),
            path("M595,170 C610,222 568,245 536,255", "none", ARTERY, 12, True),
            path("M510,377 C590,425 706,437 795,371", "none", ARTERY, 13, True),
            path("M786,345 C692,312 591,337 530,318", "none", VEIN, 13, True),
            circle(800, 355, 74, "#F7D9BE", INK, 3),
            text(480, 282, "心", 36, "#FFFFFF", "middle", 760),
            text(280, 177, "肺", 29, INK, "middle", 740), text(680, 177, "肺", 29, INK, "middle", 740),
            text(800, 363, "全身", 25, INK, "middle", 700),
            label(360, 205, 100, 130, "缺氧血到肺"),
            label(593, 205, 745, 115, "富氧血回心"),
            label(686, 417, 740, 475, "动脉送出，静脉送回"),
        ])
        return "".join(bits)
    bits = [panel_title("心脏四个腔室按顺序收缩")]
    bits.extend([
        path("M350,132 C262,86 187,168 214,270 C237,355 344,424 430,470 C516,423 623,352 650,270 C679,174 602,88 516,132 C477,151 452,179 430,207 C406,177 385,150 350,132 Z", "#D9636C", INK, 5),
        path("M430,207 L430,446", "none", "#F5C6C6", 9),
        path("M252,230 Q330,200 425,235", "none", "#F5C6C6", 7),
        path("M435,235 Q531,200 610,230", "none", "#F5C6C6", 7),
        path("M290,250 Q335,292 409,276", "none", VEIN, 6, True),
        path("M570,252 Q525,291 453,276", "none", ARTERY, 6, True),
        circle(330, 170, 13, "#F2C94C", INK, 2),
        path("M330,170 Q376,201 358,255 Q383,303 430,336 Q488,304 507,252 Q489,206 535,174", "none", "#F2C94C", 7, True),
        path("M376,284 l24,-12 l-10,25 Z M485,282 l-24,-12 l10,25 Z", "#F7E7C7", INK, 2),
        label(330, 170, 710, 120, "起搏点发出电信号"),
        label(390, 276, 710, 230, "瓣膜控制主要流向"),
        label(330, 348, 710, 340, "右心把血送往肺"),
        label(530, 348, 710, 445, "左心把血送往全身"),
    ])
    return "".join(bits)


def lung_plate(question: int) -> str:
    if question in {161, 162}:
        bits = [panel_title("吸气和呼气由胸腔与膈肌共同完成" if question == 161 else "打嗝是膈肌突然收缩后的反射")]
        centers = [280, 650] if question == 161 else [440]
        for index, cx in enumerate(centers):
            bits.extend([
                path(f"M{cx-100},132 Q{cx-145},220 {cx-105},365 Q{cx},420 {cx+105},365 Q{cx+145},220 {cx+100},132", "#F7D9C7", INK, 4),
                ellipse(cx-50, 245, 48, 100, "#F2A6B5", INK, 3), ellipse(cx+50, 245, 48, 100, "#F2A6B5", INK, 3),
                line(cx, 105, cx, 224, "#7DB3C8", 14),
            ])
            diaphragm_y = 350 if question == 162 or index == 0 else 326
            curve = f"M{cx-110},{diaphragm_y} Q{cx},{diaphragm_y+55 if question == 162 or index == 0 else diaphragm_y-35} {cx+110},{diaphragm_y}"
            bits.append(path(curve, "none", MUSCLE, 14))
        if question == 161:
            bits.extend([
                line(280, 80, 280, 128, VEIN, 7, True), line(650, 130, 650, 82, VEIN, 7, True),
                text(280, 470, "吸气：膈肌下降", 23, INK, "middle", 700),
                text(650, 470, "呼气：膈肌回升", 23, INK, "middle", 700),
                line(465, 90, 465, 470, "#CBD8DE", 3, False, "9 10"),
            ])
        else:
            bits.extend([
                line(440, 75, 440, 132, VEIN, 8, True),
                path("M417,122 Q440,105 463,122", "none", MUSCLE, 7),
                text(440, 100, "声门突然关闭", 21, INK, "middle", 700),
                label(440, 370, 700, 230, "膈肌不自主收缩"),
                label(440, 121, 700, 340, "快速进气后发出“嗝”声"),
            ])
        return "".join(bits)

    bits = [panel_title("肺里是不断分叉的气道和大量肺泡")]
    bits.extend([
        line(370, 80, 370, 195, "#76AFC4", 20),
        path("M370,180 Q308,223 286,312 M370,180 Q432,223 454,312", "none", "#76AFC4", 15),
        path("M364,205 Q324,250 305,340 M376,205 Q416,250 435,340", "none", "#76AFC4", 8),
        path("M340,132 Q225,145 218,290 Q220,414 340,420 Q368,354 360,210 Z", "#F2A6B5", INK, 4),
        path("M400,132 Q515,145 522,290 Q520,414 400,420 Q372,354 380,210 Z", "#F2A6B5", INK, 4),
        circle(690, 285, 54, "#F7C9D1", INK, 3), circle(747, 244, 48, "#F7C9D1", INK, 3), circle(765, 315, 50, "#F7C9D1", INK, 3), circle(704, 350, 48, "#F7C9D1", INK, 3),
        path("M640,190 C706,156 817,210 818,302 C819,385 710,420 641,365", "none", ARTERY, 8),
        path("M648,210 C706,181 789,222 792,297 C793,354 719,390 652,353", "none", VEIN, 8),
        line(518, 315, 624, 315, INK, 3, False, "8 8"),
        label(370, 130, 70, 105, "气管"),
        label(308, 275, 70, 255, "支气管反复分叉"),
        label(744, 245, 830, 190, "肺泡"),
        label(790, 330, 830, 355, "紧贴毛细血管"),
        text(720, 465, "局部放大", 21, INK, "middle", 650),
    ])
    return "".join(bits)


def brain_plate(question: int) -> str:
    if question == 164:
        bits = [panel_title("缩手反射先在脊髓形成快速回路")]
        bits.extend([
            path("M105,345 Q180,280 285,312 Q332,324 356,296", "none", "#F3C5AC", 45),
            circle(92, 346, 27, "#F3C5AC", INK, 3),
            path("M78,335 l-22,-18 l15,30 Z", "#F08B3E", "none"),
            rect(445, 118, 64, 296, "#F4E8CB", INK, 4, 28),
            path("M477,128 Q561,83 629,135 Q672,190 628,245 Q574,267 509,235", "#EFAE9E", INK, 4),
            path("M100,320 C205,245 335,235 465,278", "none", NERVE, 9, True),
            path("M466,296 C352,348 282,389 180,373", "none", MUSCLE, 9, True),
            path("M480,260 Q518,198 557,179", "none", "#7B62B3", 6, True, "8 7"),
            label(101, 320, 65, 110, "皮肤感到危险热度"),
            label(475, 278, 650, 190, "脊髓快速连接"),
            label(280, 389, 650, 310, "运动神经让肌肉缩手"),
            label(557, 179, 650, 420, "信号也上传大脑"),
        ])
        return "".join(bits)
    bits = [panel_title("大脑各部分分工，又通过网络合作")]
    bits.extend([
        path("M238,164 C200,85 325,62 425,104 C507,60 627,109 623,204 C682,248 639,343 570,350 C535,419 427,422 378,366 C293,402 204,351 222,280 C168,238 184,187 238,164 Z", "#EFAE9E", INK, 5),
        path("M545,338 Q610,338 630,392 Q586,437 525,398 Z", "#D8848F", INK, 4),
        path("M486,348 Q500,401 493,470", "none", "#B16C78", 24),
        path("M281,150 Q338,202 301,260 M376,115 Q426,169 398,231 M487,116 Q535,170 507,238 M254,292 Q338,276 381,341 M451,268 Q536,250 586,300", "none", "#C77780", 5),
        label(420, 132, 710, 130, "大脑：感觉、记忆与动作计划"),
        label(577, 379, 710, 245, "小脑：动作协调和平衡"),
        label(495, 385, 710, 350, "脑干：连接许多自动调节"),
        label(493, 459, 710, 450, "脊髓：连接身体"),
    ])
    return "".join(bits)


def eye_plate(question: int) -> str:
    if question == 166:
        bits = [panel_title("虹膜调节瞳孔大小，不是瞳孔自己伸缩")]
        for cx, pupil, rays, label_text in [(290, 38, 5, "亮处：瞳孔较小"), (650, 78, 2, "暗处：瞳孔较大")]:
            bits.append(circle(cx, 260, 142, "#F7F7F2", INK, 4))
            bits.append(circle(cx, 260, 106, "#6FA9A2", "#356B6C", 5))
            bits.append(circle(cx, 260, pupil, "#172839"))
            for index in range(rays):
                y = 125 + index * 25
                bits.append(line(cx - 210, y, cx - 150, y, "#F2B84B", 5, True))
            bits.append(text(cx, 465, label_text, 23, INK, "middle", 700))
        return "".join(bits)
    if question == 167:
        bits = [panel_title("眼泪持续保护眼球表面")]
        bits.extend([
            ellipse(420, 270, 238, 128, "#F7F7F2", INK, 5),
            circle(420, 270, 75, "#6FA9A2", INK, 4), circle(420, 270, 31, "#172839"),
            path("M232,183 Q202,135 260,120 Q304,139 278,181 Z", "#E9A4B6", INK, 3),
            path("M262,153 C323,130 450,138 576,201", "none", "#59ADD0", 7, True),
            path("M598,248 Q618,286 596,334 Q575,362 566,401", "none", "#59ADD0", 7, True),
            label(245, 148, 690, 135, "泪腺制造泪液"),
            label(430, 147, 690, 245, "眨眼铺开泪膜"),
            label(586, 340, 690, 355, "泪液流向鼻泪管"),
            label(420, 270, 690, 450, "冲走小颗粒并保持湿润"),
        ])
        return "".join(bits)
    bits = [panel_title("光经角膜和晶状体聚焦到视网膜")]
    bits.extend([
        path("M335,125 C520,105 654,178 676,270 C654,362 520,435 335,415 C238,400 194,340 194,270 C194,200 238,140 335,125 Z", "#F5F6F3", INK, 5),
        path("M225,181 Q163,270 225,359", "#BFE4EC", "#4C91A6", 5),
        ellipse(310, 270, 28, 97, "#E7DCC1", INK, 3),
        ellipse(268, 270, 60, 112, "none", "#6FA9A2", 16),
        path("M628,174 Q681,270 628,366", "none", "#D27B85", 12),
        line(60, 200, 225, 230, "#F0B43C", 5, True), line(60, 340, 225, 310, "#F0B43C", 5, True),
        line(225, 230, 628, 284, "#F0B43C", 4, True), line(225, 310, 628, 256, "#F0B43C", 4, True),
        path("M676,270 Q737,261 780,236", "none", NERVE, 13),
        label(210, 270, 730, 120, "角膜先折射光"),
        label(310, 270, 730, 220, "晶状体继续聚焦"),
        label(628, 270, 730, 330, "视网膜感光"),
        label(745, 252, 730, 440, "视神经送往大脑"),
    ])
    return "".join(bits)


def ear_plate() -> str:
    bits = [panel_title("声音依次经过外耳、中耳和内耳")]
    bits.extend([
        path("M176,135 C84,166 78,350 180,399 C256,420 289,349 257,309 C228,278 185,312 182,270 C179,224 239,226 250,181 C251,144 217,124 176,135 Z", "#F2B39C", INK, 5),
        path("M252,270 Q344,250 390,270", "none", "#A96F5A", 28),
        path("M390,216 L390,324", "none", "#D46A72", 10),
        path("M404,253 l32,-25 l22,31 l29,-18", "none", BONE_DARK, 10),
        path("M526,231 C616,170 670,260 607,313 C555,355 489,296 526,231 Z", "none", "#8E74B5", 18),
        path("M550,204 Q526,150 562,125 M580,202 Q585,145 620,132 M612,218 Q648,174 677,196", "none", "#6FA9A2", 11),
        path("M623,300 Q700,325 764,287", "none", NERVE, 12),
        line(48, 270, 132, 270, "#4B9EB8", 7, True),
        label(193, 188, 70, 100, "外耳收集声音"),
        label(390, 270, 710, 115, "鼓膜振动"),
        label(451, 246, 710, 220, "三块听小骨传力"),
        label(567, 263, 710, 330, "耳蜗把振动变成神经信号"),
        label(605, 170, 710, 445, "半规管参与平衡"),
    ])
    return "".join(bits)


def balance_plate() -> str:
    bits = [panel_title("大脑同时比较视觉、内耳和身体感觉")]
    bits.extend([
        circle(480, 155, 62, "#F2B39C", INK, 4),
        path("M430,217 Q480,198 530,217 L570,390 Q480,430 390,390 Z", "#75A9C9", INK, 4),
        line(425, 265, 302, 340, "#F2B39C", 27), line(535, 265, 658, 340, "#F2B39C", 27),
        line(438, 390, 395, 485, "#465F85", 31), line(522, 390, 565, 485, "#465F85", 31),
        circle(465, 150, 6, INK), circle(495, 150, 6, INK),
        path("M507,135 q35,-30 55,7 q17,31 -16,49", "none", "#8E74B5", 9),
        path("M468,175 Q485,191 502,175", "none", INK, 3),
        path("M440,125 Q480,90 520,125", "none", "#EFAE9E", 22),
        line(480, 130, 480, 78, NERVE, 5, True),
        line(550, 160, 686, 118, "#8E74B5", 5, True),
        line(398, 470, 262, 438, "#3F8C74", 5, True),
        label(480, 98, 715, 105, "眼睛看到周围方向"),
        label(555, 150, 715, 210, "内耳感受转动和加速度"),
        label(397, 470, 715, 320, "脚底、肌肉和关节报告姿势"),
        label(480, 125, 715, 430, "大脑综合后调整肌肉"),
    ])
    return "".join(bits)


def smell_plate() -> str:
    bits = [panel_title("气味分子随空气到达鼻腔上方")]
    bits.extend([
        path("M232,94 Q400,78 500,168 Q552,215 518,259 Q478,291 443,308 Q423,352 459,401 Q340,454 220,397 Q152,339 166,244 Q177,148 232,94 Z", "#F3C5AC", INK, 5),
        path("M300,168 Q396,147 467,217 Q429,246 342,252 Q292,255 266,292", "#F8D9CB", "#B67567", 4),
        path("M338,154 Q409,142 468,198", "none", "#D8A53D", 18),
        path("M369,151 Q383,115 420,91", "none", NERVE, 8, True),
        path("M80,240 C135,225 191,230 275,252", "none", "#6CB0C5", 7, True),
    ])
    for x, y in [(95, 212), (126, 259), (168, 218), (205, 270)]:
        bits.append(circle(x, y, 7, "#7F9D55"))
    bits.extend([
        label(156, 228, 690, 130, "带气味的空气进入"),
        label(382, 154, 690, 235, "嗅觉感受区"),
        label(419, 93, 690, 340, "神经信号送往大脑"),
        label(338, 250, 690, 445, "鼻腔还会加温、加湿空气"),
    ])
    return "".join(bits)


def tongue_plate(question: int) -> str:
    bits = [panel_title("味觉感受器分布在舌面许多区域")]
    bits.extend([
        path("M172,150 Q360,85 557,155 Q642,244 566,398 Q477,487 307,458 Q143,420 128,285 Q125,204 172,150 Z", "#E88B97", INK, 5),
    ])
    colors = ["#E3B53D", "#7CB069", "#DA6B62", "#6B80B4", "#9A6BB1"]
    positions = [(225,210),(310,165),(410,190),(510,220),(200,300),(300,275),(410,305),(520,315),(265,380),(380,395),(492,380)]
    for index,(x,y) in enumerate(positions): bits.append(circle(x,y,13,colors[index%len(colors)],"#FFFFFF",3))
    bits.extend([
        circle(720, 255, 92, "#F5A8B1", INK, 4),
        path("M675,260 Q720,205 765,260 Q720,318 675,260 Z", "#D86E7D", INK, 3),
        circle(700,250,7,"#F8D95C"),circle(720,237,7,"#7CB069"),circle(741,251,7,"#6B80B4"),
        line(565, 300, 620, 280, INK, 3, False, "7 7"),
        label(312, 166, 670, 115, "不同味觉感受器广泛分布"),
        label(720, 255, 805, 235, "味蕾局部放大"),
        label(720, 285, 805, 345, "甜、酸、咸、苦、鲜都要靠大脑辨认"),
        text(365, 492, "彩色点只表示多种受体混合分布，不是舌头分区地图", 19, INK, "middle", 560),
    ])
    if question == 171:
        bits.append(text(365, 80, "五种基本味觉并非五块专属区域", 21, "#A54E58", "middle", 700))
    return "".join(bits)


def teeth_plate(question: int) -> str:
    if question == 174:
        bits = [panel_title("恒牙在乳牙下面生长并逐渐替换")]
        bits.extend([
            path("M130,190 Q330,110 560,190 L560,430 Q340,500 130,430 Z", "#EBAF9F", INK, 4),
            path("M185,170 Q215,118 245,170 L238,275 Q215,305 192,275 Z", "#F8F3E8", INK, 3),
            path("M330,165 Q365,105 400,165 L392,280 Q365,313 338,280 Z", "#F8F3E8", INK, 3),
            path("M190,350 Q220,285 250,350 L245,435 Q220,458 195,435 Z", "#F3E5CB", INK, 3),
            path("M325,355 Q370,275 415,355 L405,448 Q370,470 335,448 Z", "#F3E5CB", INK, 3),
            line(220, 330, 220, 285, BONE_DARK, 6, True), line(370, 335, 370, 288, BONE_DARK, 6, True),
            label(216, 165, 680, 130, "乳牙"),
            label(220, 365, 680, 245, "下面的恒牙逐渐长大"),
            label(220, 285, 680, 355, "乳牙根逐渐被吸收"),
            label(370, 435, 680, 445, "恒牙最后进入牙列"),
        ])
        return "".join(bits)
    bits = [panel_title("门牙、犬牙和磨牙形状对应不同任务")]
    shapes = [
        (180, "门牙", "M135,168 Q180,135 225,168 L217,335 Q180,365 143,335 Z"),
        (380, "犬牙", "M330,170 Q380,126 430,170 L408,372 Q380,410 352,372 Z"),
        (620, "磨牙", "M545,185 Q575,135 608,172 Q640,130 675,172 Q710,140 735,190 L712,350 Q640,395 568,350 Z"),
    ]
    for cx,name,d in shapes:
        bits.append(path(d,"#F8F3E8",INK,4)); bits.append(text(cx,450,name,25,INK,"middle",720))
    bits.extend([
        text(180, 493, "切", 21, "#A56A42", "middle", 650),
        text(380, 493, "撕", 21, "#A56A42", "middle", 650),
        text(640, 493, "磨碎", 21, "#A56A42", "middle", 650),
        label(180, 170, 75, 105, "较薄的边缘"),
        label(380, 175, 760, 180, "较尖的牙尖"),
        label(640, 180, 760, 315, "宽阔、有多个牙尖"),
    ])
    return "".join(bits)


def digestive_plate(question: int) -> str:
    bits = [panel_title("食物沿消化道移动并逐步分解" if question == 175 else "空的胃肠仍会有规律收缩和移动气体")]
    bits.extend([
        circle(330, 105, 48, "#F3C5AC", INK, 3),
        path("M279,153 Q330,130 381,153 L420,475 Q330,515 240,475 Z", "#F5D6C4", INK, 4),
        line(330, 135, 330, 240, "#B98372", 13),
        path("M330,235 C425,220 456,292 395,340 C345,367 304,333 310,280 Z", "#E08A91", INK, 4),
        path("M290,346 C230,365 246,464 320,458 C378,452 382,389 334,382 C289,376 283,425 328,431 C389,439 437,393 412,350", "none", "#C77A62", 19),
        path("M274,350 Q220,394 258,461 Q330,500 402,461 Q444,395 391,350", "none", "#9E6A56", 13),
        line(330, 235, 330, 184, "#70A9C1", 6, True),
        label(330, 205, 650, 125, "食管把食团送到胃"),
        label(388, 280, 650, 225, "胃搅拌并加入消化液"),
        label(330, 400, 650, 330, "小肠继续分解并吸收"),
        label(270, 445, 650, 435, "大肠回收水分并形成粪便"),
    ])
    if question == 176:
        bits.extend([
            path("M285,355 Q330,335 375,355", "none", "#F0B43C", 7, True),
            path("M372,407 Q330,427 288,407", "none", "#F0B43C", 7, True),
            text(495, 495, "咕噜声来自液体、气体和肠壁运动", 20, INK, "middle", 600),
        ])
    return "".join(bits)


def urinary_plate() -> str:
    bits = [panel_title("肾脏过滤血液并调节水、盐和废物")]
    bits.extend([
        path("M255,150 C173,135 155,244 195,314 C220,355 286,334 300,275 C312,221 313,165 255,150 Z", "#B75D64", INK, 4),
        path("M515,150 C597,135 615,244 575,314 C550,355 484,334 470,275 C458,221 457,165 515,150 Z", "#B75D64", INK, 4),
        path("M258,312 Q286,380 328,438", "none", "#E8C25B", 9),
        path("M512,312 Q484,380 442,438", "none", "#E8C25B", 9),
        path("M328,438 Q385,480 442,438 Q442,498 385,511 Q328,498 328,438 Z", "#E6B85A", INK, 4),
        path("M80,220 Q160,200 206,220", "none", ARTERY, 10, True), path("M560,220 Q640,200 710,220", "none", VEIN, 10, True),
        label(245, 205, 700, 125, "两侧肾脏"),
        label(285, 350, 700, 235, "输尿管把尿液送下去"),
        label(385, 455, 700, 345, "膀胱暂时储存"),
        label(130, 210, 700, 450, "有用物质大多留在血液中"),
    ])
    return "".join(bits)


def handwash_plate() -> str:
    bits = [panel_title("肥皂、揉搓和流水一起带走污物")]
    bits.extend([
        path("M135,305 Q170,230 225,260 L246,177 Q257,150 275,174 L275,240 L300,142 Q312,118 330,146 L325,238 L354,157 Q368,132 384,162 L370,250 L397,204 Q415,177 430,207 L402,315 Q380,393 297,430 Q210,421 154,374 Z", "#F3C5AC", INK, 4),
        path("M518,305 Q553,230 608,260 L629,177 Q640,150 658,174 L658,240 L683,142 Q695,118 713,146 L708,238 L737,157 Q751,132 767,162 L753,250 L780,204 Q798,177 813,207 L785,315 Q763,393 680,430 Q593,421 537,374 Z", "#F3C5AC", INK, 4),
    ])
    for x,y in [(205,290),(270,330),(347,282),(610,300),(681,330),(748,275)]: bits.append(circle(x,y,10,"#82684F"))
    for x,y,r in [(560,210,18),(598,188,13),(636,217,16),(686,198,15),(731,224,17),(778,252,12)]: bits.append(circle(x,y,r,"#DFF3F7","#65AFC4",3))
    path_bits = path("M455,80 Q480,130 455,180 Q430,130 455,80 Z", "#58B2D1", "none") + line(455,180,455,320,"#58B2D1",12,True)
    bits.extend([
        path_bits,
        text(270, 485, "揉搓松动污物", 22, INK, "middle", 700),
        text(680, 485, "肥皂包住油污，流水带走", 22, INK, "middle", 700),
        line(470,270,520,270,INK,6,True),
    ])
    return "".join(bits)


def textbook_body(question: int, palette: tuple[str, ...]) -> str:
    del palette
    if question in {152, 153, 154, 178, 179}:
        return skin_plate(question)
    if question == 181:
        return wound_plate()
    if question in {155, 156, 157}:
        return bone_plate(question)
    if question in {158, 159}:
        return heart_plate(question)
    if question in {160, 161, 162}:
        return lung_plate(question)
    if question in {163, 164}:
        return brain_plate(question)
    if question in {165, 166, 167}:
        return eye_plate(question)
    if question == 168:
        return ear_plate()
    if question == 169:
        return balance_plate()
    if question == 170:
        return smell_plate()
    if question in {171, 172}:
        return tongue_plate(question)
    if question in {173, 174}:
        return teeth_plate(question)
    if question in {175, 176}:
        return digestive_plate(question)
    if question == 177:
        return urinary_plate()
    if question == 180:
        return handwash_plate()
    raise KeyError(f"no textbook illustration for question {question}")
