from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "images" / "post_training_taxonomy.svg"
W, H = 1180, 650


def rect(x, y, w, h, fill, stroke="#334155", sw=1.4, rx=8):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'


def text(x, y, s, size=18, weight=500, fill="#0f172a", anchor="middle"):
    return f'<text x="{x}" y="{y}" font-family="Arial, sans-serif" font-size="{size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{s}</text>'


def arrow(x1, y1, x2, y2, color="#334155", sw=2):
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}" marker-end="url(#arrow)"/>'


parts = [
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
    "<defs>",
    '<marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">',
    '<path d="M0,0 L0,6 L9,3 z" fill="#334155"/>',
    "</marker>",
    "</defs>",
    rect(0, 0, W, H, "#f8fafc", "none", 0, 0),
    text(W / 2, 46, "Modern Post-Training Stack", 30, 700),
    text(W / 2, 76, "Instruction, preference, reasoning, tool use, and safety form a closed loop", 16, 400, "#475569"),
]

steps = [
    (70, 180, "Base model", "pretrained next-token model", "#dbeafe", "#2563eb"),
    (285, 180, "SFT", "instruction following", "#ecfeff", "#0891b2"),
    (500, 180, "Preference", "RLHF / DPO / IPO", "#fef3c7", "#d97706"),
    (715, 180, "Reasoning RL", "verifiable rewards", "#fee2e2", "#dc2626"),
    (930, 180, "Tool and safety", "formats, refusal, red team", "#ede9fe", "#7c3aed"),
]

for i, (x, y, title, subtitle, fill, stroke) in enumerate(steps):
    parts.append(rect(x, y, 175, 95, fill, stroke, 1.8, 10))
    parts.append(text(x + 87.5, y + 38, title, 19, 700, stroke))
    parts.append(text(x + 87.5, y + 65, subtitle, 13, 500, "#334155"))
    if i < len(steps) - 1:
        parts.append(arrow(x + 175, y + 48, steps[i + 1][0], y + 48))

parts.append(rect(135, 385, 910, 145, "#ffffff", "#cbd5e1", 1.4, 10))
parts.append(text(590, 420, "Evaluation and data iteration", 22, 700))
parts.append(text(290, 465, "math / code", 15, 700, "#1d4ed8"))
parts.append(text(450, 465, "facts / citation", 15, 700, "#155e75"))
parts.append(text(610, 465, "format / tools", 15, 700, "#92400e"))
parts.append(text(770, 465, "safety / refusal", 15, 700, "#991b1b"))
parts.append(text(930, 465, "latency / cost", 15, 700, "#5b21b6"))
parts.append(text(590, 505, "Failures generate new data; new data updates SFT, preferences, rewards, and policies.", 15, 500, "#334155"))
parts.append(arrow(590, 385, 590, 290, "#64748b", 2))
parts.append(arrow(930, 275, 930, 385, "#64748b", 2))
parts.append("</svg>\n")

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(parts), encoding="utf-8")
print(OUT)
