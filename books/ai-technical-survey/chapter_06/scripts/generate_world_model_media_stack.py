from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "images" / "world_model_media_stack.svg"
W, H = 1180, 700


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
    text(W / 2, 46, "World Models and Generative Media", 30, 700),
    text(W / 2, 76, "Generation quality, state prediction, and action feedback are related but not identical", 16, 400, "#475569"),
]

parts.append(rect(75, 150, 300, 430, "#ffffff", "#cbd5e1", 1.4, 10))
parts.append(text(225, 185, "Generative media", 23, 700))
media = [
    ("Text prompt", "#fef3c7", "#d97706"),
    ("Image / video encoder", "#dbeafe", "#2563eb"),
    ("Diffusion / Flow / DiT", "#ede9fe", "#7c3aed"),
    ("Pixels / frames", "#fee2e2", "#dc2626"),
]
for i, (label, fill, stroke) in enumerate(media):
    y = 235 + i * 75
    parts.append(rect(125, y, 200, 46, fill, stroke, 1.5, 8))
    parts.append(text(225, y + 29, label, 15, 700, stroke))
    if i < len(media) - 1:
        parts.append(arrow(225, y + 46, 225, y + 75))

parts.append(rect(440, 150, 300, 430, "#ffffff", "#cbd5e1", 1.4, 10))
parts.append(text(590, 185, "World model", 23, 700))
wm = [
    ("Observation o_t", "#dbeafe", "#2563eb"),
    ("Latent state z_t", "#ecfeff", "#0891b2"),
    ("Dynamics F(z,a)", "#fef3c7", "#d97706"),
    ("Predicted future", "#fee2e2", "#dc2626"),
]
for i, (label, fill, stroke) in enumerate(wm):
    y = 235 + i * 75
    parts.append(rect(490, y, 200, 46, fill, stroke, 1.5, 8))
    parts.append(text(590, y + 29, label, 15, 700, stroke))
    if i < len(wm) - 1:
        parts.append(arrow(590, y + 46, 590, y + 75))

parts.append(rect(805, 150, 300, 430, "#ffffff", "#cbd5e1", 1.4, 10))
parts.append(text(955, 185, "Agent loop", 23, 700))
loop = [
    ("Plan", "#ede9fe", "#7c3aed"),
    ("Act", "#fef3c7", "#d97706"),
    ("Observe", "#dbeafe", "#2563eb"),
    ("Update belief", "#ecfeff", "#0891b2"),
]
for i, (label, fill, stroke) in enumerate(loop):
    y = 235 + i * 75
    parts.append(rect(855, y, 200, 46, fill, stroke, 1.5, 8))
    parts.append(text(955, y + 29, label, 15, 700, stroke))
    if i < len(loop) - 1:
        parts.append(arrow(955, y + 46, 955, y + 75))

parts.append(arrow(375, 365, 440, 365, "#64748b", 2))
parts.append(arrow(740, 365, 805, 365, "#64748b", 2))
parts.append(text(407, 345, "state", 13, 500, "#64748b"))
parts.append(text(772, 345, "action", 13, 500, "#64748b"))
parts.append(rect(245, 620, 690, 42, "#ffffff", "#94a3b8", 1.2, 8))
parts.append(text(590, 647, "Video realism can help world modeling, but it is not sufficient evidence of physical understanding.", 16, 600, "#334155"))
parts.append("</svg>\n")

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(parts), encoding="utf-8")
print(OUT)
