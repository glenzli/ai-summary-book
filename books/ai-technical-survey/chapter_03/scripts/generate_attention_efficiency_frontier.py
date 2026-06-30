from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "images" / "attention_efficiency_frontier.svg"

W = 1200
H = 760


def rect(x, y, w, h, fill, stroke="#1f2937", sw=1.2, rx=0, extra=""):
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}" {extra}/>'
    )


def text(x, y, content, size=20, weight=500, fill="#111827", anchor="middle"):
    return (
        f'<text x="{x}" y="{y}" font-family="Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}" '
        f'text-anchor="{anchor}">{content}</text>'
    )


def line(x1, y1, x2, y2, stroke="#334155", sw=2, dash=None):
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
        f'stroke="{stroke}" stroke-width="{sw}"{dash_attr}/>'
    )


def arrow(x1, y1, x2, y2, stroke="#334155", sw=2):
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
        f'stroke="{stroke}" stroke-width="{sw}" marker-end="url(#arrow)"/>'
    )


def matrix(x, y, n, cell, mode):
    elems = []
    for r in range(n):
        for c in range(n):
            fill = "#f8fafc"
            stroke = "#cbd5e1"
            if mode == "dense":
                if c <= r:
                    fill = "#2563eb"
            elif mode == "window":
                if c <= r and r - c <= 2:
                    fill = "#16a34a"
                if c == 0 and r >= 4:
                    fill = "#0f766e"
            elif mode == "dynamic":
                picks = {
                    0: [0],
                    1: [0, 1],
                    2: [1, 2],
                    3: [0, 2, 3],
                    4: [1, 3, 4],
                    5: [0, 2, 4, 5],
                    6: [2, 5, 6],
                    7: [0, 3, 6, 7],
                    8: [1, 4, 7, 8],
                    9: [0, 5, 8, 9],
                }
                if c in picks.get(r, []):
                    fill = "#dc2626"
            elems.append(rect(x + c * cell, y + r * cell, cell, cell, fill, stroke, 0.8))
    return "\n".join(elems)


def token_row(x, y, count, fill):
    elems = []
    for i in range(count):
        elems.append(rect(x + i * 28, y, 20, 20, fill, "#475569", 1.0, 3))
    return "\n".join(elems)


parts = [
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
    "<defs>",
    '<marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">',
    '<path d="M0,0 L0,6 L9,3 z" fill="#334155"/>',
    "</marker>",
    "</defs>",
    rect(0, 0, W, H, "#f8fafc", "none"),
    text(W / 2, 48, "Attention Efficiency Frontier", 30, 700),
    text(W / 2, 78, "Dense attention, sparse selection, latent cache, and hardware-aware kernels", 16, 400, "#475569"),
]

panel_w = 340
panel_h = 250
panels = [
    (55, 120, "Full causal attention", "O(n^2) scores"),
    (430, 120, "Window + global tokens", "O(nw) scores"),
    (805, 120, "Dynamic sparse selection", "O(nm) selected scores"),
]

for x, y, title, subtitle in panels:
    parts.append(rect(x, y, panel_w, panel_h, "#ffffff", "#cbd5e1", 1.4, 8))
    parts.append(text(x + panel_w / 2, y + 35, title, 20, 700))
    parts.append(text(x + panel_w / 2, y + 60, subtitle, 14, 400, "#64748b"))

parts.append(matrix(120, 205, 10, 16, "dense"))
parts.append(text(200, 395, "Every query sees all past tokens", 14, 500, "#334155"))

parts.append(matrix(495, 205, 10, 16, "window"))
parts.append(text(575, 395, "Recent window plus coarse global anchors", 14, 500, "#334155"))

parts.append(matrix(870, 205, 10, 16, "dynamic"))
parts.append(text(950, 395, "Indexer chooses a small relevant subset", 14, 500, "#334155"))

parts.append(rect(55, 430, 520, 245, "#ffffff", "#cbd5e1", 1.4, 8))
parts.append(text(315, 465, "MLA / latent KV cache", 22, 700))
parts.append(text(315, 492, "Store compressed latent states, reconstruct K/V when needed", 15, 400, "#64748b"))
parts.append(token_row(110, 545, 9, "#bfdbfe"))
parts.append(text(235, 530, "hidden states", 14, 500, "#334155"))
parts.append(arrow(370, 555, 425, 555))
parts.append(rect(435, 520, 80, 70, "#dbeafe", "#2563eb", 2, 8))
parts.append(text(475, 550, "latent", 16, 700, "#1d4ed8"))
parts.append(text(475, 572, "cache", 14, 500, "#1d4ed8"))
parts.append(arrow(475, 595, 475, 635))
parts.append(rect(390, 640, 170, 28, "#eff6ff", "#60a5fa", 1.4, 5))
parts.append(text(475, 660, "reconstructed K / V", 13, 600, "#1e3a8a"))

parts.append(rect(625, 430, 520, 245, "#ffffff", "#cbd5e1", 1.4, 8))
parts.append(text(885, 465, "NSA / DSA style hybrid sparsity", 22, 700))
parts.append(text(885, 492, "Local precision, global compression, and fine-grained selection", 15, 400, "#64748b"))
parts.append(rect(685, 535, 110, 52, "#ecfeff", "#0891b2", 1.6, 7))
parts.append(text(740, 568, "local window", 14, 700, "#155e75"))
parts.append(rect(835, 515, 110, 52, "#fef9c3", "#ca8a04", 1.6, 7))
parts.append(text(890, 548, "compressed", 14, 700, "#854d0e"))
parts.append(rect(985, 535, 110, 52, "#fee2e2", "#dc2626", 1.6, 7))
parts.append(text(1040, 568, "top-k select", 14, 700, "#991b1b"))
parts.append(arrow(795, 560, 835, 545))
parts.append(arrow(945, 545, 985, 560))
parts.append(line(740, 610, 1040, 610, "#94a3b8", 2, "6 6"))
parts.append(text(890, 635, "Sparse pattern must match training objective and hardware kernels", 14, 500, "#334155"))

parts.append("</svg>\n")

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(parts), encoding="utf-8")
print(OUT)
