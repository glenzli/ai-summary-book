from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "images" / "mamba_ssm_flow.svg"
W, H = 1180, 620


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
    text(W / 2, 46, "Attention vs Selective State Space", 30, 700),
    text(W / 2, 76, "Explicit token-token lookup versus compressed recurrent state", 16, 400, "#475569"),
]

parts.append(rect(60, 120, 500, 390, "#ffffff", "#cbd5e1"))
parts.append(text(310, 155, "Transformer attention", 23, 700))
parts.append(text(310, 184, "query reads a visible set of historical keys", 15, 400, "#64748b"))

token_x = [120 + i * 56 for i in range(7)]
for i, x in enumerate(token_x):
    parts.append(rect(x, 315, 38, 38, "#dbeafe", "#2563eb", 1.5, 6))
    parts.append(text(x + 19, 340, f"x{i+1}", 14, 700, "#1d4ed8"))
parts.append(rect(395, 245, 62, 50, "#fee2e2", "#dc2626", 1.8, 8))
parts.append(text(426, 275, "q_t", 17, 700, "#991b1b"))
for x in token_x[:6]:
    parts.append(arrow(426, 295, x + 19, 315, "#64748b", 1.6))
parts.append(text(310, 405, "Cost: explicit attention scores over visible tokens", 16, 500, "#334155"))
parts.append(text(310, 433, "Strength: content-addressed retrieval", 16, 500, "#334155"))

parts.append(rect(620, 120, 500, 390, "#ffffff", "#cbd5e1"))
parts.append(text(870, 155, "Mamba / selective SSM", 23, 700))
parts.append(text(870, 184, "each token updates a compressed state", 15, 400, "#64748b"))

state_y = 310
for i in range(5):
    x = 685 + i * 82
    parts.append(rect(x, state_y, 48, 48, "#ecfeff", "#0891b2", 1.6, 8))
    parts.append(text(x + 24, state_y + 30, f"h{i}", 15, 700, "#155e75"))
    if i < 4:
        parts.append(arrow(x + 48, state_y + 24, x + 82, state_y + 24))
for i in range(5):
    x = 685 + i * 82
    parts.append(rect(x, 235, 48, 32, "#fef3c7", "#d97706", 1.4, 7))
    parts.append(text(x + 24, 256, f"x{i+1}", 13, 700, "#92400e"))
    parts.append(arrow(x + 24, 267, x + 24, state_y))
    parts.append(text(x + 24, 225, "gate", 11, 500, "#92400e"))
parts.append(rect(1015, state_y, 48, 48, "#fee2e2", "#dc2626", 1.6, 8))
parts.append(text(1039, state_y + 30, "y_t", 15, 700, "#991b1b"))
parts.append(arrow(1013, state_y + 24, 1015, state_y + 24))
parts.append(text(870, 405, "Cost: linear scan plus hardware-aware kernels", 16, 500, "#334155"))
parts.append(text(870, 433, "Strength: streaming memory and long sequences", 16, 500, "#334155"))

parts.append("</svg>\n")
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(parts), encoding="utf-8")
print(OUT)
