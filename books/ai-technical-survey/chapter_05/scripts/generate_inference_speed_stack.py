from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "images" / "inference_speed_stack.svg"
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
    text(W / 2, 46, "LLM Inference Speed Stack", 30, 700),
    text(W / 2, 76, "Latency and throughput are co-designed across model, runtime, and product layers", 16, 400, "#475569"),
]

layers = [
    ("Product routing", "small/large model router, cache, streaming UI", "#ede9fe", "#7c3aed"),
    ("Serving runtime", "continuous batching, queueing, admission control", "#fee2e2", "#dc2626"),
    ("KV memory", "PagedAttention, KV quantization, prefix cache", "#fef3c7", "#d97706"),
    ("Decode acceleration", "speculative decoding, Medusa, EAGLE, MTP", "#ecfeff", "#0891b2"),
    ("Kernels and precision", "FlashAttention, FlashMLA, fused ops, FP8/INT4", "#dbeafe", "#2563eb"),
]

for i, (title, subtitle, fill, stroke) in enumerate(layers):
    y = 125 + i * 88
    parts.append(rect(160, y, 860, 62, fill, stroke, 1.8, 10))
    parts.append(text(320, y + 38, title, 19, 700, stroke))
    parts.append(text(650, y + 38, subtitle, 15, 500, "#334155"))
    if i < len(layers) - 1:
        parts.append(arrow(590, y + 62, 590, y + 88))

parts.append(rect(160, 575, 255, 42, "#ffffff", "#94a3b8", 1.2, 8))
parts.append(text(287, 601, "TTFT", 18, 700, "#334155"))
parts.append(rect(462, 575, 255, 42, "#ffffff", "#94a3b8", 1.2, 8))
parts.append(text(590, 601, "TPOT / tokens per second", 18, 700, "#334155"))
parts.append(rect(765, 575, 255, 42, "#ffffff", "#94a3b8", 1.2, 8))
parts.append(text(892, 601, "Throughput / cost", 18, 700, "#334155"))
parts.append("</svg>\n")

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(parts), encoding="utf-8")
print(OUT)
