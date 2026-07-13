from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "images" / "open_model_ecosystem.svg"
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
    text(W / 2, 46, "Open-Weight Model Ecosystem", 30, 700),
    text(W / 2, 76, "One base model can branch into adapters, distilled checkpoints, merged models, and quantized builds", 16, 400, "#475569"),
]

parts.append(rect(480, 120, 220, 72, "#dbeafe", "#2563eb", 2, 12))
parts.append(text(590, 155, "Base model", 22, 700, "#1d4ed8"))
parts.append(text(590, 178, "Qwen / Llama / DeepSeek", 13, 500, "#334155"))

branches = [
    (95, 285, "SFT / DPO", "instruction and preference", "#ecfeff", "#0891b2"),
    (330, 285, "LoRA adapters", "small task deltas", "#fef3c7", "#d97706"),
    (565, 285, "Distillation", "teacher outputs", "#fee2e2", "#dc2626"),
    (800, 285, "Model merge", "combine weight deltas", "#ede9fe", "#7c3aed"),
]

for x, y, title, subtitle, fill, stroke in branches:
    parts.append(rect(x, y, 190, 80, fill, stroke, 1.8, 10))
    parts.append(text(x + 95, y + 35, title, 18, 700, stroke))
    parts.append(text(x + 95, y + 60, subtitle, 13, 500, "#334155"))
    parts.append(arrow(590, 192, x + 95, y, "#64748b", 1.8))

parts.append(rect(310, 455, 560, 100, "#ffffff", "#94a3b8", 1.5, 10))
parts.append(text(590, 485, "Quantized checkpoints and packaging", 20, 700, "#334155"))
parts.append(text(590, 513, "Precision: INT8 / INT4 / mixed precision", 14, 500, "#475569"))
parts.append(text(590, 537, "Methods: AWQ / GPTQ   |   Container: GGUF", 14, 500, "#475569"))

for x, y, *_ in branches:
    parts.append(arrow(x + 95, y + 80, 590, 455, "#64748b", 1.6))

parts.append(rect(230, 600, 720, 45, "#ffffff", "#cbd5e1", 1.2, 8))
parts.append(text(590, 628, "Runtime: vLLM, llama.cpp, SGLang, TGI, TensorRT-LLM, routers, RAG, safety filters", 15, 600, "#334155"))
parts.append(arrow(590, 555, 590, 600, "#334155", 2))
parts.append("</svg>\n")

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(parts), encoding="utf-8")
print(OUT)
