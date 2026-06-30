import os


def box(x, y, w, h, text, fill, stroke):
    lines = text.split("\\n")
    dy = 16
    text_y = y + h / 2 - dy * (len(lines) - 1) / 2 + 5
    label = "\n".join(
        f'<text x="{x+w/2}" y="{text_y+i*dy}" text-anchor="middle" class="small">{line}</text>'
        for i, line in enumerate(lines)
    )
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill}" stroke="{stroke}" stroke-width="2"/>\n{label}'


def arrow(x1, y1, x2, y2, label=""):
    mid = ""
    if label:
        mid = f'<text x="{(x1+x2)/2}" y="{(y1+y2)/2-7}" text-anchor="middle" class="tiny">{label}</text>'
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#444" stroke-width="1.8" marker-end="url(#arrow)"/>\n{mid}'


def main():
    width, height = 980, 520
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#444"/></marker></defs>',
        '<style>text{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Arial,sans-serif;fill:#222}.title{font-size:20px;font-weight:700}.small{font-size:14px}.tiny{font-size:12px;fill:#555}</style>',
        '<text x="490" y="32" text-anchor="middle" class="title">Agent as a controlled loop: plan, act, observe, verify, revise</text>',
        box(58, 88, 168, 82, "User goal\\nconstraints", "#F5F5F5", "#666666"),
        box(326, 88, 190, 82, "Planner / LLM\\nreasoning state", "#DAE8FC", "#6C8EBF"),
        box(686, 88, 190, 82, "Tool call\\ncode / search / API", "#FFF2CC", "#D6B656"),
        box(326, 245, 190, 74, "Memory\\ncontext + retrieval", "#E1D5E7", "#9673A6"),
        box(686, 245, 190, 74, "Environment\\nobservation", "#D5E8D4", "#82B366"),
        box(326, 385, 190, 74, "Verifier\\ntests / citations / policy", "#F8CECC", "#B85450"),
        box(686, 385, 190, 74, "Final answer\\nor action", "#F5F5F5", "#666666"),
        arrow(226, 129, 326, 129, "task"),
        arrow(516, 129, 686, 129, "action"),
        arrow(781, 170, 781, 245, "result"),
        arrow(686, 282, 516, 282, "observation"),
        arrow(421, 170, 421, 245, "read / write"),
        arrow(421, 319, 421, 385, "check"),
        arrow(516, 422, 686, 422, "approved"),
        '<path d="M686 405 C590 360 550 225 516 144" fill="none" stroke="#444" stroke-width="1.8" marker-end="url(#arrow)"/>',
        '<text x="610" y="315" text-anchor="middle" class="tiny">revise</text>',
        '<text x="490" y="492" text-anchor="middle" class="small" fill="#555">Reliable agents need permission boundaries and verification, not only longer reasoning traces.</text>',
        "</svg>",
    ]

    out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "images", "agent_control_loop.svg"))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
