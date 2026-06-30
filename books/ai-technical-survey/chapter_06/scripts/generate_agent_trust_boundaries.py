import os


def box(x, y, w, h, label, fill, stroke, cls="small"):
    lines = label.split("\\n")
    base = y + h / 2 - 8 * (len(lines) - 1) + 5
    text = "\n".join(
        f'<text x="{x+w/2}" y="{base+i*16}" text-anchor="middle" class="{cls}">{line}</text>'
        for i, line in enumerate(lines)
    )
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="1.8"/>\n{text}'


def arrow(x1, y1, x2, y2, label="", color="#444", dashed=False):
    dash = ' stroke-dasharray="6 5"' if dashed else ""
    mid = ""
    if label:
        mid = f'<text x="{(x1+x2)/2}" y="{(y1+y2)/2-8}" text-anchor="middle" class="tiny">{label}</text>'
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="1.8"{dash} marker-end="url(#arrow)"/>\n{mid}'


def main():
    width, height = 1040, 560
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#444"/></marker></defs>',
        '<style>text{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Arial,sans-serif;fill:#222}.title{font-size:21px;font-weight:700}.small{font-size:14px}.tiny{font-size:12px;fill:#555}.label{font-size:13px;font-weight:700}</style>',
        '<text x="520" y="34" text-anchor="middle" class="title">Agent trust boundaries: untrusted content must not become authority</text>',
        '<rect x="60" y="76" width="410" height="392" rx="12" fill="#F7FBFF" stroke="#6C8EBF" stroke-width="2"/>',
        '<text x="265" y="102" text-anchor="middle" class="label">Trusted control plane</text>',
        '<rect x="570" y="76" width="410" height="392" rx="12" fill="#FFF9F0" stroke="#D6B656" stroke-width="2"/>',
        '<text x="775" y="102" text-anchor="middle" class="label">Untrusted data and external effects</text>',
    ]

    # Trusted side
    parts += [
        box(110, 130, 145, 62, "System policy\\npermissions", "#DAE8FC", "#6C8EBF"),
        box(285, 130, 145, 62, "User goal\\napprovals", "#F5F5F5", "#666666"),
        box(110, 250, 145, 62, "Agent runtime\\nstate machine", "#D5E8D4", "#82B366"),
        box(285, 250, 145, 62, "Guardrails\\npolicy checks", "#F8CECC", "#B85450"),
        box(198, 370, 145, 62, "Audit log\\ntrace", "#E1D5E7", "#9673A6"),
    ]

    # Untrusted side
    parts += [
        box(610, 130, 145, 62, "Web pages\\ndocuments", "#FFF2CC", "#D6B656"),
        box(790, 130, 145, 62, "Tool outputs\\nAPI results", "#FFF2CC", "#D6B656"),
        box(610, 250, 145, 62, "Secrets\\nprivate data", "#F8CECC", "#B85450"),
        box(790, 250, 145, 62, "Side effects\\nwrite / send / pay", "#F8CECC", "#B85450"),
        box(700, 370, 145, 62, "Artifacts\\nfiles / tickets", "#F5F5F5", "#666666"),
    ]

    parts += [
        arrow(255, 161, 285, 161, "policy"),
        arrow(182, 192, 182, 250, "constraints"),
        arrow(357, 192, 357, 250, "approval"),
        arrow(255, 281, 285, 281, "check"),
        arrow(357, 312, 270, 370, "record"),
        arrow(682, 192, 255, 250, "read as data", dashed=True),
        arrow(862, 192, 255, 250, "observe", dashed=True),
        arrow(755, 281, 430, 281, "never expose directly", "#B85450", True),
        arrow(430, 281, 790, 281, "approval required"),
        arrow(862, 312, 772, 370, "result"),
        '<line x1="520" y1="86" x2="520" y2="458" stroke="#B85450" stroke-width="2.2" stroke-dasharray="8 6"/>',
        '<text x="520" y="486" text-anchor="middle" class="label" fill="#B85450">trust boundary</text>',
        '<text x="520" y="524" text-anchor="middle" class="tiny">External content can provide evidence, but only trusted policy and user approval can authorize actions.</text>',
        "</svg>",
    ]

    out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "images", "agent_trust_boundaries.svg"))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
