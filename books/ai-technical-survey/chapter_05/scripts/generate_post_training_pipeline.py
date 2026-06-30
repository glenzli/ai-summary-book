import os


def box(x, y, w, h, label, fill, stroke):
    lines = label.split("\\n")
    base = y + h / 2 - 8 * (len(lines) - 1) + 5
    text = "\n".join(
        f'<text x="{x+w/2}" y="{base+i*16}" text-anchor="middle" class="small">{line}</text>'
        for i, line in enumerate(lines)
    )
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="1.8"/>\n{text}'


def arrow(x1, y1, x2, y2, label="", dashed=False):
    dash = ' stroke-dasharray="6 5"' if dashed else ""
    mid = ""
    if label:
        mid = f'<text x="{(x1+x2)/2}" y="{(y1+y2)/2-8}" text-anchor="middle" class="tiny">{label}</text>'
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#444" stroke-width="1.8"{dash} marker-end="url(#arrow)"/>\n{mid}'


def main():
    width, height = 1040, 560
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#444"/></marker></defs>',
        '<style>text{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Arial,sans-serif;fill:#222}.title{font-size:21px;font-weight:700}.small{font-size:14px}.tiny{font-size:12px;fill:#555}.label{font-size:13px;font-weight:700;fill:#333}</style>',
        '<text x="520" y="34" text-anchor="middle" class="title">Post-training map: SFT, RLHF/PPO, and DPO</text>',
        box(70, 82, 160, 66, "Base model\\nnext-token pretraining", "#F5F5F5", "#666666"),
        box(290, 82, 160, 66, "SFT model\\ninstruction following", "#DAE8FC", "#6C8EBF"),
        box(510, 82, 160, 66, "Candidate responses\\nfor same prompt", "#FFF2CC", "#D6B656"),
        box(730, 82, 160, 66, "Human preference\\ny_w > y_l", "#FFF2CC", "#D6B656"),
        arrow(230, 115, 290, 115, "SFT data"),
        arrow(450, 115, 510, 115, "sample"),
        arrow(670, 115, 730, 115, "rank"),
        '<text x="520" y="196" text-anchor="middle" class="label">Two common optimization routes</text>',
        box(170, 245, 170, 66, "Reward model\\nlearn r(x,y)", "#E1D5E7", "#9673A6"),
        box(170, 370, 170, 66, "PPO update\\nreward - KL", "#F8CECC", "#B85450"),
        box(430, 370, 170, 66, "Aligned model\\npolicy output", "#D5E8D4", "#82B366"),
        box(700, 245, 170, 66, "DPO objective\\ndirect preference loss", "#E1D5E7", "#9673A6"),
        box(700, 370, 170, 66, "Aligned model\\npolicy output", "#D5E8D4", "#82B366"),
        arrow(730, 148, 255, 245, "preference pairs"),
        arrow(255, 311, 255, 370, "score"),
        arrow(340, 403, 430, 403, "update"),
        arrow(730, 148, 785, 245, "preference pairs"),
        arrow(785, 311, 785, 370, "optimize"),
        '<path d="M290 403 C335 340 385 300 450 148" fill="none" stroke="#444" stroke-width="1.8" stroke-dasharray="6 5" marker-end="url(#arrow)"/>',
        '<text x="377" y="288" text-anchor="middle" class="tiny">KL anchor to SFT</text>',
        '<path d="M700 403 C642 330 620 260 450 148" fill="none" stroke="#444" stroke-width="1.8" stroke-dasharray="6 5" marker-end="url(#arrow)"/>',
        '<text x="613" y="302" text-anchor="middle" class="tiny">reference policy</text>',
        '<text x="260" y="476" text-anchor="middle" class="small">RLHF/PPO: train a reward model, then optimize the policy against reward plus KL.</text>',
        '<text x="785" y="476" text-anchor="middle" class="small">DPO: optimize the policy directly from preferred vs rejected responses.</text>',
        '<text x="520" y="524" text-anchor="middle" class="tiny">Both routes use preference data; both need anchors, safety data, evaluation, and monitoring.</text>',
        "</svg>",
    ]

    out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "images", "post_training_pipeline.svg"))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
