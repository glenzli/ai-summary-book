import os


def rect(x, y, w, h, label, fill, stroke):
    lines = label.split("\\n")
    text = []
    base = y + h / 2 - 8 * (len(lines) - 1) + 5
    for i, line in enumerate(lines):
        text.append(f'<text x="{x+w/2}" y="{base+i*16}" text-anchor="middle" class="small">{line}</text>')
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="1.8"/>\n' + "\n".join(text)


def arrow(x1, y1, x2, y2, label=""):
    mid = ""
    if label:
        mid = f'<text x="{(x1+x2)/2}" y="{(y1+y2)/2-8}" text-anchor="middle" class="tiny">{label}</text>'
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#444" stroke-width="1.7" marker-end="url(#arrow)"/>\n{mid}'


def main():
    width, height = 1040, 560
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#444"/></marker></defs>',
        '<style>text{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Arial,sans-serif;fill:#222}.title{font-size:21px;font-weight:700}.small{font-size:14px}.tiny{font-size:12px;fill:#555}.layer{font-size:13px;font-weight:700;fill:#333}</style>',
        '<text x="520" y="34" text-anchor="middle" class="title">Agent system stack: model ability becomes controlled software</text>',
        '<text x="54" y="92" class="layer">User and task layer</text>',
        '<text x="54" y="184" class="layer">Runtime layer</text>',
        '<text x="54" y="276" class="layer">Protocol layer</text>',
        '<text x="54" y="368" class="layer">Tool and data layer</text>',
        '<text x="54" y="460" class="layer">Observability layer</text>',
    ]

    # user/task
    parts += [
        rect(210, 64, 185, 62, "User goal\\nconstraints", "#F5F5F5", "#666666"),
        rect(430, 64, 185, 62, "Human approval\\nrisk decisions", "#F8CECC", "#B85450"),
        rect(650, 64, 185, 62, "Artifacts\\nfiles / reports", "#E1D5E7", "#9673A6"),
    ]

    # runtime
    parts += [
        rect(170, 156, 160, 62, "Planner\\ntask graph", "#DAE8FC", "#6C8EBF"),
        rect(360, 156, 160, 62, "Context builder\\nselection + memory", "#DAE8FC", "#6C8EBF"),
        rect(550, 156, 160, 62, "Model call\\nreasoning / output", "#D5E8D4", "#82B366"),
        rect(740, 156, 160, 62, "Policy gate\\npermissions", "#F8CECC", "#B85450"),
    ]

    # protocol
    parts += [
        rect(255, 248, 190, 62, "MCP\\ntools / resources", "#FFF2CC", "#D6B656"),
        rect(595, 248, 190, 62, "A2A\\nagent tasks", "#FFF2CC", "#D6B656"),
    ]

    # tools/data
    parts += [
        rect(140, 340, 150, 62, "Search\\nweb / docs", "#F5F5F5", "#666666"),
        rect(320, 340, 150, 62, "Code\\nsandbox / tests", "#F5F5F5", "#666666"),
        rect(500, 340, 150, 62, "Data\\nDB / files", "#F5F5F5", "#666666"),
        rect(680, 340, 150, 62, "Peer agents\\nspecialists", "#F5F5F5", "#666666"),
    ]

    # observability
    parts += [
        rect(245, 432, 170, 62, "Trace\\nspans / logs", "#E1D5E7", "#9673A6"),
        rect(435, 432, 170, 62, "Evaluation\\ntrajectory tests", "#E1D5E7", "#9673A6"),
        rect(625, 432, 170, 62, "Cost and latency\\nmonitoring", "#E1D5E7", "#9673A6"),
    ]

    parts += [
        arrow(302, 126, 250, 156, "goal"),
        arrow(520, 126, 820, 156, "approval"),
        arrow(615, 126, 820, 187, "write"),
        arrow(330, 187, 360, 187),
        arrow(520, 187, 550, 187),
        arrow(710, 187, 740, 187),
        arrow(820, 218, 350, 248, "authorized calls"),
        arrow(820, 218, 690, 248),
        arrow(350, 310, 215, 340),
        arrow(350, 310, 395, 340),
        arrow(350, 310, 575, 340),
        arrow(690, 310, 755, 340),
        arrow(215, 402, 330, 432, "events"),
        arrow(395, 402, 520, 432, "results"),
        arrow(575, 402, 710, 432, "metrics"),
        '<text x="520" y="532" text-anchor="middle" class="tiny">MCP standardizes tool and context access; A2A standardizes cross-agent task exchange. Runtime policy decides what is allowed.</text>',
        "</svg>",
    ]

    out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "images", "agent_system_stack.svg"))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
