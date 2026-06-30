import math
import os


def polyline(points, color):
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.8" />'


def main():
    width, height = 880, 500
    left, right, top, bottom = 85, 35, 55, 75
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_step = 80
    ymin, ymax = 1e-8, 1.0
    log_min, log_max = math.log10(ymin), math.log10(ymax)

    def xmap(t):
        return left + plot_w * t / max_step

    def ymap(y):
        y = max(y, ymin)
        return top + plot_h * (log_max - math.log10(y)) / (log_max - log_min)

    curves = [
        (1.00, "#82B366", "f = 1.00"),
        (0.99, "#6C8EBF", "f = 0.99"),
        (0.95, "#9673A6", "f = 0.95"),
        (0.90, "#D6B656", "f = 0.90"),
        (0.80, "#B85450", "f = 0.80"),
    ]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Arial,sans-serif;fill:#222}.small{font-size:13px}.label{font-size:15px}.title{font-size:20px;font-weight:700}</style>',
        f'<text x="{width/2}" y="28" text-anchor="middle" class="title">LSTM forget gate: long-range retention is multiplicative</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#444" stroke-width="1.5"/>',
        f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#444" stroke-width="1.5"/>',
    ]

    for tick in [0, 20, 40, 60, 80]:
        x = xmap(tick)
        parts.append(f'<line x1="{x:.1f}" y1="{top+plot_h}" x2="{x:.1f}" y2="{top+plot_h+6}" stroke="#444"/>')
        parts.append(f'<text x="{x:.1f}" y="{top+plot_h+26}" text-anchor="middle" class="small">{tick}</text>')

    for yval, label in [(1, "1"), (1e-2, "1e-2"), (1e-4, "1e-4"), (1e-6, "1e-6"), (1e-8, "1e-8")]:
        y = ymap(yval)
        parts.append(f'<line x1="{left-6}" y1="{y:.1f}" x2="{left}" y2="{y:.1f}" stroke="#444"/>')
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="#ddd" stroke-dasharray="3 4"/>')
        parts.append(f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" class="small">{label}</text>')

    for f, color, label in curves:
        pts = [(xmap(t), ymap(f**t)) for t in range(max_step + 1)]
        parts.append(polyline(pts, color))

    parts.extend(
        [
            f'<text x="{left+plot_w/2}" y="{height-22}" text-anchor="middle" class="label">time gap T - k</text>',
            f'<text x="22" y="{top+plot_h/2}" transform="rotate(-90 22 {top+plot_h/2})" text-anchor="middle" class="label">approx. retention product of forget gates</text>',
            f'<rect x="{width-150}" y="68" width="118" height="126" rx="8" fill="#fff" stroke="#ccc"/>',
        ]
    )

    for idx, (_, color, label) in enumerate(curves):
        y = 90 + idx * 22
        parts.append(f'<line x1="{width-132}" y1="{y}" x2="{width-102}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{width-94}" y="{y+4}" class="small">{label}</text>')

    parts.extend(
        [
            '<path d="M590 88 C540 105 510 130 475 165" fill="none" stroke="#6C8EBF" stroke-width="1.6" marker-end="url(#arrow)"/>',
            '<path d="M560 395 C505 355 460 320 408 283" fill="none" stroke="#B85450" stroke-width="1.6" marker-end="url(#arrow)"/>',
            '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#444"/></marker></defs>',
            '<text x="598" y="86" class="small">near-identity gates preserve gradients</text>',
            '<text x="560" y="410" class="small">small deviations compound over time</text>',
            "</svg>",
        ]
    )

    out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "images", "lstm_forget_gate_retention.svg"))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
