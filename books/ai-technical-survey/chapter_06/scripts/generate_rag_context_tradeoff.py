import math
import os


def points_for(xs, ys, xmap, ymap):
    return " ".join(f"{xmap(x):.1f},{ymap(y):.1f}" for x, y in zip(xs, ys))


def main():
    width, height = 980, 430
    panel_w, panel_h = 365, 250
    top = 78
    panels = [(75, top, "Cost"), (535, top, "Coverage")]
    corpus = [10 ** (4 + 4 * i / 99) for i in range(100)]
    window = 1e6
    topk_tokens = 6e4

    long_cost = [min(x, window) / 1e5 for x in corpus]
    rag_cost = [topk_tokens / 1e5 for _ in corpus]
    long_cov = [min(x, window) / x for x in corpus]
    rag_cov = [max(0.05, min(0.9, 0.25 + 0.6 * (1 - math.exp(-topk_tokens / math.sqrt(x))))) for x in corpus]

    def make_panel(px, py, title, ys1, ys2, y_max, y_label):
        log_min, log_max = 4, 8

        def xmap(x):
            return px + panel_w * (math.log10(x) - log_min) / (log_max - log_min)

        def ymap(y):
            return py + panel_h * (1 - y / y_max)

        lines = [
            f'<text x="{px+panel_w/2}" y="{py-18}" text-anchor="middle" class="subtitle">{title}</text>',
            f'<rect x="{px}" y="{py}" width="{panel_w}" height="{panel_h}" fill="#fff" stroke="#444" stroke-width="1.4"/>',
            f'<text x="{px+panel_w/2}" y="{py+panel_h+45}" text-anchor="middle" class="small">candidate corpus size (tokens)</text>',
            f'<text x="{px-50}" y="{py+panel_h/2}" transform="rotate(-90 {px-50} {py+panel_h/2})" text-anchor="middle" class="small">{y_label}</text>',
        ]
        for exp in [4, 5, 6, 7, 8]:
            x = xmap(10**exp)
            lines.append(f'<line x1="{x:.1f}" y1="{py}" x2="{x:.1f}" y2="{py+panel_h}" stroke="#ddd" stroke-dasharray="3 4"/>')
            lines.append(f'<text x="{x:.1f}" y="{py+panel_h+22}" text-anchor="middle" class="tiny">1e{exp}</text>')
        for frac in [0, 0.25, 0.5, 0.75, 1.0]:
            y = ymap(frac * y_max)
            lines.append(f'<line x1="{px}" y1="{y:.1f}" x2="{px+panel_w}" y2="{y:.1f}" stroke="#eee"/>')
        win_x = xmap(window)
        lines.append(f'<line x1="{win_x:.1f}" y1="{py}" x2="{win_x:.1f}" y2="{py+panel_h}" stroke="#B85450" stroke-width="1.5" stroke-dasharray="6 5"/>')
        lines.append(f'<polyline points="{points_for(corpus, ys1, xmap, ymap)}" fill="none" stroke="#6C8EBF" stroke-width="2.7"/>')
        lines.append(f'<polyline points="{points_for(corpus, ys2, xmap, ymap)}" fill="none" stroke="#82B366" stroke-width="2.7"/>')
        return lines

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Arial,sans-serif;fill:#222}.title{font-size:20px;font-weight:700}.subtitle{font-size:17px;font-weight:700}.small{font-size:13px}.tiny{font-size:11px;fill:#555}</style>',
        '<text x="490" y="32" text-anchor="middle" class="title">RAG vs long context: cost, recall, and attention reliability</text>',
    ]
    parts += make_panel(*panels[0], long_cost, rag_cost, 10, "relative prompt cost")
    parts += make_panel(*panels[1], long_cov, rag_cov, 1, "usable evidence coverage (toy)")
    parts += [
        '<line x1="384" y1="365" x2="414" y2="365" stroke="#6C8EBF" stroke-width="3"/><text x="424" y="370" class="small">Long context</text>',
        '<line x1="520" y1="365" x2="550" y2="365" stroke="#82B366" stroke-width="3"/><text x="560" y="370" class="small">RAG top-k</text>',
        '<line x1="646" y1="365" x2="676" y2="365" stroke="#B85450" stroke-width="2" stroke-dasharray="6 5"/><text x="686" y="370" class="small">context window</text>',
        '<text x="490" y="405" text-anchor="middle" class="tiny">Curves are conceptual: real systems depend on retrieval quality, reranking, chunking, and model-specific long-context behavior.</text>',
        "</svg>",
    ]

    out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "images", "rag_long_context_tradeoff.svg"))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
