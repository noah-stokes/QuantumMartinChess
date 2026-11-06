"""Headless architecture/circuit drawings and training plots."""

import os
from pathlib import Path


def plotting():
    os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / ".cache/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def save_diagrams(output: Path) -> dict[str, str]:
    plt = plotting()
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    from qnns.rook_endgame.model import build_circuit

    output = Path(output)
    if not output.suffix:
        output = output.with_suffix(".svg")
    if output.suffix.lower() not in (".svg", ".png", ".pdf"):
        raise ValueError("Diagram output must use .svg, .png, or .pdf")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(16, 8))
    fig.patch.set_facecolor("#f7f9fc")
    ax.set(xlim=(0, 100), ylim=(0, 100))
    ax.set_axis_off()

    def text(x, y, value, size=10, color="#14253b", **kw):
        ax.text(x, y, value, fontsize=size, color=color, ha="center", va="center", **kw)

    def box(x, y, w, h, value, fill="#ffffff", edge="#c9d5e5", size=10):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3",
                                   facecolor=fill, edgecolor=edge, zorder=3))
        text(x + w / 2, y + h / 2, value, size, zorder=4)

    def arrow(a, b):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12,
                                     color="#65748b", zorder=2))

    text(50, 96, "35-bit chess evaluator: 190 learned parameters", 20, weight="bold")
    text(50, 89, "Each column is another group from ONE board. Each bit has its own scale.", 12)
    ys = [76 - q * 5 for q in range(7)]
    for q, y in enumerate(ys):
        text(4, y, f"q{q}", weight="bold")
        ax.plot([7, 78], [y, y], color="#aab7c8", lw=1.2, zorder=1)
    for group, cx in enumerate((14, 28, 42, 56, 70)):
        text(cx, 83, f"Group {group + 1}", 11, weight="bold")
        for q, y in enumerate(ys):
            j = group * 7 + q
            fill = "#dfeeff" if j == 5 else "#fff0de" if j == 26 else "#f0f4f9"
            box(cx - 5.2, y - 1.9, 10.4, 3.8, f"bit {j} × α{j}", fill, size=8)
        mx = cx + 7
        box(mx - .7, 43, 1.4, 36, "", "#e9e1f5", "#c5b2de")
        text(mx, 61, "learned RY/RZ + CZ", 7, "#625087", rotation=90, zorder=4)
    ax.plot([78, 79], [ys[0], ys[0]], color="#65748b")
    ax.plot([78, 79], [ys[-1], ys[-1]], color="#65748b")
    ax.plot([79, 79], [ys[-1], ys[0]], color="#65748b")
    arrow((79, 65), (82, 65))
    box(82, 56, 16, 22, "Repeat five groups\nwith new mixing angles\n\nSame 35 input scales", "#f0eafb", size=9)
    arrow((90, 55), (90, 50))
    box(82, 33, 16, 16, "Final 7 RY angles\n7 Z averages\nLinear + tanh\nOne White score", "#e4f4ec", size=9)
    text(21, 37, "Blue: bit 5, part of king square", 10, "#1765c1")
    text(59, 37, "Orange: bit 26, whose turn", 10, "#af520b")
    box(4, 16, 25, 12, "Each board starts with\nfresh qubits: |0000000⟩", "#edf2f9")
    box(37, 16, 27, 12, "All boards share the\nsame learned settings", "#f0eafb")
    box(72, 16, 25, 12, "Read outputs only after\nboth passes finish", "#e4f4ec")
    arrow((29.5, 22), (36.5, 22))
    arrow((64.5, 22), (71.5, 22))
    text(50, 6, "35 input scales + 140 mixing angles + 7 final angles + 8 output parameters = 190", 11)
    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    png = output.with_suffix(".png")
    if png != output:
        fig.savefig(png, dpi=160, bbox_inches="tight")
    plt.close(fig)

    circuit, _, _ = build_circuit()
    actual = output.with_name(output.stem + "_circuit.svg")
    drawing = circuit.draw(output="mpl", fold=45, style={"fontsize": 8}, initial_state=True)
    drawing.savefig(actual, bbox_inches="tight")
    plt.close(drawing)
    return {"overview": str(output), "overview_png": str(png), "circuit": str(actual)}


def save_loss_plot(path: Path, history: list[dict], baseline_metrics: dict):
    plt = plotting()
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = [r["epoch"] for r in history]
    ax.plot(epochs, [r["train_mse"] for r in history], marker="o", label="Training MSE")
    ax.plot(epochs, [r["validation_mse"] for r in history], marker="o", label="Validation MSE")
    for key, name in (("zero", "Always zero"), ("material_mean", "Material mean")):
        ax.axhline(baseline_metrics[key]["mse"], linestyle="--", label=name)
    ax.set(xlabel="Epoch (0 = before training)", ylabel="Unweighted mean squared error",
           title="Fixed training and validation samples")
    ax.legend()
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
