"""
Draws the architecture and workflow diagrams used in the report and slides.
    python docs/make_diagrams.py
Outputs: docs/diagrams/architecture.png and docs/diagrams/workflow.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # draw to files, no window
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).parent / "diagrams"
OUT.mkdir(exist_ok=True)

BG, PANEL, TEXT, MUTED = "#0b0d17", "#151a2c", "#e7e9f3", "#8a90a8"
ORANGE, BLUE, AQUA, VIOLET = "#d95926", "#3987e5", "#199e70", "#9085e9"


def box(ax, x, y, w, h, title, sub="", color=VIOLET):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15",
                                fc=PANEL, ec=color, lw=2))
    ax.text(x + w / 2, y + h * (0.62 if sub else 0.5), title, ha="center", va="center",
            color=TEXT, fontsize=11, weight="bold")
    if sub:
        ax.text(x + w / 2, y + h * 0.3, sub, ha="center", va="center", color=MUTED, fontsize=8.5)


def arrow(ax, x1, y1, x2, y2, color=MUTED):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8, mutation_scale=14))


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h), dpi=160)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


# ---------------- Architecture ----------------
fig, ax = canvas(14.6, 7.2)
ax.text(7.3, 6.8, "MeaningMap - System Architecture", ha="center", color=TEXT, fontsize=16, weight="bold")

box(ax, 0.3, 4.7, 3.0, 1.2, "User query", "typed in the browser", VIOLET)
box(ax, 0.3, 3.0, 3.0, 1.2, "data/documents.csv", "80 docs · 8 categories", MUTED)
box(ax, 0.3, 1.1, 3.0, 1.2, "data/eval_queries.json", "37 labelled test queries", MUTED)

box(ax, 4.1, 4.7, 3.9, 1.2, "TF-IDF Vectorizer (scikit-learn)", "lowercase · tokenize · stop words · IDF", ORANGE)
box(ax, 4.1, 3.0, 3.9, 1.2, "Sentence-BERT (all-MiniLM-L6-v2)", "WordPiece · 6 transformer layers · mean pool", BLUE)
box(ax, 8.6, 3.0, 2.3, 2.9, "Cosine similarity", "+ hybrid mix\nα·sem + (1−α)·kw", AQUA)

box(ax, 11.5, 5.0, 3.0, 0.95, "Search Battle", "3 ranked lists + highlights", VIOLET)
box(ax, 11.5, 3.8, 3.0, 0.95, "Meaning Galaxy", "2-D maps + silhouette", VIOLET)
box(ax, 11.5, 2.6, 3.0, 0.95, "Similarity Lab / Under Hood", "heatmap · pipeline steps", VIOLET)
box(ax, 11.5, 1.25, 3.0, 0.95, "Evaluation", "P@1 · Recall@5 · MRR", VIOLET)

# query and documents both go through BOTH models
arrow(ax, 3.3, 5.3, 4.1, 5.3)
arrow(ax, 3.3, 5.0, 4.1, 3.8)
arrow(ax, 3.3, 3.9, 4.1, 4.9)
arrow(ax, 3.3, 3.6, 4.1, 3.6)
# models -> cosine
arrow(ax, 8.0, 5.3, 8.6, 5.1, ORANGE)
arrow(ax, 8.0, 3.6, 8.6, 3.8, BLUE)
# cosine -> features
for y in (5.47, 4.27, 3.07):
    arrow(ax, 10.9, 4.45, 11.5, y, AQUA)
arrow(ax, 10.9, 3.3, 11.5, 1.9, AQUA)
# test queries -> evaluation
arrow(ax, 3.3, 1.7, 11.5, 1.7)

ax.text(1.6, 0.35, "search_engine.py = all NLP logic        app.py = Streamlit user interface        evaluate.py = command-line metrics",
        color=MUTED, fontsize=9.5, ha="left")
fig.savefig(OUT / "architecture.png", bbox_inches="tight", facecolor=BG)
plt.close(fig)

# ---------------- Workflow ----------------
fig, ax = canvas(15, 5.4)
ax.text(7.5, 5.0, "MeaningMap - Query Workflow", ha="center", color=TEXT, fontsize=16, weight="bold")

box(ax, 0.2, 2.1, 2.0, 1.2, "1. Query", "“my PC takes forever\nto boot”", VIOLET)
# keyword lane
box(ax, 2.9, 3.4, 2.3, 1.1, "2a. Preprocess", "lower · tokens · stop words", ORANGE)
box(ax, 5.8, 3.4, 2.3, 1.1, "3a. TF-IDF vector", "sparse, ~600 dims", ORANGE)
box(ax, 8.7, 3.4, 2.3, 1.1, "4a. Cosine vs docs", "word-overlap score", ORANGE)
# semantic lane
box(ax, 2.9, 0.9, 2.3, 1.1, "2b. WordPiece", "sub-word tokens", BLUE)
box(ax, 5.8, 0.9, 2.3, 1.1, "3b. Transformer", "→ 384-dim embedding", BLUE)
box(ax, 8.7, 0.9, 2.3, 1.1, "4b. Cosine vs docs", "meaning score", BLUE)

box(ax, 11.6, 2.1, 1.6, 1.2, "5. Hybrid", "min-max + α", AQUA)
box(ax, 13.5, 2.1, 1.4, 1.2, "6. Rank", "top-k +\nexplain", VIOLET)

arrow(ax, 2.2, 2.9, 2.9, 3.9)
arrow(ax, 2.2, 2.5, 2.9, 1.5)
for x in (5.2, 8.1):
    arrow(ax, x, 3.95, x + 0.6, 3.95, ORANGE)
    arrow(ax, x, 1.45, x + 0.6, 1.45, BLUE)
arrow(ax, 11.0, 3.95, 11.6, 3.0, ORANGE)
arrow(ax, 11.0, 1.45, 11.6, 2.4, BLUE)
arrow(ax, 13.2, 2.7, 13.5, 2.7, AQUA)
fig.savefig(OUT / "workflow.png", bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("Saved diagrams to", OUT)
