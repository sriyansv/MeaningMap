# MeaningMap: Keyword vs Semantic Search

**Do computers understand what you *mean*, or only the *words* you type?**

MeaningMap is an interactive NLP application that runs three information-retrieval methods side by side on the same collection of documents:

| Method | NLP technique | Idea |
|---|---|---|
| **Keyword search** | TF-IDF + Vector Space Model + cosine similarity | Matches documents that share *words* with the query |
| **Semantic search** | Sentence-BERT transformer embeddings (`all-MiniLM-L6-v2`) + cosine similarity | Matches documents that share *meaning* with the query |
| **Hybrid search** | Weighted mix of both (min-max normalised) | Combines exact-word precision and meaning |

All three are measured on a hand-labelled test set with standard IR metrics: **Precision@1, Recall@5 and MRR**.

## Features

1. **Search**: one query, three ranked result lists side by side. Shared keywords are highlighted in yellow, and every word is tinted blue by how close its *meaning* is to the query. A verdict banner says whether the methods agree, disagree, or keyword search finds nothing (no shared words).
2. **Document map**: 2-D maps (SVD for TF-IDF, PCA for embeddings) of all documents, coloured by topic, with your query as a star and lines to its top-3 results. Silhouette scores show how well topics cluster in each space.
3. **Compare sentences**: compare any two sentences with TF-IDF vs. Sentence-BERT, plus a word-to-word meaning heatmap. Includes presets for paraphrases, polysemy ("bank"), and a known limitation (word order).
4. **How it works**: every pipeline step for your query: lower-casing, tokenization, stop-word removal, vocabulary lookup, TF-IDF weights, WordPiece sub-words, the 384-number embedding, and the cosine formula.
5. **Evaluation**: live Precision@1 / Recall@5 / MRR for all three methods, plus a per-question table showing where each method ranked the correct answer.
6. **Bring your own documents**: paste your own text in the sidebar and search it.

## Setup and run (Windows, Python 3.10+)

```bash
# 1. create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate            # (macOS/Linux: source .venv/bin/activate)

# 2. install CPU-only PyTorch first (small download), then the rest
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 3. start the app (opens http://localhost:8501)
streamlit run app.py

# optional: print the evaluation metrics in the terminal
python evaluate.py
```

> The **first run needs internet** to download the Sentence-BERT model once (~90 MB). It is cached afterwards, so the app then works offline.

## Project structure

```
NLP-Project/
├── app.py                  # Streamlit user interface (tabs, cards, charts)
├── search_engine.py        # ALL the NLP logic (TF-IDF, SBERT, hybrid, explanations, evaluation)
├── evaluate.py             # command-line evaluation -> results/
├── test_engine.py          # automated checks of the engine
├── requirements.txt
├── .streamlit/config.toml  # dark theme
├── data/
│   ├── documents.csv       # 80 short documents, 8 categories
│   └── eval_queries.json   # 37 test questions + correct document IDs
├── results/                # metrics.csv, per_query.csv (created by evaluate.py)
└── docs/
    ├── MeaningMap_Report.docx
    ├── MeaningMap_Slides.pptx
    ├── diagrams/           # architecture.png, workflow.png
    ├── screenshots/        # screenshots of the running app
    ├── make_diagrams.py    # regenerates the diagrams   (needs: pip install matplotlib)
    └── make_report.py      # regenerates the report     (needs: pip install python-docx)
```

> When submitting as a zip, leave out the `.venv/` folder (it is large and machine-specific). It can be recreated with the setup steps above.

## Results (37 test questions, 80 documents)

| Method | Precision@1 | Recall@5 | MRR |
|---|---|---|---|
| Keyword (TF-IDF) | 0.297 | 0.459 | 0.391 |
| Semantic (SBERT) | **0.919** | **1.000** | **0.959** |
| Hybrid (α = 0.7) | 0.703 | 0.986 | 0.836 |

Reproduce with `python evaluate.py`. Run the automated checks with `python test_engine.py`.

## Dataset

`data/documents.csv` is a small knowledge base of 80 two-sentence documents written for this project, 10 in each of 8 categories (Health, Technology, Space, Food, Travel, Finance, Sports, Environment). `data/eval_queries.json` holds 37 everyday-language questions, each labelled with the document(s) that answer it. The dataset is small on purpose: every result can be checked by hand during a demo.
